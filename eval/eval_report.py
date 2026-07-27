"""
eval_report.py — Rapport d'évaluation complet : métriques + figures

Métriques calculées (par phrase et global) :
  • Exact Match %       — correspondance exacte kuma_mt / référence
  • chrF / chrF++       — character F-score (sacrebleu) : kuma / google / nllb vs référence
                          (chrF++ ajoute les n-grammes de mots, word_order=2, pour un signal
                          sur l'ordre des mots que chrF seul — caractères uniquement — n'a pas)
  • CamemBERT sim       — BERTScore F1 mot-à-mot (CamemBERT) vs le FR original (déjà dans
                          le CSV) : Kuma comparé via le gloss KG (sens_fr), Google et NLLB
                          via back-traduction BM→FR (deep_translator / NLLB lui-même)

Figures générées (PNG dans le dossier courant) :
  fig_eval_global.png        — vue d'ensemble toutes métriques
  fig_eval_chrf_cat.png      — chrF par catégorie (kuma vs google vs nllb)
  fig_eval_camembert_cat.png — CamemBERT par catégorie
  fig_eval_exact_cat.png     — Exact Match % par catégorie (kuma_mt)

Usage :
  python eval_report.py                          # dernier CSV auto-détecté
  python eval_report.py resultats_bambara_X.csv  # CSV explicite
  python eval_report.py --run                    # lance test_phrases + rapport
  python eval_report.py --run --no-camembert     # sans CamemBERT (plus rapide)
"""

import sys
import os
import glob
import csv
import math
import re
from collections import defaultdict

from eval.bambara_udpipe import parse_dep_tree_string
from eval.evaluate import kendall_tau_word_order, CATEGORY_TO_CLAUSE, strip_bambara_tones
from eval.bambara_treebank_stats import pos_bigram_coverage, dep_triple_coverage

_RULE_NUM_PREFIX = re.compile(r'^rule\d+_')
_STABLE_PREFIXES = (
    'content_question', 'ccomp_dire', 'ccomp_savoir', 'ccomp',
    'experiencer', 'freq', 'refl_posture', 'refl_actif',
    'refl_accidentel', 'refl_idiom',
    'venir_de', 'verb_serial', 'impersonnel', 'comitative',
    'noun_phrase_have', 'noun_phrase', 'interrogative',
    'existential_absolute', 'existential_localized',
    'existential_nominal', 'past_have', 'future_have',
    'modal_pouvoir', 'privative', 'relative', 'prohibitive',
    'restrictive', 'optatif', 'temporal_np', 'reciprocal',
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _chrf_score(hyp: str, ref: str, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF) simplifié, compatible sacrebleu."""
    if not hyp or not ref:
        return 0.0
    try:
        from sacrebleu.metrics import CHRF
        return CHRF(beta=beta).sentence_score(hyp, [ref]).score / 100.0
    except Exception:
        return 0.0


def _chrfpp_score(hyp: str, ref: str, beta: float = 2.0) -> float:
    """chrF++ : chrF (n-grammes de caractères) + n-grammes de MOTS (word_order=2,
    ajoute unigrammes+bigrammes de mots au score) — seule différence avec chrF,
    donne un signal sur l'ordre des mots que chrF (caractères seuls) n'a pas."""
    if not hyp or not ref:
        return 0.0
    try:
        from sacrebleu.metrics import CHRF
        return CHRF(beta=beta, word_order=2).sentence_score(hyp, [ref]).score / 100.0
    except Exception:
        return 0.0


def _load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _find_latest_csv() -> str | None:
    files = sorted(glob.glob('resultats_bambara_*.csv'), reverse=True)
    return files[0] if files else None


def _safe_float(val: str) -> float | None:
    try:
        v = float(val)
        return v if not math.isnan(v) else None
    except (ValueError, TypeError):
        return None


def _parse_bertscore_f1(val: str) -> float | None:
    """Parse le F1 (3e champ) d'une chaîne 'P|R|F1' stockée par
    test_phrases.py (kuma_bertscore_prf / backtrans_bertscore_prf)."""
    if not val or not isinstance(val, str) or '|' not in val:
        return None
    parts = val.split('|')
    if len(parts) != 3:
        return None
    return _safe_float(parts[2])


# Marqueurs TAM négatifs — chargés depuis TamConfig (KG), PAS une liste
# tapée à la main : la précédente ('tùn ma', 'tùn tɛ', 'tɛ kà', 'tɛ na',
# 'ma', 'tɛ', 'mán') était incomplète (manquait 'tɛ́nà' futur négatif et
# 'bìlen' conditionnel négatif — ce dernier ne contient même pas tɛ/ma,
# donc invisible à toute heuristique par motif) et partiellement fausse
# ('mán' au lieu de 'man', 'tɛ kà'/'tɛ na' avec espace au lieu de
# 'tɛ́kà'/'tùn tɛ' réels — bug trouvé 2026-07-24, signalé par l'utilisateur
# via des exemples d'arbre bambara réel). Normalisées par
# strip_bambara_tones (tons) et espaces retirés (variantes d'écriture
# 'bɛ́nà'/'bɛ́ nà'/'bɛ na' toutes équivalentes) avant comparaison.
def _load_tam_markers(db):
    rows = db.query(
        "MATCH (t:TamConfig) WHERE t.bm IS NOT NULL AND t.bm <> '' "
        "RETURN t.bm AS bm, t.neg AS neg")
    neg, aff = set(), set()
    for r in rows:
        bm = r.get('bm')
        if not bm:
            continue
        key = strip_bambara_tones(bm).replace(' ', '').lower()
        (neg if r.get('neg') else aff).add(key)
    return neg, aff


_NEG_TAM_KEYS = None
_AFF_TAM_KEYS = None


def _ensure_tam_markers():
    global _NEG_TAM_KEYS, _AFF_TAM_KEYS
    if _NEG_TAM_KEYS is not None:
        return
    try:
        from kg.neo4j_client import Neo4jClient
        db = Neo4jClient()
        _NEG_TAM_KEYS, _AFF_TAM_KEYS = _load_tam_markers(db)
    except Exception as e:
        print(f'    [Polarité] Impossible de charger TamConfig depuis le KG ({e}) — figure ignorée')
        _NEG_TAM_KEYS, _AFF_TAM_KEYS = set(), set()


def _has_neg_marker(text: str) -> bool:
    """Cherche un marqueur TAM négatif dans `text`, par TOKEN (pas sousséquence de caractères — un marqueur court comme 'ma' matcherait sinon
    à l'intérieur de mots sans rapport, ex. 'dama', 'kuma'). Tons retirés
    (bɛ́nà ~ bɛnà) ; teste chaque token seul ET chaque paire de tokens
    consécutifs joints (gère 'bɛ́nà' en un seul mot ET 'tùn ma' en deux)."""
    toks = [strip_bambara_tones(t).lower() for t in text.split()]
    for t in toks:
        if t in _NEG_TAM_KEYS:
            return True
    for a, b in zip(toks, toks[1:]):
        if (a + b) in _NEG_TAM_KEYS:
            return True
    return False


def _polarity_match(hyp: str, ref: str) -> bool | None:
    """Polarité (affirmatif/négatif) calculée directement sur les chaînes,
    symétriquement pour les 3 systèmes — voir commentaire d'appel."""
    if not hyp or not ref:
        return None
    _ensure_tam_markers()
    if not _NEG_TAM_KEYS:
        return None
    ref_neg = _has_neg_marker(ref)
    hyp_neg = _has_neg_marker(hyp)
    return ref_neg == hyp_neg


# ── Compute metrics ────────────────────────────────────────────────────────────

def _bleu_score(hyp: str, ref: str) -> float:
    """Simplified BLEU-4 via sacrebleu."""
    if not hyp or not ref:
        return 0.0
    try:
        from sacrebleu.metrics import BLEU
        return BLEU(effective_order=4).sentence_score(hyp, [ref]).score / 100.0
    except Exception:
        return 0.0


def _normalize_for_exm(s: str) -> str:
    """Normalisation pour EXM (exact match) tolérant : casse, espaces
    multiples/en bord de chaîne, ponctuation finale — PAS les diacritiques
    tonals (a/á/à/â restent distincts : porteurs de sens en bambara, une
    différence de ton n'est pas un simple artefact de formatage)."""
    import re
    s = (s or '').strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = s.strip(' .,!?;:')
    return s


def _bleu_char_score(hyp: str, ref: str) -> float:
    """BLEU au niveau CARACTÈRE (sacrebleu tokenize='char') plutôt que mot :
    plus robuste à un simple suffixe manquant/en trop (ex: 'téri' vs 'tériw'
    — zéro n-gramme de MOT partagé sur ce token, mais la majorité des
    n-grammes de CARACTÈRES restent partagés). Utile pour le bambara, langue
    agglutinante-ish où ce genre de bug morphologique est fréquent (-w
    pluriel, -len statif, -man épithète, cf. bugs corrigés cette session)."""
    if not hyp or not ref:
        return 0.0
    try:
        from sacrebleu.metrics import BLEU
        return BLEU(tokenize='char', effective_order=True).sentence_score(hyp, [ref]).score / 100.0
    except Exception:
        return 0.0


def compute_metrics(rows: list[dict]) -> list[dict]:
    """Ajoute les champs métriques à chaque ligne."""
    enriched = []
    for r in rows:
        ref   = (r.get('bambara_attendu') or '').strip()
        kuma  = (r.get('bambara_obtenu')  or '').strip()
        goog  = (r.get('google_translate') or '').strip()
        nllb  = (r.get('nllb_translate')   or '').strip()
        bm_dep = r.get('bambara_dep_tree', '').strip()

        r['chrf_kuma']   = _chrf_score(kuma, ref)  if (kuma and ref) else None
        r['chrf_google'] = _chrf_score(goog, ref)  if (goog and ref) else None
        r['chrf_nllb']   = _chrf_score(nllb, ref)  if (nllb and ref) else None

        r['chrfpp_kuma']   = _chrfpp_score(kuma, ref)  if (kuma and ref) else None
        r['chrfpp_google'] = _chrfpp_score(goog, ref)  if (goog and ref) else None
        r['chrfpp_nllb']   = _chrfpp_score(nllb, ref)  if (nllb and ref) else None

        r['bleu_kuma']   = _bleu_score(kuma, ref)  if (kuma and ref) else None
        r['bleu_google'] = _bleu_score(goog, ref)  if (goog and ref) else None
        r['bleu_nllb']   = _bleu_score(nllb, ref)  if (nllb and ref) else None

        r['bleuchar_kuma']   = _bleu_char_score(kuma, ref)  if (kuma and ref) else None
        r['bleuchar_google'] = _bleu_char_score(goog, ref)  if (goog and ref) else None
        r['bleuchar_nllb']   = _bleu_char_score(nllb, ref)  if (nllb and ref) else None

        # EXM strict (égalité exacte après .strip(), même critère que
        # 'statut'=='PASS' pour Kuma dans test_phrases.py) — recalculé
        # explicitement ici pour Google/NLLB (qui n'ont pas de colonne
        # 'statut' dédiée), afin que les 3 systèmes soient comparés au
        # même standard plutôt que Kuma seul ayant un EXM.
        r['exact_kuma']   = (1 if kuma.strip() == ref.strip() else 0) if (kuma and ref) else None
        r['exact_google'] = (1 if goog.strip() == ref.strip() else 0) if (goog and ref) else None
        r['exact_nllb']   = (1 if nllb.strip() == ref.strip() else 0) if (nllb and ref) else None

        # EXM normalisé (casse/espaces/ponctuation finale tolérés, cf.
        # _normalize_for_exm) : capture les cas où la traduction est
        # correcte mais diffère de la référence par un détail de
        # formatage sans intérêt linguistique (ex: point final en trop/
        # manquant, casse de première lettre).
        ref_norm = _normalize_for_exm(ref)
        r['exact_kuma_norm']   = (1 if _normalize_for_exm(kuma) == ref_norm else 0) if (kuma and ref) else None
        r['exact_google_norm'] = (1 if _normalize_for_exm(goog) == ref_norm else 0) if (goog and ref) else None
        r['exact_nllb_norm']   = (1 if _normalize_for_exm(nllb) == ref_norm else 0) if (nllb and ref) else None

        r['cam_kuma']    = _safe_float(r.get('camembert_kuma'))
        r['cam_google']  = _safe_float(r.get('camembert_google'))
        r['cam_nllb']    = _safe_float(r.get('camembert_nllb'))

        # METEOR déjà calculé bambara-vs-bambara direct par test_phrases.py
        # (colonnes meteor_* présentes dans le CSV) — pas de recalcul ici.
        r['meteor_kuma']   = _safe_float(r.get('meteor_kuma'))
        r['meteor_google'] = _safe_float(r.get('meteor_google'))
        r['meteor_nllb']   = _safe_float(r.get('meteor_nllb'))

        # Kendall τ de réordonnancement (eval/evaluate.py::kendall_tau_word_order,
        # décision utilisateur 2026-07-20 : réintégré dans les figures/rapport).
        # Échelle NATIVE [-1,+1] — isolé de METEOR : ne mesure QUE l'ordre des
        # mots communs à hyp/ref (pénalité neutre p=0.5 pour un mot non
        # apparié), indépendant du choix lexical lui-même.
        r['kendall_kuma']   = kendall_tau_word_order(kuma, ref) if (kuma and ref) else None
        r['kendall_google'] = kendall_tau_word_order(goog, ref) if (goog and ref) else None
        r['kendall_nllb']   = kendall_tau_word_order(nllb, ref) if (nllb and ref) else None

        # BERTScore F1 : déjà calculé et stocké comme chaîne "P|R|F1" par
        # test_phrases.py (kuma_bertscore_prf / backtrans_bertscore_prf),
        # jamais parsé jusqu'ici par eval_report.py — seul le F1 (3e champ)
        # nous intéresse pour le rapport/les tests de significativité
        # (fixé 2026-07-24, cf. décision utilisateur : implémenter plutôt
        # que retirer du papier).
        r['bertf1_kuma']   = _parse_bertscore_f1(r.get('bertscore_prf_kuma'))
        r['bertf1_google'] = _parse_bertscore_f1(r.get('bertscore_prf_google'))
        r['bertf1_nllb']   = _parse_bertscore_f1(r.get('bertscore_prf_nllb'))

        # Polarité (affirmatif/négatif) : calculée directement sur les
        # chaînes générée/référence, symétriquement pour les 3 systèmes —
        # pas de dépendance au tree interne de Kuma (qui rendrait la
        # métrique incomparable pour Google/NLLB, lesquels n'ont pas de
        # tree). Un marqueur négatif présent dans hyp doit correspondre à
        # sa présence dans ref (fixé 2026-07-24, métrique décrite en
        # Section 4.2 mais jamais implémentée jusqu'ici).
        r['polarity_kuma']   = _polarity_match(kuma, ref) if (kuma and ref) else None
        r['polarity_google'] = _polarity_match(goog, ref) if (goog and ref) else None
        r['polarity_nllb']   = _polarity_match(nllb, ref) if (nllb and ref) else None

        # Plausibilité structurelle vs UD_Bambara-CRB (gold), indépendante du
        # français : le bigramme POS / triplet de dépendance de l'arbre kuma
        # est-il attesté dans du vrai bambara annoté ?
        bm_toks = parse_dep_tree_string(bm_dep)
        r['pos_bigram_cov'] = pos_bigram_coverage(bm_toks)
        r['dep_triple_cov'] = dep_triple_coverage(bm_toks)

        enriched.append(r)
    return enriched


def _avg(vals: list) -> float | None:
    filtered = [v for v in vals if v is not None]
    return sum(filtered) / len(filtered) if filtered else None


def global_summary(rows: list[dict]) -> dict:
    return {
        'n':             len(rows),
        'exact_kuma':    _avg([r['exact_kuma']  for r in rows]),
        'exact_google':  _avg([r['exact_google'] for r in rows]),
        'exact_nllb':    _avg([r['exact_nllb']  for r in rows]),
        'exact_kuma_norm':   _avg([r['exact_kuma_norm']   for r in rows]),
        'exact_google_norm': _avg([r['exact_google_norm'] for r in rows]),
        'exact_nllb_norm':   _avg([r['exact_nllb_norm']   for r in rows]),
        'chrf_kuma':     _avg([r['chrf_kuma']   for r in rows]),
        'chrf_google':   _avg([r['chrf_google'] for r in rows]),
        'chrf_nllb':     _avg([r['chrf_nllb']   for r in rows]),
        'chrfpp_kuma':   _avg([r['chrfpp_kuma']   for r in rows]),
        'chrfpp_google': _avg([r['chrfpp_google'] for r in rows]),
        'chrfpp_nllb':   _avg([r['chrfpp_nllb']   for r in rows]),
        'bleu_kuma':     _avg([r['bleu_kuma']   for r in rows]),
        'bleu_google':   _avg([r['bleu_google'] for r in rows]),
        'bleu_nllb':     _avg([r['bleu_nllb']   for r in rows]),
        'bleuchar_kuma':   _avg([r['bleuchar_kuma']   for r in rows]),
        'bleuchar_google': _avg([r['bleuchar_google'] for r in rows]),
        'bleuchar_nllb':   _avg([r['bleuchar_nllb']   for r in rows]),
        'cam_kuma':      _avg([r['cam_kuma']     for r in rows]),
        'cam_google':    _avg([r['cam_google']  for r in rows]),
        'cam_nllb':      _avg([r['cam_nllb']    for r in rows]),
        'meteor_kuma':   _avg([r['meteor_kuma']   for r in rows]),
        'meteor_google': _avg([r['meteor_google'] for r in rows]),
        'meteor_nllb':   _avg([r['meteor_nllb']   for r in rows]),
        'kendall_kuma':   _avg([r['kendall_kuma']   for r in rows]),
        'kendall_google': _avg([r['kendall_google'] for r in rows]),
        'kendall_nllb':   _avg([r['kendall_nllb']   for r in rows]),
        'bertf1_kuma':    _avg([r['bertf1_kuma']   for r in rows]),
        'bertf1_google':  _avg([r['bertf1_google'] for r in rows]),
        'bertf1_nllb':    _avg([r['bertf1_nllb']   for r in rows]),
        'polarity_kuma':   _avg([1.0 if r['polarity_kuma']   else (0.0 if r['polarity_kuma']   is not None else None) for r in rows]),
        'polarity_google': _avg([1.0 if r['polarity_google'] else (0.0 if r['polarity_google'] is not None else None) for r in rows]),
        'polarity_nllb':   _avg([1.0 if r['polarity_nllb']   else (0.0 if r['polarity_nllb']   is not None else None) for r in rows]),
        'pos_bigram_cov': _avg([r['pos_bigram_cov'] for r in rows]),
        'dep_triple_cov': _avg([r['dep_triple_cov'] for r in rows]),
    }


def _category_family(cat: str) -> str:
    """Regroupe une catégorie fine ('equative_sing', 'equative_plur',
    'equative_neg'...) sous sa FAMILLE grammaticale — TEST_CASES compte 230
    catégories fines, illisibles dans un même graphique (labels superposés/
    microscopiques). Le détail fin reste disponible dans le CSV brut pour
    qui en a besoin.

    Le numéro 'ruleN_' est un artefact d'organisation du corpus de test
    (numérotation arbitraire attribuée par catégorie créée), pas un concept
    grammatical — le garder produirait des groupes opaques ('rule1',
    'rule2'...) au lieu de rejoindre la vraie famille grammaticale (ex.
    'rule2_prohibitive_dem' -> 'prohibitive'). Retiré AVANT le matching de
    préfixe, même logique que clause_group() dans eval_final.py (bug
    trouvé 2026-07-24 : eval_report.py utilisait encore l'ancien split
    naïf sur '_', laissant 'rule1'/'rule2'/... visibles dans les figures)."""
    cat = cat or 'unknown'
    if cat in CATEGORY_TO_CLAUSE:
        return CATEGORY_TO_CLAUSE[cat]
    stripped = _RULE_NUM_PREFIX.sub('', cat)
    for prefix in _STABLE_PREFIXES:
        if stripped.startswith(prefix):
            return prefix
    return stripped.split('_')[0]


def by_category(rows: list[dict]) -> dict[str, dict]:
    """Regroupe par FAMILLE de catégorie (cf. _category_family), pas par
    catégorie fine — 230 catégories fines produisaient des graphiques
    illisibles (bug trouvé 2026-07-20, décision utilisateur : 'les
    figures pour catégories sont invisibles')."""
    cats: dict[str, list] = defaultdict(list)
    for r in rows:
        cats[_category_family(r.get('categorie', 'unknown'))].append(r)
    result = {}
    for cat, rs in cats.items():
        result[cat] = {
            'n':           len(rs),
            'exact_kuma':  _avg([r['exact_kuma']  for r in rs]),
            'exact_google': _avg([r['exact_google'] for r in rs]),
            'exact_nllb':   _avg([r['exact_nllb']  for r in rs]),
            'exact_kuma_norm':   _avg([r['exact_kuma_norm']   for r in rs]),
            'exact_google_norm': _avg([r['exact_google_norm'] for r in rs]),
            'exact_nllb_norm':   _avg([r['exact_nllb_norm']   for r in rs]),
            'chrf_kuma':   _avg([r['chrf_kuma']   for r in rs]),
            'chrf_google': _avg([r['chrf_google'] for r in rs]),
            'chrf_nllb':   _avg([r['chrf_nllb']   for r in rs]),
            'chrfpp_kuma':   _avg([r['chrfpp_kuma']   for r in rs]),
            'chrfpp_google': _avg([r['chrfpp_google'] for r in rs]),
            'chrfpp_nllb':   _avg([r['chrfpp_nllb']   for r in rs]),
            'bleu_kuma':   _avg([r['bleu_kuma']   for r in rs]),
            'bleu_google': _avg([r['bleu_google'] for r in rs]),
            'bleu_nllb':   _avg([r['bleu_nllb']   for r in rs]),
            'bleuchar_kuma':   _avg([r['bleuchar_kuma']   for r in rs]),
            'bleuchar_google': _avg([r['bleuchar_google'] for r in rs]),
            'bleuchar_nllb':   _avg([r['bleuchar_nllb']   for r in rs]),
            'cam_kuma':    _avg([r['cam_kuma']     for r in rs]),
            'cam_google':  _avg([r['cam_google']  for r in rs]),
            'cam_nllb':    _avg([r['cam_nllb']    for r in rs]),
            'meteor_kuma':   _avg([r['meteor_kuma']   for r in rs]),
            'meteor_google': _avg([r['meteor_google'] for r in rs]),
            'meteor_nllb':   _avg([r['meteor_nllb']   for r in rs]),
            'kendall_kuma':   _avg([r['kendall_kuma']   for r in rs]),
            'kendall_google': _avg([r['kendall_google'] for r in rs]),
            'kendall_nllb':   _avg([r['kendall_nllb']   for r in rs]),
            'bertf1_kuma':    _avg([r['bertf1_kuma']   for r in rs]),
            'bertf1_google':  _avg([r['bertf1_google'] for r in rs]),
            'bertf1_nllb':    _avg([r['bertf1_nllb']   for r in rs]),
            'polarity_kuma':   _avg([1.0 if r['polarity_kuma']   else (0.0 if r['polarity_kuma']   is not None else None) for r in rs]),
            'polarity_google': _avg([1.0 if r['polarity_google'] else (0.0 if r['polarity_google'] is not None else None) for r in rs]),
            'polarity_nllb':   _avg([1.0 if r['polarity_nllb']   else (0.0 if r['polarity_nllb']   is not None else None) for r in rs]),
            'pos_bigram_cov': _avg([r['pos_bigram_cov'] for r in rs]),
            'dep_triple_cov': _avg([r['dep_triple_cov'] for r in rs]),
        }
    return result


# ── Figures ────────────────────────────────────────────────────────────────────

COLORS = {
    'kuma':   '#2196F3',   # bleu
    'google': '#FF9800',   # orange
    'nllb':   '#4CAF50',   # vert
}


def _bar_group(ax, categories, vals_k, vals_g, vals_n, title, ylabel,
               fmt='{:.2f}', x_rot=45, xlim=(0, 1.15),
               label_vals_k=None, label_vals_g=None, label_vals_n=None):
    """Barres HORIZONTALES : une catégorie (famille) par ligne, valeur en
    abscisse. Bien plus lisible que des barres verticales dès qu'il y a
    plus d'une dizaine de catégories — labels lus normalement au lieu
    d'être pivotés/rétrécis jusqu'à devenir illisibles (bug trouvé
    2026-07-20, décision utilisateur : "les figures pour catégories sont
    invisibles", 230 catégories fines/~50 familles bien trop nombreuses
    pour tenir sur un axe horizontal)."""
    import numpy as np
    # Ordre alphabétique INVERSÉ : matplotlib empile les barh de bas en
    # haut, donc inverser ici restitue l'ordre alpha standard de haut en
    # bas à l'écran.
    categories = list(categories)[::-1]
    vals_k = list(vals_k)[::-1]
    vals_g = list(vals_g)[::-1]
    vals_n = list(vals_n)[::-1]
    label_vals_k = list(label_vals_k)[::-1] if label_vals_k is not None else vals_k
    label_vals_g = list(label_vals_g)[::-1] if label_vals_g is not None else vals_g
    label_vals_n = list(label_vals_n)[::-1] if label_vals_n is not None else vals_n

    y = np.arange(len(categories))
    h = 0.26

    def _plot(offset, vals, label_vals, label, color):
        xs = [v if v is not None else 0 for v in vals]
        bars = ax.barh(y + offset, xs, h, label=label, color=color, alpha=0.85)
        for bar, v, lv in zip(bars, vals, label_vals):
            # v != 0 (pas v > 0) : nécessaire pour les métriques signées
            # (ex: Kendall τ ∈ [-1,+1]) dont une valeur négative légitime
            # ne doit pas perdre son étiquette. Le texte affiché (lv) peut
            # différer de la position tracée (v) — ex: Kendall τ tracé
            # remis à l'échelle (τ+1)/2 mais étiqueté avec sa valeur
            # native, pour cohérence avec fig_radar et éviter d'afficher
            # deux nombres différents pour la même statistique.
            if v is not None and v != 0:
                pad = 0.01 if v >= 0 else -0.01
                ax.text(bar.get_width() + pad,
                        bar.get_y() + bar.get_height() / 2,
                        fmt.format(lv), ha=('left' if v >= 0 else 'right'),
                        va='center', fontsize=6)

    _plot(-h, vals_k, label_vals_k, 'Kuma-MT', COLORS['kuma'])
    _plot( 0, vals_g, label_vals_g, 'Google',  COLORS['google'])
    _plot(+h, vals_n, label_vals_n, 'NLLB',    COLORS['nllb'])

    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(ylabel, fontsize=9)
    ax.set_xlim(*xlim)
    if xlim[0] < 0:
        ax.axvline(0, color='black', linewidth=0.6, alpha=0.5)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='x', alpha=0.3)


def fig_global(summary: dict, out: str = 'fig_eval_global.png'):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Évaluation globale — Kuma-MT vs Google vs NLLB', fontsize=13, fontweight='bold')

    systems = ['Kuma-MT', 'Google', 'NLLB']
    colors  = [COLORS['kuma'], COLORS['google'], COLORS['nllb']]

    # ── Exact Match ──
    ax = axes[0, 0]
    val = summary['exact_kuma']
    ax.bar(['Kuma-MT'], [val or 0], color=COLORS['kuma'], alpha=0.85, width=0.4)
    if val is not None:
        ax.text(0, (val or 0) + 0.01, f'{val:.1%}', ha='center', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title('Exact Match %', fontsize=10, fontweight='bold')
    ax.set_ylabel('Score', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # ── chrF ──
    ax = axes[0, 1]
    vals = [summary['chrf_kuma'], summary['chrf_google'], summary['chrf_nllb']]
    bars = ax.bar(systems, [v or 0 for v in vals], color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        if v is not None:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title('chrF (character F-score)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Score', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # ── BLEU ──
    ax = axes[0, 2]
    vals = [summary['bleu_kuma'], summary['bleu_google'], summary['bleu_nllb']]
    bars = ax.bar(systems, [v or 0 for v in vals], color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        if v is not None:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title('BLEU (word-level)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Score', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # ── CamemBERT ──
    ax = axes[1, 0]
    vals = [summary['cam_kuma'], summary['cam_google'], summary['cam_nllb']]
    bars = ax.bar(systems, [v or 0 for v in vals], color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        if v is not None:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title('CamemBERT (BM→FR semantic)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Cosine similarity', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # ── Plausibilité structurelle vs UD_Bambara-CRB ──
    ax = axes[1, 1]
    val = summary['pos_bigram_cov']
    ax.bar(['Kuma-MT'], [val or 0], color=COLORS['kuma'], alpha=0.85, width=0.4)
    if val is not None:
        ax.text(0, (val or 0) + 0.01, f'{val:.1%}', ha='center', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title('POS bigram coverage (vs UD_Bambara-CRB)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Score', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # ── Summary text ──
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = (
        f"Total phrases: {summary['n']}\n\n"
        f"Kuma-MT dominant sur :\n"
        f"• Exact Match: {summary['exact_kuma'] or 0:.1%}\n"
        f"• chrF: {summary['chrf_kuma'] or 0:.3f}\n"
        f"• BLEU: {summary['bleu_kuma'] or 0:.3f}\n"
        f"• CamemBERT: {summary['cam_kuma'] or 0:.3f}\n"
        f"• POS bigram cov. (UD_Bambara-CRB): {summary['pos_bigram_cov'] or 0:.1%}"
    )
    ax.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_global_all_metrics(summary: dict, out: str = 'fig_eval_global_all_metrics.png'):
    """Vue d'ensemble GLOBALE (pas par catégorie) des 3 systèmes sur
    TOUTES les métriques en un seul graphique lisible (barres
    horizontales groupées, même moteur que les figures par-catégorie) :
    Exact Match, EXM normalisé, chrF, chrF++, BLEU, BLEU (char),
    CamemBERT, METEOR, Kendall τ — décision utilisateur 2026-07-20.
    Kendall τ est nativement sur [-1,+1] — REMIS À L'ÉCHELLE (τ+1)/2 pour
    partager l'axe 0-1 des autres métriques (même traitement que fig_radar) ;
    fig_eval_kendall_cat.png garde la valeur native pour qui la veut."""
    import matplotlib.pyplot as plt

    metric_labels = ['Exact Match', 'EXM norm.', 'chrF', 'chrF++',
                      'BLEU', 'BLEU (char)', 'CamemBERT', 'BERTScore F1', 'METEOR', 'Kendall τ*']
    key_fns = [
        lambda s: f'exact_{s}',
        lambda s: f'exact_{s}_norm',
        lambda s: f'chrf_{s}',
        lambda s: f'chrfpp_{s}',
        lambda s: f'bleu_{s}',
        lambda s: f'bleuchar_{s}',
        lambda s: f'cam_{s}',
        lambda s: f'bertf1_{s}',
        lambda s: f'meteor_{s}',
    ]

    def _vals(system):
        vals = [summary.get(kf(system)) for kf in key_fns]
        label_vals = list(vals)
        kendall = summary.get(f'kendall_{system}')
        vals.append((kendall + 1) / 2 if kendall is not None else None)
        # Étiquette = valeur NATIVE de Kendall τ (pas la version remise à
        # l'échelle utilisée pour la position de la barre) — cohérent avec
        # fig_radar, évite d'afficher deux nombres différents pour la même
        # statistique (bug trouvé 2026-07-24).
        label_vals.append(kendall)
        return vals, label_vals

    vk, lk = _vals('kuma')
    vg, lg = _vals('google')
    vn, ln = _vals('nllb')

    fig, ax = plt.subplots(figsize=(10, 6.5))
    _bar_group(ax, metric_labels, vk, vg, vn,
               f"Vue d'ensemble globale — Kuma-MT vs Google vs NLLB (n={summary['n']})",
               'Score', fmt='{:.3f}',
               label_vals_k=lk, label_vals_g=lg, label_vals_n=ln)
    fig.text(0.02, 0.01, '* Kendall τ remis à l\'échelle (τ+1)/2 — voir '
             'fig_eval_kendall_cat.png pour la valeur native [-1,+1]',
             fontsize=6, color='gray')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_chrf_by_cat(cat_stats: dict, out: str = 'fig_eval_chrf_cat.png'):
    import matplotlib.pyplot as plt

    cats = sorted(cat_stats)
    vk = [cat_stats[c]['chrf_kuma']   for c in cats]
    vg = [cat_stats[c]['chrf_google'] for c in cats]
    vn = [cat_stats[c]['chrf_nllb']   for c in cats]

    fig, ax = plt.subplots(figsize=(11, max(6, len(cats) * 0.32)))
    _bar_group(ax, cats, vk, vg, vn,
               'chrF par catégorie', 'chrF', fmt='{:.2f}')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_bleu_by_cat(cat_stats: dict, out: str = 'fig_eval_bleu_cat.png'):
    import matplotlib.pyplot as plt

    cats = sorted(cat_stats)
    vk = [cat_stats[c]['bleu_kuma']   for c in cats]
    vg = [cat_stats[c]['bleu_google'] for c in cats]
    vn = [cat_stats[c]['bleu_nllb']   for c in cats]

    fig, ax = plt.subplots(figsize=(11, max(6, len(cats) * 0.32)))
    _bar_group(ax, cats, vk, vg, vn,
               'BLEU par catégorie (word-level)', 'BLEU', fmt='{:.2f}')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_chrfpp_by_cat(cat_stats: dict, out: str = 'fig_eval_chrfpp_cat.png'):
    import matplotlib.pyplot as plt

    cats = sorted(cat_stats)
    vk = [cat_stats[c]['chrfpp_kuma']   for c in cats]
    vg = [cat_stats[c]['chrfpp_google'] for c in cats]
    vn = [cat_stats[c]['chrfpp_nllb']   for c in cats]

    fig, ax = plt.subplots(figsize=(11, max(6, len(cats) * 0.32)))
    _bar_group(ax, cats, vk, vg, vn,
               'chrF++ par catégorie', 'chrF++', fmt='{:.2f}')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_bleuchar_by_cat(cat_stats: dict, out: str = 'fig_eval_bleuchar_cat.png'):
    import matplotlib.pyplot as plt

    cats = sorted(cat_stats)
    vk = [cat_stats[c]['bleuchar_kuma']   for c in cats]
    vg = [cat_stats[c]['bleuchar_google'] for c in cats]
    vn = [cat_stats[c]['bleuchar_nllb']   for c in cats]

    fig, ax = plt.subplots(figsize=(11, max(6, len(cats) * 0.32)))
    _bar_group(ax, cats, vk, vg, vn,
               'BLEU par catégorie (character-level)', 'BLEU (char)', fmt='{:.2f}')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_meteor_by_cat(cat_stats: dict, out: str = 'fig_eval_meteor_cat.png'):
    import matplotlib.pyplot as plt

    cats = [c for c in sorted(cat_stats)
            if any(cat_stats[c][k] is not None
                   for k in ('meteor_kuma', 'meteor_google', 'meteor_nllb'))]
    if not cats:
        print('  [METEOR] Aucune donnée par catégorie — figure ignorée')
        return

    vk = [cat_stats[c]['meteor_kuma']   for c in cats]
    vg = [cat_stats[c]['meteor_google'] for c in cats]
    vn = [cat_stats[c]['meteor_nllb']   for c in cats]

    fig, ax = plt.subplots(figsize=(11, max(6, len(cats) * 0.32)))
    _bar_group(ax, cats, vk, vg, vn,
               'METEOR par catégorie (bambara vs bambara_attendu, direct)', 'METEOR', fmt='{:.2f}')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_kendall_by_cat(cat_stats: dict, out: str = 'fig_eval_kendall_cat.png'):
    """Kendall τ de réordonnancement (eval/evaluate.py::kendall_tau_word_order)
    par catégorie — échelle NATIVE [-1,+1] (xlim dédié, pas le [0,1] partagé
    par les autres métriques)."""
    import matplotlib.pyplot as plt

    cats = [c for c in sorted(cat_stats)
            if any(cat_stats[c][k] is not None
                   for k in ('kendall_kuma', 'kendall_google', 'kendall_nllb'))]
    if not cats:
        print('  [Kendall τ] Aucune donnée par catégorie — figure ignorée')
        return

    vk = [cat_stats[c]['kendall_kuma']   for c in cats]
    vg = [cat_stats[c]['kendall_google'] for c in cats]
    vn = [cat_stats[c]['kendall_nllb']   for c in cats]

    fig, ax = plt.subplots(figsize=(11, max(6, len(cats) * 0.32)))
    _bar_group(ax, cats, vk, vg, vn,
               "Kendall τ (ordre des mots) par catégorie", 'Kendall τ',
               fmt='{:+.2f}', xlim=(-1.15, 1.15))
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_bertf1_by_cat(cat_stats: dict, out: str = 'fig_eval_bertf1_cat.png'):
    """BERTScore F1 (contextuel, glouton — test_phrases.py::kuma_bertscore_prf /
    backtrans_bertscore_prf) par catégorie."""
    import matplotlib.pyplot as plt

    cats = [c for c in sorted(cat_stats)
            if any(cat_stats[c][k] is not None
                   for k in ('bertf1_kuma', 'bertf1_google', 'bertf1_nllb'))]
    if not cats:
        print('  [BERTScore F1] Aucune donnée par catégorie — figure ignorée')
        return

    vk = [cat_stats[c]['bertf1_kuma']   for c in cats]
    vg = [cat_stats[c]['bertf1_google'] for c in cats]
    vn = [cat_stats[c]['bertf1_nllb']   for c in cats]

    fig, ax = plt.subplots(figsize=(11, max(6, len(cats) * 0.32)))
    _bar_group(ax, cats, vk, vg, vn,
               'BERTScore F1 par catégorie', 'BERTScore F1', fmt='{:.2f}')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_polarity_by_cat(cat_stats: dict, out: str = 'fig_eval_polarity_cat.png'):
    """Polarité (affirmatif/négatif, calculée directement sur les chaînes
    générée/référence — cf. _polarity_match) par catégorie."""
    import matplotlib.pyplot as plt

    cats = [c for c in sorted(cat_stats)
            if any(cat_stats[c][k] is not None
                   for k in ('polarity_kuma', 'polarity_google', 'polarity_nllb'))]
    if not cats:
        print('  [Polarité] Aucune donnée par catégorie — figure ignorée')
        return

    vk = [cat_stats[c]['polarity_kuma']   for c in cats]
    vg = [cat_stats[c]['polarity_google'] for c in cats]
    vn = [cat_stats[c]['polarity_nllb']   for c in cats]

    fig, ax = plt.subplots(figsize=(11, max(6, len(cats) * 0.32)))
    _bar_group(ax, cats, vk, vg, vn,
               'Précision de polarité par catégorie', 'Polarité (accord)', fmt='{:.0%}')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_camembert_by_cat(cat_stats: dict, out: str = 'fig_eval_camembert_cat.png'):
    import matplotlib.pyplot as plt

    cats = [c for c in sorted(cat_stats)
            if any(cat_stats[c][k] is not None
                   for k in ('cam_kuma', 'cam_google', 'cam_nllb'))]
    if not cats:
        print('  [CamemBERT] Aucune donnée par catégorie — figure ignorée')
        return

    vk = [cat_stats[c]['cam_kuma']   for c in cats]
    vg = [cat_stats[c]['cam_google'] for c in cats]
    vn = [cat_stats[c]['cam_nllb']   for c in cats]

    fig, ax = plt.subplots(figsize=(11, max(6, len(cats) * 0.32)))
    _bar_group(ax, cats, vk, vg, vn,
               'CamemBERT sim par catégorie (BM→FR, cosine)', 'Cosine sim')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_exact_by_cat(cat_stats: dict, out: str = 'fig_eval_exact_cat.png'):
    import matplotlib.pyplot as plt
    import numpy as np

    cats = sorted(cat_stats)[::-1]  # barh empile bas→haut : inverser pour lire haut→bas
    vals = [cat_stats[c]['exact_kuma'] for c in cats]

    fig, ax = plt.subplots(figsize=(9, max(6, len(cats) * 0.32)))
    y = np.arange(len(cats))
    bars = ax.barh(y, [v or 0 for v in vals], color=COLORS['kuma'], alpha=0.85)
    for bar, v in zip(bars, vals):
        if v is not None and v > 0:
            ax.text(bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{v:.0%}', ha='left', va='center', fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.set_title('Kuma-MT — Exact Match % par catégorie', fontsize=11, fontweight='bold')
    ax.set_xlabel('Exact Match %', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_exact3_by_cat(cat_stats: dict, out: str = 'fig_eval_exact3_cat.png'):
    """Exact Match strict, 3 systèmes (contrairement à fig_exact_by_cat qui
    ne montre que Kuma — Google/NLLB recalculés en comparaison directe à
    bambara_attendu dans compute_metrics, cf. décision utilisateur
    2026-07-20)."""
    import matplotlib.pyplot as plt

    cats = sorted(cat_stats)
    vk = [cat_stats[c]['exact_kuma']   for c in cats]
    vg = [cat_stats[c]['exact_google'] for c in cats]
    vn = [cat_stats[c]['exact_nllb']   for c in cats]

    fig, ax = plt.subplots(figsize=(11, max(6, len(cats) * 0.32)))
    _bar_group(ax, cats, vk, vg, vn,
               'Exact Match (strict) par catégorie', 'Exact Match', fmt='{:.0%}')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_exactnorm_by_cat(cat_stats: dict, out: str = 'fig_eval_exactnorm_cat.png'):
    """Exact Match normalisé (casse/espaces/ponctuation finale tolérés,
    cf. _normalize_for_exm) — pas les diacritiques tonals, porteurs de sens."""
    import matplotlib.pyplot as plt

    cats = sorted(cat_stats)
    vk = [cat_stats[c]['exact_kuma_norm']   for c in cats]
    vg = [cat_stats[c]['exact_google_norm'] for c in cats]
    vn = [cat_stats[c]['exact_nllb_norm']   for c in cats]

    fig, ax = plt.subplots(figsize=(11, max(6, len(cats) * 0.32)))
    _bar_group(ax, cats, vk, vg, vn,
               'Exact Match (normalisé) par catégorie', 'Exact Match norm.', fmt='{:.0%}')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_radar(summary: dict, out: str = 'fig_eval_radar.png'):
    """Radar (spider) comparant les 3 systèmes sur toutes les métriques
    globales à échelle 0-1 partagée (EXM, EXM norm., chrF, chrF++, BLEU,
    BLEU-char, CamemBERT, METEOR, Kendall τ). EXM recalculé pour les 3
    systèmes par comparaison directe à bambara_attendu (cf. compute_metrics),
    donc comparable ici contrairement à une ancienne version Kuma-only.
    Kendall τ est nativement sur [-1,+1] (cf. fig_kendall_by_cat pour sa
    propre échelle) — REMIS À L'ÉCHELLE (τ+1)/2 uniquement pour partager
    l'axe 0-1 du radar avec les autres métriques ; le rapport texte et la
    figure par-catégorie dédiée gardent la valeur native."""
    import matplotlib.pyplot as plt
    import numpy as np

    axes_labels = ['EXM', 'EXM norm.', 'chrF', 'chrF++', 'BLEU', 'BLEU (char)',
                    'CamemBERT', 'METEOR', 'Kendall τ*']
    # summary key patterns aren't all uniform ('exact_kuma' vs
    # 'exact_kuma_norm', suffix after the system name for the norm variant)
    # — construit explicitement plutôt que de forcer un seul gabarit.
    key_fns = [
        lambda s: f'exact_{s}',
        lambda s: f'exact_{s}_norm',
        lambda s: f'chrf_{s}',
        lambda s: f'chrfpp_{s}',
        lambda s: f'bleu_{s}',
        lambda s: f'bleuchar_{s}',
        lambda s: f'cam_{s}',
        lambda s: f'meteor_{s}',
    ]

    # Les 2 premiers axes (EXM, EXM norm.) s'affichent en %, le reste en
    # décimal — indexé sur la position dans key_fns, pas sur un test de
    # substring fragile.
    _pct_axes = {0, 1}

    def _vals(system):
        # (radius plot value, texte affiché sur la valeur NATIVE — pas
        # rescale — sinon Kendall τ afficherait sa version (τ+1)/2 au lieu
        # du τ réel, source de confusion).
        plot_vals, labels = [], []
        for i, kf in enumerate(key_fns):
            v = summary.get(kf(system))
            plot_vals.append(v or 0)
            if v is None:
                labels.append('—')
            elif i in _pct_axes:
                labels.append(f'{v:.0%}')
            else:
                labels.append(f'{v:.2f}')
        kendall = summary.get(f'kendall_{system}')
        plot_vals.append((kendall + 1) / 2 if kendall is not None else 0)
        labels.append(f'{kendall:+.2f}' if kendall is not None else '—')
        return plot_vals, labels

    kuma_vals,   kuma_labels   = _vals('kuma')
    google_vals, google_labels = _vals('google')
    nllb_vals,   nllb_labels   = _vals('nllb')

    n = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    # Décalage radial par système pour éviter que les 3 étiquettes se
    # chevauchent quand les valeurs sont proches sur un même axe.
    _label_offsets = {'Kuma-MT': 0.06, 'Google': -0.03, 'NLLB': -0.11}

    for vals, labels, label, color in [
        (kuma_vals, kuma_labels, 'Kuma-MT', COLORS['kuma']),
        (google_vals, google_labels, 'Google', COLORS['google']),
        (nllb_vals, nllb_labels, 'NLLB', COLORS['nllb']),
    ]:
        v = vals + vals[:1]
        ax.plot(angles, v, color=color, linewidth=2, label=label)
        ax.fill(angles, v, color=color, alpha=0.15)
        ax.scatter(angles[:-1], vals, color=color, s=14, zorder=5)
        for ang, val, txt in zip(angles[:-1], vals, labels):
            ax.text(ang, val + _label_offsets[label], txt,
                    color=color, fontsize=7.5, fontweight='bold',
                    ha='center', va='center')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title('Kuma-MT vs Google vs NLLB — toutes métriques (0-1)',
                 fontsize=11, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.12), fontsize=9)
    ax.grid(alpha=0.3)
    fig.text(0.02, 0.02, '* Kendall τ remis à l\'échelle (τ+1)/2 pour partager l\'axe 0-1 — '
             'voir fig_eval_kendall_cat.png pour la valeur native [-1,+1]',
             fontsize=6, color='gray')

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


# ── Console summary ─────────────────────────────────────────────────────────────

def print_summary(summary: dict, csv_path: str):
    w = 72
    print(f"\n{'='*w}")
    print(f"  RAPPORT D'ÉVALUATION — {os.path.basename(csv_path)}")
    print(f"  {summary['n']} phrases analysées")
    print(f"{'='*w}")

    def _row(label, kuma, google, nllb, fmt='{:.3f}'):
        def _f(v): return fmt.format(v) if v is not None else '  —  '
        print(f"  {label:<20} Kuma: {_f(kuma):<10} Google: {_f(google):<10} NLLB: {_f(nllb)}")

    print(f"\n  {'Métrique':<20} {'Kuma-MT':<14} {'Google':<14} {'NLLB'}")
    print(f"  {'-'*68}")
    _row('Exact Match',     summary['exact_kuma'], summary['exact_google'], summary['exact_nllb'], fmt='{:.1%}')
    _row('Exact Match norm', summary['exact_kuma_norm'], summary['exact_google_norm'], summary['exact_nllb_norm'], fmt='{:.1%}')
    _row('chrF',            summary['chrf_kuma'], summary['chrf_google'], summary['chrf_nllb'])
    _row('chrF++',          summary['chrfpp_kuma'], summary['chrfpp_google'], summary['chrfpp_nllb'])
    _row('BLEU',            summary['bleu_kuma'], summary['bleu_google'], summary['bleu_nllb'])
    _row('BLEU (char)',     summary['bleuchar_kuma'], summary['bleuchar_google'], summary['bleuchar_nllb'])
    _row('CamemBERT sim',   summary['cam_kuma'],  summary['cam_google'],  summary['cam_nllb'])
    _row('METEOR',          summary['meteor_kuma'], summary['meteor_google'], summary['meteor_nllb'])
    _row('BERTScore F1',    summary['bertf1_kuma'], summary['bertf1_google'], summary['bertf1_nllb'])
    _row('Kendall τ (order)', summary['kendall_kuma'], summary['kendall_google'], summary['kendall_nllb'], fmt='{:+.3f}')
    _row('Polarity acc.',   summary['polarity_kuma'], summary['polarity_google'], summary['polarity_nllb'], fmt='{:.1%}')
    print(f"{'='*w}\n")

    print(f"\n{'='*w}")
    print(f"  PLAUSIBILITÉ STRUCTURELLE vs UD_Bambara-CRB (gold, sens ignoré)")
    print(f"{'='*w}")
    pbc = summary['pos_bigram_cov']
    print(f"  {'POS bigram coverage':<24} {(f'{pbc:.1%}' if pbc is not None else '—')}")
    dtc = summary['dep_triple_cov']
    print(f"  {'Dep triple coverage':<24} {(f'{dtc:.1%}' if dtc is not None else '—')}")
    print(f"{'='*w}\n")
    print(f"  (fraction des bigrammes POS / triplets tête-deprel-dépendant de")
    print(f"   l'arbre bambara kuma qui sont attestés dans le vrai bambara du")
    print(f"   treebank UD_Bambara-CRB — indépendant du français et du sens)")


# ── Main ────────────────────────────────────────────────────────────────────────

def run_report(csv_path: str):
    print(f"\n  Chargement : {csv_path}")
    rows = _load_csv(csv_path)
    print(f"  {len(rows)} phrases chargées")

    print('  Calcul des métriques (chrF) …')
    rows = compute_metrics(rows)

    summary  = global_summary(rows)
    cat_stat = by_category(rows)

    print_summary(summary, csv_path)

    print('  Génération des figures …')
    fig_global(summary)
    fig_global_all_metrics(summary)
    fig_chrf_by_cat(cat_stat)
    fig_chrfpp_by_cat(cat_stat)
    fig_bleu_by_cat(cat_stat)
    fig_bleuchar_by_cat(cat_stat)
    fig_camembert_by_cat(cat_stat)
    fig_meteor_by_cat(cat_stat)
    fig_kendall_by_cat(cat_stat)
    fig_bertf1_by_cat(cat_stat)
    fig_polarity_by_cat(cat_stat)
    fig_exact_by_cat(cat_stat)
    fig_exact3_by_cat(cat_stat)
    fig_exactnorm_by_cat(cat_stat)
    fig_radar(summary)
    print('\n   15 figures générées\n')


if __name__ == '__main__':
    args = sys.argv[1:]

    if '--run' in args:
        # Lance test_phrases --run puis génère le rapport
        no_cam = '--no-camembert' in args
        import importlib.util, types
        spec = importlib.util.spec_from_file_location(
            'test_phrases', os.path.join(os.path.dirname(__file__), 'test_phrases.py'))
        tp = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(tp)  # type: ignore

        sys.path.insert(0, '.')
        from pipeline.translation_engine import TranslationEngine
        from kg.neo4j_client import Neo4jClient
        db     = Neo4jClient()
        engine = TranslationEngine(db)
        csv_path = tp.run_tests(
            translate_fn=lambda s: engine.translate(s),
            no_camembert=no_cam,
        )
        run_report(csv_path)

    else:
        # CSV fourni ou auto-détecté
        csv_arg = next((a for a in args if a.endswith('.csv')), None)
        if csv_arg:
            csv_path = csv_arg
        else:
            csv_path = _find_latest_csv()

        if not csv_path or not os.path.exists(csv_path):
            print('Usage : python eval_report.py [fichier.csv]')
            print('        python eval_report.py --run [--no-camembert]')
            sys.exit(1)

        run_report(csv_path)
