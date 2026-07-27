"""
eval_final.py — Évaluation finale Kuma-MT vs Google vs NLLB-200
================================================================
Source de référence : "FINAL Data.csv" (365 phrases, 230 catégories).
- phrase_fr        : source française
- bambara_attendu   : RÉFÉRENCE gold pour toutes les évaluations bambara
- google_translate / nllb_translate : sorties baseline déjà figées
- Kuma-MT est RE-TRADUIT ici avec le moteur actuel (tous les fixes du jour
  inclus) — bambara_obtenu/camembert_kuma du CSV d'origine sont ignorés,
  seuls phrase_fr/bambara_attendu/google_translate/nllb_translate servent
  de données fixes.

Génère :
  - eval_final_<ts>.csv            détail par phrase
  - eval_final_<ts>_summary.json   métriques agrégées (paper-ready)
  - figures_final/*.png            toutes les figures (voir eval_plots.py
    restauré + nouvelles figures 3-voies Kuma/Google/NLLB)

232 catégories → trop nombreuses pour un bar chart par catégorie (demande
explicite de l'utilisateur) : regroupées via CATEGORY_TO_CLAUSE (evaluate.py)
+ un fallback générique par préfixe/suffixe pour les catégories non listées.
"""
import os
import sys
import csv
import json
import datetime
import io
import re
import contextlib
import warnings
warnings.filterwarnings('ignore')

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'eval'))

from evaluate import (CATEGORY_TO_CLAUSE, ud_health, bambara_word_order_ok,
                       tree_metrics, corpus_metrics, chrf_score, chrfpp_score,
                       bleu_char_score, kendall_tau_word_order, extract_tam_from_ref,
                       strip_bambara_tones, exm_match_normalized)
import test_phrases as tp


# ── Regroupement macro pour les 230 catégories ───────────────────────────────
_STRIP_SUFFIXES = [
    '_neg_plur', '_plur_neg', '_passe_neg', '_neg_passe',
    '_neg', '_plur', '_passe', '_past', '_futur', '_future', '_pres', '_present',
    '_abstract', '_material', '_age', '_root', '_ccomp', '_appos',
    '_interrogative', '_question', '_marker', '_alternative',
    '_dative', '_locatif', '_locative', '_genitive', '_coi', '_abs',
    '_emphase', '_dem', '_object', '_subject', '_topic', '_iobj',
    '_real_subject', '_splitting', '_inversion', '_modal_xcomp',
]


_RULE_NUM_PREFIX = re.compile(r'^rule\d+_')


def clause_group(category: str) -> str:
    """Groupe macro pour une catégorie fine. Priorité : mapping explicite
    (evaluate.CATEGORY_TO_CLAUSE) > préfixe reconnu > catégorie brute.

    Le numéro 'ruleN_' est un artefact d'organisation du corpus de test
    (numérotation arbitraire attribuée par catégorie créée), pas un concept
    grammatical — l'afficher tel quel dans les figures (ex: 'rule6 (n=5)')
    est incompréhensible. On le retire AVANT le matching de préfixe, pour
    que ex. 'rule2_prohibitive_dem'/'rule2_prohibitive_object' rejoignent le
    groupe 'prohibitive' comme n'importe quelle autre catégorie prohibitive,
    au lieu de finir dans un groupe 'rule2' opaque. Fixé 2026-07-10."""
    if category in CATEGORY_TO_CLAUSE:
        return CATEGORY_TO_CLAUSE[category]
    stripped_category = _RULE_NUM_PREFIX.sub('', category)
    # Préfixes stables de familles grammaticales connues
    # (ccomp_dire_x -> 'ccomp_dire', content_question_x -> 'content_question'…)
    for prefix in ('content_question', 'ccomp_dire', 'ccomp_savoir', 'ccomp',
                   'experiencer', 'freq', 'refl_posture', 'refl_actif',
                   'refl_accidentel', 'refl_idiom',
                   'venir_de', 'verb_serial', 'impersonnel', 'comitative',
                   'noun_phrase_have', 'noun_phrase', 'interrogative',
                   'existential_absolute', 'existential_localized',
                   'existential_nominal', 'past_have', 'future_have',
                   'modal_pouvoir', 'privative', 'relative', 'prohibitive',
                   'restrictive', 'optatif', 'temporal_np', 'reciprocal'):
        if stripped_category.startswith(prefix):
            return prefix
    stripped = stripped_category
    for suf in _STRIP_SUFFIXES:
        if stripped.endswith(suf):
            stripped = stripped[:-len(suf)]
            break
    return stripped


def load_final_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FINAL Data.csv')
    rows_in = load_final_csv(csv_path)
    total = len(rows_in)
    print(f"Chargé {total} phrases depuis FINAL Data.csv\n")

    print("Chargement du moteur Kuma-MT + tagger + CamemBERT…")
    from pipeline.translation_engine import TranslationEngine
    from pipeline.spacy_parser import SpacyParser
    from kg.neo4j_client import Neo4jClient
    db = Neo4jClient()
    eng = TranslationEngine(db)
    tagger = SpacyParser(db)

    results = []
    for i, row in enumerate(rows_in, 1):
        phrase   = row['phrase_fr']
        expected = row['bambara_attendu'].strip()
        category = row['categorie']
        goog_bm  = row.get('google_translate', '').strip()
        nllb_bm  = row.get('nllb_translate', '').strip()

        print(f"\r  [{i:3d}/{total}] {phrase[:55]:<55}", end='', flush=True)

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                res = eng.translate(phrase)
            output = res.get('bambara', '').strip()
            tree_meta = res.get('tree', {})
            kuma_tokens = res.get('tokens', [])
        except Exception as e:
            output = f'[ERR: {e}]'
            tree_meta = {}
            kuma_tokens = []

        exm_kuma  = (output == expected)
        exm_goog  = (goog_bm == expected)
        exm_nllb  = (nllb_bm == expected)

        # EXM tolérant au ton + variantes pronominales (décision 2026-07-10) :
        # l'EXM strict sous-estime fortement Google/NLLB qui ne marquent
        # jamais les tons (vérifié : EXM strict Google 1.1% -> normalisé
        # 10.1%, NLLB 0.3% -> 6.8%). Complète l'EXM strict, ne le remplace pas.
        exm_kuma_norm = exm_match_normalized(output, expected)
        exm_goog_norm = exm_match_normalized(goog_bm, expected) if goog_bm else False
        exm_nllb_norm = exm_match_normalized(nllb_bm, expected) if nllb_bm else False

        chrf_kuma = chrf_score(output, expected)
        chrf_goog = chrf_score(goog_bm, expected) if goog_bm else None
        chrf_nllb = chrf_score(nllb_bm, expected) if nllb_bm else None

        chrfpp_kuma = chrfpp_score(output, expected)
        chrfpp_goog = chrfpp_score(goog_bm, expected) if goog_bm else None
        chrfpp_nllb = chrfpp_score(nllb_bm, expected) if nllb_bm else None

        bleu_kuma_s = bleu_char_score(output, expected)
        bleu_goog_s = bleu_char_score(goog_bm, expected) if goog_bm else None
        bleu_nllb_s = bleu_char_score(nllb_bm, expected) if nllb_bm else None

        # cam_google/cam_nllb réutilisés depuis FINAL Data.csv : ils ne dépendent
        # que de phrase_fr + google_translate/nllb_translate (figés, inchangés),
        # pas du moteur Kuma actuel. Les recalculer coûterait une back-traduction
        # mot-à-mot (NLLB generate() + réseau Google) par mot et par phrase — sur
        # 365 phrases, un test a mesuré ~50s/phrase, soit ~5h pour la suite
        # complète. Seul cam_kuma (direct, sans back-traduction) est recalculé.
        with contextlib.redirect_stdout(io.StringIO()):
            # CamemBERT (mot-à-mot isolé) et BERTScore (contextuel, glouton)
            # sont deux métriques distinctes par construction (2026-07-09) :
            # CamemBERT embeddings chaque mot seul (score_kuma_word_level),
            # BERTScore encode la phrase entière en une passe (bertscore_prf,
            # via kuma_bertscore_prf). Google/NLLB : P/R/F1 laissés à None,
            # demanderaient la même back-traduction mot-à-mot coûteuse
            # (~50s/phrase) qu'on a évitée pour cam_google/cam_nllb.
            cam_kuma = tp.score_kuma_word_level(kuma_tokens)
            p_kuma, r_kuma, f1_kuma = tp.kuma_bertscore_prf(kuma_tokens, phrase, tagger)
        cam_goog = _f(row.get('camembert_google'))
        cam_nllb = _f(row.get('camembert_nllb'))
        p_goog = r_goog = f1_goog = None
        p_nllb = r_nllb = f1_nllb = None

        tm = tree_metrics(tree_meta, expected, category)
        udh = ud_health(phrase)

        # TAM (3 systèmes, méthodologie symétrique) : expected_tam est le
        # marqueur TAM grammaticalement correct pour LE VERBE FRANÇAIS, tel
        # que résolu par le KG pendant la traduction Kuma (tree_meta['tam'],
        # ex: 'être' présent équatif → 'yé') — indépendant de ce que chaque
        # système a produit. On vérifie ensuite si ce marqueur apparaît dans
        # CHAQUE sortie (Kuma, Google, NLLB), avec la même vérification.
        # Remplace l'ancien tam_ok de tree_metrics (ref_tam vs tree_tam) qui,
        # pour Kuma, comparait sa propre sortie à son propre état interne —
        # pas une vraie validation externe.
        #
        # Comparaison SANS TONS (strip_bambara_tones) : Google Translate et
        # NLLB ne produisent jamais de marquage tonal (toujours 'ye', jamais
        # 'yé') — une comparaison sur chaîne tonale ferait échouer à tort
        # 100% des cas pour ces deux systèmes, même quand ils ont choisi le
        # bon marqueur (vérifié : "Je suis enseignant" → Google/NLLB
        # produisent bien 'ye' mais la comparaison tonale exacte échouait).
        expected_tam = tree_meta.get('tam', '')

        def _tam_present(sentence):
            if not expected_tam:
                return False
            norm_tam = strip_bambara_tones(expected_tam).lower()
            norm_sentence = strip_bambara_tones(sentence).lower()
            return f' {norm_tam} ' in f' {norm_sentence} '

        tam_ok_kuma = _tam_present(output)
        tam_ok_goog = _tam_present(goog_bm) if goog_bm else None
        tam_ok_nllb = _tam_present(nllb_bm) if nllb_bm else None

        # Word-order (S-TAM-V) : uniquement Kuma-MT. Le check repose sur le
        # ClauseTemplate KG (clause_type + valeurs réelles des slots S/TAM/O/V
        # dans tree_meta) — données qui n'existent que pour la sortie Kuma ;
        # Google/NLLB ne passent jamais par l'arbre interne, donc non évaluable
        # pour ces deux systèmes (pas de fallback heuristique plus faible).
        wo_kuma = bambara_word_order_ok(tree_meta, output, eng.rule_engine.grammar)

        # Kendall τ de réordonnancement, par phrase (décision 2026-07-11) :
        # ordre des tokens communs à output/reference vs ordre dans reference,
        # PAS une corrélation chrF↔chrF++ (qui ne mesurait que l'accord entre
        # deux métriques d'overlap, jamais l'ordre réel des mots).
        tau_wo_kuma = kendall_tau_word_order(output, expected)
        tau_wo_goog = kendall_tau_word_order(goog_bm, expected) if goog_bm else None
        tau_wo_nllb = kendall_tau_word_order(nllb_bm, expected) if nllb_bm else None

        results.append({
            'idx': i, 'category': category, 'group': clause_group(category),
            'source': phrase, 'reference': expected,
            'kuma': output, 'google': goog_bm, 'nllb': nllb_bm,
            'exm_kuma': exm_kuma, 'exm_google': exm_goog, 'exm_nllb': exm_nllb,
            'exm_kuma_norm': exm_kuma_norm, 'exm_google_norm': exm_goog_norm,
            'exm_nllb_norm': exm_nllb_norm,
            'chrf_kuma': chrf_kuma, 'chrf_google': chrf_goog, 'chrf_nllb': chrf_nllb,
            'chrfpp_kuma': chrfpp_kuma, 'chrfpp_google': chrfpp_goog, 'chrfpp_nllb': chrfpp_nllb,
            'bleu_kuma': bleu_kuma_s, 'bleu_google': bleu_goog_s, 'bleu_nllb': bleu_nllb_s,
            'cam_kuma': cam_kuma, 'cam_google': cam_goog, 'cam_nllb': cam_nllb,
            'p_kuma': p_kuma, 'r_kuma': r_kuma, 'f1_kuma': f1_kuma,
            'p_google': p_goog, 'r_google': r_goog, 'f1_google': f1_goog,
            'p_nllb': p_nllb, 'r_nllb': r_nllb, 'f1_nllb': f1_nllb,
            'wo_kuma': wo_kuma,
            'clause_type_ok': tm['clause_type_ok'],
            'tam_ok_kuma': tam_ok_kuma, 'tam_ok_google': tam_ok_goog, 'tam_ok_nllb': tam_ok_nllb,
            'expected_tam': expected_tam,
            'neg_ok': tm['neg_ok'], 'slot_S': tm['slot_S_filled'],
            'slot_V': tm['slot_V_filled'], 'slot_O': tm['slot_O_filled'],
            'tree_clause': tree_meta.get('clause_type', ''),
            'ud_score': udh.get('ud_score'),
            'ud_single_root': udh.get('single_root'),
            'ud_root_pos_ok': udh.get('root_pos_ok'),
            'ud_has_subject': udh.get('has_subject'),
            'ud_connected': udh.get('connected'),
            'ud_no_dep_dep': udh.get('no_dep_dep'),
            'tau_wo_kuma': tau_wo_kuma, 'tau_wo_google': tau_wo_goog, 'tau_wo_nllb': tau_wo_nllb,
        })

    print()

    out_csv = f'eval_final_{ts}.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"CSV détail : {out_csv}")

    # BLEU corpus-level (sacrebleu) pour les 3 systèmes
    refs = [r['reference'] for r in results]
    bleu_kuma = corpus_metrics([r['kuma'] for r in results], refs)
    bleu_goog = corpus_metrics([r['google'] for r in results], refs)
    bleu_nllb = corpus_metrics([r['nllb'] for r in results], refs)

    def _avg(lst):
        v = [x for x in lst if x is not None]
        return round(sum(v) / len(v), 3) if v else None

    def _pct(lst):
        v = [x for x in lst if x is not None]
        return round(sum(v) / len(v) * 100, 2) if v else None

    # τ de réordonnancement (word order vs référence), moyenné par phrase —
    # remplace l'ancien τ chrF↔chrF++ qui mesurait l'accord entre deux
    # métriques d'overlap plutôt que l'ordre réel des mots (décision 2026-07-11).
    def _tau_avg(key):
        v = [r[key] for r in results if r[key] is not None]
        return round(sum(v) / len(v), 3) if v else 0.0

    tau_kuma = _tau_avg('tau_wo_kuma')
    tau_goog = _tau_avg('tau_wo_google')
    tau_nllb = _tau_avg('tau_wo_nllb')

    summary = {
        'timestamp': ts, 'n_phrases': total,
        'kuma': {
            'EXM': _pct([r['exm_kuma'] for r in results]),
            'EXM_normalized': _pct([r['exm_kuma_norm'] for r in results]),
            'chrF_sentence_avg': _avg([r['chrf_kuma'] for r in results]),
            'chrF_corpus': bleu_kuma['chrf_corpus'],
            'chrFpp_sentence_avg': _avg([r['chrfpp_kuma'] for r in results]),
            'chrFpp_corpus': bleu_kuma['chrfpp_corpus'],
            'BLEU_sentence_avg': _avg([r['bleu_kuma'] for r in results]),
            'BLEU_corpus': bleu_kuma['bleu_corpus'],
            'CamemBERT_avg': _avg([r['cam_kuma'] for r in results]),
            'BERTScore_P': _avg([r['p_kuma'] for r in results]),
            'BERTScore_R': _avg([r['r_kuma'] for r in results]),
            'BERTScore_F1': _avg([r['f1_kuma'] for r in results]),
            'WordOrder_STAMV': _pct([r['wo_kuma'] for r in results]),
            'CTA': _pct([r['clause_type_ok'] for r in results]),
            'TAM_acc': _pct([r['tam_ok_kuma'] for r in results if r['expected_tam']]),
            'NEG_acc': _pct([r['neg_ok'] for r in results]),
            'tau_word_order': tau_kuma,
        },
        'google': {
            'EXM': _pct([r['exm_google'] for r in results]),
            'EXM_normalized': _pct([r['exm_google_norm'] for r in results]),
            'chrF_sentence_avg': _avg([r['chrf_google'] for r in results]),
            'chrF_corpus': bleu_goog['chrf_corpus'],
            'chrFpp_sentence_avg': _avg([r['chrfpp_google'] for r in results]),
            'chrFpp_corpus': bleu_goog['chrfpp_corpus'],
            'BLEU_sentence_avg': _avg([r['bleu_google'] for r in results]),
            'BLEU_corpus': bleu_goog['bleu_corpus'],
            'CamemBERT_avg': _avg([r['cam_google'] for r in results]),
            'TAM_acc': _pct([r['tam_ok_google'] for r in results
                            if r['expected_tam'] and r['tam_ok_google'] is not None]),
            'tau_word_order': tau_goog,
        },
        'nllb': {
            'EXM': _pct([r['exm_nllb'] for r in results]),
            'EXM_normalized': _pct([r['exm_nllb_norm'] for r in results]),
            'chrF_sentence_avg': _avg([r['chrf_nllb'] for r in results]),
            'chrF_corpus': bleu_nllb['chrf_corpus'],
            'chrFpp_sentence_avg': _avg([r['chrfpp_nllb'] for r in results]),
            'chrFpp_corpus': bleu_nllb['chrfpp_corpus'],
            'BLEU_sentence_avg': _avg([r['bleu_nllb'] for r in results]),
            'BLEU_corpus': bleu_nllb['bleu_corpus'],
            'CamemBERT_avg': _avg([r['cam_nllb'] for r in results]),
            'TAM_acc': _pct([r['tam_ok_nllb'] for r in results
                            if r['expected_tam'] and r['tam_ok_nllb'] is not None]),
            'tau_word_order': tau_nllb,
        },
        'ud_score_avg_pct': _pct([
            (r['ud_score'] / 5 if r['ud_score'] is not None else None)
            for r in results]),
    }
    out_json = f'eval_final_{ts}_summary.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"JSON résumé : {out_json}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return out_csv, out_json


if __name__ == '__main__':
    main()
