"""
evaluate.py — Évaluation quantitative de Kuma-MT (FR→Bambara)
==============================================================
Métriques calculées :
  - EXM   : Exact Match Rate          (test_phrases.py comme référence)
  - chrF  : Character n-gram F-score  (sacrebleu)
  - P@1   : Embedding Precision@1     (lemme ∈ sens_fr du top-1 candidat)
  - KGH   : KG Hit Rate               (% exact match vs embedding)
  - CTA   : Clause Type Accuracy      (inférée depuis la catégorie de test)
  - τ     : Kendall's tau             (word-order reordering vs référence, par phrase)

Usage :
  python evaluate.py                  # éval complète silencieuse
  python evaluate.py --verbose        # affiche chaque phrase
  python evaluate.py --cat simple     # filtre sur une catégorie
  python evaluate.py --no-translate   # rapport sur la suite de test uniquement
"""

import sys
import csv
import json
import datetime
import argparse
import os
import re
import traceback
import unicodedata
from collections import defaultdict

from eval.bambara_ud_reference import build_reference_ud_tree


def classify_error(e: Exception) -> tuple:
    """Catégorise une exception de traduction pour le diagnostic (pas de
    hiérarchie d'exceptions dédiée dans le pipeline — tout est 'Exception'
    générique) : on retient le TYPE de l'exception + le frame le plus profond
    de la traceback (fichier:ligne dans la fonction où elle a été levée),
    plus fiable qu'un heuristique sur le texte du message. Retourne
    (error_type, error_where)."""
    error_type = type(e).__name__
    tb = traceback.extract_tb(e.__traceback__)
    if tb:
        frame = tb[-1]
        error_where = f"{os.path.basename(frame.filename)}:{frame.lineno} in {frame.name}"
    else:
        error_where = '?'
    return error_type, error_where


def strip_bambara_tones(s: str) -> str:
    """Retire les marques tonales (accents combinants : grave/aigu sur les
    voyelles) d'une chaîne bambara. Nécessaire pour comparer un marqueur TAM
    indépendamment du marquage tonal : Google Translate et NLLB ne produisent
    JAMAIS de tons (toujours 'ye', jamais 'yé'), contrairement à Kuma — une
    comparaison exacte sur chaîne tonale ferait donc échouer à tort le TAM
    de ces deux systèmes même quand ils ont choisi le bon marqueur."""
    nfd = unicodedata.normalize('NFD', s)
    return ''.join(c for c in nfd if not unicodedata.combining(c))


# Variantes de forme (pas de ton, pas de synonyme lexical) du pronom 1sg
# sujet bambara : 'n' (forme clitique, KG Pronoun.bm pour 'je'/'moi') vs
# 'ne' (forme pleine, sans ton — 'né' devient déjà 'ne' après
# strip_bambara_tones). Volontairement TRÈS restreint : uniquement des
# variantes de FORME d'un même mot déjà attestées (cf. décision 2026-07-10 —
# EXM ne doit PAS absorber des synonymes lexicaux différents comme
# 'karamɔgɔ'/'karamɔgɔkɛ', c'est le rôle de chrF/chrF++/BERTScore).
_PRONOUN_FORM_VARIANTS = {'ne': 'n'}


def exm_normalize(s: str) -> str:
    """Normalisation pour EXM tolérant : minuscule + sans tons + variantes
    de forme pronominale connues (PAS de synonymes lexicaux)."""
    s = strip_bambara_tones(s or '').lower().strip()
    tokens = [_PRONOUN_FORM_VARIANTS.get(t, t) for t in s.split()]
    return ' '.join(tokens)


def exm_match_normalized(hyp: str, ref: str) -> bool:
    """EXM tolérant au ton et aux variantes de forme pronominale — cf.
    exm_normalize(). Complète (ne remplace pas) l'EXM strict : l'EXM strict
    reste informatif (seul Kuma produit une orthographe tonale complète),
    mais sous-estime fortement Google/NLLB qui ne marquent JAMAIS les tons
    (vérifié sur le corpus complet : EXM strict Google 1.1% → 10.1%
    normalisé, NLLB 0.3% → 6.8% normalisé — décision 2026-07-10)."""
    return exm_normalize(hyp) == exm_normalize(ref)


# ── Mapping catégorie → clause_type attendu ───────────────────────────────────
CATEGORY_TO_CLAUSE = {
    'equative_sing': 'equative', 'equative_plur': 'equative',
    'equative_neg': 'equative', 'equative_past': 'equative',
    'equative_past_neg': 'equative', 'equative_relative': 'equative',
    'qualitative': 'qualitative', 'qualitative_neg': 'qualitative',
    'qualitative_past': 'qualitative', 'qualitative_past_neg': 'qualitative',
    'locative': 'locative', 'locative_neg': 'locative',
    'locative_plur': 'locative', 'locative_adv': 'locative',
    'statif': 'statif', 'statif_plur': 'statif',
    'past_intransitive': 'simple', 'past_intransitive_plur': 'simple',
    'past_transitive': 'simple', 'past_transitive_plur': 'simple',
    'past_neg': 'simple', 'habitude': 'simple',
    'simple': 'simple', 'simple_neg': 'simple', 'simple_plur': 'simple',
    'volitif': 'simple', 'volitif_neg': 'simple',
    'progressif': 'simple', 'futur': 'simple', 'futur_neg': 'simple',
    'intrans_action_present_neg': 'simple', 'intrans_action_progressif': 'simple',
    'intrans_action_passe_pos': 'simple', 'intrans_action_passe_neg': 'simple',
    'intrans_action_hab_neg': 'simple', 'intrans_action_futur': 'simple',
    'identificatory': 'identificatory', 'identificatory_neg': 'identificatory',
    'identificatory_appos': 'identificatory',
    'presentative': 'presentative', 'presentative_plur': 'presentative',
    'existential_absolute': 'existential_absolute',
    'existential_absolute_neg': 'existential_absolute',
    'existential_localized': 'existential_localized',
    'existential_localized_neg': 'existential_localized',
    'existential_nominal_complex': 'existential_nominal',
    'noun_phrase_have_material': 'noun_phrase_have',
    'noun_phrase_have_material_neg': 'noun_phrase_have',
    'noun_phrase_have_material_plur': 'noun_phrase_have',
    'noun_phrase_have_abstract': 'noun_phrase_have',
    'noun_phrase_have_abstract_neg': 'noun_phrase_have',
    'noun_phrase_have_abstract_plur': 'noun_phrase_have',
    'noun_phrase_have_age': 'noun_phrase_have',
    'interrogative': 'interrogative', 'interrogative_motion': 'interrogative',
    'interrogative_question_marker': 'interrogative',
    'interrogative_alternative': 'interrogative',
    'content_question_who': 'content_question',
    'content_question_who_inversion': 'content_question',
    'content_question_what': 'content_question',
    'content_question_where': 'content_question',
    'content_question_when': 'content_question',
    'content_question_why': 'content_question',
    'content_question_how': 'content_question',
    'content_question_which': 'content_question',
    'content_question_which_noun': 'content_question',
    'content_question_how_much': 'content_question',
    'verb_serial_dative': 'verb_serial',
    'relative_topic': 'relative_topic', 'relative_topic_neg': 'relative_topic',
    'comitative': 'comitative', 'comitative_genitive': 'comitative',
    'comitative_action': 'simple',
    'reciprocal': 'reciprocal',
    'conditional': 'conditional',
    'temporal_subordinator': 'temporal',
    'comparative': 'comparative', 'reported_comparative': 'simple',
    'nummod_subject': 'simple', 'nummod_object': 'simple',
    'nummod_locative': 'locative',
    'noun_phrase_alienable': 'noun_phrase',
    'noun_phrase_genitive': 'noun_phrase',
    'noun_phrase_coord': 'noun_phrase',
    'imperative': 'imperative', 'prohibitive': 'prohibitive',
    'ownership': 'ownership',
    'infinitive': 'infinitive', 'infinitive_neg': 'infinitive',
    'infinitive_list': 'infinitive',
    'privative_pron': 'privative_pred', 'privative_clause': 'simple',
    'privative_noun': 'noun_phrase', 'privative_verb': 'simple',
    'deictique': 'presentative', 'deictique_plur': 'presentative',
}

CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}

# TAM markers connus — pour extraction depuis la référence Bambara.
# Repris EXACTEMENT de la table (:TamConfig) du KG (rules/rule_engine.py,
# requête "MATCH (t:TamConfig) RETURN t.tense, t.neg, t.bm") — ne pas
# ajouter de marqueur ici sans l'y vérifier d'abord. 'dòn' (copule
# identificatoire/présentative), 'ka' (obligation/optatif) et 'kàna'
# (prohibitif) ne sont PAS des TAM : bug fixé 2026-07-08, ils faisaient
# rater à tort la vérification d'ordre des mots (bambara_word_order_ok)
# sur toute clause équative/existentielle/identificatoire/statif se
# terminant légitimement par 'dòn' sans verbe après (ex: "n de dòn",
# "búuru bɛ" [existentiel, bɛ = prédicat lui-même, pas un TAM ici — voir
# note dans bambara_word_order_ok]).
_TAM_MARKERS = [
    'tùn yé', 'tùn ma', 'tùn bɛ', 'tùn tɛ',
    'bɛ kà', 'tɛ kà', 'bɛ na', 'tɛ na',
    'yé', 'ma', 'bɛ', 'tɛ', 'mán',
]

# ── UD Parse Health ───────────────────────────────────────────────────────────

_SPACY_NLP = None

def _get_nlp():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            import spacy
            _SPACY_NLP = spacy.load('fr_dep_news_trf')
        except Exception:
            pass
    return _SPACY_NLP


def ud_health(sentence: str) -> dict:
    """5 checks structurels sur le parse spaCy français.
    Score 0–5 : nombre de checks réussis.

    1. single_root   — exactement un token ROOT
    2. root_pos_ok   — ROOT est VERB / AUX / NOUN / PROPN
    3. has_subject   — au moins un nsubj / expl / nsubj:pass
    4. connected     — pas de cycle trivial (tout token hors ROOT a head ≠ lui-même)
    5. no_dep_dep    — aucun token avec dep='dep' (relation non résolue)
    """
    nlp = _get_nlp()
    if nlp is None:
        return {'ud_score': None, 'ud_pct': None}
    doc    = nlp(sentence)
    roots  = [t for t in doc if t.dep_ == 'ROOT']
    checks = {
        'single_root':  len(roots) == 1,
        'root_pos_ok':  bool(roots) and roots[0].pos_ in ('VERB', 'AUX', 'NOUN', 'PROPN', 'ADJ'),
        'has_subject':  any(t.dep_ in ('nsubj', 'expl', 'nsubj:pass', 'expl:subj') for t in doc),
        'connected':    all(t.head.i != t.i or t.dep_ == 'ROOT' for t in doc),
        'no_dep_dep':   not any(t.dep_ == 'dep' for t in doc),
    }
    score = sum(checks.values())
    return {**checks, 'ud_score': score, 'ud_pct': round(score / 5 * 100, 1)}


# ── Bambara word-order (contre le template KG du clause_type) ────────────────
# Ancienne version : heuristique générique "TAM précédé d'un sujet ET suivi
# d'un verbe" sur la chaîne finale — cassait sur existential_absolute
# ('{O} bɛ yan', bɛ pas forcément suivi d'un verbe) et sur les TAM composés
# ('tùn bɛ'). Remplacée par une comparaison directe à la règle KG du
# clause_type (ClauseTemplate, via lookup_clause_template) : on relit l'ordre
# des slots {S}/{TAM}/{O}/{V}/... prescrit par le template, et on vérifie que
# les VALEURS RÉELLES de ces slots (déjà connues via tree_meta — S/TAM/O/V
# tels que construits pendant la traduction, donc le TAM composé est déjà la
# bonne chaîne complète) apparaissent dans le même ordre dans la sortie.

def bambara_word_order_ok(tree_meta: dict, output: str, G_kg: dict) -> bool:
    """Vérifie que l'ordre des slots (S, TAM, O, V, ...) dans la sortie
    respecte le ClauseTemplate KG du clause_type détecté.
    Retourne None si aucun template KG n'existe pour ce clause_type
    (non évaluable — exclu de la moyenne, même convention que p1/kgh)."""
    from rules.kg_rule_engine import lookup_clause_template

    clause_type = tree_meta.get('clause_type', '')
    template = lookup_clause_template(clause_type, G_kg)
    if not template:
        return None

    slot_order = re.findall(r'\{(\w+)\}', template)

    last_pos = -1
    for slot in slot_order:
        value = (tree_meta.get(slot) or tree_meta.get(slot.upper())
                  or tree_meta.get(slot.lower()) or '')
        value = value.strip() if isinstance(value, str) else ''
        if not value:
            continue   # slot vide (ex: S='' en existential_absolute) → ignoré
        pos = output.find(value, last_pos + 1)
        if pos == -1 or pos <= last_pos:
            return False
        last_pos = pos
    return True


def ud_structure_ok(tree_meta: dict, output: str):
    """Vérifie que la sortie contient les mots attendus (S/TAM/O/V/OBL...)
    dans le bon ordre de surface ET avec les bons rôles UD (POS+deprel),
    selon le schéma documenté par Aplonova & Tyers (build_reference_ud_tree).
    Retourne None si le clause_type n'est pas couvert (non évaluable — même
    convention que p1/kgh, exclu de la moyenne) ; sinon True/False selon que
    la résolution a réussi (liste non vide) ou échoué (liste vide — échec
    réel de structure, compte comme faux, n'est PAS exclu de la moyenne)."""
    ref = build_reference_ud_tree(tree_meta, output)
    if ref is None:
        return None
    return bool(ref)


# ── Métriques corpus (chrF, chrF++, BLEU) ────────────────────────────────────

def corpus_metrics(hyps: list, refs: list) -> dict:
    """Calcule chrF, chrF++ et BLEU (car-level) au niveau corpus."""
    try:
        from sacrebleu.metrics import CHRF, BLEU
        chrf_obj   = CHRF(word_order=0)
        chrfpp_obj = CHRF(word_order=2)
        bleu_obj   = BLEU(tokenize='char')
        return {
            'chrf_corpus':   round(chrf_obj.corpus_score(hyps, [refs]).score, 2),
            'chrfpp_corpus': round(chrfpp_obj.corpus_score(hyps, [refs]).score, 2),
            'bleu_corpus':   round(bleu_obj.corpus_score(hyps, [refs]).score, 2),
        }
    except Exception as e:
        return {'chrf_corpus': None, 'chrfpp_corpus': None, 'bleu_corpus': None}


def extract_tam_from_ref(reference: str) -> str:
    """Extrait le marqueur TAM depuis une traduction Bambara de référence."""
    ref = reference.strip()
    for tam in _TAM_MARKERS:
        if f' {tam} ' in f' {ref} ':
            return tam
    return ''


# ── Métriques tree-level ───────────────────────────────────────────────────────

def tree_metrics(tree_meta: dict, reference: str, category: str) -> dict:
    """
    Calcule les métriques structurelles à partir du tree snapshot.

    Retourne :
      clause_type_ok  : clause_type détecté correspond au type attendu
      tam_ok          : TAM détecté correspond au TAM de la référence
      neg_ok          : négation correctement détectée
      slot_S_filled   : sujet non vide dans le tree
      slot_V_filled   : verbe non vide dans le tree
      slot_O_filled   : objet non vide (pour les phrases transitives)
    """
    expected_clause = CATEGORY_TO_CLAUSE.get(category, '')
    detected_clause = tree_meta.get('clause_type', '')
    clause_ok = bool(expected_clause and detected_clause
                     and expected_clause in detected_clause)

    # TAM : comparer tree['tam'] avec le TAM extrait de la référence
    ref_tam = extract_tam_from_ref(reference)
    tree_tam = tree_meta.get('tam', '')
    tam_ok = bool(ref_tam and tree_tam and ref_tam == tree_tam)

    # Négation : est-ce que la référence contient tɛ/ma ?
    ref_neg = any(f' {m} ' in f' {reference} '
                  for m in ('tɛ', 'ma', 'mán', 'kàna'))
    tree_neg = bool(tree_meta.get('neg', False))
    neg_ok = (ref_neg == tree_neg)

    # Slot fill : mesure de complétude structurelle
    slot_S = bool(tree_meta.get('S', '').strip())
    slot_V = bool(tree_meta.get('V', '').strip())
    slot_O = bool(tree_meta.get('O', '').strip())

    return {
        'clause_type_ok': clause_ok,
        'tam_ok':         tam_ok,
        'neg_ok':         neg_ok,
        'slot_S_filled':  slot_S,
        'slot_V_filled':  slot_V,
        'slot_O_filled':  slot_O,
        'ref_tam':        ref_tam,
        'tree_tam':       tree_tam,
    }


# ── Métriques token-level ──────────────────────────────────────────────────────

def embedding_p1(tokens, source_lemmas: dict) -> float:
    """
    P@1 = % de tokens de contenu dont le lemme source apparaît dans sens_fr(top-1).
    source_lemmas : {orig_index: lemma_fr}
    """
    content = [t for t in tokens
               if t.get('pos') in CONTENT_POS
               and t.get('role') == 'content'
               and t.get('sens_fr')
               and t.get('lemma')]
    if not content:
        return None
    hits = sum(
        1 for t in content
        if t['lemma'].lower() in t['sens_fr'].lower()
    )
    return hits / len(content)


def kg_hit_rate(tokens) -> float:
    """KGH = % de tokens de contenu traduits par exact match KG."""
    content = [t for t in tokens
               if t.get('pos') in CONTENT_POS
               and t.get('role') == 'content'
               and t.get('_match_type')]
    if not content:
        return None
    hits = sum(1 for t in content if t.get('_match_type') == 'exact')
    return hits / len(content)


def chrf_score(hypothesis: str, reference: str) -> float:
    """chrF sentence-level score (sacrebleu ≥ 2.x)."""
    try:
        from sacrebleu.metrics import CHRF
        return CHRF(word_order=0).sentence_score(hypothesis, [reference]).score
    except Exception:
        h = set(hypothesis[i:i+3] for i in range(len(hypothesis)-2))
        r = set(reference[i:i+3] for i in range(len(reference)-2))
        if not h or not r:
            return 100.0 if hypothesis.strip() == reference.strip() else 0.0
        p   = len(h & r) / len(h)
        rec = len(h & r) / len(r)
        return 2 * p * rec / (p + rec + 1e-8) * 100


def chrfpp_score(hypothesis: str, reference: str) -> float:
    """chrF++ sentence-level score (chrF + sensibilité à l'ordre des mots,
    word_order=2). Utilisé uniquement pour le Kendall's τ chrF↔chrF++
    (cf. kendall_tau) — pas de fallback manuel, chrF++ nécessite sacrebleu."""
    from sacrebleu.metrics import CHRF
    return CHRF(word_order=2).sentence_score(hypothesis, [reference]).score


def bleu_char_score(hypothesis: str, reference: str) -> float:
    """BLEU (tokenize='char') sentence-level score, pour les scatters
    phrase-par-phrase. effective_order=True : évite l'effondrement à 0 que
    donnerait le BLEU standard dès qu'un seul ordre de n-gramme (souvent le
    4-gramme) a une précision nulle sur une phrase courte — sacrebleu réduit
    alors l'ordre effectivement utilisé au lieu de forcer un facteur nul dans
    le produit géométrique. Vérifié : phrases courtes/non-liées donnent un
    score proportionnel (>0), pas un 0 plat qui écraserait le scatter."""
    from sacrebleu.metrics import BLEU
    return BLEU(tokenize='char', effective_order=True).sentence_score(
        hypothesis, [reference]).score


# ── Kendall's tau ──────────────────────────────────────────────────────────────

def kendall_tau(scores_a: list, scores_b: list) -> float:
    """
    Kendall's τ-b entre deux listes de scores (même ordre de phrases).
    Mesure la concordance de classement entre deux métriques.
    Générique — utilisée hors contexte word-order (cf. kendall_tau_word_order
    pour la métrique de reordering MT, seule utilisation actuelle dans le pipeline).
    """
    if len(scores_a) < 2:
        return 0.0
    from scipy.stats import kendalltau
    tau, _ = kendalltau(scores_a, scores_b)
    return 0.0 if tau != tau else float(tau)   # tau != tau ⇔ NaN (variance nulle)


def kendall_tau_word_order(hyp: str, ref: str, p_penalty: float = 0.5):
    """
    Kendall's τ de réordonnancement, distance étendue façon Fagin/Kumar-
    Vassilvitskii pour comparer deux "rankings" non identiques (ici : les
    tokens de ref et hyp ne sont pas le même ensemble — mots manquants,
    substitués). Décision 2026-07-11, remplace la version qui n'alignait que
    les tokens communs et laissait tomber silencieusement les tokens de
    référence absents de l'hypothèse (ex: 'Musa yé dònso yé' vs Google
    'Musa ye kungotigi ye' → 'dònso' jamais comparé → τ=1.0 malgré un nom
    traduit complètement différent, alors que TOUS les mots de la phrase
    doivent compter dans l'évaluation).

    Univers = tous les tokens (occurrences) de ref ∪ hyp, appariés par
    identité de surface (sans tons) en glouton gauche→droite pour gérer les
    répétitions (ex: les deux 'yé' d'une clause équative appariés dans
    l'ordre où ils apparaissent). Pour CHAQUE paire de tokens distincts de
    cet univers :
      - les deux appariés (positions connues dans ref ET hyp) : distance 0
        si leur ordre relatif concorde entre ref et hyp, 1 sinon ;
      - un apparié + un non-apparié (présent dans un seul des deux textes) :
        pénalité fixe p_penalty (0.5 par défaut — valeur "neutre", celle
        attendue si on ne savait rien de l'ordre relatif) ;
      - les deux non-appariés : pénalité fixe p_penalty également.

    La distance totale est normalisée par le nombre de paires, puis
    reconvertie sur l'échelle τ classique [-1, +1] via 1 − 2×distance
    (distance=0 → τ=+1 ordre parfait ; distance=1 → τ=-1 ordre totalement
    inversé/discordant ; p_penalty=0.5 → contribution nulle à τ, cohérent
    avec son rôle de valeur neutre).

    Retourne None si l'univers a moins de 2 tokens (rien à comparer).
    """
    # Même normalisation que exm_match_normalized (tons + variantes de forme
    # pronominale, ex: 'ne'->'n') pour ne pas pénaliser Google/NLLB sur des
    # différences de forme qui ne changent pas le mot (décision 2026-07-11).
    def _toks(s):
        s = strip_bambara_tones(s or '').lower()
        return [_PRONOUN_FORM_VARIANTS.get(t, t) for t in s.split()]

    ref_toks = _toks(ref)
    hyp_toks = _toks(hyp)

    from collections import defaultdict, deque
    ref_occ = defaultdict(list)
    for i, t in enumerate(ref_toks):
        ref_occ[t].append(i)
    hyp_occ = defaultdict(list)
    for j, t in enumerate(hyp_toks):
        hyp_occ[t].append(j)

    matched, ref_missing, hyp_missing = [], [], []
    for t in set(ref_occ) | set(hyp_occ):
        r_pos = deque(ref_occ.get(t, []))
        h_pos = deque(hyp_occ.get(t, []))
        while r_pos and h_pos:
            matched.append((r_pos.popleft(), h_pos.popleft()))
        ref_missing.extend(r_pos)
        hyp_missing.extend(h_pos)

    n_m, n_rm, n_hm = len(matched), len(ref_missing), len(hyp_missing)
    if n_m + n_rm + n_hm < 2:
        return None

    def _choose2(n):
        return n * (n - 1) // 2

    total_distance = 0.0
    total_pairs = 0

    # Paires appariées ↔ appariées : concordance réelle sur les positions.
    for i in range(n_m):
        r1, h1 = matched[i]
        for j in range(i + 1, n_m):
            r2, h2 = matched[j]
            concordant = (r1 < r2) == (h1 < h2)
            total_distance += 0.0 if concordant else 1.0
    total_pairs += _choose2(n_m)

    # Paires appariée ↔ non-appariée (un seul côté a une position) : pénalité fixe.
    n_one_sided = n_rm + n_hm
    total_distance += p_penalty * n_m * n_one_sided
    total_pairs += n_m * n_one_sided

    # Paires non-appariée ↔ non-appariée (aucun des deux comparable) : pénalité fixe.
    n_both_missing_pairs = _choose2(n_rm) + _choose2(n_hm) + n_rm * n_hm
    total_distance += p_penalty * n_both_missing_pairs
    total_pairs += n_both_missing_pairs

    if total_pairs == 0:
        return None
    normalized_distance = total_distance / total_pairs
    return 1.0 - 2.0 * normalized_distance


# ── Rapport ────────────────────────────────────────────────────────────────────

def print_report(results: list, verbose: bool = False):
    total = len(results)
    if total == 0:
        print("Aucun résultat.")
        return

    exm_scores  = [r['exm']  for r in results]
    chrf_scores = [r['chrf'] for r in results]
    p1_scores   = [r['p1']   for r in results if r['p1'] is not None]
    kgh_scores  = [r['kgh']  for r in results if r['kgh'] is not None]

    exm_rate = sum(exm_scores) / total * 100
    chrf_avg = sum(chrf_scores) / total
    p1_avg   = sum(p1_scores) / len(p1_scores) * 100 if p1_scores else None
    kgh_avg  = sum(kgh_scores) / len(kgh_scores) * 100 if kgh_scores else None

    # Tree metrics
    cta_list  = [r['clause_type_ok'] for r in results if r.get('clause_type_ok') is not None]
    tam_list  = [r['tam_ok']         for r in results if r.get('ref_tam')]
    neg_list  = [r['neg_ok']         for r in results]
    sfr_s     = [r['slot_S']         for r in results if r.get('slot_S') is not None]
    sfr_v     = [r['slot_V']         for r in results if r.get('slot_V') is not None]

    cta_global = sum(cta_list) / len(cta_list) * 100 if cta_list else None
    tam_acc    = sum(tam_list) / len(tam_list) * 100  if tam_list else None
    neg_acc    = sum(neg_list) / len(neg_list) * 100  if neg_list else None
    sfr_s_avg  = sum(sfr_s)   / len(sfr_s)   * 100   if sfr_s   else None
    sfr_v_avg  = sum(sfr_v)   / len(sfr_v)   * 100   if sfr_v   else None

    # Kendall τ de réordonnancement, par phrase : ordre des tokens communs
    # (obtenu vs référence), moyenné sur le corpus (cf. décision 2026-07-11).
    wo_tau_scores = [kendall_tau_word_order(r['output'], r['reference']) for r in results]
    wo_tau_scores = [t for t in wo_tau_scores if t is not None]
    tau_word_order = sum(wo_tau_scores) / len(wo_tau_scores) if wo_tau_scores else 0.0

    # Corpus-level chrF / chrF++ / BLEU
    hyps_all = [r['output']    for r in results]
    refs_all = [r['reference'] for r in results]
    corp = corpus_metrics(hyps_all, refs_all)

    # UD parse health
    ud_scores = [r['ud_score'] for r in results if r.get('ud_score') is not None]
    ud_checks = ['ud_single_root', 'ud_root_pos_ok', 'ud_has_subject',
                 'ud_connected', 'ud_no_dep_dep']
    ud_labels = ['ROOT unique', 'ROOT pos valide', 'Sujet présent',
                 'Arbre connexe', 'Pas de dep=dep']
    ud_check_rates = {}
    for chk in ud_checks:
        vals = [r[chk] for r in results if r.get(chk) is not None]
        ud_check_rates[chk] = sum(vals) / len(vals) * 100 if vals else None

    # Bambara word order
    wo_list = [r['word_order_ok'] for r in results if r.get('word_order_ok') is not None]
    wo_rate = sum(wo_list) / len(wo_list) * 100 if wo_list else None

    # UD structure (Aplonova & Tyers scheme, cf. build_reference_ud_tree)
    ud_struct_list = [r['ud_structure_ok'] for r in results if r.get('ud_structure_ok') is not None]
    ud_struct_rate = sum(ud_struct_list) / len(ud_struct_list) * 100 if ud_struct_list else None

    print(f"\n{'═'*70}")
    print(f"  KUMA-MT — RAPPORT D'ÉVALUATION   ({total} phrases)")
    print(f"{'═'*70}")

    print(f"\n  ── Surface ────────────────────────────────────────")
    print(f"  EXM    Exact Match Rate     : {exm_rate:6.1f}%")
    print(f"  chrF   Sentence avg        : {chrf_avg:6.1f}")
    if corp['chrf_corpus']   is not None: print(f"  chrF   Corpus              : {corp['chrf_corpus']:6.2f}")
    if corp['chrfpp_corpus'] is not None: print(f"  chrF++ Corpus (word-order) : {corp['chrfpp_corpus']:6.2f}")
    if corp['bleu_corpus']   is not None: print(f"  BLEU   Corpus (char)       : {corp['bleu_corpus']:6.2f}")
    print(f"  τ      Kendall (word order, vs ref) : {tau_word_order:+.3f}")

    print(f"\n  ── Sémantique ─────────────────────────────────────")
    if p1_avg  is not None: print(f"  P@1   Embedding Precision  : {p1_avg:6.1f}%")
    if kgh_avg is not None: print(f"  KGH   KG Hit Rate          : {kgh_avg:6.1f}%")

    print(f"\n  ── Structure (tree) ───────────────────────────────")
    if cta_global is not None: print(f"  CTA   Clause Type Accuracy : {cta_global:6.1f}%")
    if tam_acc    is not None: print(f"  TAM   TAM Marker Accuracy  : {tam_acc:6.1f}%")
    if neg_acc    is not None: print(f"  NEG   Negation Accuracy    : {neg_acc:6.1f}%")
    if sfr_s_avg  is not None: print(f"  SFR-S Slot Sujet Fill Rate : {sfr_s_avg:6.1f}%")
    if sfr_v_avg  is not None: print(f"  SFR-V Slot Verbe Fill Rate : {sfr_v_avg:6.1f}%")
    if wo_rate    is not None: print(f"  WO    Word Order (S-TAM-V) : {wo_rate:6.1f}%")
    if ud_struct_rate is not None: print(f"  UDS   UD Structure (Aplonova & Tyers) : {ud_struct_rate:6.1f}%")

    print(f"\n  ── UD Parse Health (FR parse) ─────────────────────")
    if ud_scores:
        ud_avg = sum(ud_scores) / len(ud_scores) / 5 * 100
        print(f"  UD Score moyen             : {ud_avg:6.1f}%  (sur 5 checks)")
        for chk, lbl in zip(ud_checks, ud_labels):
            pct = ud_check_rates.get(chk)
            if pct is not None:
                bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
                print(f"  {lbl:<22} : {bar}  {pct:.1f}%")
    else:
        print("  (spaCy non disponible)")

    # Breakdown par catégorie générale (clause_type_group)
    type_groups = defaultdict(lambda: {'n': 0, 'exm': 0, 'chrf': []})
    for r in results:
        g = CATEGORY_TO_CLAUSE.get(r['category'], r['category'])
        type_groups[g]['n'] += 1
        type_groups[g]['exm'] += r['exm']
        type_groups[g]['chrf'].append(r['chrf'])

    print(f"\n  {'Clause type':<22} {'N':>4} {'EXM%':>6} {'chrF':>6}")
    print(f"  {'-'*42}")
    for g, v in sorted(type_groups.items(), key=lambda x: -x[1]['n']):
        e = v['exm'] / v['n'] * 100
        c = sum(v['chrf']) / len(v['chrf'])
        print(f"  {g:<22} {v['n']:>4} {e:>6.1f} {c:>6.1f}")

    # Erreurs de traduction (exceptions), par type + lieu
    error_rows = [r for r in results if r.get('error_type')]
    if error_rows:
        err_counts = defaultdict(int)
        for r in error_rows:
            err_counts[(r['error_type'], r['error_where'])] += 1
        print(f"\n  ── ERREURS de traduction ({len(error_rows)}/{total}) ──")
        for (etype, ewhere), n in sorted(err_counts.items(), key=lambda x: -x[1]):
            print(f"  {n:>3}×  {etype:<20} @ {ewhere}")

    # Erreurs
    failures = [r for r in results if not r['exm']]
    if failures:
        print(f"\n  ── ÉCHECS chrF < 60 ({len([r for r in failures if r['chrf']<60])}) ──")
        for r in sorted(failures, key=lambda x: x['chrf'])[:10]:
            print(f"\n  [{r['category']}] {r['source']}")
            print(f"    ATT : {r['reference']}")
            print(f"    GOT : {r['output']}   (chrF={r['chrf']:.1f})")

    print(f"{'═'*70}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def load_cases_from_final_csv(path: str) -> list:
    """Charge (phrase_fr, bambara_attendu, categorie) depuis FINAL Data.csv —
    remplace test_phrases.TEST_CASES comme source de données/référence tout
    en réutilisant l'intégralité des métriques de ce module."""
    with open(path, encoding='utf-8') as f:
        return [(row['phrase_fr'], row['bambara_attendu'], row['categorie'])
                for row in csv.DictReader(f)]


def run_evaluation(filter_cat: str = None, verbose: bool = False,
                   no_translate: bool = False, export_csv: bool = True,
                   make_plots: bool = False, cases: list = None):

    if cases is None:
        from test_phrases import TEST_CASES
        cases = TEST_CASES
    if filter_cat:
        cases = [(p, e, c) for p, e, c in cases if filter_cat in c]
        print(f"Filtre catégorie '{filter_cat}' : {len(cases)} phrases")

    results = []

    if no_translate:
        # Rapport sur la suite de test uniquement
        for phrase, expected, category in cases:
            results.append({
                'source': phrase, 'reference': expected,
                'output': '', 'category': category,
                'exm': False, 'chrf': 0.0,
                'p1': None, 'kgh': None, 'clause_type_ok': None,
            })
        print_report(results, verbose)
        return

    # ── Traduction + métriques ────────────────────────────────────────────────
    print("Chargement du moteur…")
    import warnings; warnings.filterwarnings('ignore')
    # Rediriger stdout pendant la traduction pour supprimer les DEBUG
    import io, contextlib

    from pipeline.translation_engine import TranslationEngine
    from pipeline.spacy_parser import SpacyParser
    from kg.neo4j_client import Neo4jClient
    import test_phrases as tp
    db  = Neo4jClient()
    eng = TranslationEngine(db)
    tagger = SpacyParser(db)

    total = len(cases)
    print(f"Évaluation sur {total} phrases…\n")

    for i, (phrase, expected, category) in enumerate(cases, 1):
        print(f"\r  [{i:3d}/{total}] {phrase[:55]:<55}", end='', flush=True)

        # Traduction (silencieuse)
        buf = io.StringIO()
        res = {}
        error_type, error_where = '', ''
        try:
            with contextlib.redirect_stdout(buf):
                res = eng.translate(phrase)
            output = res.get('bambara', '').strip()
        except Exception as e:
            error_type, error_where = classify_error(e)
            output = f'[ERR:{error_type} @ {error_where}] {e}'

        # Tokens pour P@1 et KGH (dernier clause traduite)
        tokens = getattr(eng, '_current_clause_tokens', []) or []

        # EXM
        exm = (output == expected.strip())

        # chrF / chrF++ (sentence-level ; chrF++ seulement pour Kendall's τ)
        chrf   = chrf_score(output, expected)
        chrfpp = chrfpp_score(output, expected)

        # P@1
        p1 = embedding_p1(tokens, {})

        # KG Hit Rate
        kgh = kg_hit_rate(tokens)

        # CamemBERT (mot-à-mot isolé) et BERTScore (contextuel, glouton) sont
        # deux métriques distinctes par construction (cf. décision 2026-07-09).
        with contextlib.redirect_stdout(io.StringIO()):
            cam_kuma = tp.score_kuma_word_level(tokens)
            p_bs, r_bs, f1_bs = tp.kuma_bertscore_prf(tokens, phrase, tagger)

        # Métriques tree
        tree_meta = res.get('tree', {})
        tm = tree_metrics(tree_meta, expected, category)

        if verbose:
            status = '' if exm else f' chrF={chrf:.0f}'
            print(f"\n  {status}  {phrase}")
            if not exm:
                print(f"     ATT : {expected}")
                print(f"     GOT : {output}")
            print(f"     tree: clause={tree_meta.get('clause_type','?')} "
                  f"tam={tree_meta.get('tam','?')} neg={tree_meta.get('neg','?')} "
                  f"S={bool(tree_meta.get('S'))} V={bool(tree_meta.get('V'))}")

        # UD parse health
        udh = ud_health(phrase)

        # Bambara word order (contre le ClauseTemplate KG du clause_type)
        wo_ok = bambara_word_order_ok(tree_meta, output, eng.rule_engine.grammar)

        # UD structure (schéma Aplonova & Tyers, cf. build_reference_ud_tree)
        uds_ok = ud_structure_ok(tree_meta, output)

        results.append({
            'source':          phrase,
            'reference':       expected,
            'output':          output,
            'category':        category,
            'error_type':      error_type,
            'error_where':     error_where,
            'exm':             exm,
            'chrf':            chrf,
            'chrfpp':          chrfpp,
            'p1':              p1,
            'kgh':             kgh,
            # CamemBERT / BERTScore
            'cam':             cam_kuma,
            'bs_p':            p_bs,
            'bs_r':            r_bs,
            'bs_f1':           f1_bs,
            # tree metrics
            'clause_type_ok':  tm['clause_type_ok'],
            'tam_ok':          tm['tam_ok'],
            'neg_ok':          tm['neg_ok'],
            'slot_S':          tm['slot_S_filled'],
            'slot_V':          tm['slot_V_filled'],
            'slot_O':          tm['slot_O_filled'],
            'ref_tam':         tm['ref_tam'],
            'tree_tam':        tm['tree_tam'],
            'tree_clause':     tree_meta.get('clause_type', ''),
            # UD parse health
            'ud_score':          udh.get('ud_score'),
            'ud_single_root':    udh.get('single_root'),
            'ud_root_pos_ok':    udh.get('root_pos_ok'),
            'ud_has_subject':    udh.get('has_subject'),
            'ud_connected':      udh.get('connected'),
            'ud_no_dep_dep':     udh.get('no_dep_dep'),
            # word order
            'word_order_ok':   wo_ok,
            'ud_structure_ok': uds_ok,
        })

    print()  # saut de ligne après le curseur

    print_report(results, verbose)

    if export_csv:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f'eval_{ts}.csv'
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'source','reference','output','category','error_type','error_where',
                'exm','chrf','chrfpp','p1','kgh',
                'cam','bs_p','bs_r','bs_f1',
                'clause_type_ok','tam_ok','neg_ok','slot_S','slot_V','slot_O',
                'ref_tam','tree_tam','tree_clause',
                'ud_score','ud_single_root','ud_root_pos_ok','ud_has_subject',
                'ud_connected','ud_no_dep_dep','word_order_ok','ud_structure_ok'])
            writer.writeheader()
            writer.writerows(results)
        print(f"  CSV exporté : {path}")

        # Export JSON résumé (pour paper) — tout recalculé localement
        def _avg(lst):
            return round(sum(lst) / len(lst) * 100, 2) if lst else None

        _cta  = [r['clause_type_ok'] for r in results if r.get('clause_type_ok') is not None]
        _tam  = [r['tam_ok']         for r in results if r.get('ref_tam')]
        _neg  = [r['neg_ok']         for r in results]
        _sfrS = [r['slot_S']         for r in results if r.get('slot_S') is not None]
        _sfrV = [r['slot_V']         for r in results if r.get('slot_V') is not None]

        _corp = corpus_metrics(
            [r['output'] for r in results],
            [r['reference'] for r in results])
        _ud   = [r['ud_score'] for r in results if r.get('ud_score') is not None]
        _wo   = [r['word_order_ok'] for r in results if r.get('word_order_ok') is not None]
        _uds  = [r['ud_structure_ok'] for r in results if r.get('ud_structure_ok') is not None]
        summary = {
            'timestamp':    ts,
            'n_phrases':    len(results),
            'EXM':          round(sum(r['exm'] for r in results) / len(results) * 100, 2),
            'chrF_sentence_avg': round(sum(r['chrf'] for r in results) / len(results), 2),
            'chrF_corpus':  _corp.get('chrf_corpus'),
            'chrFpp_corpus': _corp.get('chrfpp_corpus'),
            'BLEU_corpus':  _corp.get('bleu_corpus'),
            'P@1':          _avg([r['p1']  for r in results if r['p1']  is not None]),
            'KGH':          _avg([r['kgh'] for r in results if r['kgh'] is not None]),
            'CamemBERT':    _avg([r['cam']   for r in results if r['cam']   is not None]),
            'BERTScore_P':  _avg([r['bs_p']  for r in results if r['bs_p']  is not None]),
            'BERTScore_R':  _avg([r['bs_r']  for r in results if r['bs_r']  is not None]),
            'BERTScore_F1': _avg([r['bs_f1'] for r in results if r['bs_f1'] is not None]),
            'tau_word_order': (lambda v: round(sum(v) / len(v), 3) if v else 0.0)(
                                [t for t in (kendall_tau_word_order(r['output'], r['reference'])
                                             for r in results) if t is not None]),
            'CTA':   _avg(_cta),
            'TAM':   _avg(_tam),
            'NEG':   _avg(_neg),
            'SFR_S': _avg(_sfrS),
            'SFR_V': _avg(_sfrV),
            'UD_score_avg': round(sum(_ud) / len(_ud) / 5 * 100, 2) if _ud else None,
            'WO_S_TAM_V':  round(sum(_wo) / len(_wo) * 100, 2) if _wo else None,
            'UD_structure_Aplonova_Tyers': round(sum(_uds) / len(_uds) * 100, 2) if _uds else None,
        }
        json_path = f'eval_{ts}_summary.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  JSON résumé : {json_path}")

        # Génération des graphiques
        if make_plots:
            try:
                import eval_plots
                outdir = 'figures'
                os.makedirs(outdir, exist_ok=True)
                from evaluate import CATEGORY_TO_CLAUSE as _map
                plot_rows = eval_plots.load_csv(path)
                print(f"\n  Génération des figures dans {outdir}/ …")
                eval_plots.fig1a_surface_metrics(plot_rows, outdir)
                eval_plots.fig1b_semantic_metrics(plot_rows, outdir)
                eval_plots.fig1c_structure_metrics(plot_rows, outdir)
                eval_plots.fig_by_clause_type(plot_rows, outdir, _map)
                eval_plots.fig_chrf_distribution(plot_rows, outdir)
                eval_plots.fig_clause_confusion(plot_rows, outdir, _map)
                eval_plots.fig_chrf_vs_p1(plot_rows, outdir)
                eval_plots.fig_tam_matrix(plot_rows, outdir)
            except Exception as e:
                print(f"    Graphiques non générés : {e}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Évaluation Kuma-MT')
    parser.add_argument('--verbose',       action='store_true')
    parser.add_argument('--cat',           default=None, help='Filtre catégorie')
    parser.add_argument('--no-translate',  action='store_true')
    parser.add_argument('--no-csv',        action='store_true')
    parser.add_argument('--plots',         action='store_true',
                        help='Générer les graphiques après l\'évaluation')
    args = parser.parse_args()

    run_evaluation(
        filter_cat   = args.cat,
        verbose      = args.verbose,
        no_translate = args.no_translate,
        export_csv   = not args.no_csv,
        make_plots   = args.plots,
    )
