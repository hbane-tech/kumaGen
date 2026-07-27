"""
eval/bambara_treebank_stats.py — Statistiques structurelles tirées du gold
standard UD_Bambara-CRB, pour évaluer si les arbres bambara générés par
kuma_mt sont structurellement plausibles (POS/dépendances attestées dans du
vrai bambara), indépendamment du sens des mots et indépendamment du français.

Contrairement à eval_report.py's pos_acc/deprel_sim (qui comparent l'arbre
bambara à l'arbre français d'entrée), ceci compare l'arbre bambara à un
référentiel de patterns syntaxiques réellement attestés dans le corpus
UD_Bambara-CRB — la question posée est « cette structure existe-t-elle en
vrai bambara ? », pas « ressemble-t-elle au français ? ».

Deux métriques de couverture (0-1, plus haut = plus plausible) :
  pos_bigram_coverage : fraction des bigrammes UPOS consécutifs de l'arbre
                        kuma qui apparaissent quelque part dans le gold.
  dep_triple_coverage : fraction des triplets (UPOS tête, deprel, UPOS
                        dépendant) de l'arbre kuma qui apparaissent dans le
                        gold (le deprel est réduit à son label principal,
                        avant ':', pour ne pas être trop strict sur les
                        sous-types).
"""
import os
from functools import lru_cache

from eval.bambara_parser_eval import read_conllu, UD_TEST, download_treebank, main_label


def _build_stats(conllu_path: str = UD_TEST):
    if not os.path.exists(conllu_path) and not download_treebank(conllu_path):
        return set(), set()

    sents = read_conllu(conllu_path)
    pos_bigrams = set()
    dep_triples = set()
    for _text, toks, _units in sents:
        by_id = {t['id']: t for t in toks}
        ordered = sorted(toks, key=lambda t: t['id'])
        for a, b in zip(ordered, ordered[1:]):
            pos_bigrams.add((a['upos'], b['upos']))
        for t in toks:
            head = by_id.get(t['head'])
            head_upos = head['upos'] if head else 'ROOT'
            dep_triples.add((head_upos, main_label(t['deprel']), t['upos']))
    return pos_bigrams, dep_triples


@lru_cache(maxsize=1)
def get_gold_stats():
    """Cached: (pos_bigrams: set[(upos,upos)], dep_triples: set[(head_upos,deprel,dep_upos)])."""
    return _build_stats()


def pos_bigram_coverage(tokens: list[dict]) -> float | None:
    """Fraction of consecutive-UPOS bigrams in `tokens` attested in the gold treebank."""
    if not tokens or len(tokens) < 2:
        return None
    gold_bigrams, _ = get_gold_stats()
    if not gold_bigrams:
        return None
    ordered = sorted(tokens, key=lambda t: t['id'])
    pairs = [(a['upos'], b['upos']) for a, b in zip(ordered, ordered[1:])]
    if not pairs:
        return None
    hits = sum(1 for p in pairs if p in gold_bigrams)
    return hits / len(pairs)


def dep_triple_coverage(tokens: list[dict]) -> float | None:
    """Fraction of (head_upos, deprel, dep_upos) triples in `tokens` attested in the gold treebank."""
    if not tokens:
        return None
    _, gold_triples = get_gold_stats()
    if not gold_triples:
        return None
    by_id = {t['id']: t for t in tokens}
    triples = []
    for t in tokens:
        head = by_id.get(t['head'])
        head_upos = head['upos'] if head else 'ROOT'
        triples.append((head_upos, main_label(t['deprel']), t['upos']))
    if not triples:
        return None
    hits = sum(1 for tr in triples if tr in gold_triples)
    return hits / len(triples)
