"""
eval/compare_kuma_vs_gold.py — Compare une sortie Kuma-MT à l'exemple gold
UD_Bambara-CRB structurellement le plus proche.
================================================================================
Ne construit AUCUN arbre "de référence" auto-généré (contrairement à
bambara_ud_reference.py::build_reference_ud_tree, qui dérive un pseudo-gold
depuis tree_meta) : les deux arbres affichés ici viennent de sources
indépendantes de KumaGen —
  1. la sortie Kuma est parsée par le vrai modèle UDPipe entraîné sur
     UD_Bambara (Aplonova & Tyers), via eval/bambara_udpipe.py::parse_bambara ;
  2. l'exemple "de référence" est une phrase RÉELLE du treebank gold
     UD_Bambara-CRB (eval/ud/bm_crb-ud-test.conllu), retrouvée par similarité
     structurelle (recouvrement de deprels + longueur), pas construite.

Similarité structurelle = indice de Jaccard entre les multi-ensembles de
deprels (label principal, avant ':') des deux phrases, avec pénalité de
longueur en départage. Une mesure volontairement simple (pas d'alignement
d'arbres) — suffisante pour retrouver "quel type de construction gold
ressemble le plus à ceci", pas pour scorer une accuracy.

Usage :
  python -m eval.compare_kuma_vs_gold "a yé porofesɛri yé"
  python -m eval.compare_kuma_vs_gold "a yé porofesɛri yé" --top 3
"""
import os
import sys
import argparse
from collections import Counter

from eval.bambara_udpipe import parse_bambara
from eval.bambara_parser_eval import read_conllu, UD_TEST, download_treebank

_GOLD_CACHE = None


def _main_label(dep):
    return (dep or '').split(':')[0].lower()


def _load_gold():
    global _GOLD_CACHE
    if _GOLD_CACHE is not None:
        return _GOLD_CACHE
    if not os.path.exists(UD_TEST) and not download_treebank():
        raise FileNotFoundError(f"Treebank introuvable : {UD_TEST}")
    sents = read_conllu(UD_TEST)
    # ne garde que les phrases avec texte + tokens (élimine les blocs vides)
    _GOLD_CACHE = [(text, toks) for text, toks, _ in sents if text and toks]
    return _GOLD_CACHE


def _signature(tokens):
    """Multi-ensemble des deprels principaux — la 'forme' structurelle."""
    return Counter(_main_label(t['deprel']) for t in tokens)


def _jaccard(sig_a: Counter, sig_b: Counter) -> float:
    inter = sum((sig_a & sig_b).values())
    union = sum((sig_a | sig_b).values())
    return inter / union if union else 0.0


def best_gold_matches(kuma_tokens, top_k=3):
    """Retourne les top_k (score, text, gold_tokens) les plus proches
    structurellement de kuma_tokens, parmi le treebank gold UD_Bambara-CRB."""
    sig_kuma = _signature(kuma_tokens)
    n_kuma = len(kuma_tokens)

    scored = []
    for text, toks in _load_gold():
        sig_gold = _signature(toks)
        score = _jaccard(sig_kuma, sig_gold)
        if score == 0.0:
            continue
        len_penalty = abs(len(toks) - n_kuma) / max(len(toks), n_kuma, 1)
        # score composite : similarité structurelle en premier, longueur
        # proche en départage (poids mineur, 0.1 — ne doit jamais inverser
        # un net avantage de recouvrement de deprels)
        composite = score - 0.1 * len_penalty
        scored.append((composite, score, text, toks))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(s, text, toks) for _, s, text, toks in scored[:top_k]]


def _print_tree(title, tokens):
    print(f"  {title}")
    print(f"  {'id':<4}{'form':<16}{'upos':<8}{'head':<6}{'deprel'}")
    for t in tokens:
        print(f"  {t['id']:<4}{t['form']:<16}{t['upos']:<8}{t['head']:<6}{t['deprel']}")


def compare(sentence: str, top_k: int = 3):
    kuma_tokens = parse_bambara(sentence)
    if not kuma_tokens:
        print(f"  Le parseur UD_Bambara n'a rien retourné pour : {sentence!r}")
        return

    print("=" * 78)
    print(f"SORTIE KUMA-MT (parsée par le modèle UDPipe UD_Bambara, indépendant)")
    print("=" * 78)
    _print_tree(f"'{sentence}'", kuma_tokens)

    matches = best_gold_matches(kuma_tokens, top_k=top_k)
    if not matches:
        print("\n  Aucun exemple gold avec un deprel en commun — phrase trop atypique.")
        return

    print()
    print("=" * 78)
    print(f"TOP {len(matches)} EXEMPLES GOLD UD_Bambara-CRB LES PLUS PROCHES "
          f"(similarité de deprels)")
    print("=" * 78)
    for i, (score, text, toks) in enumerate(matches, 1):
        print(f"\n  ── Match #{i}  (similarité deprel = {score:.2f}) ──")
        _print_tree(f"'{text}'", toks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('sentence', help="Phrase bambara produite par Kuma-MT")
    ap.add_argument('--top', type=int, default=3, help="Nombre d'exemples gold à afficher")
    args = ap.parse_args()
    compare(args.sentence, top_k=args.top)


if __name__ == '__main__':
    main()
