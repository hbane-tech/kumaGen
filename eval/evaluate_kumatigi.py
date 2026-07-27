"""
evaluate_kumatigi.py — Évaluation de Kuma-MT sur le dataset externe
Tysby101/kumatigi (HuggingFace, fra->bam, 55k/4.5k/4.9k train/val/test).

Ce dataset n'a pas de colonne 'categorie' (clause_type Kuma) : on réutilise
son champ 'tier' comme catégorie de regroupement pour le rapport (permet un
breakdown par provenance : high_quality / real_fra_reviewed / real_bam_reviewed
/ synthetic_reviewed), mais les métriques CTA/TAM/NEG (qui dépendent du
mapping CATEGORY_TO_CLAUSE) ne sont pas significatives ici et doivent être
ignorées dans le rapport - seules EXM/chrF/P@1/KGH/UD structure/word order
sont pertinentes pour ce corpus hors-suite-de-test.

Usage:
  python -m eval.evaluate_kumatigi --split test --n 500 --seed 0
"""

import argparse
import random

from eval.evaluate import run_evaluation


def load_cases(split: str, n: int, seed: int, min_quality: int = None) -> list:
    from datasets import load_dataset
    ds = load_dataset('Tysby101/kumatigi')[split]
    idxs = list(range(len(ds)))
    if min_quality is not None:
        idxs = [i for i in idxs if ds[i]['quality_bin'] >= min_quality]
    random.seed(seed)
    if n < len(idxs):
        idxs = random.sample(idxs, n)
    cases = []
    for i in idxs:
        r = ds[i]
        cases.append((r['src_text'], r['tgt_text'], r['tier']))
    return cases


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Évaluation Kuma-MT sur kumatigi (HF)')
    ap.add_argument('--split', default='test', choices=['train', 'validation', 'test'])
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-quality', type=int, default=None,
                     help='Filtre quality_bin >= valeur (0-3)')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--no-csv', action='store_true')
    args = ap.parse_args()

    cases = load_cases(args.split, args.n, args.seed, args.min_quality)
    print(f"kumatigi/{args.split} : {len(cases)} phrases échantillonnées (seed={args.seed})")

    run_evaluation(
        verbose=args.verbose,
        export_csv=not args.no_csv,
        cases=cases,
    )
