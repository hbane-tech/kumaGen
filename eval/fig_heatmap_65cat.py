"""
fig_heatmap_65cat.py — Heatmap 3 systèmes x 65 catégories (chrF++), format
compact horizontal (catégories en colonnes, systèmes en lignes).

Lit resultats_bambara_<ts>.csv (le CSV utilisé par eval_report.py, celui qui
reproduit exactement les chiffres du Tableau 1 de l'article), calcule
chrF++ par phrase pour Kuma-MT/Google/NLLB, moyenne par catégorie fine
regroupée via eval_final.clause_group() (= les 65 "grammatical construction
types" cités dans le papier, vérifié : clause_group() appliqué aux 230
catégories brutes du CSV donne exactement 65 groupes distincts).

Catégories triées par score Kuma-MT décroissant (pas de dendrogramme —
inutile en format 3-lignes, et évite le sur-encombrement vertical du
premier essai).

Usage:
    python eval/fig_heatmap_65cat.py [resultats_bambara_XXX.csv]
"""
import sys
import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.eval_final import clause_group
from eval.eval_report import _chrfpp_score


def _avg(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'resultats_bambara_20260724_005832.csv'
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))

    groups = defaultdict(lambda: {'kuma': [], 'google': [], 'nllb': []})
    for r in rows:
        ref = (r.get('bambara_attendu') or '').strip()
        kuma = (r.get('bambara_obtenu') or '').strip()
        goog = (r.get('google_translate') or '').strip()
        nllb = (r.get('nllb_translate') or '').strip()
        if not ref:
            continue
        g = clause_group(r['categorie'])
        if kuma:
            groups[g]['kuma'].append(_chrfpp_score(kuma, ref) * 100)
        if goog:
            groups[g]['google'].append(_chrfpp_score(goog, ref) * 100)
        if nllb:
            groups[g]['nllb'].append(_chrfpp_score(nllb, ref) * 100)

    cats = list(groups.keys())
    n_by_cat = {c: len(groups[c]['kuma']) for c in cats}
    scored = [
        (c, _avg(groups[c]['kuma']) or 0, _avg(groups[c]['google']) or 0, _avg(groups[c]['nllb']) or 0)
        for c in cats
    ]
    scored.sort(key=lambda x: -x[1])  # tri par score Kuma-MT décroissant

    ordered_cats = [s[0] for s in scored]
    ordered_vals = [(s[1], s[2], s[3]) for s in scored]

    # AAAI double-column width ≈ 7.1in — un seul bandeau de 65 colonnes
    # illisible à cette largeur. On découpe en N_STRIPS bandeaux empilés,
    # chacun tenant sur la pleine largeur double-colonne.
    N_STRIPS = 3
    n = len(ordered_cats)
    chunk = -(-n // N_STRIPS)  # ceil division

    fig, axes = plt.subplots(N_STRIPS, 1, figsize=(7.1, 2.05 * N_STRIPS))
    im = None
    for k in range(N_STRIPS):
        lo, hi = k * chunk, min((k + 1) * chunk, n)
        cats_k = ordered_cats[lo:hi]
        matrix_k = np.array(ordered_vals[lo:hi]).T  # 3 x len(cats_k)
        ax = axes[k]
        im = ax.imshow(matrix_k, aspect='auto', cmap='Blues', vmin=0, vmax=100)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Kuma-MT', 'Google', 'NLLB-200'], fontsize=7, fontweight='bold')
        ax.set_xticks(range(len(cats_k)))
        ax.set_xticklabels(cats_k, rotation=90, fontsize=5.5, ha='center')
        ax.tick_params(axis='x', pad=2)
        for i in range(matrix_k.shape[0]):
            for j in range(matrix_k.shape[1]):
                val = matrix_k[i, j]
                color = 'white' if val > 55 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=4.5, color=color)

    axes[0].set_title(
        'chrF++ across all 65 grammatical construction types, sorted by Kuma-MT score '
        '(n per category in Table 3 / supplementary material)',
        fontsize=9, pad=8)
    fig.subplots_adjust(hspace=1.6, top=0.94, bottom=0.04)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, shrink=0.8)
    cbar.set_label('chrF++', fontsize=8)

    outdir = 'figures_final'
    os.makedirs(outdir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(outdir, f'fig_heatmap_65cat.{ext}'), dpi=300, bbox_inches='tight')
    print(f' figures_final/fig_heatmap_65cat.png / .pdf  ({len(cats)} catégories)')


if __name__ == '__main__':
    main()
