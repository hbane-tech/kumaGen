"""
eval/fig_upos_uas_las_by_group.py — UPOS / UAS / LAS par groupe grammatical,
entre l'arbre de référence (build_reference_ud_tree, schéma Aplonova & Tyers
+ tree_meta) et l'arbre produit par le parseur UD_Bambara indépendant
(parse_bambara), sur des sorties Kuma-MT réelles.

Source : eval/kuma_tree_upos_uas_las.csv (voir eval/tree_metrics_upos_uas_las.py)

Usage : python -m eval.fig_upos_uas_las_by_group
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 11,
    'font.family': 'DejaVu Sans',
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

C_UPOS = '#0072B2'   # bleu — Okabe-Ito
C_UAS  = '#E69F00'   # orange
C_LAS  = '#009E73'   # vert

CSV_PATH = os.path.join(os.path.dirname(__file__), 'kuma_tree_upos_uas_las.csv')
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'figures_final')


def main():
    with open(CSV_PATH, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    by_group = defaultdict(list)
    for r in rows:
        by_group[r['group']].append(r)

    items = sorted(by_group.items(), key=lambda x: -len(x[1]))
    names = [f"{g} (n={len(v)})" for g, v in items]
    upos = [sum(float(r['upos']) for r in v) / len(v) for _, v in items]
    uas  = [sum(float(r['uas'])  for r in v) / len(v) for _, v in items]
    las  = [sum(float(r['las'])  for r in v) / len(v) for _, v in items]

    y = np.arange(len(names))
    h = 0.25
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.7)))
    ax.barh(y - h, upos, height=h, color=C_UPOS, edgecolor='black', linewidth=0.4, label='UPOS')
    ax.barh(y,     uas,  height=h, color=C_UAS,  edgecolor='black', linewidth=0.4, label='UAS')
    ax.barh(y + h, las,  height=h, color=C_LAS,  edgecolor='black', linewidth=0.4, label='LAS')

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel('%  (agreement between reference tree and the independent UD_Bambara parser)')
    ax.set_title('Kuma-MT output — UPOS / UAS / LAS vs. independent UD_Bambara parse\n'
                  f'(sampled sentences, n={len(rows)} total)')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUTDIR, f'fig17_upos_uas_las_by_group.{ext}'),
                    bbox_inches='tight')
    print(f"Saved fig17_upos_uas_las_by_group.png/.pdf to {OUTDIR}")


if __name__ == '__main__':
    main()
