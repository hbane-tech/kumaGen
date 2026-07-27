"""
eval/fig_bertscore_3systems.py — BERTScore Precision/Recall/F1 pour les 3
systèmes (Kuma-MT, Google, NLLB-200), n=365.

Source :
  - Google/NLLB : eval/baseline_bertscore.csv (back-traduction mot-à-mot,
    cf. eval/compute_baseline_bertscore.py — calculé sur
    resultats_bambara_20260720_074041.csv)
  - Kuma : PAS ENCORE DISPONIBLE — le vrai BERTScore contextuel Kuma exige
    les gloses KG par token (kuma_tokens), non exportées dans le run du
    2026-07-20 qui a produit ce CSV. test_phrases.py calcule désormais
    p_kuma/r_kuma/f1_kuma en direct (colonne 'bertscore_prf_kuma' du CSV,
    format 'P|R|F1') — un prochain `python test_phrases.py --run` les
    capturera pour un vrai comparatif 3 systèmes (décision utilisateur
    2026-07-20).

Même palette/style que eval_final_figures.py (Okabe-Ito).

Usage : python -m eval.fig_bertscore_3systems
"""
import csv
import os

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

C_KUMA   = '#0072B2'
C_GOOGLE = '#E69F00'
C_NLLB   = '#009E73'

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(_ROOT, 'figures_final')


def _avg(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in ('', 'None', None)]
    return sum(vals) / len(vals) * 100 if vals else 0.0


def main():
    with open(os.path.join(_ROOT, 'eval', 'baseline_bertscore.csv'), encoding='utf-8') as f:
        base_rows = list(csv.DictReader(f))

    goog = [_avg(base_rows, 'p_google'), _avg(base_rows, 'r_google'), _avg(base_rows, 'f1_google')]
    nllb = [_avg(base_rows, 'p_nllb'), _avg(base_rows, 'r_nllb'), _avg(base_rows, 'f1_nllb')]

    labels = ['Precision', 'Recall', 'F1']
    x = np.arange(len(labels))
    w = 0.3

    fig, ax = plt.subplots(figsize=(7.5, 5))
    b2 = ax.bar(x - w / 2, goog, width=w, color=C_GOOGLE, edgecolor='black', linewidth=0.5, label='Google')
    b3 = ax.bar(x + w / 2, nllb, width=w, color=C_NLLB,  edgecolor='black', linewidth=0.5, label='NLLB-200')

    for bars in (b2, b3):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 1.2, f'{h:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel('%')
    ax.set_title('BERTScore — Precision / Recall / F1 by system (n=365)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.text(0.02, 0.97,
            'Kuma-MT : en attente (prochain run capture bertscore_prf_kuma)',
            transform=ax.transAxes, fontsize=8, va='top', style='italic', color='gray')

    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUTDIR, f'fig18_bertscore_3systems.{ext}'), bbox_inches='tight')
    print(f"Saved fig18_bertscore_3systems.png/.pdf to {OUTDIR}")


if __name__ == '__main__':
    main()
