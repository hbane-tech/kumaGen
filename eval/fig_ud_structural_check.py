"""
eval/fig_ud_structural_check.py — Figure : santé structurelle UD de la sortie
Kuma-MT, mesurée par le VRAI parseur UD_Bambara (Aplonova & Tyers), par groupe.
================================================================================
Source : eval/kuma_ud_structural_check.csv (généré par parsing direct de la
sortie Kuma avec eval/bambara_udpipe.py::parse_bambara — AUCUN arbre
auto-construit depuis tree_meta, contrairement à bambara_ud_reference.py).

Même palette/style que eval_final_figures.py (Okabe-Ito, C_KUMA bleu).
Deux groupes (prohibitive, imperative) sont coloriés à part : en bambara,
l'impératif/prohibitif n'a structurellement pas de sujet explicite (2e
personne implicite) — un has_subject=0% y est attendu grammaticalement, pas
un échec, contrairement aux autres groupes à faible score (ex: equative,
qualitative) qui restent des cas à investiguer.

Usage : python -m eval.fig_ud_structural_check
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

C_KUMA = '#0072B2'   # bleu — Okabe-Ito, cohérent avec eval_final_figures.py
C_GREY = '#999999'   # groupes où l'absence de sujet est grammaticalement attendue

_NO_SUBJECT_EXPECTED = {'prohibitive', 'imperative'}

CSV_PATH = os.path.join(os.path.dirname(__file__), 'kuma_ud_structural_check.csv')
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'figures_final')


def _pct(vals):
    v = [x == 'True' for x in vals if x != '']
    return round(sum(v) / len(v) * 100, 1) if v else 0.0


def main():
    with open(CSV_PATH, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for r in rows:
        if r['root_upos'] != '':
            groups[r['group']].append(r)

    items = sorted(groups.items(), key=lambda x: -len(x[1]))
    names = [f"{g} (n={len(v)})" for g, v in items]
    has_subj = [_pct([r['has_subject'] for r in v]) for _, v in items]
    colors = [C_GREY if g in _NO_SUBJECT_EXPECTED else C_KUMA for g, _ in items]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, max(5, len(names) * 0.28)))
    ax.barh(y, has_subj, color=colors, edgecolor='black', linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel('% sentences with nsubj detected by the real UD_Bambara parser')
    ax.set_title(f'Kuma-MT output — subject detection by the independent\n'
                 f'UD_Bambara parser (Aplonova & Tyers) — ALL {len(names)} groups')

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=C_KUMA, edgecolor='black', label='Subject expected (unexplained gap = candidate error)'),
        Patch(facecolor=C_GREY, edgecolor='black', label='No subject expected by Bambara grammar (imperative/prohibitive)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=7, framealpha=0.9)

    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUTDIR, f'fig16_kuma_ud_structural_check.{ext}'),
                    bbox_inches='tight')
    print(f"Saved fig16_kuma_ud_structural_check.png/.pdf to {OUTDIR}")


if __name__ == '__main__':
    main()
