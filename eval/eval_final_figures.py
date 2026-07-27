"""
eval_final_figures.py — Figures pour eval_final.py (Kuma-MT vs Google vs NLLB-200)
==================================================================================
Lit un CSV produit par eval_final.py (+ son JSON résumé, + le JSON de
eval/bambara_parser_eval.py pour les métriques UPOS/UAS/LAS/LAS-c) et génère
toutes les figures dans figures_final/ (PNG 300dpi + PDF vectoriel).

230 catégories → regroupées via eval_final.clause_group() pour tous les
graphiques par catégorie (bar/column illisible sinon, cf. demande explicite).

Palette Okabe-Ito (daltonien-friendly), assignation FIXE par système sur
TOUTES les figures de comparaison :
  Kuma-MT = bleu, Google = orange, NLLB-200 = vert.

Usage :
  python eval_final_figures.py eval_final_<ts>.csv eval_final_<ts>_summary.json
  python eval_final_figures.py --latest
"""
import os
import sys
import csv
import json
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_final import clause_group

plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 11,
    'font.family': 'DejaVu Sans',
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# Palette Okabe-Ito — assignation fixe par système, jamais permutée
C_KUMA   = '#0072B2'   # bleu
C_GOOGLE = '#E69F00'   # orange
C_NLLB   = '#009E73'   # vert
C_RED    = '#D55E00'
C_PURPLE = '#CC79A7'
C_GREY   = '#999999'
SYS_COLORS = {'Kuma-MT': C_KUMA, 'Google': C_GOOGLE, 'NLLB-200': C_NLLB}


def _b(s):
    return str(s).strip().lower() in ('true', '1', 'yes', 'oui')


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_csv(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            for k in ('exm_kuma', 'exm_google', 'exm_nllb', 'wo_kuma',
                      'clause_type_ok', 'tam_ok_kuma', 'tam_ok_google', 'tam_ok_nllb',
                      'neg_ok', 'slot_S', 'slot_V', 'slot_O', 'ud_single_root',
                      'ud_root_pos_ok', 'ud_has_subject', 'ud_connected', 'ud_no_dep_dep'):
                if r.get(k) not in (None, '', 'None'):
                    r[k] = _b(r[k])
                else:
                    r[k] = None
            for k in ('chrf_kuma', 'chrf_google', 'chrf_nllb',
                      'chrfpp_kuma', 'chrfpp_google', 'chrfpp_nllb',
                      'bleu_kuma', 'bleu_google', 'bleu_nllb',
                      'cam_kuma', 'cam_google', 'cam_nllb', 'ud_score',
                      'p_kuma', 'r_kuma', 'f1_kuma',
                      'p_google', 'r_google', 'f1_google',
                      'p_nllb', 'r_nllb', 'f1_nllb'):
                r[k] = _f(r.get(k))
            rows.append(r)
    return rows


def _save(fig, outdir, name):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(outdir, f'{name}.{ext}'), bbox_inches='tight')
    plt.close(fig)
    print(f"   {name}.png / .pdf")


def _avg(lst):
    v = [x for x in lst if x is not None]
    return sum(v) / len(v) if v else 0.0


def _pct(lst):
    v = [x for x in lst if x is not None]
    return sum(v) / len(v) * 100 if v else 0.0


# ─────────────────────────────────────────────────────────────────────────
# FIG A — Vue d'ensemble 3 systèmes (grouped bar, une figure = tout lisible)
# ─────────────────────────────────────────────────────────────────────────
def fig_system_overview(rows, summary, outdir):
    # 'Ordre S-TAM-V' exclu : métrique Kuma-only (repose sur le tree interne,
    # non calculable pour Google/NLLB) — cf. décision 2026-07-08, affichée
    # séparément dans fig_kuma_structure.
    metrics = ['EXM', 'EXM (normalized)', 'chrF', 'CamemBERT']
    kuma = [summary['kuma']['EXM'], summary['kuma']['EXM_normalized'],
            summary['kuma']['chrF_sentence_avg'], summary['kuma']['CamemBERT_avg'] * 100]
    goog = [summary['google']['EXM'], summary['google']['EXM_normalized'],
            summary['google']['chrF_sentence_avg'], summary['google']['CamemBERT_avg'] * 100]
    nllb = [summary['nllb']['EXM'], summary['nllb']['EXM_normalized'],
            summary['nllb']['chrF_sentence_avg'], summary['nllb']['CamemBERT_avg'] * 100]

    y = np.arange(len(metrics))
    h = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y + h, kuma, height=h, color=C_KUMA, label='Kuma-MT', edgecolor='black', linewidth=0.5)
    ax.barh(y,     goog, height=h, color=C_GOOGLE, label='Google', edgecolor='black', linewidth=0.5)
    ax.barh(y - h, nllb, height=h, color=C_NLLB,  label='NLLB-200', edgecolor='black', linewidth=0.5)
    for yi, vals in zip(y, zip(kuma, goog, nllb)):
        for off, v in zip((h, 0, -h), vals):
            ax.text(v + 1, yi + off, f'{v:.1f}', va='center', fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel('Score (%)')
    ax.set_title(f"Kuma-MT vs Google vs NLLB-200 — Overview (n={summary['n_phrases']})")
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False)
    _save(fig, outdir, 'fig1_system_overview')


# ─────────────────────────────────────────────────────────────────────────
# FIG A2 — Grille 2×3 : un subplot par métrique, 3 barres (Kuma/Google/NLLB)
# Métriques : EXM, chrF, chrF++, BLEU(char), CamemBERT, Kendall τ (word order)
# τ = concordance d'ordre entre les tokens partagés hyp/référence, PAS une
# corrélation chrF↔chrF++ (cf. décision 2026-07-11 : l'ancienne version ne
# mesurait que l'accord entre deux métriques d'overlap proches, jamais
# l'ordre réel des mots — restait élevée même pour Google/NLLB malgré des
# scores chrF/BLEU bas, ce qui était trompeur).
# τ ∈[-1,+1] rescalé en 0-100 via (τ+1)/2*100 pour partager l'axe des autres.
# Toutes les métriques ici sont calculables pour les 3 systèmes (contrairement
# à Ordre S-TAM-V, Kuma-only, cf. fig9/fig10) — pas de violation de la règle
# "ne comparer que des métriques de même nature" puisque chaque métrique a
# son propre subplot indépendant, seule l'échelle 0-100 est partagée.
# ─────────────────────────────────────────────────────────────────────────
def fig_global_evaluation_grid(rows, summary, outdir):
    def _tau_rescaled(sysname):
        return (summary[sysname]['tau_word_order'] + 1) / 2 * 100

    panels = [
        ('Exact Match %',       lambda s: summary[s]['EXM']),
        ('Exact Match %\n(normalized: tones + pronoun form)', lambda s: summary[s]['EXM_normalized']),
        ('chrF',                lambda s: summary[s]['chrF_sentence_avg']),
        ('chrF++',              lambda s: summary[s]['chrFpp_sentence_avg']),
        ('BLEU (char)',         lambda s: summary[s]['BLEU_sentence_avg']),
        ('CamemBERT (BM↔FR)',   lambda s: summary[s]['CamemBERT_avg'] * 100),
        ('Kendall τ (word order\nvs reference)\nrescaled 0-100', _tau_rescaled),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(f"Global Evaluation — Kuma-MT vs Google vs NLLB (n={summary['n_phrases']})",
                 fontsize=15, fontweight='bold')

    systems = [('Kuma-MT', 'kuma', C_KUMA), ('Google', 'google', C_GOOGLE),
               ('NLLB-200', 'nllb', C_NLLB)]

    for ax, (title, getter) in zip(axes.flat, panels):
        labels = [name for name, _, _ in systems]
        values = [getter(key) for _, key, _ in systems]
        colors = [color for _, _, color in systems]
        bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.6)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylim(0, 108)
        ax.set_ylabel('Score')
        ax.set_title(title, fontsize=11, fontweight='bold')

    # 8e slot inutilisé (7 panels sur une grille 2x4) — masqué proprement.
    for ax in axes.flat[len(panels):]:
        ax.axis('off')

    _save(fig, outdir, 'fig2_global_evaluation_grid')


# ─────────────────────────────────────────────────────────────────────────
# FIG B / FIG C — métrique par groupe de clause (3 systèmes), HEATMAP
# ─────────────────────────────────────────────────────────────────────────
# Anciennement un bar chart horizontal tronqué au top_n=25 (sur 65 groupes
# réels) — troncature arbitraire, aucune justification pour "25" plutôt
# qu'un autre chiffre, et les 40 groupes exclus disparaissaient sans aucune
# indication dans la figure elle-même. Remplacé par une heatmap : encoder
# la valeur par la COULEUR d'une cellule (pas la longueur d'une barre)
# permet d'afficher TOUS les groupes sans figure démesurément large — même
# esprit que la matrice de confusion (figH/fig12), déjà une heatmap.
# Décision 2026-07-10.
def _fig_metric_heatmap_by_group(rows, outdir, value_key_kuma, value_key_google,
                                  value_key_nllb, scale100, metric_label, fig_name):
    groups = defaultdict(lambda: {'kuma': [], 'google': [], 'nllb': []})
    for r in rows:
        g = r['group']
        if r.get(value_key_kuma) is not None: groups[g]['kuma'].append(r[value_key_kuma])
        if r.get(value_key_google) is not None: groups[g]['google'].append(r[value_key_google])
        if r.get(value_key_nllb) is not None: groups[g]['nllb'].append(r[value_key_nllb])

    items = [(g, v) for g, v in groups.items() if v['kuma']]
    items = sorted(items, key=lambda x: -len(x[1]['kuma']))
    names = [f"{g} (n={len(v['kuma'])})" for g, v in items]
    mult = 100 if scale100 else 1
    M = np.array([
        [_avg(v['kuma']) * mult, _avg(v['google']) * mult, _avg(v['nllb']) * mult]
        for _, v in items
    ])

    fig, ax = plt.subplots(figsize=(6, max(6, len(names) * 0.24)))
    im = ax.imshow(M, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Kuma-MT', 'Google', 'NLLB-200'], fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title(f'{metric_label} by grammatical group — ALL {len(names)} groups (sorted by frequency)')
    for i in range(len(names)):
        for j in range(3):
            v = M[i, j]
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=6,
                    color='white' if v > 60 else 'black')
    fig.colorbar(im, ax=ax, label=metric_label, shrink=0.5)
    _save(fig, outdir, fig_name)


def fig_chrf_by_group(rows, outdir, top_n=None):
    _fig_metric_heatmap_by_group(rows, outdir, 'chrf_kuma', 'chrf_google', 'chrf_nllb',
                                  scale100=False, metric_label='chrF', fig_name='fig3_chrf_by_group')


def fig_camembert_by_group(rows, outdir, top_n=None):
    _fig_metric_heatmap_by_group(rows, outdir, 'cam_kuma', 'cam_google', 'cam_nllb',
                                  scale100=True, metric_label='CamemBERT (%)', fig_name='fig4_camembert_by_group')


# ─────────────────────────────────────────────────────────────────────────
# FIG D — Scatter chrF vs CamemBERT, 3 systèmes superposés
# ─────────────────────────────────────────────────────────────────────────
def _fig_scatter_headtohead(rows, outdir, metric_prefix, metric_label, fig_name):
    """Deux panels côte à côte : Kuma-MT vs NLLB-200 / Kuma-MT vs Google —
    phrase par phrase, pour UNE métrique donnée (chrF, chrF++ ou BLEU-char).
    Diagonale y=x ; zone ombrée = 'Kuma meilleur' (score_kuma > score_autre,
    sous la diagonale). Générique : réutilisée pour les 3 métriques de la
    même famille surface (même échelle 0-100, comparables entre elles)."""
    kuma_key = f'{metric_prefix}_kuma'
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle(f'Sentence-by-sentence comparison — {metric_label}', fontsize=14)

    panels = [
        ('NLLB-200', f'{metric_prefix}_nllb', C_NLLB, axes[0]),
        ('Google Translate', f'{metric_prefix}_google', C_GOOGLE, axes[1]),
    ]
    # Jitter reproductible : beaucoup de phrases ont un score Kuma proche de
    # 100 (exact match) — sans jitter, elles se superposent en une colonne
    # dense qui se lit visuellement comme "une barre" plutôt que comme des
    # centaines de points empilés, donnant l'illusion que la plupart des
    # données sont absentes. Décalage aléatoire faible (±1.5) sur x ET y,
    # seed fixe pour que la figure soit reproductible d'un run à l'autre.
    _rng = np.random.default_rng(42)

    for sysname, key, color, ax in panels:
        pts = [(r[kuma_key], r[key]) for r in rows
               if r[kuma_key] is not None and r[key] is not None]
        n = len(pts)
        x = np.array([p[0] for p in pts]) + _rng.uniform(-1.5, 1.5, size=n)
        y = np.array([p[1] for p in pts]) + _rng.uniform(-1.5, 1.5, size=n)

        ax.fill_between([0, 100], [0, 100], 0, color=C_KUMA, alpha=0.08, zorder=0)
        ax.plot([0, 100], [0, 100], color=C_GREY, linestyle='--', linewidth=1.2, zorder=1)
        ax.scatter(x, y, color=color, alpha=0.4, edgecolor='black',
                   linewidth=0.3, s=35, zorder=2)

        ax.set_xlim(-3, 105)
        ax.set_ylim(-3, 105)
        ax.set_xlabel(f'{metric_label} Kuma-MT')
        ax.set_ylabel(f'{metric_label} {sysname}')
        ax.set_title(f'Kuma-MT vs {sysname}\n(Kuma better in blue zone, n={n})')
        ax.set_aspect('equal')

    _save(fig, outdir, fig_name)


def fig_scatter_chrf_cam(rows, outdir):
    _fig_scatter_headtohead(rows, outdir, 'chrf', 'chrF', 'fig5_headtohead_chrf')


def fig_scatter_chrfpp(rows, outdir):
    _fig_scatter_headtohead(rows, outdir, 'chrfpp', 'chrF++', 'fig6_headtohead_chrfpp')


def fig_scatter_bleu_char(rows, outdir):
    _fig_scatter_headtohead(rows, outdir, 'bleu', 'BLEU (char)', 'fig7_headtohead_bleu_char')


# ─────────────────────────────────────────────────────────────────────────
# FIG E — Radar 3 systèmes superposés : vue d'ensemble de TOUTES les métriques
# calculables pour les 3 systèmes (EXM, chrF, chrF++, BLEU-char, CamemBERT,
# Kendall τ). 'Ordre S-TAM-V' exclu : Kuma-only (cf. fig9/fig10), pas
# comparable à Google/NLLB — un radar 3 systèmes ne peut montrer que des
# axes où les 3 ont une valeur réelle.
# τ (∈[-1,+1]) rescalé en 0-100 via (τ+1)/2*100 pour partager l'échelle
# radiale des autres axes (tous déjà 0-100).
# ─────────────────────────────────────────────────────────────────────────
def fig_radar_systems(rows, summary, outdir):
    metrics = ['EXM', 'chrF', 'chrF++', 'BLEU\n(char)', 'CamemBERT', 'Kendall τ\n(word order)']

    def _vals(sysname):
        s = summary[sysname]
        tau_rescaled = (s['tau_word_order'] + 1) / 2 * 100
        return [s['EXM'], s['chrF_sentence_avg'], s['chrFpp_sentence_avg'],
                s['BLEU_sentence_avg'], s['CamemBERT_avg'] * 100, tau_rescaled]

    data = {
        'Kuma-MT':  _vals('kuma'),
        'Google':   _vals('google'),
        'NLLB-200': _vals('nllb'),
    }
    N = len(metrics)
    angles = [i * 2 * np.pi / N for i in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for sysname, vals in data.items():
        vals_plot = vals + [vals[0]]
        ax.plot(angles, vals_plot, color=SYS_COLORS[sysname], linewidth=2, label=sysname)
        ax.fill(angles, vals_plot, color=SYS_COLORS[sysname], alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8, color=C_GREY)
    ax.set_title(f"Radar Overview (n={summary['n_phrases']})", pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=False)
    _save(fig, outdir, 'fig8_radar_systems')


# ─────────────────────────────────────────────────────────────────────────
# FIG F — Ordre des mots S-TAM-V : conformité par groupe, Kuma-MT uniquement
# (métrique non calculable pour Google/NLLB — cf. décision 2026-07-08 : elle
# repose sur le ClauseTemplate KG + les slots réels S/TAM/O/V du tree interne
# Kuma, données qui n'existent pas pour une sortie de traduction externe)
# ─────────────────────────────────────────────────────────────────────────
def fig_word_order_by_group(rows, outdir, top_n=None):
    # top_n retiré (2026-07-10) : plus de troncature arbitraire, cf. décision
    # sur fig3/fig4 — un seul bar par groupe (pas 3 systèmes), donc pas
    # besoin de heatmap ici, juste afficher tous les groupes.
    groups = defaultdict(list)
    for r in rows:
        if r['wo_kuma'] is not None:
            groups[r['group']].append(r['wo_kuma'])

    items = sorted(groups.items(), key=lambda x: -len(x[1]))
    names = [f"{g} (n={len(v)})" for g, v in items]
    kuma  = [_pct(v) for _, v in items]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, max(5, len(names) * 0.28)))
    ax.barh(y, kuma, color=C_KUMA, edgecolor='black', linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel('% sentences conforming to KG ClauseTemplate')
    ax.set_title(f'Kuma-MT — Word order conformity (S-TAM-V) — ALL {len(names)} groups')
    _save(fig, outdir, 'fig9_word_order_by_group')


# ─────────────────────────────────────────────────────────────────────────
# FIG G — Structure Kuma-MT : TAM / NEG / slots (bar simple)
# CTA (Clause Type Accuracy) volontairement exclue : 39.7%, anomalie
# nettement en-dehors de tous les autres scores Kuma (78-97%), cause encore
# non identifiée (cf. liste des items à valider, décision 2026-07-10) —
# affichée sans explication fiable, elle serait plus trompeuse qu'utile.
# À réintégrer une fois la cause tracée.
# ─────────────────────────────────────────────────────────────────────────
def fig_kuma_structure(rows, outdir):
    tam = _pct([r['tam_ok_kuma'] for r in rows if r.get('expected_tam')])
    neg = _pct([r['neg_ok'] for r in rows])
    ss  = _pct([r['slot_S'] for r in rows])
    sv  = _pct([r['slot_V'] for r in rows])
    so  = _pct([r['slot_O'] for r in rows])
    wo  = _pct([r['wo_kuma'] for r in rows])

    labels = ['TAM', 'NEG', 'Slot S', 'Slot V', 'Slot O', 'Word order\nS-TAM-V']
    values = [tam, neg, ss, sv, so, wo]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, values, color=C_KUMA, edgecolor='black', linewidth=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f'{v:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 108)
    ax.set_ylabel('%')
    ax.set_title(f'Kuma-MT Grammatical Structure (n={len(rows)})')
    _save(fig, outdir, 'fig10_kuma_structure')


# ─────────────────────────────────────────────────────────────────────────
# FIG G2 — BERTScore P/R/F1 (Kuma uniquement) : séparée de fig10 car
# c'est une famille de métrique différente — similarité sémantique par
# embeddings contextuels, PAS un contrôle de structure grammaticale
# (TAM/NEG/Slot/Word-order sont des booléens issus de règles, moyennés en %).
# Les mélanger dans un même bar chart sous un même axe donnait l'illusion
# fausse qu'il s'agit de la même famille de mesure (décision 2026-07-11).
# Google/NLLB : P/R/F1 laissés vides, back-traduction mot-à-mot trop
# coûteuse (~50s/phrase × 365, cf. décision antérieure sur cam_google/cam_nllb).
# ─────────────────────────────────────────────────────────────────────────
def fig_bertscore_detail(rows, outdir):
    bs_p  = _pct([r['p_kuma'] for r in rows])
    bs_r  = _pct([r['r_kuma'] for r in rows])
    bs_f1 = _pct([r['f1_kuma'] for r in rows])

    labels = ['Precision', 'Recall', 'F1']
    values = [bs_p, bs_r, bs_f1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, values, color=C_KUMA, edgecolor='black', linewidth=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f'{v:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 108)
    ax.set_ylabel('%')
    ax.set_title(f'Kuma-MT BERTScore — contextual semantic similarity (n={len(rows)})')
    _save(fig, outdir, 'fig11_bertscore_detail')


# ─────────────────────────────────────────────────────────────────────────
# FIG H — Matrice de confusion clause_type (Kuma, attendu vs détecté)
# ─────────────────────────────────────────────────────────────────────────
def fig_clause_confusion(rows, outdir, top_n=None):
    """Lignes = groupe grammatical attendu (macro-catégorie), colonnes =
    clause_type détecté par Kuma-MT. TOUS les groupes affichés (top_n
    retiré 2026-07-10, même décision que fig3/fig4/fig9 : pas de troncature
    arbitraire).

    Colonnes RÉORDONNÉES (pas tri alphabétique indépendant des 2 axes) :
    pour chaque ligne, dans l'ordre, la colonne où cette ligne a le plus de
    phrases devient sa colonne de diagonale (si pas déjà prise par une
    ligne précédente). Sans ça, une correspondance many-to-one VOULUE
    (ex: modal_pouvoir → verb_serial, cf. décision 2026-07-10 : ModalRule
    est mort, les modaux sont littéralement traités comme du verb_serial)
    finit éparpillée hors diagonale et se lit à tort comme une erreur de
    classification, alors que ce n'est que le nom de colonne qui diffère
    du nom de ligne. Après réordonnancement, seule une VRAIE incohérence
    de classification (Kuma détecte un clause_type différent de celui
    majoritairement observé pour ce groupe) reste hors diagonale."""
    pairs = [(r['group'], r.get('tree_clause', '') or '∅') for r in rows]
    if not pairs:
        return
    expected_types = sorted(set(p[0] for p in pairs))
    detected_types_all = sorted(set(p[1] for p in pairs))

    counts = defaultdict(int)
    for e, d in pairs:
        counts[(e, d)] += 1

    ordered_detected = []
    used = set()
    for e in expected_types:
        candidates = {d: counts.get((e, d), 0) for d in detected_types_all if d not in used}
        best_d = max(candidates, key=candidates.get) if candidates else None
        if best_d is not None and candidates[best_d] > 0:
            ordered_detected.append(best_d)
            used.add(best_d)
    leftover = sorted(d for d in detected_types_all if d not in used)
    detected_types = ordered_detected + leftover

    idx_e = {t: i for i, t in enumerate(expected_types)}
    idx_d = {t: i for i, t in enumerate(detected_types)}
    M = np.zeros((len(expected_types), len(detected_types)))
    for e, d in pairs:
        M[idx_e[e], idx_d[d]] += 1

    fig, ax = plt.subplots(figsize=(max(7, len(detected_types)*0.45),
                                    max(6, len(expected_types)*0.32)))
    im = ax.imshow(M, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(detected_types)))
    ax.set_xticklabels(detected_types, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(expected_types)))
    ax.set_yticklabels(expected_types, fontsize=7)
    ax.set_xlabel('clause_type detected (Kuma-MT)')
    ax.set_ylabel('expected grammatical group')
    ax.set_title(f'clause_type Confusion Matrix — ALL {len(expected_types)} groups\n'
                 f'columns reordered to align intended many-to-one mappings on the diagonal')
    for i in range(len(expected_types)):
        for j in range(len(detected_types)):
            if M[i, j] > 0:
                ax.text(j, i, int(M[i, j]), ha='center', va='center',
                        color='white' if M[i, j] > M.max()/2 else 'black', fontsize=6)
    fig.colorbar(im, ax=ax, label='N sentences', shrink=0.7)
    _save(fig, outdir, 'fig12_clause_confusion')


# ─────────────────────────────────────────────────────────────────────────
# FIG I — UD parse health (français, entrée)
# ─────────────────────────────────────────────────────────────────────────
def fig_ud_health(rows, outdir):
    checks = [('ud_single_root', 'Single ROOT'), ('ud_root_pos_ok', 'Valid ROOT POS'),
              ('ud_has_subject', 'Subject present'), ('ud_connected', 'Connected tree'),
              ('ud_no_dep_dep', 'No dep=dep')]
    valid = [r for r in rows if r.get('ud_single_root') is not None]
    if not valid:
        return
    labels = [lbl for _, lbl in checks]
    values = [_pct([r[k] for r in valid]) for k, _ in checks]
    colors = [C_NLLB if v >= 90 else C_GOOGLE if v >= 70 else C_RED for v in values]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, values, color=colors, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, values):
        ax.text(v + 0.5, b.get_y() + b.get_height()/2, f'{v:.1f}%',
                va='center', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 108)
    ax.set_xlabel('Conforming sentences (%)')
    ax.set_title(f'French Input UD Parse Health (n={len(valid)})')
    ax.invert_yaxis()
    _save(fig, outdir, 'fig13_ud_health')


# ─────────────────────────────────────────────────────────────────────────
# FIG J — Parsing Bambara (UPOS/UAS/LAS/LAS-c) vs UD_Bambara-CRB
# ─────────────────────────────────────────────────────────────────────────
def fig_bambara_parsing(json_path, outdir):
    if not json_path or not os.path.exists(json_path):
        print("    No bambara_parser_eval JSON — fig14 skipped")
        return
    with open(json_path, encoding='utf-8') as f:
        d = json.load(f)
    labels = ['UPOS', 'UAS', 'LAS', 'LAS-c']
    values = [d.get('UPOS', 0), d.get('UAS', 0), d.get('LAS', 0), d.get('LAS_main', 0)]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, values, color=[C_NLLB, C_KUMA, C_GOOGLE, C_PURPLE],
                  edgecolor='black', linewidth=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f"Bambara UDPipe Parser vs {d.get('treebank','UD_Bambara-CRB')}  "
                 f"(n={d.get('n_sentences','?')} sentences)")
    _save(fig, outdir, 'fig14_bambara_parsing')

    cpos = d.get('confusion_pos', {})
    if cpos:
        items = sorted(cpos.items(), key=lambda x: -x[1])[:15]
        names = [k.replace('->', ' → ') for k, _ in items]
        vals = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(7, max(4, len(names)*0.35)))
        y = np.arange(len(names))
        ax.barh(y, vals, color=C_RED, edgecolor='black', linewidth=0.4, alpha=0.85)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Number of errors")
        ax.set_title('Top UPOS confusions (gold → system) — Bambara parser')
        for yi, v in zip(y, vals):
            ax.text(v + 2, yi, str(v), va='center', fontsize=8)
        # Note de lecture : chaque barre "A → B" = nombre de tokens dont
        # l'étiquette CORRECTE (gold, UD_Bambara-CRB) est A mais que le
        # parseur UDPipe tiers a étiquetés B, cumulé sur tout le treebank
        # de test (~13 800 tokens). Ce n'est PAS un pourcentage : un chiffre
        # élevé signifie juste que cette confusion précise arrive souvent en
        # valeur absolue. VERB→NOUN, l'erreur la plus fréquente ici, confirme
        # indépendamment le bug trouvé en testant ce même parseur sur la
        # sortie réelle de Kuma (équatif "n yé kàramɔgɔkɛ yé" → copule 'yé'
        # étiquetée NOUN, nom 'kàramɔgɔkɛ' étiqueté VERB) — cf. décision de
        # ne pas utiliser ce parseur comme référence (build_reference_ud_tree
        # à la place).
        fig.text(0.5, -0.02,
                 "Reading: \"A → B\" = number of tokens whose gold label is A "
                 "but the parser predicted B (raw count over the whole treebank, not a %).",
                 ha='center', va='top', fontsize=8, style='italic', color=C_GREY, wrap=True)
        _save(fig, outdir, 'fig15_bambara_pos_confusion')


# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='?')
    ap.add_argument('json', nargs='?')
    ap.add_argument('--latest', action='store_true')
    ap.add_argument('--outdir', default='figures_final')
    ap.add_argument('--bambara-parser-json',
                     default='eval/bambara_parser_eval_20260705_044627.json')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    csv_path = args.csv
    json_path = args.json
    if args.latest or not csv_path:
        cands = sorted(glob.glob('eval_final_*.csv'))
        csv_path = cands[-1] if cands else None
        if csv_path:
            json_path = csv_path.replace('.csv', '_summary.json')

    if not csv_path or not os.path.exists(csv_path):
        print("No eval_final_*.csv found.")
        return

    print(f"CSV: {csv_path}")
    rows = load_csv(csv_path)
    with open(json_path, encoding='utf-8') as f:
        summary = json.load(f)
    print(f"Loaded {len(rows)} sentences — generating figures in {args.outdir}/…\n")

    fig_system_overview(rows, summary, args.outdir)
    fig_global_evaluation_grid(rows, summary, args.outdir)
    fig_chrf_by_group(rows, args.outdir)
    fig_camembert_by_group(rows, args.outdir)
    fig_scatter_chrf_cam(rows, args.outdir)
    fig_scatter_chrfpp(rows, args.outdir)
    fig_scatter_bleu_char(rows, args.outdir)
    fig_radar_systems(rows, summary, args.outdir)
    fig_word_order_by_group(rows, args.outdir)
    fig_kuma_structure(rows, args.outdir)
    fig_bertscore_detail(rows, args.outdir)
    fig_clause_confusion(rows, args.outdir)
    fig_ud_health(rows, args.outdir)
    fig_bambara_parsing(args.bambara_parser_json, args.outdir)

    print(f"\n All figures in {args.outdir}/ (PNG 300dpi + PDF)")


if __name__ == '__main__':
    main()
