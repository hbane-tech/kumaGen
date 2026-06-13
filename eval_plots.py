"""
eval_plots.py — Génération des graphiques d'évaluation pour le paper Kuma-MT
============================================================================
Lit un CSV produit par evaluate.py et génère des figures publication-ready
(300 DPI, palette sobre) dans un dossier figures/.

Usage :
  python eval_plots.py eval_20260608_120000.csv
  python eval_plots.py eval_20260608_120000.csv --outdir figures/
  python eval_plots.py --latest          # prend le CSV eval_*.csv le plus récent
"""

import os
import sys
import csv
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')          # backend non-interactif (serveur)
import matplotlib.pyplot as plt


# ── Style publication ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'font.size':         11,
    'font.family':       'DejaVu Sans',   # supporte les glyphes bambara (ɛ ɔ ɲ)
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# Palette daltonien-friendly (Okabe-Ito)
C_BLUE   = '#0072B2'
C_ORANGE = '#E69F00'
C_GREEN  = '#009E73'
C_RED    = '#D55E00'
C_PURPLE = '#CC79A7'
C_GREY   = '#999999'


def _str2bool(s):
    return str(s).strip().lower() in ('true', '1', 'yes', 'oui')


def load_csv(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            r['exm']  = _str2bool(r.get('exm', ''))
            r['chrf'] = float(r.get('chrf') or 0)
            r['p1']   = float(r['p1'])  if r.get('p1')  not in (None, '', 'None') else None
            r['kgh']  = float(r['kgh']) if r.get('kgh') not in (None, '', 'None') else None
            for b in ('clause_type_ok', 'tam_ok', 'neg_ok', 'slot_S', 'slot_V', 'slot_O',
                      'ud_single_root', 'ud_root_pos_ok', 'ud_has_subject',
                      'ud_connected', 'ud_no_dep_dep', 'word_order_ok'):
                if b in r:
                    r[b] = _str2bool(r.get(b, ''))
            r['ud_score'] = (float(r['ud_score'])
                             if r.get('ud_score') not in (None, '', 'None') else None)
            rows.append(r)
    return rows


def _clause_group(category, mapping):
    return mapping.get(category, category)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Métriques globales (bar chart)
# ─────────────────────────────────────────────────────────────────────────────
def fig_global_metrics(rows, outdir):
    n = len(rows)
    exm  = sum(r['exm'] for r in rows) / n * 100
    chrf = sum(r['chrf'] for r in rows) / n
    p1   = _avg_pct([r['p1']  for r in rows if r['p1']  is not None])
    kgh  = _avg_pct([r['kgh'] for r in rows if r['kgh'] is not None])
    cta  = _avg_pct([r['clause_type_ok'] for r in rows if 'clause_type_ok' in r])
    tam  = _avg_pct([r['tam_ok'] for r in rows if r.get('ref_tam')])
    neg  = _avg_pct([r['neg_ok'] for r in rows if 'neg_ok' in r])

    labels = ['EXM', 'chrF', 'P@1', 'KGH', 'CTA', 'TAM', 'NEG']
    values = [exm, chrf, p1, kgh, cta, tam, neg]
    groups = ['Surface', 'Surface', 'Sémantique', 'Sémantique',
              'Structure', 'Structure', 'Structure']
    colors = {'Surface': C_BLUE, 'Sémantique': C_ORANGE, 'Structure': C_GREEN}
    bar_colors = [colors[g] for g in groups]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=bar_colors, edgecolor='black', linewidth=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f'{v:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_ylabel('Score (%)  /  chrF (0–100)')
    ax.set_title(f'Métriques globales Kuma-MT (n={n})')

    # Légende des groupes — placée sous le titre, hors des barres
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=colors[g], label=g) for g in colors]
    ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, -0.10),
              ncol=3, frameon=False)

    _save(fig, outdir, 'fig1_global_metrics')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — EXM & chrF par clause_type (grouped bar horizontal)
# ─────────────────────────────────────────────────────────────────────────────
def fig_by_clause_type(rows, outdir, mapping):
    groups = defaultdict(lambda: {'exm': [], 'chrf': []})
    for r in rows:
        g = _clause_group(r['category'], mapping)
        groups[g]['exm'].append(r['exm'])
        groups[g]['chrf'].append(r['chrf'])

    # trier par nombre décroissant
    items = sorted(groups.items(), key=lambda x: -len(x[1]['exm']))
    names = [f"{g}  (n={len(v['exm'])})" for g, v in items]
    exm   = [np.mean(v['exm']) * 100 for _, v in items]
    chrf  = [np.mean(v['chrf'])       for _, v in items]

    y = np.arange(len(names))
    h = 0.38
    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.42)))
    ax.barh(y + h/2, exm,  height=h, color=C_BLUE,   label='EXM (%)',  edgecolor='black', linewidth=0.4)
    ax.barh(y - h/2, chrf, height=h, color=C_ORANGE, label='chrF',     edgecolor='black', linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel('Score')
    ax.set_title('Performance par type de clause')
    ax.legend(loc='lower right', frameon=False)
    _save(fig, outdir, 'fig2_by_clause_type')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Distribution des scores chrF (histogramme)
# ─────────────────────────────────────────────────────────────────────────────
def fig_chrf_distribution(rows, outdir):
    chrf = [r['chrf'] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(chrf, bins=20, range=(0, 100), color=C_BLUE,
            edgecolor='black', linewidth=0.6, alpha=0.85)
    mean = np.mean(chrf)
    median = np.median(chrf)
    ax.axvline(mean,   color=C_RED,    linestyle='--', linewidth=1.5, label=f'Moyenne {mean:.1f}')
    ax.axvline(median, color=C_GREEN,  linestyle=':',  linewidth=1.5, label=f'Médiane {median:.1f}')
    ax.set_xlabel('chrF')
    ax.set_ylabel('Nombre de phrases')
    ax.set_title('Distribution des scores chrF')
    ax.legend(frameon=False)
    _save(fig, outdir, 'fig3_chrf_distribution')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Matrice de confusion clause_type (attendu vs détecté)
# ─────────────────────────────────────────────────────────────────────────────
def fig_clause_confusion(rows, outdir, mapping):
    pairs = [(mapping.get(r['category'], r['category']),
              r.get('tree_clause', '') or '∅')
             for r in rows if r.get('tree_clause') is not None]
    if not pairs:
        return
    expected_types = sorted(set(p[0] for p in pairs))
    detected_types = sorted(set(p[1] for p in pairs))
    idx_e = {t: i for i, t in enumerate(expected_types)}
    idx_d = {t: i for i, t in enumerate(detected_types)}

    M = np.zeros((len(expected_types), len(detected_types)))
    for e, d in pairs:
        M[idx_e[e], idx_d[d]] += 1

    fig, ax = plt.subplots(figsize=(max(7, len(detected_types)*0.7),
                                    max(6, len(expected_types)*0.5)))
    im = ax.imshow(M, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(detected_types)))
    ax.set_xticklabels(detected_types, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(expected_types)))
    ax.set_yticklabels(expected_types, fontsize=8)
    ax.set_xlabel('clause_type détecté')
    ax.set_ylabel('clause_type attendu')
    ax.set_title('Matrice de confusion — classification des clauses')
    # Annoter les cellules non nulles
    for i in range(len(expected_types)):
        for j in range(len(detected_types)):
            if M[i, j] > 0:
                ax.text(j, i, int(M[i, j]), ha='center', va='center',
                        color='white' if M[i, j] > M.max()/2 else 'black',
                        fontsize=8)
    fig.colorbar(im, ax=ax, label='N phrases', shrink=0.7)
    _save(fig, outdir, 'fig4_clause_confusion')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Corrélation chrF vs P@1 (scatter + Kendall τ)
# ─────────────────────────────────────────────────────────────────────────────
def fig_chrf_vs_p1(rows, outdir):
    pts = [(r['chrf'], r['p1'] * 100) for r in rows if r['p1'] is not None]
    if len(pts) < 3:
        return
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    tau = _kendall_tau(x, y)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(x, y, color=C_PURPLE, alpha=0.6, edgecolor='black', linewidth=0.4, s=40)
    # ligne de tendance
    if len(set(x)) > 1:
        z = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, np.poly1d(z)(xs), color=C_RED, linewidth=1.5, linestyle='--')
    ax.set_xlabel('chrF (surface)')
    ax.set_ylabel('P@1 sémantique (%)')
    ax.set_title(f'Corrélation surface ↔ sémantique  (τ = {tau:+.3f})')
    _save(fig, outdir, 'fig5_chrf_vs_p1')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — TAM accuracy : attendu vs détecté (matrice)
# ─────────────────────────────────────────────────────────────────────────────
def fig_tam_matrix(rows, outdir):
    pairs = [(r.get('ref_tam', '') or '∅', r.get('tree_tam', '') or '∅')
             for r in rows if r.get('ref_tam')]
    if not pairs:
        return
    tam_ref = sorted(set(p[0] for p in pairs))
    tam_det = sorted(set(p[1] for p in pairs))
    idx_r = {t: i for i, t in enumerate(tam_ref)}
    idx_d = {t: i for i, t in enumerate(tam_det)}
    M = np.zeros((len(tam_ref), len(tam_det)))
    for r_, d_ in pairs:
        M[idx_r[r_], idx_d[d_]] += 1

    fig, ax = plt.subplots(figsize=(max(6, len(tam_det)*0.8),
                                    max(5, len(tam_ref)*0.6)))
    im = ax.imshow(M, cmap='Greens', aspect='auto')
    ax.set_xticks(range(len(tam_det)))
    ax.set_xticklabels(tam_det, rotation=45, ha='right')
    ax.set_yticks(range(len(tam_ref)))
    ax.set_yticklabels(tam_ref)
    ax.set_xlabel('TAM détecté (tree)')
    ax.set_ylabel('TAM attendu (référence)')
    ax.set_title('Matrice TAM — marqueurs aspecto-temporels')
    for i in range(len(tam_ref)):
        for j in range(len(tam_det)):
            if M[i, j] > 0:
                ax.text(j, i, int(M[i, j]), ha='center', va='center',
                        color='white' if M[i, j] > M.max()/2 else 'black', fontsize=9)
    fig.colorbar(im, ax=ax, label='N', shrink=0.7)
    _save(fig, outdir, 'fig6_tam_matrix')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — Parsing vs UD French GSD (UPOS / UAS / LAS) + confusions POS
# ─────────────────────────────────────────────────────────────────────────────
def fig_parser(json_path, outdir):
    """Lit le JSON de eval/parser_eval.py et trace métriques + confusions POS."""
    import json
    with open(json_path, encoding='utf-8') as f:
        d = json.load(f)

    # — 7a : barres UPOS / UAS / LAS / LAS-c —
    labels = ['UPOS', 'UAS', 'LAS', 'LAS-c']
    values = [d.get('UPOS', 0), d.get('UAS', 0),
              d.get('LAS', 0), d.get('LAS_main', 0)]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, values, color=[C_GREEN, C_BLUE, C_ORANGE, C_PURPLE],
                  edgecolor='black', linewidth=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f"Parsing vs {d.get('treebank','UD')}  "
                 f"(n={d.get('n_sentences','?')} phrases, "
                 f"{d.get('n_scored','?')} tokens)")
    _save(fig, outdir, 'fig7_parser_metrics')

    # — 7b : top confusions POS (barres horizontales) —
    cpos = d.get('confusion_pos', {})
    if cpos:
        items = sorted(cpos.items(), key=lambda x: -x[1])[:12]
        names = [k.replace('->', ' → ') for k, _ in items]
        vals  = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(7, max(4, len(names)*0.4)))
        y = np.arange(len(names))
        ax.barh(y, vals, color=C_RED, edgecolor='black', linewidth=0.4, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Nombre d\'erreurs')
        ax.set_title('Top confusions POS (gold → système)')
        for yi, v in zip(y, vals):
            ax.text(v + 0.3, yi, str(v), va='center', fontsize=9)
        _save(fig, outdir, 'fig8_pos_confusion')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9 — UD Parse Health (5 checks + score moyen)
# ─────────────────────────────────────────────────────────────────────────────
def fig_ud_health(rows, outdir):
    checks = [
        ('ud_single_root',   'ROOT unique'),
        ('ud_root_pos_ok',   'ROOT pos valide'),
        ('ud_has_subject',   'Sujet présent'),
        ('ud_connected',     'Arbre connexe'),
        ('ud_no_dep_dep',    'Pas de dep=dep'),
    ]
    valid = [r for r in rows if r.get('ud_single_root') is not None]
    if not valid:
        print("  ⚠️  Pas de données UD — fig9 ignorée")
        return

    labels = [lbl for _, lbl in checks]
    values = [_avg_pct([r[k] for r in valid]) for k, _ in checks]
    ud_avg = sum(r['ud_score'] for r in valid if r.get('ud_score') is not None)
    ud_avg /= max(1, sum(1 for r in valid if r.get('ud_score') is not None))
    ud_pct  = ud_avg / 5 * 100

    colors = [C_GREEN if v >= 90 else C_ORANGE if v >= 70 else C_RED for v in values]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, values, color=colors, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, values):
        ax.text(v + 0.5, b.get_y() + b.get_height()/2,
                f'{v:.1f}%', va='center', fontsize=10, fontweight='bold')
    ax.axvline(ud_pct, color=C_PURPLE, linestyle='--', linewidth=1.5,
               label=f'Score UD moyen {ud_pct:.1f}%')
    ax.set_xlim(0, 108)
    ax.set_xlabel('Phrases conformes (%)')
    ax.set_title(f'Santé du parse UD (spaCy FR, n={len(valid)})')
    ax.legend(frameon=False, loc='lower right')
    ax.invert_yaxis()
    _save(fig, outdir, 'fig9_ud_health')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 10 — Radar chart : vue d'ensemble de toutes les métriques
# ─────────────────────────────────────────────────────────────────────────────
def fig_radar(rows, outdir, parser_json=None):
    n = len(rows)

    def _p(lst):
        return sum(lst) / len(lst) * 100 if lst else 0.0

    metrics = {
        'EXM':    _avg_pct([r['exm']  for r in rows]),
        'chrF':   _avg_pct([r['chrf'] for r in rows]),
        'P@1':    _p([r['p1']  * 100 for r in rows if r['p1']  is not None]),
        'KGH':    _p([r['kgh'] * 100 for r in rows if r['kgh'] is not None]),
        'CTA':    _p([r['clause_type_ok'] for r in rows if 'clause_type_ok' in r]),
        'TAM':    _p([r['tam_ok'] for r in rows if r.get('ref_tam')]),
        'NEG':    _p([r['neg_ok'] for r in rows if 'neg_ok' in r]),
        'WO':     _p([r['word_order_ok'] for r in rows
                      if r.get('word_order_ok') is not None]),
        'UD':     sum(r['ud_score'] for r in rows if r.get('ud_score') is not None)
                  / max(1, sum(1 for r in rows if r.get('ud_score') is not None))
                  / 5 * 100,
    }

    # Parser metrics from JSON if available
    if parser_json and os.path.exists(parser_json):
        import json
        with open(parser_json, encoding='utf-8') as f:
            pj = json.load(f)
        metrics['LAS']  = pj.get('LAS', 0)
        metrics['UPOS'] = pj.get('UPOS', 0)

    labels = list(metrics.keys())
    values = list(metrics.values())
    N = len(labels)
    angles = [i * 2 * np.pi / N for i in range(N)] + [0]
    values_plot = values + [values[0]]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    ax.plot(angles, values_plot, color=C_BLUE, linewidth=2)
    ax.fill(angles, values_plot, color=C_BLUE, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8, color=C_GREY)
    ax.set_title(f'Vue d\'ensemble Kuma-MT (n={n})', pad=20, fontsize=12)
    # Annoter chaque point
    for angle, val, lbl in zip(angles[:-1], values, labels):
        ax.text(angle, val + 7, f'{val:.0f}', ha='center', va='center',
                fontsize=9, fontweight='bold', color=C_BLUE)
    _save(fig, outdir, 'fig10_radar')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 11 — Word order OK par clause_type (stacked bar)
# ─────────────────────────────────────────────────────────────────────────────
def fig_word_order(rows, outdir, mapping):
    valid = [r for r in rows if r.get('word_order_ok') is not None]
    if not valid:
        print("  ⚠️  Pas de données word_order — fig11 ignorée")
        return

    groups = defaultdict(lambda: {'ok': 0, 'fail': 0})
    for r in valid:
        g = _clause_group(r['category'], mapping)
        if r['word_order_ok']:
            groups[g]['ok']   += 1
        else:
            groups[g]['fail'] += 1

    items = sorted(groups.items(), key=lambda x: -(x[1]['ok'] + x[1]['fail']))
    names  = [g for g, _ in items]
    ok_pct = [v['ok'] / (v['ok'] + v['fail']) * 100 for _, v in items]
    fa_pct = [100 - p for p in ok_pct]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.40)))
    ax.barh(y, ok_pct, color=C_GREEN,  edgecolor='black', linewidth=0.4,
            label='S-TAM-V ✓', alpha=0.9)
    ax.barh(y, fa_pct, left=ok_pct, color=C_RED, edgecolor='black', linewidth=0.4,
            label='S-TAM-V ✗', alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel('% phrases')
    ax.set_title(f'Conformité ordre des mots S-TAM-V (n={len(valid)})')
    ax.legend(loc='lower right', frameon=False)
    _save(fig, outdir, 'fig11_word_order')


# ── Helpers ────────────────────────────────────────────────────────────────────
def _avg_pct(lst):
    return sum(lst) / len(lst) * 100 if lst else 0.0


def _kendall_tau(a, b):
    n = len(a)
    c = d = 0
    for i in range(n):
        for j in range(i+1, n):
            s = (a[i]-a[j]) * (b[i]-b[j])
            if s > 0: c += 1
            elif s < 0: d += 1
    tot = n*(n-1)/2
    return (c-d)/tot if tot else 0.0


def _save(fig, outdir, name):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {name}.png / .pdf")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Graphiques évaluation Kuma-MT')
    parser.add_argument('csv', nargs='?', help='CSV produit par evaluate.py')
    parser.add_argument('--latest', action='store_true', help='Dernier eval_*.csv')
    parser.add_argument('--outdir', default='figures', help='Dossier de sortie')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # CSV traduction (optionnel) : eval_*.csv
    csv_candidates = sorted(glob.glob('eval_*.csv'))
    csv_path = args.csv or (csv_candidates[-1] if csv_candidates else None)

    # JSON parsing (optionnel) : parser_eval_*.json
    parser_candidates = sorted(glob.glob('parser_eval_*.json'))
    parser_json = parser_candidates[-1] if parser_candidates else None
    if parser_json:
        print(f"JSON parsing : {parser_json}")
    else:
        print("ℹ️  Aucun parser_eval_*.json — figures parsing ignorées "
              "(lance : python -m eval.parser_eval)")

    if csv_path:
        print(f"CSV traduction : {csv_path}")
        rows = load_csv(csv_path)
        print(f"Chargé : {len(rows)} phrases\nGénération des figures dans {args.outdir}/ …\n")
        try:
            from evaluate import CATEGORY_TO_CLAUSE as mapping
        except Exception:
            mapping = {}
        # Figures traduction
        fig_global_metrics(rows, args.outdir)
        fig_by_clause_type(rows, args.outdir, mapping)
        fig_chrf_distribution(rows, args.outdir)
        fig_clause_confusion(rows, args.outdir, mapping)
        fig_chrf_vs_p1(rows, args.outdir)
        fig_tam_matrix(rows, args.outdir)
        # Nouvelles figures
        fig_ud_health(rows, args.outdir)
        fig_radar(rows, args.outdir, parser_json=parser_json)
        fig_word_order(rows, args.outdir, mapping)
    else:
        print("ℹ️  Aucun eval_*.csv — figures traduction ignorées.")

    if parser_json:
        print(f"\nGénération des figures parsing …")
        fig_parser(parser_json, args.outdir)

    print(f"\n✅ Figures dans {args.outdir}/  (PNG 300dpi + PDF vectoriel)")


if __name__ == '__main__':
    main()
