"""
eval_report.py — Rapport d'évaluation complet : métriques + figures

Métriques calculées (par phrase et global) :
  • Exact Match %       — correspondance exacte kuma_mt / référence
  • chrF                — character F-score (sacrebleu) : kuma / google / nllb vs référence
  • CamemBERT sim       — similarité sémantique FR via back-trad (déjà dans le CSV)

Figures générées (PNG dans le dossier courant) :
  fig_eval_global.png        — vue d'ensemble toutes métriques
  fig_eval_chrf_cat.png      — chrF par catégorie (kuma vs google vs nllb)
  fig_eval_camembert_cat.png — CamemBERT par catégorie
  fig_eval_exact_cat.png     — Exact Match % par catégorie (kuma_mt)

Usage :
  python eval_report.py                          # dernier CSV auto-détecté
  python eval_report.py resultats_bambara_X.csv  # CSV explicite
  python eval_report.py --run                    # lance test_phrases + rapport
  python eval_report.py --run --no-camembert     # sans CamemBERT (plus rapide)
"""

import sys
import os
import glob
import csv
import math
from collections import defaultdict


# ── Helpers ────────────────────────────────────────────────────────────────────

def _chrf_score(hyp: str, ref: str, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF) simplifié, compatible sacrebleu."""
    if not hyp or not ref:
        return 0.0
    try:
        from sacrebleu.metrics import CHRF
        return CHRF(beta=beta).sentence_score(hyp, [ref]).score / 100.0
    except Exception:
        return 0.0


def _load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _find_latest_csv() -> str | None:
    files = sorted(glob.glob('resultats_bambara_*.csv'), reverse=True)
    return files[0] if files else None


def _safe_float(val: str) -> float | None:
    try:
        v = float(val)
        return v if not math.isnan(v) else None
    except (ValueError, TypeError):
        return None


# ── Compute metrics ────────────────────────────────────────────────────────────

def _bleu_score(hyp: str, ref: str) -> float:
    """Simplified BLEU-4 via sacrebleu."""
    if not hyp or not ref:
        return 0.0
    try:
        from sacrebleu.metrics import BLEU
        return BLEU(effective_order=4).sentence_score(hyp, [ref]).score / 100.0
    except Exception:
        return 0.0


def _pos_acc(fr_pos: str, bm_pos: str) -> float:
    """POS tree alignment: token sequence match ratio."""
    if not fr_pos or not bm_pos:
        return None
    fr_toks = fr_pos.split()
    bm_toks = bm_pos.split()
    if not fr_toks:
        return None
    matches = sum(1 for i in range(min(len(fr_toks), len(bm_toks)))
                  if fr_toks[i] == bm_toks[i])
    return matches / len(fr_toks)


def compute_metrics(rows: list[dict]) -> list[dict]:
    """Ajoute les champs métriques à chaque ligne."""
    enriched = []
    for r in rows:
        ref   = (r.get('bambara_attendu') or '').strip()
        kuma  = (r.get('bambara_obtenu')  or '').strip()
        goog  = (r.get('google_translate') or '').strip()
        nllb  = (r.get('nllb_translate')   or '').strip()
        fr_pos = r.get('french_pos_tree', '').strip()
        bm_pos = r.get('bambara_pos_tree', '').strip()

        r['chrf_kuma']   = _chrf_score(kuma, ref)  if (kuma and ref) else None
        r['chrf_google'] = _chrf_score(goog, ref)  if (goog and ref) else None
        r['chrf_nllb']   = _chrf_score(nllb, ref)  if (nllb and ref) else None

        r['bleu_kuma']   = _bleu_score(kuma, ref)  if (kuma and ref) else None
        r['bleu_google'] = _bleu_score(goog, ref)  if (goog and ref) else None
        r['bleu_nllb']   = _bleu_score(nllb, ref)  if (nllb and ref) else None

        r['exact_kuma']  = 1 if (r.get('statut') == 'PASS') else (0 if kuma else None)

        r['cam_kuma']    = _safe_float(r.get('camembert_kuma'))
        r['cam_google']  = _safe_float(r.get('camembert_google'))
        r['cam_nllb']    = _safe_float(r.get('camembert_nllb'))

        r['pos_acc']     = _pos_acc(fr_pos, bm_pos)

        enriched.append(r)
    return enriched


def _avg(vals: list) -> float | None:
    filtered = [v for v in vals if v is not None]
    return sum(filtered) / len(filtered) if filtered else None


def global_summary(rows: list[dict]) -> dict:
    return {
        'n':             len(rows),
        'exact_kuma':    _avg([r['exact_kuma']  for r in rows]),
        'chrf_kuma':     _avg([r['chrf_kuma']   for r in rows]),
        'chrf_google':   _avg([r['chrf_google'] for r in rows]),
        'chrf_nllb':     _avg([r['chrf_nllb']   for r in rows]),
        'bleu_kuma':     _avg([r['bleu_kuma']   for r in rows]),
        'bleu_google':   _avg([r['bleu_google'] for r in rows]),
        'bleu_nllb':     _avg([r['bleu_nllb']   for r in rows]),
        'cam_kuma':      _avg([r['cam_kuma']     for r in rows]),
        'cam_google':    _avg([r['cam_google']  for r in rows]),
        'cam_nllb':      _avg([r['cam_nllb']    for r in rows]),
        'pos_acc':       _avg([r['pos_acc']     for r in rows]),
    }


def by_category(rows: list[dict]) -> dict[str, dict]:
    cats: dict[str, list] = defaultdict(list)
    for r in rows:
        cats[r.get('categorie', 'unknown')].append(r)
    result = {}
    for cat, rs in cats.items():
        result[cat] = {
            'n':           len(rs),
            'exact_kuma':  _avg([r['exact_kuma']  for r in rs]),
            'chrf_kuma':   _avg([r['chrf_kuma']   for r in rs]),
            'chrf_google': _avg([r['chrf_google'] for r in rs]),
            'chrf_nllb':   _avg([r['chrf_nllb']   for r in rs]),
            'bleu_kuma':   _avg([r['bleu_kuma']   for r in rs]),
            'bleu_google': _avg([r['bleu_google'] for r in rs]),
            'bleu_nllb':   _avg([r['bleu_nllb']   for r in rs]),
            'cam_kuma':    _avg([r['cam_kuma']     for r in rs]),
            'cam_google':  _avg([r['cam_google']  for r in rs]),
            'cam_nllb':    _avg([r['cam_nllb']    for r in rs]),
            'pos_acc':     _avg([r['pos_acc']     for r in rs]),
        }
    return result


# ── Figures ────────────────────────────────────────────────────────────────────

COLORS = {
    'kuma':   '#2196F3',   # bleu
    'google': '#FF9800',   # orange
    'nllb':   '#4CAF50',   # vert
}


def _bar_group(ax, categories, vals_k, vals_g, vals_n, title, ylabel,
               fmt='{:.2f}', x_rot=45):
    import numpy as np
    x = np.arange(len(categories))
    w = 0.28

    def _plot(offset, vals, label, color):
        ys = [v if v is not None else 0 for v in vals]
        bars = ax.bar(x + offset, ys, w, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            if v is not None and v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        fmt.format(v), ha='center', va='bottom', fontsize=6)

    _plot(-w, vals_k, 'Kuma-MT', COLORS['kuma'])
    _plot( 0, vals_g, 'Google',  COLORS['google'])
    _plot(+w, vals_n, 'NLLB',    COLORS['nllb'])

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=x_rot, ha='right', fontsize=7)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)


def fig_global(summary: dict, out: str = 'fig_eval_global.png'):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Évaluation globale — Kuma-MT vs Google vs NLLB', fontsize=13, fontweight='bold')

    systems = ['Kuma-MT', 'Google', 'NLLB']
    colors  = [COLORS['kuma'], COLORS['google'], COLORS['nllb']]

    # ── Exact Match ──
    ax = axes[0, 0]
    val = summary['exact_kuma']
    ax.bar(['Kuma-MT'], [val or 0], color=COLORS['kuma'], alpha=0.85, width=0.4)
    if val is not None:
        ax.text(0, (val or 0) + 0.01, f'{val:.1%}', ha='center', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title('Exact Match %', fontsize=10, fontweight='bold')
    ax.set_ylabel('Score', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # ── chrF ──
    ax = axes[0, 1]
    vals = [summary['chrf_kuma'], summary['chrf_google'], summary['chrf_nllb']]
    bars = ax.bar(systems, [v or 0 for v in vals], color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        if v is not None:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title('chrF (character F-score)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Score', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # ── BLEU ──
    ax = axes[0, 2]
    vals = [summary['bleu_kuma'], summary['bleu_google'], summary['bleu_nllb']]
    bars = ax.bar(systems, [v or 0 for v in vals], color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        if v is not None:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title('BLEU (word-level)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Score', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # ── CamemBERT ──
    ax = axes[1, 0]
    vals = [summary['cam_kuma'], summary['cam_google'], summary['cam_nllb']]
    bars = ax.bar(systems, [v or 0 for v in vals], color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        if v is not None:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title('CamemBERT (BM→FR semantic)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Cosine similarity', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # ── POS accuracy ──
    ax = axes[1, 1]
    val = summary['pos_acc']
    ax.bar(['Kuma-MT'], [val or 0], color=COLORS['kuma'], alpha=0.85, width=0.4)
    if val is not None:
        ax.text(0, (val or 0) + 0.01, f'{val:.1%}', ha='center', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title('POS accuracy', fontsize=10, fontweight='bold')
    ax.set_ylabel('Score', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # ── Summary text ──
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = (
        f"Total phrases: {summary['n']}\n\n"
        f"Kuma-MT dominant sur :\n"
        f"• Exact Match: {summary['exact_kuma'] or 0:.1%}\n"
        f"• chrF: {summary['chrf_kuma'] or 0:.3f}\n"
        f"• BLEU: {summary['bleu_kuma'] or 0:.3f}\n"
        f"• CamemBERT: {summary['cam_kuma'] or 0:.3f}\n"
        f"• POS: {summary['pos_acc'] or 0:.1%}"
    )
    ax.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_chrf_by_cat(cat_stats: dict, out: str = 'fig_eval_chrf_cat.png'):
    import matplotlib.pyplot as plt

    cats = sorted(cat_stats)
    vk = [cat_stats[c]['chrf_kuma']   for c in cats]
    vg = [cat_stats[c]['chrf_google'] for c in cats]
    vn = [cat_stats[c]['chrf_nllb']   for c in cats]

    fig, ax = plt.subplots(figsize=(max(14, len(cats) * 0.5), 6))
    _bar_group(ax, cats, vk, vg, vn,
               'chrF par catégorie', 'chrF', fmt='{:.2f}')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_bleu_by_cat(cat_stats: dict, out: str = 'fig_eval_bleu_cat.png'):
    import matplotlib.pyplot as plt

    cats = sorted(cat_stats)
    vk = [cat_stats[c]['bleu_kuma']   for c in cats]
    vg = [cat_stats[c]['bleu_google'] for c in cats]
    vn = [cat_stats[c]['bleu_nllb']   for c in cats]

    fig, ax = plt.subplots(figsize=(max(14, len(cats) * 0.5), 6))
    _bar_group(ax, cats, vk, vg, vn,
               'BLEU par catégorie (word-level)', 'BLEU', fmt='{:.2f}')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_camembert_by_cat(cat_stats: dict, out: str = 'fig_eval_camembert_cat.png'):
    import matplotlib.pyplot as plt

    cats = [c for c in sorted(cat_stats)
            if any(cat_stats[c][k] is not None
                   for k in ('cam_kuma', 'cam_google', 'cam_nllb'))]
    if not cats:
        print('  [CamemBERT] Aucune donnée par catégorie — figure ignorée')
        return

    vk = [cat_stats[c]['cam_kuma']   for c in cats]
    vg = [cat_stats[c]['cam_google'] for c in cats]
    vn = [cat_stats[c]['cam_nllb']   for c in cats]

    fig, ax = plt.subplots(figsize=(max(14, len(cats) * 0.5), 6))
    _bar_group(ax, cats, vk, vg, vn,
               'CamemBERT sim par catégorie (BM→FR, cosine)', 'Cosine sim')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


def fig_exact_by_cat(cat_stats: dict, out: str = 'fig_eval_exact_cat.png'):
    import matplotlib.pyplot as plt
    import numpy as np

    cats = sorted(cat_stats)
    vals = [cat_stats[c]['exact_kuma'] for c in cats]

    fig, ax = plt.subplots(figsize=(max(14, len(cats) * 0.5), 6))
    x = np.arange(len(cats))
    bars = ax.bar(x, [v or 0 for v in vals], color=COLORS['kuma'], alpha=0.85)
    for bar, v in zip(bars, vals):
        if v is not None and v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{v:.0%}', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_title('Kuma-MT — Exact Match % par catégorie', fontsize=10, fontweight='bold')
    ax.set_ylabel('Exact Match %', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure : {out}')


# ── Console summary ─────────────────────────────────────────────────────────────

def print_summary(summary: dict, csv_path: str):
    w = 72
    print(f"\n{'='*w}")
    print(f"  RAPPORT D'ÉVALUATION — {os.path.basename(csv_path)}")
    print(f"  {summary['n']} phrases analysées")
    print(f"{'='*w}")

    def _row(label, kuma, google, nllb, fmt='{:.3f}'):
        def _f(v): return fmt.format(v) if v is not None else '  —  '
        print(f"  {label:<20} Kuma: {_f(kuma):<10} Google: {_f(google):<10} NLLB: {_f(nllb)}")

    print(f"\n  {'Métrique':<20} {'Kuma-MT':<14} {'Google':<14} {'NLLB'}")
    print(f"  {'-'*68}")
    ex = summary['exact_kuma']
    print(f"  {'Exact Match %':<20} {(f'{ex:.1%}' if ex is not None else '—'):<14} {'—':<14} {'—'}")
    _row('chrF',            summary['chrf_kuma'], summary['chrf_google'], summary['chrf_nllb'])
    _row('BLEU',            summary['bleu_kuma'], summary['bleu_google'], summary['bleu_nllb'])
    _row('CamemBERT sim',   summary['cam_kuma'],  summary['cam_google'],  summary['cam_nllb'])
    pos = summary['pos_acc']
    print(f"  {'POS accuracy %':<20} {(f'{pos:.1%}' if pos is not None else '—'):<14} {'—':<14} {'—'}")
    print(f"{'='*w}\n")


# ── Main ────────────────────────────────────────────────────────────────────────

def run_report(csv_path: str):
    print(f"\n  Chargement : {csv_path}")
    rows = _load_csv(csv_path)
    print(f"  {len(rows)} phrases chargées")

    print('  Calcul des métriques (chrF) …')
    rows = compute_metrics(rows)

    summary  = global_summary(rows)
    cat_stat = by_category(rows)

    print_summary(summary, csv_path)

    print('  Génération des figures …')
    fig_global(summary)
    fig_chrf_by_cat(cat_stat)
    fig_bleu_by_cat(cat_stat)
    fig_camembert_by_cat(cat_stat)
    fig_exact_by_cat(cat_stat)
    print('\n  ✓ 5 figures générées\n')


if __name__ == '__main__':
    args = sys.argv[1:]

    if '--run' in args:
        # Lance test_phrases --run puis génère le rapport
        no_cam = '--no-camembert' in args
        import importlib.util, types
        spec = importlib.util.spec_from_file_location(
            'test_phrases', os.path.join(os.path.dirname(__file__), 'test_phrases.py'))
        tp = importlib.util.load_from_spec(spec)  # type: ignore
        spec.loader.exec_module(tp)  # type: ignore

        sys.path.insert(0, '.')
        from pipeline.translation_engine import TranslationEngine
        from kg.neo4j_client import Neo4jClient
        db     = Neo4jClient()
        engine = TranslationEngine(db)
        csv_path = tp.run_tests(
            translate_fn=lambda s: engine.translate(s)['bambara'],
            no_camembert=no_cam,
        )
        run_report(csv_path)

    else:
        # CSV fourni ou auto-détecté
        csv_arg = next((a for a in args if a.endswith('.csv')), None)
        if csv_arg:
            csv_path = csv_arg
        else:
            csv_path = _find_latest_csv()

        if not csv_path or not os.path.exists(csv_path):
            print('Usage : python eval_report.py [fichier.csv]')
            print('        python eval_report.py --run [--no-camembert]')
            sys.exit(1)

        run_report(csv_path)
