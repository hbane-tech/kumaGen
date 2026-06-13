"""
baseline_eval.py — Comparaison Kuma-MT vs NLLB vs Google Translate
===================================================================
Lit le CSV produit par `test_phrases.py --run`, traduit chaque phrase FR
en bambara via NLLB et Google Translate, puis compare les trois systèmes
sur les métriques : chrF, chrF++, EXM, word-order S-TAM-V.

Génère :
  baseline_translations.json  — cache des traductions (évite de re-tourner NLLB)
  baseline_results.csv        — résultats complets phrase par phrase
  figures/fig_baseline_*.{png,pdf}

Usage :
  python baseline_eval.py                         # dernier resultats_bambara_*.csv
  python baseline_eval.py --csv resultats_X.csv   # CSV spécifique
  python baseline_eval.py --no-nllb               # Google + Kuma-MT seulement
  python baseline_eval.py --no-google             # NLLB + Kuma-MT seulement
  python baseline_eval.py --figures-only          # lit le cache, régénère les figures
"""

import argparse
import csv
import glob
import json
import os
import re
import time
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Style (cohérent avec eval_plots.py) ──────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.size': 11, 'font.family': 'DejaVu Sans',
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_KUMA   = '#0072B2'   # bleu  — Kuma-MT
C_NLLB   = '#E69F00'   # orange — NLLB
C_GOOGLE = '#009E73'   # vert  — Google Translate
C_GREY   = '#999999'

CACHE_FILE   = 'baseline_translations.json'
RESULTS_CSV  = 'baseline_results.csv'
FIGURES_DIR  = 'figures'

# Bambara TAM markers (pour évaluation word-order)
_TAM_RE = re.compile(
    r'\b(tùn yé|tùn ma|tùn bɛ|tùn tɛ|bɛ kà|tɛ kà|bɛ na|tɛ na'
    r'|yé|ma|bɛ|tɛ|dòn|kàna)\b'
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Chargement CSV test_phrases
# ═══════════════════════════════════════════════════════════════════════════════

def load_test_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            # Accepte les deux formats : resultats_bambara (statut=PASS/FAIL/ERROR)
            # et eval_*.csv (chrf, p1, etc.)
            rows.append({
                'id':        r.get('#', r.get('', '')),
                'categorie': r.get('categorie', r.get('category', '')),
                'phrase_fr': r.get('phrase_fr', r.get('source', '')),
                'reference': r.get('bambara_attendu', r.get('reference', '')),
                'kuma_mt':   r.get('bambara_obtenu', r.get('output', '')),
                'statut':    r.get('statut', ''),
            })
    # Filtrer les lignes sans phrase source
    rows = [r for r in rows if r['phrase_fr'].strip()]
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Traduction Google Translate
# ═══════════════════════════════════════════════════════════════════════════════

def translate_google(phrases: list[str], delay: float = 0.5) -> dict[str, str]:
    """Traduit FR→Bambara via Google Translate (deep_translator).
    Retourne {phrase_fr: traduction_bm}. Ajoute un délai entre les appels."""
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source='fr', target='bm')
    results = {}
    n = len(phrases)
    for i, phrase in enumerate(phrases):
        print(f"\r  Google [{i+1:3d}/{n}] {phrase[:50]:<50}", end='', flush=True)
        try:
            results[phrase] = translator.translate(phrase) or ''
        except Exception as e:
            print(f"\n  ⚠️  Google error on '{phrase[:30]}': {e}")
            results[phrase] = ''
        if delay > 0:
            time.sleep(delay)
    print()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Traduction NLLB
# ═══════════════════════════════════════════════════════════════════════════════

def translate_nllb(phrases: list[str],
                   model_name: str = 'facebook/nllb-200-distilled-600M',
                   batch_size: int = 8) -> dict[str, str]:
    """Traduit FR→Bambara via NLLB-200 (HuggingFace transformers).
    Langue source : fra_Latn  |  Langue cible : bam_Latn (Bambara).
    Retourne {phrase_fr: traduction_bm}."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    SRC_LANG = 'fra_Latn'
    TGT_LANG = 'bam_Latn'

    print(f"  Chargement NLLB ({model_name}) …")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    print(f"  NLLB prêt sur {device.upper()}")

    results = {}
    n       = len(phrases)
    for start in range(0, n, batch_size):
        batch  = phrases[start:start + batch_size]
        print(f"\r  NLLB [{start+len(batch):3d}/{n}] "
              f"{batch[0][:40]:<40}", end='', flush=True)
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256,
            src_lang=SRC_LANG,
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
                max_new_tokens=128,
                num_beams=4,
            )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for phrase, translation in zip(batch, decoded):
            results[phrase] = translation
    print()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Métriques
# ═══════════════════════════════════════════════════════════════════════════════

def _chrf(hyp: str, ref: str, word_order: int = 0) -> float:
    try:
        from sacrebleu.metrics import CHRF
        return CHRF(word_order=word_order).sentence_score(hyp, [ref]).score
    except Exception:
        return 0.0


def _word_order_ok(bambara: str) -> bool:
    tokens = bambara.strip().split()
    if not tokens:
        return False
    m = _TAM_RE.search(bambara)
    if not m:
        return True   # phrase nominale sans TAM → neutre
    tam_pos   = len(bambara[:m.start()].split())
    tam_width = len(m.group().split())
    return tam_pos > 0 and (tam_pos + tam_width) < len(tokens)


def _token_f1(hyp: str, ref: str) -> float:
    h = defaultdict(int)
    r = defaultdict(int)
    for w in hyp.lower().split(): h[w] += 1
    for w in ref.lower().split(): r[w] += 1
    common = sum(min(h[w], r[w]) for w in h)
    p   = common / len(hyp.split())  if hyp.split()  else 0
    rec = common / len(ref.split())  if ref.split()  else 0
    return 2 * p * rec / (p + rec) if (p + rec) else 0.0


def compute_metrics(hyp: str, ref: str) -> dict:
    return {
        'chrf':     round(_chrf(hyp, ref, word_order=0), 2),
        'chrfpp':   round(_chrf(hyp, ref, word_order=2), 2),
        'exm':      hyp.strip() == ref.strip(),
        'token_f1': round(_token_f1(hyp, ref), 3),
        'word_order_ok': _word_order_ok(hyp),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Comparaison globale
# ═══════════════════════════════════════════════════════════════════════════════

def compare(rows: list[dict]) -> dict:
    """Agrège les métriques par système."""
    systems = ['kuma_mt', 'nllb', 'google']
    agg = {s: defaultdict(list) for s in systems}

    for r in rows:
        ref = r['reference'].strip()
        for s in systems:
            hyp = r.get(s, '').strip()
            if not hyp or not ref:
                continue
            m = compute_metrics(hyp, ref)
            for k, v in m.items():
                agg[s][k].append(v)

    summary = {}
    for s in systems:
        d = agg[s]
        n = len(d['chrf'])
        if n == 0:
            continue
        summary[s] = {
            'n':           n,
            'chrf':        round(sum(d['chrf'])     / n, 2),
            'chrfpp':      round(sum(d['chrfpp'])   / n, 2),
            'exm_pct':     round(sum(d['exm'])      / n * 100, 1),
            'token_f1':    round(sum(d['token_f1']) / n, 3),
            'word_order':  round(sum(d['word_order_ok']) / n * 100, 1),
        }

    # Corpus-level chrF / chrF++ avec sacrebleu
    try:
        from sacrebleu.metrics import CHRF, BLEU
        refs = [r['reference'].strip() for r in rows if r['reference'].strip()]
        for s in systems:
            hyps = [r.get(s, '').strip() for r in rows if r['reference'].strip()]
            if not any(hyps):
                continue
            summary[s]['chrf_corpus']   = round(
                CHRF(word_order=0).corpus_score(hyps, [refs]).score, 2)
            summary[s]['chrfpp_corpus'] = round(
                CHRF(word_order=2).corpus_score(hyps, [refs]).score, 2)
            summary[s]['bleu_corpus']   = round(
                BLEU(tokenize='char').corpus_score(hyps, [refs]).score, 2)
    except Exception:
        pass

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Figures
# ═══════════════════════════════════════════════════════════════════════════════

_SYS_LABELS = {'kuma_mt': 'Kuma-MT', 'nllb': 'NLLB-200', 'google': 'Google'}
_SYS_COLORS = {'kuma_mt': C_KUMA, 'nllb': C_NLLB, 'google': C_GOOGLE}

def _save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGURES_DIR, f'{name}.{ext}'), bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {name}.png / .pdf")


def fig_global_comparison(summary: dict):
    """Bar chart : chrF / chrF++ / EXM / WO pour les 3 systèmes."""
    systems = [s for s in ('kuma_mt', 'nllb', 'google') if s in summary]
    metrics = [
        ('chrf',       'chrF (sentence)'),
        ('chrfpp',     'chrF++'),
        ('exm_pct',    'Exact Match %'),
        ('word_order', 'Word Order S-TAM-V %'),
    ]

    x      = np.arange(len(metrics))
    width  = 0.28
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, sys in enumerate(systems):
        vals = [summary[sys].get(m[0], 0) for m in metrics]
        offset = (i - len(systems)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=_SYS_LABELS[sys],
                      color=_SYS_COLORS[sys],
                      edgecolor='black', linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 1.2,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics], fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Score')
    ax.set_title('Comparaison Kuma-MT vs NLLB-200 vs Google Translate (FR→Bambara)')
    ax.legend(frameon=False)
    _save(fig, 'fig_baseline_global')


def fig_corpus_metrics(summary: dict):
    """Bar chart : chrF corpus / chrF++ corpus / BLEU corpus."""
    systems = [s for s in ('kuma_mt', 'nllb', 'google') if s in summary
               and 'chrf_corpus' in summary[s]]
    if not systems:
        return
    metrics = [
        ('chrf_corpus',   'chrF corpus'),
        ('chrfpp_corpus', 'chrF++ corpus'),
        ('bleu_corpus',   'BLEU corpus (char)'),
    ]
    x     = np.arange(len(metrics))
    width = 0.28
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, sys in enumerate(systems):
        vals   = [summary[sys].get(m[0], 0) for m in metrics]
        offset = (i - len(systems)/2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width,
                        label=_SYS_LABELS[sys], color=_SYS_COLORS[sys],
                        edgecolor='black', linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.8,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics], fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Score')
    ax.set_title('Métriques corpus FR→Bambara')
    ax.legend(frameon=False)
    _save(fig, 'fig_baseline_corpus')


def fig_per_category(rows: list[dict], mapping: dict):
    """chrF par clause_type pour les 3 systèmes (grouped horizontal bars)."""
    systems = [s for s in ('kuma_mt', 'nllb', 'google')
               if any(r.get(s) for r in rows)]

    groups = defaultdict(lambda: {s: [] for s in systems})
    for r in rows:
        ref = r['reference'].strip()
        if not ref:
            continue
        g = mapping.get(r['categorie'], r['categorie'])
        for s in systems:
            hyp = r.get(s, '').strip()
            if hyp:
                groups[g][s].append(_chrf(hyp, ref))

    items = sorted(groups.items(), key=lambda x: -len(x[1].get('kuma_mt', [])))
    names = [g for g, _ in items]

    y     = np.arange(len(names))
    h     = 0.8 / len(systems)
    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.5)))

    for i, sys in enumerate(systems):
        vals   = [np.mean(v[sys]) if v[sys] else 0 for _, v in items]
        offset = (i - len(systems)/2 + 0.5) * h
        ax.barh(y + offset, vals, height=h,
                label=_SYS_LABELS[sys], color=_SYS_COLORS[sys],
                edgecolor='black', linewidth=0.3, alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel('chrF moyen')
    ax.set_title('chrF par type de clause — comparaison 3 systèmes')
    ax.legend(frameon=False, loc='lower right')
    _save(fig, 'fig_baseline_per_category')


def fig_chrf_scatter(rows: list[dict]):
    """Scatter chrF Kuma-MT vs NLLB et Kuma-MT vs Google."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pairs = [('nllb', 'NLLB-200', C_NLLB), ('google', 'Google Translate', C_GOOGLE)]

    for ax, (sys, label, color) in zip(axes, pairs):
        pts = [(r.get('kuma_mt', ''), r.get(sys, ''), r['reference'])
               for r in rows if r.get('kuma_mt') and r.get(sys) and r['reference']]
        if not pts:
            ax.set_visible(False)
            continue
        x = [_chrf(h1, ref) for h1, _, ref in pts]
        y = [_chrf(h2, ref) for _, h2, ref in pts]
        ax.scatter(x, y, color=color, alpha=0.5, edgecolor='black',
                   linewidth=0.3, s=35)
        lim = [0, 105]
        ax.plot(lim, lim, 'k--', linewidth=0.8, alpha=0.5)   # diagonale
        ax.fill_between(lim, lim, 105, alpha=0.05, color=C_KUMA)   # zone Kuma > baseline
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.set_xlabel('chrF Kuma-MT')
        ax.set_ylabel(f'chrF {label}')
        ax.set_title(f'Kuma-MT vs {label}\n'
                     f'(Kuma meilleur dans zone bleue, n={len(pts)})')
    fig.suptitle('Comparaison phrase par phrase — chrF', fontsize=12)
    _save(fig, 'fig_baseline_scatter')


def fig_word_order_comparison(rows: list[dict]):
    """Stacked bar : conformité S-TAM-V par système."""
    systems = [s for s in ('kuma_mt', 'nllb', 'google')
               if any(r.get(s) for r in rows)]
    labels  = [_SYS_LABELS[s] for s in systems]
    wo_pct  = []
    for s in systems:
        hyps = [r.get(s, '').strip() for r in rows if r.get(s, '').strip()]
        ok   = sum(_word_order_ok(h) for h in hyps)
        wo_pct.append(round(ok / len(hyps) * 100, 1) if hyps else 0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors  = [_SYS_COLORS[s] for s in systems]
    bars    = ax.bar(labels, wo_pct, color=colors, edgecolor='black', linewidth=0.6)
    for b, v in zip(bars, wo_pct):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.set_ylabel('Phrases conformes (%)')
    ax.set_title('Conformité ordre des mots S-TAM-V (Bambara)\nKuma-MT vs NLLB vs Google')
    _save(fig, 'fig_baseline_word_order')


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Rapport texte
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(summary: dict):
    print(f"\n{'═'*68}")
    print(f"  BASELINE COMPARISON — FR → Bambara")
    print(f"{'═'*68}")
    print(f"  {'Métrique':<22} {'Kuma-MT':>10} {'NLLB-200':>10} {'Google':>10}")
    print(f"  {'-'*52}")
    metrics = [
        ('n',           'N phrases',         '{:.0f}'),
        ('chrf',        'chrF (sentence)',    '{:.2f}'),
        ('chrfpp',      'chrF++',             '{:.2f}'),
        ('chrf_corpus', 'chrF corpus',        '{:.2f}'),
        ('chrfpp_corpus','chrF++ corpus',     '{:.2f}'),
        ('bleu_corpus', 'BLEU corpus (chr)',  '{:.2f}'),
        ('exm_pct',     'Exact Match %',      '{:.1f}%'),
        ('token_f1',    'Token F1',           '{:.3f}'),
        ('word_order',  'Word Order S-TAM-V', '{:.1f}%'),
    ]
    for key, label, fmt in metrics:
        vals = []
        for s in ('kuma_mt', 'nllb', 'google'):
            v = summary.get(s, {}).get(key)
            vals.append(fmt.format(v) if v is not None else '—')
        print(f"  {label:<22} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")
    print(f"{'═'*68}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ═══════════════════════════════════════════════════════════════════════════════

def _latest_results_csv() -> str | None:
    candidates = sorted(glob.glob('resultats_bambara_*.csv'))
    return candidates[-1] if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',          help='CSV produit par test_phrases.py --run')
    parser.add_argument('--no-nllb',      action='store_true')
    parser.add_argument('--no-google',    action='store_true')
    parser.add_argument('--figures-only', action='store_true',
                        help='Lit baseline_translations.json, régénère les figures')
    parser.add_argument('--nllb-model',   default='facebook/nllb-200-distilled-600M')
    parser.add_argument('--google-delay', type=float, default=0.5,
                        help='Délai entre appels Google (secondes)')
    args = parser.parse_args()

    # ── Chargement du CSV ────────────────────────────────────────────────────
    csv_path = args.csv or _latest_results_csv()
    if not csv_path:
        print("❌ Aucun CSV trouvé. Lance d'abord : python test_phrases.py --run")
        return
    print(f"  CSV : {csv_path}")
    rows = load_test_csv(csv_path)
    print(f"  {len(rows)} phrases chargées")

    # ── Cache ────────────────────────────────────────────────────────────────
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding='utf-8') as f:
            cache = json.load(f)
        print(f"  Cache chargé : {CACHE_FILE}  "
              f"({len(cache.get('nllb', {}))} NLLB, "
              f"{len(cache.get('google', {}))} Google)")

    if not args.figures_only:
        phrases = [r['phrase_fr'] for r in rows]

        # Google Translate
        if not args.no_google:
            google_cache = cache.get('google', {})
            missing_g    = [p for p in phrases if p not in google_cache]
            if missing_g:
                print(f"\n  ▶ Google Translate ({len(missing_g)} phrases) …")
                new_g = translate_google(missing_g, delay=args.google_delay)
                google_cache.update(new_g)
                cache['google'] = google_cache
            else:
                print(f"  ✅ Google : {len(google_cache)} phrases en cache")
        else:
            cache.setdefault('google', {})

        # NLLB
        if not args.no_nllb:
            nllb_cache = cache.get('nllb', {})
            missing_n  = [p for p in phrases if p not in nllb_cache]
            if missing_n:
                print(f"\n  ▶ NLLB ({len(missing_n)} phrases, {args.nllb_model}) …")
                new_n = translate_nllb(missing_n, model_name=args.nllb_model)
                nllb_cache.update(new_n)
                cache['nllb'] = nllb_cache
            else:
                print(f"  ✅ NLLB : {len(nllb_cache)} phrases en cache")
        else:
            cache.setdefault('nllb', {})

        # Sauvegarde cache
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        print(f"\n  Cache sauvegardé : {CACHE_FILE}")

    # ── Peupler les lignes ───────────────────────────────────────────────────
    for r in rows:
        p = r['phrase_fr']
        r['nllb']   = cache.get('nllb', {}).get(p, '')
        r['google'] = cache.get('google', {}).get(p, '')

    # ── Métriques ────────────────────────────────────────────────────────────
    summary = compare(rows)
    print_report(summary)

    # ── Export CSV résultats ─────────────────────────────────────────────────
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'id', 'categorie', 'phrase_fr', 'reference',
            'kuma_mt', 'nllb', 'google',
            'chrf_kuma', 'chrf_nllb', 'chrf_google',
            'wo_kuma', 'wo_nllb', 'wo_google',
        ])
        writer.writeheader()
        for r in rows:
            ref = r['reference'].strip()
            writer.writerow({
                'id':         r['id'],
                'categorie':  r['categorie'],
                'phrase_fr':  r['phrase_fr'],
                'reference':  ref,
                'kuma_mt':    r.get('kuma_mt', ''),
                'nllb':       r.get('nllb', ''),
                'google':     r.get('google', ''),
                'chrf_kuma':  round(_chrf(r.get('kuma_mt', ''), ref), 2),
                'chrf_nllb':  round(_chrf(r.get('nllb', ''), ref), 2),
                'chrf_google':round(_chrf(r.get('google', ''), ref), 2),
                'wo_kuma':    _word_order_ok(r.get('kuma_mt', '')),
                'wo_nllb':    _word_order_ok(r.get('nllb', '')),
                'wo_google':  _word_order_ok(r.get('google', '')),
            })
    print(f"  Résultats : {RESULTS_CSV}")

    # ── Figures ──────────────────────────────────────────────────────────────
    print(f"\n  Génération des figures dans {FIGURES_DIR}/ …")
    try:
        from evaluate import CATEGORY_TO_CLAUSE as mapping
    except Exception:
        mapping = {}

    fig_global_comparison(summary)
    fig_corpus_metrics(summary)
    fig_per_category(rows, mapping)
    fig_chrf_scatter(rows)
    fig_word_order_comparison(rows)

    print(f"\n  ✅ Figures dans {FIGURES_DIR}/  (PNG 300dpi + PDF vectoriel)")


if __name__ == '__main__':
    main()
