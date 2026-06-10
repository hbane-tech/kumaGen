"""
eval/parse_trans_corr.py — Corrélation parsing ↔ traduction
============================================================
Teste l'hypothèse de robustesse de l'architecture hybride :
  « les erreurs de parsing ne se propagent pas linéairement à la sortie ».

Sur les phrases UD French GSD, on croise pour chaque phrase :
  X = LAS local      (vrai, calculé contre le gold UD)
  Y = proxies de qualité de traduction SANS référence bambara :
       - coverage : 1 − (placeholders [..] / mots)   → complétude lexicale
       - tree_ok  : clause_type non-fallback ET slots S+V remplis

Si la corrélation X↔Y est faible, le parsing n'est pas le goulot : la couche
règles+LLM absorbe les erreurs de dépendance. Métriques : Pearson, Spearman, τ.

⚠️ coverage/tree_ok sont des PROXIES (complétude/structure), pas la correction
sémantique — déclaré explicitement. Borne supérieure de propagation d'erreur.

Usage :
  python -m eval.parse_trans_corr --limit 100
"""

import os
import re
import sys
import argparse

from eval.parser_eval import read_conllu, align, norm_deprel, UD_TEST


def las_local(sys_toks, gold, units):
    """LAS d'une phrase : % de tokens (mono-token alignés) avec head+dep corrects."""
    aligned = align(sys_toks, units)
    gold_by_id = {g['id']: g for g in gold}
    sysorig2goldid = {sys_toks[si].get('orig_index'): u['ids'][0]
                      for si, u in aligned if len(u['ids']) == 1}
    n = ok = 0
    for si, u in aligned:
        if len(u['ids']) != 1:
            continue
        st = sys_toks[si]
        gt = gold_by_id[u['ids'][0]]
        n += 1
        shi, so = st.get('head_index', -1), st.get('orig_index', -1)
        shg = 0 if (shi == so or shi < 0) else sysorig2goldid.get(shi, -999)
        if shg == gt['head'] and norm_deprel(st.get('dep')) == norm_deprel(gt['deprel']):
            ok += 1
    return (ok / n) if n else None, n


def translation_quality(bambara: str, tree_meta: dict):
    """Proxies sans référence : coverage lexicale + bonne formation du tree."""
    words = bambara.split()
    n_words = len(words)
    n_placeholder = len(re.findall(r'\[[^\]]+\]', bambara))
    coverage = 1 - n_placeholder / n_words if n_words else 0

    _fallback_clauses = {'', 'simple'}   # 'simple' = défaut générique
    clause = tree_meta.get('clause_type', '')
    tree_ok = bool(clause and clause not in _fallback_clauses
                   and tree_meta.get('S') and tree_meta.get('V'))
    return coverage, tree_ok


# ── Corrélations ────────────────────────────────────────────────────────────────
def _pearson(x, y):
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x)/n, sum(y)/n
    num = sum((a-mx)*(b-my) for a, b in zip(x, y))
    dx = sum((a-mx)**2 for a in x) ** 0.5
    dy = sum((b-my)**2 for b in y) ** 0.5
    return num / (dx*dy) if dx and dy else 0.0


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0]*len(v)
    for rank, i in enumerate(order):
        r[i] = rank
    return r


def _spearman(x, y):
    return _pearson(_rank(x), _rank(y))


def _kendall(x, y):
    n = len(x); c = d = 0
    for i in range(n):
        for j in range(i+1, n):
            s = (x[i]-x[j])*(y[i]-y[j])
            if s > 0: c += 1
            elif s < 0: d += 1
    tot = n*(n-1)/2
    return (c-d)/tot if tot else 0.0


# ── Main ─────────────────────────────────────────────────────────────────────────
def run(limit=100, export=True):
    import warnings, io, contextlib
    warnings.filterwarnings('ignore')

    sents = read_conllu(UD_TEST)
    sents = [s for s in sents if s[0]][:limit]
    print(f"Corrélation parsing↔traduction sur {len(sents)} phrases GSD\n")

    from pipeline.translation_engine import TranslationEngine
    from kg.neo4j_client import Neo4jClient
    db  = Neo4jClient()
    eng = TranslationEngine(db)

    rows = []
    for i, (text, gold, units) in enumerate(sents, 1):
        print(f"\r  [{i:3d}/{len(sents)}] {text[:48]:<48}", end='', flush=True)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                res = eng.translate(text)
            sys_toks = res.get('tokens', [])
            bambara  = res.get('bambara', '')
            tree     = res.get('tree', {})
        except Exception:
            continue

        las, n_tok = las_local(sys_toks, gold, units)
        if las is None or n_tok < 3:
            continue
        cov, tree_ok = translation_quality(bambara, tree)
        rows.append({'text': text, 'las': las, 'coverage': cov,
                     'tree_ok': tree_ok, 'n_tok': n_tok})

    print()
    if not rows:
        print("Aucune donnée.")
        return

    xs  = [r['las'] for r in rows]
    cov = [r['coverage'] for r in rows]
    trk = [1.0 if r['tree_ok'] else 0.0 for r in rows]

    print(f"\n{'═'*60}")
    print(f"  CORRÉLATION PARSING ↔ TRADUCTION  (n={len(rows)})")
    print(f"{'═'*60}")
    print(f"  LAS local moyen        : {sum(xs)/len(xs)*100:5.1f}%")
    print(f"  Coverage moyenne       : {sum(cov)/len(cov)*100:5.1f}%")
    print(f"  Tree bien formé        : {sum(trk)/len(trk)*100:5.1f}%")
    print(f"\n  ── LAS ↔ Coverage lexicale ──")
    print(f"  Pearson  r : {_pearson(xs, cov):+.3f}")
    print(f"  Spearman ρ : {_spearman(xs, cov):+.3f}")
    print(f"  Kendall  τ : {_kendall(xs, cov):+.3f}")
    print(f"\n  ── LAS ↔ Tree bien formé ──")
    print(f"  Pearson  r : {_pearson(xs, trk):+.3f}")
    print(f"  Spearman ρ : {_spearman(xs, trk):+.3f}")

    # Interprétation auto
    r = _pearson(xs, cov)
    verdict = ('FAIBLE — parsing peu prédictif de la sortie (robustesse)'
               if abs(r) < 0.3 else
               'MODÉRÉE' if abs(r) < 0.6 else 'FORTE — parsing prédit la sortie')
    print(f"\n  ➜ Corrélation LAS↔coverage : {verdict}")
    print(f"{'═'*60}\n")

    if export:
        import json, datetime, csv
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'parse_trans_corr_{ts}.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['text', 'las', 'coverage', 'tree_ok', 'n_tok'])
            w.writeheader(); w.writerows(rows)
        summary = {
            'n': len(rows),
            'las_mean': round(sum(xs)/len(xs)*100, 2),
            'coverage_mean': round(sum(cov)/len(cov)*100, 2),
            'tree_ok_rate': round(sum(trk)/len(trk)*100, 2),
            'pearson_las_cov':  round(_pearson(xs, cov), 3),
            'spearman_las_cov': round(_spearman(xs, cov), 3),
            'kendall_las_cov':  round(_kendall(xs, cov), 3),
            'pearson_las_tree': round(_pearson(xs, trk), 3),
        }
        with open(f'parse_trans_corr_{ts}.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  Exporté : parse_trans_corr_{ts}.csv / .json")

    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Corrélation parsing↔traduction')
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--no-export', action='store_true')
    args = p.parse_args()
    run(limit=args.limit, export=not args.no_export)
