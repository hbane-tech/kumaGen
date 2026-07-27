"""
eval/tree_metrics_upos_uas_las.py — UPOS / UAS / LAS entre l'arbre de
référence (build_reference_ud_tree, dérivé de tree_meta + schéma Aplonova &
Tyers) et l'arbre produit par le VRAI parseur UD_Bambara indépendant
(parse_bambara), sur les sorties Kuma-MT elles-mêmes.
================================================================================
Ne PAS confondre avec eval/bambara_parser_eval.py (qui évalue parse_bambara
contre le treebank gold UD_Bambara-CRB en général) : ici, les deux arbres
décrivent la MÊME phrase (une sortie Kuma), et on mesure si le parseur
indépendant confirme la structure que Kuma prétend avoir produite.

Alignement reference<->système : par forme de surface (insensible aux tons),
en ordre séquentiel (gère les répétitions comme le 'yé' du copule équatif).

Nécessite une RE-TRADUCTION (tree_meta complet, pas seulement les booléens
slot_S/V/O du CSV eval_final) — échantillonne donc un nombre limité de
phrases par groupe pour rester dans un temps raisonnable (moteur LLM lent).

Usage : python -m eval.tree_metrics_upos_uas_las --per-group 5
"""
import argparse
import csv
import io
import contextlib
import os
import sys
import random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.bambara_ud_reference import build_reference_ud_tree
from eval.bambara_udpipe import parse_bambara, dep_tree_string
from eval.evaluate import strip_bambara_tones

_COVERED_CLAUSE_TYPES = {
    'equative', 'identificatory', 'locative',
    'existential_absolute', 'existential_localized', 'presentative',
    'simple', 'statif',
}


def _norm(s):
    return strip_bambara_tones(s or '').lower().strip()


def align(ref_nodes, sys_nodes):
    """Aligne par forme de surface normalisée, séquentiellement (gère les
    répétitions). Retourne liste de (ref_node, sys_node) pour les paires
    résolues des deux côtés."""
    sys_by_form = defaultdict(list)
    for i, s in enumerate(sys_nodes):
        sys_by_form[_norm(s['form'])].append(i)

    used = set()
    pairs = []
    for r in ref_nodes:
        candidates = [i for i in sys_by_form.get(_norm(r['form']), []) if i not in used]
        if not candidates:
            continue
        i = candidates[0]
        used.add(i)
        pairs.append((r, sys_nodes[i]))
    return pairs


def score_pair(ref_nodes, sys_nodes):
    """UPOS/UAS/LAS entre ref (schéma Aplonova&Tyers via tree_meta) et
    sys (parse_bambara indépendant), sur les tokens alignables."""
    pairs = align(ref_nodes, sys_nodes)
    if not pairs:
        return None

    # id ref -> id sys, pour résoudre les têtes une fois les deux côtés alignés
    ref_id_to_sys_id = {}
    for r, s in pairs:
        ref_id_to_sys_id[r['id']] = s['id']

    n = len(pairs)
    n_upos = n_uas = n_las = 0
    for r, s in pairs:
        if r['upos'].upper() == s['upos'].upper():
            n_upos += 1
        # tête ref exprimée en id ref (0 = root) -> id sys attendu
        expected_sys_head = 0 if r['head'] == 0 else ref_id_to_sys_id.get(r['head'])
        uas_ok = expected_sys_head is not None and s['head'] == expected_sys_head
        if uas_ok:
            n_uas += 1
        if uas_ok and r['deprel'].split(':')[0].lower() == s['deprel'].split(':')[0].lower():
            n_las += 1

    return {'n': n, 'upos': n_upos / n, 'uas': n_uas / n, 'las': n_las / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='eval_final_20260711_105728.csv')
    ap.add_argument('--per-group', type=int, default=5,
                     help='Max sentences re-translated per covered clause_type group')
    ap.add_argument('--out', default='eval/kuma_tree_upos_uas_las.csv')
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding='utf-8')))
    by_group = defaultdict(list)
    for r in rows:
        if r['group'] in _COVERED_CLAUSE_TYPES:
            by_group[r['group']].append(r)

    random.seed(0)
    sample = []
    for g, items in by_group.items():
        random.shuffle(items)
        sample.extend(items[:args.per_group])
    print(f"Sampling {len(sample)} sentences across {len(by_group)} covered groups "
          f"(max {args.per_group}/group)")

    print("Loading TranslationEngine...")
    from pipeline.translation_engine import TranslationEngine
    from kg.neo4j_client import Neo4jClient
    db = Neo4jClient()
    eng = TranslationEngine(db)

    out_rows = []
    for i, r in enumerate(sample, 1):
        phrase = r['source']
        print(f"\r  [{i:3d}/{len(sample)}] {phrase[:50]:<50}", end='', flush=True)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                res = eng.translate(phrase)
            output = res.get('bambara', '').strip()
            tree_meta = res.get('tree', {})
        except Exception:
            continue

        ref_nodes = build_reference_ud_tree(tree_meta, output)
        if not ref_nodes:
            continue
        sys_nodes = parse_bambara(output)
        if not sys_nodes:
            continue

        s = score_pair(ref_nodes, sys_nodes)
        if s is None:
            continue

        out_rows.append({
            'group': r['group'], 'source': phrase,
            'reference_bm': r['reference'], 'kuma_bm': output,
            'ref_tree': dep_tree_string(ref_nodes),
            'sys_tree': dep_tree_string(sys_nodes),
            'n_aligned': s['n'],
            'upos': round(s['upos'] * 100, 1),
            'uas': round(s['uas'] * 100, 1),
            'las': round(s['las'] * 100, 1),
        })

    print()
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {args.out} — {len(out_rows)} scored sentences")


if __name__ == '__main__':
    main()
