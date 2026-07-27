"""
eval/compute_baseline_bertscore.py — Calcule réellement BERTScore P/R/F1 pour
Google Translate et NLLB-200, par back-traduction mot-à-mot (bambara -> français)
puis comparaison contextuelle à la phrase source (test_phrases.py::backtrans_bertscore_prf).

Remplace le "—" de Table 1 par un vrai chiffre, plutôt que d'invoquer un
argument de coût pour ne pas le mesurer.

Usage : python -m eval.compute_baseline_bertscore
"""
import csv
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_phrases as tp

CSV_PATH = 'resultats_bambara_20260720_074041.csv'
OUT_PATH = 'eval/baseline_bertscore.csv'


def main():
    rows = list(csv.DictReader(open(CSV_PATH, encoding='utf-8')))
    print(f"Loaded {len(rows)} rows")

    from pipeline.spacy_parser import SpacyParser
    from kg.neo4j_client import Neo4jClient
    db = Neo4jClient()
    tagger = SpacyParser(db)

    out_rows = []
    for i, r in enumerate(rows, 1):
        phrase = r['phrase_fr']
        goog_bm = r['google_translate']
        nllb_bm = r['nllb_translate']
        print(f"\r[{i:3d}/{len(rows)}] {phrase[:45]:<45}", end='', flush=True)

        pg, rg, f1g = tp.backtrans_bertscore_prf(
            phrase, goog_bm, tp.backtranslate_google_bm_to_fr, tagger) if goog_bm else (None, None, None)
        pn, rn, f1n = tp.backtrans_bertscore_prf(
            phrase, nllb_bm, tp.backtranslate_nllb_bm_to_fr, tagger) if nllb_bm else (None, None, None)

        out_rows.append({
            'idx': r['#'], 'source': phrase, 'categorie': r.get('categorie', ''),
            'p_google': pg, 'r_google': rg, 'f1_google': f1g,
            'p_nllb': pn, 'r_nllb': rn, 'f1_nllb': f1n,
        })

        if i % 20 == 0:
            _dump(out_rows)

    print()
    _dump(out_rows)
    _summary(out_rows)


def _dump(rows):
    with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _summary(rows):
    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) * 100 if vals else None

    print(f"Google  P/R/F1 : {avg('p_google'):.1f} / {avg('r_google'):.1f} / {avg('f1_google'):.1f}")
    print(f"NLLB    P/R/F1 : {avg('p_nllb'):.1f} / {avg('r_nllb'):.1f} / {avg('f1_nllb'):.1f}")


if __name__ == '__main__':
    main()
