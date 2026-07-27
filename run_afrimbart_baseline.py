"""
run_afrimbart_baseline.py — Runs masakhane/afrimbart_fr_bam_news over the full
365-sentence benchmark and writes a new CSV with an 'afrimbart_translate'
column, for use as a Bambara-specific baseline alongside Google/NLLB in
eval_report.py (reviewer request: narrow baseline comparison).

Usage:
    python run_afrimbart_baseline.py resultats_bambara_20260724_005832.csv
"""
import sys
import csv
import time

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'resultats_bambara_20260724_005832.csv'
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))

    print('Loading afrimbart_fr_bam_news...')
    tok = AutoTokenizer.from_pretrained('masakhane/afrimbart_fr_bam_news')
    model = AutoModelForSeq2SeqLM.from_pretrained('masakhane/afrimbart_fr_bam_news')
    model.eval()

    n = len(rows)
    t0 = time.time()
    for i, r in enumerate(rows):
        fr = (r.get('phrase_fr') or '').strip()
        if not fr:
            r['afrimbart_translate'] = ''
            continue
        inputs = tok(fr, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=64, num_beams=4)
        decoded = tok.batch_decode(out, skip_special_tokens=True)[0]
        r['afrimbart_translate'] = decoded
        if (i + 1) % 20 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f'  {i+1}/{n}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)')

    out_path = csv_path.replace('.csv', '_with_afrimbart.csv')
    fieldnames = list(rows[0].keys())
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f' Wrote {out_path}')


if __name__ == '__main__':
    main()
