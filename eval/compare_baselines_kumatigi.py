"""
compare_baselines_kumatigi.py — Kuma vs NLLB-200 vs Google Translate sur le
même échantillon kumatigi (HuggingFace Tysby101/kumatigi), même métriques que
evaluate.py (chrF, chrF++, BLEU-char), pour une comparaison directe avec la
Table 5 du papier Kumatigi (Cissé, 2026).

NLLB-200-distilled-600M : modèle EXACT utilisé par le papier comme baseline
zero-shot (pas de fine-tuning) — cf. facebook/nllb-200-distilled-600M déjà
utilisé ailleurs dans ce repo pour la back-traduction (test_phrases.py).
Google Translate : via deep_translator.GoogleTranslator (confirmé supporter
'bm' dans ce repo, contrairement à googletrans qui ne le supporte pas).

Usage:
  python -m eval.compare_baselines_kumatigi --split test --n 500 --seed 0
  python -m eval.compare_baselines_kumatigi --split test --n 500 --seed 0 --kuma-csv eval_20260727_XXXXXX.csv
"""

import argparse
import csv
import json
import time
import datetime

from eval.evaluate_kumatigi import load_cases
from eval.evaluate import chrf_score, chrfpp_score, bleu_char_score, corpus_metrics


# ── NLLB-200-distilled-600M (zero-shot, pas de fine-tuning — comme le papier) ──

_nllb_model = None
_nllb_tokenizer = None


def _load_nllb():
    global _nllb_model, _nllb_tokenizer
    if _nllb_model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        name = 'facebook/nllb-200-distilled-600M'
        print(f"  [NLLB] Chargement de {name} …")
        _nllb_tokenizer = AutoTokenizer.from_pretrained(name, src_lang='fra_Latn')
        _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(name)
        _nllb_model.eval()
        print("  [NLLB] Modèle chargé.")
    return _nllb_model, _nllb_tokenizer


def translate_nllb(text: str) -> str:
    if not text or not text.strip():
        return ''
    try:
        model, tokenizer = _load_nllb()
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        gen = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids('bam_Latn'),
            max_new_tokens=256,
        )
        return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    except Exception as e:
        return f'[ERR:NLLB] {e}'


# ── Google Translate (deep_translator, non-officiel) ──────────────────────────

def translate_google(text: str, retries: int = 2, delay: float = 1.0) -> str:
    if not text or not text.strip():
        return ''
    from deep_translator import GoogleTranslator
    for attempt in range(retries + 1):
        try:
            out = GoogleTranslator(source='fr', target='bm').translate(text)
            return out or ''
        except Exception as e:
            if attempt < retries:
                time.sleep(delay)
                continue
            return f'[ERR:Google] {e}'


# ── Main ───────────────────────────────────────────────────────────────────────

def run_comparison(split: str, n: int, seed: int, min_quality=None,
                    skip_nllb=False, skip_google=False, google_delay=1.0):
    cases = load_cases(split, n, seed, min_quality)
    total = len(cases)
    print(f"kumatigi/{split} : {total} phrases échantillonnées (seed={seed})")

    results = []
    for i, (phrase, expected, category) in enumerate(cases, 1):
        print(f"\r  [{i:4d}/{total}] {phrase[:50]:<50}", end='', flush=True)

        nllb_out = translate_nllb(phrase) if not skip_nllb else ''
        google_out = translate_google(phrase, delay=google_delay) if not skip_google else ''
        if not skip_google:
            time.sleep(google_delay)  # politeness delay, avoid rate-limit/block

        row = {
            'source': phrase,
            'reference': expected,
            'category': category,
            'nllb_output': nllb_out,
            'nllb_chrf': chrf_score(nllb_out, expected) if nllb_out else 0.0,
            'nllb_chrfpp': chrfpp_score(nllb_out, expected) if nllb_out else 0.0,
            'nllb_bleu': bleu_char_score(nllb_out, expected) if nllb_out else 0.0,
            'google_output': google_out,
            'google_chrf': chrf_score(google_out, expected) if google_out else 0.0,
            'google_chrfpp': chrfpp_score(google_out, expected) if google_out else 0.0,
            'google_bleu': bleu_char_score(google_out, expected) if google_out else 0.0,
        }
        results.append(row)

    print()

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f'baselines_kumatigi_{ts}.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV exporté : {csv_path}")

    refs = [r['reference'] for r in results]
    nllb_hyps = [r['nllb_output'] for r in results]
    google_hyps = [r['google_output'] for r in results]

    summary = {
        'timestamp': ts,
        'n_phrases': total,
        'nllb': {
            **corpus_metrics(nllb_hyps, refs),
            'chrf_sentence_avg': round(sum(r['nllb_chrf'] for r in results) / total, 2),
        },
        'google': {
            **corpus_metrics(google_hyps, refs),
            'chrf_sentence_avg': round(sum(r['google_chrf'] for r in results) / total, 2),
        },
    }
    json_path = f'baselines_kumatigi_{ts}_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  JSON résumé : {json_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return results, summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Kuma vs NLLB vs Google sur kumatigi')
    ap.add_argument('--split', default='test', choices=['train', 'validation', 'test'])
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-quality', type=int, default=None)
    ap.add_argument('--skip-nllb', action='store_true')
    ap.add_argument('--skip-google', action='store_true')
    ap.add_argument('--google-delay', type=float, default=1.0,
                     help='Délai (s) entre appels Google Translate, anti rate-limit')
    args = ap.parse_args()

    run_comparison(
        split=args.split, n=args.n, seed=args.seed, min_quality=args.min_quality,
        skip_nllb=args.skip_nllb, skip_google=args.skip_google,
        google_delay=args.google_delay,
    )
