#!/usr/bin/env python3
"""
Word order analysis using Kendall tau.
Compares reference (correct) Bambara vs system output.
"""

import csv
from pathlib import Path
from typing import List, Optional
from scipy.stats import kendalltau

# Bambara word order markers and classifiers
TAM_MARKERS = {'bɛ', 'yé', 'ma', 'tɛ', 'tùn', 'kà', 'ka', 'na', 'ra', 'tún'}
PRONOUNS = {'n', 'i', 'a', 'o', 'u', 'anw', 'aw', 'ùw', 'inw', 'iw'}
PARTICLES = {'wà', 'dun', 'foyi', 'ni', 'le', 'te', 'den', 'min', 'don', 'dòn', 'ko', 'ní', 'mána', 'la', 'kɔnɔ'}

def extract_pos(text: str) -> List[str]:
    """Extract POS sequence from Bambara text."""
    if not text:
        return []

    text = text.replace('?', ' ?').replace(',', ' ,')
    words = text.split()

    pos_list = []
    for word in words:
        word_clean = word.strip('.,?;:«»')
        if not word_clean:
            continue

        if word_clean in TAM_MARKERS:
            pos_list.append('AUX')
        elif word_clean in PRONOUNS:
            pos_list.append('PRON')
        elif word_clean in PARTICLES:
            pos_list.append('PART')
        elif word_clean[0].isupper():
            pos_list.append('PROPN')
        elif word_clean.endswith(('li', 'ra', 'na')):
            pos_list.append('VERB')
        else:
            pos_list.append('NOUN')

    return pos_list

def kendall_tau_reordering(pos1: List[str], pos2: List[str]) -> Optional[float]:
    """Calculate Kendall tau between two POS sequences."""
    if len(pos1) < 2 or len(pos2) < 2 or len(pos1) != len(pos2):
        return None

    try:
        tau, _ = kendalltau(list(range(len(pos1))), list(range(len(pos2))))
        return tau
    except:
        return None

# Find latest eval CSV
csv_files = sorted(Path('.').glob('eval_*.csv'))
if not csv_files:
    print("❌ No eval CSV found")
    exit(1)

csv_path = csv_files[-1]
print(f"\n📊 WORD ORDER ANALYSIS (Kendall Tau)")
print(f"{'='*75}")
print(f"File: {csv_path.name}\n")

results = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        results.append(row)

reordering_data = []
for i, result in enumerate(results[:25]):
    ref = result.get('reference', '')
    actual = result.get('output', '')
    category = result.get('category', '')
    source = result.get('source', '')[:45]

    if not ref or not actual:
        continue

    ref_pos = extract_pos(ref)
    actual_pos = extract_pos(actual)

    tau = kendall_tau_reordering(ref_pos, actual_pos)

    if tau is not None:
        match = '✅' if ref == actual else '❌'
        tau_quality = "🟢" if tau > 0.7 else "🟡" if tau > 0.3 else "🔴"

        print(f"[{i+1:2d}] {match} {source}...")
        print(f"     Category: {category}")
        print(f"     Ref POS:    {' '.join(ref_pos)}")
        print(f"     Actual POS: {' '.join(actual_pos)}")
        print(f"     Kendall τ = {tau:+.3f} {tau_quality}")
        print(f"     Reference: {ref}")
        print(f"     Output:    {actual}")
        print()

        reordering_data.append({'tau': tau, 'match': ref == actual, 'category': category})

# Summary
if reordering_data:
    print(f"\n{'='*75}")
    print(f"📈 SUMMARY")
    print(f"{'='*75}\n")

    taus = [d['tau'] for d in reordering_data]
    matches = [d['match'] for d in reordering_data]

    avg = sum(taus) / len(taus)
    correct = sum(matches)

    print(f"Kendall τ (average):    {avg:+.3f}")
    print(f"Range:                  {min(taus):+.3f} to {max(taus):+.3f}")
    print(f"Exact matches:          {correct}/{len(matches)} ({100*correct/len(matches):.0f}%)")
    print()
    print(f"Interpretation:")
    print(f"  τ > +0.7  → Excellent (word order matches reference)")
    print(f"  τ 0 to +0.7 → Good (acceptable reordering)")
    print(f"  τ < 0  → Problem (reversed order)")
