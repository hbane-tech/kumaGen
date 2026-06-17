#!/usr/bin/env python3
"""
Quick word order analysis using existing CSV data.
No pipeline needed - just analyzes reference vs output word order.
"""

import csv
from pathlib import Path
from typing import List, Optional
from scipy.stats import kendalltau

# Bambara word order markers
TAM_MARKERS = {'bɛ', 'yé', 'ma', 'tɛ', 'tùn', 'kà', 'ka', 'na', 'ra', 'tún'}
PRONOUNS = {'n', 'i', 'a', 'o', 'u', 'anw', 'aw', 'ùw', 'inw', 'iw', 'a', 'e', 'á'}
PARTICLES = {'wà', 'dun', 'foyi', 'ni', 'le', 'te', 'den', 'min', 'don', 'dòn', 'ko', 'ní', 'mána', 'la', 'kɔnɔ'}

def extract_pos(text: str) -> List[str]:
    """Simple POS extraction for Bambara text."""
    if not text:
        return []
    
    # Clean and tokenize
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
        elif word_clean in ('ka', 'o', 'ba'):  # Common copulas
            pos_list.append('AUX')
        else:
            pos_list.append('NOUN')
    
    return pos_list

def kendall_tau_reordering(pos1: List[str], pos2: List[str]) -> Optional[float]:
    """Calculate Kendall tau between two POS sequences."""
    if len(pos1) < 2 or len(pos2) < 2 or len(pos1) != len(pos2):
        return None
    
    try:
        ranks1 = list(range(len(pos1)))
        ranks2 = list(range(len(pos2)))
        
        # Create position mapping for each unique POS
        unique_pos = {}
        idx = 0
        for p in pos1 + pos2:
            if p not in unique_pos:
                unique_pos[p] = idx
                idx += 1
        
        # Map to indices
        indices1 = [unique_pos[p] for p in pos1]
        indices2 = [unique_pos[p] for p in pos2]
        
        tau, _ = kendalltau(indices1, indices2)
        return tau
    except:
        return None

# Find latest CSV
csv_files = list(Path('.').glob('resultats_bambara_*.csv'))
if not csv_files:
    print("No CSV found. Checking for old file...")
    csv_path = Path('resultats_bambara_20260614_031740.csv')
    if not csv_path.exists():
        print("❌ No test results found")
        exit(1)
else:
    csv_path = sorted(csv_files)[-1]

print(f"\n📊 WORD ORDER ANALYSIS (Reference vs Output)")
print(f"{'='*70}")
print(f"Using: {csv_path.name}\n")

results = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        results.append(row)

# Analyze first 20 results
reordering_data = []
for i, result in enumerate(results[:20]):
    ref = result.get('bambara_attendu', '')
    actual = result.get('bambara_obtenu', '')
    category = result.get('categorie', '')
    phrase_fr = result.get('phrase_fr', '')[:40]
    status = result.get('statut', '')
    
    if not ref or not actual:
        continue
    
    ref_pos = extract_pos(ref)
    actual_pos = extract_pos(actual)
    
    tau = kendall_tau_reordering(ref_pos, actual_pos)
    
    reordering_data.append({
        'phrase': phrase_fr,
        'category': category,
        'status': status,
        'ref_pos': ' '.join(ref_pos) if ref_pos else 'N/A',
        'actual_pos': ' '.join(actual_pos) if actual_pos else 'N/A',
        'tau': tau,
        'ref': ref,
        'actual': actual
    })
    
    if tau is not None:
        status_icon = '✅' if status == 'PASS' else '❌'
        tau_quality = "🟢" if tau > 0.7 else "🟡" if tau > 0.3 else "🔴"
        print(f"[{i+1}] {status_icon} {phrase_fr}...")
        print(f"    {category}")
        print(f"    Ref:    {' '.join(ref_pos)}")
        print(f"    Actual: {' '.join(actual_pos)}")
        print(f"    Kendall τ = {tau:+.3f} {tau_quality}")
        print(f"    Expected: {ref}")
        print(f"    Got:      {actual}")
        print()

# Summary
print(f"\n{'='*70}")
print(f"📈 SUMMARY")
print(f"{'='*70}\n")

taus = [d['tau'] for d in reordering_data if d['tau'] is not None]
if taus:
    avg = sum(taus) / len(taus)
    print(f"Average Kendall τ: {avg:+.3f}")
    print(f"Range: {min(taus):+.3f} to {max(taus):+.3f}")
    
    passes = [d['tau'] for d in reordering_data if d['status'] == 'PASS' and d['tau'] is not None]
    fails = [d['tau'] for d in reordering_data if d['status'] != 'PASS' and d['tau'] is not None]
    
    if passes:
        print(f"\nPASS cases: avg τ = {sum(passes)/len(passes):+.3f} ({len(passes)} cases)")
    if fails:
        print(f"FAIL cases: avg τ = {sum(fails)/len(fails):+.3f} ({len(fails)} cases)")

print("\n💡 Interpretation:")
print("  τ > +0.7  = Excellent word order (matches reference)")
print("  τ 0 to +0.7 = Moderate reordering (acceptable)")
print("  τ < 0  = Reversed order (problem)")
