#!/usr/bin/env python3
"""
Word Order Analysis using Kendall Tau
Compares syntactic reordering between:
1. French input → Bambara output (reordering analysis)
2. Reference Bambara → Actual Bambara (accuracy against expected word order)
3. System outputs vs each other (consistency analysis)

Kendall Tau: rank correlation coefficient measuring how much relative ordering changed
- τ = +1: perfect agreement (same order)
- τ = 0: no correlation
- τ = -1: perfect reversal
"""

import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.stats import kendalltau
from collections import defaultdict


def extract_pos_sequence(pos_tree_str: str) -> List[str]:
    """Parse POS tree string into list of POS tags."""
    if not pos_tree_str or pos_tree_str == '':
        return []
    return pos_tree_str.strip().split()


def create_position_indices(sequence: List[str]) -> List[int]:
    """
    Create position indices for Kendall tau calculation.
    Each unique POS gets a rank based on first appearance.

    Example:
        ['PRON', 'VERB', 'DET', 'NOUN'] → [0, 1, 2, 3]
    """
    pos_to_rank = {}
    rank = 0
    indices = []
    for pos in sequence:
        if pos not in pos_to_rank:
            pos_to_rank[pos] = rank
            rank += 1
        indices.append(pos_to_rank[pos])
    return indices


def kendall_tau_reordering(french_pos: List[str], output_pos: List[str]) -> Optional[float]:
    """
    Calculate Kendall tau between French word order and output word order.

    Measures: How much did the system reorder tokens?

    Returns:
        τ in range [-1, +1]:
        +1 = perfect agreement (no reordering)
        0 = random/no correlation
        -1 = perfect reversal
        None = sequences incomparable (different lengths, <2 items)
    """
    if len(french_pos) < 2 or len(output_pos) < 2:
        return None

    if len(french_pos) != len(output_pos):
        return None

    # Create mapping: for each POS in output, find its position in French
    # This only works if same POS tokens exist
    try:
        french_indices = create_position_indices(french_pos)
        output_indices = create_position_indices(output_pos)

        tau, p_value = kendalltau(french_indices, output_indices)
        return tau
    except Exception as e:
        return None


def extract_word_order_rules() -> Dict[str, Dict]:
    """
    Extract Bambara word order rules from source code analysis.
    These are the EXPECTED word orders based on clause types.

    Returns dict mapping clause_type → {pattern, description, sov_order}
    """
    rules = {
        'simple': {
            'pattern': 'S TAM [O] V [V_ACT] [obliques]',
            'order': ['S', 'TAM', 'O', 'V', 'V_ACT', 'OBL'],
            'description': 'Basic clause: Subject-TAM-Object-Verb (SOV-like with TAM)',
            'sov_type': 'TAM-between-S-O'
        },
        'conditional': {
            'pattern': 'ní S TAM [O] V [obliques]',
            'order': ['ní', 'S', 'TAM', 'O', 'V', 'OBL'],
            'description': 'Condition marker (ní) + Subject-TAM-Object-Verb',
            'sov_type': 'marked-SOV'
        },
        'temporal': {
            'pattern': 'tuma min S TAM [O] V [obliques]',
            'order': ['tuma_min', 'S', 'TAM', 'O', 'V', 'OBL'],
            'description': 'Temporal marker + Subject-TAM-Object-Verb',
            'sov_type': 'marked-SOV'
        },
        'restrictive': {
            'pattern': 'S TAM foyi yé ni ATTR tɛ',
            'order': ['S', 'TAM', 'foyi', 'yé', 'ni', 'ATTR', 'tɛ'],
            'description': 'Rule 6: ne...que restrictive → "only one who..." structure',
            'sov_type': 'specialized'
        },
        'quest_ce_que': {
            'pattern': 'mún S TAM V V_ACT [obliques]',
            'order': ['mún', 'S', 'TAM', 'V', 'V_ACT', 'OBL'],
            'description': 'Rule 7: Qu\'est-ce que → mún + modal serial',
            'sov_type': 'VSO-like'
        },
        'refl_absolute': {
            'pattern': 'S TAM S [yɛrɛ] V [obliques]',
            'order': ['S', 'TAM', 'S', 'yɛrɛ', 'V', 'OBL'],
            'description': 'Reflexive: S TAM S [yɛrɛ] V',
            'sov_type': 'SOV-reflexive'
        },
        'verb_serial': {
            'pattern': 'S TAM [bɔra ka] V_ACT [obliques]',
            'order': ['S', 'TAM', 'bɔra', 'ka', 'V_ACT', 'OBL'],
            'description': 'Serial verb: S TAM bɔra ka V_ACTION (recent past)',
            'sov_type': 'TAM-serial'
        },
        'interrogative': {
            'pattern': 'S TAM [O] V wà ?',
            'order': ['S', 'TAM', 'O', 'V', 'wà', '?'],
            'description': 'Yes-no question: S TAM [O] V wà ?',
            'sov_type': 'SOV-question'
        },
        'passive': {
            'pattern': 'S TAM V+ra [agent]',
            'order': ['S', 'TAM', 'V+ra', 'AGENT', 'OBL'],
            'description': 'Passive: S TAM V+ra (agent optional)',
            'sov_type': 'SOV-passive'
        },
        'imperative': {
            'pattern': 'V [O] [obliques]',
            'order': ['V', 'O', 'OBL'],
            'description': 'Command: Verb-Object (no subject, no TAM)',
            'sov_type': 'VO-imperative'
        }
    }
    return rules


def analyze_word_order_csv(csv_path: str):
    """Analyze word order reordering in CSV results using Kendall tau."""

    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        return

    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    if not results:
        print("❌ No results in CSV")
        return

    rules = extract_word_order_rules()

    print(f"\n{'='*80}")
    print(f"🔤 WORD ORDER ANALYSIS (Kendall Tau)")
    print(f"{'='*80}\n")
    print("Kendall τ ranges from -1 (complete reversal) to +1 (perfect agreement)")
    print("Interpretation:")
    print("  τ > 0.7  : High ordering consistency (good word order)")
    print("  τ 0.3-0.7: Moderate reordering (acceptable)")
    print("  τ < 0.3  : Major reordering (potential issues)")
    print("  τ < 0    : Reverse ordering\n")

    # Analysis by category
    by_category = defaultdict(list)
    reordering_stats = []

    print(f"{'='*80}")
    print(f"Individual Comparisons (sample)")
    print(f"{'='*80}\n")

    # Handle both old format (french_pos_tree) and new format (reference/output)
    is_eval_format = 'reference' in (results[0] if results else {})

    for i, result in enumerate(results[:15]):  # Show first 15
        category = result.get('categorie', result.get('category', 'unknown'))

        if is_eval_format:
            # Eval CSV format
            ref = result.get('reference', '')
            out = result.get('output', '')
            source = result.get('source', '')[:40]
            phrase_display = source
        else:
            # Old format
            ref = result.get('bambara_attendu', '')
            out = result.get('bambara_obtenu', '')
            phrase_display = result.get('phrase_fr', '')[:40]

        if not ref or not out:
            continue

        ref_pos = extract_pos_sequence(ref)
        out_pos = extract_pos_sequence(out)

        # Calculate Kendall tau
        tau = kendall_tau_reordering(ref_pos, out_pos)

        if tau is not None:
            status_symbol = "✅" if ref == out else "❌"
            tau_bar = "█" * int((tau + 1) * 10) + "░" * (20 - int((tau + 1) * 10))
            print(f"[{i+1}] {status_symbol} {phrase_display}...")
            print(f"    Category: {category}")
            print(f"    Ref POS:  {' '.join(ref_pos)}")
            print(f"    Out POS:  {' '.join(out_pos)}")
            print(f"    Kendall τ: {tau:+.3f} {tau_bar}")
            if is_eval_format:
                print(f"    Reference: {ref}")
                print(f"    Output:    {out}")
            print()

            by_category[category].append(tau)
            reordering_stats.append({
                'phrase': phrase_display,
                'category': category,
                'ref_pos': ref_pos,
                'output_pos': out_pos,
                'tau': tau,
                'match': ref == out
            })

    # Statistics
    print(f"\n{'='*80}")
    print(f"📊 WORD ORDER STATISTICS BY CATEGORY")
    print(f"{'='*80}\n")

    all_taus = []
    match_taus = []
    diff_taus = []

    for category in sorted(by_category.keys()):
        taus = by_category[category]
        avg_tau = sum(taus) / len(taus) if taus else 0
        all_taus.extend(taus)

        # Separate by match
        cat_match = [s['tau'] for s in reordering_stats
                     if s['category'] == category and s.get('match', True)]
        cat_diff = [s['tau'] for s in reordering_stats
                    if s['category'] == category and not s.get('match', True)]

        if cat_match:
            match_taus.extend(cat_match)
        if cat_diff:
            diff_taus.extend(cat_diff)

        print(f"Category: {category}")
        print(f"  Count:           {len(taus)}")
        print(f"  Avg τ:           {avg_tau:+.3f}")
        print(f"  Min:             {min(taus):+.3f}")
        print(f"  Max:             {max(taus):+.3f}")
        if cat_match:
            print(f"  Exact match avg τ: {sum(cat_match)/len(cat_match):+.3f} ({len(cat_match)} cases)")
        if cat_diff:
            print(f"  Different avg τ:   {sum(cat_diff)/len(cat_diff):+.3f} ({len(cat_diff)} cases)")
        print()

    # Overall statistics
    if all_taus:
        print(f"\n{'='*80}")
        print(f"📈 OVERALL REORDERING STATISTICS")
        print(f"{'='*80}\n")

        avg_all = sum(all_taus) / len(all_taus)
        print(f"All phrases:")
        print(f"  Total:       {len(all_taus)}")
        print(f"  Avg τ:       {avg_all:+.3f}")
        print(f"  Min:         {min(all_taus):+.3f}")
        print(f"  Max:         {max(all_taus):+.3f}")

        if match_taus:
            print(f"\nExact match cases ({len(match_taus)}):")
            print(f"  Avg τ:       {sum(match_taus)/len(match_taus):+.3f}")
            print(f"  Min:         {min(match_taus):+.3f}")
            print(f"  Max:         {max(match_taus):+.3f}")

        if diff_taus:
            print(f"\nDifferent cases ({len(diff_taus)}):")
            print(f"  Avg τ:       {sum(diff_taus)/len(diff_taus):+.3f}")
            print(f"  Min:         {min(diff_taus):+.3f}")
            print(f"  Max:         {max(diff_taus):+.3f}")

    # Word order rules reference
    print(f"\n{'='*80}")
    print(f"📋 EXTRACTED BAMBARA WORD ORDER RULES")
    print(f"{'='*80}\n")

    for clause_type, rule_info in sorted(rules.items()):
        print(f"{clause_type.upper()}")
        print(f"  Pattern:  {rule_info['pattern']}")
        print(f"  Order:    {' → '.join(rule_info['order'])}")
        print(f"  Type:     {rule_info['sov_type']}")
        print(f"  Note:     {rule_info['description']}")
        print()

    # Interpretation guide
    print(f"\n{'='*80}")
    print(f"💡 INTERPRETATION GUIDE")
    print(f"{'='*80}\n")
    print("""
KENDALL TAU INTERPRETATION:

τ ≈ +1.0  : Reordering matches perfectly (same word order in input and output)
           → Input and output have identical relative ordering
           → System maintains French word order

τ ≈ +0.7  : High ordering consistency
           → Minor reordering acceptable for SOV transformation
           → System mostly follows rule patterns

τ ≈ +0.3  : Moderate consistency
           → Significant reordering occurring
           → Check if this matches expected Bambara SOV pattern

τ ≈ 0.0   : No correlation
           → Word order completely scrambled
           → System may not be following word order rules

τ < 0     : Reverse ordering
           → Words appear in opposite order
           → Serious word order issue

COMBINED METRICS:

High τ + High Sequence Similarity = Good: word order matches, structure preserved
High τ + Low Sequence Similarity = Insertion/deletion, not reordering
Low τ + High Depth Change = Major restructuring (check if intentional)
Low τ + PASS = Acceptable reordering per rules (e.g., French SVO → Bambara SOV)
    """)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        analyze_word_order_csv(csv_file)
    else:
        # Find latest CSV (try resultats first, then eval)
        csv_files = list(Path('.').glob('resultats_bambara_*.csv'))
        if not csv_files:
            csv_files = list(Path('.').glob('eval_*.csv'))

        if csv_files:
            latest_csv = sorted(csv_files)[-1]
            print(f"Using latest CSV: {latest_csv}")
            analyze_word_order_csv(str(latest_csv))
        else:
            print("❌ No CSV files found")
