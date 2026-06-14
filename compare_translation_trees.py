#!/usr/bin/env python3
"""
Compare syntactic trees (POS sequences) between:
- kuma_mt (our system)
- Google Translate
- NLLB (No Language Left Behind)

Usage: python compare_translation_trees.py <csv_file>
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
from difflib import SequenceMatcher


def extract_pos_from_text(pos_tree_str: str) -> List[str]:
    """Parse POS tree string into list of POS tags."""
    if not pos_tree_str or pos_tree_str == '':
        return []
    return pos_tree_str.strip().split()


def calculate_tree_similarity(tree1: List[str], tree2: List[str]) -> float:
    """Calculate similarity between two POS sequences (0-1 scale)."""
    if not tree1 or not tree2:
        return 0.0

    matcher = SequenceMatcher(None, tree1, tree2)
    return matcher.ratio()


def get_tree_depth(pos_tree: List[str]) -> int:
    """Estimate tree depth from POS sequence (simple heuristic)."""
    return len(pos_tree)


def get_tree_complexity(pos_tree: List[str]) -> Dict:
    """Analyze tree complexity metrics."""
    if not pos_tree:
        return {'depth': 0, 'unique_pos': 0, 'pos_distribution': {}}

    pos_dist = {}
    for pos in pos_tree:
        pos_dist[pos] = pos_dist.get(pos, 0) + 1

    return {
        'depth': len(pos_tree),
        'unique_pos': len(set(pos_tree)),
        'pos_distribution': pos_dist,
        'most_common': max(pos_dist.items(), key=lambda x: x[1])[0] if pos_dist else None,
    }


def compare_trees(french_pos: str, kuma_pos: str) -> Dict:
    """
    Compare French input tree with kuma_mt output tree.

    Args:
        french_pos: POS tree from French input
        kuma_pos: POS tree from kuma_mt output

    Returns:
        Dict with comparison metrics
    """
    fr_tree = extract_pos_from_text(french_pos)
    km_tree = extract_pos_from_text(kuma_pos)

    return {
        'french_depth': len(fr_tree),
        'kuma_depth': len(km_tree),
        'depth_change': len(km_tree) - len(fr_tree),
        'similarity': calculate_tree_similarity(fr_tree, km_tree),
        'french_complexity': get_tree_complexity(fr_tree),
        'kuma_complexity': get_tree_complexity(km_tree),
    }


def analyze_csv_file(csv_path: str):
    """Analyze the test results CSV file and generate comparison report."""

    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        return

    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    if not results:
        print("❌ No results in CSV file")
        return

    print(f"\n{'='*80}")
    print(f"📊 SYNTACTIC TREE ANALYSIS")
    print(f"{'='*80}")
    print(f"\nAnalyzing {len(results)} translations...\n")

    # Collect statistics
    total_comparisons = 0
    similarity_scores = []
    depth_changes = []

    # Display individual comparisons
    print(f"{'='*80}")
    print(f"Individual Comparisons (Sample)")
    print(f"{'='*80}\n")

    for i, result in enumerate(results[:10]):  # Show first 10
        french_pos = result.get('french_pos_tree', '')
        kuma_pos = result.get('bambara_pos_tree', '')

        if french_pos and kuma_pos:
            comparison = compare_trees(french_pos, kuma_pos)
            similarity_scores.append(comparison['similarity'])
            depth_changes.append(comparison['depth_change'])
            total_comparisons += 1

            print(f"[{i+1}] {result['phrase_fr'][:50]}...")
            print(f"    Status: {result['statut']}")
            print(f"    French tree:  {french_pos}")
            print(f"    Kuma tree:    {kuma_pos}")
            print(f"    Similarity:   {comparison['similarity']:.2%}")
            print(f"    Depth change: {comparison['depth_change']:+d} tokens")
            print()

    # Overall statistics
    print(f"\n{'='*80}")
    print(f"📈 OVERALL STATISTICS")
    print(f"{'='*80}\n")

    if similarity_scores:
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        print(f"Average Tree Similarity:     {avg_similarity:.2%}")
        print(f"Similarity Range:           {min(similarity_scores):.2%} - {max(similarity_scores):.2%}")
        print(f"Median Depth Change:        {sorted(depth_changes)[len(depth_changes)//2]:+d} tokens")
        print(f"Max Expansion:              {max(depth_changes):+d} tokens")
        print(f"Max Compression:            {min(depth_changes):+d} tokens")

    print(f"\n{'='*80}")
    print(f"⚠️  NEXT STEPS FOR FULL COMPARISON:")
    print(f"{'='*80}\n")
    print("To compare with Google Translate and NLLB baselines:")
    print("\n1. Add API calls to generate translations:")
    print("   - Google Translate API (requires credentials)")
    print("   - NLLB model via HuggingFace Transformers")
    print("\n2. Extract POS trees from baseline outputs:")
    print("   - Tokenize baseline outputs")
    print("   - Run spaCy or Bambara tokenizer on results")
    print("\n3. Add baseline columns to CSV:")
    print("   - google_translate_result")
    print("   - nllb_result")
    print("   - google_pos_tree")
    print("   - nllb_pos_tree")
    print("\n4. Generate comparison matrix:")
    print("   - French → Kuma vs French → Google vs French → NLLB")
    print("   - Similarity scores between all pairs")
    print("   - Tree depth analysis")
    print("\n5. Visualization:")
    print("   - Side-by-side POS comparisons")
    print("   - Performance charts")
    print("   - Rule coverage heatmaps")
    print()


def generate_comparison_template():
    """Generate a template for extending the comparison."""

    template = '''#!/usr/bin/env python3
"""
Extended comparison with Google Translate and NLLB baselines.
This template shows how to integrate baseline translations.
"""

import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import requests


class BaselineComparator:
    """Compare kuma_mt with baseline translation systems."""

    def __init__(self):
        # Initialize NLLB model (example)
        # model_name = "facebook/nllb-200-distilled-600M"
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pass

    def google_translate(self, text: str, source_lang: str = 'fr',
                         target_lang: str = 'bm') -> str:
        """
        Translate using Google Translate API.
        Requires: google-cloud-translate

        pip install google-cloud-translate
        """
        # from google.cloud import translate
        # client = translate.TranslationServiceClient()
        # result = client.translate_text(...)
        pass

    def nllb_translate(self, text: str) -> str:
        """
        Translate using NLLB model.
        Requires: transformers, torch
        """
        # inputs = self.tokenizer(text, return_tensors="pt")
        # outputs = self.model.generate(**inputs)
        # return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        pass

    def extract_baseline_pos(self, text: str) -> str:
        """Extract POS from baseline output using spaCy or similar."""
        pass


if __name__ == '__main__':
    print("Template for baseline comparison - implement as needed")
'''

    return template


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        analyze_csv_file(csv_file)
    else:
        # Find latest CSV file
        csv_files = list(Path('.').glob('resultats_bambara_*.csv'))
        if csv_files:
            latest_csv = sorted(csv_files)[-1]
            print(f"Using latest CSV: {latest_csv}")
            analyze_csv_file(str(latest_csv))
        else:
            print("❌ No CSV files found. Run 'python test_phrases.py --run' first")
            print("\nGenerating comparison template...")
            template = generate_comparison_template()
            with open('compare_baselines.py', 'w') as f:
                f.write(template)
            print("✅ Created compare_baselines.py template")
