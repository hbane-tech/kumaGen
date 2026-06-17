#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/bane/Desktop/kuma_mt')

from kg.neo4j_client import Neo4jClient
from pipeline.translation_engine import TranslationEngine

try:
    db = Neo4jClient()
    engine = TranslationEngine(db)

    # Test case with xcomp object
    test_phrases = [
        "cela vaut la peine de prendre un fusil",
    ]

    for phrase in test_phrases:
        print(f"\n{'='*60}")
        print(f"FR > {phrase}")
        result = engine.translate(phrase)
        print(f"BM > {result}")
        print(f"{'='*60}")

except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
