#!/usr/bin/env python3
"""
Test script to debug reflexive verb handling.
Runs a single reflexive sentence through the pipeline with debug output.
"""
import sys
from kg.neo4j_client import Neo4jClient
from pipeline.translation_engine import TranslationEngine

def test_reflexive():
    print("\n" + "="*80)
    print("TESTING REFLEXIVE VERB DETECTION")
    print("="*80)

    # Initialize database and pipeline
    db = Neo4jClient()
    engine = TranslationEngine(db)

    # Test sentence
    test_sentence = "il est en train de se laver"

    print(f"\nINPUT: {test_sentence}")
    print("\nRunning translation pipeline...\n")

    # Translate
    result = engine.translate(test_sentence)

    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"Bambara: {result['bambara']}")
    print(f"Frame: {result['frame']}")
    print(f"Tree metadata:")
    tree = result.get('tree', {})
    print(f"  - clause_type: {tree.get('clause_type')}")
    print(f"  - tense: {tree.get('tense')}")
    print(f"  - tam: {tree.get('tam')}")
    print(f"  - S: {tree.get('S')}")
    print(f"  - V: {tree.get('V')}")
    print("="*80)

if __name__ == '__main__':
    try:
        test_reflexive()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
