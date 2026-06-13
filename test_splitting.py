#!/usr/bin/env python3
"""Test clause splitting for relative clauses"""

from kg.neo4j_client import Neo4jClient
from pipeline.translation_engine import TranslationEngine

db = Neo4jClient()
engine = TranslationEngine(db)

# Test case: Rule 8
sentence = "Toi qui prends l'ennemi vivant"
result = engine.translate(sentence)
print(f"INPUT:  {sentence}")
print(f"OUTPUT: {result['bambara']}")
print(f"Expected: i bɛ júgu mɔ́n (with proper relative clause handling)")
print()

# Also test the clause splitting directly
tokens = engine.tagger.tag_and_parse(sentence)
print("TOKENS:")
for t in tokens:
    print(f"  {t['orig_index']:2}: {t['surface']:12} pos={t.get('pos', 'N/A'):6} dep={t.get('dep', 'N/A'):10} head_index={t.get('head_index')}")

print("\nCLAUSES:")
clauses = engine._split_clauses(sentence, tokens)
for i, c in enumerate(clauses):
    print(f"  Clause {i+1}: {c}")
