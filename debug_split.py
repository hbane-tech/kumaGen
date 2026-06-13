#!/usr/bin/env python3
"""Debug clause splitting"""

from pipeline.spacy_parser import SpacyParser
from kg.neo4j_client import Neo4jClient

db = Neo4jClient()
parser = SpacyParser(db)

# Test relative clause
sentence = "Toi qui prends l'ennemi vivant"
tokens = parser.parse(sentence)

print(f"Sentence: {sentence}")
print("\nTokens:")
for t in tokens:
    print(f"  {t['orig_index']}: '{t['surface']}' pos={t.get('pos')} dep={t.get('dep')} head_index={t.get('head_index')}")

print("\nLooking for acl:relcl children:")
for tok in tokens:
    if tok.get('dep') == 'acl:relcl':
        print(f"  Found acl:relcl verb: '{tok['surface']}' orig_index={tok['orig_index']}")
        children = [t for t in tokens if t.get('head_index') == tok['orig_index']]
        print(f"  Children (tokens with head_index={tok['orig_index']}): {len(children)}")
        for c in children:
            print(f"    - '{c['surface']}' head_index={c.get('head_index')} orig_index={c.get('orig_index')}")

        if children:
            relative_marker = min(children, key=lambda t: t['orig_index'])
            print(f"  → Selected relative marker: '{relative_marker['surface']}'")

print("\nTest sentence.find():")
print(f"  sentence.find('qui', 0) = {sentence.find('qui', 0)}")
print(f"  sentence[0:4] = '{sentence[0:4]}'")
print(f"  sentence[4:] = '{sentence[4:]}'")

# Test the actual clause splitting
print("\n\nNow testing _split_clauses():")
from pipeline.translation_engine import TranslationEngine
engine = TranslationEngine(db)
clauses = engine._split_clauses(sentence, tokens=tokens)
print(f"Result: {clauses}")
for i, c in enumerate(clauses):
    print(f"  Clause {i+1}: '{c}'")
