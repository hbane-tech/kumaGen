#!/usr/bin/env python3
"""
Test de la validation sémantique LLM pour la traduction.
Vérifie que le LLM peut correctement valider/rejeter les candidats.
"""

from kg.neo4j_client import Neo4jClient
from pipeline.translation_engine import TranslationEngine

db = Neo4jClient()
engine = TranslationEngine(db)

# Test cases: (input_fr, expected_bm_or_meaning)
test_cases = [
    ("il se lave", "laver (cleaner avec eau)"),
    ("elle court", "courir (run)"),
    ("je pense", "penser (think)"),
    ("tu chantes", "chanter (sing)"),
]

print("=" * 75)
print("🧪 TEST VALIDATION SÉMANTIQUE LLM")
print("=" * 75)

for sentence_fr, description in test_cases:
    print(f"\n📝 Entrée: {sentence_fr}")
    print(f"   Attendu: {description}")
    print("-" * 75)

    result = engine.translate(sentence_fr)
    bambara = result.get('bambara', '')

    print(f"✅ Résultat Bambara: {bambara}")
    print()

print("=" * 75)
print("Observations:")
print("- Les validations sémantiques REJETTENT les mauvaises correspondances?")
print("- Les meilleures traductions sont sélectionnées après le LLM reranking?")
print("=" * 75)

db.driver.close()
