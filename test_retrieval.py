"""
test_retrieval.py
End-to-end translation test showing top 5 candidates per token.
"""

from pipeline.translation_engine import TranslationEngine
from kg.neo4j_client import Neo4jClient

db     = Neo4jClient()
engine = TranslationEngine(db)

test_sentences = [
    "je mange du riz",
    "il boit de l'eau",
    "la femme mange",
    "le chien court",
    "ronfler la nuit",
]

print("\n🧪 Retrieval / translation test\n")
for s in test_sentences:
    result = engine.translate(s)
    print(f"  ✅ '{s}' → '{result['bambara']}'\n")
