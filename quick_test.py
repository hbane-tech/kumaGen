#!/usr/bin/env python3
from kg.neo4j_client import Neo4jClient
from pipeline.translation_engine import TranslationEngine

db = Neo4jClient()
engine = TranslationEngine(db)

sentence = "Toi qui prends l'ennemi vivant"
print(f"Testing: {sentence}\n")
result = engine.translate(sentence)
print(f"\nFinal result: {result['bambara']}")
