#!/usr/bin/env python3
from kg.neo4j_client import Neo4jClient
from pipeline.translation_engine import TranslationEngine

db = Neo4jClient()
engine = TranslationEngine(db)

for sentence in ["chaque jour", "tous les jours", "chacun d entre vous"]:
    print(f"Testing: {sentence}")
    result = engine.translate(sentence)
    print(f"Result: {result['bambara']}\n")
