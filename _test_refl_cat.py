#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/bane/Desktop/kuma_mt')
from kg.neo4j_client import Neo4jClient
from pipeline.translation_engine import TranslationEngine

db = Neo4jClient()
te = TranslationEngine(db)

tests = [
    "il s'est évanoui",
    "elle se lave",
    "il s'est blessé",
    "il s'est mis à pleurer",
    "il s'est lavé lui même",
    "il se souvient",
]
for t in tests:
    r = te.translate(t)
    bambara = r.get('bambara') if isinstance(r, dict) else r
    print(f"RESULT: {t!r:42s} → {bambara}")
