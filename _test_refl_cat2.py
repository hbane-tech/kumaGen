#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/bane/Desktop/kuma_mt')
from kg.neo4j_client import Neo4jClient
from pipeline.translation_engine import TranslationEngine

db = Neo4jClient()
te = TranslationEngine(db)

tests = [
    ("il s'est lavé",                "a yé a jó"),
    ("elle se lave",                 "a bɛ a jó"),
    ("il s'est lavé lui même",       "a yé a yɛrɛ jó"),
    ("il s'est blessé",              "a yé a yɛrɛ màjógin"),
    ("il s'est assis",               "a yé a sìgi"),
    ("il s'est mis à pleurer",       "a yé a yɛrɛ bìla kàsi la"),
    ("je me lavais",                 "n tùn bɛ n jó"),
    ("j'étais en train de me laver", "n tùn bɛ ka n jó"),
]
for (t, expected) in tests:
    r = te.translate(t)
    bambara = r.get('bambara') if isinstance(r, dict) else r
    ok = '✓' if bambara == expected else '✗'
    print(f"RESULT: {ok} {t!r:42s} attendu={expected!r} obtenu={bambara!r}")
