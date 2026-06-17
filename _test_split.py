import sys
sys.path.insert(0, '/Users/bane/Desktop/kuma_mt')
from kg.neo4j_client import Neo4jClient
from pipeline.translation_engine import TranslationEngine
db = Neo4jClient()
eng = TranslationEngine(db)

phrases = [
    # Avoir non-présent → sɔrɔ
    ("J'aurais une voiture",              "n bɛ na watiri sɔrɔ"),
    ("J'ai eu une voiture",               "n yé watiri sɔrɔ"),
    ("Je n'ai pas eu de voiture",         "n ma watiri sɔrɔ"),
    ("J'avais eu une voiture",            "n tùn yé watiri sɔrɔ"),
    ("Je n'avais pas eu de voiture",      "n tùn ma watiri sɔrɔ"),
    # Avoir présent → bóló
    ("j'ai une voiture",                  "watiri bɛ n bóló"),
    ("j'ai un frère",                     "bálimakɛ bɛ n fɛ"),
    # Regressions
    ("il voit ce vieux",                  "a bɛ màakɔrɔlama yé"),
    ("je mange une pomme",                "n bɛ pɔmu dún"),
]
for (p, expected) in phrases:
    r = eng.translate(p)
    got = r.get('bambara', '')
    ok = '✓' if got == expected else '✗'
    print(f"{ok} {p}")
    print(f"   attendu : {expected}")
    print(f"   obtenu  : {got}")
