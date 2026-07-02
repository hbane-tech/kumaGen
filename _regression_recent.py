#!/usr/bin/env python3
"""
Regression test — phrases corrigées récemment.
Lance le moteur une seule fois et teste tous les cas.
"""
import os, sys
os.environ.setdefault('DEBUG_TRANSLATE', '0')   # silence les DEBUG verbeux

from pipeline.translation_engine import TranslationEngine
from kg.neo4j_client import Neo4jClient

db     = Neo4jClient()
engine = TranslationEngine(db)

def t(fr):
    return engine.translate(fr)['bambara'].strip()

TESTS = [
    # ─── expl:pass fix (elle se regarde) ──────────────────────────────────────
    ("elle se regarde",
     lambda r: 'yɛrɛ' in r and 'fílɛ' in r,
     "doit contenir 'yɛrɛ fílɛ'  →  a bɛ a yɛrɛ fílɛ"),

    # ─── refl_yere actif (il se regarde) ──────────────────────────────────────
    ("il se regarde",
     lambda r: 'yɛrɛ' in r and 'fílɛ' in r,
     "doit contenir 'yɛrɛ fílɛ'  →  a bɛ a yɛrɛ fílɛ"),

    # ─── se mettre à fix ──────────────────────────────────────────────────────
    ("il s'est mis à travailler",
     lambda r: 'yɛrɛ' in r and 'bìla' in r and 'báara' in r,
     "doit contenir 'yɛrɛ bìla báara'  →  a yé a yɛrɛ bìla báara la"),

    # ─── xcomp obj fix (fusil absent) ─────────────────────────────────────────
    ("cela vaut la peine de prendre un fusil",
     lambda r: 'gàtan' in r,
     "doit contenir 'gàtan' (fusil)"),

    # ─── iobj lexicalisé (Comment t'appelles-tu ?) ────────────────────────────
    ("Comment t'appelles-tu ?",
     lambda r: not r.rstrip(' ?').endswith('i yé'),
     "ne doit PAS se terminer par 'i yé'"),

    # ─── statif relcl (qui est malade → -len dòn) ─────────────────────────────
    ("Je veux manger avec mon mari et ma fille qui est malade",
     lambda r: 'len' in r and 'dòn' in r,
     "doit contenir 'len dòn' (forme statif)"),

    # ─── double 'ni' comitative ────────────────────────────────────────────────
    ("Je veux manger avec mon mari et ma fille qui est malade",
     lambda r: 'ni ni' not in r,
     "ne doit PAS contenir 'ni ni' (doublon comitative)"),

    # ─── avoir_relcl_override (qui a eu un enfant → mìn yé ... sɔrɔ) ──────────
    ("Une femme qui a eu un enfant ne peut pas abandonner son enfant",
     lambda r: 'sɔrɔ' in r and 'sɔrɔla' not in r,
     "doit contenir 'sɔrɔ' sans suffixe '-la'  →  mìn yé dén sɔrɔ"),

    # ─── avoir du courage (ABSTRACT, pas STATIF) ──────────────────────────────
    ("j'ai du courage",
     lambda r: 'dùsukololen' not in r,
     "ne doit PAS contenir 'dùsukololen' (forme statif incorrecte)"),

    # ─── ccomp interrogative (sais-tu que …) — statif, pas de li kɛ ──────────
    ("sais-tu que je suis un enfant ?",
     lambda r: ('dɔ́n' in r or 'don' in r.lower()) and 'li' not in r,
     "doit contenir 'dɔ́n' SANS 'li kɛ'  →  i bɛ dɔ́n ko n yé dén yé wà ?"),

    # ─── coordination copule (elle chante et danse) ───────────────────────────
    ("elle chante et danse",
     lambda r: 'wa' in r,
     "doit contenir 'wa' (coordination)"),

    # ─── venir de + lieu ──────────────────────────────────────────────────────
    ("tu viens d'où ?",
     lambda r: 'bɔ' in r and ('?' in r or 'wà' in r or 'mín' in r),
     "doit contenir 'bɔ' et marqueur interrogatif"),

    # ─── est-ce que copule ────────────────────────────────────────────────────
    ("Est-ce qu'elle travaille ?",
     lambda r: 'báara' in r,
     "doit contenir 'báara'  (travailler)"),

    # ─── se blesser (accidentel, yɛrɛ) ───────────────────────────────────────
    ("il s'est blessé",
     lambda r: 'yɛrɛ' in r and 'màjógin' in r,
     "doit contenir 'yɛrɛ màjógin'  →  a yé a yɛrɛ màjógin"),

    # ─── nous nous préparons (réflexif matériel pluriel, pas réciproque) ──────
    ("nous nous préparons",
     lambda r: 'yɛrɛ' in r and 'ɲɔgɔn' not in r,
     "doit contenir 'yɛrɛ' SANS 'ɲɔgɔn'  →  anw bɛ anw yɛrɛ fìri"),

    # ─── Est-ce que (Yala + bɛ + wà ?, pas optatif ka) ───────────────────────
    ("Est-ce qu'il mange ?",
     lambda r: 'Yala' in r and 'bɛ' in r and 'wà' in r and 'ka' not in r,
     "doit contenir 'Yala' + 'bɛ' + 'wà'  →  Yala a bɛ dúnli kɛ wà ?"),

    # ─── avoir passé simple + possessif (sɔrɔ, pas dèli ; possessif présent) ──
    ("quand il eut ton appel",
     lambda r: 'sɔrɔ' in r and 'dèli' not in r and ('i' in r.split()),
     "doit contenir 'sɔrɔ' SANS 'dèli' avec 'i'  →  tuma min a yé i wéle sɔrɔ"),

    # ─── avoir futur → sɔrɔ (pas bóló) ────────────────────────────────────────
    ("tu auras une voiture",
     lambda r: 'sɔrɔ' in r and 'bóló' not in r and 'na' in r,
     "doit contenir 'sɔrɔ' + 'na' SANS 'bóló'  →  i bɛ na wátiri sɔrɔ"),

    # ─── avoir présent MATERIAL → bóló (pas sɔrɔ) ─────────────────────────────
    ("tu as une voiture",
     lambda r: 'bóló' in r and 'sɔrɔ' not in r,
     "doit contenir 'bóló' SANS 'sɔrɔ'  →  wátiri bɛ i bóló"),
]

print()
print("═" * 65)
print("  REGRESSION — phrases récemment corrigées")
print("═" * 65)

passed = 0
failed = 0
errors = []

for fr, check, desc in TESTS:
    try:
        result = t(fr)
        ok = check(result)
    except Exception as e:
        result = f"[ERREUR: {e}]"
        ok = False

    icon = "✅" if ok else "❌"
    if ok:
        passed += 1
    else:
        failed += 1
        errors.append((fr, result, desc))

    print(f"\n{icon}  FR : {fr}")
    print(f"    BM : {result}")
    if not ok:
        print(f"    ⚠️  attendu : {desc}")

print()
print("═" * 65)
print(f"  {passed}/{passed+failed} tests passés"
      + (f"  —  {failed} échec(s)" if failed else "  ✅ tout bon"))
print("═" * 65)
print()
