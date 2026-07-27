"""
utils/language.py
Le pipeline ne traite que le français (parseur spaCy, grammaire des règles,
lexique KG — Sense.fr — sont tous français uniquement ; aucune traduction
anglaise n'est jamais scrapée, Sense.en est vide à 100%). Décision
2026-07-17 : retirer la détection de langue anglais/français, qui ne
faisait que router vers un pipeline anglais non fonctionnel.
"""


def detect_language(sentence: str, db=None) -> str:
    return 'fr'
