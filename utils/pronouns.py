"""
utils/pronouns.py
Fetches pronoun mappings from the KG at runtime.
No hardcoded data.
"""

_cache = {}


def get_pronoun_bm(db, surface: str, role: str = 'subject') -> str:
    """Returns Bambara equivalent of a pronoun, or None."""
    key = (surface.lower(), role)
    if key not in _cache:
        res = db.query("""
        MATCH (p:Pronoun {surface: $surface, role: $role})
        RETURN p.bm AS bm LIMIT 1
        """, {"surface": surface.lower(), "role": role})
        _cache[key] = res[0]['bm'] if res else None
    return _cache[key]


def get_all_surfaces(db, lang: str) -> set:
    """Returns set of all pronoun surface forms for a language."""
    res = db.query("""
    MATCH (p:Pronoun {lang: $lang})
    RETURN p.surface AS surface
    """, {"lang": lang})
    return {r['surface'] for r in res}


def is_negation_marker(db, surface: str, lang: str) -> bool:
    """Returns True if this surface form is a negation marker."""
    res = db.query("""
    MATCH (n:NegMarker {surface: $surface, lang: $lang})
    RETURN count(n) AS cnt
    """, {"surface": surface.lower(), "lang": lang})
    return res[0]['cnt'] > 0 if res else False