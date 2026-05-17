"""
utils/language.py
Language detection using existing KG nodes.
No hardcoded lexicons, no separate marker list.
"""

_lang_markers_cache = None


def _load_lang_markers(db) -> dict:
    """
    Builds language marker sets from existing KG nodes.
    Uses Pronoun and FunctionWord nodes — already in KG, no extra data needed.
    """
    global _lang_markers_cache
    if _lang_markers_cache is not None:
        return _lang_markers_cache

    res = db.query("""
    MATCH (n)
    WHERE n:Pronoun OR n:FunctionWord OR n:Article OR n:NegMarker
    RETURN n.surface AS surface, n.lang AS lang
    """)

    markers = {'en': set(), 'fr': set()}
    for r in res:
        lang = r.get('lang')
        if lang in markers and r.get('surface'):
            markers[lang].add(r['surface'].lower())

    _lang_markers_cache = markers
    return markers


def detect_language(sentence: str, db=None) -> str:
    """
    Detects 'fr' or 'en'. Defaults to 'fr'.
    Priority:
      1. langdetect library
      2. French accent characters
      3. KG node overlap heuristic
    """
    try:
        from langdetect import detect
        lang = detect(sentence)
        return lang if lang in ('fr', 'en') else 'fr'
    except Exception:
        pass

    accents = set('àâäéèêëîïôùûüÿçœæÀÂÄÉÈÊËÎÏÔÙÛÜŸÇŒÆ')
    if any(c in accents for c in sentence):
        return 'fr'

    if db:
        markers  = _load_lang_markers(db)
        words    = set(sentence.lower().split())
        en_score = len(words & markers.get('en', set()))
        fr_score = len(words & markers.get('fr', set()))
        if en_score > fr_score:
            return 'en'
        if fr_score > en_score:
            return 'fr'

    return 'fr'