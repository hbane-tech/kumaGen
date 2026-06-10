"""
kg/retriever.py

Two-stage retrieval + reranking with hard POS filtering and semantic
context expansion using SemanticGroup nodes from the KG.

  Stage 1 — Exact French text match using toLower CONTAINS.
             \b word boundary regex breaks on accented chars (é, è, à...)
             so we use CONTAINS which is reliable for all French words.
             Hard filtered by POS: VERB tokens only return Verb senses.
  Stage 2 — Sentence-transformer cosine similarity, same POS filter.
  Stage 3 — Reranking: exactness + POS + conciseness + context + match bonus.
"""

import re
import numpy as np
from embeddings.word2vec_encoder import encode
from utils.normalize import normalize_token, clean_gloss


_SPACY_TO_KG_POS = {
    'NOUN':  ['Noun'],
    'VERB':  ['Verb'],
    'ADJ':   ['Adjective'],
    'ADV':   ['Adverb'],
    'PRON':  ['Pronoun'],
    'ADP':   ['Preposition'],
    'CONJ':  ['Conjunction'],
    'PART':  ['Particle'],
    'INTJ':  ['Interjection'],
    'AUX':   ['Auxiliary', 'Verb'],
}


def cosine(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def same_embedding_space(query_vec, stored_vec, tolerance=3.0):
    qn = np.linalg.norm(query_vec)
    sn = np.linalg.norm(stored_vec)
    if qn == 0 or sn == 0:
        return False
    return (max(qn, sn) / min(qn, sn)) <= tolerance


def _allowed_kg_pos(spacy_pos: str) -> list:
    return _SPACY_TO_KG_POS.get(spacy_pos, [])


def _pos_matches(spacy_pos: str, kg_pos: str) -> bool:
    allowed = _allowed_kg_pos(spacy_pos)
    if not allowed:
        return True
    return kg_pos in allowed


# ------------------------------------------------------------------
# Reranking signals
# ------------------------------------------------------------------

def _fr_exactness_bonus(fr_raw: str, token: str) -> float:
    fr = clean_gloss(fr_raw or '').lower().strip()
    t  = token.lower().strip()
    if not fr or not t:
        return 0.0
    if fr == t:
        return 0.08
    if re.match(r'^' + re.escape(t) + r'\s*,', fr):
        return 0.05
    if fr.startswith(t + ' '):
        return 0.04
    if fr.startswith(t):
        return 0.03
    return 0.0


def _pos_bonus(spacy_pos: str, kg_pos: str) -> float:
    if not spacy_pos:
        return 0.0
    return 0.05 if _pos_matches(spacy_pos, kg_pos) else 0.0


def _conciseness_bonus(fr_raw: str) -> float:
    fr    = clean_gloss(fr_raw or '').strip()
    words = len(fr.split())
    if words == 1: return 0.04
    if words == 2: return 0.02
    if words <= 4: return 0.01
    return 0.0


def _context_bonus(fr_raw: str, context_tokens: list,
                   semantic_groups: list = None) -> float:
    """
    Bonus when candidate gloss matches context tokens.
    Includes semantic group expansion loaded from KG SemanticGroup nodes.
    No hardcoded lexicons.
    """
    if not context_tokens or not fr_raw:
        return 0.0
    fr = clean_gloss(fr_raw).lower()

    # Direct token match — use simple CONTAINS for accented words
    direct = sum(
        1 for t in context_tokens
        if t.lower() in fr
    )

    # Semantic group expansion
    semantic = 0
    for group in (semantic_groups or []):
        ctx_hit   = any(t.lower() in group for t in context_tokens)
        gloss_hit = any(w in fr for w in group)
        if ctx_hit and gloss_hit:
            semantic += 1

    total = direct + semantic
    if total >= 2: return 0.10
    if total == 1: return 0.04
    return 0.0


def _gloss_match_score(fr_raw: str, token: str) -> float:
    """
    Score how well the French gloss matches the token.
    Uses CONTAINS logic — no \b boundary (breaks on accented chars).

    Scoring tiers (privilege exact matches):
    1.00 — gloss equals token exactly (e.g., "ami" == "ami")
    0.98 — gloss equals token + period (e.g., "ami." == "ami")
    0.96 — gloss starts with token + space/comma (e.g., "ami prêt" but NOT "ami, autre")
    0.90 — first comma-segment equals token (e.g., "ami" in "ami, bien-aimé" but PENALIZED)
    0.82 — gloss contains token as substring with word boundary
    0.00 — no match
    """
    fr = clean_gloss(fr_raw or '').lower().strip()
    t  = token.lower().strip()
    if not fr or not t:
        return 0.0

    # Tier 1a: Perfect exact match
    if fr == t:
        return 1.00

    # Tier 1b: Exact + period (e.g., "ami.")
    if fr == t + '.':
        return 0.98

    # Tier 1c: Token followed by space/punctuation (NOT comma list)
    # e.g., "ami prêt" is good, but "ami, autre" is NOT
    if fr.startswith(t + ' ') or fr.startswith(t + '.'):
        # Check if it's a simple extension (one more word) vs compound definition
        after_token = fr[len(t):].lstrip('.').strip()
        if after_token and not after_token.startswith(','):
            # Single extension like "ami proche" → good
            return 0.96

    # Tier 2: First segment before comma matches token
    # But PENALIZE if there's significant content after comma
    first_segment = fr.split(',')[0].strip()
    if first_segment == t or first_segment == t + '.':
        # Check if there's additional definitions after comma
        after_comma = fr.split(',', 1)[1].strip() if ',' in fr else ''
        if after_comma and len(after_comma.split()) > 2:
            # Significant definition after comma → "ami, bien-aimé" gets penalized
            return 0.80
        else:
            # Minimal or no additional content → good
            return 0.93
    elif first_segment.startswith(t + ' '):
        # "ami proche, autre" → less good than exact first segment
        after_comma = fr.split(',', 1)[1].strip() if ',' in fr else ''
        return 0.92 if after_comma else 0.94

    # Tier 3: gloss contains token as a meaningful word boundary
    import re as _re
    pattern = r'(^|[ ,;(])' + _re.escape(t) + r'($|[ .,;)])'
    if _re.search(pattern, fr):
        idx = fr.find(t)
        words_before = len(fr[:idx].split()) if idx > 0 else 0
        if words_before >= 4:
            return 0.0
        return 0.82

    return 0.0


def _rerank(candidates: list, token: str, spacy_pos: str,
            context_tokens: list = None,
            semantic_groups: list = None) -> list:
    for c in candidates:
        b1 = _fr_exactness_bonus(c['fr'], token)
        b2 = _pos_bonus(spacy_pos, c.get('pos', ''))
        b3 = _conciseness_bonus(c['fr'])
        # Context bonus removed — scoring based on semantic meaning only
        # Context was causing wrong words to score higher due to
        # coincidental neighboring word matches
        b5 = 0.15 if c.get('match') == 'exact' else 0.0
        final = c['score'] + b1 + b2 + b3 + b5
        # ⚠️  CRITICAL: Cap final_score at 1.0 so the hierarchy is maintained:
        #    exact matches (1.0) > embedding (0.85) > synonyms
        # Without capping, bonuses stack and "ami intime" gets 1.82 > 0.85,
        # defeating the purpose of the 0.85 embedding cap.
        c['final_score'] = round(min(final, 1.0), 4)
        c['bonuses'] = {'exact': b1, 'pos': b2, 'concise': b3,
                        'context': 0.0, 'match': b5}
    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    return candidates


class KGRetriever:

    def __init__(self, db):
        self.db               = db
        self._semantic_groups = self._load_semantic_groups()

    def _load_semantic_groups(self) -> list:
        """Load semantic groups from KG — no hardcoded lexicons."""
        try:
            res = self.db.query("""
            MATCH (g:SemanticGroup)
            RETURN g.keywords AS keywords
            """)
            return [
                set(r['keywords'])
                for r in res
                if r.get('keywords')
            ]
        except Exception:
            return []

    def retrieve(self, token: str, frame: str,
                 spacy_pos: str = None, top_k: int = 10,
                 lang: str = 'fr', context_tokens: list = None,
                 is_verbal_noun: bool = False):
        norm = normalize_token(token)
        if not norm:
            return []

        # Verbal nouns (de lire, à manger): skip POS filter
        # 'lire' finds kàlan (pos=Verb) not restricted to Noun senses
        eff_pos = None if is_verbal_noun else spacy_pos

        exact = self._exact_match(norm, frame, lang=lang, spacy_pos=eff_pos)

        # Improved embedding with context (if available)
        embed = self._embedding_match_with_context(
            norm, frame,
            exclude_bm={c['bm'] for c in exact if c['score'] >= 0.90},
            context_tokens=context_tokens,
            lang=lang, spacy_pos=eff_pos
        )

        combined = exact + embed
        combined = _rerank(combined, norm, eff_pos or '')
        return combined[:top_k]

    def _exact_match(self, norm: str, frame: str,
                     lang: str = 'fr', spacy_pos: str = None):
        """
        Exact French text match using toLower CONTAINS.

        The old regex (?i).*\\b{token}\\b.* breaks on accented French
        characters — \\b treats é, è, à as non-word characters so
        'école' would never match pattern \\bécole\\b.

        Fix: use toLower(s.fr) CONTAINS toLower($token) as primary check.
        This correctly matches 'école.', 'école primaire.', etc.
        Secondary: also try without accents via regex for robustness.
        """
        allowed_pos = _allowed_kg_pos(spacy_pos)
        pos_filter  = "AND s.pos IN $allowed_pos" if allowed_pos else ""

        # Primary: CONTAINS match — reliable for all French words including
        # those with accents (é, è, à, ç, ô, û, etc.)
        results = self.db.query(f"""
        MATCH (w:Word)-[:HAS_SENSE]->(s:Sense)
        WHERE toLower(s.fr) CONTAINS toLower($token)
        {pos_filter}
        RETURN s.bm AS bm, s.fr AS fr, s.en AS en,
               s.frame AS frame, s.pos AS pos
        LIMIT 40
        """, {
            "token":       norm,
            "allowed_pos": allowed_pos,
        })

        candidates = []
        seen_bm    = set()

        for r in results:
            base = _gloss_match_score(r['fr'], norm)
            if base == 0.0:
                continue
            if r['bm'] in seen_bm:
                continue
            seen_bm.add(r['bm'])
            score = self._apply_frame(base, r['frame'], frame)
            candidates.append({
                'bm': r['bm'], 'fr': r['fr'], 'en': r['en'],
                'frame': r['frame'], 'pos': r['pos'],
                'score': round(min(score, 1.0), 4), 'match': 'exact',
            })

        return candidates

    def _embedding_match_with_context(self, norm: str, frame: str,
                                     exclude_bm: set,
                                     context_tokens: list = None,
                                     lang: str = 'fr',
                                     spacy_pos: str = None):
        """
        Enhanced embedding matching that uses surrounding context
        to build a richer query representation.

        Important: Embedding scores are capped at 0.85 so that exact matches
        (which score 1.0) are ALWAYS preferred. This prevents good embedding
        results from overshadowing perfect exact matches.

        If context_tokens available: encode(token + context) for better signal
        Fallback to simple token encoding if no context.
        """
        allowed_pos = _allowed_kg_pos(spacy_pos)
        pos_filter = "AND s.pos IN $allowed_pos" if allowed_pos else ""

        # Build query with context if available
        if context_tokens and len(context_tokens) >= 2:
            # Combine token + context for richer semantic representation
            context_str = " ".join(context_tokens[:5])
            query_text = f"{norm} {context_str}".strip()
            query_vec = encode(query_text)
        else:
            # Fallback to simple token
            query_vec = encode(norm)

        if np.linalg.norm(query_vec) == 0:
            return []

        results = self.db.query(f"""
        MATCH (w:Word)-[:HAS_SENSE]->(s:Sense)
        WHERE s.embedding IS NOT NULL
        {pos_filter}
        RETURN s.bm AS bm, s.fr AS fr, s.en AS en,
               s.frame AS frame, s.pos AS pos,
               s.embedding AS emb
        """, {"allowed_pos": allowed_pos})

        candidates = []
        for r in results:
            if r['bm'] in exclude_bm:
                continue
            stored_vec = np.array(r['emb'], dtype=np.float32)
            if not same_embedding_space(query_vec, stored_vec):
                continue
            score = cosine(query_vec, stored_vec)
            score = self._apply_frame(score, r['frame'], frame)

            # ⚠️  CRITICAL: Cap embedding scores at 0.85 so exact matches (1.0)
            # are ALWAYS preferred over embedding results, even very good ones.
            # This prevents "fìlaninteri" from beating "ami." via synonym fallback.
            score = min(score, 0.85)

            candidates.append({
                'bm': r['bm'], 'fr': r['fr'], 'en': r['en'],
                'frame': r['frame'], 'pos': r['pos'],
                'score': round(score, 4), 'match': 'embed',
            })

        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates

    def _apply_frame(self, score: float, candidate_frame: str,
                     query_frame: str) -> float:
        if query_frame == 'GENERIC':
            return score
        if candidate_frame == query_frame:
            return score * 1.10
        if candidate_frame != 'GENERIC':
            return score * 0.80
        return score