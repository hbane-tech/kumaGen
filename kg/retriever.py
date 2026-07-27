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
from embeddings.labse_encoder import encode
from utils.normalize import normalize_token, clean_gloss


_SPACY_TO_KG_POS = {
    'NOUN':  ['Noun'],
    'VERB':  ['Verb'],
    'ADJ':   ['adj'],
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


def _allowed_kg_pos(spacy_pos: str, adj_predicative: bool = False) -> list:
    allowed = list(_SPACY_TO_KG_POS.get(spacy_pos, []))
    # Bambara VQ (verbe de qualité) : le KG tague souvent le même sens
    # "être ADJ." en pos='Verb' plutôt que 'adj' (le bambara n'a pas de
    # copule séparée pour les VQ) — un Verb ne peut se substituer à un
    # ADJ français QU'EN POSITION PRÉDICATIVE (ROOT+cop), jamais en
    # position attributive ("belle maison"). Sans cette allowance, ces
    # sens ne sont trouvés qu'en repêchage cross-POS et plafonnés sous le
    # seuil de confiance automatique, les envoyant à tort en validation
    # LLM (bug trouvé 2026-07-19 : 'jɛ́'="être sûr." rejeté pour 'sûr').
    if spacy_pos == 'ADJ' and adj_predicative and 'Verb' not in allowed:
        allowed.append('Verb')
    return allowed


def _pos_matches(spacy_pos: str, kg_pos: str) -> bool:
    allowed = _allowed_kg_pos(spacy_pos)
    if not allowed:
        return True
    return kg_pos in allowed


# ------------------------------------------------------------------
# Reranking signals
# ------------------------------------------------------------------

def _pos_bonus(spacy_pos: str, kg_pos: str) -> int:
    """POS match bonus — +2 pts if matches, 0 otherwise."""
    if not spacy_pos:
        return 0
    return 2 if _pos_matches(spacy_pos, kg_pos) else 0


def _conciseness_bonus(fr_raw: str) -> int:
    """Conciseness bonus — +1 pt if single word, 0 otherwise."""
    fr = clean_gloss(fr_raw or '').strip()
    return 1 if len(fr.split()) == 1 else 0


def _primacy_key(sense_index, corpus_freq, gloss_fr: str = '') -> tuple:
    """
    Départage entre deux gloses de score par ailleurs identique (jamais
    utilisé pour dépasser un écart de score gloss réel — uniquement pour
    trier des ex-aequo). Trois signaux, dans cet ordre (décision utilisateur
    2026-07-20, remplace l'ancien tie-break par tranche de fréquence log10) :

    0. filtre attesté/non-attesté (freq=0 vs freq>0) — élimine d'abord un
       sens JAMAIS attesté dans le corpus, quel que soit son sense_index.
       Ex: 'fòlofolo'="abandonner, négliger..." sens n°1/freq=0 ne doit PAS
       battre 'bàn'="abandonner" sens n°4/freq=240 juste parce qu'il est
       sens n°1 de sa propre entrée — un sens "principal" jamais attesté
       n'est pas un signal de primauté fiable face à un synonyme bien
       attesté (bug trouvé 2026-07-19). Binaire (attesté/non) plutôt qu'une
       tranche log10 continue : deux fréquences bien attestées mais d'ordres
       de grandeur proches (3110 vs 3726) ne doivent PAS être séparées par
       ce filtre, seulement le cas structurel freq=0 vs freq>0.
    1. sense_index — position du sens DANS SA PROPRE entrée ("1 •", "2 •"...),
       décide ENTRE candidats déjà tous attestés (freq>0) ou déjà tous non-
       attestés (freq=0) : sense_index=1 signale le sens canonique/principal
       du headword. Ex: 'dón'="jour, date." sens n°1/freq=3110 bat 'tìle'=
       "jour, 24 heures." sens n°3/freq=3726 — l'ancien tie-break par
       tranche log10(freq) les séparait à tort en deux tranches adjacentes
       (arrondi 3 vs 4) à cause d'un simple artefact de seuil d'arrondi
       entre deux fréquences pourtant du même ordre de grandeur, empêchant
       sense_index de jamais trancher (bug trouvé 2026-07-20).
    2. corpus_freq — départage UNIQUEMENT entre candidats à sense_index égal
       (ex: 'dá'="jour." freq=27 vs 'dón'="jour, date." freq=3110, tous deux
       sens n°1 de leur propre entrée — la fréquence corpus tranche alors).
    3. absence de liste dans la glose — dernier recours, seulement quand
       tout le reste est À ÉGALITÉ. Une glose à virgules qui matche par son
       premier élément (ex: 'peyizan'="paysan, Mali peyizan, marque de
       semence.") est souvent une entrée mal éclatée regroupant plusieurs
       sens distincts, pas de vrais quasi-synonymes comme 'dón'="jour,
       date." — à fréquence égale, un gloss propre à un seul sens
       ('wúlakɔnɔmɔgɔ'="paysan.") est préféré.
    """
    try:
        freq = int(corpus_freq) if corpus_freq else 0
    except (TypeError, ValueError):
        freq = 0
    try:
        idx = float(sense_index) if sense_index else 1.0
    except (TypeError, ValueError):
        idx = 1.0
    is_list = 1 if ',' in (gloss_fr or '').rstrip('.').strip() else 0
    is_attested = 1 if freq > 0 else 0
    return (is_attested, -idx, freq, -is_list)



def _gloss_match_score(fr_raw: str, token: str, copula_fr: str = '') -> float:
    """
    Score gloss match on 0-100 scale. Ultra-simple hierarchy.

    100 — perfect exact match ("ami" == "ami")
    70  — composite match ("ami, bien-aimé" / "ami intime")
    50  — substring match with word boundary
    0   — no match
    """
    if not fr_raw or not token:
        return 0.0

    fr = clean_gloss(fr_raw or '').lower().strip()
    t = token.lower().strip()

    if not fr or not t:
        return 0.0

    # Point final = simple ponctuation de fin de glose dans le dictionnaire
    # source, pas un marqueur sémantique — sa présence/absence est inconstante
    # d'une entrée à l'autre ("maison." vs "maison") et ne doit pas dégrader
    # un match par ailleurs parfait (bug trouvé 2026-07-17 : 'tìn' fr='maison'
    # sans point battait 'só' fr='maison.' avec point à cause de ce seul écart
    # de score 100 vs 80 — départage réel désormais via sense_index/_primacy_key).
    fr_bare = fr.rstrip('.').strip()

    # Perfect exact match (point final ignoré)
    if fr_bare == t:
        return 100.0

    # Glose lexicographique "COPULE X." (ex: "être sûr.") : convention du
    # dictionnaire source pour gloser un mot-qualité/adjectif prédicatif
    # bambara (le bambara n'a pas de copule séparée pour les VQ) — ce n'est
    # PAS une variante restreinte, c'est la même notion sous forme verbale.
    # Sans ce garde, ce match tombait au palier substring (50 pts) et
    # repassait par la validation LLM, qui l'invalidait à tort comme
    # "catégorie restreinte" (bug trouvé 2026-07-19 : 'jɛ́'="être sûr."
    # rejeté pour 'sûr', menant au placeholder ; 213 gloses KG suivent ce
    # même patron "être X"). copula_fr vient du KG (FunctionWord role=
    # 'copula'), jamais en dur.
    if copula_fr and fr_bare == copula_fr + ' ' + t:
        return 100.0

    # Composite list (comma-separated) où le PREMIER élément = token :
    # "fille, nièce", "ami, camarade" — c'est le sens PRINCIPAL de l'entrée,
    # un match textuellement aussi certain qu'un exact bare (le mot cherché
    # EST le premier sens listé). Même score que l'exact match — sinon deux
    # candidats par ailleurs équivalents (ex: 'dá'="jour." vs 'dón'="jour,
    # date.") ne sont jamais à égalité et le départage par fréquence corpus
    # (_primacy_key) n'a jamais l'occasion de s'appliquer (bug trouvé
    # 2026-07-17 : 'dá', 27 occurrences, battait 'dón', 3110 occurrences,
    # uniquement parce que 100 > 90). Un match plus loin dans la liste
    # (sens secondaire/dérivé) reste à un score inférieur.
    first_segment = fr_bare.split(',')[0].strip()
    if first_segment == t and ',' in fr_bare:
        return 100.0
    if first_segment == t or fr_bare.startswith(t + ' '):
        return 70.0

    # Substring match with word boundary — le palier le MOINS fiable : le
    # mot est présent tel quel, mais rien ne dit s'il est le SUJET de la
    # glose ou juste le qualificatif d'un autre mot ailleurs dans la glose
    # (ex: "lâche" dans "échappement lâche, échappement trop lâche.").
    # Tenté à 30 pts (2026-07-21) pour réduire ce palier — REVERT : ça fait
    # chuter des cas légitimes (ex. "encore actuellement" dans une glose-
    # liste, 52 pts avant) dans la même fourchette qu'un candidat embedding
    # sans rapport, créant une égalité arbitrale gagnée par le mauvais
    # candidat au tie-break. La protection contre les faux positifs de ce
    # palier existe déjà ailleurs (validation LLM par segment isolé pour
    # les gloses-listes ; garde LLM sur le repli "mot le plus proche") —
    # baisser ce score ici n'ajoutait rien et cassait un cas protégé.
    import re as _re
    pattern = r'(^|[ ,;(])' + _re.escape(t) + r'($|[ .,;)])'
    if _re.search(pattern, fr):
        return 50.0

    return 0.0


def _rerank(candidates: list, _token: str, spacy_pos: str,
            context_tokens: list = None,
            semantic_groups: list = None) -> list:
    """Rerank candidates on 0-100 point scale. Ultra-simple: +2 POS, +1 concise."""
    for c in candidates:
        b_pos = _pos_bonus(spacy_pos, c.get('pos', ''))
        b_concise = _conciseness_bonus(c['fr'])
        primacy_key = _primacy_key(c.get('sense_index', 1), c.get('corpus_freq', 0), c.get('fr', ''))

        final = c['score'] + b_pos + b_concise

        # Cap at 100 pts (primauté exclue du plafond — sinon deux candidats
        # à 100 pts plafonnés perdraient le départage entre eux)
        final = min(final, 100.0)

        c['final_score'] = round(final, 1)
        c['_primacy'] = primacy_key
        c['bonuses'] = {'pos': b_pos, 'concise': b_concise, 'primacy': primacy_key}

    candidates.sort(key=lambda x: (x['final_score'], x['_primacy']), reverse=True)
    return candidates


class KGRetriever:

    def __init__(self, db):
        self.db               = db
        self._semantic_groups = self._load_semantic_groups()
        self._copula_fr       = self._load_copula_fr()

    def _load_copula_fr(self) -> str:
        """Lexème copule français depuis le KG (FunctionWord role='copula')."""
        try:
            res = self.db.query(
                "MATCH (f:FunctionWord {lang:'fr', role:'copula'}) "
                "RETURN f.surface AS s LIMIT 1")
            return (res[0].get('s') or '').lower() if res else ''
        except Exception:
            return ''

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

    def retrieve(self, token: str,
                 spacy_pos: str = None, top_k: int = 10,
                 lang: str = 'fr', context_tokens: list = None,
                 is_verbal_noun: bool = False, adj_predicative: bool = False):
        # Handle hyphenated words (là-bas, week-end, etc.)
        # Strategy: try full word first, then split if no matches
        if '-' in token:
            # Try full hyphenated word (preserve hyphen for KG lookup)
            import unicodedata
            token_lower = unicodedata.normalize('NFC', token.lower().strip())

            allowed_pos = _allowed_kg_pos(None if is_verbal_noun else spacy_pos)
            pos_filter = "AND s.pos IN $allowed_pos" if allowed_pos else ""

            # Direct query for hyphenated word
            hyphen_results = self.db.query(f"""
            MATCH (w:Word)-[:HAS_SENSE]->(s:Sense)
            WHERE toLower(s.fr) CONTAINS toLower($token)
            {pos_filter}
            RETURN s.bm AS bm, s.fr AS fr, s.en AS en,
                   s.pos AS pos
            LIMIT 5
            """, {
                "token": token_lower,
                "allowed_pos": allowed_pos,
            })

            if hyphen_results:
                # Found full hyphenated word
                candidates = []
                for r in hyphen_results:
                    score = _gloss_match_score(r['fr'], token_lower, self._copula_fr)
                    if score > 0:
                        candidates.append({
                            'bm': r['bm'],
                            'fr': r['fr'],
                            'score': score,
                            'pos': r['pos']
                        })
                if candidates:
                    candidates = _rerank(candidates, token_lower, spacy_pos or '')
                    return candidates[:top_k]

            # Si pas de correspondance directe, tenter le split par trait-d'union
            # SEULEMENT si les parties ont elles-mêmes des correspondances KG utiles
            # (évite que 'week-end' → 'end' → 'endroit' par faux ami linguistique).
            parts = token.split('-')
            combined_results = []
            _all_parts_have_match = True
            for part in parts:
                if part.strip():
                    part_results = self.retrieve(
                        part.strip(),
                        spacy_pos=spacy_pos,
                        top_k=3,
                        lang=lang,
                        context_tokens=context_tokens,
                        is_verbal_noun=is_verbal_noun
                    )
                    if part_results and part_results[0].get('score', 0) >= 40:
                        combined_results.append(part_results[0])
                    else:
                        _all_parts_have_match = False

            # N'utiliser le résultat combiné que si TOUTES les parties ont
            # une correspondance fiable (score ≥ 40) — sinon laisser le
            # retriever embedding traiter le composé entier.
            if combined_results and _all_parts_have_match:
                combined_bm = ' '.join(r['bm'] for r in combined_results)
                return [{
                    'bm': combined_bm,
                    'fr': '-'.join(parts),
                    'score': 50,
                    'final_score': 50.0,
                    'pos': 'Combined'
                }]
            # Pas de split fiable → laisser tomber dans le retriever embedding
            # avec le token complet (ex: 'week-end' → embedding → dɔ́gɔkun)

        norm = normalize_token(token)
        if not norm:
            return []

        # Verbal nouns (de lire, à manger): skip POS filter
        # 'lire' finds kàlan (pos=Verb) not restricted to Noun senses
        eff_pos = None if is_verbal_noun else spacy_pos

        exact = self._exact_match(norm, lang=lang, spacy_pos=eff_pos,
                                   adj_predicative=adj_predicative)

        # ── FALLBACK CROSS-POS : aucun match exact de même POS ────────────────
        # Le KG n'a parfois qu'un homographe d'une AUTRE catégorie grammaticale
        # pour un mot (ex: "finale" tagué ADJ par spaCy dans "décision finale",
        # mais le KG n'a que 'fínali' en Noun — le sens sportif "la finale").
        # MAIS spaCy donne un POS fiable — l'absence de candidat de même POS
        # signale le plus souvent un vrai trou lexical (ex: "ferme" tagué NOUN
        # ["la ferme" = la ferme agricole] n'a aucun Noun dans le KG), pas une
        # invitation à emprunter un HOMOGRAPHE SANS RAPPORT d'une autre
        # catégorie (ex: "ferme" Adverbe = "solide, dur" — mot différent).
        # Score plafonné sous le seuil de confiance automatique (90 pts,
        # cf. _validate_candidate_semantics) : le candidat passe donc quand
        # même par la validation sémantique LLM avant d'être retenu, au lieu
        # d'être accepté aveuglément comme un vrai match de même POS.
        # Bug trouvé 2026-07-17 : "finale" tombait en placeholder alors que
        # 'fínali' existe avec une glose identique.
        if eff_pos and not any(c['score'] >= 90.0 for c in exact):
            _cross_pos = self._exact_match(norm, lang=lang, spacy_pos=None)
            _seen_bm = {c['bm'] for c in exact}
            for c in _cross_pos:
                if c['score'] >= 90.0 and c['bm'] not in _seen_bm:
                    c['score'] = 85.0
                    c['match'] = 'cross_pos'
                    exact.append(c)
                    _seen_bm.add(c['bm'])

        # Improved embedding with context (if available)
        # Exclude exact matches with high score (90+ pts) from embedding results
        # (cross_pos inclus malgré son score plafonné à 85 — même bm, pas
        # besoin d'un doublon embedding pour le même candidat).
        embed = self._embedding_match_with_context(
            norm,
            exclude_bm={c['bm'] for c in exact
                        if c['score'] >= 90.0 or c.get('match') == 'cross_pos'},
            context_tokens=context_tokens,
            lang=lang, spacy_pos=eff_pos
        )

        combined = exact + embed
        combined = _rerank(combined, norm, eff_pos or '')
        top = combined[:top_k]

        # Un match "exact" à score élevé (70/100 pts) peut n'être qu'une
        # coïncidence de substring dans une glose composée ("situation DE
        # GROUPE" contient "situation" mais dénote tout autre chose) — la
        # validation sémantique en aval (LLM) les invalide presque toujours.
        # Si tous les matches exacts sont de ce type et remplissent déjà
        # top_k, un vrai synonyme sémantique (embed, score plus bas car
        # capé à 40 pts) n'a jamais la chance d'être vu par la validation
        # (bug trouvé 2026-07-20 : "situation" → aucun candidat embed
        # (ex: 'kísa'="statut, état.", cosinus 0.69) n'atteignait jamais le
        # pool car noyé par 7 gloses composées "situation de X" à 70 pts,
        # toutes invalidées ensuite). Garantir un minimum de candidats embed
        # dans le pool retourné, même si ça dépasse top_k de quelques unités
        # — le pool complet passe de toute façon par la validation
        # sémantique en aval, qui tranchera leur pertinence réelle.
        _min_embed = 5
        _embed_in_top = [c for c in top if c.get('match') == 'embed']
        if len(_embed_in_top) < _min_embed:
            _embed_ranked = [c for c in combined if c.get('match') == 'embed']
            _missing = _min_embed - len(_embed_in_top)
            _extra = [c for c in _embed_ranked if c not in top][:_missing]
            top = top + _extra
        return top

    def _exact_match(self, norm: str,
                     lang: str = 'fr', spacy_pos: str = None,
                     adj_predicative: bool = False):
        """
        Exact French text match using toLower CONTAINS.
        Returns candidates with scores on 0-100 point scale.

        The old regex (?i).*\\b{token}\\b.* breaks on accented French
        characters — \\b treats é, è, à as non-word characters so
        'école' would never match pattern \\bécole\\b.

        Fix: use toLower(s.fr) CONTAINS toLower($token) as primary check.
        This correctly matches 'école.', 'école primaire.', etc.
        """
        allowed_pos = _allowed_kg_pos(spacy_pos, adj_predicative)
        pos_filter  = "AND s.pos IN $allowed_pos" if allowed_pos else ""

        # Primary: CONTAINS match — reliable for all French words including
        # those with accents (é, è, à, ç, ô, û, etc.)
        results = self.db.query(f"""
        MATCH (w:Word)-[:HAS_SENSE]->(s:Sense)
        WHERE toLower(s.fr) CONTAINS toLower($token)
        {pos_filter}
        RETURN s.bm AS bm, s.fr AS fr, s.en AS en,
               s.pos AS pos,
               coalesce(s.sense_index, 1) AS sense_index,
               coalesce(s.corpus_freq, 0) AS corpus_freq
        ORDER BY
          CASE
            WHEN toLower(s.fr) = toLower($token) THEN 0
            WHEN toLower(s.fr) = toLower($token + '.') THEN 1
            WHEN toLower(trim(s.fr)) STARTS WITH toLower($token) THEN 2
            ELSE 3
          END ASC,
          coalesce(s.sense_index, 1) ASC,
          coalesce(s.corpus_freq, 0) DESC,
          size(s.fr) ASC
        LIMIT 40
        """, {
            "token":       norm,
            "allowed_pos": allowed_pos,
        })

        # DEBUG: Show all KG results before scoring
        print(f"     [RETRIEVE] Query returned {len(results)} results for token='{norm}':")
        for r in results[:10]:  # Show first 10
            print(f"       - fr='{r['fr']}' | bm='{r['bm']}'")

        candidates = []
        seen_bm    = set()

        for r in results:
            base = _gloss_match_score(r['fr'], norm, self._copula_fr)  # 0-100 pts
            if base == 0.0:
                continue
            if r['bm'] in seen_bm:
                continue
            seen_bm.add(r['bm'])
            # Cap at 100 pts
            score = min(base, 100.0)
            candidates.append({
                'bm': r['bm'], 'fr': r['fr'], 'en': r['en'],
                'pos': r['pos'],
                'score': round(score, 1), 'match': 'exact',
                'sense_index': r.get('sense_index', 1),
                'corpus_freq': r.get('corpus_freq', 0),
            })

        return candidates

    def _embedding_match_with_context(self, norm: str,
                                     exclude_bm: set,
                                     context_tokens: list = None,
                                     lang: str = 'fr',
                                     spacy_pos: str = None):
        """
        Embedding matching on 0-100 point scale.
        STRICTLY capped at 40 pts so gloss matches (100/70/50) always win.

        If context_tokens available: encode(token + context) for better signal
        Fallback to simple token encoding if no context.
        """
        allowed_pos = _allowed_kg_pos(spacy_pos)
        pos_filter = "AND s.pos IN $allowed_pos" if allowed_pos else ""

        # Encoder le mot NU en plus de la version contexte-augmentée : la
        # glose KG candidate est elle-même toujours SANS contexte (juste
        # "recrutement.", pas "recrutement université programme..."). Comparer
        # un vecteur contexte-dilué à une glose nue est structurellement
        # défavorable même à un match évident — ex: "recruter" (nu) vs
        # "recrutement." = cosine 0.93, mais "recruter université actuellement
        # participants" (avec contexte) vs "recrutement." chute à 0.56, assez
        # pour faire disparaître le seul bon candidat sous des candidats
        # contexte-pollués sans rapport (bug trouvé 2026-07-20). Le contexte
        # reste utile pour désambiguïser un mot polysémique (ex: "avocat"
        # fruit/métier) — donc on garde les deux vecteurs et on prend le MAX
        # par candidat, au lieu de remplacer le signal nu par le signal
        # contextualisé.
        query_vec_bare = encode(norm)
        query_vec_ctx = None
        if context_tokens and len(context_tokens) >= 2:
            context_str = " ".join(context_tokens[:5])
            query_text = f"{norm} {context_str}".strip()
            query_vec_ctx = encode(query_text)

        if np.linalg.norm(query_vec_bare) == 0:
            return []

        results = self.db.query(f"""
        MATCH (w:Word)-[:HAS_SENSE]->(s:Sense)
        WHERE s.embedding IS NOT NULL
        {pos_filter}
        RETURN s.bm AS bm, s.fr AS fr, s.en AS en,
               s.pos AS pos, s.embedding AS emb,
               coalesce(s.sense_index, 1) AS sense_index,
               coalesce(s.corpus_freq, 0) AS corpus_freq
        """, {"allowed_pos": allowed_pos})

        query_dim = query_vec_bare.shape[0]
        candidates = []
        for r in results:
            if r['bm'] in exclude_bm:
                continue
            stored_vec = np.array(r['emb'], dtype=np.float32)
            if stored_vec.shape[0] != query_dim:
                continue  # stale embedding from previous model (different dim)
            if not same_embedding_space(query_vec_bare, stored_vec):
                continue
            # Cosine similarity (0-1) → scale to 0-40 pts. Max du signal nu
            # et contexte-augmenté (voir note plus haut) — le contexte ne
            # doit jamais faire moins bien que le mot seul, seulement mieux
            # pour désambiguïser un terme polysémique.
            cosine_sim = cosine(query_vec_bare, stored_vec)
            if query_vec_ctx is not None:
                cosine_sim = max(cosine_sim, cosine(query_vec_ctx, stored_vec))
            score = cosine_sim * 40.0

            candidates.append({
                'bm': r['bm'], 'fr': r['fr'], 'en': r['en'],
                'pos': r['pos'],
                'score': round(score, 1), 'match': 'embed',
                'sense_index': r.get('sense_index', 1),
                'corpus_freq': r.get('corpus_freq', 0),
            })

        candidates.sort(key=lambda x: (
            x['score'], -x.get('sense_index', 1), x.get('corpus_freq', 0)
        ), reverse=True)
        return candidates

