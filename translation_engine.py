"""
pipeline/translation_engine.py
Translates FR/EN -> Bambara using hybrid LLM+KG+Rules architecture.

Layer 1: LLM morphological parsing (lemma extraction)
Layer 2: KG sense retrieval — exact French gloss match + embedding fallback
Layer 3: LLM contextual reranking — picks best sense for context
Layer 4: Bambara grammar rule engine (R1-R60)

Fallback chain:
  1. KG exact match          — best case
  2. KG embedding match      — acceptable
  3. LLM reranking           — disambiguation
  4. -nen suffix (ADJ)       — morphological derivation
  5. Synonym lookup (LLM+KG) — semantic expansion
     Synonym candidate injected into ranked list, competes by score.
     Guard: skip synonym if any existing candidate's gloss already
     contains the search word (embedding found the right word).
  6. Raw lemma               — last resort

Synonym validation (FIX):
  The embedding-based cosine check (v_syn vs v_gloss) is circular:
  if the underlying embed space already maps 'assaillir' near 'appliqué'
  (the retrieval bug), the same model will confirm sim ≥ 0.75 and
  accept a completely wrong result.
  → Replaced with a tiny LLM yes/no call that breaks out of the
    embedding space and reasons about meaning directly.

Low-signal guard (FIX):
  When all candidates are embedding-only AND top score < 0.50,
  the embedding space has zero useful signal; synonym candidates
  from the same space are equally unreliable.
  → _needs_synonym now returns False in this case, emitting a
    placeholder immediately instead of wasting an LLM call.
"""

import re
import requests
from pipeline.tokenizer import tokenize
from pipeline.frame_parser import FrameParser
from kg.retriever import KGRetriever
from embeddings.word2vec_encoder import encode as _embed
from rules.r1_r60_engine import RuleEngine
from config.settings import LLM_BACKEND, LLM_MODEL
import os
from google import genai
from google.genai import types
# from rules.r1_r60_engine import build_tree
from embeddings.word2vec_encoder import _get_model

MIN_SCORE      = 0.12
TOP_K          = 10
ADJ_CONFIDENCE = 0.70
VERB_MIN_SCORE = 0.75

# Minimum embed top-score below which the embedding space is considered
# dead — synonyms from the same space won't rescue the search.
_EMBED_DEAD_ZONE = 0.50


class TranslationEngine:

    def __init__(self, db):
        self.db                = db
        self.retriever         = KGRetriever(db)
        self.frame_parser      = FrameParser(db)
        self.rule_engine       = RuleEngine(db)
        self._current_sentence = ''
        self.model = _get_model()

    # ------------------------------------------------------------------
    # LLM CALLS
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, max_tokens: int = 10) -> str:
        """Generic LLM call — returns raw text response."""
        backend = LLM_BACKEND
        model   = LLM_MODEL

        if backend == 'ollama':
            payload = {
                "model":   model or 'qwen2.5:3b',
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0,
                            "num_predict": max_tokens},
            }
            r = requests.post(
                "http://localhost:11434/api/generate",
                json=payload, timeout=120)
            return r.json()["response"].strip()

        elif backend == 'gemini':
            client   = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
            response = client.models.generate_content(
                model=model or 'gemini-2.0-flash',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            )
            return response.text.strip()

        return ''

    def _call_llm_index(self, prompt: str):
        """Call LLM expecting a number — returns 0-based index or None."""
        resp  = self._call_llm(prompt, max_tokens=5)
        match = re.search(r'\d+', resp)
        return int(match.group()) - 1 if match else None

    # ------------------------------------------------------------------
    # SYNONYM NEED GUARD
    # ------------------------------------------------------------------

    def _needs_synonym(self, tok_lemma: str, candidates: list,
                       all_embed: bool) -> bool:
        """
        Decide whether synonym fallback should be attempted.

        Do NOT attempt synonym when:
        - A candidate's French gloss already contains the search word
          (embedding found semantically correct result)
        - Top candidate is an exact match with high score
        - FIX: all_embed AND top_score < _EMBED_DEAD_ZONE
          The embedding space has zero useful signal; synonym candidates
          retrieved via the same embedding will be equally unreliable.
          Short-circuit to placeholder immediately.

        DO attempt synonym when:
        - No candidates
        - Top score < 0.70
        - All embedding results AND no gloss contains the search word
        """
        if not candidates:
            return True

        top_score   = candidates[0]['final_score']
        lemma_lower = tok_lemma.lower()

        # Check if any candidate's French gloss contains the search word
        gloss_match = any(
            lemma_lower in c.get('fr', '').lower()
            for c in candidates[:5]
        )
        if gloss_match:
            return False

        # FIX: dead embedding zone — synonym from the same space won't help.
        # Score this low means the model found nothing semantically related;
        # any synonym it retrieves will be equally random. Emit placeholder.
        if all_embed and top_score < _EMBED_DEAD_ZONE:
            print(f"     📭 [{tok_lemma}] — embed dead zone "
                  f"(score={top_score:.2f} < {_EMBED_DEAD_ZONE}), "
                  f"skipping synonym fallback")
            return False

        # Low confidence — always try synonym
        if top_score < 0.70:
            return True

        # All embedding with no gloss match — synonym may find exact match
        if all_embed:
            return True

        return False

    # ------------------------------------------------------------------
    # LLM CONTEXTUAL RERANKING
    # ------------------------------------------------------------------

    def _rerank_with_llm(self, token_lemma: str,
                         candidates: list,
                         tok_pos: str = 'NOUN',
                         context_tokens: list = None) -> list:
        """
        Use LLM to pick the best candidate sense.

        Triggers:
        1. Multiple exact matches with close scores AND simpler exists
        2. All embedding results with low confidence (< 0.75)
        3. VERB with multiple exact matches very close (<=0.05 gap)
           AND top candidate has compound gloss
        """
        if len(candidates) < 2:
            return candidates

        top       = candidates[0]
        top_score = top['final_score']
        if top_score >= 0.80:
            return candidates


        exact_matches = [c for c in candidates if c.get('match') == 'exact']
        close         = [c for c in candidates
                         if top_score - c['final_score'] <= 0.10]
        top_fr_words  = len(top['fr'].strip().rstrip('.').split())
        has_simpler   = any(
            len(c['fr'].strip().rstrip('.').split()) < top_fr_words
            for c in candidates[1:]
        )
        all_embed      = all(c.get('match') == 'embed' for c in candidates)
        low_confidence = top_score < 0.75

        second_exact_score = exact_matches[1]['final_score'] \
                             if len(exact_matches) >= 2 else 0

        context_match_in_lower = any(
            any(ctx.lower() in c.get('fr','').lower()
                for ctx in (context_tokens or []))
            for c in candidates[1:4]
        ) if context_tokens else False

        is_context_sensitive = (
            tok_pos == 'VERB'
            and not all_embed
            and len(exact_matches) >= 2
            and (top_score - second_exact_score) <= 0.05
            and (top_fr_words >= 2 or context_match_in_lower)
        )

        should_rerank = (
            (len(exact_matches) >= 2 and len(close) >= 2 and has_simpler)
            or (all_embed and low_confidence)
            or is_context_sensitive
        )

        if not should_rerank:
            return candidates

        if all_embed and low_confidence:
            rerank_pool = candidates
        elif is_context_sensitive:
            rerank_pool = [c for c in candidates
                           if top_score - c['final_score'] <= 0.15]
        else:
            rerank_pool = [c for c in candidates
                           if top_score - c['final_score'] <= 0.10]

        if len(rerank_pool) < 2:
            return candidates

        options = '\n'.join(
            f"{i+1}. {c['bm']} — {c['fr']}"
            for i, c in enumerate(rerank_pool[:10])
        )

        prompt = (
            f'You are a semantic disambiguation expert.\n\n'
            f'Sentence: "{self._current_sentence}"\n'
            f'Word to translate: "{token_lemma}"\n\n'
            f'The options below are Bambara translations with their '
            f'French meanings.\n'
            f'Choose the one whose French meaning is semantically '
            f'closest to "{token_lemma}" as used in this sentence.\n\n'
            f'Consider the full sentence context carefully.\n'
            f'Prefer simple direct meanings over compound or '
            f'specialized ones.\n'
            f'If none match well, choose the semantically '
            f'closest option.\n\n'
            f'Options:\n{options}\n\n'
            f'Reply with ONLY the number (1-10).'
        )

        try:
            chosen_idx = self._call_llm_index(prompt)
            if chosen_idx is not None and \
               0 <= chosen_idx < len(rerank_pool):
                chosen = rerank_pool[chosen_idx]
                if chosen in candidates:
                    candidates.remove(chosen)
                    candidates.insert(0, chosen)
                    print(f"     🧠 LLM reranked → #{chosen_idx+1} "
                          f"'{chosen['bm']}' ({chosen['fr']})")
        except Exception as e:
            print(f"     ⚠️  LLM rerank failed: {e}")

        return candidates

    # ------------------------------------------------------------------
    # SYNONYM FALLBACK
    # ------------------------------------------------------------------

    def _validate_synonym_candidate(self, original_lemma: str,
                                    syn: str, best: dict) -> bool:
        """
        Validate that a KG candidate is a genuine translation of `syn`
        (and therefore of `original_lemma`).

        FIX: The previous approach compared cosine(embed(syn), embed(gloss)).
        This is circular: the same embedding model that retrieved a wrong
        candidate (e.g. tíminandi/appliqué for assaillir/attaquer) will
        also report high similarity between 'assaillir' and 'appliqué'
        because both words live near the same compressed cluster.

        New approach: ask the LLM directly. A one-shot yes/no call is
        cheap (max_tokens=3) and semantically correct.

        Fallback (if LLM call fails): check whether the gloss textually
        starts with or contains the synonym stem. This is weaker but
        safe — it never accepts completely unrelated glosses.
        """
        best_fr   = best.get('fr', '').lower().rstrip('.').strip()
        syn_lower = syn.lower().strip()

        # Fast text pass: if the gloss contains the synonym word directly,
        # it's clearly valid — skip the LLM call.
        if syn_lower in best_fr or best_fr.startswith(syn_lower):
            return True

        # LLM semantic judge — breaks out of the embedding space.
        # Q: does the French gloss mean approximately the same as the synonym?
        prompt = (
            f'Does the French word or expression "{best_fr}" mean '
            f'approximately the same as "{syn}"?\n'
            f'Reply with YES or NO only.'
        )
        try:
            resp = self._call_llm(prompt, max_tokens=3).strip().upper()
            is_valid = resp.startswith('Y')
            if not is_valid:
                print(f"     ⚠️  LLM semantic check: "
                      f"'{best_fr}' ≠ '{syn}' → rejected "
                      f"(bm='{best.get('bm')}', "
                      f"original='{original_lemma}')")
            return is_valid
        except Exception as e:
            print(f"     ⚠️  LLM validation failed: {e}")
            # Conservative fallback: only accept if gloss contains
            # at least the first 5 characters of the synonym
            stem = syn_lower[:5]
            return len(stem) >= 4 and stem in best_fr

    def _try_synonym_fallback(self, tok: dict, frame: str,
                               context_lemmas: list):
        """
        Ask LLM for synonyms, retrieve KG candidates, return best dict.

        The returned candidate is injected into the main candidate list
        so it competes fairly by score.

        Guard in _needs_synonym() prevents false positives.
        Validation in _validate_synonym_candidate() uses LLM (not
        embedding cosine) to break out of the compressed embedding space.
        """
        lemma = tok['lemma']
        lang  = tok.get('lang', 'fr')
        pos   = tok.get('pos', 'VERB')

        pos_label = {
            'VERB': 'verbs',
            'NOUN': 'nouns',
            'ADJ':  'adjectives',
            'ADV':  'adverbs',
        }.get(pos, 'words')

        prompt = (
            f'List 3 French {pos_label} that mean the same as "{lemma}".\n'
            f'Do NOT include "{lemma}" itself in the list.\n'
            f'Reply with ONLY 3 different French words separated by commas.\n'
            f'Examples:\n'
            f'- construire -> bâtir, édifier, ériger\n'
            f'- étudier -> apprendre, réviser, lire\n'
            f'- reculé -> lointain, éloigné, isolé\n'
            f'- marcher -> avancer, se déplacer, aller\n'
            f'{lemma} ->'
        )

        try:
            response = self._call_llm(prompt, max_tokens=25)
            if not response:
                return None

            synonyms = [s.strip().lower()
                        for s in response.replace('->', '').split(',')]
            lemma_lower = lemma.lower()
            synonyms = [
                s for s in synonyms
                if s
                and len(s) > 1
                and s != lemma_lower
                and lemma_lower not in s
            ][:3]

            if not synonyms:
                return None

            print(f"     🔧 Synonyms for '{lemma}': {synonyms}")

            for syn in synonyms:
                syn_candidates = self.retriever.retrieve(
                    syn, frame,
                    spacy_pos=pos,
                    top_k=3,
                    lang=lang,
                )

                best = syn_candidates[0] if syn_candidates else None
                if not best or best['final_score'] < 0.90:
                    continue

                # FIX: LLM-based semantic validation instead of
                # embedding cosine similarity.
                # Embedding cosine is circular — the same compressed
                # vector space that produced the wrong retrieval will
                # also report high similarity between an unrelated gloss
                # and the synonym (observed: assaillir ≈ appliqué @ 0.986).
                is_valid = self._validate_synonym_candidate(
                    original_lemma=lemma,
                    syn=syn,
                    best=best,
                )

                if is_valid:
                    best['via_synonym'] = syn
                    print(f"🔧 Synonym match: '{syn}' → "
                          f"'{best['bm']}' ({best['fr']}) "
                          f"score={best['final_score']:.3f}")
                    return best
                else:
                    print(f"⚠️  Rejected: '{best['bm']}' "
                          f"({best['fr']}) — LLM: '{best.get('fr','')}' "
                          f"≠ '{syn}'")

        except Exception as e:
            print(f"⚠️  Synonym fallback failed: {e}")

        return None

    # ------------------------------------------------------------------
    # SEMANTIC CLASS DETECTION (LLM + KG cache)
    # ------------------------------------------------------------------
    def _detect_statif_adj(self, lemma: str) -> str:
        if not hasattr(self, '_statif_cache'):
            self._statif_cache = {}
        if lemma in self._statif_cache:
            return self._statif_cache[lemma]

        import time

        # Question 1 : est-ce un participe passé ?
        prompt1 = (
            f'Is "{lemma}" a French past participle from a verb?\n'
            f'Examples YES: cuit (cuire), fini (finir), cassé (casser), '
            f'fermé (fermer), ouvert (ouvrir), brûlé (brûler)\n'
            f'Examples NO: grand, vrai, beau, rapide, assis, fatigué\n'
            f'Answer YES or NO only.'
        )
        for _attempt in range(3):
            try:
                import requests as _req
                payload = {'model': 'qwen2.5:0.5b', 'prompt': prompt1,
                           'stream': False, 'options': {'temperature': 0, 'num_predict': 3}}
                r = _req.post('http://localhost:11434/api/generate',
                              json=payload, timeout=15)
                raw1 = r.json().get('response', '').strip().upper()
                print(f"     🔍 qwen participe? '{raw1}'")
                if raw1.startswith('Y'):
                    self._statif_cache[lemma] = 'PARTICIPE'
                    return 'PARTICIPE'
                break
            except Exception as e:
                print(f"     🔍 attempt {_attempt+1} failed: {e}")
                time.sleep(1)

        # Question 2 : est-ce un statif (état d'une personne) ?
        prompt2 = (
            f'Is "{lemma}" a French adjective describing a physical or '
            f'emotional state of a PERSON?\n'
            f'Examples YES: assis (seated), fatigué (tired), fâché (angry), '
            f'blessé (wounded), sûr (sure)\n'
            f'Examples NO: grand, vrai, beau, rapide, cuit, fini\n'
            f'Answer YES or NO only.'
        )
        for _attempt in range(3):
            try:
                payload = {'model': 'qwen2.5:0.5b', 'prompt': prompt2,
                           'stream': False, 'options': {'temperature': 0, 'num_predict': 3}}
                r = _req.post('http://localhost:11434/api/generate',
                              json=payload, timeout=15)
                raw2 = r.json().get('response', '').strip().upper()
                print(f"     🔍 qwen statif? '{raw2}'")
                if raw2.startswith('Y'):
                    self._statif_cache[lemma] = 'STATIF'
                    return 'STATIF'
                break
            except Exception as e:
                print(f"     🔍 attempt {_attempt+1} failed: {e}")
                time.sleep(1)

        # Question 3 : est-ce une valeur abstraite ?
        prompt3 = (
            f'Is "{lemma}" a French adjective expressing an abstract logical '
            f'truth or value (not a physical quality)?\n'
            f'Examples YES: vrai (true), faux (false), juste (correct)\n'
            f'Examples NO: grand, beau, rapide, assis, cuit\n'
            f'Answer YES or NO only.'
        )
        for _attempt in range(3):
            try:
                payload = {'model': 'qwen2.5:0.5b', 'prompt': prompt3,
                           'stream': False, 'options': {'temperature': 0, 'num_predict': 3}}
                r = _req.post('http://localhost:11434/api/generate',
                              json=payload, timeout=15)
                raw3 = r.json().get('response', '').strip().upper()
                print(f"     🔍 qwen valeur? '{raw3}'")
                if raw3.startswith('Y'):
                    self._statif_cache[lemma] = 'VALEUR'
                    return 'VALEUR'
                break
            except Exception as e:
                print(f"     🔍 attempt {_attempt+1} failed: {e}")
                time.sleep(1)

        self._statif_cache[lemma] = 'QUALITE'
        return 'QUALITE'
    
    # def _detect_statif_adj(self, lemma: str) -> str:
    #     """Retourne 'statif', 'participe' ou 'none'"""
    #     if not hasattr(self, '_statif_cache'):
    #         self._statif_cache = {}
    #     if lemma in self._statif_cache:
    #         return self._statif_cache[lemma]

    #     # prompt = (
    #     #     f'Answer with ONE word only: STATIF, PARTICIPE, VALEUR, or QUALITE.\n\n'
    #     #     f'STATIF = posture or emotion of a PERSON:\n'
    #     #     f'  assis (seated), fatigué (tired), sûr (sure/certain),\n'
    #     #     f'  fâché (angry), blessé (wounded), debout (standing)\n\n'
    #     #     f'PARTICIPE = result of an action on an OBJECT or FOOD:\n'
    #     #     f'  cuit (cooked), fini (finished), fermé (closed),\n'
    #     #     f'  cassé (broken), rempli (filled), brûlé (burned), ouvert (open)\n\n'
    #     #     f'VALEUR = abstract logical truth (no action, no person):\n'
    #     #     f'  vrai (true), faux (false), juste (correct), faux (wrong)\n\n'
    #     #     f'QUALITE = natural or physical quality:\n'
    #     #     f'  grand (tall), beau (beautiful), rapide (fast), lourd (heavy)\n\n'
    #     #     f'Classify "{lemma}": '
    #     # )
    #     prompt = (
    #         f'"{lemma}" - choose ONE:\n'
    #         f'STATIF (person state: assis, fatigué, fâché)\n'
    #         f'PARTICIPE (object result: cuit, fini, fermé, cassé)\n'
    #         f'VALEUR (truth: vrai, faux)\n'
    #         f'QUALITE (quality: grand, beau, rapide)\n'
    #         f'Answer:'
    #     )

    #     import time
    #     for _attempt in range(3):
    #         try:
    #             import requests as _req
    #             payload = {'model': 'qwen2.5:0.5b', 'prompt': prompt,
    #                        'stream': False, 'options': {'temperature': 0, 'num_predict': 5}}
    #             r = _req.post('http://localhost:11434/api/generate',
    #                           json=payload, timeout=120)
    #             raw = r.json().get('response', '').strip().upper().split()[0]
    #             print(f"     🔍 qwen raw response: '{raw}'")
    #             result = raw if raw in ('STATIF', 'PARTICIPE', 'VALEUR', 'QUALITE') else 'none'
    #             self._statif_cache[lemma] = result
    #             return result
    #         except Exception as e:
    #             print(f"     🔍 _detect_statif_adj attempt {_attempt+1} failed: {e}")
    #             time.sleep(2)
    #     self._statif_cache[lemma] = 'none'
    #     return 'none'
    
    def _load_semantic_classes(self) -> str:
        if not hasattr(self, '_semantic_class_cache'):
            try:
                res = self.db.query("""
                    MATCH (c:SemanticClass)
                    RETURN c.name AS name, c.description AS description
                    ORDER BY c.name
                """)
                self._semantic_class_cache = res if res else []
            except Exception:
                self._semantic_class_cache = []
        return self._semantic_class_cache

    def _detect_semantic_class(self, lemma: str, bm: str) -> str:
        try:
            res = self.db.query("""
                MATCH (s:Sense {bm: $bm})
                WHERE s.semantic_class IS NOT NULL
                RETURN s.semantic_class AS cls LIMIT 1
            """, {'bm': bm})
            if res and res[0].get('cls'):
                return res[0]['cls']
        except Exception:
            pass

        valid_names = {
            'motion', 'biological', 'posture', 'spontaneous',
            'perception', 'meteorological',
            'consumption', 'preparation',
            'action', 'technique', 'craft', 'communication',
            'copula', 'having', 'other',
        }

        prompt = (
            f'Classify this French verb "{lemma}" into ONE of these semantic classes:\n'
            f'- motion: marcher, courir, partir, aller, venir\n'
            f'- biological: dormir, respirer, grandir, vieillir\n'
            f'- posture: s\'asseoir, se lever, se coucher, se tenir\n'
            f'- spontaneous: rire, pleurer, tousser, éternuer\n'
            f'- perception: voir, entendre, sentir, ressentir\n'
            f'- consumption: manger, boire, avaler (sans objet précis)\n'
            f'- preparation: cuisiner, préparer, mijoter\n'
            f'- action: travailler, construire, creuser, frapper\n'
            f'- technique: tisser, forger, sculpter, réparer\n'
            f'- craft: coudre, tricoter, peindre, dessiner\n'
            f'- communication: parler, dire, chanter, crier\n'
            f'- copula: être (état, identité)\n'
            f'- having: avoir, posséder\n'
            f'- other: tout ce qui ne rentre pas dans les catégories ci-dessus\n\n'
            f'Reply with ONLY the class name, nothing else.'
        )

        try:
            cls = self._call_llm(prompt, max_tokens=5).strip().lower()
            cls = cls.split()[0] if cls else 'other'
            if cls not in valid_names:
                cls = 'other'
            try:
                self.db.query("""
                    MATCH (s:Sense {bm: $bm})
                    SET s.semantic_class = $cls
                """, {'bm': bm, 'cls': cls})
            except Exception:
                pass
            print(f"     🏷️  semantic_class('{lemma}') = {cls}")
            return cls
        except Exception as e:
            print(f"     ⚠️  semantic class detection failed: {e}")
            return 'other'
        
    # ------------------------------------------------------------------
    # TOKEN TRANSLATION
    # ------------------------------------------------------------------

    def _get_kg_label(self, spacy_pos: str) -> str | None:
        if not hasattr(self, '_pos_label_cache'):
            try:
                res = self.db.query("""
                    MATCH (m:PosMapping)
                    RETURN m.spacy AS spacy, m.kg_label AS kg_label
                """)
                self._pos_label_cache = {
                    r['spacy']: r['kg_label'] for r in res
                } if res else {}
            except Exception:
                self._pos_label_cache = {}
        return self._pos_label_cache.get(spacy_pos)

    def _translate_token(self, tok: dict, frame: str,
                         context_lemmas: list):
        surface = tok['surface']
        lemma   = tok['lemma']
        lang    = tok.get('lang', 'fr')
        pos     = tok['pos']

        if tok.get('bm') and tok.get('role') != 'interrogative':
            return tok, []

        if pos == 'PUNCT':
            return tok, []

        if pos == 'PROPN':
            tok['bm'] = tok.get('lemma') or surface
            return tok, []

        kg_label = self._get_kg_label(pos)

        if kg_label:
            try:
                res = self.db.query(f"""
                    MATCH (n:{kg_label})
                    WHERE toLower(n.surface) = toLower($surface)
                       OR toLower(n.lemma)   = toLower($lemma)
                       OR toLower(n.fr)      = toLower($lemma)
                    RETURN n.bm AS bm
                    ORDER BY
                        CASE WHEN toLower(n.surface) = toLower($surface)
                             THEN 0 ELSE 1 END
                    LIMIT 1
                """, {'surface': surface, 'lemma': lemma})
                if res and res[0].get('bm'):
                    tok['bm'] = res[0]['bm']
                    return tok, []
            except Exception as e:
                print(f"     ⚠️  KG label query failed ({kg_label}): {e}")

        structural_labels = {'Preposition', 'Article', 'Auxiliary'}
        if kg_label in structural_labels:
            # SÉCURISATION DES MOTS DE LIAISON SYNTAXIQUE (ZÉRO HARDCODE)
            # Si le mot est une préposition mais qu'il possède un rôle structurel fort
            # configuré par le KG, on lui interdit de quitter la fonction prématurément.
            # Il doit descendre jusqu'à la méthode retriever.retrieve pour charger sa glose !
            _ROLES_A_TRADUIRE = {'purposive', 'purpose', 'comitative', 'conjunction', 'interrogative'}
            
            if tok.get('role') not in _ROLES_A_TRADUIRE:
                return tok, []
            # S'il est dans la liste, on ignore le return et on le laisse continuer !

        if tok.get('role') == 'auxiliary':
            return tok, []

        sense_label  = self._get_kg_label('NOUN')
        is_sense_pos = kg_label == sense_label or kg_label == 'Sense'

        if not is_sense_pos:
            prompt = (
                f'Does the French {pos} word "{lemma}" have a '
                f'direct Bambara equivalent? '
                f'If yes reply with ONLY the Bambara word. '
                f'If no direct equivalent (like articles le/la/les) '
                f'reply with EMPTY. '
                f'Reply with ONE word or EMPTY.'
            )
            try:
                result = self._call_llm(prompt, max_tokens=8).strip()
                if result and result.upper() != 'EMPTY':
                    tok['bm'] = result
                    if kg_label:
                        try:
                            self.db.query(f"""
                                MERGE (n:{kg_label} {{fr: $lemma, lang: $lang}})
                                SET n.bm = $bm, n.surface = $surface,
                                    n.pos = $pos
                            """, {'lemma': lemma, 'lang': lang,
                                  'bm': result, 'surface': surface,
                                  'pos': pos})
                        except Exception:
                            pass
            except Exception as e:
                print(f"     ⚠️  LLM function word failed: {e}")
            return tok, []

        candidates = self.retriever.retrieve(
            lemma, frame,
            spacy_pos=tok['pos'],
            top_k=TOP_K,
            lang=lang,
            is_verbal_noun=tok.get('is_verbal_noun', False),
        )
        
        # ── AFFICHAGE DU TOP 5/6 DES CANDIDATS SENSE DU KG (DIAGNOSTIC VISUEL) ──
        print(f"\n     🔎 [TRANSLATION ENGINE] Jeton: '{surface}' | Lemme: '{lemma}' | POS: {pos}")
        if not candidates:
            print("        📭 Aucun candidat disponible dans la liste du moteur.")
        else:
            # On affiche les 5 ou 6 premiers candidats présents dans la pile finale de décision [S4]
            for idx, cand in enumerate(candidates[:6]):
                match_type = cand.get('match', 'unknown').upper()
                score = cand.get('final_score', 0.0)
                bm_glose = cand.get('bm', '[vide]')
                fr_sens = cand.get('fr', '[vide]')
                via_syn = f" (via synonyme: '{cand['via_synonym']}')" if 'via_synonym' in cand else ""
                
                print(f"        Rang #{idx+1} [{match_type}] Score: {score:.3f} | Bambara: '{bm_glose}' → Sens FR: \"{fr_sens}\"{via_syn}")
        print("     " + "="*65)

        candidates = self._rerank_with_llm(
            lemma, candidates, tok_pos=tok['pos'],
        )

        all_embed = bool(candidates) and all(
            c.get('match') == 'embed' for c in candidates)

        if self._needs_synonym(tok['lemma'], candidates, all_embed):
            syn_candidate = self._try_synonym_fallback(
                tok, frame, context_lemmas)
            if syn_candidate:
                candidates.append(syn_candidate)
                candidates.sort(key=lambda c: c['final_score'],
                                reverse=True)
            else:
                # Synonym validation failed — but if the top embed candidate
                # is strong enough, use it directly instead of a placeholder.
                # Typical case: 'end' in 'week-end' → lában (score=0.948).
                # The synonym check failed because synonyms were verbal forms
                # ('conclure') while the word is a noun.
                top_fallback = candidates[0] if candidates else None
                if top_fallback and top_fallback['final_score'] >= 0.90:
                    # Assign bm directly and return — don't fall through to
                    # the all_embed guards below which would re-reject it.
                    print(f" ↩️  No synonym matched — using top embed '{top_fallback['bm']}' (score={top_fallback['final_score']:.3f})")
                    tok['bm'] = top_fallback['bm']
                    # return tok, candidates
                else:
                    print(f"     📭 [{tok['lemma']}] — no valid translation found")
                    tok['bm'] = f"[{tok['lemma']}]"

                    # Détecter statif/participe avant de retourner
                    if tok['pos'] == 'ADJ':
                        # Utiliser le cache directement sans rappeler LLM
                        _result = getattr(self, '_statif_cache', {}).get(tok['lemma'])
                        if _result is None:
                            _result = self._detect_statif_adj(tok['lemma'])
                        _result_lower = str(_result).lower() if _result else ''
                        if _result_lower == 'statif' or _result is True:
                            tok['is_statif'] = True
                        elif _result_lower == 'participe':
                            tok['is_participe_passe'] = True
                            print(f"     🏷️  participe_passe('{tok['lemma']}') = True")
                        elif _result_lower == 'valeur':
                            tok['is_valeur'] = True

                    print(f"DEBUG après détection: tok flags = is_statif={tok.get('is_statif')}, is_participe_passe={tok.get('is_participe_passe')}")
                    return tok, candidates
                    # Ne pas retourner ici — continuer pour détecter is_statif

        top_score = candidates[0]['final_score'] if candidates else 0

        if all_embed and top_score < 0.75:
            gloss_match = any(
                tok['lemma'].lower() in c.get('fr', '').lower()
                for c in candidates[:5])
            if not gloss_match:
                print(f"     📭 [{tok['lemma']}] — embeddings too distant")
                tok['bm'] = f"[{tok['lemma']}]"
                return tok, candidates

        if not candidates:
            tok['bm'] = f"[{tok['lemma']}]"
            return tok, []

        best      = candidates[0]
        tok['bm'] = best['bm'] if best['final_score'] >= MIN_SCORE \
                    else f"[{tok['lemma']}]"

        if tok['pos'] == 'VERB' and tok.get('bm'):
            tok['semantic_class'] = self._detect_semantic_class(
                tok['lemma'], tok['bm'])
            _sc = tok.get('semantic_class', '')

            _morph_str = str(tok.get('morph', ''))
            _is_part_pass = ('VerbForm=Part' in _morph_str
                             and 'Voice=Pass' in _morph_str)
            if _is_part_pass:
                if _sc in ('posture', 'biological', 'spontaneous', 'consumption'):
                    _is_statif = True
                else:
                    try:
                        _is_statif = self._detect_statif_adj(tok['lemma'])
                    except Exception:
                        _is_statif = False
                if _is_statif:
                    tok['is_statif'] = True
                    tok['pos'] = 'ADJ'
                    print(f"     🏷️  participe_statif('{tok['lemma']}') = True")

            # Catégorie B1 : intransitif absolu → verbe nu
            if _sc in ('motion', 'biological', 'posture', 'spontaneous',
                       'perception', 'meteorological'):
                tok['intransitive_type'] = 'absolute'

            # Catégorie B2 : nom support dédié → action_noun + kɛ
            elif _sc in ('consumption', 'preparation'):
                tok['intransitive_type'] = 'support'
                # Récupérer le nom d'action depuis le KG
                try:
                    _res = self.db.query(
                        "MATCH (s:Sense {bm: $bm}) "
                        "RETURN s.action_noun AS an LIMIT 1",
                        {'bm': tok['bm']})
                    _an = _res[0].get('an') if _res and _res[0] else None
                except Exception:
                    _an = None
                if _an:
                    tok['action_noun'] = _an
                else:
                    # Fallback LLM : demander la forme nominale bambara
                    try:
                        _prompt = (
                            f'What is the Bambara action noun form of the verb "{tok["bm"]}" '
                            f'(French: "{tok["lemma"]}")? '
                            f'Example: dún (manger) → dúmúní, tobí (cuisiner) → tobíli. '
                            f'Reply with ONLY the Bambara noun form.'
                        )
                        _an = self._call_llm(_prompt, max_tokens=10).strip()
                        if _an and _an.upper() != 'EMPTY':
                            tok['action_noun'] = _an
                            # Cacher dans KG pour la prochaine fois
                            try:
                                self.db.query(
                                    "MATCH (s:Sense {bm: $bm}) SET s.action_noun = $an",
                                    {'bm': tok['bm'], 'an': _an})
                            except Exception:
                                pass
                    except Exception:
                        # Fallback morphologique en dernier recours
                        _v_bm = tok.get('bm', '')
                        if _v_bm:
                            tok['action_noun'] = (_v_bm + 'ni'
                                                  if _v_bm.endswith('n')
                                                  else _v_bm + 'li')

            # Catégorie B3 : nominalisation -li/-ni → V+li + kɛ
            elif _sc in ('action', 'technique', 'craft', 'communication'):
                tok['intransitive_type'] = 'nominalized'

        # if tok['pos'] == 'ADJ' and tok.get('bm'):
        #     _result = self._detect_statif_adj(lemma)
        #     if _result == 'statif' or _result is True:
        #         tok['is_statif'] = True
        #     elif _result == 'participe':
        #         tok['is_participe_passe'] = True

        if tok['pos'] == 'ADJ' and tok.get('bm'):
            _result = self._detect_statif_adj(lemma)
            if _result == 'statif' or _result is True:
                tok['is_statif'] = True
                # Si traduction KG absente ou mauvaise, demander au LLM
                if tok.get('bm', '').startswith('[') or all_embed:
                    try:
                        _p = (f'Translate the French verb "{lemma}" to Bambara. '
                              f'Reply with ONE Bambara word only.')
                        _bm = self._call_llm(_p, max_tokens=8).strip()
                        if _bm and not _bm.startswith('['):
                            tok['bm'] = _bm
                    except Exception:
                        pass
            elif _result == 'participe':
                tok['is_participe_passe'] = True
            elif _result == 'valeur':
                tok['is_valeur'] = True


        return tok, candidates

    # ------------------------------------------------------------------
    # CLAUSE SPLITTING
    # ------------------------------------------------------------------

    def _split_clauses(self, sentence: str) -> list:
        import re
        parts = re.split(r',|(?<=[a-zA-ZÀ-ÿ])\.(?=\s+[A-ZÀ-Ÿ]|\s*$)', sentence)
        parts = [p.strip().strip('.') for p in parts]
        return [p for p in parts if p and len(p.split()) > 1]

    # ------------------------------------------------------------------
    # SINGLE CLAUSE TRANSLATION
    # ------------------------------------------------------------------

    def _translate_clause(self, clause: str, frame: str) -> str:
        self._current_sentence = clause

        clause_vector = self.model.encode(clause)

        if hasattr(self.db, 'search_semantic_phrase'):
            global_match = self.db.search_semantic_phrase(clause_vector, threshold=0.85)
            if global_match:
                print(f"\n     🚀 GLOBAL SEMANTIC MATCH FOUND (KG):")
                print(f"     '{clause}'  ≈  '{global_match['fr']}'")
                print(f"     → Result: '{global_match['bm']}'")
                return global_match['bm']

        tokens = tokenize(clause, db=self.db,
                          backend=LLM_BACKEND, model=LLM_MODEL)
        if not tokens:
            return ''

        in_quotes = False
        for tok in tokens:
            surface = tok.get('surface', '')
            if '«' in surface or '"' in surface:
                in_quotes = True
            if in_quotes:
                tok['bm'] = surface
                tok['pos'] = 'PROPN'
                tok['is_protected'] = True
            if '»' in surface or '"' in surface:
                in_quotes = False

        concepts = []

        # Pré-détection statif en une seule passe avant traduction
        _adj_lemmas = [
            t['lemma'] for t in tokens
            if t.get('pos') == 'ADJ'
        ]
        for _lemma in _adj_lemmas:
            if _lemma not in getattr(self, '_statif_cache', {}):
                if not hasattr(self, '_statif_cache'):
                    self._statif_cache = {}
                self._statif_cache[_lemma] = self._detect_statif_adj(_lemma)

        for tok in tokens:
            surf = str(tok.get('surface', '')).strip().lower()
            pos = tok.get('pos', '')

            # 1. NETTOYAGE GÉNÉRIQUE DE PONCTUATION ET CARACTÈRES ORPHELINS (ZÉRO HARDCODE)
            # Si le jeton est classé en ponctuation, ou s'il s'agit d'un caractère isolé non-alphanumérique 
            # (comme une apostrophe droite, courbe, un tiret), on le vide et on passe immédiatement au suivant.
            if pos in ('PUNCT', 'SYM') or (len(surf) <= 1 and not surf.isalnum()):
                tok['bm'] = ''
                continue

            # 2. HARMONISATION GÉNÉRIQUE DES PRONOMS SUJETS SINGULIERS
            if pos == 'PRON' and surf == 'j':
                tok['bm'] = 'n'
                tok['role'] = 'pronoun'
                continue

            if pos == 'PRON' and surf in ("c'", 'ce', 'cela', 'ça') and tok.get('role') in ('expletive', 'pronoun'):
                tok['bm'] = 'o'
                continue

            # Exécution de la traduction unifiée du jeton si valide
            tok, candidates = self._translate_token(tok, frame, [])

        _has_purposive_acl = any(
            t.get('pos') == 'VERB'
            and t.get('role') == 'content'
            and t.get('dep') == 'acl'
            and any(
                tt.get('head_index') == t.get('orig_index')
                and tt.get('role') == 'purposive'
                for tt in tokens
            )
            for t in tokens
        )
        _has_conjunction = any(
            t.get('role') == 'conjunction' for t in tokens
        )

        if _has_purposive_acl and _has_conjunction:
            from pipeline.proposition_parser import translate_propositions
            bambara = translate_propositions(tokens)
        else:
            bambara = self.rule_engine.apply(tokens, frame)

        return bambara


    # ------------------------------------------------------------------
    # MAIN TRANSLATE
    # ------------------------------------------------------------------

    def translate(self, sentence: str) -> dict:
        self._current_sentence = sentence

        all_tokens = tokenize(sentence, db=self.db,
                              backend=LLM_BACKEND, model=LLM_MODEL)
        if not all_tokens:
            return {'bambara': '', 'frame': 'GENERIC', 'concepts': []}

        lang_tag = all_tokens[0].get('lang', 'fr').upper()
        lemmas   = [t['lemma'] for t in all_tokens if t['role'] == 'content']
        frame    = self.frame_parser.detect_frame(lemmas)

        print(f"\n🧠 INPUT : {sentence}  [{lang_tag}]")
        print(f"🔤 TOKENS: "
              f"{[(t['lemma'], t['pos'], t['dep'], t['role']) for t in all_tokens]}")
        print(f"📊 FRAME : {frame}")
        print("=" * 75)

        clauses      = self._split_clauses(sentence)
        all_bambara  = []
        all_concepts = []

        for i, clause in enumerate(clauses):
            if len(clauses) > 1:
                print(f"\n{'─'*40}")
                print(f"📌 CLAUSE {i+1}/{len(clauses)}: {clause}")
                print(f"{'─'*40}")

            self._current_sentence = clause
            bm = self._translate_clause(clause, frame)

            print(f"  ✂️  Clause {i+1} -> '{bm}'")

            if bm:
                all_bambara.append(bm)

        bambara_output = ', '.join(b for b in all_bambara if b)

        print(f"\n🇲🇱 BAMBARA : {bambara_output}")
        print("=" * 75)

        return {
            'bambara':  bambara_output,
            'frame':    frame,
            'concepts': all_concepts,
        }