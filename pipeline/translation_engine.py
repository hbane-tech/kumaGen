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
from rules import build_tree, tree_to_bambara, RuleEngine
from config.settings import LLM_BACKEND, LLM_MODEL, GEMINI_API_KEY, GEMINI_MODEL
import os
from embeddings.word2vec_encoder import _get_model

MIN_SCORE      = 12    # 0-100 point scale
TOP_K          = 10
ADJ_CONFIDENCE = 70    # 0-100 point scale
VERB_MIN_SCORE = 75    # 0-100 point scale

# Minimum embed top-score below which the embedding space is considered
# dead — synonyms from the same space won't rescue the search.
_EMBED_DEAD_ZONE = 50  # 0-100 point scale


class TranslationEngine:

    def __init__(self, db):
        self.db                = db
        self.retriever         = KGRetriever(db)
        self.frame_parser      = FrameParser(db)
        self.rule_engine       = RuleEngine(db)
        self._current_sentence = ''
        self.model = _get_model()
        self._warmup_llm()

    def _warmup_llm(self):
        """Premier appel léger pour charger le modèle en mémoire dès le démarrage.
        Retente une fois si le premier appel dépasse le timeout (modèle à froid)."""
        if LLM_BACKEND != 'ollama':
            return
        import requests as _req
        model = LLM_MODEL or 'qwen2.5:3b'
        payload = {'model': model, 'prompt': 'OUI', 'stream': False,
                   'options': {'num_predict': 1}}
        for attempt, wait in enumerate([90, 120], start=1):
            try:
                r = _req.post("http://localhost:11434/api/generate",
                              json=payload, timeout=wait)
                r.raise_for_status()
                print(f"✅  LLM Ollama prêt ({model})")
                return
            except _req.exceptions.ConnectionError:
                print("⚠️  Ollama non joignable — vérifier que 'ollama serve' tourne")
                return
            except _req.exceptions.Timeout:
                if attempt == 1:
                    print(f"⚠️  Ollama lent au démarrage (>{wait}s), nouvelle tentative...")
                else:
                    print(f"⚠️  Ollama timeout ({wait}s) — fonctionnement sans LLM")
            except Exception as e:
                print(f"⚠️  LLM Ollama erreur : {type(e).__name__}: {e}")
                return

    # ------------------------------------------------------------------
    # LLM CALLS
    # ------------------------------------------------------------------

    _llm_circuit_open_until: float = 0.0  # epoch time; 0 = circuit closed
    _LLM_TIMEOUT   = 90   # hard ceiling per Ollama call (seconds)
    _LLM_RETRY_GAP = 0    # retry immédiatement après chaque échec

    def _call_llm(self, prompt: str, max_tokens: int = 10, timeout: int = 8) -> str:  # noqa: ARG002
        """Generic LLM call — Ollama (qwen) uniquement."""
        import time as _time
        ollama_ok = _time.time() >= TranslationEngine._llm_circuit_open_until

        if ollama_ok and LLM_BACKEND == 'ollama':
            payload = {
                'model': LLM_MODEL or 'qwen2.5:3b',
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': 0, 'num_predict': max_tokens}
            }
            try:
                r = requests.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=TranslationEngine._LLM_TIMEOUT)
                return r.json()["response"].strip()
            except (requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectTimeout,
                    requests.exceptions.ConnectionError) as _e:
                TranslationEngine._llm_circuit_open_until = (
                    _time.time() + TranslationEngine._LLM_RETRY_GAP)
                print(f"⚠️  Ollama _call_llm échoué : {type(_e).__name__} — model={LLM_MODEL}")

        return ''


    def _call_llm_index(self, prompt: str):
        """Call LLM expecting a number — returns 0-based index or None."""
        resp  = self._call_llm(prompt, max_tokens=5)
        match = re.search(r'\d+', resp)
        return int(match.group()) - 1 if match else None

    # ------------------------------------------------------------------
    # SYNONYM NEED GUARD
    # ------------------------------------------------------------------

    def _enrich_token_context(self, tok: dict, all_tokens: list) -> None:
        """
        Use LLM to analyze grammatical context and enrich token with context tags.
        Modifies tok in-place by adding 'context_type' field.

        Examples:
        - "fille" in "la fille de Moussa" → context_type='genitive_object'
        - "médecin" in "Il est médecin" → context_type='predicate'
        - "chat" in "le chat noir" → context_type='modified_noun'
        """
        lemma = tok.get('lemma', '').lower().strip()
        if not lemma:
            return

        # Build sentence for context
        sentence = " ".join(t.get('surface', '') for t in all_tokens).strip()
        if not sentence:
            return

        # Find adjacent tokens for context clues
        tok_idx = next((i for i, t in enumerate(all_tokens) if t.get('orig_index') == tok.get('orig_index')), None)
        if tok_idx is None:
            return

        adjacent = []
        if tok_idx > 0:
            adjacent.append(all_tokens[tok_idx - 1].get('surface', ''))
        if tok_idx < len(all_tokens) - 1:
            adjacent.append(all_tokens[tok_idx + 1].get('surface', ''))
        adjacent_str = " ".join(adjacent).strip()

        # LLM analyzes grammatical role
        prompt = (
            f'Dans la phrase: "{sentence}"\n'
            f'Le mot "{lemma}" (entouré par: {adjacent_str}) a quel rôle grammatical?\n'
            f'Réponds UNIQUEMENT par UNE de ces catégories:\n'
            f'- genitive_object: complément de nom (X de Y)\n'
            f'- possessive: relation de possession\n'
            f'- predicate: attribut du sujet (Il est X)\n'
            f'- modified_noun: nom avec adjectif/modificateur\n'
            f'- agent: acteur d\'une action\n'
            f'- patient: objet d\'une action\n'
            f'- other: autre\n'
            f'Catégorie:'
        )

        try:
            resp = self._call_llm(prompt, max_tokens=3).strip().lower()
            # Extract category
            for cat in ['genitive_object', 'possessive', 'predicate', 'modified_noun', 'agent', 'patient', 'other']:
                if cat in resp:
                    tok['context_type'] = cat
                    print(f"     [CONTEXT] '{lemma}' → context_type={cat}")
                    return
        except Exception as e:
            print(f"     [CONTEXT ERROR] {lemma}: {e}")

    def _boost_scores_with_context(self, candidates: list,
                                   context_type: str) -> list:
        """
        Use LLM to validate if candidates match the grammatical context.
        Adds +10 pts boost if LLM confirms match, 0 pts otherwise.

        Example:
        - context_type='possessive' + gloss='fille, nièce' → LLM: YES → +10 pts
        - context_type='possessive' + gloss='femme' → LLM: NO → +0 pts
        """
        if not context_type or not candidates:
            return candidates

        for c in candidates[:5]:  # Only check top 5 for speed
            gloss = c.get('fr', '').lower().rstrip('.').strip()
            if not gloss:
                continue

            context_desc = {
                'possessive': 'a possessive/familial relationship (parent, child, sibling, relative, friend)',
                'genitive_object': 'a genitive relation of belonging (X de Y)',
                'agent': 'the agent/actor performing an action',
                'predicate': 'a profession, role, or predicate after "to be"',
                'modified_noun': 'a noun modified by an adjective',
                'patient': 'the object/patient receiving an action',
                'other': 'a general noun',
            }.get(context_type, 'a general noun')

            prompt = (
                f'Does the French word "{gloss}" represent {context_desc}?\n'
                f'Reply with YES or NO only.'
            )

            try:
                resp = self._call_llm(prompt, max_tokens=3).strip().upper()
                is_match = resp.startswith('Y')
                if is_match:
                    c['final_score'] = c.get('final_score', 0) + 10
                    print(f"     [CONTEXT BOOST] '{gloss}' → {context_type}: +10 pts")
                else:
                    print(f"     [CONTEXT SKIP] '{gloss}' → {context_type}: no match")
            except Exception as e:
                print(f"     ⚠️  Context boost failed for '{gloss}': {e}")

        # Re-sort by final_score after context boosts
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return candidates

    def _needs_synonym(self, tok_lemma: str, candidates: list,
                       all_embed: bool) -> bool:
        """
        Decide whether synonym fallback should be attempted (0-100 point scale).

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
        - Top score < 70 pts
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
            print(f"     [{tok_lemma}] — embed dead zone "
                  f"(score={top_score:.1f} < {_EMBED_DEAD_ZONE}), "
                  f"skipping synonym fallback")
            return False

        # Low confidence → try synonym ONLY if top_score very low
        # Raised threshold from 70 to 50 pts to reduce bad synonyms
        # (e.g., "belle" for "gentil" - different semantic field)
        if top_score < 50:
            return True

        # All embedding with very low score → try synonym as last resort
        if all_embed and top_score < 60:
            return True

        return False

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # SENS_FR RE-RANKING (déterministe, sans LLM)
    # ------------------------------------------------------------------

    def _rerank_by_sens_fr(self, source_lemma: str, candidates: list) -> list:
        """
        Re-classe les candidats en vérifiant si le lemme source apparaît
        dans le sens_fr du candidat (KG). Deux passes sur 0-100 point scale.

        1. String check (gratuit) :
           - Lemme comme MOT dans sens_fr → boost fort (+35 pts)
           - Lemme comme sous-chaîne     → boost modéré (+15 pts)
           - Pas de match                → pas de boost

        2. Embedding check (seulement si top-1 sans match string, embed-only) :
           - cosine_sim(encode(lemme), encode(sens_fr)) * 20 → boost fin

        Correspond directement à la métrique Embedding P@1 du paper.
        """
        import numpy as np

        lemma_lower = source_lemma.lower().strip()
        if not lemma_lower or not candidates:
            return candidates

        # ── Passe 1 : string match sens_fr ────────────────────────────
        # IMPORTANT: These boosts are MINIMAL to avoid collapsing the hierarchy.
        # Boosts should only slightly reorder ties, not change categorical ranking.
        for c in candidates:
            raw = c.get('fr', '')
            sens = raw.lower().replace(',', ' ').replace('.', ' ').replace(';', ' ')
            words = set(sens.split())
            if lemma_lower in words:
                c['_sens_boost'] = 3       # mot entier → minimal signal
            elif lemma_lower in raw.lower():
                c['_sens_boost'] = 1       # sous-chaîne → minimal signal
            else:
                c['_sens_boost'] = 0

        # ── Passe 2 : embedding sens_fr (seulement si top sans match) ─
        # cosine(lemme, glose) est CIRCULAIRE dans cet espace : il sert
        # uniquement à départager l'ordre des candidats embed, JAMAIS à
        # gonfler la confiance (final_score). Stocké à part → le seuil de
        # rerank/synonyme ne se laisse pas berner par un faux ami à 0.97.
        top = candidates[0]
        if top.get('_sens_boost', 0) == 0 and top.get('match') == 'embed':
            try:
                src_vec = self.model.encode(lemma_lower)
                src_norm = np.linalg.norm(src_vec)
                for c in candidates[:5]:
                    if c.get('match') == 'embed' and c.get('fr'):
                        sv = self.model.encode(c['fr'])
                        sim = float(np.dot(src_vec, sv) /
                                    (src_norm * np.linalg.norm(sv) + 1e-8))
                        c['_embed_rank_boost'] = sim * 2  # 0-100 scale (minimal)
            except Exception:
                pass

        # ── Application des boosts et re-tri ──────────────────────────
        # Seul le boost string (passe 1) entre dans final_score (confiance).
        # Le boost cosinus (passe 2) n'agit que sur la clé de tri.
        for c in candidates[:10]:
            old_score = c.get('final_score', 0)
            c['final_score'] = c['final_score'] + c.get('_sens_boost', 0)
            embed_boost = c.get('_embed_rank_boost', 0)
            # DEBUG
            if c.get('_sens_boost', 0) != 0 or embed_boost != 0:
                print(f"     [RERANK_SENS] {c.get('bm', '?')}: {old_score:.1f} + {c.get('_sens_boost', 0)} (sens) + {embed_boost:.1f} (embed) = {c['final_score']:.1f}")

        return sorted(
            candidates,
            key=lambda x: -(x['final_score'] + x.get('_embed_rank_boost', 0)))

    # LLM CONTEXTUAL RERANKING
    # ------------------------------------------------------------------

    def _rerank_with_llm(self, token_lemma: str,
                         candidates: list,
                         tok_pos: str = 'NOUN',
                         context_tokens: list = None) -> list:
        """
        Use LLM to pick the best candidate sense (0-100 point scale).

        Triggers:
        1. Multiple exact matches with close scores AND simpler exists
        2. All embedding results with low confidence (< 75 pts)
        3. VERB with multiple exact matches very close (<=5 pts gap)
           AND top candidate has compound gloss
        """
        if len(candidates) < 2:
            return candidates

        top       = candidates[0]
        top_score = top['final_score']
        all_embed = all(c.get('match') == 'embed' for c in candidates)
        # Un score élevé ne court-circuite le rerank que pour un match EXACT.
        # Un top 'embed' à score élevé est souvent un faux ami (espace
        # dégénéré : "monté" → fɔ/dire à 0.97) → toujours passer par le LLM.
        if top_score >= 80 and not all_embed:
            return candidates
        # Exception : embed très haute confiance (score bien au-dessus du max
        # cosinus 85 pts grâce aux bonus frame+gloss) → le LLM ne peut pas faire mieux.
        # Le seuil 105 ne se déclenche pas pour les faux amis à 85 pts.
        if all_embed and top_score >= 105:
            return candidates

        exact_matches = [c for c in candidates if c.get('match') == 'exact']
        close         = [c for c in candidates
                         if top_score - c['final_score'] <= 10]
        top_fr_words  = len(top['fr'].strip().rstrip('.').split())
        has_simpler   = any(
            len(c['fr'].strip().rstrip('.').split()) < top_fr_words
            for c in candidates[1:]
        )
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
            and (top_score - second_exact_score) <= 5
            and (top_fr_words >= 2 or context_match_in_lower)
        )

        should_rerank = (
            (len(exact_matches) >= 2 and len(close) >= 2 and has_simpler)
            or all_embed   # tout embed → score peu fiable, toujours reranker
            or is_context_sensitive
        )

        if not should_rerank:
            return candidates

        if all_embed:
            rerank_pool = candidates
        elif is_context_sensitive:
            rerank_pool = [c for c in candidates
                           if top_score - c['final_score'] <= 15]
        else:
            rerank_pool = [c for c in candidates
                           if top_score - c['final_score'] <= 10]

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
    # SEMANTIC VALIDATION OF KG CANDIDATES
    # ------------------------------------------------------------------

    def _validate_candidate_semantics(self, token_fr: str,
                                      candidates: list,
                                      top_k: int = 5) -> list:
        """
        Filter out false positives using LLM semantic validation.

        Ultra-simple: Only penalize candidates that are semantically invalid.
        - Text match or embedding ≥50 pts: validate with LLM
        - Valid: keep as-is
        - Invalid: penalize -30 pts (drop below 50 pts)
        """
        if not candidates or not token_fr:
            return candidates

        token_lower = token_fr.lower().strip()

        for c in candidates[:top_k]:
            score = c.get('score', 0)

            # Perfect exact (100 pts) — trust it, no validation needed
            if score >= 100:
                continue

            # Composite/substring (50+ pts) — validate with LLM
            if score >= 50:
                gloss_fr = c.get('fr', '').lower().rstrip('.').strip()

                # Text contains token — trust it
                if token_lower in gloss_fr or gloss_fr.startswith(token_lower):
                    print(f"     ✅ Match textuel: '{gloss_fr}' ≈ '{token_lower}' → {c['bm']}")
                    continue

                # Ask LLM: is gloss semantically valid for token?
                prompt = (
                    f'Les expressions françaises \"{gloss_fr}\" et \"{token_lower}\" '
                    f'ont-elles à peu près la même signification?\n'
                    f'Réponds uniquement par OUI ou NON.'
                )

                try:
                    resp = self._call_llm(prompt, max_tokens=3).strip().upper()
                    is_valid = resp.startswith('O')

                    if is_valid:
                        print(f"     ✅ LLM valide: '{gloss_fr}' ≈ '{token_lower}' → {c['bm']}")
                    else:
                        # Penalize false positives heavily
                        c['score'] = max(0, score - 30)
                        c['final_score'] = c['score']
                        print(f"     ⚠️  LLM invalide: '{gloss_fr}' ≠ '{token_lower}' → {c['bm']} (-30 pts)")

                except Exception as e:
                    print(f"     ⚠️  LLM validation failed: {e}")
                    # Conservative: penalize on LLM failure
                    c['score'] = max(0, score - 15)
                    c['final_score'] = c['score']

        # Re-sort by score
        candidates.sort(key=lambda x: x.get('final_score', x.get('score', 0)), reverse=True)
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
            f'Donne 3 {pos_label} français qui signifient EXACTEMENT la même chose que "{lemma}".\n'
            f'Synonymes VRAIS seulement, pas de sens différent.\n'
            f'Ne PAS inclure "{lemma}" lui-même.\n'
            f'Réponds UNIQUEMENT par 3 mots séparés par des virgules.'
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

    def _detect_possession_type(self, lemma: str, semantic_class: str = '') -> str:
        """
        Classify noun for 'avoir' construction.
        Returns: AGE / MATERIAL / ABSTRACT / PAIN
        """
        if not lemma:
            return 'ABSTRACT'

        # Classification via semantic_class KG si disponible
        _age_classes = {'time', 'duration', 'age'}
        _material_classes = {'object', 'tool', 'vehicle', 'building',
                            'money', 'food_item', 'clothing', 'furniture'}
        _abstract_classes = {'feeling', 'emotion', 'biological_state',
                            'sensation', 'mental_state', 'physiological'}
        _pain_classes = {'pain', 'illness', 'disease', 'symptom'}

        if semantic_class in _age_classes:
            return 'AGE'
        if semantic_class in _material_classes:
            return 'MATERIAL'
        if semantic_class in _abstract_classes:
            return 'ABSTRACT'
        if semantic_class in _pain_classes:
            return 'PAIN'

        # Fallback LLM
        prompt = (
            f"Le nom français '{lemma}' dans la construction 'avoir + {lemma}' "
            f"appartient à quelle catégorie ?\n"
            f"AGE : notion de temps, d'âge, d'années (âge, ans, siècle)\n"
            f"MATERIAL : objet physique concret possédable (maison, voiture, téléphone, argent, clé)\n"
            f"ABSTRACT : état interne biologique ou psychologique non palpable "
            f"(faim, soif, peur, honte, chance, envie, idée, confiance)\n"
            f"PAIN : douleur physique ou maladie (mal, douleur, fièvre, migraine)\n"
            f"Réponds UNIQUEMENT par : AGE, MATERIAL, ABSTRACT ou PAIN"
        )

        for attempt in range(3):
            try:
                result = self._call_llm(prompt, max_tokens=5).strip().upper()
                if result in ('AGE', 'MATERIAL', 'ABSTRACT', 'PAIN'):
                    return result
            except Exception:
                continue

        # Fallback timeout → ABSTRACT (plus sûr)
        return 'ABSTRACT'

    def _detect_statif_adj(self, lemma: str) -> str:
        prompt = (
            f'The French adjective "{lemma}" used predicatively — which Bambara construction?\n'
            f'QUALITE: permanent quality, physical property, or RELATIONAL property '
            f'that describes the subject. '
            f'Examples: grand, beau, fort, rapide, rouge, intelligent, '
            f'égal, semblable, différent, pareil, équivalent. '
            f'Bambara: subject + ka + adjective.\n'
            f'STATIF: temporary emotional or epistemic state the subject has entered. '
            f'Examples: fatigué, content, triste, prêt, malade, inquiet, libre, occupé, '
            f'sûr, certain, convaincu, conscient. '
            f'Bambara: adjective + -len/-nen dòn.\n'
            f'VALEUR: abstract truth-value presented as a FACT/NOUN (you would say '
            f'"c\'est la X"), NOT an adjective describing the subject. '
            f'Examples: vrai, faux, réel. '
            f'Bambara: noun dòn (presentative).\n'
            f'PARTICIPE: state resulting from a past action done to the subject. '
            f'Examples: blessé, fermé, cassé, ouvert, cuit. '
            f'Bambara: verb + -ra/-la/-na.\n'
            f'Reply with ONLY one word: QUALITE, STATIF, VALEUR, or PARTICIPE.'
        )

        for _attempt in range(3):
            try:
                raw = self._call_llm(prompt, max_tokens=5).strip().upper().split()[0]
                print(f"     🔍 LLM classify? '{raw}'")
                result = raw if raw in ('STATIF', 'PARTICIPE', 'QUALITE', 'VALEUR') else 'QUALITE'
                return result
            except Exception as e:
                print(f"     🔍 attempt {_attempt+1} failed: {e}")

        return 'QUALITE'

    def _classify_adj_state(self, tok, all_embed=False):
        """Classe un ADJ prédicatif en STATIF/PARTICIPE/VALEUR/QUALITE et pose
        le flag correspondant (QUALITE → aucun flag : qualitative par défaut).

        Même logique que les branches inline, mais appelable aussi sur les
        chemins où bm est déjà fourni par le KG : sans ça, l'early-return
        sautait la classification (ex: 'capable' → sénkola → jamais classé).
        """
        if tok.get('pos') != 'ADJ':
            return
        # Déjà classé en amont → ne pas relancer le LLM
        if (tok.get('is_statif') or tok.get('is_participe_passe')
                or tok.get('is_valeur')):
            return
        _clause_toks  = getattr(self, '_current_clause_tokens', [])
        _passive_subj = any(t.get('dep') == 'nsubj:pass' for t in _clause_toks)
        _has_obl_arg  = (tok.get('dep') == 'ROOT' and any(
            t.get('dep') == 'obl:arg'
            and t.get('head_index') == tok.get('orig_index')
            for t in _clause_toks))
        if _has_obl_arg:
            _result = 'STATIF'
        else:
            _result = self._detect_statif_adj(tok.get('lemma', ''))
            if str(_result).upper() == 'QUALITE' and _passive_subj and tok.get('dep') == 'ROOT':
                _result = 'PARTICIPE'
        _result_norm = str(_result).upper() if _result else ''
        if _result_norm == 'STATIF' or _result is True:
            tok['is_statif'] = True
            if all_embed:
                tok['bm'] = f"[{tok.get('lemma')}]"
        elif _result_norm == 'PARTICIPE':
            tok['is_participe_passe'] = True
        elif _result_norm == 'VALEUR':
            tok['is_valeur'] = True

    def _detect_reflexive_type(self, lemma: str) -> str:
        """Classifie le type de construction réflexive du verbe.
        Retourne RECIPROCAL, REFLEXIVE, PASSIVE ou IDIOMATIC."""
        prompt = (
            f'Verb: "{lemma}". Used with reflexive "se".\n'
            f'RECIPROCAL: two or more participants perform the action on each other '
            f'(meet, fight, kiss, marry, see each other).\n'
            f'REFLEXIVE: the action is intentionally directed back to the subject as an object; '
            f'the subject consciously acts on themselves '
            f'(hurt oneself, blame oneself, judge oneself, examine oneself, punish oneself).\n'
            f'PASSIVE: "se" has no semantic role; the subject undergoes the action '
            f'or the construction is impersonal/passive '
            f'(be sold, be called, be done, happen, be used).\n'
            f'IDIOMATIC: includes body-care/grooming actions (wash, dress, shave, comb, '
            f'bathe, dry oneself), AND verbs that require "se" to express a state, change '
            f'of state, emotion, cognition, movement, or fixed meaning '
            f'(realize, remember, get angry, hurry, leave, wonder, concentrate, '
            f'make a mistake, get up, lie down, sit down, get dressed, get bored).\n'
            f'Return one word only: RECIPROCAL, REFLEXIVE, PASSIVE, or IDIOMATIC.'
        )
        for _attempt in range(3):
            try:
                raw = self._call_llm(prompt, max_tokens=5).strip().upper()
                parts = raw.split()
                if not parts:
                    raise ValueError('empty response')
                raw = parts[0]
                print(f"     🔍 reflexive_type? '{raw}'")
                result = raw if raw in ('RECIPROCAL', 'REFLEXIVE', 'PASSIVE', 'IDIOMATIC') else 'IDIOMATIC'
                return result
            except Exception as e:
                print(f"     🔍 attempt {_attempt+1} failed: {e}")
        return 'IDIOMATIC'

    def _load_semantic_classes(self) -> str:
        try:
            res = self.db.query("""
                MATCH (c:SemanticClass)
                RETURN c.name AS name, c.description AS description
                ORDER BY c.name
            """)
            return res if res else []
        except Exception:
            return []

    def _detect_relational_noun(self, lemma: str, bm: str) -> bool:
        """
        Détermine si un nom est INALIENABLE en bambara → pas de 'ka'.
        Principe linguistique : possession inalienable = relation constitutive et
        indissociable entre possesseur et possédé. Possession aliénable = le possédé
        existe indépendamment du possesseur et peut en être séparé → 'ka'.
        Retourne True  → inalienable → pas de 'ka'
        Retourne False → aliénable   → 'ka' requis
        Fallback : True (pas de 'ka') — plus sûr grammaticalement.
        """
        if not lemma or not bm:
            return True

        prompt = (
            f"En bambara, RÈGLE ABSOLUE : deux entités de même nature ne prennent JAMAIS 'ka'.\n"
            f"Les relations de parenté, de famille, et les relations sociales entre personnes "
            f"sont toujours INALIENABLES (sans 'ka') : père, mère, frère, sœur, fils, fille, "
            f"oncle, tante, cousin, grand-père, grand-mère, mari, femme, enfant, ami, ennemi, "
            f"voisin, collègue, patron, etc.\n"
            f"Les parties du corps sont aussi INALIENABLES (sans 'ka') : tête, bras, jambe, main, etc.\n"
            f"Seuls les objets physiques SÉPARABLES et TRANSFÉRABLES prennent 'ka' (possession ALIÉNABLE) : "
            f"maison, voiture, livre, vêtement, argent, champ, outil, etc.\n"
            f"Le mot français '{lemma}' représente-t-il une relation INALIENABLE (OUI) "
            f"ou un objet ALIÉNABLE (NON) ?\n"
            f"Réponds UNIQUEMENT par : OUI ou NON"
        )
        for _ in range(2):
            result_str = self._call_llm(prompt, max_tokens=5).strip().upper()
            if 'OUI' in result_str:
                print(f"  🔗 [RELATIONAL] '{lemma}' ({bm}) → INALIENABLE (sans 'ka')")
                return True
            if 'NON' in result_str:
                print(f"  📦 [RELATIONAL] '{lemma}' ({bm}) → ALIÉNABLE (avec 'ka')")
                return False

        # LLM indisponible → pas de 'ka' par défaut (plus sûr grammaticalement)
        print(f"  ❓ [RELATIONAL] '{lemma}' ({bm}) → LLM indisponible, pas de 'ka' par défaut")
        return True

    def _detect_intransitive_type(self, lemma: str, semantic_class: str = '') -> str:

        # ── PRIORITÉ CLASSE SÉMANTIQUE : classes autonomes → ABSOLU ───────────
        # Les classes sémantiques autonomes (spontaneous, motion, posture…) sont
        # intrinsèquement intransitives : leur sens central ne porte pas sur un
        # objet direct, même si la grammaire le permet ('perdre ses clés').
        # On respecte cette classe AVANT d'interroger le LLM (qui répondrait
        # ACTION sur la simple possibilité grammaticale d'un COD).
        # Aligné sur _AUTONOMOUS_SC (advcl.py) et la logique B1 (intransitif absolu).
        _AUTONOMOUS_SC = {'motion', 'biological', 'posture', 'spontaneous',
                          'perception', 'meteorological'}
        if semantic_class in _AUTONOMOUS_SC:
            print(f"  🔍 [TRANSITIVITY] '{lemma}' class={semantic_class} "
                  f"→ ABSOLU (classe autonome, LLM ignoré)")
            return 'ABSOLU'

        # ── LLM ──────────────────────────────────────────────────────────────
        # On demande l'usage COURANT (pas la possibilité grammaticale) : 'travailler'
        # peut grammaticalement avoir un COD (travailler le bois) mais s'emploie
        # habituellement sans objet → ABSOLU. Les verbes cités sont de simples
        # exemples illustratifs, pas une liste exhaustive.
        prompt = (
            f"Le verbe français '{lemma}', dans son usage le plus COURANT, "
            f"s'emploie-t-il avec un objet direct ?\n"
            f"ACTION = habituellement AVEC un objet direct, "
            f"par exemple : manger, voir, prendre, lire, boire…\n"
            f"ABSOLU = habituellement SANS objet, intransitif, "
            f"par exemple : dormir, travailler, partir, courir, parler…\n"
            f"Réponds UNIQUEMENT par : ACTION ou ABSOLU"
        )
        _result = None
        for attempt in range(3):
            try:
                _raw = self._call_llm(prompt, max_tokens=5).strip().upper()
                print(f"  🔬 [TRANSITIVITY raw] attempt {attempt+1}: {_raw!r}")
                if 'ACTION' in _raw:
                    _result = 'ACTION'; break
                if 'ABSOLU' in _raw:
                    _result = 'ABSOLU'; break
            except Exception:
                continue

        verdict = _result
        print(f"  🔍 [TRANSITIVITY] LLM verdict for '{lemma}' (class={semantic_class}) → {verdict}  [raw: {_raw!r}]")
        return verdict
    
    
    def _normalize_verb_to_infinitive(self, verb: str) -> str:
        """Normalise un verbe conjugué à sa forme infinitive.

        Utilise le LLM pour une conversion fiable (sans hardcode).
        Exemples: 'lave' → 'laver', 'mangé' → 'manger', 'vais' → 'aller'
        """
        verb_lower = verb.lower().strip()
        if not verb_lower:
            return verb

        prompt = (
            f'Trouve la forme infinitive du verbe français "{verb}".\n\n'
            f'Exemples:\n'
            f'- lave, laves, lavent → laver\n'
            f'- mange, manges, mangent → manger\n'
            f'- viens, venons, vient → venir\n'
            f'- suis, sommes, êtes, sont → être\n'
            f'- ai, avons, avez, ont → avoir\n\n'
            f'Réponds UNIQUEMENT par l\'infinitif (un seul mot), minuscules, sans ponctuation.'
        )

        try:
            infinitive = self._call_llm(prompt, max_tokens=8).strip().lower()
            if infinitive and len(infinitive) > 1:
                return infinitive
        except Exception as e:
            print(f"     ⚠️  Infinitive normalization failed: {e}")

        return verb

    def _detect_semantic_class(self, lemma: str) -> str:
        """Détecte la classe sémantique d'un verbe via LLM.

        Le LLM classe le verbe en autonome ou transitif basé sur sa sémantique.
        """
        prompt = (
            f'Quelle est la nature sémantique du verbe français "{lemma}" ?\n\n'
            f'Catégories AUTONOMES (intransitifs, n\'acceptent pas de COD direct):\n'
            f'  motion=déplacement dans l\'espace (aller, venir, courir, marcher...)\n'
            f'  biological=processus vital du corps (vivre, mourir, naître, respirer...)\n'
            f'  posture=position/changement de position (rester, dormir, se lever...)\n'
            f'  spontaneous=réaction involontaire (rire, crier, pleurer...)\n'
            f'  perception=voir, entendre, sentir (perception directe)\n'
            f'  meteorological=phénomène atmosphérique (pleuvoir, neiger...)\n'
            f'  copula=lien attributif (être, sembler, paraître...)\n\n'
            f'Catégories TRANSITIVES (acceptent souvent un COD):\n'
            f'  action=action intentionnelle (faire, donner, prendre, manger...)\n'
            f'  consumption=ingestion (manger, boire...)\n'
            f'  preparation=transformation (cuisiner, préparer...)\n'
            f'  technique=travail spécialisé (construire, réparer...)\n'
            f'  craft=création artistique (peindre, écrire...)\n'
            f'  communication=parole (dire, raconter, demander...)\n'
            f'  having=possession (avoir, posséder...)\n\n'
            f'  other=aucune catégorie ne convient.\n\n'
            f'Réponds UNIQUEMENT par le nom de la catégorie.'
        )

        try:
            import re as _re
            cls_raw = self._call_llm(prompt, max_tokens=15).strip().lower()
            words   = _re.findall(r'[a-z]+', cls_raw)
            cls     = words[0] if words else 'other'
            print(f"     🏷️  semantic_class('{lemma}') = {cls}  [LLM]")
            return cls
        except Exception as e:
            print(f"     ⚠️  semantic class detection failed: {e}")
            return 'other'
        
    # ------------------------------------------------------------------
    # TOKEN TRANSLATION
    # ------------------------------------------------------------------

    def _get_kg_label(self, spacy_pos: str):
        try:
            res = self.db.query("""
                MATCH (m:PosMapping)
                RETURN m.spacy AS spacy, m.kg_label AS kg_label
            """)
            mapping = {r['spacy']: r['kg_label'] for r in res} if res else {}
        except Exception:
            mapping = {}
        return mapping.get(spacy_pos)

    def _translate_token(self, tok: dict, frame: str,
                         context_lemmas: list, all_tokens: list = None):
        surface = tok['surface']
        lemma   = tok['lemma']
        lang    = tok.get('lang', 'fr')
        pos     = tok['pos']

        # DEBUG: Track which tokens are processed
        print(f"     [TRANSLATE_TOKEN] surface='{surface}' lemma='{lemma}' pos={pos} dep={tok.get('dep')}")

        if tok.get('bm'):
            if pos == 'NOUN' and not tok['bm'].startswith('[') and 'is_relational' not in tok:
                tok['is_relational'] = self._detect_relational_noun(tok['lemma'], tok['bm'])
            elif pos == 'ADJ':
                self._classify_adj_state(tok)
            return tok, []

        if pos == 'PUNCT':
            return tok, []
        # ── NÉGATION : ne pas assigner de bm aux tokens de négation ──────────
        # ne/n'/pas/jamais/plus/rien → role='negation' ou dans neg_surfaces KG
        # Ces tokens doivent rester bm='' pour ne pas parasiter les obliques
        _surf_neg = str(surface).lower().rstrip("'").rstrip('\u2019').rstrip('\u2018')
        _neg_surfs = self.rule_engine.grammar.get('neg_surfaces', set())
        if tok.get('role') == 'negation' or _surf_neg in _neg_surfs:
            tok['bm'] = ''
            return tok, []

        if pos == 'PROPN':
            tok['bm'] = tok.get('lemma') or surface
            return tok, []

        # ── PRÉ-CONTRÔLE SYNTAXIQUE : rôle nominal vs adjectif ───────────────
        # Pour les ADJ ROOT dans une construction copulative avec expletif (c'est):
        #   - ADJ avec det enfant (c'est le vrai)  → rôle nominal → is_nominal_adj=True
        #   - ADJ sans det                         → rôle adjectif (qualité ou valeur)
        if pos == 'ADJ' and tok.get('dep') == 'ROOT':
            _ctoks_pre = getattr(self, '_current_clause_tokens', [])
            _has_expl_pre = any(t.get('role') == 'expletive' for t in _ctoks_pre)
            if _has_expl_pre:
                _has_det_on_adj = any(
                    t.get('dep') == 'det'
                    and t.get('head_index') == tok.get('orig_index')
                    and t.get('role') not in ('expletive',)
                    for t in _ctoks_pre
                )
                if _has_det_on_adj:
                    tok['is_nominal_adj'] = True

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
                    if pos == 'NOUN' and not tok['bm'].startswith('['):
                        tok['is_relational'] = self._detect_relational_noun(tok['lemma'], tok['bm'])
                    elif pos == 'ADJ':
                        self._classify_adj_state(tok)
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

        # ── NORMALISER LES VERBES À L'INFINITIF AVANT KG RETRIEVAL ─────────────
        # Les verbes dans le KG sont à l'infinitif ('laver', pas 'lave')
        kg_search_lemma = lemma
        if tok['pos'] == 'VERB' and lemma:
            # Convertir en infinitif pour une meilleure recherche KG
            # "lave" → "laver", "mangé" → "manger", etc.
            infinitive = self._normalize_verb_to_infinitive(lemma)
            if infinitive and infinitive != lemma:
                kg_search_lemma = infinitive
                print(f"     🔄 Verbe normalisé: '{lemma}' → '{kg_search_lemma}'")

        # Enrich token with grammatical context (LLM analysis)
        if all_tokens:
            self._enrich_token_context(tok, all_tokens)
            if tok.get('context_type'):
                print(f"     [CONTEXT] '{lemma}' → context_type={tok['context_type']}")
            else:
                print(f"     [CONTEXT] '{lemma}' → NONE (LLM may have failed)")

        candidates = self.retriever.retrieve(
            kg_search_lemma, frame,
            spacy_pos=tok['pos'],
            top_k=TOP_K,
            lang=lang,
            context_tokens=context_lemmas,
            is_verbal_noun=tok.get('is_verbal_noun', False),
        )

        # ── CONTEXT-AWARE BOOSTING: LLM validates grammatical match ──
        # If token has a context_type, boost scores for matching candidates
        if tok.get('context_type') and candidates:
            candidates = self._boost_scores_with_context(
                candidates, tok['context_type'])

        # ── SEMANTIC VALIDATION: Filter out false positives ──
        # LLM checks if candidate gloss actually matches the token semantically
        candidates = self._validate_candidate_semantics(lemma, candidates, top_k=10)

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
                
                print(f"        Rang #{idx+1} Score: {score:.1f} pts | Bambara: '{bm_glose}' → Sens FR: \"{fr_sens}\"{via_syn}")
        print("     " + "="*65)

        candidates = self._rerank_with_llm(
            lemma, candidates, tok_pos=tok['pos'],
        )

        all_embed = bool(candidates) and all(
            c.get('match') == 'embed' for c in candidates)

        # SYNONYM FALLBACK DISABLED: Use KG/Embedding results as-is
        # Synonyms were creating bad matches like 'belle' for 'gentil'
        # Prefer placeholder over wrong synonym

        # Fallback NOUN pour les ADJ prédicatifs (professions/rôles)
        # ex: "je suis étudiant" → spaCy=ADJ, mais KG a kàlandenba (NOUN)
        if tok.get('pos') == 'ADJ' and tok.get('dep') == 'ROOT':
            _cop_context = any(
                t.get('dep') == 'cop'
                for t in getattr(self, '_current_clause_tokens', []))
            if _cop_context:
                # Classifier d'abord : STATIF/PARTICIPE/QUALITE
                # NOUN fallback disabled — interferes with correct adjective ranking
                self._classify_adj_state(tok)
                # if not tok.get('is_participe_passe') and not tok.get('is_statif'):
                #     _noun_cands = self.retriever.retrieve(
                #         lemma, frame, spacy_pos='NOUN',
                #         top_k=TOP_K, lang=lang)
                #     _noun_cands = self._rerank_by_sens_fr(lemma, _noun_cands)
                #     if _noun_cands and _noun_cands[0]['final_score'] >= 50:
                #         _best_n = _noun_cands[0]
                #         tok['bm'] = _best_n['bm']
                #         tok['_adj_is_nominal_pred'] = True
                #         print(f"     🔄 [NOUN fallback] '{lemma}' → '{_best_n['bm']}' ({_best_n['fr']})")
                #         return tok, _noun_cands
        # Détecter statif/participe AVANT le return
        if tok['pos'] == 'ADJ':
            _clause_toks2 = getattr(self, '_current_clause_tokens', [])
            _passive_subj = any(t.get('dep') == 'nsubj:pass' for t in _clause_toks2)
            _has_obl_arg2 = (tok.get('dep') == 'ROOT' and any(
                t.get('dep') == 'obl:arg' and t.get('head_index') == tok.get('orig_index')
                for t in _clause_toks2))
            if _has_obl_arg2:
                _r = 'STATIF'
            else:
                _r = self._detect_statif_adj(tok['lemma'])
                if str(_r).upper() == 'QUALITE' and _passive_subj and tok.get('dep') == 'ROOT':
                    _r = 'PARTICIPE'
            _rn = str(_r).upper() if _r else ''
            if _rn == 'STATIF' or _r is True:
                tok['is_statif'] = True
            elif _rn == 'PARTICIPE':
                tok['is_participe_passe'] = True
            elif _rn == 'VALEUR':
                tok['is_valeur'] = True
        print(f"DEBUG après détection: tok flags = is_statif={tok.get('is_statif')}, is_participe_passe={tok.get('is_participe_passe')}")

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
        tok['_match_type']  = best.get('match', 'unknown')   # 'exact' | 'embed'
        tok['_top_score']   = best.get('final_score', 0.0)
        # Stocker le sens FR du meilleur candidat pour détecter les subsomptions
        # dans les chaînes génitives (ex: venue→jɔ̀kun sens_fr="raison de la venue")
        if best['final_score'] >= MIN_SCORE:
            tok['sens_fr'] = best.get('fr', '')

        if tok['pos'] == 'VERB' and tok.get('bm'):
            tok['semantic_class'] = self._detect_semantic_class(tok['lemma'])
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

            # Signal lexical KG : nom support dédié → action_noun + kɛ
            try:
                _res = self.db.query(
                    "MATCH (s:Sense {bm: $bm}) "
                    "RETURN s.action_noun AS an LIMIT 1",
                    {'bm': tok['bm']})
                _an = _res[0].get('an') if _res and _res[0] else None
            except Exception:
                _an = None
            if _an:
                tok['intransitive_type'] = 'support'
                tok['action_noun'] = _an

            # Catégorie B1 : intransitif absolu → verbe nu
            elif _sc in ('motion', 'biological', 'posture', 'spontaneous',
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

            # Catégorie B3 : nominalisation -li/-ni → V+li + kɛ
            elif _sc in ('action', 'technique', 'craft', 'communication'):
                tok['intransitive_type'] = 'nominalized'

        if tok['pos'] == 'NOUN' and tok.get('bm') and not tok['bm'].startswith('['):
            tok['is_relational'] = self._detect_relational_noun(tok['lemma'], tok['bm'])

        # Détection du type réflexif quand le verbe a un pronom réflexif dans la clause
        if tok['pos'] == 'VERB' and tok.get('dep') == 'ROOT':
            _ctoks_v = getattr(self, '_current_clause_tokens', [])
            _nsubj_v       = next((t for t in _ctoks_v
                                   if t.get('dep') in ('nsubj', 'nsubj:pass')), None)
            _nsubj_morph_v = str(_nsubj_v.get('morph', '')) if _nsubj_v else ''
            _nsubj_surf_v  = str(_nsubj_v.get('surface', '')).lower() if _nsubj_v else ''
            _nsubj_pos_v   = _nsubj_v.get('pos', '') if _nsubj_v else ''
            _is_plural_subj = (
                'Number=Plur' in _nsubj_morph_v
                or (_nsubj_v and _nsubj_v.get('is_plural'))
                or _nsubj_surf_v == 'on')
            _is_dem_noun = ('PronType=Dem' in _nsubj_morph_v
                            or _nsubj_pos_v == 'NOUN')

            # Règle syntaxique forte : même surface sujet+réflexif (nous nous, ils ils)
            # → toujours réciproque si pluriel, sans appel LLM
            _same_surf_refl = next((
                t for t in _ctoks_v
                if t.get('dep') in ('expl:comp', 'obj')
                and t.get('pos') == 'PRON'
                and _nsubj_surf_v
                and str(t.get('surface', '')).lower() == _nsubj_surf_v
            ), None)

            _has_refl_pron = (
                _same_surf_refl is not None
                or any(
                    t.get('dep') in ('expl:comp', 'obj', 'iobj')
                    and t.get('pos') == 'PRON'
                    and ('Reflex=Yes' in str(t.get('morph', ''))
                         or t.get('dep') == 'iobj')
                    for t in _ctoks_v
                )
            )

            if _has_refl_pron:
                # NOUN singulier / Dem → passif réflexif
                # NOUN pluriel → traité comme les autres pluriels (LLM ou même-surface)
                if _is_dem_noun and not _is_plural_subj:
                    tok['is_refl_passive'] = True
                elif _same_surf_refl and _is_plural_subj:
                    # nous nous, ils ils → réciproque certain, pas de LLM
                    tok['is_reciprocal'] = True
                elif _is_plural_subj:
                    # se/me/Reflex=Yes + pluriel : LLM décide RECIPROCAL vs REFLEXIVE
                    _rtype = self._detect_reflexive_type(tok['lemma'])
                    if _rtype == 'RECIPROCAL':
                        tok['is_reciprocal'] = True
                    elif _rtype == 'PASSIVE':
                        tok['is_refl_passive'] = True
                    elif _rtype == 'IDIOMATIC':
                        tok['is_refl_idiomatic'] = True
                else:
                    # Singulier (expl:comp ou obj) : LLM décide IDIOMATIC vs REFLEXIVE vs PASSIVE
                    _refl_sing = next((t for t in _ctoks_v
                                       if t.get('dep') in ('expl:comp', 'obj')
                                       and t.get('pos') == 'PRON'), None)
                    if _refl_sing:
                        _rtype = self._detect_reflexive_type(tok['lemma'])
                        if _rtype == 'PASSIVE':
                            tok['is_refl_passive'] = True
                        elif _rtype == 'IDIOMATIC':
                            tok['is_refl_idiomatic'] = True

        if tok['pos'] == 'ADJ' and tok.get('bm'):
            _clause_toks = getattr(self, '_current_clause_tokens', [])
            _passive_subj = any(t.get('dep') == 'nsubj:pass' for t in _clause_toks)
            _has_obl_arg  = (tok.get('dep') == 'ROOT' and any(
                t.get('dep') == 'obl:arg' and t.get('head_index') == tok.get('orig_index')
                for t in _clause_toks))
            if _has_obl_arg:
                # ADJ avec complément (sûr de, content de, capable de) → état épistémique
                _result = 'STATIF'
            else:
                _result = self._detect_statif_adj(lemma)
                if str(_result).upper() == 'QUALITE' and _passive_subj and tok.get('dep') == 'ROOT':
                    _result = 'PARTICIPE'

            _result_norm = str(_result).upper() if _result else ''
            if _result_norm == 'STATIF' or _result is True:
                tok['is_statif'] = True
                if all_embed:
                    tok['bm'] = f"[{lemma}]"
            elif _result_norm == 'PARTICIPE':
                tok['is_participe_passe'] = True
            elif _result_norm == 'VALEUR':
                tok['is_valeur'] = True
            # QUALITE → rien, le moteur décide (qualitative par défaut)

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

    def _translate_clause(self, clause: str, frame: str, lang: str = 'fr') -> str:
        self._current_sentence = clause

        clause_vector = self.model.encode(clause)

        if hasattr(self.db, 'search_semantic_phrase'):
            global_match = self.db.search_semantic_phrase(clause_vector, threshold=0.85)
            if global_match:
                print(f"\n     🚀 GLOBAL SEMANTIC MATCH FOUND (KG):")
                print(f"     '{clause}'  ≈  '{global_match['fr']}'")
                print(f"     → Result: '{global_match['bm']}'")
                return global_match['bm']

        # tokens = tokenize(clause, db=self.db,
        #                   backend=LLM_BACKEND, model=LLM_MODEL)
        
        tokens = tokenize(clause, db=self.db,
                          backend=LLM_BACKEND, model=LLM_MODEL)
        # Forcer la langue détectée depuis la phrase principale
        for tok in tokens:
            tok['lang'] = lang

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
        self._current_clause_tokens = tokens

        for tok_idx, tok in enumerate(tokens):
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


            # Build context from adjacent tokens for better embedding retrieval
            # Filter out punctuation and function words that don't add semantic value
            context_lemmas = []
            for offset in [-1, 1]:  # previous and next
                idx = tok_idx + offset
                if 0 <= idx < len(tokens):
                    neighbor = tokens[idx]
                    lem = neighbor.get('lemma', '').strip()
                    pos = neighbor.get('pos', '')
                    # Skip if: punctuation, empty, or too common/structural
                    if lem and pos not in ('PUNCT', 'SYM') and len(lem) > 1:
                        context_lemmas.append(lem)
            context_lemmas = [l.lower() for l in context_lemmas if l]  # lowercase

            # Exécution de la traduction unifiée du jeton si valide

            # APRÈS — appelé APRÈS _translate_token (semantic_class rempli)
            tok, candidates = self._translate_token(tok, frame, context_lemmas, tokens)

            # Détection de transitivité aussi sur les verbes advcl/conj (clauses
            # purposives : 'pour cuisiner et manger') pour que _purp_verb_bm choisisse
            # la forme correcte (transitif sans COD → V+li kɛ ; intransitif → V nu).
            if tok.get('pos') == 'VERB' and (tok.get('is_root')
                                             or tok.get('dep') in ('xcomp', 'advcl', 'conj')):
                if tok.get('action_noun'):
                    tok['intransitive_type'] = 'support'
                else:
                    tok['intransitive_type'] = self._detect_intransitive_type(
                        tok.get('lemma', ''), tok.get('semantic_class', ''))
            
            # Dans la boucle for tok in tokens, après _translate_token :
            if (tok.get('dep') == 'obj'
                    and any(t.get('lemma', '').lower() == 'avoir'
                            and (t.get('is_root') or t.get('dep') == 'ROOT')
                            for t in tokens)):
                _ptype = self._detect_possession_type(
                    tok.get('lemma', ''), tok.get('semantic_class', ''))
                tok['possession_type'] = _ptype

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
            # bm = self._translate_clause(clause, frame)
            _main_lang = all_tokens[0].get('lang', 'fr') if all_tokens else 'fr'
            bm = self._translate_clause(clause, frame, lang=_main_lang)

            print(f"  ✂️  Clause {i+1} -> '{bm}'")

            if bm:
                all_bambara.append(bm)

        bambara_output = ', '.join(b for b in all_bambara if b)

        print(f"\n🇲🇱 BAMBARA : {bambara_output}")
        print("=" * 75)

        # Snapshot du dernier tree pour l'évaluation
        _tree = getattr(self.rule_engine, '_last_tree', {}) or {}
        _main = _tree.get('main', {})
        _tree_meta = {
            'clause_type': _tree.get('clause_type', ''),
            'tense':       _tree.get('tense', ''),
            'neg':         _tree.get('neg', False),
            'tam':         _tree.get('tam', ''),
            'S':           _main.get('S', ''),
            'V':           _main.get('V', ''),
            'O':           _main.get('O', ''),
            'n_obls':      len(_main.get('OBL_ALL', [])),
        }

        # Snapshot des tokens (tagging) pour l'évaluation du parsing
        _tokens_meta = [
            {
                'surface':    t.get('surface', ''),
                'lemma':      t.get('lemma', ''),
                'pos':        t.get('pos', ''),
                'dep':        t.get('dep', ''),
                'role':       t.get('role', ''),
                'head_index': t.get('head_index', -1),
                'orig_index': t.get('orig_index', -1),
            }
            for t in all_tokens
        ]

        return {
            'bambara':  bambara_output,
            'frame':    frame,
            'concepts': all_concepts,
            'tree':     _tree_meta,
            'tokens':   _tokens_meta,
        }
