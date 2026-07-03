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
from config.settings import (LLM_BACKEND, LLM_MODEL, GEMINI_API_KEY, GEMINI_MODEL,
                             OLLAMA_GENERATE_URL)
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
        # Inject LLM classifiers into grammar so build_tree steps can call them
        self.rule_engine.grammar['_classify_refl_verb'] = self._classify_refl_verb
        self.rule_engine.grammar['_classify_privative_noun'] = self._classify_privative_noun

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
                r = _req.post(OLLAMA_GENERATE_URL,
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

    def _call_llm(self, prompt: str, max_tokens: int = 10, timeout: int = None) -> str:
        """Generic LLM call — Ollama (qwen) uniquement.
        timeout=None (défaut) : ceiling standard _LLM_TIMEOUT (90s). Les appelants
        dont la réponse attendue est courte/peu coûteuse peuvent passer un timeout
        explicite plus serré pour échouer vite plutôt que de bloquer jusqu'à 90s."""
        import time as _time
        ollama_ok = _time.time() >= TranslationEngine._llm_circuit_open_until
        _timeout = timeout if timeout is not None else TranslationEngine._LLM_TIMEOUT

        if ollama_ok and LLM_BACKEND == 'ollama':
            payload = {
                'model': LLM_MODEL or 'qwen2.5:3b',
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': 0, 'num_predict': max_tokens}
            }
            try:
                r = requests.post(
                    OLLAMA_GENERATE_URL,
                    json=payload,
                    timeout=_timeout)
                return r.json().get("response", "").strip()
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
                    # print(f"     [CONTEXT BOOST] '{gloss}' → {context_type}: +10 pts")
                # else:
                    # print(f"     [CONTEXT SKIP] '{gloss}' → {context_type}: no match")
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
                         context_tokens: list = None,
                         modifier_lemmas: list = None) -> list:
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
            # Pool serré pour all_embed : 5 pts max (vs 10 avant).
            # Un écart de 10 pts inclut des candidats sémantiquement éloignés
            # (ex: "réellement" → hàáli/très au lieu de bɛ́rɛ/vraiment).
            rerank_pool = [c for c in candidates
                           if top_score - c['final_score'] <= 5]
            if len(rerank_pool) < 2:
                rerank_pool = candidates[:3]  # fallback : top 3 si pool trop petit
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

        _modifier_hint = (
            f'This word is modified by: {", ".join(modifier_lemmas)}.\n'
            f'Prefer the option whose French meaning specifically incorporates '
            f'these modifiers, even if compound.\n'
            if modifier_lemmas else ''
        )
        _simple_pref = (
            '' if modifier_lemmas
            else 'Prefer simple direct meanings over compound or specialized ones.\n'
        )
        prompt = (
            f'You are a French-Bambara lexicon expert.\n\n'
            f'Sentence: "{self._current_sentence}"\n'
            f'Word to translate: "{token_lemma}"\n'
            f'{_modifier_hint}'
            f'Each option below is a Bambara word with its French gloss.\n'
            f'Choose the option whose French gloss is the most DIRECT SYNONYM '
            f'of "{token_lemma}" — same denotation, not just same semantic field.\n'
            f'Example: "réellement" → prefer "vraiment/effectivement" over "très/tout à fait".\n'
            f'Example: "rapidement" → prefer "vite/rapidement" over "tôt/bientôt".\n'
            f'{_simple_pref}'
            f'If none match well, choose the closest synonym.\n\n'
            f'Options:\n{options}\n\n'
            f'Reply with ONLY the number (1-{min(10, len(rerank_pool))}).'
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

        Validation threshold: ≥ 20 pts (covers both exact/substring matches
        and embedding results). Perfect exact (100 pts) is always trusted.
        No text-bypass: LaBSE + LLM decide, not substring heuristics.
        - Valid: keep as-is
        - Invalid: penalize → 0 pts
        - LLM unavailable: penalize → score - 15 pts (conservative)

        FALLBACK: If all top candidates fail validation, restore the best one
        to 50% of its original score (vs full disqualification at 0 pts).
        This handles cases like "faire" (bare verb) matched only to "faire X"
        (compound verbs) — all fail validation but we need a fallback.
        """
        if not candidates or not token_fr:
            return candidates

        token_lower = token_fr.lower().strip()

        # Track invalidated candidates to restore fallback if needed
        original_scores = {i: c.get('score', 0) for i, c in enumerate(candidates[:top_k])}
        invalidated_count = 0

        for i, c in enumerate(candidates[:top_k]):
            score = c.get('score', 0)

            # Perfect exact (100 pts) — trust it unconditionally
            if score >= 100:
                continue

            # Validate all candidates ≥ 20 pts with LLM semantic check
            if score >= 20:
                gloss_fr = c.get('fr', '').lower().rstrip('.').strip()

                # Ask LLM: is gloss semantically equivalent to token?
                # The gloss may be a synonym list ("trop, très, beaucoup")
                # or a compound noun ("plaque de cuisson").
                prompt = (
                    f'Dans un dictionnaire, la glose d\'un mot bambara est: \"{gloss_fr}\".\n'
                    f'Le mot français cherché est: \"{token_lower}\".\n'
                    f'Est-ce que \"{token_lower}\" correspond à l\'une des significations '
                    f'ou synonymes de cette glose (liste ou expression)?\n'
                    f'Réponds uniquement par OUI ou NON.'
                )

                try:
                    resp = self._call_llm(prompt, max_tokens=3).strip().upper()
                    is_valid = resp.startswith('O')

                    if is_valid:
                        print(f"     ✅ LLM valide: '{gloss_fr}' ≈ '{token_lower}' → {c['bm']}")
                    else:
                        # Candidate is semantically wrong → disqualify completely
                        c['score'] = 0
                        c['final_score'] = 0
                        invalidated_count += 1
                        # print(f"     ⚠️  LLM invalide: '{gloss_fr}' ≠ '{token_lower}' → {c['bm']} (→ 0 pts)")

                except Exception as e:
                    print(f"     ⚠️  LLM validation failed: {e}")
                    # Conservative: penalize on LLM failure
                    c['score'] = max(0, score - 15)
                    c['final_score'] = c['score']

        # FALLBACK: if all EXACT matches were invalidated and KG has no bare match,
        # restore only if the compound is linguistically very close to the bare token.
        # E.g., "faire peur" is close to "faire" (1 word apart, same starting word)
        # but "plaque de cuisson" is far from "cuisson" (2+ words apart, different starting word).
        # This handles compound verbs without hardcoding.
        exact_matches = [c for c in candidates[:top_k] if c.get('match') == 'exact']
        invalid_exact = sum(1 for c in exact_matches if c.get('final_score', 0) == 0)

        if invalid_exact > 0 and invalid_exact == len(exact_matches) and exact_matches:
            # Find the best exact match by original score
            best_exact = max(exact_matches, key=lambda c: original_scores.get(candidates.index(c), 0))
            best_idx = candidates.index(best_exact)
            best_orig_score = original_scores.get(best_idx, 0)
            best_gloss = best_exact.get('fr', '').lower()

            # Check linguistic similarity: restore only if compound is very close to bare token
            # "faire peur" (1 word apart, starts with "faire") → restore
            # "plaque de cuisson" (2+ words from "cuisson", doesn't start with "cuisson") → don't restore
            words_in_gloss = [w.rstrip('.,;:!?') for w in best_gloss.split()]
            word_distance = len(words_in_gloss) - 1  # extra words beyond the token
            starts_with_token = best_gloss.startswith(token_lower + ' ')

            # Only restore if: (1) gloss contains token, (2) very close (≤1 extra word), (3) starts with token
            is_close_variant = (
                token_lower in words_in_gloss
                and word_distance <= 1
                and starts_with_token
            )

            if best_orig_score > 0 and is_close_variant:
                restore_score = int(best_orig_score * 0.5)  # 50% of original
                best_exact['score'] = restore_score
                best_exact['final_score'] = restore_score
                # Move this restored exact match to the front after re-sorting
                candidates.remove(best_exact)
                candidates.insert(0, best_exact)
                print(f"     ⚠️  [FALLBACK] Restauré exact match à {restore_score} pts "
                      f"(50% de {best_orig_score}): '{best_gloss}' est proche de '{token_lower}' "
                      f"(distance={word_distance})")

        # Re-sort by score
        candidates.sort(key=lambda x: x.get('final_score', x.get('score', 0)), reverse=True)

        # FINAL FALLBACK: if all exact matches failed validation AND we only have
        # embedding results left, prefer a placeholder over a weak embedding match.
        # This prevents "cuisson" from falling back to "jírisi" (menuiserie) just because
        # it's an embedding match.
        exact_matches = [c for c in candidates if c.get('match') == 'exact']
        embed_only = not exact_matches or all(c.get('final_score', 0) == 0 for c in exact_matches)
        top_is_embed = candidates and candidates[0].get('match') == 'embed'
        top_score = candidates[0].get('final_score', 0) if candidates else 0

        if embed_only and top_is_embed and top_score < 30:
            # All exact matches failed validation; only weak embedding remains
            # Clear candidates to force placeholder fallback downstream
            print(f"     📭 [NO FALLBACK TO EMBED] All exact matches invalidated & "
                  f"top embed too weak ({top_score} pts) → use placeholder instead")
            return []

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

        # Pré-check KG : Sense node avec possession_type → évite appel LLM inutile
        try:
            _kg_pt = self.db.query(
                "MATCH (n:Sense) WHERE toLower(n.fr) = toLower($fr) AND n.possession_type IS NOT NULL "
                "RETURN n.possession_type AS pt LIMIT 1",
                {'fr': lemma})
            if _kg_pt and _kg_pt[0].get('pt'):
                _pt = str(_kg_pt[0]['pt']).upper()
                if _pt in ('AGE', 'MATERIAL', 'EXPERIENCER', 'ABSTRACT', 'STATIF'):
                    print(f"  🔍 [POSS_TYPE KG] '{lemma}' → {_pt}")
                    return _pt
        except Exception:
            pass

        # Le LLM classe le nom en utilisant semantic_class comme indice contextuel.
        # Pas de tables de correspondance en dur : VerbNet ne couvre pas les noms,
        # donc semantic_class est rarement rempli pour un NOUN ; le LLM reste juge.
        _sc_hint = f" (classe sémantique disponible : '{semantic_class}')" if semantic_class else ""
        prompt = (
            f"Le nom français '{lemma}'{_sc_hint} est complément direct de 'avoir'.\n"
            f"Quelle catégorie lui correspond ?\n"
            f"AGE : durée ou âge (âge, ans, siècle, heure)\n"
            f"MATERIAL : objet physique concret qu'on peut tenir ou posséder "
            f"(voiture, maison, téléphone, argent, clé, vêtement, vélo, sac, outil, arme, couteau, bâton)\n"
            f"ABSTRACT : possession non physique — relation humaine, lien social, concept "
            f"ou DISPOSITION VOLITIONNELLE que le sujet peut mobiliser volontairement "
            f"(frère, ami, enfant, mari, famille, idée, droit, talent, chance, avis, "
            f"courage, confiance, patience, volonté, détermination, persévérance, orgueil)\n"
            f"EXPERIENCER : sensation physique externe subie par le corps — la sensation "
            f"est le sujet grammatical EN BAMBARA (faim, soif, chaud, froid, sommeil, "
            f"fièvre, nausée, douleur, mal, vertige, fatigue)\n"
            f"STATIF : état émotionnel PASSIF et INVOLONTAIRE — le sujet ne peut pas "
            f"le déclencher volontairement (test : 'Sois X!' est impossible ou absurde) "
            f"— s'exprime en bambara par une forme participiale "
            f"(peur, honte, joie, colère, jalousie, tristesse, regret, envie, nostalgie)\n"
            f"Réponds UNIQUEMENT par : AGE, MATERIAL, ABSTRACT, EXPERIENCER ou STATIF"
        )

        for _ in range(3):
            try:
                result = self._call_llm(prompt, max_tokens=5).strip().upper()
                if result in ('AGE', 'MATERIAL', 'EXPERIENCER', 'ABSTRACT', 'STATIF'):
                    print(f"  🔍 [POSS_TYPE] '{lemma}' sc={semantic_class!r} → {result}")
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

    def _detect_classifying_adj(self, lemma: str) -> bool:
        """Returns True if the adjective is CLASSIFIANT (no -man), False if QUALIFIANT (needs -man).
        CLASSIFIANT = nationality, category, domain, type (français, international, médical).
        QUALIFIANT = quality, property, characteristic (grand, beau, rouge, chaud)."""
        prompt = (
            f'The French adjective "{lemma}" is used as an epithet (modifying a noun).\n'
            f'QUALIFIANT: describes a quality, property or characteristic of the noun — '
            f'grand, beau, vieux, rouge, rapide, chaud, intelligent, bon, mauvais, simple.\n'
            f'CLASSIFIANT: indicates a category, nationality, domain or type, NOT a quality — '
            f'français, international, européen, médical, électronique, national, politique.\n'
            f'Reply with ONLY one word: QUALIFIANT or CLASSIFIANT.'
        )
        for _attempt in range(3):
            try:
                raw = self._call_llm(prompt, max_tokens=5).strip().upper().split()[0]
                print(f"     🔍 adj_classify? '{lemma}' → '{raw}'")
                if raw in ('QUALIFIANT', 'CLASSIFIANT'):
                    return raw == 'CLASSIFIANT'
            except Exception as e:
                print(f"     🔍 adj_classify attempt {_attempt+1} failed: {e}")
        return False  # fallback: assume qualifying → add -man

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
        Retourne RECIPROCAL, REFLEXIVE, PASSIVE ou SUBJECTIVE."""
        prompt = (
            f'Verb: "{lemma}". Used with reflexive "se".\n'
            f'RECIPROCAL: Two or more participants perform the action on each other '
            f'(meet, fight, kiss, marry, see each other).\n'

            f'REFLEXIVE: The subject consciously and literally performs the action on themselves '
            f'as an object. This includes grooming/body-care '
            f'(wash, dress, shave, comb, hurt oneself, blame oneself, examine oneself).\n'

            f'PASSIVE: "se" has no semantic role; the subject undergoes the action '
            f'or the construction is impersonal/passive '
            f'(be sold, be called, be done, happen, be used).\n'

            f'SUBJECTIVE: "se" is an intrinsic part of the verb to express a state, change '
            f'of state, emotion, cognition, movement, or a completely non-literal meaning '
            f'(realize, remember, get angry, hurry, leave, wonder, concentrate, '
            f'make a mistake, get up, lie down, sit down, get bored).\n'

            f'Return one word only: RECIPROCAL, REFLEXIVE, PASSIVE, or SUBJECTIVE.'

        )
        for _attempt in range(3):
            try:
                raw = self._call_llm(prompt, max_tokens=5).strip().upper()
                parts = raw.split()
                if not parts:
                    raise ValueError('empty response')
                raw = parts[0]
                print(f"     🔍 reflexive_type? '{raw}'")
                result = raw if raw in ('RECIPROCAL', 'REFLEXIVE', 'PASSIVE', 'SUBJECTIVE') else 'SUBJECTIVE'
                return result
            except Exception as e:
                print(f"     🔍 attempt {_attempt+1} failed: {e}")
        return 'SUBJECTIVE'

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
            f"voisin, collègue, patron, maître, chef, roi, dirigeant, responsable, leader, etc.\n"
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

        # ── PRIORITÉ CLASSE SÉMANTIQUE : VerbNet détermine la transitivité ────
        # Niveau 1 — classes intransitives pures → ABSOLU (jamais de COD).
        _INTRANSITIVE_SC = {'motion', 'biological', 'posture', 'spontaneous',
                            'meteorological'}
        if semantic_class in _INTRANSITIVE_SC:
            print(f"  🔍 [TRANSITIVITY] '{lemma}' class={semantic_class} "
                  f"→ ABSOLU (classe autonome, LLM ignoré)")
            return 'ABSOLU'

        # Niveau 2 — classes transitives → ACTION (prennent un COD ; sans COD
        # → V+li kɛ / action_noun kɛ dans step7_final). Pas de LLM nécessaire.
        _TRANSITIVE_SC = {'consumption', 'preparation', 'action', 'craft', 'perception'}
        if semantic_class in _TRANSITIVE_SC:
            print(f"  🔍 [TRANSITIVITY] '{lemma}' class={semantic_class} "
                  f"→ ACTION (classe transitive, LLM ignoré)")
            return 'ACTION'

        # ── LLM : présence d'un COD (complément d'objet direct) ──────────────
        # Note : ne pas inclure la semantic_class dans le prompt — le label 'action'
        # induit le LLM à répondre ACTION même pour les intransitifs de classe action
        # (travailler, courir, danser…). La transitivité s'évalue indépendamment.
        prompt = (
            f"Le verbe français '{lemma}' peut-il prendre un COD (complément d'objet direct) ?\n"
            f"ABSOLU  → jamais de COD (intransitif strict) : courir, dormir, régner, exister\n"
            f"ACTION  → COD possible (transitif) : manger, couper, aider, donner\n"
            f"Réponds UNIQUEMENT par ABSOLU ou ACTION."
        )
        _result = None
        _raw = ''
        for attempt in range(3):
            try:
                _raw = self._call_llm(prompt, max_tokens=5).strip().upper()
                print(f"  🔬 [TRANSITIVITY raw] attempt {attempt+1}: {_raw!r}")
                if _raw.startswith('ABSOLU'):
                    _result = 'ABSOLU'; break
                if _raw.startswith('ACTION'):
                    _result = 'ACTION'; break
            except Exception:
                continue

        verdict = _result
        print(f"  🔍 [TRANSITIVITY] LLM verdict for '{lemma}' (class={semantic_class}) → {verdict}  [raw: {_raw!r}]")
        return verdict

    def _classify_refl_verb(self, lemma: str) -> str:
        """
        Classe un verbe réfléchi français selon son comportement syntaxique
        (catégories alignées sur les classes VerbeNet — le VerbNet français
        de Danlos et al., https://github.com/aymara/verbenet — plutôt que
        sur le seul critère volontaire/accidentel) :

          PRONOMINAL    : n'existe PAS sans 'se' (s'évanouir, se souvenir,
                          se méfier, se taire). ≈ pas d'équivalent VerbeNet
                          (classe purement French-specific, absente du
                          VerbNet anglais source).
          SOIN_CORPOREL : toilette/soin du corps, objet réfléchi par défaut.
                          ≈ VerbeNet floss-41.2.1 (laver, raser, se préparer,) +
                          braid-41.2.2 (coiffer, maquiller, peigner) +
                          dress-41.1.1 (habiller, vêtir).
          POSTURE       : position/changement de position, pas une action
                          sur un objet (s'asseoir, se lever, se coucher,
                          se pencher, s'agenouiller). ≈ VerbeNet
                          assuming_position-50, qui liste justement les
                          formes pronominales ('asseoir s'', 'coucher se',
                          'agenouiller s'') séparément de leurs variantes
                          transitives-causatives (spatial_configuration-47.6 :
                          'asseoir qqn', 'lever qqch').
          ACCIDENTEL    : le sujet subit un évènement fortuit qui l'atteint
                          physiquement (se blesser, se couper, se brûler).
                          ≈ VerbeNet hurt-40.8.3 (blesser, brûler, casser,
                          déchirer, écorcher).
          ACTIF         : toute autre action volontaire du sujet sur lui-même
                          (se déguiser, se défendre..).

        Retourne : 'pronominal' | 'soin_corporel' | 'posture' | 'accidentel' | 'actif'
        """
        # Pas de cache KG : un verdict LLM périmé/erroné (timeout → défaut
        # 'actif') se figeait sinon indéfiniment. On rappelle le LLM à chaque
        # fois.
        prompt = (
            f'Verbe réfléchi à classer : "{lemma}". Catégorie :\n\n'
            f'PRONOMINAL : "se" transforme V transitif en son équivalent INTRANSITIF automatique\n'
            f'  (le sujet subit l\'événement sans agir délibérément sur lui-même),\n'
            f'  OU verbe impossible sans "se", OU sens différent de V.\n'
            f'  Ex: réveiller qqn → se réveiller (le réveil arrive), endormir → s\'endormir,\n'
            f'      fermer → se fermer, évanouir → s\'évanouir, taire → se taire, tromper → se tromper.\n\n'
            f'SOIN_CORPOREL : action d\'hygiène/toilette ACTIVE et intentionnelle sur son propre corps.\n'
            f'  Ex: laver, raser, coiffer, maquiller, habiller, brosser.\n\n'
            f'POSTURE : UNIQUEMENT changement de POSITION PHYSIQUE du corps (où le corps EST).\n'
            f'  Ex: asseoir (debout→assis), coucher (debout→allongé), pencher, agenouiller.\n'
            f'  ATTENTION : réveiller et endormir NE SONT PAS des postures (états de conscience).\n\n'
            f'ACCIDENTEL : blessure ou dommage physique que subit le sujet (volontaire ou non).\n'
            f'  → blesser, couper, brûler, casser (bras/jambe), écorcher = TOUJOURS ACCIDENTEL.\n\n'
            f'ACTIF : autre action intentionnelle du sujet sur lui-même.\n'
            f'  Ex: déguiser, défendre, préparer (mental).\n\n'
            f'Réponds OBLIGATOIREMENT par UN SEUL MOT parmi : PRONOMINAL, SOIN_CORPOREL, POSTURE, ACCIDENTEL ou ACTIF\n'
            f'NE répète PAS le verbe.'
        )

        _CATS = ('PRONOMINAL', 'SOIN_CORPOREL', 'POSTURE', 'ACCIDENTEL', 'ACTIF')
        _result = None
        _raw = ''
        for attempt in range(3):
            try:
                # timeout court (15s) : réponse attendue = un seul mot-catégorie,
                # pas besoin du ceiling 90s — échoue vite sur les 3 tentatives
                # plutôt que de bloquer jusqu'à 4'30 quand Ollama est indisponible.
                _raw = self._call_llm(prompt, max_tokens=6, timeout=15).strip().upper()
                print(f"  🔬 [REFL_CAT raw] attempt {attempt+1}: {_raw!r}")
                _hit = next((c for c in _CATS if c in _raw), None)
                if _hit:
                    _result = _hit.lower(); break
                # LLM a répété la construction réflexive au lieu d'une catégorie.
                # Distinguer : blessure physique (ACCIDENTEL) vs pronominal idiomatique.
                if _raw.strip() == f'SE {lemma.upper()}':
                    try:
                        _chk = self._call_llm(
                            f'Le verbe "{lemma}" décrit-il une blessure ou un dommage physique '
                            f'(couper, brûler, casser…) ? OUI ou NON',
                            max_tokens=3, timeout=10).strip().upper()
                        _result = 'accidentel' if 'OUI' in _chk else 'pronominal'
                    except Exception:
                        _result = 'pronominal'
                    break
            except Exception:
                continue

        cat = _result or 'actif'  # défaut si LLM indisponible
        print(f"  🔄 [REFL_CAT] '{lemma}' → {cat}  [LLM, raw={_raw!r}]")
        return cat

    def _classify_privative_noun(self, lemma: str) -> str:
        """
        Classe un nom français pour choisir le marqueur privatif bambara :
          ACTION : nom d'action/processus dérivé d'un verbe (cuisson, nettoyage,
                   traitement, construction, réparation, formation...).
                   → marqueur 'bali' (sans faire l'action).
          CHOSE  : nom de chose, substance ou état (sel, eau, sucre, argent,
                   lumière, permission, bruit...).
                   → marqueur 'tan' (sans la chose).
        Retourne : 'action' | 'chose'
        """
        prompt = (
            f'Le nom français "{lemma}" est-il un NOM D\'ACTION '
            f'(dérivé d\'un verbe, représentant un processus ou une activité) '
            f'ou un NOM DE CHOSE (substance, objet, état) ?\n\n'
            f'NOM D\'ACTION : cuisson (de cuire), nettoyage (de nettoyer), '
            f'traitement (de traiter), construction, formation, réparation...\n'
            f'NOM DE CHOSE : sel, eau, sucre, argent, lumière, permission, bruit...\n\n'
            f'Réponds UNIQUEMENT par : ACTION ou CHOSE'
        )
        _raw = ''
        try:
            _raw = self._call_llm(prompt, max_tokens=4, timeout=10).strip().upper()
            if 'ACTION' in _raw:
                return 'action'
        except Exception:
            pass
        return 'chose'

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

        for attempt in range(3):
            try:
                infinitive = self._call_llm(prompt, max_tokens=8).strip().lower()
                if infinitive and len(infinitive) > 1:
                    return infinitive
            except Exception as e:
                print(f"     ⚠️  Infinitive normalization failed (attempt {attempt+1}): {e}")

        return verb

    def _detect_semantic_class(self, lemma: str) -> str:
        """Détecte la classe sémantique d'un verbe via LLM."""
        prompt = (
            f'Quelle est la nature sémantique du verbe français "{lemma}" ?\n\n'
            f'Catégories AUTONOMES (intransitifs, n\'acceptent pas de COD direct):\n'
            f'  motion=le sujet change de lieu ou se déplace — déplacements, départs, arrivées, directions\n'
            f'    (aller, venir, courir, marcher, arriver, sortir, entrer, monter, descendre...)\n'
            f'  biological=processus vital ou TRANSITION D\'ÉTAT corporel ponctuelle :\n'
            f'    vivre, mourir, naître, respirer, réveiller (rompt le sommeil),\n'
            f'    lever (oppose la gravité, déclenche le passage couché→debout).\n'
            f'    Voix active : "lever quelqu\'un" = déclencher un changement d\'état.\n'
            f'  posture=CONFIGURATION SPATIALE STABLE du corps (état maintenu) :\n'
            f'    asseoir (place sur un support), coucher (allonge sur une surface),\n'
            f'    pencher, accroupir. Voix active : "asseoir un enfant" = positionner sur surface.\n'
            f'    NE PAS classer "lever" en posture : lever = processus transformationnel (biological).\n'
            f'  spontaneous=réaction involontaire (rire, crier, pleurer...)\n'
            f'  perception=voir, entendre, sentir (perception directe)\n'
            f'  meteorological=phénomène atmosphérique (pleuvoir, neiger...)\n'
            f'  copula=lien attributif (être, sembler, paraître...)\n'
            f'  stative_cognitive=ÉTAT MENTAL STATIQUE, atélique, incompatible avec le progressif\n'
            f'    ("je suis en train de savoir" est impossible) :\n'
            f'    savoir, connaître, croire, penser, supposer, ignorer, comprendre,\n'
            f'    reconnaître (au sens de "admettre"), douter, se souvenir, oublier.\n'
            f'    TEST : l\'état est homogène — on sait ou on ne sait pas, sans transition.\n'
            f'    EXCLURE les verbes d\'ACTIVITÉ (travailler, jouer, étudier, lire) :\n'
            f'    "il est en train de travailler" EST possible → c\'est action, pas stative_cognitive.\n'
            f'    EXCLURE aimer/détester (=psych_emotion) et vouloir/souhaiter (=modal).\n'
            f'  psych_emotion=ÉTAT AFFECTIF du sujet envers qqch/qqun :\n'
            f'    aimer, adorer, détester, haïr, apprécier, chérir, craindre, redouter, préférer.\n'
            f'    EXCLURE vouloir/souhaiter/désirer (=modal) et savoir/croire (=stative_cognitive).\n'
            f'  modal=VOLITION ou INTENTION du sujet (semi-auxiliaire suivi d\'un infinitif) :\n'
            f'    vouloir, souhaiter, désirer, oser, prétendre (avoir l\'intention de).\n'
            f'    EXCLURE aimer/détester (=psych_emotion) et devoir/falloir (=obligation).\n'
            f'  obligation=NÉCESSITÉ ou OBLIGATION du sujet (semi-auxiliaire) :\n'
            f'    devoir, falloir, il faut.\n'
            f'    EXCLURE vouloir/souhaiter (=modal), pouvoir (=autre).\n\n'
            f'Catégories TRANSITIVES (acceptent souvent un COD):\n'
            f'  action=ACTIVITÉ INTRANSITIVE par nature (pas de COD habituel) :\n'
            f'    travailler, courir, marcher, nager, danser, voyager, jouer (sans objet),\n'
            f'    étudier (intransitif), lire (intransitif).\n'
            f'    ⚠️ NE PAS classer ici les verbes qui prennent normalement un COD :\n'
            f'    acheter, vendre, donner, prendre, chercher, trouver → utiliser "other".\n'
            f'  consumption_liquid=ingestion de LIQUIDE (boire, siroter — jamais consumption pour boire)\n'
            f'  consumption=ingestion SOLIDE uniquement (manger, croquer, dévorer, avaler qqch de solide — ≠ boire)\n'
            f'  preparation=transformation (cuisiner, préparer...)\n'
            f'  technique=travail spécialisé (construire, réparer...)\n'
            f'  craft=création artistique (peindre, écrire...)\n'
            f'  communication=parole (dire, raconter, demander...)\n'
            f'  having=ÉTAT de POSSESSION STATIQUE uniquement (posséder, détenir, contenir,\n'
            f'    appartenir, garder, tenir). Avoir = having UNIQUEMENT au sens possessif.\n'
            f'    ⚠️ acheter ≠ having (acheter = transaction → other)\n'
            f'  other=verbe transitif direct standard sans catégorie propre :\n'
            f'    acheter, vendre, donner, prendre, chercher, trouver, voir, rencontrer,\n'
            f'    envoyer, recevoir, ouvrir, fermer, casser, porter, mettre, garder...\n\n'
            f'Réponds UNIQUEMENT par le nom de la catégorie.'
        )
        _VALID_SC = {
            'motion', 'biological', 'posture', 'spontaneous', 'perception',
            'meteorological', 'copula', 'stative_cognitive', 'psych_emotion', 'modal', 'obligation',
            'action', 'consumption_liquid', 'consumption',
            'preparation', 'technique', 'craft', 'communication', 'having', 'other',
        }
        try:
            import re as _re
            cls_raw = self._call_llm(prompt, max_tokens=15).strip().lower()
            # Inclure '_' pour préserver 'consumption_liquid' (ne pas split en 'consumption')
            words   = _re.findall(r'[a-z_]+', cls_raw)
            # Chercher la première correspondance exacte avec une classe valide
            cls = next((w for w in words if w in _VALID_SC), 'other')

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
            elif pos == 'VERB' and 'semantic_class' not in tok:
                # bm pré-assigné (parseur/KG) : détecter quand même la classe
                # sémantique pour que la transitivité (li kɛ / la / nu) soit juste.
                # Normaliser d'abord le lemme (forme fléchie → infinitif) pour que
                # VerbNet trouve le verbe ; sinon 'venue' → VerbNet miss → LLM.
                _pre_lem = tok['lemma']
                _pre_inf = self._normalize_verb_to_infinitive(_pre_lem)
                if _pre_inf and _pre_inf != _pre_lem:
                    tok['lemma'] = _pre_inf
                tok['semantic_class'] = self._detect_semantic_class(tok['lemma'])
            return tok, []

        if pos == 'PUNCT':
            return tok, []
        # ── NÉGATION : ne pas assigner de bm aux tokens de négation ──────────
        # ne/n'/pas/jamais/plus/rien → role='negation' ou dans neg_surfaces KG
        # Ces tokens doivent rester bm='' pour ne pas parasiter les obliques
        _surf_neg = str(surface).lower().rstrip("'").rstrip('\u2019').rstrip('\u2018')
        _neg_surfs = self.rule_engine.grammar.get('neg_surfaces', set())
        if tok.get('role') == 'negation' or _surf_neg in _neg_surfs:
            # N\u00e9gateurs PORTEURS DE CONTENU (plus\u2192bilen, rien\u2192foyi) : on conserve
            # le bm pos\u00e9 par le parseur depuis le KG, il sera rendu en fin de
            # clause. N\u00e9gateurs PURS (ne/pas/jamais) : aucun bm KG \u2192 vid\u00e9s pour
            # ne pas parasiter les obliques. Discrimination par le bm du KG.
            if not tok.get('bm'):
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
                       OR toLower(n.fr)      = toLower($lemma) + '.'
                    RETURN n.bm AS bm, n.pos AS sense_pos, n.semantic_class AS sense_sc
                    ORDER BY
                        CASE WHEN toLower(n.surface) = toLower($surface)
                             THEN 0 ELSE 1 END,
                        CASE WHEN toLower(n.fr) = toLower($lemma) THEN 0
                             WHEN toLower(n.fr) = toLower($lemma) + '.' THEN 1
                             ELSE 2 END,
                        CASE WHEN toLower(coalesce(n.pos,'')) IN ['verb','verbe'] AND $pos='VERB'
                             THEN 0
                             WHEN toLower(coalesce(n.pos,'')) IN ['noun','nom','n'] AND $pos='NOUN'
                             THEN 0
                             WHEN toLower(coalesce(n.pos,'')) IN ['adjective','adj'] AND $pos='ADJ'
                             THEN 0
                             ELSE 1 END
                    LIMIT 1
                """, {'surface': surface, 'lemma': lemma, 'pos': pos})
                if res and res[0].get('bm'):
                    tok['bm'] = res[0]['bm']
                    _sense_pos = str(res[0].get('sense_pos') or '').lower()
                    _sense_sc  = res[0].get('sense_sc') or ''
                    _sense_fr  = str(res[0].get('fr') or res[0].get('bm', '')).strip()
                    # Lire semantic_class depuis KG si disponible (évite appel LLM nondéterministe)
                    if _sense_sc and not tok.get('semantic_class'):
                        tok['semantic_class'] = _sense_sc
                    # Si spaCy a mal tagué un nom comme ADJ, corriger via le KG
                    if pos == 'ADJ' and _sense_pos in ('noun', 'n', 'nom'):
                        tok['pos'] = 'NOUN'
                        pos = 'NOUN'
                    if pos == 'NOUN' and not tok['bm'].startswith('['):
                        tok['is_relational'] = self._detect_relational_noun(tok['lemma'], tok['bm'])
                    elif pos == 'ADJ':
                        self._classify_adj_state(tok)
                    # Pour les NOUN avec correspondance KG courte (fr = lemma exact en 1 mot),
                    # continuer vers le retriever sémantique : le modèle embedding peut trouver
                    # une traduction plus précise (ex: fille→dénmuso plutôt que mùsoma).
                    # Si fr contient plusieurs mots ou ponctuation, la correspondance est
                    # déjà spécifique → retour anticipé justifié.
                    _kg_fr_words = [w for w in _sense_fr.rstrip('.').split() if w]
                    if pos == 'NOUN' and len(_kg_fr_words) <= 1:
                        pass  # continuer vers le retriever sémantique
                    else:
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
                # Pas de persistance KG : un mot-outil mal traduit par le LLM
                # (ex: 'avec' → 'avec', simple écho) se figeait sinon en dur.
                result = self._call_llm(prompt, max_tokens=8).strip()
                if result and result.upper() != 'EMPTY':
                    tok['bm'] = result
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
                tok['raw_lemma'] = lemma   # conserver le lemme spacy original avant normalisation
                tok['lemma'] = infinitive  # propager l'infinitif → VerbNet/semantic_class/refl utilisent la forme correcte
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

        # ── MODIFIER-GLOSS BOOSTING: compound noun sense selection ──
        # When a noun has nmod/amod modifiers, boost compound senses whose FR gloss
        # contains the modifier's lemma or surface form. This anchors "patte de devant"
        # → ɲɛ́sen without needing LLM reranking.
        _modifier_lemmas = set()
        if pos == 'NOUN' and all_tokens and candidates:
            _tok_idx = tok.get('orig_index')
            for _t in all_tokens:
                if (_t.get('dep') in ('nmod', 'amod')
                        and _t.get('head_index') == _tok_idx):
                    if _t.get('lemma'):
                        _modifier_lemmas.add(_t['lemma'].lower())
                    if _t.get('surface'):
                        _modifier_lemmas.add(_t['surface'].lower())
            if _modifier_lemmas:
                for _c in candidates:
                    _fr = _c.get('fr', '').lower()
                    if any(_ml in _fr for _ml in _modifier_lemmas):
                        _c['final_score'] = _c.get('final_score', 0) + 25
                candidates.sort(key=lambda x: x.get('final_score', 0), reverse=True)

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
            modifier_lemmas=list(_modifier_lemmas) if _modifier_lemmas else None,
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
        # Classification épithète : QUALIFIANT vs CLASSIFIANT (adjectifs attributifs)
        if tok.get('pos') == 'ADJ' and tok.get('dep') == 'amod':
            if self._detect_classifying_adj(tok['lemma']):
                tok['is_classifying_adj'] = True
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
            # Validation linguistique : stative_cognitive exige un sujet Experiencer.
            # Si context_type='agent' (le sujet fait l'action), c'est une contradiction
            # → le verbe est une activité dynamique, pas un état cognitif statique.
            if (tok['semantic_class'] == 'stative_cognitive'
                    and tok.get('context_type') == 'agent'):
                tok['semantic_class'] = 'action'
                print(f"     🏷️  semantic_class('{tok['lemma']}') reclassifié "
                      f"stative_cognitive→action [context_type=agent]")
            _sc = tok.get('semantic_class', '')

            _morph_str = str(tok.get('morph', ''))
            _is_part_pass = ('VerbForm=Part' in _morph_str
                             and 'Voice=Pass' in _morph_str)
            if _is_part_pass:
                if _sc in ('posture', 'biological', 'spontaneous', 'consumption'):
                    _is_statif = True
                else:
                    # _detect_statif_adj renvoie une des 4 catégories
                    # QUALITE/STATIF/VALEUR/PARTICIPE (jamais vide) : seule
                    # 'STATIF' correspond à un participe-adjectif statif
                    # (-len/-nen dòn). Un check de vérité générique sur la
                    # chaîne était toujours vrai (VALEUR/QUALITE/PARTICIPE
                    # sont aussi des chaînes non-vides) → tout participe
                    # passif finissait classé statif (ex: 'faire' → VALEUR
                    # → is_statif=True à tort).
                    try:
                        _is_statif = (self._detect_statif_adj(tok['lemma']) == 'STATIF')
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
            # NB: 'perception' EXCLU — voir/entendre/sentir sont TRANSITIFS
            #   (il l'a vu = a yé a yé ; il voit = a bɛ yéli kɛ). Sans objet
            #   ils suivent le chemin ACTION (V+li kɛ), pas l'intransitif absolu.
            elif _sc in ('motion', 'biological', 'posture', 'spontaneous',
                       'meteorological'):
                tok['intransitive_type'] = 'ABSOLU'

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

            # Construction périphrastique "se V à/de INF" (se mettre à, se
            # décider à, s'apprêter à...) : le verbe réflexif régit un
            # complément infinitif (xcomp) → inchoatif/aspectuel par
            # construction syntaxique, signal fiable sans appel LLM (le LLM,
            # ne voyant que le lemme nu hors contexte "à V", confond souvent
            # ce sens avec le sens littéral du verbe, ex: mettre = "placer").
            _governs_xcomp_inf = any(
                t.get('dep') == 'xcomp' and t.get('pos') == 'VERB'
                and t.get('head_index') == tok.get('orig_index')
                for t in _ctoks_v)

            if _has_refl_pron and _governs_xcomp_inf:
                tok['is_refl_Subjective'] = True
            elif _has_refl_pron:
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
                    elif _rtype == 'SUBJECTIVE':
                        tok['is_refl_Subjective'] = True
                else:
                    # Singulier (expl:comp ou obj) : LLM décide SUBJECTIVE vs REFLEXIVE vs PASSIVE
                    _refl_sing = next((t for t in _ctoks_v
                                       if t.get('dep') in ('expl:comp', 'obj')
                                       and t.get('pos') == 'PRON'), None)
                    if _refl_sing:
                        _rtype = self._detect_reflexive_type(tok['lemma'])
                        if _rtype == 'PASSIVE':
                            tok['is_refl_passive'] = True
                        elif _rtype == 'SUBJECTIVE':
                            tok['is_refl_Subjective'] = True

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

    def _split_clauses(self, sentence: str, tokens: list = None) -> list:
        """
        Phase 1 — split at major comma boundaries (anteposed appositive, introductory NP).
        Phase 2 — within each comma segment, split at dep-based clause boundaries
                   (acl:relcl, advcl, ccomp, xcomp).
        """
        if not tokens:
            import re
            parts = re.split(r',|(?<=[a-zA-ZÀ-ÿ])\.(?=\s+[A-ZÀ-Ÿ]|\s*$)', sentence)
            parts = [p.strip().strip('.') for p in parts]
            return [p for p in parts if p and len(p.split()) > 1]

        _sorted_toks = sorted(tokens, key=lambda x: x['orig_index'])

        # ── Phase 1 : comma split ────────────────────────────────────────────────
        _split_start_tok = None

        # a) Appositive anteposée : dep=appos vient avant son head
        for tok in _sorted_toks:
            if tok.get('dep') == 'appos':
                _ai = tok.get('orig_index', -1)
                _hi = tok.get('head_index', -1)
                if _ai < _hi:
                    _ht = next((t for t in tokens if t.get('orig_index') == _hi), None)
                    if _ht:
                        _pre = [t for t in tokens
                                if t.get('head_index') == _hi
                                and t.get('dep') in ('flat', 'flat:name', 'det')
                                and t['orig_index'] < _hi]
                        _split_start_tok = min([_ht] + _pre, key=lambda x: x['orig_index'])
                        break

        # b) Fallback : phrase nominale introductive avant virgule
        #    premier token NOUN/PROPN + aucun verbe fléchi avant la virgule
        #    ≠ "sans moi, tu…" (premier token ADP)
        # Note: le rôle du token virgule est 'content' (pas 'punct') — on filtre
        # sur pos='PUNCT' ou dep='punct' à la place.
        if not _split_start_tok:
            _comma_toks = [t for t in _sorted_toks
                           if t.get('surface') == ','
                           and (t.get('pos') == 'PUNCT' or t.get('dep') == 'punct')]
            for _ct in _comma_toks:
                _ci     = _ct['orig_index']
                _before = [t for t in _sorted_toks if t['orig_index'] < _ci]
                _after  = [t for t in _sorted_toks
                           if t['orig_index'] > _ci
                           and t.get('pos') != 'PUNCT'
                           and t.get('dep') != 'punct'
                           and t.get('surface') not in (',', '.')]
                if not _before or not _after:
                    continue
                _verb_before = any(
                    t.get('pos') == 'VERB'
                    and t.get('dep') not in ('acl', 'acl:relcl', 'amod')
                    for t in _before
                )
                if (not _verb_before
                        and _before[0].get('pos') in ('NOUN', 'PROPN')
                        and _after[0].get('pos') in ('NOUN', 'PROPN', 'PRON')
                        and _after[0].get('dep') in ('nsubj', 'flat', 'flat:name',
                                                      'ROOT', 'nsubj:pass')):
                    _split_start_tok = _after[0]
                    break

        # c) Virgule avant une relative (qui/que/dont) après clause principale complète
        #    "S V ..., qui/que/dont SUBORD" → split en deux unités de traduction
        if not _split_start_tok:
            for _ct in _comma_toks:
                _ci    = _ct['orig_index']
                _before = [t for t in _sorted_toks if t['orig_index'] < _ci]
                _after  = [t for t in _sorted_toks
                           if t['orig_index'] > _ci
                           and t.get('pos') != 'PUNCT'
                           and t.get('dep') != 'punct'
                           and t.get('surface') not in (',', '.')]
                if not _before or not _after:
                    continue
                _has_root_verb = any(
                    t.get('pos') == 'VERB'
                    and t.get('dep') not in ('acl', 'acl:relcl', 'amod')
                    for t in _before
                )
                if _has_root_verb and _after[0].get('role') == 'relative':
                    _split_start_tok = _after[0]
                    break

        # d) Clauses coordonnées : "S V1, je/tu/il V2, ..."
        #    Virgule entre deux clauses indépendantes à sujet pronominal distinct.
        #    y compris "S V1, peut-il V2 ?" (inversion interrogative)
        if not _split_start_tok:
            for _ct in _comma_toks:
                _ci    = _ct['orig_index']
                _before = [t for t in _sorted_toks if t['orig_index'] < _ci]
                _after  = [t for t in _sorted_toks
                           if t['orig_index'] > _ci
                           and t.get('pos') != 'PUNCT'
                           and t.get('dep') != 'punct'
                           and t.get('surface') not in (',', '.')]
                if not _before or not _after:
                    continue
                _has_root_verb = any(
                    t.get('pos') == 'VERB'
                    and t.get('dep') not in ('acl', 'acl:relcl', 'amod')
                    for t in _before
                )
                # Match: pronoun subject (normal clause)
                _is_pron_subj = (_after[0].get('pos') == 'PRON'
                                 and _after[0].get('role') != 'relative'
                                 and _after[0].get('dep') in ('nsubj', 'nsubj:pass'))
                # Match: auxiliary/modal verb with inverted pronoun (interrogative)
                # e.g., "peut-il manger"
                _is_modal_inversion = (_after[0].get('pos') == 'VERB'
                                      and _after[0].get('dep') in ('ROOT', 'aux')
                                      and len(_after) > 1
                                      and _after[1].get('pos') == 'PRON'
                                      and _after[1].get('dep') in ('nsubj', 'nsubj:pass'))
                if _has_root_verb and (_is_pron_subj or _is_modal_inversion):
                    _split_start_tok = _after[0]
                    break

        # e) Conjonction de coordination après clause principale
        #    "S V O1, ainsi que O2" / "S V1, et S V2"
        if not _split_start_tok:
            for _ct in _comma_toks:
                _ci    = _ct['orig_index']
                _before = [t for t in _sorted_toks if t['orig_index'] < _ci]
                _after  = [t for t in _sorted_toks
                           if t['orig_index'] > _ci
                           and t.get('pos') != 'PUNCT'
                           and t.get('dep') != 'punct'
                           and t.get('surface') not in (',', '.')]
                if not _before or not _after:
                    continue
                _has_root_verb = any(
                    t.get('pos') == 'VERB'
                    and t.get('dep') not in ('acl', 'acl:relcl', 'amod')
                    for t in _before
                )
                if _has_root_verb and _after[0].get('dep') == 'cc':
                    _split_start_tok = _after[0]
                    break

        # f) Clause adverbiale anteposée : "Si/Quand X, ROOT_clause"
        #    advcl VERB avant la virgule, ROOT (ou premier token non-ponct) après
        if not _split_start_tok:
            for _ct in _comma_toks:
                _ci    = _ct['orig_index']
                _before = [t for t in _sorted_toks if t['orig_index'] < _ci]
                _after  = [t for t in _sorted_toks
                           if t['orig_index'] > _ci
                           and t.get('pos') != 'PUNCT'
                           and t.get('dep') != 'punct'
                           and t.get('surface') not in (',', '.')]
                if not _before or not _after:
                    continue
                _has_advcl_verb = any(
                    t.get('dep') == 'advcl' and t.get('pos') in ('VERB', 'AUX')
                    for t in _before)
                _root_in_after = any(t.get('dep') == 'ROOT' for t in _after)
                if _has_advcl_verb and _root_in_after:
                    _split_start_tok = _after[0]
                    break

        # g) Fallback : clauses indépendantes séparées par virgule
        #    "S V1, S V2, S V3, ..." (énumération de clauses)
        if not _split_start_tok and _comma_toks:
            for _ct in _comma_toks:
                _ci    = _ct['orig_index']
                _before = [t for t in _sorted_toks if t['orig_index'] < _ci
                          and t.get('pos') != 'PUNCT' and t.get('dep') != 'punct']
                _after  = [t for t in _sorted_toks
                           if t['orig_index'] > _ci
                           and t.get('pos') != 'PUNCT'
                           and t.get('dep') != 'punct'
                           and t.get('surface') not in (',', '.')]
                if not _before or not _after:
                    continue
                # Vérifier que before ET after ont un VERB ROOT indépendant
                _before_has_root = any(
                    t.get('dep') == 'ROOT' and t.get('pos') in ('VERB', 'AUX')
                    for t in _before)
                _after_has_root = any(
                    t.get('dep') == 'ROOT' and t.get('pos') in ('VERB', 'AUX')
                    for t in _after)
                if _before_has_root and _after_has_root:
                    _split_start_tok = _after[0]
                    break

        # h) supprimé — le split se fait uniquement sur virgule.

        # Si un split est trouvé : séparer texte + tokens, puis appliquer
        # le dep-split indépendamment dans chaque segment
        if _split_start_tok:
            _si       = _split_start_tok.get('orig_index', -1)
            seg1_toks = [t for t in tokens if t['orig_index'] < _si]
            seg2_toks = [t for t in tokens if t['orig_index'] >= _si]
            _surf     = _split_start_tok.get('surface', '')
            _si_idx   = _split_start_tok.get('orig_index', -1)
            _n_before = sum(1 for t in _sorted_toks
                            if t.get('surface') == _surf
                            and t['orig_index'] < _si_idx)
            _pos = -1
            for _ in range(_n_before + 1):
                _pos = sentence.find(_surf, _pos + 1)
            if _pos > 0 and seg1_toks and seg2_toks:
                seg1_text = sentence[:_pos].strip().rstrip(',').strip()
                seg2_text = sentence[_pos:].strip()
                result = []
                result.extend(self._split_clauses(seg1_text, seg1_toks))
                result.extend(self._split_clauses(seg2_text, seg2_toks))
                if result:
                    return result

        # Pas de comma split : dep-split direct
        return self._split_at_deps(sentence, tokens)

    def _split_at_deps(self, sentence: str, _tokens: list) -> list:
        """Le rule engine gère ccomp/advcl inline (step3 ccomp block, step5 advcl.py).
        Un split dep-based casse les frontières de clause (ex: ccomp sur le prédicat
        nominal strande l'article dans la clause précédente). Phrase traitée en entier."""
        if not sentence.strip():
            return []
        return [sentence.strip()]

    # ------------------------------------------------------------------
    # SINGLE CLAUSE TRANSLATION
    # ------------------------------------------------------------------

    def _translate_clause(self, clause: str, frame: str, lang: str = 'fr') -> str:
        self._current_sentence = clause

        # ── FIXED PHRASES (Rule 5) ──────────────────────────────────────
        _FIXED_PHRASES = {
            'ainsi donc': 'ola sa',
        }
        clause_normalized = clause.lower().strip().rstrip('?!.,')
        for fr_phrase, bm_phrase in _FIXED_PHRASES.items():
            if clause_normalized == fr_phrase:
                print(f"\n     🎯 FIXED PHRASE MATCH: '{clause}' → '{bm_phrase}'")
                return bm_phrase

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

            # Détection des dimensions sur TOUT verbe : transitivité, classe sém.,
            # consumption solid/liquid, volition, agentivité.
            if tok.get('pos') == 'VERB':
                # Garantir que la classe sémantique est détectée par le LLM pour
                # TOUT verbe, quelle que soit l'origine du bm (KG label, retrieve…).
                # Sans ça, 'venir' (bm='nà' via KG label) sortait sans classe →
                # transitivité ACTION → 'nàli kɛ' au lieu de 'nà' (motion).
                if not tok.get('semantic_class'):
                    tok['semantic_class'] = self._detect_semantic_class(tok.get('lemma', ''))
                # Transitité déterminée par VerbNet (via semantic_class) ou
                # LLM en fallback — jamais court-circuitée par action_noun.
                # action_noun reste dans le token pour le rendu (step7_final).
                tok['intransitive_type'] = self._detect_intransitive_type(
                    tok.get('lemma', ''), tok.get('semantic_class', ''))
            
            # Dans la boucle for tok in tokens, après _translate_token :
            if (tok.get('dep') == 'obj'
                    and any((t.get('lemma', '').lower() == 'avoir'
                             or t.get('semantic_class') == 'having')
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

        # Use spaCy dependencies to auto-detect clause boundaries
        clauses      = self._split_clauses(sentence, tokens=all_tokens)
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
                'bm':         t.get('bm', ''),
                'sens_fr':    t.get('sens_fr', ''),
            }
            for t in all_tokens
        ]

        # Glose sémantique KG : premier sens FR de chaque token de contenu traduit.
        # Utilisée pour l'évaluation sémantique sans back-translation.
        _sens_fr_parts = []
        for t in all_tokens:
            if t.get('role') == 'content' and t.get('sens_fr'):
                _first = t['sens_fr'].split('.')[0].split(',')[0].strip()
                if _first:
                    _sens_fr_parts.append(_first)
        _sens_fr_gloss = ' '.join(_sens_fr_parts)

        return {
            'bambara':       bambara_output,
            'frame':         frame,
            'concepts':      all_concepts,
            'tree':          _tree_meta,
            'tokens':        _tokens_meta,
            'sens_fr_gloss': _sens_fr_gloss,
        }
