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
from kg.retriever import KGRetriever, _primacy_key
from embeddings.labse_encoder import encode as _embed
from rules import build_tree, tree_to_bambara, RuleEngine
from rules.core import _is_avoir
from config.settings import (LLM_BACKEND, LLM_MODEL, GEMINI_API_KEY, GEMINI_MODEL,
                             OLLAMA_GENERATE_URL)
import os
from embeddings.labse_encoder import _get_model

# Floor set above the maximum possible bonus contribution (POS +2, conciseness
# +1 = 3 pts max) so no candidate can be accepted on bonuses alone — it must
# always carry some genuine gloss or embedding signal (G(s,t) or 40*cosine > 0).
MIN_SCORE      = 5     # 0-100 point scale
TOP_K          = 5
ADJ_CONFIDENCE = 70    # 0-100 point scale
VERB_MIN_SCORE = 75    # 0-100 point scale

# Minimum embed top-score below which the embedding space is considered
# dead — synonyms from the same space won't rescue the search.
_EMBED_DEAD_ZONE = 50  # 0-100 point scale


class TranslationEngine:

    def __init__(self, db):
        self.db                = db
        self.retriever         = KGRetriever(db)
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
                print(f"  LLM Ollama prêt ({model})")
                return
            except _req.exceptions.ConnectionError:
                print("  Ollama non joignable — vérifier que 'ollama serve' tourne")
                return
            except _req.exceptions.Timeout:
                if attempt == 1:
                    print(f"  Ollama lent au démarrage (>{wait}s), nouvelle tentative...")
                else:
                    print(f"  Ollama timeout ({wait}s) — fonctionnement sans LLM")
            except Exception as e:
                print(f"  LLM Ollama erreur : {type(e).__name__}: {e}")
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
                print(f"  Ollama _call_llm échoué : {type(_e).__name__} — model={LLM_MODEL}")

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

        # Garde structurel : ROOT + enfant dep='cop' = attribut du sujet, sans
        # ambiguïté possible ("le travail... est meilleur" → 'meilleur' ROOT,
        # 'être' cop) — pas besoin de deviner via LLM, qui peut se tromper
        # (ex: 'meilleur' classé à tort 'modified_noun' au lieu de 'predicate',
        # ce qui filtre le candidat KG fìsaman/Adjective au profit du candidat
        # fìsamannci/Noun, un mismatch de POS). Décision 2026-07-13.
        _orig_idx = tok.get('orig_index')
        if (tok.get('dep') == 'ROOT'
                and any(t.get('dep') == 'cop' and t.get('head_index') == _orig_idx
                        for t in all_tokens)):
            tok['context_type'] = 'predicate'
            print(f"     [CONTEXT] '{lemma}' → context_type=predicate (structurel : ROOT+cop)")
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
            # Un candidat déjà à 100 pts (match exact ou premier sens d'une
            # liste "mot, synonyme") n'a aucune ambiguïté à lever : le LLM
            # (petit modèle, verdict instable d'un candidat à l'autre pour
            # des gloses qui dénotent pourtant le même mot-sens) ne fait
            # qu'introduire du bruit non-déterministe dans le départage —
            # ex: "ami." jugé NO mais "ami, bien-aimé." jugé YES pour le
            # même context_type='possessive', alors que les deux DÉNOTENT
            # "ami" (bug trouvé 2026-07-20 : "mon ami" → 'díyanyemɔgɔ'
            # (freq=0, sens "bien-aimé") au lieu de 'téri' (freq=253, sens
            # "ami." simple), le boost écrasant le départage par fréquence).
            # Réserver le LLM aux matches déjà imparfaits (<100 pts), où il
            # sert réellement à distinguer un sens approximatif pertinent
            # d'un faux-positif substring.
            if c.get('final_score', 0) >= 100:
                continue
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
                print(f"       Context boost failed for '{gloss}': {e}")

        # Re-sort by final_score after context boosts — départage via
        # _primacy_key (kg/retriever.py), identique au tri du retriever
        # (attesté d'abord, puis sense_index, puis corpus_freq) — un tuple
        # ad-hoc local divergeait sur le filtre "attesté d'abord" (bug
        # trouvé 2026-07-21 : 'homme' → un sens sense_index=1 jamais
        # attesté pouvait battre 'cɛ̀' sense_index=2/freq=3223).
        candidates.sort(key=lambda x: (
            x['final_score'],
            _primacy_key(x.get('sense_index', 1), x.get('corpus_freq', 0), x.get('fr', ''))
        ), reverse=True)
        return candidates

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

        Trigger: ONLY when there is no reliable exact match at all (every
        candidate came from the embedding fallback). If any exact match is
        present, the KG's own ordering (score + corpus_freq/sense_index tie-
        break) is trusted as-is — the LLM is never asked to arbitrate between
        exact matches, since it has no real signal for near-duplicate glosses
        and can override a common word with a rare synonym at random.
        """
        if len(candidates) < 2:
            return candidates

        top       = candidates[0]
        top_score = top['final_score']
        all_embed = all(c.get('match') == 'embed' for c in candidates)
        exact_matches = [c for c in candidates if c.get('match') == 'exact']

        # Le LLM ne tranche QUE quand il n'y a AUCUN match exact fiable —
        # avec un exact match, l'ordre KG (score + tri corpus_freq/sense_index
        # déjà appliqué en amont) est le signal de confiance. Un appel LLM
        # pour départager deux exacts (même à égalité parfaite) n'a souvent
        # aucun signal sémantique réel à exploiter (gloses quasi-identiques,
        # ex: "argent" vs "argent.") et peut écraser un mot 100x plus fréquent
        # (wári, corpus_freq=1360) par un synonyme rare (ɲàga, corpus_freq=13)
        # au hasard (bug trouvé 2026-07-22).
        if exact_matches:
            return candidates

        # Exception : embed très haute confiance (score au max du barème
        # 0-100 pts) → le LLM ne peut pas faire mieux.
        if all_embed and top_score >= 100:
            return candidates

        should_rerank = all_embed  # tout embed → score peu fiable, toujours reranker

        if not should_rerank:
            return candidates

        # should_rerank == all_embed ici (seul cas restant) : pool serré,
        # 5 pts max (vs 10 avant) — un écart de 10 pts inclut des candidats
        # sémantiquement éloignés (ex: "réellement" → hàáli/très au lieu de
        # bɛ́rɛ/vraiment).
        rerank_pool = [c for c in candidates
                       if top_score - c['final_score'] <= 5]
        if len(rerank_pool) < 2:
            rerank_pool = candidates[:3]  # fallback : top 3 si pool trop petit

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
                    # Le rerank est un jugement sémantique explicite du LLM parmi
                    # un pool de candidats à score faible (souvent tous ~0 pts,
                    # cf. all_embed) — sans ce plancher, le choix validé est
                    # aussitôt rejeté par le seuil MIN_SCORE en aval et remplacé
                    # par un placeholder, rendant le rerank sans effet (bug
                    # observé sur 'appel' → wélewele choisi puis jeté).
                    if chosen['final_score'] < MIN_SCORE:
                        chosen['final_score'] = MIN_SCORE
                    print(f"      LLM reranked → #{chosen_idx+1} "
                          f"'{chosen['bm']}' ({chosen['fr']})")
        except Exception as e:
            print(f"       LLM rerank failed: {e}")

        return candidates

    # ------------------------------------------------------------------
    # SEMANTIC VALIDATION OF KG CANDIDATES
    # ------------------------------------------------------------------

    def _validate_candidate_semantics(self, token_fr: str,
                                      candidates: list,
                                      top_k: int = 5,
                                      tok: dict = None,
                                      all_tokens: list = None) -> list:
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

        def _ensure_min_score(c):
            # Un candidat validé sémantiquement par le LLM ne doit pas être
            # rejeté par le seuil MIN_SCORE en aval juste parce que son score
            # de récupération (souvent un embed faible) est bas — même
            # logique que le plancher du rerank ci-dessus (ligne ~548).
            if c.get('final_score', c.get('score', 0)) < MIN_SCORE:
                c['final_score'] = MIN_SCORE
                c['score'] = MIN_SCORE

        _tok_is_person_cache = {}

        def _tok_is_person():
            # Calculé au plus une fois par token (paresseux — coûte un appel
            # LLM), pas par candidat.
            if 'v' not in _tok_is_person_cache:
                _tok_is_person_cache['v'] = bool(
                    tok and tok.get('pos') == 'NOUN'
                    and self._classify_common_noun_is_person(token_lower))
            return _tok_is_person_cache['v']

        for i, c in enumerate(candidates[:top_k]):
            score = c.get('score', 0)

            # Perfect exact (100 pts) OU premier segment d'une glose multi-sens
            # (90 pts, ex: "jour" == 1er élément de "jour, date.") — les deux
            # sont des matches textuels structurellement certains, pas des
            # heuristiques floues à re-vérifier par LLM. Seuil relevé de 100
            # à 90 (bug trouvé 2026-07-17 : le LLM qwen2.5:3b invalidait à
            # tort 'jour, date.' pour 'jour', mettant 'dón' [fréquent, 3110
            # occurrences] à 0 pts au profit de 'dá' [rare, 27 occurrences]
            # qui passait à 100 pts et échappait à cette vérification).
            if score >= 90:
                continue

            # Décision 2026-07-20 : plus de plancher de score pour
            # proposer un candidat à la validation LLM — TOUT candidat
            # (y compris un embed à 5-15 pts) est soumis au LLM, qui
            # tranche seul par le sens plutôt qu'un seuil numérique
            # arbitraire pré-filtrant ce qui a même le droit d'être jugé.
            # Le score sert ensuite uniquement à départager les candidats
            # validés entre eux (ordre), pas à décider qui est validable.
            gloss_fr = c.get('fr', '').lower().rstrip('.').strip()

            # Identité orthographique avec la forme de SURFACE (pas le
            # lemme) : spaCy lemmatise l'adjectif à sa forme canonique
            # (masculin, "coopératif"), mais le KG peut ne stocker que la
            # forme fléchie réellement utilisée dans la phrase (ex:
            # "programme coopérative" → surface="coopérative", glose KG=
            # "coopérative." — taguée Noun dans le dico source, mais c'est
            # littéralement le même mot que l'adjectif cherché, variante de
            # genre). Une correspondance EXACTE avec la forme de surface est
            # aussi certaine structurellement qu'un match sur le lemme — le
            # LLM, lui, juge seulement le sens du gloss isolé et rejette à
            # tort ces variantes cross-POS car il ne voit pas ce contexte
            # flexionnel (bug trouvé 2026-07-20 : "programme coopératif" →
            # placeholder alors que 'kóperatifu' est la traduction correcte).
            _tok_surface = (tok.get('surface') or '').lower().strip() if tok else ''
            if _tok_surface and gloss_fr == _tok_surface:
                c['_llm_validated'] = True
                _ensure_min_score(c)
                print(f"      Identité de surface (variante flexionnelle) : "
                      f"'{gloss_fr}' == '{_tok_surface}' → {c['bm']}")
                continue

            # ADJ cherché vs candidat NOUN de la même famille dérivationnelle
            # (ex: "coopératif" vs "coopérative." — glosé comme une
            # INSTITUTION en français, un concept réellement différent de la
            # qualité "coopératif"). Le check générique juge à raison EN
            # FRANÇAIS que ces deux sens divergent — mais un emprunt bambara
            # ('kóperatifu') sert souvent aux deux catégories à la fois, sans
            # distinction de genre/POS. Décidé par similarité FastText
            # (composition en n-grammes de caractères, modèle entraîné sur
            # le corpus des gloses KG — cf. embeddings/fasttext_encoder.py),
            # PAS par similarité de forme brute (rejeté : la ressemblance
            # orthographique seule ne garantit pas le sens, ex. "sur"/"sûr")
            # ni par un prompt LLM à exemples figés (rejeté : mémorise des
            # mots précis au lieu de généraliser). Seuil 0.85 calibré sur
            # des paires connues : coopératif/coopérative=0.99, final/
            # finale=0.87 (racine partagée) vs cuisson/cuisant=0.41,
            # sur/sûr=0.31 (mots différents malgré la ressemblance). Bug
            # trouvé 2026-07-20 : "programme coopératif" → placeholder alors
            # que 'kóperatifu' est correct.
            if tok and tok.get('pos') == 'ADJ' and str(c.get('pos', '')).lower() in ('noun', 'nom'):
                _gloss_root = gloss_fr.split(',')[0].split()[0] if gloss_fr else ''
                from embeddings.fasttext_encoder import same_root, root_similarity
                if _gloss_root and same_root(token_lower, _gloss_root):
                    c['_llm_validated'] = True
                    _ensure_min_score(c)
                    print(f"      Même racine FastText ({root_similarity(token_lower, _gloss_root):.2f}) : "
                          f"'{_gloss_root}' ≈ '{token_lower}' → {c['bm']}")
                    continue

            # PERSONNE cherchée vs glose ABSTRAITE de même racine (action/
            # événement) : dérivation bambara 'tigi' ("possesseur/personne
            # associée à X") transforme un nom abstrait en référent-personne
            # — ex. jɔ̀yɔrɔ (participation) + tigi = jɔ̀yɔrɔtigi (celui qui a
            # une participation = un participant). Règle structurelle
            # générale (pas un mot précis) : seulement pour un candidat
            # EMBED à score significatif (similarité sémantique/lexicale
            # réelle établie, pas un nom abstrait choisi au hasard) — que ce
            # candidat vienne du repêchage cross-POS ou déjà du pool normal
            # (même POS=Noun des deux côtés, comme "participation"/
            # "participant"). Bug trouvé 2026-07-20 : "participants" →
            # 'jɔ̀yɔrɔ' rejeté à raison par le check générique (ce n'est PAS
            # un participant, c'est la participation elle-même) faute d'un
            # candidat KG "personne" générique — 'jɔ̀yɔrɔtigi' comble ce trou
            # sans rien halluciner de neuf.
            if (c.get('match') == 'embed' and c.get('score', 0) >= 25
                    and tok and tok.get('pos') == 'NOUN' and _tok_is_person()):
                _cand_is_person = self._classify_common_noun_is_person(gloss_fr)
                if not _cand_is_person:
                    c['bm'] = (c.get('bm') or '') + 'tigi'
                    c['_llm_validated'] = True
                    # Dérivation SYNTHÉTISÉE (jamais attestée telle quelle
                    # dans le KG) : ne doit jamais surclasser un candidat
                    # réellement attesté qui a, lui, passé la validation
                    # sémantique normale. Remise à l'échelle proportionnelle
                    # (facteur, pas plancher fixe) : un plancher unique à
                    # MIN_SCORE effaçait aussi l'ordre RELATIF entre
                    # dérivations tigi elles-mêmes, faisant perdre le
                    # meilleur match ("participation"→jɔ̀yɔrɔtigi) au profit
                    # d'un moins bon ("réunion"→jɛ̀ɛrɛtigi) une fois tous à
                    # égalité (bug trouvé 2026-07-23, "participants"). Un
                    # facteur multiplicatif préserve cet ordre tout en
                    # gardant l'ensemble sous un vrai mot attesté (ex.
                    # 'bànbaganci'=23.2 pts, non affecté ici) — corrige
                    # "djihadistes" → 'bólofaratigi' [affluent+tigi, gonflé
                    # à 29-32 pts] sans régresser "participants".
                    c['score'] = max(MIN_SCORE, round(c.get('score', 0) * 0.3, 1))
                    c['final_score'] = c['score']
                    print(f"      Personne↔abstrait : '{gloss_fr}' (non-personne) + "
                          f"'tigi' → '{c['bm']}' pour '{token_lower}' (personne)")
                    continue

            # Verbe réflexif ("il SE lève") : une glose "se {verbe}" n'est
            # PAS une variante restreinte du verbe nu à pénaliser — c'est
            # exactement le sens cherché ICI, le clitique réfléchi de la
            # phrase le confirme structurellement. Sans ce garde, la
            # validation générique (ci-dessous) traite "se lever" comme
            # un sens plus spécifique que "lever" (vrai en isolation :
            # "lever" nu peut aussi être transitif) et l'invalide au
            # profit d'un faux-ami non-réfléchi mieux placé dans une glose-
            # liste (bug trouvé 2026-07-20 : "il se lève" → 'láyɛ̀lɛn'
            # ["hausser, faire monter, lever"] au lieu de 'wúli' ["se
            # lever."], le vrai clitique réfléchi ignoré par la validation).
            _tok_is_reflexive_here = bool(tok) and any(
                x.get('dep') in ('expl:comp', 'expl:pass')
                and x.get('head_index') == tok.get('orig_index')
                and ('Reflex=Yes' in str(x.get('morph', ''))
                     or x.get('role') == 'reflexive')
                for x in (all_tokens or []))
            if _tok_is_reflexive_here and (
                    gloss_fr == f'se {token_lower}'
                    or gloss_fr == f"s'{token_lower}"):
                c['_llm_validated'] = True
                _ensure_min_score(c)
                print(f"      Réflexif structurel : '{gloss_fr}' ≈ 'se {token_lower}' → {c['bm']}")
                continue

            # Ask LLM: is gloss semantically equivalent to token?
            # The gloss may be a synonym list ("trop, très, beaucoup")
            # or a compound noun ("plaque de cuisson").
            #
            # max_tokens=3 forçait un verdict immédiat sans aucun
            # raisonnement — le petit modèle local (qwen2.5:3b) échouait
            # alors de façon déterministe (5/5) même sur des cas triviaux.
            # Autoriser un raisonnement bref avant un marqueur terminal
            # 'REPONSE=' (même technique que _classify_refl_verb) restaure
            # un verdict fiable — mais SEULEMENT si la question posée
            # correspond à la STRUCTURE réelle de la glose (bug trouvé
            # 2026-07-19, deux formes très différentes) :
            #   - glose-liste ("méticuleusement, clairement et dans le
            #     détail...") : le mot cherché est-il un des sens listés ?
            #     → "clairement" dans cette liste = OUI.
            #   - glose qualifiée SANS virgule ("gens de caste", "main
            #     droite") : le qualificatif RESTREINT le sens du mot nu
            #     à une sous-catégorie — ce n'est PAS un synonyme direct.
            #     Sans ce garde, "gens de caste" (caste artisanale
            #     spécifique) validait à tort pour "gens" (personnes en
            #     général), pareil "main droite" pour "main".
            # Piège supplémentaire au sein même des gloses-listes : un mot
            # peut apparaître à l'INTÉRIEUR d'un des segments sans EN
            # ÊTRE un lui-même (ex: "air sympathique, qualité d'être
            # sympathique, don de plaire aux gens" — "gens" y est le
            # complément du 3e segment, pas un synonyme listé de "gens" ;
            # contraste avec "clairement" qui EST le 2e segment de sa
            # liste). D'où "LUI-MÊME" : il ne suffit pas que le mot soit
            # présent dans le texte, il doit être un sens/synonyme à part
            # entière de la liste.
            _is_list_gloss = ',' in gloss_fr
            if _is_list_gloss:
                # Le verdict "LUI-MÊME" seul (sans exemples) oscillait selon
                # la formulation. Isoler le SEGMENT contenant le mot (au lieu
                # de faire raisonner le LLM sur la liste entière) + des
                # exemples travaillés (renforçateur→OUI vs complément d'un
                # autre mot→NON) stabilise le verdict. Exemples neutres
                # (aucun ne nomme le mot réellement recherché — sinon on
                # fige la réponse pour CE mot précis au lieu d'enseigner
                # la règle générale, cf. décision 2026-07-19 sur 'regarder'/
                # 'observer' : le fix doit généraliser, pas mémoriser un cas).
                import re as _re_seg
                _segments = [s.strip() for s in gloss_fr.split(',')]
                _segment = next(
                    (s for s in _segments
                     if _re_seg.search(r'(^|\s)' + _re_seg.escape(token_lower) + r'($|\s)', s)),
                    gloss_fr)
                # Identité triviale : le segment isolé EST le mot cherché,
                # sans rien d'autre (ex: glose "statut, situation." pour
                # "situation" → segment="situation"). Structurellement
                # certain, comme le score>=90 plus haut — inutile de
                # demander au LLM, qui produisait des réponses incohérentes
                # sur ce cas dégénéré non représenté dans les exemples
                # few-shot (bug trouvé 2026-07-20 : "situation" rejeté à
                # tort pour 'jɔ̀sen'/"statut, situation." et 'kísa').
                if _segment.strip().lower() == token_lower:
                    c['_llm_validated'] = True
                    _ensure_min_score(c)
                    print(f"      Segment = mot cherché (identité) : "
                          f"'{gloss_fr}' ⊇ '{token_lower}' → {c['bm']}")
                    continue
                prompt = (
                    f'Exemples :\n'
                    f'- Segment "encore aujourd\'hui", mot "aujourd\'hui" -> REPONSE=OUI '
                    f'(encore est un simple renforçateur, aujourd\'hui garde son sens plein)\n'
                    f'- Segment "envie de parler aux voisins", mot "voisins" -> REPONSE=NON '
                    f'(voisins est le complément de "parler à", le sens central du segment est '
                    f'"envie/désir de parler", pas "voisins")\n'
                    f'- Segment "très grand", mot "grand" -> REPONSE=OUI (très est un renforçateur)\n'
                    f'- Segment "chemin de fer", mot "fer" -> REPONSE=NON (fer est un complément '
                    f'du nom "chemin", le sens du segment est un TYPE de chemin, pas "fer")\n\n'
                    f'Maintenant applique la même logique :\n'
                    f'Segment "{_segment}", mot "{token_lower}" -> ?\n'
                    f'Réponds uniquement par REPONSE=OUI ou REPONSE=NON (pas d\'explication).'
                )
            else:
                prompt = (
                    f'Dans un dictionnaire, la glose d\'un mot bambara est: \"{gloss_fr}\".\n'
                    f'Le mot français cherché (SEUL, sans qualificatif) est: \"{token_lower}\".\n'
                    f'Réfléchis brièvement (1 phrase) : cette glose désigne-t-elle la même chose '
                    f'que \"{token_lower}\" EN GÉNÉRAL, ou seulement une catégorie/variante '
                    f'restreinte de \"{token_lower}\" (auquel cas ce N\'EST PAS un synonyme '
                    f'direct du mot seul) ?\n'
                    f'Termine ta réponse par une nouvelle ligne : REPONSE=OUI (synonyme direct, '
                    f'sens général identique) ou REPONSE=NON (sens plus restreint/spécifique)'
                )

            try:
                resp = self._call_llm(prompt, max_tokens=100).strip().upper()
                _tail = resp.rsplit('REPONSE=', 1)[-1] if 'REPONSE=' in resp else resp
                is_valid = _tail.strip().startswith('OUI')

                if is_valid:
                    c['_llm_validated'] = True
                    _ensure_min_score(c)
                    print(f"      LLM valide: '{gloss_fr}' ≈ '{token_lower}' → {c['bm']}")
                else:
                    # Candidate sémantiquement rejeté → exclu de la sélection
                    # via un flag, jamais en écrasant son score à 0 (le score
                    # d'origine reste une donnée de diagnostic valable — c'est
                    # l'exclusion de la liste plus bas qui l'empêche d'être
                    # choisi, pas une falsification de son score).
                    c['_llm_invalidated'] = True
                    invalidated_count += 1
                    # print(f"       LLM invalide: '{gloss_fr}' ≠ '{token_lower}' → {c['bm']} (exclu)")

            except Exception as e:
                print(f"       LLM validation failed: {e}")
                # LLM indisponible : score conservé tel quel (pas de pénalité
                # arbitraire ni de mise à 0), le candidat reste jugeable sur
                # son mérite de récupération.

        # FALLBACK: if all EXACT matches were invalidated and KG has no bare match,
        # restore only if the compound is linguistically very close to the bare token.
        # E.g., "faire peur" is close to "faire" (1 word apart, same starting word)
        # but "plaque de cuisson" is far from "cuisson" (2+ words apart, different starting word).
        # This handles compound verbs without hardcoding.
        exact_matches = [c for c in candidates[:top_k] if c.get('match') == 'exact']
        invalid_exact = sum(1 for c in exact_matches if c.get('_llm_invalidated'))

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

            # Contextual guard: the gloss's extra word(s) beyond the bare token
            # (e.g. "peur" in "faire peur") must be attested by a real complement
            # of this token in the sentence — not just lexical proximity in the
            # gloss text. Without this, "faire" in "qu'est-ce qu'il a fait ?"
            # (object = interrogative "que", no concrete complement) wrongly
            # restored "faire peur" → 'yògoro' just because it was the closest
            # gloss variant, regardless of what the sentence actually says.
            if is_close_variant and tok is not None and all_tokens:
                _extra_words = [w for w in words_in_gloss if w != token_lower]
                _tok_idx = tok.get('orig_index')
                # Include both lemma and surface: adjective lemmas get normalized
                # to masculine singular by spaCy (e.g. "douces" -> "doux"), which
                # doesn't lexically match a feminine gloss form like "patate douce".
                # Comparing surface forms too (via prefix match, since "douces"
                # starts with "douce") catches these irregular agreement cases.
                _complement_forms = {
                    (t.get('lemma') or t.get('bm') or '').lower()
                    for t in all_tokens
                    if t.get('head_index') == _tok_idx
                    and t.get('dep') in ('obj', 'nmod', 'obl:arg', 'xcomp', 'amod')
                    and t.get('role') not in ('interrogative', 'relative')
                } | {
                    (t.get('surface') or t.get('text') or '').lower()
                    for t in all_tokens
                    if t.get('head_index') == _tok_idx
                    and t.get('dep') in ('obj', 'nmod', 'obl:arg', 'xcomp', 'amod')
                    and t.get('role') not in ('interrogative', 'relative')
                }
                _forms_match = any(
                    ew == cf or cf.startswith(ew) or ew.startswith(cf)
                    for ew in _extra_words for cf in _complement_forms if cf
                )
                if _extra_words and not _forms_match:
                    is_close_variant = False
                    print(f"       [FALLBACK] '{best_gloss}' rejeté : aucun "
                          f"complément réel du token ne correspond à {_extra_words}")

            if best_orig_score > 0 and is_close_variant:
                restore_score = int(best_orig_score * 0.5)  # 50% of original
                best_exact['score'] = restore_score
                best_exact['final_score'] = restore_score
                best_exact['_llm_invalidated'] = False
                # Move this restored exact match to the front after re-sorting
                candidates.remove(best_exact)
                candidates.insert(0, best_exact)
                print(f"       [FALLBACK] Restauré exact match à {restore_score} pts "
                      f"(50% de {best_orig_score}): '{best_gloss}' est proche de '{token_lower}' "
                      f"(distance={word_distance})")

        # Candidats invalidés par le LLM et non restaurés ci-dessus : exclus
        # de la liste plutôt que laissés avec un score mis à 0 — le score
        # n'est jamais falsifié, un candidat rejeté disparaît simplement des
        # options considérées.
        candidates = [c for c in candidates if not c.get('_llm_invalidated')]

        # Re-sort by score — départage via _primacy_key (cf. décision 2026-07-21).
        candidates.sort(key=lambda x: (
            x.get('final_score', x.get('score', 0)),
            _primacy_key(x.get('sense_index', 1), x.get('corpus_freq', 0), x.get('fr', ''))
        ), reverse=True)

        # FINAL FALLBACK: if all exact matches failed validation AND we only have
        # embedding results left, prefer a placeholder over a weak embedding match.
        # This prevents "cuisson" from falling back to "jírisi" (menuiserie) just because
        # it's an embedding match — UNLESS that embedding candidate was itself
        # explicitly confirmed by the LLM check above (_llm_validated), in which
        # case wiping it out here would silently discard a positive semantic
        # judgment just because its retrieval score happens to be low (bug
        # trouvé 2026-07-19 : "fructueux" → nàfama validé " LLM valide:
        # 'profitable, rentable' ≈ 'fructueux'" puis effacé quand même par ce
        # garde, produisant un placeholder [fructueux] au lieu du mot validé).
        exact_matches = [c for c in candidates if c.get('match') == 'exact']
        embed_only = not exact_matches
        top_is_embed = candidates and candidates[0].get('match') == 'embed'
        top_score = candidates[0].get('final_score', 0) if candidates else 0
        top_validated = candidates and candidates[0].get('_llm_validated')
        # NOTE (bug trouvé 2026-07-20, revert) : un plancher basé sur
        # corpus_freq>0 pour contourner ce seuil a été essayé ici, dans
        # l'idée qu'une fréquence positive signale un "vrai" mot. Ça a
        # laissé passer 'pàriti'="parti." (candidat NON validé, jamais
        # passé par la boucle LLM ci-dessus) pour "participant" — la
        # fréquence atteste que le MOT existe, pas qu'il est sémantiquement
        # lié à la requête. Retour au comportement strict : sans validation
        # LLM positive, un embed <30 pts reste un placeholder.
        if embed_only and top_is_embed and top_score < 30 and not top_validated:
            # All exact matches failed validation; only weak embedding remains
            # Clear candidates to force placeholder fallback downstream
            print(f"      [NO FALLBACK TO EMBED] All exact matches invalidated & "
                  f"top embed too weak ({top_score} pts) → use placeholder instead")
            return []

        return candidates

    # ------------------------------------------------------------------
    # GLOSS MATCH VALIDATION (fast KG-label lookup path)
    # ------------------------------------------------------------------

    def _validate_gloss_match(self, token_fr: str, gloss_fr: str) -> bool:
        """
        Validate that a KG gloss genuinely carries the meaning of the
        source French word, for the quick surface/lemma/gloss-prefix
        lookup in _translate_token (bypasses the full retrieve+rerank
        pipeline). A dictionary gloss can textually contain the source
        word (e.g. "tout, tout entier") while the Bambara headword
        actually carries a narrower or idiomatic connotation not visible
        in the gloss text alone (e.g. bákuru ~ "tout perdu/ruiné", not a
        neutral pre-adjectival intensifier) — only a semantic check
        catches this, not string matching.
        """
        gloss_clean = (gloss_fr or '').lower().rstrip('.').strip()
        token_clean = (token_fr or '').lower().strip()
        if not gloss_clean or not token_clean:
            return bool(gloss_clean)

        prompt = (
            f'Dans un dictionnaire, la glose d\'un mot bambara est: "{gloss_clean}".\n'
            f'Le mot français cherché est: "{token_clean}".\n'
            f'Est-ce que "{token_clean}" correspond vraiment au sens de cette glose '
            f'dans un usage courant (pas seulement un chevauchement de mots) ?\n'
            f'Réponds uniquement par OUI ou NON.'
        )
        try:
            resp = self._call_llm(prompt, max_tokens=3).strip().upper()
            return resp.startswith('O')
        except Exception as e:
            print(f"       [KG-LABEL] Validation LLM échouée: {e}")
            # Conservatif : sans validation possible, ne pas faire confiance
            # à un match non-exact.
            return False

    # ------------------------------------------------------------------
    # SYNONYM FALLBACK
    # ------------------------------------------------------------------

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
                    print(f"   [POSS_TYPE KG] '{lemma}' → {_pt}")
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
                    print(f"   [POSS_TYPE] '{lemma}' sc={semantic_class!r} → {result}")
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
                print(f"      LLM classify? '{raw}'")
                result = raw if raw in ('STATIF', 'PARTICIPE', 'QUALITE', 'VALEUR') else 'QUALITE'
                return result
            except Exception as e:
                print(f"      attempt {_attempt+1} failed: {e}")

        return 'QUALITE'

    def _detect_classifying_adj(self, lemma: str) -> bool:
        """Returns True if the adjective is CLASSIFIANT (no -man), False if QUALIFIANT (needs -man).

        L'ancienne version demandait au LLM de trancher directement entre
        QUALIFIANT/CLASSIFIANT sur une définition abstraite (catégorie/
        nationalité/domaine vs qualité) — verdict non fiable et répétable à
        tort (température=0, vérifié via appel direct Ollama) pour "français"
        (répondait QUALIFIANT alors qu'il figurait en PREMIER exemple
        CLASSIFIANT de son propre prompt) ; une reformulation en termes de
        grammaire française pure ("adjectif qualificatif" vs "adjectif de
        relation") échoue aussi totalement (qwen2.5:3b répond QUALIFICATIF
        pour tous les mots testés, aucune discrimination). Remplacé par un
        test étroit et empiriquement fiable : "{lemma} est-il une couleur ?"
        (bug trouvé 2026-07-20, décision utilisateur : ne suffixer -man QUE
        pour les couleurs, classifiant par défaut sinon — cf. bìlen/bìlenman
        déjà présents comme entrées KG distinctes pour les couleurs)."""
        prompt = (
            'Q: "bleu" est-il une couleur ? R: OUI\n'
            'Q: "grand" est-il une couleur ? R: NON\n'
            f'Q: "{lemma}" est-il une couleur ? R:'
        )
        for _attempt in range(3):
            try:
                raw = self._call_llm(prompt, max_tokens=5).strip().upper()
                _is_color = raw.startswith('OUI')
                print(f"      adj_classify (couleur?) '{lemma}' → {'OUI' if _is_color else 'NON'}")
                return not _is_color
            except Exception as e:
                print(f"      adj_classify attempt {_attempt+1} failed: {e}")
        return True  # fallback: pas prouvé couleur → classifiant, pas de -man

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
        elif not self._detect_classifying_adj(tok.get('lemma', '')):
            # _detect_classifying_adj retourne False précisément pour les
            # couleurs (cf. sa docstring) — les couleurs sont STATIF en
            # bambara (V+-len/-nen dòn : "bìlenlen dòn"), PAS qualitatif
            # (S ka ADJ). _detect_statif_adj lui-même liste pourtant "rouge"
            # comme exemple QUALITE dans son propre prompt, garantissant la
            # mauvaise classification à chaque appel pour tout adjectif de
            # couleur (bug trouvé 2026-07-20 : "ce légume est rouge" →
            # "nin lègimu in ka bìlen" au lieu de "nin lègimu in bìlenlen
            # dòn", régression du fix couleur=statif du commit 07d26fe une
            # fois le tag KG semantic_class='color' disparu). Réutilise le
            # test couleur déjà validé fiable pour le suffixe -man.
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
                print(f"      reflexive_type? '{raw}'")
                result = raw if raw in ('RECIPROCAL', 'REFLEXIVE', 'PASSIVE', 'SUBJECTIVE') else 'SUBJECTIVE'
                return result
            except Exception as e:
                print(f"      attempt {_attempt+1} failed: {e}")
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

    def _classify_common_noun_is_person(self, surface: str, is_substantivized_adjective: bool = False) -> bool:
        """
        LLM en dernier recours pour pipeline/possession_ner.py::detect_category :
        un nom COMMUN français (pas une entité nommée) désigne-t-il une
        personne (ex: "frère", "maître", "ami") ? Appelé seulement quand
        spaCy NER + regex année + is_currency n'ont rien trouvé — voir le
        docstring du module pour la liste des alternatives non-LLM essayées
        et rejetées (spaCy NER/vecteurs, Stanza NER, LEFFF).

        is_substantivized_adjective : fait structurel transmis par l'appelant
        (le token est tagué ADJ par spaCy mais employé ici sans nom tête,
        ex: "le VIEUX" comme possesseur) — jamais une liste de mots figée,
        seulement l'information grammaticale réelle issue du parse.
        """
        if not surface:
            return False
        # Format Q/R avec exemples positifs (parenté ET rôle social) + un
        # négatif : la formulation déclarative précédente ("désigne-t-il...
        # Exemples OUI/NON...") répondait NON à tort et de façon répétable
        # (température=0, pas un aléa — un biais du prompt, vérifié via appel
        # direct Ollama) pour des noms de parenté pourtant listés dans ses
        # propres exemples ("mère" — bug trouvé 2026-07-20, même famille que
        # le biais déjà corrigé pour _classify_noun_is_object). Le style Q/R
        # ci-dessous suit le patron déjà validé par l'utilisateur pour cette
        # même classe de biais.
        _structural_hint = (
            f'Note grammaticale : "{surface}" est ici un adjectif employé '
            f'sans nom, donc substantivé (il joue le rôle d\'un nom).\n'
            if is_substantivized_adjective else ''
        )
        prompt = (
            f'{_structural_hint}'
            'Q: "frère" désigne-t-il une personne ? R: OUI\n'
            'Q: "roi" désigne-t-il une personne ? R: OUI\n'
            'Q: "voiture" désigne-t-il une personne ? R: NON\n'
            f'Q: "{surface}" désigne-t-il une personne ? R:'
        )
        for _ in range(2):
            result = self._call_llm(prompt, max_tokens=5).strip().upper()
            if 'OUI' in result:
                print(f"   [PER_NOUN] '{surface}' → personne (OUI)")
                return True
            if 'NON' in result:
                print(f"   [PER_NOUN] '{surface}' → non-personne (NON)")
                return False
        return False

    def _classify_noun_is_object(self, lemma: str) -> bool:
        """
        Sous un déterminant possessif pronominal ("sa maison", "mes devoirs"),
        le possesseur est toujours une personne — seul le nom possédé décide
        de l'aliénabilité. Une partie du corps ou un lien de parenté est
        inaliénable (pas de 'ka' : "n bolo" = ma main) ; un objet/possession
        ordinaire est aliénable ("n ka bìlakun" = mes devoirs).
        """
        if not lemma:
            return False
        # Nom COLLECTIF (famille, équipe, communauté, peuple...) : associé au
        # vocabulaire de la parenté/des personnes mais désigne un GROUPE, pas
        # un lien de parenté individuel (père, frère) — la question corps/
        # parenté ci-dessous répond systématiquement OUI pour "famille" faute
        # de cette distinction (bug trouvé 2026-07-20, décision utilisateur :
        # "ma famille" doit être ALIÉNABLE avec 'ka', contrairement à un lien
        # de parenté individuel). Question isolée sans exemple contrastif
        # (aucun mot de parenté à citer comme contre-exemple) — vérifiée
        # stable sur famille/équipe/communauté/peuple (OUI) vs jambe/main/
        # livre/voiture (NON).
        _collective_prompt = (
            f'Est-ce que "{lemma}" désigne un groupe de plusieurs personnes '
            f'(pas un individu) ?\nRéponds UNIQUEMENT par OUI ou NON.'
        )
        _collective_result = self._call_llm(_collective_prompt, max_tokens=5).strip().upper()
        if _collective_result.startswith('OUI'):
            print(f"   [ALIENABLE_NOUN] '{lemma}' → objet (aliénable, nom collectif)")
            return True
        # Format Q/R avec exemples positifs (corps ET parenté) + un négatif :
        # une formulation déclarative ("désigne-t-il...Exemples OUI/NON...")
        # répondait NON à tort et de façon répétable pour "main"/"tête" même
        # à température=0 (pas un aléa — un biais du prompt). Le style Q/R
        # ci-dessous a été vérifié manuellement stable sur main/tête/œil/
        # frère/père/devoir/sac/voiture/maison.
        prompt = (
            'Q: "jambe" est-il une partie du corps ou un lien de parenté ? R: OUI\n'
            'Q: "père" est-il une partie du corps ou un lien de parenté ? R: OUI\n'
            'Q: "livre" est-il une partie du corps ou un lien de parenté ? R: NON\n'
            f'Q: "{lemma}" est-il une partie du corps ou un lien de parenté ? R:'
        )
        result = self._call_llm(prompt, max_tokens=5).strip().upper()
        _is_object = 'NON' in result
        _mark = 'objet (aliénable)' if _is_object else 'corps/parenté (inaliénable)'
        print(f"  {'' if _is_object else ''} [ALIENABLE_NOUN] '{lemma}' → {_mark}")
        return _is_object

    def _detect_relational_noun(self, tok, bm: str) -> bool:
        """
        Détermine si un nom est INALIENABLE en bambara → pas de 'ka'.

        Bug fixé 2026-07-07 : l'ancienne heuristique (LLM + liste lexicale
        parenté/corps vs objets physiques) ne regardait QUE le nom possédé,
        jamais le possesseur — "les maîtres du jeu" traitait 'maître' comme
        ALIÉNABLE isolément, ignorant que le possesseur ('jeu') n'est même
        pas une entité nommée. Remplacé par une table possesseur×possédé
        basée sur la catégorie NER des deux (voir pipeline/possession_ner.py) :
        l'aliénabilité dépend de la PAIRE de catégories, pas du nom possédé
        seul (ex: PER possesseur + tout sauf PER → ALIÉNABLE ; presque tout
        le reste → NON-ALIÉNABLE).

        Retourne True  → inalienable → pas de 'ka'
        Retourne False → aliénable   → 'ka' requis
        Fallback : True (pas de 'ka') — plus sûr grammaticalement.
        """
        lemma = tok.get('lemma', '') if isinstance(tok, dict) else str(tok)
        if not lemma or not bm:
            return True

        from pipeline.possession_ner import detect_category, is_alienable

        # Lemme (forme canonique singulier), pas surface fléchie : le pluriel
        # ("enfants") fait échouer la classification LLM person alors que le
        # lemme singulier ("enfant") — pourtant l'exemple même du prompt —
        # réussit à 100% (bug trouvé 2026-07-17, reproductible à chaque essai,
        # pas de la non-déterminisme LLM comme d'abord supposé).
        possessed_cat = detect_category(
            lemma, self._classify_common_noun_is_person)

        # Chercher le possesseur : nom en dep='nmod' dont le head est ce token
        # ("le maître DU JEU" : jeu.dep='nmod', jeu.head_index=maître.orig_index).
        # ADJ inclus : un adjectif substantivé ("le VIEUX", "le jeune"...) peut
        # lui-même être le possesseur ("le champ DU VIEUX" = "the old man's
        # field") — spaCy le tague ADJ (nature lexicale) même en emploi
        # nominal, donc l'exclure ratait ces cas et retombait à tort sur "pas
        # de possesseur trouvé" → aucun 'ka' (bug pré-existant, corrigé
        # 2026-07-16).
        _clause_toks = getattr(self, '_current_clause_tokens', [])
        _tok_idx = tok.get('orig_index') if isinstance(tok, dict) else None
        _possessor_tok = next((t for t in _clause_toks
                               if t.get('dep') == 'nmod'
                               and t.get('head_index') == _tok_idx
                               and t.get('pos') in ('NOUN', 'PROPN', 'ADJ')), None)

        if _possessor_tok is not None:
            _possessor_is_adj = _possessor_tok.get('pos') == 'ADJ'
            if _possessor_is_adj:
                # Adjectif substantivé employé comme possesseur ("le champ DU
                # VIEUX", "la maison DU JEUNE") : structurellement, un ADJ
                # sans nom tête employé en position de possesseur nominal
                # désigne quasi-toujours une personne — pas besoin du LLM
                # (qui s'est trompé sur "vieux", classé à tort non-personne).
                possessor_cat = 'PER'
            else:
                _possessor_lemma = _possessor_tok.get('lemma') or _possessor_tok.get('surface', '')
                possessor_cat = detect_category(
                    _possessor_lemma, self._classify_common_noun_is_person)
        else:
            # Pas de possesseur nominal explicite trouvé (déterminant possessif
            # pronominal : "sa maison", "mes devoirs"...). Le possesseur est
            # toujours une PERSONNE (son/ma/ton...). Si le possédé DÉNOTE
            # LUI-MÊME une personne (ami, collègue, voisin...), la même règle
            # PER×PER=inaliénable que pour un possesseur nominal explicite
            # ("le fils DU ROI" → sans 'ka') s'applique directement — "son
            # ami" est structurellement identique à "l'ami DE PIERRE" une
            # fois le possesseur substitué par un pronom, ce n'est pas parce
            # que le déterminant est pronominal que la paire de catégories
            # change (bug trouvé 2026-07-20 : "ses amis" recevait 'ka' à
            # tort, la vérification corps/parenté ne reconnaissant "ami" ni
            # comme partie du corps ni comme lien de parenté). Seulement si
            # le possédé N'EST PAS une personne (maison, devoirs, main...) on
            # retombe sur le test corps/parenté vs objet, qui lui distingue
            # correctement une partie du corps (inaliénable) d'un bien
            # ordinaire (aliénable) — distinction que la table PER/ORG/LOC
            # ne sait pas faire (aucune des deux n'étant "PER").
            _possede_is_person = detect_category(
                lemma, self._classify_common_noun_is_person) == 'PER'
            if _possede_is_person:
                _alienable_pp = is_alienable('PER', 'PER')
                _mark = "ALIÉNABLE (avec 'ka')" if _alienable_pp else "INALIENABLE (sans 'ka')"
                print(f"   [RELATIONAL] possessif pronominal × possédé='{lemma}'(PER) → {_mark}")
                return not _alienable_pp
            _is_object = self._classify_noun_is_object(lemma)
            _mark = "ALIÉNABLE (avec 'ka')" if _is_object else "INALIENABLE (sans 'ka')"
            print(f"   [RELATIONAL] possessif pronominal × possédé='{lemma}' → {_mark}")
            return not _is_object

        _alienable = is_alienable(possessor_cat, possessed_cat)
        _mark = 'ALIÉNABLE (avec \'ka\')' if _alienable else "INALIENABLE (sans 'ka')"
        print(f"   [RELATIONAL] possesseur={_possessor_tok.get('surface')}"
              f"({possessor_cat}) × possédé='{lemma}'({possessed_cat}) → {_mark}")
        return not _alienable

    def _detect_intransitive_type(self, lemma: str, semantic_class: str = '') -> str:

        # ── PRIORITÉ CLASSE SÉMANTIQUE : VerbNet détermine la transitivité ────
        # Niveau 1 — classes intransitives pures → ABSOLU (jamais de COD).
        _INTRANSITIVE_SC = {'motion', 'biological', 'posture', 'spontaneous',
                            'meteorological'}
        if semantic_class in _INTRANSITIVE_SC:
            print(f"   [TRANSITIVITY] '{lemma}' class={semantic_class} "
                  f"→ ABSOLU (classe autonome, LLM ignoré)")
            return 'ABSOLU'

        # Niveau 2 — classes transitives → ACTION (prennent un COD ; sans COD
        # → V+li kɛ / action_noun kɛ dans step7_final). Pas de LLM nécessaire.
        _TRANSITIVE_SC = {'consumption', 'preparation', 'action', 'craft', 'perception'}
        if semantic_class in _TRANSITIVE_SC:
            print(f"   [TRANSITIVITY] '{lemma}' class={semantic_class} "
                  f"→ ACTION (classe transitive, LLM ignoré)")
            return 'ACTION'

        # ── LLM : présence d'un COD (complément d'objet direct) ──────────────
        # Note : ne pas inclure la semantic_class dans le prompt — le label 'action'
        # induit le LLM à répondre ACTION même pour les intransitifs de classe action
        # (travailler, courir, danser…). La transitivité s'évalue indépendamment.
        prompt = (
            f"Le verbe français '{lemma}' peut-il prendre un COD (complément d'objet direct) ?\n"
            f"ABSOLU  → jamais de COD (intransitif strict) : courir, dormir, régner, exister\n"
            f"ACTION  → COD possible (transitif) : manger, couper, aider\n"
            f"Réponds UNIQUEMENT par ABSOLU ou ACTION."
        )
        _result = None
        _raw = ''
        for attempt in range(3):
            try:
                _raw = self._call_llm(prompt, max_tokens=5).strip().upper()
                print(f"   [TRANSITIVITY raw] attempt {attempt+1}: {_raw!r}")
                if _raw.startswith('ABSOLU'):
                    _result = 'ABSOLU'; break
                if _raw.startswith('ACTION'):
                    _result = 'ACTION'; break
            except Exception:
                continue

        verdict = _result
        print(f"   [TRANSITIVITY] LLM verdict for '{lemma}' (class={semantic_class}) → {verdict}  [raw: {_raw!r}]")
        return verdict

    def _classify_refl_verb(self, lemma: str, context: str = '') -> str:
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
        # Le verbe seul est souvent ambigu/figuré (sens propre vs métaphorique)
        # → le LLM tranche sans contexte et varie d'un run à l'autre. Fournir
        # la phrase complète quand disponible ancre le verdict dans l'usage réel.
        _context_line = f'Phrase : "{context}"\n\n' if context else ''
        prompt = (
            f'{_context_line}'
            f'Verbe réfléchi à classer (dans le sens où il est employé ci-dessus '
            f'si une phrase est donnée) : "{lemma}". Catégorie :\n\n'
            f'PRONOMINAL : "se" transforme V transitif en son équivalent INTRANSITIF automatique\n'
            f'  (le sujet subit l\'événement sans agir délibérément sur lui-même),\n'
            f'  OU verbe impossible sans "se", OU sens différent de V.\n'
            f'  Ex: réveiller qqn → se réveiller (le réveil arrive), endormir → s\'endormir,\n'
            f'      fermer → se fermer, évanouir → s\'évanouir, taire → se taire, tromper → se tromper,\n'
            f'      souvenir → se souvenir (aucun sens sans "se").\n'
            f'  ATTENTION : si le verbe garde EXACTEMENT son sens transitif normal, avec "se"\n'
            f'  comme simple complément d\'objet réfléchi (le sujet fait V à lui-même au lieu\n'
            f'  de qqn d\'autre — ex: juger qqn → se juger = juger soi-même), CE N\'EST PAS\n'
            f'  pronominal → c\'est ACTIF.\n\n'
            f'SOIN_CORPOREL : action d\'hygiène/toilette ACTIVE et intentionnelle sur son propre corps,\n'
            f'  où les MAINS manipulent physiquement le corps (laver, raser, coiffer...).\n'
            f'  Ex: laver, raser, coiffer, maquiller, habiller, brosser.\n'
            f'  ATTENTION : la perception sensorielle SEULE (voir/regarder/écouter, même dans\n'
            f'  un miroir, sans manipulation physique) N\'EST PAS du soin corporel → ACTIF.\n\n'
            f'POSTURE : UNIQUEMENT changement de POSITION PHYSIQUE du corps humain/animal\n'
            f'  (où le corps EST : debout, assis, allongé, penché...), PAS un déplacement\n'
            f'  vers/dans un lieu ou un état.\n'
            f'  Ex: asseoir (debout→assis), coucher (debout→allongé), pencher, agenouiller.\n'
            f'  ATTENTION : réveiller et endormir NE SONT PAS des postures (états de conscience) ;\n'
            f'  voir, regarder, écouter, sentir NE SONT PAS des postures (perception sensorielle,\n'
            f'  aucun changement de position du corps) — ce sont des actions intentionnelles\n'
            f'  du sujet (→ ACTIF quand dirigées sur soi-même).\n\n'
            f'ACCIDENTEL : dommage, dégradation ou détérioration progressive que subit le sujet\n'
            f'  (volontaire ou non), y compris au sens figuré (le sujet SUBIT un processus\n'
            f'  négatif progressif, ce n\'est PAS une posture du corps, même si le verbe\n'
            f'  évoque littéralement un mouvement ou une position).\n'
            f'  → blesser, couper, brûler, casser (bras/jambe), écorcher, sombrer, effondrer,\n'
            f'  dégrader = TOUJOURS ACCIDENTEL.\n\n'
            f'ACTIF : autre action intentionnelle du sujet sur lui-même, y compris une\n'
            f'  perception dirigée sur soi (le sujet observe/perçoit sa propre personne).\n'
            f'  Ex: déguiser, défendre, préparer (mental), observer qqn → s\'observer\n'
            f'  (percevoir soi-même, ex. dans un miroir).\n\n'
            f'Réfléchis brièvement (1-2 phrases) à ce que fait le sujet, PUIS termine\n'
            f'ta réponse par une nouvelle ligne exactement : CATEGORIE=<mot>\n'
            f'où <mot> est UN SEUL MOT parmi : PRONOMINAL, SOIN_CORPOREL, POSTURE, ACCIDENTEL, ACTIF.'
        )

        _CATS = ('PRONOMINAL', 'SOIN_CORPOREL', 'POSTURE', 'ACCIDENTEL', 'ACTIF')
        _result = None
        _raw = ''
        for attempt in range(3):
            try:
                # max_tokens=6 empêchait tout raisonnement et forçait un
                # verdict immédiat, souvent faux sur ce petit modèle local
                # (bug trouvé 2026-07-19 : "se regarder" → POSTURE/PRONOMINAL
                # à froid, mais ACTIF une fois qu'un raisonnement bref est
                # autorisé avant la catégorie finale). Le marqueur
                # 'CATEGORIE=' ancre l'extraction sur le VERDICT FINAL plutôt
                # que sur un nom de catégorie mentionné en passant pendant le
                # raisonnement ("ce n'est pas PRONOMINAL, donc...").
                _raw = self._call_llm(prompt, max_tokens=120, timeout=20).strip().upper()
                print(f"   [REFL_CAT raw] attempt {attempt+1}: {_raw!r}")
                _tail = _raw.rsplit('CATEGORIE=', 1)[-1] if 'CATEGORIE=' in _raw else _raw
                _hit = next((c for c in _CATS if c in _tail), None)
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
        print(f"   [REFL_CAT] '{lemma}' → {cat}  [LLM, raw={_raw!r}]")
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
            f'Exemples (formes conjuguées):\n'
            f'- lave, laves, lavent → laver\n'
            f'- mange, manges, mangent → manger\n'
            f'- viens, venons, vient → venir\n'
            f'- suis, sommes, êtes, sont → être\n'
            f'- ai, avons, avez, ont → avoir\n\n'
            f'Exemples (participes passés irréguliers, parfois employés comme '
            f'adjectifs — attention aux verbes en -uire/-uire dont le participe '
            f'ne se devine pas par simple suffixe "-er"):\n'
            f'- cuit, cuite, cuits, cuites → cuire\n'
            f'- fait, faite → faire\n'
            f'- pris, prise → prendre\n'
            f'- mis, mise → mettre\n'
            f'- dit, dite → dire\n'
            f'- écrit, écrite → écrire\n'
            f'- ouvert, ouverte → ouvrir\n'
            f'- né, née → naître\n'
            f'- mort, morte → mourir\n\n'
            f'Réponds UNIQUEMENT par l\'infinitif (un seul mot), minuscules, sans ponctuation.'
        )

        for attempt in range(3):
            try:
                infinitive = self._call_llm(prompt, max_tokens=8).strip().lower()
                if infinitive and len(infinitive) > 1:
                    return infinitive
            except Exception as e:
                print(f"       Infinitive normalization failed (attempt {attempt+1}): {e}")

        return verb

    def _detect_semantic_class(self, lemma: str) -> str:
        """Détecte la classe sémantique d'un verbe. Pré-check KG d'abord
        (Sense.semantic_class) — évite un appel LLM non-déterministe pour
        les lemmes déjà classés une fois (ex: 'pleurer' -> 'spontaneous'
        reclassé différemment à chaque appel LLM, cassant tour à tour la
        transitivité, le réflexif, etc. selon la session)."""
        if not lemma:
            return 'other'
        _VALID_SC = {
            'motion', 'biological', 'posture', 'spontaneous', 'perception',
            'meteorological', 'copula', 'stative_cognitive', 'psych_emotion', 'modal', 'obligation',
            'action', 'consumption_liquid', 'consumption',
            'preparation', 'technique', 'craft', 'communication', 'communication_transitive',
            'having', 'giving', 'other', 'color',
        }
        try:
            _kg_sc = self.db.query(
                "MATCH (n:Sense) WHERE (toLower(n.fr) = toLower($fr) "
                "OR toLower(n.fr) = toLower($fr) + '.') AND n.semantic_class IS NOT NULL "
                "RETURN n.semantic_class AS sc "
                "ORDER BY CASE WHEN toLower(n.fr) = toLower($fr) THEN 0 ELSE 1 END "
                "LIMIT 1",
                {'fr': lemma})
            if _kg_sc and _kg_sc[0].get('sc'):
                _sc = str(_kg_sc[0]['sc']).lower()
                if _sc in _VALID_SC:
                    print(f"       semantic_class('{lemma}') = {_sc}  [KG]")
                    return _sc
        except Exception:
            pass
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
            f'  spontaneous=réaction involontaire (rire, crier...)\n'
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
            f'     NE PAS classer ici les verbes qui prennent normalement un COD :\n'
            f'    acheter, vendre, prendre, chercher, trouver → utiliser "other"\n'
            f'    (donner, envoyer, offrir… → utiliser "giving", voir plus bas).\n'
            f'  consumption_liquid=ingestion de LIQUIDE (boire, siroter — jamais consumption pour boire)\n'
            f'  consumption=ingestion SOLIDE uniquement (manger, croquer, dévorer, avaler qqch de solide — ≠ boire)\n'
            f'  preparation=transformation (cuisiner, préparer...)\n'
            f'  technique=travail spécialisé (construire, réparer...)\n'
            f'  craft=création artistique (peindre, écrire...)\n'
            f'  communication=parole SANS lien fixe entre sujet et interlocuteur '
            f'(dire, raconter, demander...)\n'
            f'  communication_transitive=verbe de DÉNOMINATION qui attribue une '
            f'étiquette/un nom à quelqu\'un ou dont l\'action lie durablement deux '
            f'personnes l\'une à l\'autre (appeler [nommer], nommer, surnommer, nomination).\n'
            f'     "appeler" au sens de "téléphoner à" ou "héler" reste communication.\n'
            f'  having=ÉTAT de POSSESSION STATIQUE uniquement (posséder, détenir, contenir,\n'
            f'    appartenir, garder, tenir). Avoir = having UNIQUEMENT au sens possessif.\n'
            f'     acheter ≠ having (acheter = transaction → other)\n'
            f'  giving=TRANSFERT DE POSSESSION vers un destinataire — VerbNet classe 13.1\n'
            f'    (Give Verbs) : donner, envoyer, offrir, prêter, remettre, céder, expédier,\n'
            f'    livrer, transmettre, rendre. Le sujet cesse de posséder, le destinataire\n'
            f'    commence à posséder — jamais "having" (qui est un ÉTAT statique sans\n'
            f'    transfert ni destinataire).\n'
            f'     donner ≠ having (donner = transfert → giving, jamais possession statique)\n'
            f'  other=verbe transitif direct standard sans catégorie propre :\n'
            f'Réponds UNIQUEMENT par le nom de la catégorie.'
        )
        try:
            import re as _re
            cls_raw = self._call_llm(prompt, max_tokens=15).strip().lower()
            # Inclure '_' pour préserver 'consumption_liquid' (ne pas split en 'consumption')
            words   = _re.findall(r'[a-z_]+', cls_raw)
            # Chercher la première correspondance exacte avec une classe valide
            cls = next((w for w in words if w in _VALID_SC), 'other')

            # 'having' route tout root_tok vers noun_phrase_have (rules/core.py
            # _is_avoir), une reclassification structurelle lourde — un faux
            # positif y est bien plus coûteux qu'ailleurs (le vrai verbe
            # disparaît du rendu). La classification multi-catégories ci-
            # dessus s'est montrée non-fiable spécifiquement pour ce bucket
            # (bug trouvé 2026-07-20 : "bénir" classé 'having' → "Que Dieu
            # vous bénisse" rendu "aw bɛ Dieu", verbe et sens perdus). Second
            # avis via un test étroit OUI/NON, même patron que le test
            # couleur déjà validé fiable pour le suffixe -man.
            if cls == 'having':
                _confirm_prompt = (
                    'Q: "posséder" signifie-t-il avoir/détenir quelque chose ? R: OUI\n'
                    'Q: "donner" signifie-t-il avoir/détenir quelque chose ? R: NON\n'
                    f'Q: "{lemma}" signifie-t-il avoir/détenir quelque chose ? R:'
                )
                _confirm = self._call_llm(_confirm_prompt, max_tokens=5).strip().upper()
                if not _confirm.startswith('OUI'):
                    print(f"       semantic_class('{lemma}') = having rejeté "
                          f"(second avis: NON) → other  [LLM]")
                    return 'other'

            print(f"       semantic_class('{lemma}') = {cls}  [LLM]")
            return cls
        except Exception as e:
            print(f"       semantic class detection failed: {e}")
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

    def _classify_verb_transitivity(self, tok: dict) -> None:
        """Classifie intransitive_type + action_noun pour un VERB dont le bm est
        déjà connu. Centralisé et appelé depuis LES DEUX chemins de _translate_token
        (bm pré-assigné par le parseur/KG label, ou résolu ici via KG retrieval)
        pour que la même règle grammaticale (transitivité, COD, semantic_class)
        s'applique quel que soit l'endroit du pipeline où le verbe apparaît —
        ROOT d'une clause simple ou xcomp infinitif sous un modal ('il faut
        manger'). Avant cette centralisation, le chemin bm-pré-assigné ne posait
        que semantic_class et retournait immédiatement, sans jamais classifier
        intransitive_type/action_noun → 'manger' sans COD sous 'falloir' perdait
        la nominalisation irrégulière (dumuni) que 'il mange' recevait bien.
        """
        # Ne pas recalculer si déjà posé (ex: depuis s.semantic_class du KG,
        # spécifiquement pour éviter un appel LLM nondéterministe redondant).
        if not tok.get('semantic_class'):
            tok['semantic_class'] = self._detect_semantic_class(tok['lemma'])
        # Validation linguistique : stative_cognitive exige un sujet Experiencer.
        # Si context_type='agent' (le sujet fait l'action), c'est une contradiction
        # → le verbe est une activité dynamique, pas un état cognitif statique.
        if (tok['semantic_class'] == 'stative_cognitive'
                and tok.get('context_type') == 'agent'):
            tok['semantic_class'] = 'action'
            print(f"       semantic_class('{tok['lemma']}') reclassifié "
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
                print(f"       participe_statif('{tok['lemma']}') = True")

        # Signal lexical KG : nom support dédié → action_noun + kɛ
        try:
            _res = self.db.query(
                "MATCH (s:Sense {bm: $bm, pos: 'Verb'}) "
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

        # Catégorie B3 : nominalisation -li/-ni → V+li + kɛ
        elif _sc in ('action', 'technique', 'craft', 'communication'):
            tok['intransitive_type'] = 'nominalized'

    def _try_deverbal_noun_fallback(self, verb_lemma: str) -> dict:
        """Verbe sans AUCUN candidat KG direct de pos=Verb (ex: 'recruter') :
        retenter la recherche embedding SANS filtre de POS. Le nom d'action
        dérivé (recrutement, organisation, nettoyage...) a souvent une
        similarité embedding TRÈS élevée avec le verbe source ("recruter" vs
        "recrutement." → cosine 0.93) mais était structurellement exclu par
        le filtre spacy_pos='VERB' de la recherche principale (allowed_pos=
        ['Verb'] uniquement) — le candidat ne participait même pas au calcul
        de similarité. Utilisé ensuite comme nom-support (V_ACTION kɛ), même
        mécanisme que Sense.action_noun déjà pour 'manger'→dumuni kɛ.
        Bug trouvé 2026-07-20 : 'recruter' n'a ni entrée verbe directe ni
        action_noun KG ; 'recrutement.'→cɛ̀ta existe et matche très fort en
        embedding, mais jamais vu par la recherche filtrée pos=Verb."""
        if not verb_lemma:
            return None
        try:
            _cands = self.retriever.retrieve(
                verb_lemma, spacy_pos=None, top_k=5, lang='fr')
        except Exception:
            return None
        _noun_cands = [c for c in _cands
                       if str(c.get('pos', '')).lower() in ('noun', 'nom')
                       and c.get('match') == 'embed' and c.get('bm')]
        if not _noun_cands:
            return None
        best = _noun_cands[0]
        if best.get('score', 0) < 30:  # cosine < 0.75 sur l'échelle 0-40 pts
            return None
        return {'bm': best['bm'], 'fr': best.get('fr', ''),
                'final_score': MIN_SCORE, 'score': MIN_SCORE,
                'match': 'deverbal_noun'}

    def _translate_token(self, tok: dict,
                         context_lemmas: list, all_tokens: list = None):
        surface = tok['surface']
        lemma   = tok['lemma']
        lang    = tok.get('lang', 'fr')
        pos     = tok['pos']

        # DEBUG: Track which tokens are processed
        print(f"     [TRANSLATE_TOKEN] surface='{surface}' lemma='{lemma}' pos={pos} dep={tok.get('dep')}")

        if tok.get('bm'):
            if pos == 'NOUN' and not tok['bm'].startswith('[') and 'is_relational' not in tok:
                tok['is_relational'] = self._detect_relational_noun(tok, tok['bm'])
            elif pos == 'ADJ':
                self._classify_adj_state(tok)
                # Classification épithète QUALIFIANT/CLASSIFIANT manquait sur
                # ce chemin bm-pré-assigné (bug trouvé 2026-07-19 : "français",
                # "international" en amod prenaient toujours le suffixe -man
                # comme un adjectif qualifiant ordinaire, faute d'avoir jamais
                # atteint ce check plus bas dans la fonction, réservé au
                # chemin de retrieval complet).
                if tok.get('dep') == 'amod' and 'is_classifying_adj' not in tok:
                    if self._detect_classifying_adj(tok['lemma']):
                        tok['is_classifying_adj'] = True
            elif pos == 'VERB' and 'semantic_class' not in tok:
                # bm pré-assigné (parseur/KG) : détecter quand même la classe
                # sémantique pour que la transitivité (li kɛ / la / nu) soit juste.
                # Normaliser d'abord le lemme (forme fléchie → infinitif) pour que
                # VerbNet trouve le verbe ; sinon 'venue' → VerbNet miss → LLM.
                _pre_lem = tok['lemma']
                _pre_inf = self._normalize_verb_to_infinitive(_pre_lem)
                if _pre_inf and _pre_inf != _pre_lem:
                    tok['lemma'] = _pre_inf
                # Classification centralisée (intransitive_type + action_noun) :
                # un verbe avec bm déjà posé (ex: infinitif xcomp sous 'falloir')
                # doit recevoir EXACTEMENT la même règle grammaticale qu'un verbe
                # résolu par KG retrieval plus bas dans cette fonction — sinon
                # "il ne faut pas manger" perdait le nom d'action (dumuni) et le
                # statut support-verbe que "il mange" recevait correctement.
                self._classify_verb_transitivity(tok)
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
            # La plupart des PROPN (noms propres/personnes/lieux : Marie,
            # Moussa) n'ont pas de "traduction" — la forme française est
            # gardée telle quelle. Mais certains PROPN dénotent un référent
            # commun que le KG traduit explicitement (ex: "Dieu" → 'Ála',
            # Sense.fr='Dieu.'), pas un nom propre arbitraire. Match EXACT
            # uniquement (pas de fuzzy/substring) pour ne jamais mistraduire
            # un vrai nom de personne qui ressemblerait à un mot commun (bug
            # trouvé 2026-07-20 : "Que Dieu vous bénisse" gardait "Dieu" en
            # français au lieu de 'Ála', faute de tenter le KG pour les PROPN).
            _lemma_propn = tok.get('lemma') or surface
            try:
                _propn_res = self.db.query(
                    "MATCH (n:Sense) WHERE toLower(n.fr) = toLower($lemma) "
                    "OR toLower(n.fr) = toLower($lemma) + '.' "
                    "RETURN n.bm AS bm "
                    "ORDER BY CASE WHEN coalesce(n.corpus_freq,0) > 0 THEN 0 ELSE 1 END, "
                    "coalesce(n.sense_index,1) ASC, "
                    "coalesce(n.corpus_freq,0) DESC LIMIT 1",
                    {'lemma': _lemma_propn})
            except Exception:
                _propn_res = None
            if _propn_res and _propn_res[0].get('bm'):
                tok['bm'] = _propn_res[0]['bm']
            else:
                tok['bm'] = _lemma_propn
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

        # "se mettre à V" ≠ "mettre" : construction réfléchie-inchoative
        # (clitic expl:comp + xcomp marqué locatif 'à') — même structure que
        # celle déjà reconnue par step3_verbe.py::_xcomp_locative_mark. Le
        # lookup exact bare-lemme ci-dessous est ambigu pour certains verbes
        # dans CETTE construction précise (ex: 'mettre.' a 2 sens KG au même
        # gloss exact, kɛ́ et bìla, sans signal distinctif) et pioche
        # arbitrairement le mauvais ("il s'est mis à travailler" → kɛ́/faire
        # au lieu du sens inchoatif). Chercher "se X à" (phrase complète, pas
        # juste un contexte) laisse le KG trancher via son propre sens dédié
        # sans ambiguïté (ex: 'se mettre à.' → bìn, 100 pts exact, contre
        # 70 pts pour un match partiel "se X ..." générique).
        # Portée volontairement étroite : SEULEMENT quand la xcomp locative
        # est présente — les réfléchis sans cette structure (se blesser, se
        # regarder, se laver, se lever, se préparer…) ne passent jamais ce
        # test et gardent leur résolution actuelle inchangée.
        _refl_clitic = next((
            t for t in (all_tokens or [])
            if t.get('dep') == 'expl:comp' and t.get('role') == 'reflexive'
            and t.get('head_index') == tok.get('orig_index')), None)
        _refl_xcomp_locative_mark = next((
            m.get('surface', '') for x in (all_tokens or [])
            if x.get('dep') == 'xcomp' and x.get('head_index') == tok.get('orig_index')
            for m in (all_tokens or [])
            if m.get('dep') == 'mark' and m.get('role') == 'locative'
            and m.get('head_index') == x.get('orig_index')
        ), None) if (pos == 'VERB' and _refl_clitic) else None
        if _refl_xcomp_locative_mark:
            _refl_query = 'se ' + lemma + ' ' + str(_refl_xcomp_locative_mark).lower()
            _refl_candidates = self.retriever.retrieve(
                _refl_query, spacy_pos='VERB', top_k=1, lang=lang)
            _top_refl = _refl_candidates[0] if _refl_candidates else None
            if (_top_refl and _top_refl.get('match') == 'exact'
                    and _top_refl.get('score', 0) >= 90):
                tok['bm'] = _top_refl['bm']
                tok['sens_fr'] = _top_refl.get('fr', '')
                print(f"      [REFLEXIVE-SENSE] '{_refl_query}' → match KG dédié : "
                      f"{tok['bm']} (\"{_top_refl.get('fr')}\", {_top_refl.get('score')} pts)")
                self._classify_verb_transitivity(tok)
                return tok, _refl_candidates

        kg_label = self._get_kg_label(pos)

        # PRON substantivé avec article ("le vieux", "la petite") : spaCy
        # tague ces ADJ employés comme noms en PRON faute d'antécédent
        # explicite. Un vrai pronom clitique (il/elle/eux/celui...) n'a
        # jamais de déterminant enfant. Sans ce garde-fou, le lookup KG
        # {Pronoun} échoue (ces mots n'existent que sous Sense/adj ou
        # Sense/nom), et le token retombe dans le chemin "mot-outil" (LLM
        # EMPTY/un-mot, ligne ~2394) qui suppose à tort qu'un PRON n'a
        # jamais de contenu lexical propre → bm reste vide → le token est
        # ensuite silencieusement éliminé par le filtre PRON-sans-bm de
        # step5_obliques, perdant tout le complément (bug trouvé 2026-07-25 :
        # "il va causer avec le vieux" perdait tout le comitatif "avec le
        # vieux"). Reclassifier en NOUN avant le lookup KG lui donne la même
        # recherche Sense complète qu'un nom ordinaire.
        if pos == 'PRON' and all_tokens:
            _has_det = any(t.get('dep') == 'det' and t.get('head_index') == tok.get('orig_index')
                           for t in all_tokens)
            if _has_det:
                tok['pos'] = 'NOUN'
                pos = 'NOUN'
                kg_label = self._get_kg_label('NOUN')

        # ADJ prédicatif (copule) : le bambara exprime souvent "être ADJ" par
        # un VERBE qualitatif plutôt qu'un adjectif (ex: 'ɲì' [vq, très
        # fréquent] pour "beau/bon", vs 'ɲùman' [adjectif, rare]). Autoriser
        # un Verb à concurrencer un Adjective sur la fréquence UNIQUEMENT en
        # position prédicative — pas en position attributive ("belle maison"),
        # où un verbe ne peut pas se substituer syntaxiquement à l'adjectif.
        # Bug trouvé 2026-07-17 : 'ɲì' (freq=857) perdait systématiquement
        # contre 'ɲùman' (freq=0) car le palier POS de cette requête bloquait
        # tout candidat Verb avant même que la fréquence ne soit consultée.
        _adj_predicative = (
            pos == 'ADJ' and tok.get('dep') == 'ROOT'
            and any(t.get('dep') == 'cop' for t in (all_tokens or [])))

        if kg_label:
            try:
                res = self.db.query(f"""
                    MATCH (n:{kg_label})
                    WHERE toLower(n.surface) = toLower($surface)
                       OR toLower(n.lemma)   = toLower($lemma)
                       OR toLower(n.fr)      = toLower($lemma)
                       OR toLower(n.fr)      = toLower($lemma) + '.'
                       // gloss multi-mots séparés par virgule (ex: 'meilleur,
                       // préférable.') : reconnaître le lemme comme PREMIER
                       // sens listé, pas seulement en gloss unique exacte —
                       // sinon un sens de bon POS (fìsaman/Adjective,
                       // fr='meilleur, préférable.') est invisible à cette
                       // requête et un sens de mauvais POS mais gloss exacte
                       // (fìsamannci/Noun, fr='meilleur') gagne par défaut.
                       // Décision 2026-07-13.
                       OR toLower(n.fr) STARTS WITH toLower($lemma) + ','
                    RETURN n.bm AS bm, n.pos AS sense_pos, n.semantic_class AS sense_sc, n.fr AS fr,
                        (toLower(n.surface) = toLower($surface)
                         OR toLower(n.lemma) = toLower($lemma)
                         OR toLower(n.fr) = toLower($lemma)
                         OR toLower(n.fr) = toLower($lemma) + '.'
                         // Glose multi-sens dont le PREMIER segment est le lemme
                         // exact ("jour, date." pour lemma='jour') : même
                         // confiance qu'un match exact, pas une simple
                         // ressemblance textuelle à valider par LLM — sinon
                         // routé à tort vers _validate_gloss_match (non-déterministe,
                         // a rejeté 'jour, date.' pour 'jour', perdant tout le
                         // candidat correct 'dón'. Bug trouvé 2026-07-17).
                         // MAIS seulement si le POS concorde : un homographe
                         // français peut recouvrir DEUX MOTS SANS RAPPORT de
                         // catégories différentes ("ferme" nom=exploitation
                         // agricole vs "ferme, solide, bien dur..." adverbe=
                         // solide) — un même texte de surface, pas le même
                         // sens. Sans ce garde-fou, ce candidat sautait la
                         // validation LLM et gagnait à tort (bug trouvé
                         // 2026-07-17 : "il travaille dans une ferme" →
                         // 'gójogojo' [ferme=solide, Adverbe] au lieu d'un
                         // mot pour l'exploitation agricole).
                         OR (toLower(n.fr) STARTS WITH toLower($lemma) + ','
                             AND (toLower(coalesce(n.pos,'')) IN ['verb','verbe'] AND $pos='VERB'
                                  OR toLower(coalesce(n.pos,'')) IN ['noun','nom','n'] AND $pos='NOUN'
                                  OR toLower(coalesce(n.pos,'')) IN ['adjective','adj'] AND $pos='ADJ'
                                  OR toLower(coalesce(n.pos,'')) IN ['adverb','adv'] AND $pos='ADV'
                                  OR toLower(coalesce(n.pos,'')) IN ['verb','verbe'] AND $pos='ADJ' AND $adj_predicative))
                        ) AS is_exact_match
                    ORDER BY
                        CASE WHEN toLower(n.surface) = toLower($surface)
                             THEN 0 ELSE 1 END,
                        // POS-match AVANT l'exactitude du gloss FR : un gloss FR
                        // identique mais de POS incompatible (ex: 'fìsamannci'
                        // Noun, fr='meilleur' exact) ne doit pas l'emporter sur
                        // un gloss FR proche mais de bon POS (ex: 'fìsaman'
                        // Adjective, fr='meilleur, préférable.') — sinon le tok
                        // ADJ se fait "corriger" à tort en NOUN plus bas
                        // (décision 2026-07-13, bug : "meilleur" ADJ→NOUN).
                        CASE WHEN toLower(coalesce(n.pos,'')) IN ['verb','verbe'] AND $pos='VERB'
                             THEN 0
                             WHEN toLower(coalesce(n.pos,'')) IN ['noun','nom','n'] AND $pos='NOUN'
                             THEN 0
                             WHEN toLower(coalesce(n.pos,'')) IN ['adjective','adj'] AND $pos='ADJ'
                             THEN 0
                             WHEN toLower(coalesce(n.pos,'')) IN ['verb','verbe'] AND $pos='ADJ' AND $adj_predicative
                             THEN 0
                             WHEN toLower(coalesce(n.pos,'')) IN ['adverb','adv'] AND $pos='ADV'
                             THEN 0
                             ELSE 1 END,
                        // Exact bare/point/premier-segment-virgule = même palier
                        // (tous structurellement certains — voir kg/retriever.py
                        // _gloss_match_score pour la même logique côté retriever
                        // générique). Départage réel ensuite par fréquence corpus
                        // puis primauté du sens (bug trouvé 2026-07-17 : ce
                        // chemin KG-LABEL séparé, emprunté par tous les VERB,
                        // n'appliquait JAMAIS ces signaux — 'jí' [eau, freq=5]
                        // gagnait contre 'mìn' [boire, freq=8] pour 'boire',
                        // faute de tie-break ici).
                        CASE WHEN toLower(n.fr) = toLower($lemma) THEN 0
                             WHEN toLower(n.fr) = toLower($lemma) + '.' THEN 0
                             WHEN toLower(n.fr) STARTS WITH toLower($lemma) + ',' THEN 0
                             ELSE 2 END,
                        // Filtre attesté/non-attesté d'abord (freq=0 vs freq>0,
                        // ex: 'fòlofolo' freq=0 ne doit jamais battre 'bàn'
                        // freq=240 malgré un sense_index plus bas), PUIS
                        // sense_index (sens canonique du headword) départage
                        // entre candidats déjà également attestés/non-attestés,
                        // corpus_freq seulement pour départager un sense_index
                        // égal — même logique que kg/retriever.py::_primacy_key
                        // (décision utilisateur 2026-07-20 : l'ancienne tranche
                        // log10(freq) plaçait 'tìle' [jour, sens 3, freq=3726]
                        // avant 'dón' [jour, sens 1, freq=3110] à cause d'un
                        // simple artefact de seuil d'arrondi entre deux
                        // fréquences pourtant du même ordre de grandeur,
                        // empêchant sense_index de jamais trancher).
                        CASE WHEN coalesce(n.corpus_freq, 0) > 0 THEN 0 ELSE 1 END,
                        coalesce(n.sense_index, 1) ASC,
                        coalesce(n.corpus_freq, 0) DESC,
                        CASE WHEN n.fr CONTAINS ',' THEN 1 ELSE 0 END ASC
                    LIMIT 1
                """, {'surface': surface, 'lemma': lemma, 'pos': pos,
                      'adj_predicative': _adj_predicative})
                # Rang 1 non-exact (matché seulement via le préfixe de glose
                # multi-sens, ex. lemma='tout' ⊆ fr='tout, tout entier') : la
                # glose peut porter une nuance/connotation absente du texte
                # brut (ex. bákuru = "tout perdu/ruiné", pas un intensificateur
                # neutre) → valider sémantiquement avant d'adopter, au lieu de
                # faire confiance à la seule correspondance textuelle. Match
                # exact (surface/lemma/fr identique) : confiance directe, comme
                # le seuil ≥100 pts dans _validate_candidate_semantics.
                if res and res[0].get('bm') and not res[0].get('is_exact_match'):
                    if not self._validate_gloss_match(lemma, res[0].get('fr', '')):
                        print(f"       [KG-LABEL] Rejeté : glose '{res[0].get('fr','')}' "
                              f"ne correspond pas à '{lemma}' → repli sur le retriever complet")
                        res = None
                if res and res[0].get('bm'):
                    tok['bm'] = res[0]['bm']
                    _sense_pos = str(res[0].get('sense_pos') or '').lower()
                    _sense_sc  = res[0].get('sense_sc') or ''
                    _sense_fr  = str(res[0].get('fr') or res[0].get('bm', '')).strip()
                    tok['sens_fr'] = _sense_fr
                    # Lire semantic_class depuis KG si disponible (évite appel LLM nondéterministe)
                    if _sense_sc and not tok.get('semantic_class'):
                        tok['semantic_class'] = _sense_sc
                    # Si spaCy a mal tagué un nom comme ADJ, corriger via le KG —
                    # SAUF si le token est en position amod (épithète d'un autre
                    # nom, ex: "étudiants JAPONAIS") : cette position est
                    # structurellement adjectivale quel que soit le POS lexical
                    # du sens KG retenu ("Japonais." peut être glosé comme nom
                    # de nationalité), et la reclassifier en NOUN le fait sortir
                    # du filtre build_np._pos(t)=='ADJ', perdant l'épithète
                    # silencieusement (bug trouvé 2026-07-20 : "étudiants
                    # japonais" → "kàlandenbaw" sans 'zapɔnɛ').
                    if (pos == 'ADJ' and _sense_pos in ('noun', 'n', 'nom')
                            and tok.get('dep') != 'amod'):
                        tok['pos'] = 'NOUN'
                        pos = 'NOUN'
                    if pos == 'NOUN' and not tok['bm'].startswith('['):
                        tok['is_relational'] = self._detect_relational_noun(tok, tok['bm'])
                    elif pos == 'ADJ':
                        self._classify_adj_state(tok)
                        if tok.get('dep') == 'amod' and 'is_classifying_adj' not in tok:
                            if self._detect_classifying_adj(tok['lemma']):
                                tok['is_classifying_adj'] = True
                    elif pos == 'VERB':
                        # Même règle de classification (intransitive_type +
                        # action_noun) qu'ailleurs dans le pipeline — ce chemin
                        # de résolution KG-label (match exact lemme/surface)
                        # retourne juste après (ligne ~1557) sans jamais passer
                        # par les deux autres points d'appel de cette méthode,
                        # ce qui laissait "il ne faut pas manger"/"il mange"
                        # sans action_noun (dumuni) ni intransitive_type='support'.
                        self._classify_verb_transitivity(tok)
                    # Pour les NOUN avec correspondance KG courte (fr = lemma exact en 1 mot),
                    # continuer vers le retriever sémantique : le modèle embedding peut trouver
                    # une traduction plus précise (ex: fille→dénmuso plutôt que mùsoma).
                    # Si fr contient plusieurs mots ou ponctuation, la correspondance est
                    # déjà spécifique → retour anticipé justifié.
                    _kg_fr_words = [w for w in _sense_fr.rstrip('.').split() if w]
                    # ADJ avec nmod enfant (ex: "bon" + "à rien") : le sens KG
                    # trouvé pour l'ADJ seul (ɲùman) peut être correct pour "bon"
                    # isolé mais faux si l'ADJ+nmod forme un idiome dédié ("bon à
                    # rien" → kólon). Laisser tomber vers le retriever plus bas,
                    # qui tente d'abord la phrase composée ADJ+nmod (ligne ~1821)
                    # avant de retomber sur ce sens mot-à-mot déjà trouvé.
                    _adj_has_nmod = (pos == 'ADJ' and all_tokens and any(
                        t.get('dep') == 'nmod' and t.get('head_index') == tok.get('orig_index')
                        for t in all_tokens))
                    if (pos == 'NOUN' and len(_kg_fr_words) <= 1) or _adj_has_nmod:
                        pass  # continuer vers le retriever sémantique / phrase composée
                    else:
                        return tok, []
            except Exception as e:
                print(f"       KG label query failed ({kg_label}): {e}")

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
                print(f"       LLM function word failed: {e}")
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
                print(f"      Verbe normalisé: '{lemma}' → '{kg_search_lemma}'")

        # Enrich token with grammatical context (LLM analysis)
        if all_tokens:
            self._enrich_token_context(tok, all_tokens)
            if tok.get('context_type'):
                print(f"     [CONTEXT] '{lemma}' → context_type={tok['context_type']}")
            else:
                print(f"     [CONTEXT] '{lemma}' → NONE (LLM may have failed)")

        # ── COMPOUND-PHRASE RETRIEVAL: try "N de N2" / "ADJ à N2" as a fixed KG entry first ──
        # Ex: "raison de la venue" a un sens KG dédié (jɔ̀kun/nàkun) distinct
        # de "raison" seule (jó). "bon à rien" a de même un sens KG dédié
        # (kólon/fàdensago) distinct de "bon" seul (ɲùman) — sans ce chemin,
        # 'bon' se traduit isolément et 'rien' (nmod orphelin) fuit vers un
        # oblique indépendant en aval (ex: "... foyi la" parasite).
        # Une recherche mot-à-mot ne peut jamais préférer le composé (score
        # borné par un simple bonus additif) même quand le composé est la
        # traduction la plus juste. On cherche donc d'abord la phrase entière
        # (reconstituée depuis le texte source, avec ses articles) ; en cas de
        # match KG net, on l'utilise directement et on neutralise le modifieur
        # (déjà inclus dans le sens composé).
        # Sinon repli intégral sur la traduction mot-à-mot ci-dessous.
        if pos in ('NOUN', 'ADJ') and all_tokens:
            _tok_idx0 = tok.get('orig_index')
            # 'nmod' EN PRIORITÉ, 'amod' en repli (ex: "patates douces" —
            # aucun nmod ici, seulement l'amod "douces") : même mécanisme
            # de repêchage phrase-entière, juste élargi au type de
            # dépendant. Toujours construit depuis les formes de SURFACE
            # (jamais le lemme) — priorité au mot tel qu'écrit dans la
            # phrase, la normalisation ne doit pas intervenir avant cette
            # recherche (bug trouvé 2026-07-21 : "douces" lemmatisé en
            # "doux" par spaCy ne correspondait plus à la glose KG "patate
            # douce" accordée au féminin — chercher la PHRASE surface
            # directement contourne le problème sans jamais comparer de
            # lemme).
            _nmod_tok = next((t for t in all_tokens
                              if t.get('dep') == 'nmod'
                              and t.get('head_index') == _tok_idx0
                              and t.get('lemma')), None) or next(
                (t for t in all_tokens
                 if t.get('dep') == 'amod'
                 and t.get('head_index') == _tok_idx0
                 and t.get('lemma')), None)
            if _nmod_tok is not None and _nmod_tok.get('orig_index') is not None and _tok_idx0 is not None:
                _lo, _hi = sorted((_tok_idx0, _nmod_tok['orig_index']))
                _span = sorted((t for t in all_tokens
                                if t.get('orig_index') is not None
                                and _lo <= t['orig_index'] <= _hi
                                and t.get('pos') != 'PUNCT'),
                               key=lambda t: t['orig_index'])
                _compound_phrase = ' '.join(t.get('surface', '') for t in _span).strip()
                if _compound_phrase and _compound_phrase.lower() != lemma.lower():
                    _compound_candidates = self.retriever.retrieve(
                        _compound_phrase, spacy_pos=pos, top_k=3, lang=lang)
                    _top_compound = _compound_candidates[0] if _compound_candidates else None
                    # Repli SINGULIER si le pluriel de surface ne matche rien
                    # d'exact : le KG stocke ses gloses au singulier ("patate
                    # douce"), la phrase de surface peut être au pluriel
                    # ("patates douces") — un simple retrait du 's' final de
                    # chaque mot est une normalisation orthographique de
                    # NOMBRE, pas une lemmatisation de genre/POS (qui, elle,
                    # perdrait l'accord féminin, cf. "doux" vs "douce" plus
                    # haut) : "douces" → "douce" reste le mot féminin exact
                    # de la glose, seul le nombre change. Bug trouvé
                    # 2026-07-21 : "patates douces" → 0 match exact (pluriel)
                    # alors que "patate douce" matche à 100 pts.
                    if not (_top_compound and _top_compound.get('match') == 'exact'
                            and _top_compound.get('score', 0) >= 70):
                        _singular_phrase = ' '.join(
                            w[:-1] if w.endswith('s') and len(w) > 3 else w
                            for w in _compound_phrase.split())
                        if _singular_phrase != _compound_phrase:
                            _sg_candidates = self.retriever.retrieve(
                                _singular_phrase, spacy_pos=pos, top_k=3, lang=lang)
                            _sg_top = _sg_candidates[0] if _sg_candidates else None
                            if (_sg_top and _sg_top.get('match') == 'exact'
                                    and _sg_top.get('score', 0) >= 70):
                                _compound_candidates = _sg_candidates
                                _top_compound = _sg_top
                                _compound_phrase = _singular_phrase
                    # score==100 (glose == phrase composée mot pour mot) : confiance
                    # directe, comme is_exact_match plus bas. score∈[70,100) (glose
                    # liste/préfixe, ex. "raison de la venue, motif" ⊇ phrase) :
                    # le texte matche mais le sens réel du mot bambara peut différer
                    # (même faille que le raccourci KG-label single-mot) → valider
                    # sémantiquement avant d'adopter, sans quoi le résultat est
                    # accepté sur la seule preuve d'un chevauchement textuel.
                    if (_top_compound and _top_compound.get('match') == 'exact'
                            and _top_compound.get('score', 0) >= 70
                            and (_top_compound.get('score', 0) >= 100
                                 or self._validate_gloss_match(_compound_phrase, _top_compound.get('fr', '')))):
                        tok['bm'] = _top_compound['bm']
                        tok['sens_fr'] = _top_compound.get('fr', '')
                        _nmod_tok['bm'] = ''
                        _nmod_tok['_consumed_by_compound'] = True
                        # Le modifieur absorbé peut lui-même avoir ses propres
                        # dépendants non consommés par la phrase composée (ex:
                        # "venue DE QUELQU'UN" : 'quelqu'un' est nmod de 'venue',
                        # hors du span "raison de la venue"). Sans reroutage,
                        # ce complément resterait orphelin (tête='venue' muette)
                        # → on le rattache à la tête du composé pour qu'il soit
                        # repris comme un nmod normal de celle-ci en aval.
                        for _orph in all_tokens:
                            if (_orph.get('head_index') == _nmod_tok['orig_index']
                                    and not (_lo <= _orph.get('orig_index', -1) <= _hi)):
                                _orph['head_index'] = _tok_idx0
                        # Neutraliser le dep du modifieur absorbé : sinon il
                        # reste visible comme nmod de tok (même head_index) et
                        # court-circuite en premier les recherches `next(...)`
                        # en aval (ex: _build_subj_chain), masquant le vrai
                        # dépendant reroute juste au-dessus.
                        _nmod_tok['dep'] = '_absorbed_by_compound'
                        print(f"      [COMPOUND] '{_compound_phrase}' → match KG exact : "
                              f"{tok['bm']} (\"{_top_compound.get('fr')}\", {_top_compound.get('score')} pts)")
                        return tok, _compound_candidates
                    elif (_top_compound and _top_compound.get('match') == 'exact'
                          and _top_compound.get('score', 0) >= 70):
                        print(f"       [COMPOUND] Rejeté : glose \"{_top_compound.get('fr')}\" "
                              f"({_top_compound.get('score')} pts) ne correspond pas sémantiquement "
                              f"à '{_compound_phrase}' → repli sur la traduction mot-à-mot")

        # ── PARTICIPE PASSÉ EMPLOYÉ COMME ADJ PRÉDICATIF : normaliser AVANT la
        # recherche KG. spaCy lemmatise parfois un participe passé adjectivé
        # ("cuit", "c'est cuit") sous sa propre forme au lieu de l'infinitif
        # du verbe source ("cuire"). Le KG stocke ces sens sous l'infinitif
        # (ex: tóbi = "cuire."), donc chercher "cuit" tel quel ne trouve que
        # du bruit lexical (biscuit, riz cuit, mi-cuit...). Sans cette
        # normalisation précoce, la classification PARTICIPE n'arrivait
        # qu'après la recherche KG (trop tard pour corriger le lemme).
        # spaCy tag ADJ("cuit") ne décrit plus le mot une fois normalisé vers
        # son infinitif verbal ("cuire") : chercher avec spacy_pos='ADJ' fait
        # pénaliser le KG comme cross_pos (85 pts au lieu de 100/exact) et
        # expose le résultat à la validation sémantique LLM, peu fiable sur
        # le petit modèle local pour ce genre de cas. On recherche donc avec
        # spacy_pos='VERB' une fois le lemme normalisé.
        _search_pos = tok['pos']
        if (pos == 'ADJ' and tok.get('dep') == 'ROOT'
                and not (tok.get('is_statif') or tok.get('is_participe_passe') or tok.get('is_valeur'))
                and any(t.get('dep') == 'cop' for t in getattr(self, '_current_clause_tokens', []))):
            _r0 = self._detect_statif_adj(lemma)
            _rn0 = str(_r0).upper() if _r0 else ''
            if _rn0 == 'PARTICIPE':
                tok['is_participe_passe'] = True
                _infinitive0 = self._normalize_verb_to_infinitive(lemma)
                if _infinitive0 and _infinitive0 != lemma:
                    kg_search_lemma = _infinitive0
                    tok['raw_lemma'] = lemma
                    tok['lemma'] = _infinitive0
                    lemma = _infinitive0
                    _search_pos = 'VERB'
                    print(f"      Participe normalisé: '{tok['raw_lemma']}' → '{kg_search_lemma}'")
                # semantic_class du verbe source : certaines classes (ex.
                # 'preparation'/'technique', comme 'cuire') rendent le participe
                # passé prédicatif en V+ra/la/na résultatif plutôt qu'en V+len
                # dòn statif générique — step6_copule en a besoin pour aiguiller.
                tok['semantic_class'] = self._detect_semantic_class(tok['lemma'])
            elif _rn0 == 'STATIF':
                tok['is_statif'] = True
            elif _rn0 == 'VALEUR':
                tok['is_valeur'] = True

        # Bambara VQ (verbe de qualité) : un sens KG tagué pos='Verb' peut
        # être l'équivalent exact d'un adjectif français EN POSITION
        # PRÉDICATIVE (ROOT+cop) — le bambara n'a pas de copule séparée pour
        # les VQ, d'où la glose "être ADJ." sur une entrée Verb (cf. ligne
        # ~1871 pour le même calcul sur le chemin KG-label). Sans ce signal,
        # ces sens ne sont trouvés qu'en repêchage cross-POS et plafonnés
        # sous le seuil de confiance automatique (bug trouvé 2026-07-19 :
        # 'jɛ́'="être sûr." rejeté pour 'sûr').
        _adj_predicative_retr = (
            _search_pos == 'ADJ' and tok.get('dep') == 'ROOT'
            and any(t.get('dep') == 'cop' for t in (all_tokens or [])))

        candidates = self.retriever.retrieve(
            kg_search_lemma,
            spacy_pos=_search_pos,
            top_k=TOP_K,
            lang=lang,
            context_tokens=context_lemmas,
            is_verbal_noun=tok.get('is_verbal_noun', False),
            adj_predicative=_adj_predicative_retr,
        )

        # ── RECHERCHE AUSSI SUR LA SURFACE ORIGINALE (pas seulement le lemme) ──
        # Un adjectif français fléchi en genre/nombre ("finale") peut être la
        # forme sous laquelle le KG stocke sa glose (fr="finale", pas "final"),
        # alors que le lemme masculin canonique ("final") ne matche plus rien
        # d'exact ("finale" ne finit pas sur une frontière de mot après
        # "final" → score 0 partout). Bug trouvé 2026-07-17 : "la décision
        # finale" tombait en placeholder alors que 'fínali' (fr="finale")
        # existe bel et bien dans le KG.
        #
        # Exclusion : un VERBE en dep='acl' (participe modifiant un nom, ex.
        # "chercheur associé" — "associé" rattaché à "chercheur") a DÉJÀ
        # tranché structurellement pour le sens verbal via l'infinitif
        # ("associer") — rechercher aussi la forme brute du participe
        # ("associé") réintroduit un homographe NOM sans rapport (ɲɔ̀gɔn =
        # "un associé/collègue", personne) qui n'est pas une variante
        # d'inflexion du même mot, contrairement à "finale"/"final". Bug
        # trouvé 2026-07-19 : "chercheur associé à l'Institut..." perdait le
        # sens verbal 'jɛ̀' (s'associer à) au profit du nom 'ɲɔ̀gɔn'.
        _is_participle_acl = tok.get('pos') == 'VERB' and tok.get('dep') == 'acl'
        if surface and surface.lower() != kg_search_lemma.lower() and not _is_participle_acl:
            _surface_candidates = self.retriever.retrieve(
                surface,
                spacy_pos=_search_pos,
                top_k=TOP_K,
                lang=lang,
                context_tokens=context_lemmas,
                is_verbal_noun=tok.get('is_verbal_noun', False),
            )
            _seen_bm = {c.get('bm') for c in candidates}
            for _sc in _surface_candidates:
                if _sc.get('bm') not in _seen_bm:
                    candidates.append(_sc)
                    _seen_bm.add(_sc.get('bm'))
            candidates.sort(key=lambda x: (
                x.get('final_score', x.get('score', 0)),
                _primacy_key(x.get('sense_index', 1), x.get('corpus_freq', 0), x.get('fr', ''))
            ), reverse=True)

        # ── CROSS-POS EMBEDDING RETRY quand la recherche stricte POS n'a
        # rien trouvé de confiant ──
        # Certains mots français partagent l'orthographe presque à
        # l'identique avec un sens KG d'une AUTRE catégorie grammaticale
        # (variante de genre régulière : "coopératif"(ADJ)/"coopérative."
        # (Noun), "actif"/"active"...) mais le filtre spacy_pos (eff_pos)
        # s'applique aussi à la recherche embedding, pas seulement au match
        # exact — le sens reste donc invisible même en embedding, quel que
        # soit son score de similarité réel (même défaut structurel que pour
        # les verbes sans entrée directe, cf. _try_deverbal_noun_fallback /
        # bug 'recruter', 2026-07-20). Déclenché seulement quand AUCUN match
        # exact/textuel n'existe (candidats uniquement embed ou absents) —
        # PAS dès que le score est bas, car des candidats exacts de bonne
        # qualité (ex: gloses composées "participant de fête.") peuvent
        # légitimement scorer sous 90 pts sans être un trou lexical ; les y
        # ajouter du bruit cross-POS (28 candidats hors-sujet) a fait
        # dérailler le rerank LLM vers un faux-ami plausible (bug trouvé
        # 2026-07-20 : "participants" → 'jɔ̀yɔrɔ'="participation." au lieu
        # d'un des 7 vrais candidats "participant de ..."). La validation
        # sémantique plus bas (identité de surface ou jugement LLM) reste
        # seule responsable d'accepter ou rejeter le résultat — cette
        # relance ne fait qu'élargir le pool quand il n'y a structurellement
        # rien d'autre à juger.
        if not any(c.get('match') == 'exact' for c in candidates):
            # top_k=10 (pas le TOP_K=5 global) : cette relance interroge SANS
            # filtre POS, un pool structurellement plus bruité (adjectifs,
            # verbes... mélangés aux noms) où le bon candidat peut se
            # retrouver noyé sous des voisins vectoriels fortuits, en
            # particulier une fois la requête augmentée par le contexte de
            # la phrase (context_tokens) — un candidat par ailleurs correct
            # ("rebelle, terroriste." pour "djihadiste") disparaissait sous
            # top_k=5 alors qu'il ressortait avec un pool plus large, laissant
            # la place à des faux-amis sans rapport (bug trouvé 2026-07-21).
            _cross_pos_cands = []
            try:
                _cross_pos_cands += self.retriever.retrieve(
                    kg_search_lemma, spacy_pos=None, top_k=10,
                    lang=lang, context_tokens=context_lemmas,
                )
            except Exception:
                pass
            if surface and surface.lower() != kg_search_lemma.lower():
                try:
                    _cross_pos_cands += self.retriever.retrieve(
                        surface, spacy_pos=None, top_k=10,
                        lang=lang, context_tokens=context_lemmas,
                    )
                except Exception:
                    pass
            _seen_bm = set()
            for _cc in _cross_pos_cands:
                if _cc.get('bm') and _cc.get('bm') not in _seen_bm:
                    _cc['_cross_pos'] = True
                    candidates.append(_cc)
                    _seen_bm.add(_cc.get('bm'))
            if candidates:
                candidates.sort(key=lambda x: (
                    x.get('final_score', x.get('score', 0)),
                    _primacy_key(x.get('sense_index', 1), x.get('corpus_freq', 0), x.get('fr', ''))
                ), reverse=True)
                print(f"      [CROSS-POS RETRY] '{lemma}' — {len(candidates)} "
                      f"candidat(s) trouvé(s) hors filtre POS={_search_pos}")

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
                candidates.sort(key=lambda x: (
                    x.get('final_score', 0),
                    _primacy_key(x.get('sense_index', 1), x.get('corpus_freq', 0), x.get('fr', ''))
                ), reverse=True)

        # Conservé AVANT validation pour le repli "plus proche mot existant"
        # (cf. plus bas, cas 'élite' : rien ne passe la validation LLM
        # stricte, mais un placeholder [élite] est moins utile qu'un mot
        # imparfait mais réel). Décision 2026-07-21 de l'utilisateur : PAS
        # de mot inventé par le LLM (risque d'hallucination sur une langue
        # peu dotée) — seulement le meilleur candidat KG déjà attesté, choisi
        # par similarité CamemBERT brute (FastText écarté : pas discriminant
        # pour une proximité sémantique générale, seulement pour la parenté
        # de racine, cf. décision précédente sur 'élite'/'soldat'=0.84).
        _pre_validation_candidates = list(candidates)

        # ── SEMANTIC VALIDATION: Filter out false positives ──
        # LLM checks if candidate gloss actually matches the token semantically
        # top_k=20 (pas 10) : la relance cross-POS fusionne DEUX requêtes
        # (lemme singulier + surface, ex: 'djihadiste'+'djihadistes'), chacune
        # pouvant remonter jusqu'à ~15-19 candidats embed après l'extension
        # _min_embed du retriever — un candidat pertinent de la 1ère requête
        # peut donc se retrouver classé après les ~9 premiers résultats de la
        # 2e requête (dont les scores embed, gonflés sur une forme plurielle,
        # ne reflètent pas une pertinence réelle) et ne jamais atteindre la
        # validation si la fenêtre reste à 10 (bug trouvé 2026-07-23 :
        # 'bànbaganci'="rebelle, terroriste." classé 11e pour "djihadistes",
        # juste hors fenêtre, laissant gagner des candidats "+tigi" fabriqués
        # à partir de gloses sans rapport comme "affluent").
        candidates = self._validate_candidate_semantics(lemma, candidates, top_k=20,
                                                        tok=tok, all_tokens=all_tokens)

        # ── AFFICHAGE DU TOP 5/6 DES CANDIDATS SENSE DU KG (DIAGNOSTIC VISUEL) ──
        print(f"\n      [TRANSLATION ENGINE] Jeton: '{surface}' | Lemme: '{lemma}' | POS: {pos}")
        if not candidates:
            print("         Aucun candidat disponible dans la liste du moteur.")
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

        # Ne pas reranker si le candidat en tête a déjà été explicitement
        # confirmé par la validation sémantique (question précise : "ce token
        # correspond-il à cette glose ?"). Le reranking pose une question plus
        # large ("quel est le meilleur synonyme dans ce pool ?") sur un pool
        # qui inclut des candidats jamais validés — sans cette garde, une
        # confirmation positive déjà obtenue peut être écrasée par un choix
        # moins fiable (bug trouvé 2026-07-19 : 'nàfama' validé pour
        # 'fructueux' puis remplacé par 'gèren' [vert, non-mûr] via rerank).
        # Même garde que ci-dessus, cas symétrique : la validation sémantique
        # vient de conclure explicitement qu'AUCUN candidat ne dénote le mot
        # cherché (tous invalidés à 0 pts, aucun restauré par le fallback
        # 50%) — un fait tout aussi décisif que la confirmation positive
        # ci-dessus. Sans cette garde, rerank_with_llm repose la question
        # ("quel est le plus proche synonyme ?") sur un pool déjà rejeté et
        # FORCE un choix (plancher MIN_SCORE, ligne ~548 de cette même
        # fonction) même quand rien ne correspond réellement — produisant
        # une traduction fausse au lieu d'un placeholder honnête (bug trouvé
        # 2026-07-20 : "situation" → 'lújura' [glosé "difficulté, situation
        # pénible"] choisi de force parmi des gloses composées "situation de
        # X" toutes invalidées, alors qu'aucune ne dénote "situation" seul).
        # Depuis la décision 2026-07-20 sur le score (plus jamais mis à 0),
        # les candidats invalidés sont directement retirés de la liste dans
        # _validate_candidate_semantics plutôt que laissés à 0 pts — le "tous
        # invalidés" se traduit donc maintenant par une liste VIDE, pas par
        # des scores nuls à détecter ici.
        if not (candidates and candidates[0].get('_llm_validated')) and candidates:
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
                
        # Détecter statif/participe AVANT le return (sauf si déjà classé
        # en amont, ex: participe prédicatif normalisé avant la recherche KG)
        if tok['pos'] == 'ADJ' and not (
                tok.get('is_statif') or tok.get('is_participe_passe') or tok.get('is_valeur')):
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
                print(f"      [{tok['lemma']}] — embeddings too distant")
                _deverbal = self._try_deverbal_noun_fallback(tok['lemma']) if tok.get('pos') == 'VERB' else None
                if _deverbal:
                    tok['bm'] = _deverbal['bm']
                    tok['action_noun'] = _deverbal['bm']
                    tok['intransitive_type'] = 'support'
                    tok['sens_fr'] = _deverbal.get('fr', '')
                    print(f"      [DEVERBAL FALLBACK] '{tok['lemma']}' → nom d'action "
                          f"'{_deverbal['bm']}' (\"{_deverbal.get('fr','')}\") + kɛ")
                    return tok, [_deverbal]
                tok['bm'] = f"[{tok['lemma']}]"
                return tok, candidates

        if not candidates:
            _deverbal = self._try_deverbal_noun_fallback(tok['lemma']) if tok.get('pos') == 'VERB' else None
            if _deverbal:
                tok['bm'] = _deverbal['bm']
                tok['action_noun'] = _deverbal['bm']
                tok['intransitive_type'] = 'support'
                tok['sens_fr'] = _deverbal.get('fr', '')
                print(f"      [DEVERBAL FALLBACK] '{tok['lemma']}' → nom d'action "
                      f"'{_deverbal['bm']}' (\"{_deverbal.get('fr','')}\") + kɛ")
                return tok, [_deverbal]
            # DERNIER RECOURS : aucun candidat n'a passé la validation LLM
            # (trou lexical réel, ex. "élite" — aucun mot bambara direct au
            # KG), mais un placeholder [élite] est moins utile qu'un mot
            # RÉEL, déjà attesté au KG, même imparfait. Choisi par
            # similarité CamemBERT brute (pas d'invention LLM — décision
            # 2026-07-21 : trop de risque d'hallucination sur une langue peu
            # dotée). Marqué distinctement (_closest_word_fallback) pour
            # rester visuellement différent d'un match validé.
            #
            # UNIQUEMENT match='embed' (similarité sémantique CamemBERT) —
            # jamais un match='exact' textuel : un candidat exact a déjà été
            # explicitement rejeté par la validation LLM ci-dessus (ex.
            # "lâche" vs 'tɛ̀rɛku'="échappement lâche, échappement trop
            # lâche." — 50 pts de score textuel car "lâche" apparaît comme
            # mot isolé dans la glose, mais sémantiquement c'est un composé
            # technique distinct, à raison invalidé). Reprendre ce même
            # candidat ici via son score textuel brut annule silencieusement
            # le jugement de la validation (bug trouvé 2026-07-21).
            #
            # GARDE LLM (bug trouvé 2026-07-21) : le score CamemBERT seul
            # n'est pas fiable pour départager sur des mots rares — pour
            # "pleutre" (lâche, couard), CamemBERT classe TOUS les mots de
            # la pluie ("pluie", "détremper", "averse"...) à 0.70-0.76 de
            # cosinus, largement AU-DESSUS des vrais synonymes ("lâche"
            # 0.39, "couard" 0.34) — confusion probable avec la racine
            # "pleu-" de "pleuvoir", pas juste du bruit. FastText et LaBSE
            # testés en comparaison ne règlent pas non plus le problème de
            # façon fiable (cf. session). On ajoute donc un garde LLM léger
            # ("même concept général, au moins approximativement ?") sur
            # les meilleurs candidats avant acceptation — pas une invention
            # de mot (le candidat reste toujours un bm réel du KG), juste un
            # jugement grossier, dans l'esprit du reste de la validation
            # sémantique de cette fonction.
            _embed_candidates = sorted(
                (c for c in _pre_validation_candidates
                 if c.get('bm') and c.get('match') == 'embed'
                 and c.get('score', 0) >= 15),
                key=lambda c: c.get('score', 0), reverse=True)[:5]
            _closest = None
            for _ec in _embed_candidates:
                _sanity_prompt = (
                    f'Glose bambara : "{_ec.get("fr", "")}".\n'
                    f'Mot français cherché : "{lemma}".\n'
                    f'Est-ce que cette glose pourrait représenter, même de '
                    f'façon approximative ou par un sens voisin, le mot '
                    f'"{lemma}" ? Réponds UNIQUEMENT par OUI ou NON.'
                )
                try:
                    _sanity_resp = self._call_llm(_sanity_prompt, max_tokens=5).strip().upper()
                except Exception:
                    _sanity_resp = ''
                if 'OUI' in _sanity_resp:
                    _closest = _ec
                    break
                print(f"       [MOT LE PLUS PROCHE] rejeté par garde LLM : "
                      f"'{_ec.get('fr','')}' ≠ '{lemma}'")
            if _closest:
                tok['bm'] = _closest['bm']
                tok['sens_fr'] = _closest.get('fr', '')
                tok['_closest_word_fallback'] = True
                print(f"      [MOT LE PLUS PROCHE] '{tok['lemma']}' — aucun candidat "
                      f"validé, repli sur '{_closest['bm']}' (\"{_closest.get('fr','')}\", "
                      f"{_closest.get('score')} pts CamemBERT, confirmé par garde LLM)")
                return tok, [_closest]
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
            self._classify_verb_transitivity(tok)

        if tok['pos'] == 'NOUN' and tok.get('bm') and not tok['bm'].startswith('['):
            tok['is_relational'] = self._detect_relational_noun(tok, tok['bm'])

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

        if (tok['pos'] == 'ADJ' and tok.get('bm')
                and not (tok.get('is_statif') or tok.get('is_participe_passe') or tok.get('is_valeur'))):
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

    def _split_clauses(self, sentence: str, tokens: list = None, _rel_flags: list = None) -> list:
        """
        Phase 1 — split unconditionally at every comma (each comma-delimited
        segment becomes its own translation unit).
        Phase 2 — within each comma segment, split at dep-based clause boundaries
                   (acl:relcl, advcl, ccomp, xcomp).

        `_rel_flags`, si fourni, reçoit un booléen par segment retourné :
        True si ce segment démarre sur un pronom relatif ('qui/que/dont')
        ayant perdu son antécédent acl:relcl au moment du split (cf. translate()).
        """
        if not tokens:
            import re
            parts = re.split(r',|(?<=[a-zA-ZÀ-ÿ])\.(?=\s+[A-ZÀ-Ÿ]|\s*$)', sentence)
            parts = [p.strip().strip('.') for p in parts]
            parts = [p for p in parts if p and len(p.split()) > 1]
            if _rel_flags is not None:
                _rel_flags.extend([False] * len(parts))
            return parts

        _sorted_toks = sorted(tokens, key=lambda x: x['orig_index'])

        # ── Phase 1 : comma split ────────────────────────────────────────────────
        # Note: le rôle du token virgule est 'content' (pas 'punct') — on filtre
        # sur pos='PUNCT' ou dep='punct' à la place.
        _split_start_tok = None
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
            _split_start_tok = _after[0]
            break

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
                sub1 = self._split_clauses(seg1_text, seg1_toks, _rel_flags)
                _seg2_start = len(_rel_flags) if _rel_flags is not None else -1
                sub2 = self._split_clauses(seg2_text, seg2_toks, _rel_flags)
                # Le segment qui commence pile sur le pronom relatif ("qui/que/
                # dont" ayant perdu son antécédent acl:relcl au split) est
                # marqué pour que _translate_clause l'empêche d'être reclassé
                # en interrogatif (cf. _is_misparsed_relative_qui, rules/core.py).
                if (_rel_flags is not None and sub2
                        and _split_start_tok.get('role') == 'relative'
                        and 0 <= _seg2_start < len(_rel_flags)):
                    _rel_flags[_seg2_start] = True
                result = sub1 + sub2
                if result:
                    return result

        # Pas de comma split : dep-split direct
        _segs = self._split_at_deps(sentence, tokens)
        if _rel_flags is not None:
            _rel_flags.extend([False] * len(_segs))
        return _segs

    def _split_at_deps(self, sentence: str, _tokens: list) -> list:
        """Le rule engine gère ccomp/advcl inline (step3 ccomp block, step5 advcl.py).
        Un split dep-based casse les frontières de clause (ex: ccomp sur le prédicat
        nominal strande l'article dans la clause précédente). Phrase traitée en entier."""
        if not sentence.strip():
            return []
        return [sentence.strip()]

    def _get_relative_marker_bm(self, surface: str, lang: str) -> str:
        """FunctionWord{surface, role:'relative'}.bm (ex: 'qui'/fr → 'mìn') —
        distinct du nœud Pronoun{surface:'qui'} dont le bm ('jɔn') sert
        l'interrogatif "qui ?" (who)."""
        cache = getattr(self, '_relative_marker_bm_cache', None)
        if cache is None:
            cache = {}
            self._relative_marker_bm_cache = cache
        key = (surface.lower(), lang)
        if key not in cache:
            # NB: le KG contient un doublon FunctionWord{surface:'qui',
            # role:'relative'} avec deux bm différents ('mìn' correct,
            # 'jɔn' erroné — dupliqué à tort du nœud Pronoun interrogatif).
            # pos='PRON' ne distingue que le nœud correct ; le filtrer
            # explicitement évite un pick non-déterministe entre les deux.
            rows = self.db.query(
                "MATCH (f:FunctionWord {role:'relative', pos:'PRON'}) "
                "WHERE toLower(f.surface)=$s AND f.lang=$l RETURN f.bm AS bm LIMIT 1",
                {'s': surface.lower(), 'l': lang})
            cache[key] = rows[0]['bm'] if rows else ''
        return cache[key]

    # ------------------------------------------------------------------
    # SINGLE CLAUSE TRANSLATION
    # ------------------------------------------------------------------

    def _translate_clause(self, clause: str, lang: str = 'fr',
                          is_relative_continuation: bool = False) -> str:
        self._current_sentence = clause

        # ── FIXED PHRASES (Rule 5) ──────────────────────────────────────
        _FIXED_PHRASES = {
            'ainsi donc': 'ola sa',
        }
        clause_normalized = clause.lower().strip().rstrip('?!.,')
        for fr_phrase, bm_phrase in _FIXED_PHRASES.items():
            if clause_normalized == fr_phrase:
                print(f"\n      FIXED PHRASE MATCH: '{clause}' → '{bm_phrase}'")
                return bm_phrase

        clause_vector = self.model.encode(clause)

        if hasattr(self.db, 'search_semantic_phrase'):
            global_match = self.db.search_semantic_phrase(clause_vector, threshold=0.85)
            if global_match:
                print(f"\n      GLOBAL SEMANTIC MATCH FOUND (KG):")
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

        # Fragment issu du split d'une relative après virgule ("Mali, qui
        # s'enfonce...") : la retokenisation isolée perd l'antécédent
        # acl:relcl, donc 'qui' redevient nsubj du ROOT et serait mal
        # reclassé en interrogatif (cf. _is_misparsed_relative_qui,
        # rules/core.py). On marque le pronom explicitement et on force son
        # bm au marqueur relatif KG (FunctionWord{surface:'qui', role:
        # 'relative'}.bm = 'mìn') plutôt que le bm interrogatif 'jɔn' porté
        # par le nœud Pronoun (utilisé pour "qui ?" = who).
        if is_relative_continuation:
            _rel_tok = next((t for t in tokens
                             if t.get('pos') == 'PRON' and t.get('dep') == 'nsubj'
                             and t.get('role') == 'relative'), None)
            if _rel_tok:
                _rel_tok['_relative_continuation'] = True
                _rel_bm = self._get_relative_marker_bm(_rel_tok.get('surface', 'qui'), lang)
                if _rel_bm:
                    _rel_tok['bm'] = _rel_bm

        # '"' sert à la fois d'ouverture ET de fermeture (contrairement à «/»).
        # Sans le garde `just_opened`, le token OUVRANT se refermait sur
        # lui-même immédiatement (même caractère testé aux deux lignes) →
        # in_quotes retombait à False avant même d'atteindre le contenu cité,
        # qui n'était donc jamais protégé (bug sur "bonjour" mais pas « bonjour »).
        in_quotes = False
        for tok in tokens:
            surface = tok.get('surface', '')
            just_opened = False
            if not in_quotes and ('«' in surface or '"' in surface):
                in_quotes = True
                just_opened = True
            if in_quotes:
                tok['bm'] = surface
                tok['pos'] = 'PROPN'
                tok['is_protected'] = True
            if in_quotes and not just_opened and ('»' in surface or '"' in surface):
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

            # Déjà absorbé par un composé KG traité en amont (ex: 'venue'
            # dans "raison de la venue" → jɔ̀kun) : ne pas retraduire seul.
            if tok.get('_consumed_by_compound'):
                tok['bm'] = ''
                continue

            # 2. HARMONISATION GÉNÉRIQUE DES PRONOMS SUJETS SINGULIERS
            if pos == 'PRON' and surf == 'j':
                tok['bm'] = self.rule_engine.grammar.get('pron_1sg', '') or 'n'
                tok['role'] = 'pronoun'
                continue

            _expl_demo_surfaces = self.rule_engine.grammar.get('expletive_demonstrative_surfaces', {'ce', 'cela', 'ça'})
            if (pos == 'PRON' and surf.rstrip("'").rstrip('’') in _expl_demo_surfaces
                    and tok.get('role') in ('expletive', 'pronoun')):
                tok['bm'] = self.rule_engine.grammar.get('expletive_demonstrative_pronoun', '') or 'o'
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
            tok, candidates = self._translate_token(tok, context_lemmas, tokens)

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
                    and any(_is_avoir(t)
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
            from pipeline.proposition_parser import translate_propositions, set_grammar
            set_grammar(self.rule_engine.grammar)
            bambara = translate_propositions(tokens)
        else:
            bambara = self.rule_engine.apply(tokens)

        return bambara


    # ------------------------------------------------------------------
    # MAIN TRANSLATE
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_straight_quotes(sentence: str) -> str:
        """Remplace des paires de guillemets droits ("/") par « » (alternance
        ouverture/fermeture). Ne touche pas aux apostrophes (' seul n'est
        jamais transformé — seul le guillemet double ASCII l'est)."""
        if '"' not in sentence and '“' not in sentence and '”' not in sentence:
            return sentence
        out = []
        opening = True
        for ch in sentence:
            if ch in ('"', '“', '”'):
                out.append('«' if opening else '»')
                opening = not opening
            else:
                out.append(ch)
        return ''.join(out)

    def translate(self, sentence: str) -> dict:
        # Normaliser les guillemets droits (") en guillemets français (« »)
        # AVANT le parsing : le modèle spaCy fr gère "..." et «...» de façon
        # incohérente (ex: "bonjour" → dep=xcomp alors que « bonjour » →
        # dep=obl:arg pour le même mot dans la même phrase), ce qui cassait
        # silencieusement toute phrase citée avec des guillemets droits.
        sentence = self._normalize_straight_quotes(sentence)
        self._current_sentence = sentence

        all_tokens = tokenize(sentence, db=self.db,
                              backend=LLM_BACKEND, model=LLM_MODEL)
        if not all_tokens:
            return {'bambara': '', 'concepts': []}

        lang_tag = all_tokens[0].get('lang', 'fr').upper()

        print(f"\n INPUT : {sentence}  [{lang_tag}]")
        print(f" TOKENS: "
              f"{[(t['orig_index'], t['lemma'], t['pos'], t['dep'], t['role'], t.get('head_index')) for t in all_tokens]}")
        print("=" * 75)

        # Use spaCy dependencies to auto-detect clause boundaries
        _rel_flags   = []
        clauses      = self._split_clauses(sentence, tokens=all_tokens, _rel_flags=_rel_flags)
        all_bambara  = []
        all_concepts = []
        # `all_tokens` above is the pre-split sentence-level tokenize() call —
        # it's NEVER mutated with bm/sens_fr (that happens on a separate
        # re-tokenization of each clause substring inside _translate_clause).
        # Collect the real, translated per-clause tokens here instead, so
        # callers relying on result['tokens'] see actual bm/sens_fr values.
        _translated_tokens = []

        for i, clause in enumerate(clauses):
            if len(clauses) > 1:
                print(f"\n{'─'*40}")
                print(f" CLAUSE {i+1}/{len(clauses)}: {clause}")
                print(f"{'─'*40}")

            self._current_sentence = clause
            _main_lang = all_tokens[0].get('lang', 'fr') if all_tokens else 'fr'
            _is_rel_cont = _rel_flags[i] if i < len(_rel_flags) else False
            bm = self._translate_clause(clause, lang=_main_lang,
                                        is_relative_continuation=_is_rel_cont)
            _translated_tokens.extend(getattr(self, '_current_clause_tokens', []) or [])

            print(f"    Clause {i+1} -> '{bm}'")

            if bm:
                all_bambara.append(bm)

        bambara_output = ', '.join(b for b in all_bambara if b)

        print(f"\n BAMBARA : {bambara_output}")
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
            'obl_all':     [{'head': o.get('HEAD', ''), 'marker': o.get('MARKER', '')}
                             for o in _main.get('OBL_ALL', [])],
            'applied_template': _tree.get('_applied_template', ''),
        }

        # Snapshot des tokens (tagging) pour l'évaluation du parsing.
        # `all_tokens` porte le parse UD complet de la phrase (dep/head_index
        # cohérents pour LAS/UAS — voir eval/parse_trans_corr.py) mais n'est
        # JAMAIS mis à jour avec bm/sens_fr (ceux-ci sont posés sur une
        # re-tokenisation séparée par clause, `_translated_tokens`). Fusionner
        # bm/sens_fr par position quand les deux séquences ont la même
        # longueur (cas standard : découpage en clauses = simple partition,
        # même nombre de tokens) ; sinon laisser bm/sens_fr vides plutôt que
        # de risquer un alignement faux sur une phrase multi-clauses.
        _bm_by_pos = {}
        if len(_translated_tokens) == len(all_tokens):
            _bm_by_pos = {i: t for i, t in enumerate(_translated_tokens)}

        _tokens_meta = [
            {
                'surface':    t.get('surface', ''),
                'lemma':      t.get('lemma', ''),
                'pos':        t.get('pos', ''),
                'dep':        t.get('dep', ''),
                'role':       t.get('role', ''),
                'head_index': t.get('head_index', -1),
                'orig_index': t.get('orig_index', -1),
                'bm':         _bm_by_pos.get(i, {}).get('bm', '')
                              or t.get('bm', ''),
                'sens_fr':    _bm_by_pos.get(i, {}).get('sens_fr', '')
                              or t.get('sens_fr', ''),
            }
            for i, t in enumerate(all_tokens)
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
            'concepts':      all_concepts,
            'tree':          _tree_meta,
            'tokens':        _tokens_meta,
            'sens_fr_gloss': _sens_fr_gloss,
        }
