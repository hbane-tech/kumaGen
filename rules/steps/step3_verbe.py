"""
rules/steps/step3_verbe.py
Étape 3 : verbe ROOT (m['V']), xcomp, négation, prohibitif, impératif,
infinitif, F1 tense transfer, déjà, privatif, ADJ/NOUN ROOT avec copule.
"""
from rules.core import j, _is_copula, _is_avoir, _resolve_tam, adj_man


def run(T, tree, m, processed_indices, G_kg, NX_G,
        root_tok, xcomp_verb_tok, _neg_surfaces):
    """Retourne aux_tense_tok."""

    # ── DÉICTIQUE : voilà/voici → présentatif [O] félé ─────────────────────
    if root_tok and root_tok.get('role') == 'deictique':
        tree['clause_type'] = 'presentative'
        processed_indices.add(root_tok['orig_index'])
        # step2 may have wrongly captured the obj noun as a fallback subject;
        # release all children of the deictique root so step4 can use them as O
        for _t in T:
            if _t.get('head_index') == root_tok['orig_index']:
                processed_indices.discard(_t['orig_index'])
        m['V'] = ''
        m['S'] = ''
        return None

    # ── ÉTAPE 2 : VERBE ROOT ─────────────────────────────────────────────────
    if root_tok and root_tok.get('pos') in ('VERB', 'AUX'):
        _root_is_copula = (
            _is_copula(root_tok)
            or (_is_avoir(root_tok)
                and root_tok.get('dep') in ('aux', 'aux:tense', 'aux:pass', 'cop'))
            or any(x.get('dep') == 'case'
                  and x.get('role') == 'locative'
                  and x.get('head_index') == root_tok['orig_index']
                  for x in T)
        )
        if not _root_is_copula:
            root_bm = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
            if root_tok.get('bm_suffix'):
                root_bm += root_tok['bm_suffix']
            m['V'] = root_bm
            processed_indices.add(root_tok['orig_index'])
            _v_advmods = [x for x in T
                          if x.get('dep') == 'advmod'
                          and x.get('head_index') == root_tok['orig_index']
                          and x.get('bm')
                          and x['orig_index'] not in processed_indices
                          and str(x.get('surface', '')).lower().rstrip("'").rstrip('\u2019')
                          not in _neg_surfaces
                          and x.get('role') not in ('temporal', 'temporal_already')
                          and x.get('pos') != 'ADV']
            for _va in _v_advmods:
                _va_sub = [x for x in T
                           if x.get('dep') == 'advmod'
                           and x.get('head_index') == _va['orig_index']
                           and x.get('bm')
                           and x['orig_index'] not in processed_indices
                           and str(x.get('surface', '')).lower().rstrip("'").rstrip('\u2019')
                           not in _neg_surfaces]
                _va_bm = _va.get('bm')
                for _vs in _va_sub:
                    _va_bm = j(_va_bm, _vs.get('bm'))
                    processed_indices.add(_vs['orig_index'])
                m['V'] = j(m['V'], _va_bm)
                processed_indices.add(_va['orig_index'])

    # ── VERBE CONJ coordonné avec ROOT (ex: manger ET dormir) ────────────────
    # Seulement si le verbe conj n'a pas son propre sujet explicite
    if root_tok and root_tok.get('pos') in ('VERB', 'AUX') and m.get('V'):
        _conj_root_verbs = [x for x in T
                            if x.get('dep') == 'conj'
                            and x.get('head_index') == root_tok['orig_index']
                            and x.get('pos') == 'VERB'
                            and x['orig_index'] not in processed_indices
                            and not any(d.get('dep') in ('nsubj', 'nsubj:pass')
                                        and d.get('head_index') == x['orig_index']
                                        for d in T)]
        _root_is_inf = 'Inf' in str(root_tok.get('morph', ''))
        for _crv in _conj_root_verbs:
            _crv_bm = _crv.get('bm') or f"[{_crv.get('lemma')}]"
            _cc_rv  = (next((x for x in T if x.get('dep') == 'cc'
                             and x.get('head_index') == root_tok['orig_index']), None)
                       or next((x for x in T if x.get('dep') == 'cc'
                                and x.get('head_index') == _crv['orig_index']), None))
            # Collect obliques that belong exclusively to this conj verb
            _crv_obl_parts = []
            for _crv_obl in T:
                if (_crv_obl.get('dep') in ('obl:mod', 'obl', 'obl:arg')
                        and _crv_obl.get('head_index') == _crv['orig_index']
                        and _crv_obl['orig_index'] not in processed_indices
                        and _crv_obl.get('bm')):
                    _crv_obl_bm = _crv_obl.get('bm')
                    _crv_obl_parts.append(_crv_obl_bm)
                    processed_indices.add(_crv_obl['orig_index'])
                    for _ctok in T:
                        if (_ctok.get('dep') == 'case'
                                and _ctok.get('head_index') == _crv_obl['orig_index']
                                and _ctok['orig_index'] not in processed_indices):
                            processed_indices.add(_ctok['orig_index'])
            if _root_is_inf:
                # Infinitifs coordonnés : ani ka V2 ARGS2
                m['V'] = j(m['V'], 'ani', 'ka', _crv_bm, *_crv_obl_parts)
            else:
                # Verb coordination: 'et' between verbs → 'wa' (not 'ni' which joins nouns)
                _crv_obj = next((x for x in T
                                 if x.get('dep') == 'obj'
                                 and x.get('head_index') == _crv['orig_index']
                                 and x['orig_index'] not in processed_indices), None)
                _crv_obj_bm = ((_crv_obj.get('bm') or f"[{_crv_obj.get('lemma')}]")
                               if _crv_obj else '')
                # Anaphoric subject pronoun for the second coordinated clause
                _subj_nsubj = next((x for x in T
                                    if x.get('dep') in ('nsubj', 'nsubj:pass')
                                    and x.get('head_index') == root_tok['orig_index']), None)
                if _subj_nsubj and _subj_nsubj.get('pos') == 'PRON' and _subj_nsubj.get('bm'):
                    _coord_pron = _subj_nsubj['bm']
                elif _subj_nsubj and _subj_nsubj.get('is_plural'):
                    _coord_pron = 'uw'
                else:
                    _coord_pron = 'a'
                if _crv_obj_bm:
                    # S TAM O1 V1 wa PRON TAM O2 V2
                    _conj_tam = (_resolve_tam(
                                     root_tok.get('tense', 'pres'),
                                     tree.get('neg', False) or root_tok.get('is_neg', False),
                                     G_kg)
                                 or G_kg.get('tam_default', 'bɛ'))
                    m['V'] = j(m['V'], 'wa', _coord_pron, _conj_tam, _crv_obj_bm, _crv_bm,
                               *_crv_obl_parts)
                    processed_indices.add(_crv_obj['orig_index'])
                else:
                    m['V'] = j(m['V'], 'wa', _crv_bm, *_crv_obl_parts)
            processed_indices.add(_crv['orig_index'])
            if _cc_rv:
                processed_indices.add(_cc_rv['orig_index'])

    # ── XCOMP VERB ────────────────────────────────────────────────────────────
    if xcomp_verb_tok:
        m['V_ACTION'] = xcomp_verb_tok.get('bm') or f"[{xcomp_verb_tok.get('lemma')}]"
        processed_indices.add(xcomp_verb_tok['orig_index'])
        _xcomp_obj = next((x for x in T
                           if x.get('dep') == 'obj'
                           and x.get('head_index') == xcomp_verb_tok['orig_index']
                           and x.get('pos') in ('NOUN', 'PROPN')), None)
        if _xcomp_obj and _xcomp_obj['orig_index'] not in processed_indices:
            _xcomp_obj_bm = _xcomp_obj.get('bm') or f"[{_xcomp_obj.get('lemma')}]"
            _xcomp_poss   = next((x for x in T
                                  if x.get('dep') == 'det'
                                  and x.get('role') in ('pronoun', 'possessive')
                                  and x.get('head_index') == _xcomp_obj['orig_index']), None)
            if _xcomp_poss:
                _poss_bm      = _xcomp_poss.get('bm', '')
                _xcomp_obj_bm = (j('n', _xcomp_obj_bm) if _poss_bm == 'n'
                                 else j(_poss_bm, _xcomp_obj_bm))
                processed_indices.add(_xcomp_poss['orig_index'])
            m['O'] = _xcomp_obj_bm
            processed_indices.add(_xcomp_obj['orig_index'])

    # ── NÉGATION ─────────────────────────────────────────────────────────────
    # Exclure les tokens n\u00e9gatifs dont la t\u00eate est un ADJ/ADV \u2192 comparatif (plus important)
    # Un vrai n\u00e9gatif (ne...plus, ne...pas) a pour t\u00eate un VERB.
    def _is_genuine_neg(tok, tokens):
        # Tete VERB -> vraie negation (ne...pas, ne...plus).
        # Tete ADJ/ADV : comparatif (plus importante QUE soi) -> pas une
        # negation ; negation predicative (n'est pas grande) -> vraie negation.
        # On tranche : comparatif si la tete porte une marque SCONJ 'que'
        # placee APRES l'adjectif.
        _h = next((t for t in tokens if t.get('orig_index') == tok.get('head_index')), None)
        if not (_h and _h.get('pos') in ('ADJ', 'ADV')):
            return True
        _has_comp_mark = any(
            x.get('dep') == 'mark'
            and x.get('pos') == 'SCONJ'
            and x.get('role') != 'temporal'
            and x.get('orig_index', -1) > _h.get('orig_index', -1)
            for x in tokens)
        return not _has_comp_mark

    _has_neg_adv = any(
        str(x.get('surface', '')).lower().rstrip("'").rstrip('\u2019') in _neg_surfaces
        and x.get('dep') in ('advmod', 'fixed', 'mark')
        and _is_genuine_neg(x, T)
        for x in T)
    if _has_neg_adv:
        tree['neg'] = True
    for _nt in T:
        surf = str(_nt.get('surface', '')).lower().rstrip("'").rstrip('\u2019')
        if (surf in _neg_surfaces and _nt.get('dep') in ('advmod', 'fixed', 'mark')
                and _is_genuine_neg(_nt, T)):
            processed_indices.add(_nt['orig_index'])

    # ── PROHIBITIF ────────────────────────────────────────────────────────────
    _is_prohibitive = (root_tok and root_tok.get('pos') == 'VERB'
                       and not m.get('S') and tree.get('neg')
                       and not any(x.get('dep') in ('nsubj', 'nsubj:pass') for x in T))
    if _is_prohibitive:
        tree['clause_type'] = 'prohibitive'
        tree['neg'] = False

    # ── EST-CE QUE : marqueur interrogatif → Yala ────────────────────────────
    # Signal : -ce dep='nsubj' + que dep='mark' + être ROOT
    _est_ce_que = any(
        str(x.get('surface', '')).lower().rstrip('-').lstrip('-') == 'ce'
        and x.get('dep') in ('nsubj', 'expl:subj')
        for x in T)
    _has_que_mark = any(
        str(x.get('surface', '')).lower() in ('que', "qu'")
        and x.get('dep') == 'mark'
        for x in T)
    if _est_ce_que and _has_que_mark:
        tree['est_ce_que'] = True
        # Ignorer -ce comme sujet
        for _cx in T:
            if str(_cx.get('surface', '')).lower().rstrip('-').lstrip('-') == 'ce':
                processed_indices.add(_cx['orig_index'])
        # Ignorer 'que'
        for _qx in T:
            if str(_qx.get('surface', '')).lower() in ('que', "qu'"):
                processed_indices.add(_qx['orig_index'])

    # ── EST-CE QUE + PRON ROOT interrogatif ──────────────────────────────────
    # Qui est-ce qu'elle aime ? → content_question avec acl:relcl comme verbe
    _pron_root_interrog = (root_tok
                           and root_tok.get('pos') == 'PRON'
                           and root_tok.get('dep') == 'ROOT'
                           and 'Int' in str(root_tok.get('morph', '')))
    if _pron_root_interrog:
        _acl_verb = next((x for x in T
                          if x.get('dep') == 'acl:relcl'
                          and x.get('pos') == 'VERB'
                          and x.get('bm')), None)
        if _acl_verb:
            tree['clause_type'] = 'content_question'
            # Le verbe acl:relcl devient le verbe principal
            m['V'] = _acl_verb.get('bm', '')
            processed_indices.add(_acl_verb['orig_index'])
            # Le sujet est le nsubj du acl:relcl
            _acl_subj = next((x for x in T
                              if x.get('dep') in ('nsubj', 'nsubj:pass')
                              and x.get('head_index') == _acl_verb['orig_index']
                              and x.get('bm')), None)
            if _acl_subj:
                m['S'] = _acl_subj.get('bm', '')
                processed_indices.add(_acl_subj['orig_index'])
            # L'interrogatif ROOT devient O
            m['O'] = root_tok.get('bm', '')
            processed_indices.add(root_tok['orig_index'])
            # Ignorer est-ce que
            for _cx in T:
                if str(_cx.get('surface', '')).lower() in ('est', '-ce', "qu'", 'que'):
                    processed_indices.add(_cx['orig_index'])

    # ── EXISTENTIEL : il y a / il n'y a pas ──────────────────────────────────
    # il y a personne → [personne] tɛ
    # Signal : y dep='expl:comp' + avoir ROOT + obj
    _has_y_expl = any(
        str(x.get('surface', '')).lower().rstrip("'").rstrip('\u2019') == 'y'
        and x.get('dep') in ('expl:comp', 'advmod', 'expl:subj', 'expl')
        for x in T) or any(
        x.get('dep') in ('expl:subj', 'expl:comp')
        for x in T)
    _avoir_root = (root_tok
                   and root_tok.get('lemma', '').lower() == 'avoir'
                   and root_tok.get('pos') == 'VERB')
    
    if _has_y_expl and _avoir_root:
        tree['clause_type'] = 'existential_absolute'
        tree['_existential_locked'] = True
        m['V'] = ''
        m['S'] = ''
        _exist_obj = next((x for x in T
                           if x.get('dep') == 'obj'
                           and x.get('pos') in ('NOUN', 'PROPN', 'PRON')), None)
        if _exist_obj:
            _es_bm = _exist_obj.get('bm', '')
            # Négation existentielle : personne → mɔgɔ si bm vide
            if not _es_bm and _exist_obj.get('role') == 'negation':
                _neg_bm = G_kg.get('funcs', {}).get(
                    (_exist_obj.get('surface', '').lower(), 'fr'), {}).get('bm', '')
                _es_bm = _neg_bm or f"[{_exist_obj.get('lemma', '')}]"
            if (_exist_obj.get('is_plural')
                    or str(_exist_obj.get('surface', '')).endswith('s')):
                if not _es_bm.endswith('w'):
                    _es_bm += 'w'
            m['O'] = _es_bm
            processed_indices.add(_exist_obj['orig_index'])
        processed_indices.add(root_tok['orig_index'])
        for _yt in T:
            if (str(_yt.get('surface', '')).lower().rstrip("'").rstrip('\u2019') == 'y'
                    or _yt.get('dep') in ('expl:subj', 'expl:comp')):
                processed_indices.add(_yt['orig_index'])
        return None  # ← retour anticipé, bloque tout step suivant

    # ── AVOIR ROOT (possession/état/âge/douleur) ─────────────────────────────
    _avoir_poss = (root_tok
                   and root_tok.get('lemma', '').lower() == 'avoir'
                   and root_tok.get('pos') in ('VERB', 'AUX')
                   and not _has_y_expl)
    if _avoir_poss:
        _obj_tok = next((x for x in T
                             if x.get('dep') in ('obj', 'nsubj', 'nsubj:pass', 'obl:arg')
                             and x.get('pos') in ('NOUN', 'PROPN')), None)
        if _obj_tok:
            # _det_interrog = any(
            #     x.get('dep') in ('det', 'amod')
            #     and (x.get('role') == 'interrogative'
            #          or 'Int' in str(x.get('morph', ''))
            #          or 'PronType=Int' in str(x.get('morph', '')))
            #     and x.get('head_index') == _obj_tok['orig_index']
            #     for x in T)
            _det_interrog = any(
                x.get('dep') in ('det', 'amod', 'obj', 'advmod')
                and (x.get('role') == 'interrogative'
                     or 'Int' in str(x.get('morph', ''))
                     or 'PronType=Int' in str(x.get('morph', '')))
                for x in T)
            
            _has_body_obl = any(
                x.get('dep') in ('obl', 'obl:arg')
                and x.get('head_index') == root_tok['orig_index']
                for x in T)
            _obj_is_pain = (_obj_tok.get('role') == 'pain'
                            or _obj_tok.get('semantic_class') == 'pain')
            if _det_interrog:
                # AGE seulement si l'objet est un nom d'âge/temps
                _obj_sc = _obj_tok.get('semantic_class', '')
                _obj_lemma = _obj_tok.get('lemma', '').lower()
                _age_lemmas = {'âge', 'an', 'ans', 'année', 'années'}
                if _obj_sc in ('time', 'duration', 'age') or _obj_lemma in _age_lemmas:
                    tree['have_type'] = 'AGE'
                else:
                    tree['have_type'] = 'ABSTRACT'
            elif _obj_is_pain or (_has_body_obl and not _det_interrog):
                tree['have_type'] = 'PAIN'
            else:
                tree['have_type'] = _obj_tok.get('possession_type', 'UNKNOWN')
                tree['have_obj_lemma'] = _obj_tok.get('lemma', '')
                # Assigner O = nom (enfant, voiture...) et stocker quantité interrogative
            # Ne réassigner m['O'] que si step1_avoir n'a pas déjà traité cet objet
            # (step1_avoir ajoute _obj_poss à processed_indices après avoir inclus les conj)
            if _obj_tok['orig_index'] not in processed_indices:
                _obj_bm = _obj_tok.get('bm', '')
                if _obj_tok.get('is_plural') and _obj_bm and not _obj_bm.endswith('w'):
                    _obj_bm += 'w'
                if _obj_bm:
                    m['O'] = _obj_bm
                    processed_indices.add(_obj_tok['orig_index'])
            # Stocker combien/quel comme quantité interrogative
            if _det_interrog:
                _interrog_qty = next((x for x in T
                                      if x.get('role') == 'interrogative'
                                      and x.get('bm')), None)
                if _interrog_qty:
                    tree['interrog_qty'] = _interrog_qty.get('bm', '')
                    processed_indices.add(_interrog_qty['orig_index'])

            if tree.get('have_type'):
                tree['clause_type'] = 'noun_phrase_have'

    # ── COMPOUND VERB → infinitif ─────────────────────────────────────────────
    # Cas : spaCy parse ADJ comme ROOT avec VERB compound (ex: publier une info crédible)
    # → traiter comme infinitif
    if (root_tok and root_tok.get('pos') == 'ADJ'
            and not root_tok.get('is_statif')
            and not any(x.get('dep') in ('cop', 'aux:pass') for x in T)):
        _compound_verb = next((x for x in T
                               if x.get('dep') in ('compound', 'ROOT', 'advcl', 'amod')
                               and x.get('lemma', '').lower() not in ('', 'none')
                               and x.get('orig_index') != root_tok.get('orig_index')), None)
        if _compound_verb:
            tree['clause_type'] = 'infinitive'
            tree['tam'] = ''
            # m['V'] = _compound_verb.get('bm')
            m['V'] = _compound_verb.get('bm') or f"[{_compound_verb.get('lemma', '')}]"
            processed_indices.add(_compound_verb['orig_index'])
            # L'ADJ ROOT devient amod sur l'objet
            _obj_tok = next((x for x in T
                             if x.get('dep') in ('obj', 'nsubj', 'nsubj:pass', 'obl:arg')
                             and x.get('pos') in ('NOUN', 'PROPN')), None)
            if _obj_tok:
                _amod = next((x for x in T
                              if x.get('dep') == 'amod'
                              and x.get('head_index') == _obj_tok['orig_index']
                              and x.get('bm')), None)
                # L'ADJ ROOT est aussi un modificateur de l'objet
                _obj_bm = _obj_tok.get('bm', '')
                # ADJ ROOT → modificateur de l'objet (après le nom en bambara)
                if root_tok.get('pos') == 'ADJ' and root_tok.get('bm'):
                    _obj_bm = j(_obj_bm, root_tok.get('bm', ''))
                    processed_indices.add(root_tok['orig_index'])
                if _amod and _amod['orig_index'] not in processed_indices:
                    _obj_bm = j(_obj_bm, _amod.get('bm', ''))
                    processed_indices.add(_amod['orig_index'])
                m['O'] = _obj_bm
                processed_indices.add(_obj_tok['orig_index'])

    # ── INFINITIF : ka + O + V ───────────────────────────────────────────────
    # ex: manger du riz → ka iri dún
    if (root_tok and 'VerbForm=Inf' in str(root_tok.get('morph', ''))
            and not m.get('S')
            and not any(x.get('dep') in ('nsubj', 'nsubj:pass') for x in T)):
        tree['clause_type'] = 'infinitive'
        tree['tam'] = ''

    # ── IMPÉRATIF AFFIRMATIF ──────────────────────────────────────────────────
    _has_excl = any(str(x.get('surface', '')).strip() == '!' for x in T)
    _no_subj  = not any(x.get('dep') in ('nsubj', 'nsubj:pass') for x in T)
    # Note : vérifier la négation ORIGINALE (avant que _is_prohibitive la remette à False)
    _has_neg_original = any(
        str(x.get('surface', '')).lower().rstrip("'").rstrip('\u2019')
        in _neg_surfaces
        and x.get('dep') in ('advmod', 'fixed', 'mark')
        for x in T)
    _is_imperative_affirm = (root_tok and root_tok.get('pos') == 'VERB'
                             and not m.get('S') and not _has_neg_original
                             and _no_subj
                             and 'VerbForm=Inf' not in str(root_tok.get('morph', '')))
    if _is_imperative_affirm:
        tree['clause_type'] = 'imperative'
        tree['tam'] = ''

    # ── RÉFLEXIF / RÉCIPROQUE ────────────────────────────────────────────────
    # Détection via Reflex=Yes (spaCy morph) — pas de surfaces codées en dur.
    # Trois cas :
    #   1. dep='iobj'      → passif réflexif (s'appeler) : is_refl_passive=True
    #   2. dep='obj'/'expl:comp' + singulier → agentif (se blesser) → yɛrɛ
    #   3. dep='expl:comp' + pluriel → réciproque (se battre) → ɲɔgɔn
    _refl_tok = next((x for x in T
                      if x.get('dep') in ('expl:comp', 'obj', 'iobj')
                      and x.get('pos') == 'PRON'
                      # Exclure l'apostrophe nue d'élision (j'/c'/qu' → token "'")
                      # mal taguée expl:comp : ce n'est PAS un clitique réfléchi.
                      # Un vrai clitique (s'/se/me/te/nous) a une surface non vide
                      # une fois les apostrophes retirées.
                      and str(x.get('surface', '')).strip().strip("'''") != ''
                      and ('Reflex=Yes' in str(x.get('morph', ''))
                           # dep='expl:comp' sur un PRON = clitique réfléchi (UD fr),
                           # même quand spaCy omet Reflex=Yes (tagging incohérent)
                           or x.get('dep') == 'expl:comp'
                           or any(s.get('dep') in ('nsubj', 'nsubj:pass')
                                  and s.get('pos') == 'PRON'
                                  and str(s.get('surface', '')).lower()
                                  == str(x.get('surface', '')).lower()
                                  for s in T))), None)
    print(f"DEBUG [REFLEXIVE CHECK] _refl_tok after main check: {_refl_tok.get('surface') if _refl_tok else None}")
    # Fallback : iobj/obj PRON lié au verbe racine (t' tokenisé sans Reflex=Yes).
    # MAIS uniquement si COREFERENT avec le sujet (même personne) : 'tu t''=2/2
    # est réflexif, 'je te'=1/2 est un datif (COI), pas réflexif → laissé au
    # bloc datif. Sans ce garde, 'te' (datif) était happé à tort comme réflexif.
    # Aussi: obj au lieu de iobj pour 'se laver' (spaCy tague 'se' comme obj).
    if not _refl_tok and root_tok and root_tok.get('orig_index'):
        _subj_x = next((x for x in T if x.get('dep') in ('nsubj', 'nsubj:pass')), None)
        _subj_pers = next((p.split('=')[1] for p in str(_subj_x.get('morph', '')).split('|')
                           if p.startswith('Person=')), '') if _subj_x else ''
        print(f"DEBUG [REFLEXIVE FALLBACK] root_tok={root_tok.get('surface')}, _subj_x={_subj_x.get('surface') if _subj_x else None}, _subj_pers={_subj_pers}")
        _refl_tok = next((x for x in T
                          if x.get('dep') in ('iobj', 'obj')
                          and x.get('pos') == 'PRON'
                          and x.get('head_index') == root_tok.get('orig_index')
                          and _subj_pers
                          and f'Person={_subj_pers}' in str(x.get('morph', ''))), None)
        print(f"DEBUG [REFLEXIVE FALLBACK] _refl_tok after fallback: {_refl_tok.get('surface') if _refl_tok else None}")
    _refl_abs_classes = {'posture', 'motion', 'biological', 'spontaneous'}
    _nsubj_tok = next((x for x in T if x.get('dep') in ('nsubj', 'nsubj:pass')), None)
    _subj_is_plural = bool(
        (_nsubj_tok and ('Number=Plur' in str(_nsubj_tok.get('morph', ''))
                         or _nsubj_tok.get('is_plural')))
        or (root_tok and root_tok.get('is_plural')))
    # expl:comp présent → toujours traiter (quelle que soit la classe sémantique)
    _sc_ok = (not root_tok
              or root_tok.get('semantic_class', '') not in _refl_abs_classes
              or (_refl_tok and _refl_tok.get('dep') == 'expl:comp'))
    # Passé composé avec participe passé (on s'est battu) : autoriser VerbForm=Part
    _has_aux_tense = any(x.get('dep') == 'aux:tense' for x in T)
    # Nomination / voix moyenne : un verbe réfléchi SUIVI d'un NOM PROPRE
    # (je m'appelle Hawa) n'est pas un réfléchi agentif (se blesser). Le nom
    # propre est l'attribut, le verbe reste nu → passif réflexif, PAS de yɛrɛ.
    if (_refl_tok and root_tok and any(
            x.get('pos') == 'PROPN'
            and x.get('dep') in ('xcomp', 'obj', 'attr', 'appos', 'dep')
            and x.get('head_index') == root_tok['orig_index']
            and x['orig_index'] != (_nsubj_tok['orig_index'] if _nsubj_tok else -1)
            for x in T)):
        root_tok['is_refl_passive'] = True
    if (_refl_tok and root_tok and root_tok.get('pos') == 'VERB'
            and not root_tok.get('is_statif')
            and ('VerbForm=Part' not in str(root_tok.get('morph', ''))
                 or _has_aux_tense)
            and _sc_ok):
        processed_indices.add(_refl_tok['orig_index'])
        if _refl_tok.get('dep') == 'iobj':
            # Passif réflexif (s'appeler, se souvenir) : le sujet est patient
            root_tok['is_refl_passive'] = True
        elif _refl_tok.get('dep') == 'expl:comp':
            _nsubj_morph = str(_nsubj_tok.get('morph', '')) if _nsubj_tok else ''
            _nsubj_pos   = _nsubj_tok.get('pos', '') if _nsubj_tok else ''
            _nsubj_surf  = str(_nsubj_tok.get('surface', '')).lower() if _nsubj_tok else ''
            # "on" = groupe → pluriel sémantique (ɲɔgɔn, pas yɛrɛ)
            _effective_plural = _subj_is_plural or _nsubj_surf == 'on'
            # NOUN singulier / Dem → passif réflexif
            # NOUN pluriel → chemin réciproque (les enfants s'aiment ≠ la porte se ferme)
            _nsubj_is_dem_or_noun = (
                'PronType=Dem' in _nsubj_morph
                or (_nsubj_pos == 'NOUN' and not _effective_plural))
            # Signal syntaxique fort : même surface sujet+réflexif (nous nous, ils ils)
            _refl_surf  = str(_refl_tok.get('surface', '')).lower()
            _same_surf  = bool(_nsubj_surf and _refl_surf == _nsubj_surf)
            if _nsubj_is_dem_or_noun or (root_tok.get('is_refl_passive') and not _effective_plural):
                # Passif réflexif (sujet Dem/NOUN singulier) → verbe nu
                root_tok['is_refl_passive'] = True
            elif _effective_plural and (_same_surf or root_tok.get('is_reciprocal')) and tree.get('clause_type') != 'content_question':
                # Réciproque : même surface (nous nous) OU LLM dit RECIPROCAL → ɲɔgɔn
                if not root_tok.get('is_plural'):
                    root_tok['is_plural'] = True
                tree['clause_type'] = 'reciprocal'
            # REFLEXIVE singulier (se blesser, se laver, s'asseoir) : pas de
            # clause_type ici → traité par refl_absolute en aval (avec ou sans
            # yɛrɛ selon que le verbe est agentif transitif ou inhérent).
        elif _subj_is_plural and tree.get('clause_type') != 'content_question':
            # Réciproque pluriel → ɲɔgɔn
            tree['clause_type'] = 'reciprocal'
        # Singulier dep='obj' : traité par refl_absolute en aval (refl_yere contrôle yɛrɛ)

    # ── RÉFLEXIF → S TAM S V (réfléchi, pas passif) ──────────────────────────
    # Tout verbe réflexif (clitique 's''/'se' = expl:comp) reprend le sujet par un
    # pronom objet : il s'est blessé → a yé a [blesser] (et NON le résultatif
    # passif a [blesser]ra, réservé au passif « il est lavé »).
    # Vaut quelle que soit la transitivité (ACTION incluse) : la distinction
    # réfléchi/passif vient de la présence du clitique, pas de l'intransitive_type.
    # is_refl_passive (s'appeler, se souvenir) reste exclu → verbe nu.
    if (_refl_tok and root_tok and root_tok.get('pos') == 'VERB'
            and not root_tok.get('is_statif')
            and not root_tok.get('is_refl_passive')
            and tree.get('clause_type') != 'reciprocal'):   # ne pas écraser réciproque
        print(f"DEBUG [REFL_ABSOLUTE SET] _refl_tok={_refl_tok.get('surface')}, root_tok={root_tok.get('surface')}, is_statif={root_tok.get('is_statif')}, is_refl_passive={root_tok.get('is_refl_passive')}, clause_type={tree.get('clause_type')}")
        tree['clause_type'] = 'refl_absolute'
        tree['is_transitive'] = True   # évite le F6 résultatif (V+ra)
        # yɛrɛ (soi-même) seulement pour les réflexifs AGENTIFS TRANSITIFS
        # (se blesser → a yé a yɛrɛ màjógin). Les inhérents posture/soin
        # (s'asseoir, se laver ∈ _refl_abs_classes) → pas de yɛrɛ : a yé a V.
        tree['refl_yere'] = (root_tok.get('intransitive_type') == 'ACTION'
                             and root_tok.get('semantic_class') not in _refl_abs_classes
                             and not root_tok.get('is_refl_idiomatic'))
        processed_indices.add(_refl_tok['orig_index'])
        # Capturer le verbe NU ici : step6 (participe résultatif du passé composé)
        # écraserait sinon m['V'] en V+ra et viderait le TAM.
        tree['refl_verb'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
        # Pronom de reprise = pronom sujet si présent, sinon 'a' (3sg)
        _rsubj = next((x for x in T
                       if x.get('dep') in ('nsubj', 'nsubj:pass')), None)
        if _rsubj and _rsubj.get('pos') == 'PRON' and _rsubj.get('bm'):
            tree['refl_pron'] = _rsubj.get('bm')
        else:
            tree['refl_pron'] = 'a'
        print(f"DEBUG [REFL_ABSOLUTE DONE] tree['refl_verb']={tree.get('refl_verb')}, tree['refl_pron']={tree.get('refl_pron')}, tree['refl_yere']={tree.get('refl_yere')}")

    # ── CLITIQUES PRONOMINAUX IOBJ SANS TRADUCTION ───────────────────────────
    # PRON dep='iobj' sans bm valide = clitique adverbial (en, y…) partie du
    # groupe verbal, sans équivalent bambara direct.
    # Ajouté à processed_indices pour empêcher step5 de créer un oblique parasite.
    _dative_mk = G_kg.get('dative_marker', 'ma') or 'ma'
    for _iobj_tok in T:
        if (_iobj_tok.get('dep') == 'iobj'
                and _iobj_tok.get('pos') == 'PRON'
                and _iobj_tok['orig_index'] not in processed_indices):
            _iobj_bm = str(_iobj_tok.get('bm', ''))
            if _iobj_bm and not _iobj_bm.startswith('['):
                # Clitique datif pronominal (te→i, me→n, lui→a) → "bm ma" (i ma)
                # placé après le verbe. Distinct du datif 'à'+NP (chemin obl:arg).
                m['OBL_ALL'].append({
                    'HEAD': j(_iobj_bm, _dative_mk), 'MARKER': '',
                    'local_clause_type': 'simple', 'COMPOUND': '', 'MOD': '',
                    'DEM_PREF': '', 'DEM_SUFF': '', 'DEP_TYPE': 'iobj',
                    'COMPOUND_IS_QUANTIFIER': False, 'MARKER_IS_PREFIX': False,
                })
                processed_indices.add(_iobj_tok['orig_index'])
            else:
                # Clitique adverbial sans bm (en/y) → ignorer (pas d'équivalent bm)
                processed_indices.add(_iobj_tok['orig_index'])
                for _ct in T:
                    if (_ct.get('dep') == 'case'
                            and _ct.get('head_index') == _iobj_tok['orig_index']
                            and _ct['orig_index'] not in processed_indices):
                        processed_indices.add(_ct['orig_index'])

    # ── INFINITIF ─────────────────────────────────────────────────────────────
    _is_infinitive = (root_tok and root_tok.get('pos') == 'VERB'
                      and 'Inf' in str(root_tok.get('morph', ''))
                      and not m.get('S'))
    if _is_infinitive:
        tree['clause_type'] = 'infinitive'

    # ── CONDITIONNEL : ní (Si...) / mána (Si jamais / Dès que) ──────────────
    _cond_mark = next((x for x in T
                       if x.get('dep') == 'mark'
                       and x.get('role') == 'conditional'
                       and x['orig_index'] not in processed_indices), None)
    if _cond_mark:
        _cond_bm = _cond_mark.get('bm') or 'ní'
        _has_jamais = any(
            str(x.get('surface', '')).lower() in ('jamais', 'jamais')
            for x in T if x['orig_index'] not in processed_indices)
        tree['conditional_marker'] = ('mána'
                                      if (_cond_bm == 'mána' or _has_jamais)
                                      else 'ní')
        tree['clause_type'] = 'conditional'
        processed_indices.add(_cond_mark['orig_index'])

    # ── TEMPOREL : quand / lorsque → marqueur postposé (ex: tuma min) ───────
    _tmp_mark = next((x for x in T
                      if x.get('dep') == 'mark'
                      and x.get('role') == 'temporal'
                      and x.get('bm')
                      and x['orig_index'] not in processed_indices), None)
    if _tmp_mark and tree.get('clause_type') not in ('conditional', 'infinitive'):
        tree['temporal_marker'] = _tmp_mark.get('bm')
        tree['clause_type'] = 'temporal'
        processed_indices.add(_tmp_mark['orig_index'])

    # ── F1 : TENSE TRANSFER ───────────────────────────────────────────────────
    aux_tense_tok = next((x for x in T
                          if x.get('dep') in ('aux', 'aux:tense', 'aux:pass', 'cop')
                          and x.get('tense') in ('past', 'hab', 'plup', 'imp')), None)
    print(f"DEBUG aux_tense_tok={aux_tense_tok}")
    print(f"DEBUG T_tenses={[(x.get('surface'), x.get('dep'), x.get('tense')) for x in T]}")

    if not aux_tense_tok and root_tok and root_tok.get('tense') == 'past':
        _aux_pass = next((x for x in T if x.get('dep') == 'aux:pass'), None)
        if _aux_pass and not root_tok.get('is_statif'):
            aux_tense_tok = root_tok

    if aux_tense_tok and root_tok:
        tree['tense'] = aux_tense_tok.get('tense', 'pres')
        tree['neg']   = tree['neg'] or aux_tense_tok.get('is_neg', False)
        tree['tam']   = _resolve_tam(tree['tense'], tree['neg'], G_kg)
    elif (root_tok and root_tok.get('tense') in ('past', 'fut', 'cond', 'hab', 'prog')
          and not root_tok.get('is_statif')):
        tree['tense'] = root_tok['tense']
        tree['neg']   = tree['neg'] or root_tok.get('is_neg', False)
        tree['tam']   = _resolve_tam(tree['tense'], tree['neg'], G_kg)
    else:
        copula_tok_f1 = next((x for x in T if x.get('dep') == 'cop'), None)
        aux_tok_f1    = next((x for x in T
                              if x.get('dep') in ('aux', 'aux:tense')
                              and x.get('role') == 'auxiliary'), None)
        if aux_tok_f1 and not aux_tense_tok:
            tree['tense'] = aux_tok_f1.get('tense', 'pres')
            tree['tam']   = _resolve_tam(tree['tense'], tree['neg'], G_kg)
        elif copula_tok_f1:
            if tree.get('clause_type') == 'locative':
                tree['tam'] = _resolve_tam('pres', tree.get('neg', False), G_kg) or 'bɛ'
        else:
            tree['tam'] = (G_kg.get('tam_default', '')
                           or _resolve_tam('pres', tree['neg'], G_kg))

    # NB: l'application des suffixes verbaux (li kɛ / la / kɛ) est centralisée
    # dans step6_copule ("Capturer root non encore traité comme V"). On ne fait
    # rien ici pour éviter le double-suffixage.

    # ── DÉJÀ ──────────────────────────────────────────────────────────────────
    _deja_tok = next((x for x in T if x.get('role') == 'temporal_already'), None)
    if _deja_tok:
        _deja_modifies_state = (root_tok and root_tok.get('pos') in ('NOUN', 'ADJ'))
        if tree.get('neg'):
            tree['already_marker'] = _deja_tok.get('bm_neg', 'fɔ́lɔ') or 'fɔ́lɔ'
        elif tree.get('tense') == 'past':
            tree['already_marker'] = _deja_tok.get('bm', 'kàban') or 'kàban'
        elif _deja_modifies_state:
            tree['already_marker'] = 'kàban'
        else:
            tree['already_marker'] = 'fɔ́lɔ'
        tree['already_position'] = ('HEAD'
                                    if _deja_tok.get('orig_index', 99) == 0
                                    and not _deja_modifies_state
                                    else 'END')
        processed_indices.add(_deja_tok['orig_index'])

    # ── PARTICIPIAL_TO ────────────────────────────────────────────────────────
    part_to_tok = next((x for x in T if x.get('role') == 'participial_to'
                        and x.get('bm')), None)
    if part_to_tok:
        s_bm = (next((x.get('bm') for x in T if x.get('dep') == 'nsubj'), None)
                or next((x.get('bm') for x in T if x.get('role') == 'pronoun'), None) or '')
        m['ADV'] = j(s_bm, part_to_tok.get('bm', '') + 'tɔ')
        processed_indices.add(part_to_tok['orig_index'])

    # ── PRÉDICAT PRIVATIF NOUN ROOT ───────────────────────────────────────────
    _priv_case = next((x for x in T
                       if x.get('dep') == 'case' and x.get('role') == 'privative'
                       and root_tok and x.get('head_index') == root_tok['orig_index']
                       and root_tok.get('pos') == 'NOUN'), None)
    if _priv_case:
        tree['clause_type'] = 'privative_pred'
        _priv_marker = _priv_case.get('bm_marker', 'tan')
        _priv_type   = _priv_case.get('privative_type', 'suffix')
        _root_bm     = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
        m['O'] = _root_bm + _priv_marker if _priv_type == 'suffix' else j(_root_bm, _priv_marker)
        processed_indices.add(root_tok['orig_index'])
        processed_indices.add(_priv_case['orig_index'])

    _priv_on_root_pron = next((x for x in T
                               if x.get('dep') == 'case' and x.get('role') == 'privative'
                               and root_tok and x.get('head_index') == root_tok['orig_index']
                               and root_tok.get('pos') in ('PRON', 'NOUN')), None)
    if _priv_on_root_pron and root_tok and not _priv_case:
        tree['clause_type'] = 'privative_pred'
        _priv_marker = (G_kg.get('privative_pron_marker', 'kɔ')
                        if root_tok.get('pos') == 'PRON'
                        else _priv_on_root_pron.get('bm_marker', 'tan'))
        _root_bm = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
        m['O'] = j(_root_bm, _priv_marker)
        m['S'] = ''
        processed_indices.add(root_tok['orig_index'])
        processed_indices.add(_priv_on_root_pron['orig_index'])

    # ── ADJ/NOUN ROOT AVEC COPULE → O ────────────────────────────────────────
    _root_has_nmod_chain = (any(x.get('dep') == 'nmod'
                                and x.get('head_index') == root_tok['orig_index']
                                for x in T) if root_tok else False)
    # Rôles de cas spatiaux (locatif au sens large) : dans/à (locative),
    # sur (surface→kan), sous (under→kɔrɔ), chez/près (associative→fɛ).
    # 'X est PREP Y' avec l'un de ces cas = clause LOCATIVE, pas équative.
    _loc_case_roles = G_kg.get('locative_case_roles',
                               {'locative', 'surface', 'under', 'associative'})
    _root_has_loc_case = (root_tok and any(
        x.get('dep') == 'case' and x.get('role') in _loc_case_roles
        and x.get('head_index') == root_tok['orig_index']
        for x in T))
    print(f"DEBUG _root_has_loc_case={_root_has_loc_case}")
    print(f"DEBUG case_tokens={[(x.get('surface'), x.get('role')) for x in T if x.get('dep') == 'case']}")

    _has_expletive = any(x.get('role') in G_kg.get('expletive_roles', {'expletive'}) for x in T)

    _has_comitative = any(x.get('role') == 'comitative' for x in T)
    if (root_tok
            and root_tok.get('pos') in ('ADJ', 'NOUN')
            and any(x.get('dep') == 'cop' for x in T)
            and root_tok['orig_index'] not in processed_indices
            and not _root_has_loc_case
            and not root_tok.get('is_participe_passe')
            and not root_tok.get('is_statif')
            and not (_has_expletive and root_tok.get('bm', '').startswith('['))
            and (not _root_has_nmod_chain or _has_comitative)):
        if _root_has_nmod_chain and _has_comitative:
            from rules.steps.step4_objet.objet_standard import _build_genitive_chain

            def _collect_chain_idx(tok, all_toks):
                result = set()
                nmod = next((t for t in all_toks if t.get('dep') == 'nmod'
                             and t.get('head_index') == tok['orig_index']), None)
                poss = next((t for t in all_toks if t.get('dep') == 'det'
                             and t.get('role') in ('pronoun', 'possessive')
                             and t.get('head_index') == tok['orig_index']
                             and t.get('bm')), None)
                for _ct in all_toks:
                    if _ct.get('dep') == 'case' and _ct.get('head_index') == tok['orig_index']:
                        result.add(_ct['orig_index'])
                if poss:
                    result.add(poss['orig_index'])
                if nmod:
                    result.add(nmod['orig_index'])
                    result |= _collect_chain_idx(nmod, all_toks)
                return result

            _root_bm = _build_genitive_chain(root_tok, T, G_kg)
            processed_indices.update(_collect_chain_idx(root_tok, T))
        elif _root_has_nmod_chain:
            # nmod chain sans comitative → laisser step6 gérer (équatif/qualificatif)
            return aux_tense_tok
        else:
            _root_bm = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
            _poss_on_root = next((x for x in T
                                  if x.get('dep') == 'det'
                                  and x.get('role') in ('pronoun', 'possessive')
                                  and x.get('head_index') == root_tok['orig_index']
                                  and x.get('bm')), None)
            if _poss_on_root:
                _poss_bm = _poss_on_root.get('bm', '')
                _root_bm = j('n', _root_bm) if _poss_bm == 'n' else j(_poss_bm, _root_bm)
                processed_indices.add(_poss_on_root['orig_index'])
            # Capturer amod (ex: grande fille → mùsoma bòn)
            _amod_on_root = next((x for x in T
                                  if x.get('dep') == 'amod'
                                  and x.get('head_index') == root_tok['orig_index']
                                  and x.get('bm')), None)
            if _amod_on_root:
                _amod_bm = _amod_on_root.get('bm', '')
                if _amod_on_root.get('pos') == 'ADJ':
                    _amod_bm = adj_man(_amod_bm)
                _root_bm = j(_root_bm, _amod_bm)
                processed_indices.add(_amod_on_root['orig_index'])
        m['O'] = _root_bm
        processed_indices.add(root_tok['orig_index'])

    elif _root_has_loc_case and root_tok and root_tok['orig_index'] not in processed_indices:
        tree['clause_type'] = 'locative'
        m['S'] = next((x.get('bm', '') for x in T
                       if x.get('dep') in ('nsubj', 'nsubj:pass') and x.get('bm')),
                      m.get('S', ''))
        _loc_case_root = next((x for x in T
                               if x.get('dep') == 'case' and x.get('role') in _loc_case_roles
                               and x.get('head_index') == root_tok['orig_index']), None)
        _loc_marker = _loc_case_root.get('bm_marker', 'la') if _loc_case_root else 'la'
        if not _loc_marker:
            _loc_marker = 'la'
        m['OBL_ALL'].append({
            'HEAD': root_tok.get('bm') or f"[{root_tok.get('lemma')}]",
            'MARKER': _loc_marker, 'local_clause_type': 'locative',
            'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
            'DEP_TYPE': 'case', 'COMPOUND_IS_QUANTIFIER': False,
            'MARKER_IS_PREFIX': False,
        })
        processed_indices.add(root_tok['orig_index'])
        if _loc_case_root:
            processed_indices.add(_loc_case_root['orig_index'])

    return aux_tense_tok