"""
rules/steps/step3_verbe.py
Étape 3 : verbe ROOT (m['V']), xcomp, négation, prohibitif, impératif,
infinitif, F1 tense transfer, déjà, privatif, ADJ/NOUN ROOT avec copule.
"""
from rules.core import j, _is_copula, _is_avoir, _resolve_tam, adj_man, INTRANS_SC


def run(T, tree, m, processed_indices, G_kg, NX_G,
        root_tok, xcomp_verb_tok, _neg_surfaces):
    """Retourne aux_tense_tok."""

    # ── DÉICTIQUE : voilà/voici → présentatif [O] félé ─────────────────────
    if root_tok and root_tok.get('role') == 'deictique':
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
        # Ne pas écraser m['V'] si step1 (avoir/possession) a déjà traité ce token
        _root_already_processed = root_tok['orig_index'] in processed_indices
        if not _root_is_copula and not _root_already_processed:
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
                          and x.get('role') not in ('temporal', 'temporal_already', 'reflexive')
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
                _vci = G_kg.get('verbal_coord_inf', '')
                m['V'] = j(m['V'], _vci, _crv_bm, *_crv_obl_parts)
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
                    _coord_pron = G_kg.get('pronoun_3pl_coord', '')
                else:
                    _coord_pron = G_kg.get('pronoun_3sg_default', '')
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
                    # No object: apply TAM + intransitive routing for the conj verb
                    _crv_tense = _crv.get('tense', tree.get('tense', 'pres'))
                    _crv_neg   = tree.get('neg', False)
                    _crv_tam   = _resolve_tam(_crv_tense, _crv_neg, G_kg) or 'bɛ'
                    _crv_is_pres = _crv_tense in ('pres', 'hab')
                    _crv_is_past_pos = (_crv_tense == 'past' and not _crv_neg)
                    _crv_sc    = _crv.get('semantic_class', '')
                    _crv_it    = _crv.get('intransitive_type', '')
                    _crv_valid = _crv_bm and not _crv_bm.startswith('[')
                    _crv_is_absolu  = (_crv_it == 'ABSOLU' or _crv_sc in INTRANS_SC)
                    _crv_needs_nom  = (
                        _crv_it in ('nominalized', 'ACTION', 'support')
                        and _crv_sc not in INTRANS_SC
                        and _crv_sc not in ('having', 'technique', 'action')
                        and _crv_sc != 'consumption_liquid'
                        and _crv_valid)
                    _crv_is_action  = (_crv_sc == 'action' and _crv_valid)

                    if _crv_is_past_pos and _crv_is_absolu and _crv_valid:
                        # INTRANS_SC/ABSOLU passé positif → résultatif V+ra/la/na (no TAM)
                        _morpho_res = G_kg.get('morpho_rules', {}).get('resultative', {})
                        from rules.kg_rule_engine import apply_morpho_suffix as _ams
                        _crv_res = _ams(_crv_bm, _morpho_res) if _morpho_res else _crv_bm
                        m['V'] = j(m['V'], G_kg.get('coord_verb_marker', ''), _crv_res, *_crv_obl_parts)
                    elif _crv_needs_nom:
                        # Transitif sans COD → nominalisé V+li kɛ avec TAM propre
                        _nom_sfx = G_kg.get('nominalization_verb_suffix', '')
                        _nom = (_crv_bm + _nom_sfx
                                if _nom_sfx and not _crv_bm.endswith((G_kg.get('nominalization_verb_suffix', ''), G_kg.get('nominalization_verb_suffix_alt', ''))) else _crv_bm)
                        m['V'] = j(m['V'], G_kg.get('coord_verb_marker', ''), _coord_pron, _crv_tam,
                                   _nom, G_kg.get('coord_action_suffix', ''), *_crv_obl_parts)
                    elif _crv_is_action:
                        # Nom-ACTION (báara) : V la (pres) ou V kɛ (past) avec TAM
                        _act_sfx = G_kg.get('action_pres_suffix', '') if _crv_is_pres else G_kg.get('coord_action_suffix', '')
                        m['V'] = j(m['V'], G_kg.get('coord_verb_marker', ''), _coord_pron, _crv_tam,
                                   _crv_bm, _act_sfx, *_crv_obl_parts)
                    else:
                        # Verbe coordonné sans COD (boire, partir…) : inclure le pronom
                        # et le TAM pour éviter la confusion avec l'impératif ("wa mìn")
                        m['V'] = j(m['V'], G_kg.get('coord_verb_marker', ''),
                                   _coord_pron, _crv_tam, _crv_bm, *_crv_obl_parts)
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
            # m['O'] n'est pas encore rempli à cette étape (step4 vient après).
            # On regarde directement dans T si le ROOT a son propre obj direct.
            _root_has_own_obj = any(
                x.get('dep') == 'obj'
                and x.get('head_index') == root_tok['orig_index']
                and x.get('pos') in ('NOUN', 'PROPN', 'PRON')
                for x in T
            )
            if _root_has_own_obj:
                m['O_XCOMP'] = _xcomp_obj_bm
            else:
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
    # Restrictif "ne... que/qu" : marquer 'que/qu' (dep=advmod) comme trait\u00e9
    # pour que step5 ne le convertisse pas en oblique temporel.
    if _has_neg_adv:
        _que_adv = next((x for x in T
                         if x.get('dep') == 'advmod'
                         and str(x.get('surface', '')).lower().rstrip("'") in ('que', 'qu')), None)
        if _que_adv:
            processed_indices.add(_que_adv['orig_index'])

    # ── RESTRICTIVE NE...QUE (Rule 6) ─────────────────────────────────────
    # Detect: ne + que (restrictive "only") = foyi yé ni X tɛ
    # Example: "tu ne serais qu'un pleutre" → i bɛ yé foyi yé ni sègɛ tɛ
    _has_ne = tree.get('neg')
    _que_restrictive = next((x for x in T
                             if str(x.get('surface', '')).lower().replace(''', "'").replace(''', "'") in ('que', "qu'")
                             and x.get('dep') in ('mark', 'advmod')
                             and x.get('orig_index') not in processed_indices), None)
    if _has_ne and _que_restrictive:
        # This is ne...que restrictive
        _attr = next((x for x in T
                      if x.get('orig_index') > _que_restrictive['orig_index']
                      and x.get('pos') in ('NOUN', 'ADJ', 'PROPN')
                      and x.get('orig_index') not in processed_indices), None)
        if _attr:
            tree['restrictive_attr'] = _attr.get('bm') or f"[{_attr.get('lemma')}]"
            tree['restrictive_neg'] = bool(_has_ne)  # store neg for TAM in renderer
            processed_indices.add(_que_restrictive['orig_index'])
            processed_indices.add(_attr['orig_index'])
            tree['neg'] = False  # éviter double-négation dans le reste du pipeline

    # ── PROHIBITIF ────────────────────────────────────────────────────────────
    # expl:subj avec role='subject' = pronom personnel réel (spaCy mal-étiqueté pour
    # les passés composés avec être : "il est tombé" → il=expl:subj, pas vrai expletif)
    _has_real_subj = (
        any(x.get('dep') in ('nsubj', 'nsubj:pass') for x in T)
        or any(x.get('dep') == 'expl:subj' and x.get('role') == 'subject' for x in T)
    )
    _is_prohibitive = (root_tok and root_tok.get('pos') == 'VERB'
                       and not m.get('S') and tree.get('neg')
                       and not _has_real_subj
                       and not any(x.get('role') == 'interrogative' for x in T))
    if _is_prohibitive:
        tree['neg'] = False
        # Poser clause_type='prohibitive' dès step3 pour que step7 saute
        # la nominalization (kàna sò, pas kàna sòli kɛ pour psych_emotion).
        tree['clause_type'] = 'prohibitive'

    # ── EST-CE QUE : marqueur interrogatif → Yala ────────────────────────────
    # Signal : -ce dep='nsubj' + que dep='mark' + être ROOT
    _est_ce_que = any(
        str(x.get('surface', '')).lower().rstrip('-').lstrip('-') == 'ce'
        and x.get('dep') in ('nsubj', 'expl:subj', 'dep', 'ROOT')  # 'ROOT' : "-ce" ROOT dans "Est-ce qu'il dort?"
        for x in T)
    _has_que_mark = any(
        str(x.get('surface', '')).lower() in ('que', "qu'")
        and x.get('dep') in ('mark', 'dep')  # 'dep' pour "Qu'est-ce qu'il pourrait faire?"
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

    # ── QU'EST-CE QUE (Rule 7) ───────────────────────────────────────────────
    # Detect: qu' (interrogative PRON root) + est-ce + que (mark)
    # Pattern: Qu' (ROOT) + être (dep/aux) + -ce (dep) + que (mark) + subject
    # "Qu'est-ce qu'il pourrait t'arriver ?" → mún S TAM se ka t' sé yèn ?
    _qu_root = (root_tok
                and root_tok.get('pos') == 'PRON'
                and root_tok.get('dep') == 'ROOT'
                and 'Int' in str(root_tok.get('morph', ''))
                and str(root_tok.get('surface', '')).lower().startswith('qu'))
    # Check for 'ce' token and 'que' mark (flexible dep matching for spaCy variations)
    _has_ce = any(
        'ce' in str(x.get('surface', '')).lower().rstrip('-').lstrip('-')
        for x in T)
    _has_subj_or_expl = any(
        x.get('dep') in ('expl:subj', 'nsubj')
        and x.get('pos') == 'PRON'
        for x in T)
    _has_ce_que = _has_ce and _has_que_mark and _has_subj_or_expl

    if _qu_root and _has_ce_que:
        # This is qu'est-ce que construction (Rule 7)
        m['QUEST_WORD'] = 'mún'  # What
        processed_indices.add(root_tok['orig_index'])

        # Assign correct TAM for quest_ce_que : tense from modal verb (pourrait=fut, peut=pres)
        _qcq_modal = next((x for x in T
                           if x.get('semantic_class') == 'modal'
                           and x.get('pos') == 'VERB'
                           and x.get('tense')
                           and x.get('tense') != 'pres'), None)
        _qcq_tense = _qcq_modal.get('tense') if _qcq_modal else tree.get('tense', 'pres')
        if _qcq_tense and _qcq_tense != tree.get('tense', 'pres'):
            tree['tense'] = _qcq_tense
        tree['tam'] = _resolve_tam(tree.get('tense', 'pres'), tree.get('neg', False), G_kg) or 'bɛ'

        # Handle subject: use only REAL subjects (nsubj), skip expletive subjects (expl:subj)
        # "Qu'est-ce qu'il pourrait arriver" → 'il' is expletive → no S
        # "Qu'est-ce qu'il peut faire" → 'il' is real subject → S=a
        _expl_subj = next((x for x in T if x.get('dep') == 'expl:subj' and x.get('role') == 'subject'), None)
        if _expl_subj:
            # Clear S if subject is expletive (no subject in output)
            m['S'] = ''
            processed_indices.add(_expl_subj['orig_index'])

        # Extract modal verb (pouvoir → se) for quest_ce_que + modal structure
        # "Qu'est-ce qu'il pourrait t'arriver" → dep='dep' ; "pourrait faire" → dep='acl:relcl'
        _modal_verb = next((x for x in T
                           if x.get('pos') == 'VERB'
                           and x.get('dep') in ('dep', 'aux', 'acl:relcl')
                           and x.get('semantic_class') == 'modal'
                           and x != root_tok
                           and x.get('bm')), None)
        if _modal_verb and not m.get('V'):
            m['V'] = _modal_verb.get('bm')
            processed_indices.add(_modal_verb['orig_index'])

            # Construction impersonnelle + modal : "Qu'est-ce qu'il pourrait t'arriver ?"
            # → mún bɛ se ka i se yèn ? (S=mún, V=modal, O=datif, V_ACT=xcomp_verb)
            if _expl_subj:
                # mún = sujet (le "quoi" qui peut arriver)
                if not m.get('S'):
                    m['S'] = root_tok.get('bm', '')
                # xcomp verb (arriver → V_ACTION infinitif après ka)
                _xcomp_verb = next((x for x in T
                                    if x.get('dep') == 'xcomp'
                                    and x.get('pos') == 'VERB'
                                    and x.get('bm')), None)
                if _xcomp_verb:
                    m['V_ACTION'] = _xcomp_verb.get('bm', '')
                    processed_indices.add(_xcomp_verb['orig_index'])
                # Datif (t' → i) = bénéficiaire de l'action = m['O']
                _xcomp_subj = next((x for x in T
                                   if x.get('pos') == 'PRON'
                                   and x.get('dep') in ('nsubj', 'iobj')
                                   and x.get('role') == 'object'
                                   and x.get('bm')), None)
                if _xcomp_subj:
                    m['O'] = _xcomp_subj.get('bm', '')
                    processed_indices.add(_xcomp_subj['orig_index'])
                # Template quest_ce_que_modal : {S} {TAM} {V} ka {O} {V_ACT} {OBL} ?
                tree['clause_type'] = 'quest_ce_que_modal'
                tree['est_ce_que']  = True
            else:
                # quest_ce_que normal (sujet réel) : "Qu'est-ce qu'il pourrait faire?"
                # → a bɛ se ka mún V_ACT ? (template quest_ce_que_modal)
                # Forcer quest_ce_que_modal pour tous les tenses (pas seulement présent).
                _xcomp_verb_q = next((x for x in T
                                      if x.get('dep') == 'xcomp'
                                      and x.get('pos') == 'VERB'
                                      and x.get('bm')), None)
                if _xcomp_verb_q and not m.get('V_ACTION'):
                    m['V_ACTION'] = _xcomp_verb_q.get('bm', '')
                    processed_indices.add(_xcomp_verb_q['orig_index'])
                tree['clause_type'] = 'quest_ce_que_modal'
                tree['est_ce_que']  = True
                # t' stocké comme XCOMP_SUBJ pour usage aval
                _xcomp_subj = next((x for x in T
                                   if x.get('pos') == 'PRON'
                                   and x.get('dep') in ('nsubj', 'iobj')
                                   and any(c.get('dep') == 'xcomp'
                                          and c.get('head_index') == _modal_verb['orig_index']
                                          for c in T)
                                   and x.get('bm')), None)
                if _xcomp_subj:
                    m['XCOMP_SUBJ'] = _xcomp_subj.get('bm')
                    processed_indices.add(_xcomp_subj['orig_index'])

        # Mark ce and que as processed
        for _cx in T:
            if str(_cx.get('surface', '')).lower().rstrip('-').lstrip('-') == 'ce':
                processed_indices.add(_cx['orig_index'])
        for _qx in T:
            if str(_qx.get('surface', '')).lower() in ('que', "qu'"):
                processed_indices.add(_qx['orig_index'])

    # ── DISTRIBUTIF : chacun d'entre [PRON] → [PRON] kélen kélenna bɛ ─────────
    _distrib_one_root = (root_tok
                         and root_tok.get('pos') == 'PRON'
                         and root_tok.get('role') == 'distributive_one')
    if _distrib_one_root:
        _nmod_compl = next((x for x in T
                            if x.get('dep') == 'nmod'
                            and x.get('head_index') == root_tok['orig_index']
                            and x.get('bm')), None)
        _base_bm = (_nmod_compl.get('bm') if _nmod_compl
                    else root_tok.get('bm') or '')
        _distrib_bm = root_tok.get('bm') or G_kg.get('distributive_one_bm', '')
        m['S'] = j(_base_bm, _distrib_bm)
        processed_indices.add(root_tok['orig_index'])
        if _nmod_compl:
            processed_indices.add(_nmod_compl['orig_index'])
        # Marquer les tokens de la chaîne d'entre (case/fixed) traités
        for _cx in T:
            if (_cx.get('dep') in ('case', 'fixed')
                    and _nmod_compl
                    and _cx.get('head_index') in (_nmod_compl['orig_index'],
                                                   root_tok['orig_index'])):
                processed_indices.add(_cx['orig_index'])

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
        if not _acl_verb:
            # "Qu'est-ce qu'il fait?" : spaCy parse le vrai verbe avec dep='dep' role='content'
            _acl_verb = next((x for x in T
                              if x.get('dep') == 'dep'
                              and x.get('pos') == 'VERB'
                              and x.get('role') == 'content'
                              and x.get('bm')), None)
        if _acl_verb:
            # Le verbe acl:relcl devient le verbe principal
            m['V'] = _acl_verb.get('bm', '')
            processed_indices.add(_acl_verb['orig_index'])
            # Mémoriser le tense du verbe acl:relcl pour override TAM après le bloc F1
            # (le bloc else ligne ~1225 écrase sinon avec le tense du ROOT PRON=pres)
            _acl_tense = _acl_verb.get('tense', 'pres')
            if _acl_tense and _acl_tense != 'pres':
                tree['_cq_acl_tense'] = _acl_tense
            # Le sujet est le nsubj du acl:relcl
            # Exclure les clitiques datifs/accusatifs (role='object') mislabélés
            # dep='nsubj' : dans "il pourrait t'arriver", t' est COI, pas sujet.
            _acl_subj = next((x for x in T
                              if x.get('dep') in ('nsubj', 'nsubj:pass')
                              and x.get('head_index') == _acl_verb['orig_index']
                              and x.get('bm')
                              and x.get('role') != 'object'), None)
            if _acl_subj:
                m['S'] = _acl_subj.get('bm', '')
                processed_indices.add(_acl_subj['orig_index'])
            # L'interrogatif ROOT devient O (sauf si quest_ce_que impersonnel l'a déjà mis en S)
            if not m.get('S') or m.get('S') != root_tok.get('bm', ''):
                if not m.get('O'):  # ne pas écraser si quest_ce_que a posé O=datif
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
                    or str(_exist_obj.get('surface', '')).endswith(G_kg.get('plural_fr_ending', 's'))):
                _plur_sfx = G_kg.get('plural_noun_suffix', '')
                if _plur_sfx and not _es_bm.endswith(_plur_sfx):
                    _es_bm += _plur_sfx
            m['O'] = _es_bm
            processed_indices.add(_exist_obj['orig_index'])
        processed_indices.add(root_tok['orig_index'])
        for _yt in T:
            if (str(_yt.get('surface', '')).lower().rstrip("'").rstrip('\u2019') == 'y'
                    or _yt.get('dep') in ('expl:subj', 'expl:comp')):
                processed_indices.add(_yt['orig_index'])
        return None  # ← retour anticipé, bloque tout step suivant

    # ── AVOIR ROOT (possession/état/âge/douleur) ─────────────────────────────
    # Exclure si step1_avoir a déjà posé noun_phrase_have (sensation passée composée)
    _avoir_poss = (root_tok
                   and root_tok.get('lemma', '').lower() == 'avoir'
                   and root_tok.get('pos') in ('VERB', 'AUX')
                   and not _has_y_expl
                   and tree.get('clause_type') != 'noun_phrase_have')
    if _avoir_poss:
        _obj_tok = next((x for x in T
                             if x.get('dep') in ('obj', 'nsubj', 'nsubj:pass', 'obl:arg')
                             and x.get('pos') in ('NOUN', 'PROPN')), None)
        if _obj_tok:
            _avoir_morph = str(root_tok.get('morph', ''))
            _avoir_is_present_indic = (
                root_tok.get('tense') in ('pres', None)
                and 'Mood=Cnd' not in _avoir_morph
                and 'VerbForm=Part' not in _avoir_morph
            )

            if not _avoir_is_present_indic:
                # Non-présent (conditionnel, passé, futur) → S TAM O sɔrɔ
                _obj_bm = _obj_tok.get('bm', '')
                if _obj_tok.get('is_plural') and _obj_bm:
                    _plur = G_kg.get('plural_noun_suffix', '')
                    if _plur and not _obj_bm.endswith(_plur):
                        _obj_bm += _plur
                # Déterminant possessif sur l'objet (ton appel → i ka wéle) :
                # même logique que step4_objet.objet_standard, car ce bloc
                # marque l'objet 'processed' avant que step4 ne puisse agir.
                _poss_obj = next((x for x in T
                                  if x.get('dep') == 'det'
                                  and x.get('role') in ('pronoun', 'possessive')
                                  and x.get('head_index') == _obj_tok['orig_index']), None)
                if _poss_obj and _obj_bm:
                    _poss_bm = _poss_obj.get('bm') or _poss_obj.get('surface', '')
                    if _poss_bm:
                        _is_rel = (_obj_tok.get('is_relational', False)
                                   or _obj_bm in G_kg.get('relational_bms', set()))
                        if _is_rel:
                            _obj_bm = j(_poss_bm, _obj_bm)
                        else:
                            _gen_mk = G_kg.get('genitive_marker', 'ka') or 'ka'
                            _obj_bm = j(_poss_bm, _gen_mk, _obj_bm)
                        processed_indices.add(_poss_obj['orig_index'])
                if _obj_bm and _obj_tok['orig_index'] not in processed_indices:
                    m['O'] = _obj_bm
                    processed_indices.add(_obj_tok['orig_index'])
                m['V'] = G_kg.get('possession_past_verb', '')
            else:
                # Présent indicatif → construction possessive O bɛ S bóló
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
                    _pt = _obj_tok.get('possession_type', 'UNKNOWN')
                    tree['have_type'] = _pt
                    tree['have_obj_lemma'] = _obj_tok.get('lemma', '')
                    if _pt == 'MATERIAL':
                        tree['possession_type'] = 'material'
                    elif _pt in ('ABSTRACT', 'PAIN', 'AGE'):
                        tree['possession_type'] = _pt.lower()
                if _obj_tok['orig_index'] not in processed_indices:
                    _obj_bm = _obj_tok.get('bm', '')
                    _plur = G_kg.get('plural_noun_suffix', '')
                    if _obj_tok.get('is_plural') and _obj_bm and _plur and not _obj_bm.endswith(_plur):
                        _obj_bm += _plur
                    if _obj_bm:
                        m['O'] = _obj_bm
                        processed_indices.add(_obj_tok['orig_index'])
                if _det_interrog:
                    _interrog_qty = next((x for x in T
                                          if x.get('role') == 'interrogative'
                                          and x.get('bm')), None)
                    if _interrog_qty:
                        tree['interrog_qty'] = _interrog_qty.get('bm', '')
                        processed_indices.add(_interrog_qty['orig_index'])

    # ── COMPOUND VERB → infinitif ─────────────────────────────────────────────
    # Cas : spaCy parse ADJ comme ROOT avec VERB compound (ex: publier une info crédible)
    # → traiter comme infinitif
    if (root_tok and root_tok.get('pos') == 'ADJ'
            and not root_tok.get('is_statif')
            and not any(x.get('dep') in ('cop', 'aux:pass') for x in T)):
        _compound_verb = next((x for x in T
                               if x.get('dep') in ('compound', 'ROOT', 'advcl', 'amod')
                               and x.get('tense') != 'participial_to'
                               and x.get('role') != 'participial_to'
                               and x.get('lemma', '').lower() not in ('', 'none')
                               and x.get('orig_index') != root_tok.get('orig_index')
                               and x['orig_index'] not in processed_indices), None)
        if _compound_verb:
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
        else:
            # ADJ ROOT sans compound_verb → check participial_to (complètes en venant)
            _has_part_to = any(x.get('role') == 'participial_to' for x in T)
            if root_tok.get('bm') and _has_part_to:
                m['V'] = root_tok.get('bm', '')
                tree['tam'] = ''
                processed_indices.add(root_tok['orig_index'])

    # ── INFINITIF : ka + O + V ───────────────────────────────────────────────
    # ex: manger du riz → ka iri dún
    if (root_tok and 'VerbForm=Inf' in str(root_tok.get('morph', ''))
            and not m.get('S')
            and not any(x.get('dep') in ('nsubj', 'nsubj:pass') for x in T)):
        tree['tam'] = ''

    # ── IMPÉRATIF AFFIRMATIF ──────────────────────────────────────────────────
    _has_excl = any(str(x.get('surface', '')).strip() == '!' for x in T)
    _no_subj  = not _has_real_subj
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
        tree['tam'] = ''
        tree['clause_type'] = 'imperative'
        tree['_imp_tam_done'] = True  # verrou : F1 ne doit pas écraser

    # ── OPTATIF/JUSSIF : "Que Dieu t'aide" (sub + sujet explicite) ───────────
    # tense='sub' (posé par le mark SCONJ 'que' sur le ROOT, cf spacy_parser)
    # n'était consommé nulle part en aval (aucune entrée 'sub' dans la table
    # TAM) → retombait sur le TAM par défaut (bɛ, présent), perdant le sens
    # optatif. Avec sujet explicite (≠ impératif bare ci-dessus) : S ka O V.
    elif root_tok and root_tok.get('tense') == 'sub' and not tree.get('est_ce_que'):
        tree['tense'] = 'sub'  # propagé pour que PatternRule optative + TransformRule KG gèrent le TAM

    # ── RÉFLEXIF / RÉCIPROQUE ────────────────────────────────────────────────
    # Détection via Reflex=Yes (spaCy morph) — pas de surfaces codées en dur.
    # Trois cas :
    #   1. dep='iobj'      → passif réflexif (s'appeler) : is_refl_passive=True
    #   2. dep='obj'/'expl:comp' + singulier → agentif (se blesser) → yɛrɛ
    #   3. dep='expl:comp' + pluriel → réciproque (se battre) → ɲɔgɔn
    _refl_tok = next((x for x in T
                      if x.get('dep') in ('expl:comp', 'expl:pass', 'obj', 'iobj')
                      and x.get('pos') == 'PRON'
                      # Exclure l'apostrophe nue d'élision (j'/c'/qu' → token "'")
                      # mal taguée expl:comp : ce n'est PAS un clitique réfléchi.
                      # Un vrai clitique (s'/se/me/te/nous) a une surface non vide
                      # une fois les apostrophes retirées.
                      and str(x.get('surface', '')).strip().strip("'''") != ''
                      and ('Reflex=Yes' in str(x.get('morph', ''))
                           # dep='expl:comp'/'expl:pass' sur un PRON = clitique réfléchi (UD fr),
                           # même quand spaCy omet Reflex=Yes (tagging incohérent)
                           or x.get('dep') in ('expl:comp', 'expl:pass')
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
        _subj_x = next((x for x in T
                        if x.get('dep') in ('nsubj', 'nsubj:pass')
                        or (x.get('dep') == 'dep' and x.get('role') == 'subject')), None)
        _subj_pers = next((p.split('=')[1] for p in str(_subj_x.get('morph', '')).split('|')
                           if p.startswith('Person=')), '') if _subj_x else ''
        print(f"DEBUG [REFLEXIVE FALLBACK] root_tok={root_tok.get('surface')}, _subj_x={_subj_x.get('surface') if _subj_x else None}, _subj_pers={_subj_pers}")
        _refl_tok = next((x for x in T
                          if x.get('dep') in ('iobj', 'obj')
                          and x.get('pos') == 'PRON'
                          and x.get('head_index') == root_tok.get('orig_index')
                          and _subj_pers
                          and f'Person={_subj_pers}' in str(x.get('morph', ''))
                          # 3e personne : le/la/l'/les NE sont PAS réfléchis
                          # (seul 'se/s'' l'est → Reflex=Yes). 1re/2e personne :
                          # me/te/nous/vous coréférents au sujet = réfléchis.
                          # Sans ce garde, 'il l'a vu' (il=3, l'=3) était happé à
                          # tort comme réfléchi → 'a yé a yɛrɛ yé' au lieu de
                          # 'a yé a yé'.
                          and (_subj_pers in ('1', '2')
                               or 'Reflex=Yes' in str(x.get('morph', '')))), None)
        print(f"DEBUG [REFLEXIVE FALLBACK] _refl_tok after fallback: {_refl_tok.get('surface') if _refl_tok else None}")
    _refl_abs_classes = {'posture', 'motion', 'biological', 'spontaneous'}
    _nsubj_tok = next((x for x in T if x.get('dep') in ('nsubj', 'nsubj:pass')), None)
    _subj_is_plural = bool(
        (_nsubj_tok and ('Number=Plur' in str(_nsubj_tok.get('morph', ''))
                         or _nsubj_tok.get('is_plural')))
        or (root_tok and root_tok.get('is_plural')))
    # Signal réciproque fort : sujet pluriel + (même surface "nous nous" OU is_reciprocal).
    # Ce signal prime sur la classe sémantique (aimer/spontaneous ne doit pas
    # bloquer "nous nous aimons" → réciproque ɲɔgɔn).
    _nsubj_surf_early = str(_nsubj_tok.get('surface', '')).lower() if _nsubj_tok else ''
    _refl_surf_early  = str(_refl_tok.get('surface', '')).lower() if _refl_tok else ''
    _recip_signal = bool(
        _subj_is_plural
        and ((root_tok and root_tok.get('is_reciprocal'))
             or (_nsubj_surf_early and _refl_surf_early == _nsubj_surf_early)))
    # expl:comp présent → toujours traiter (quelle que soit la classe sémantique)
    # Signal réciproque fort → traiter aussi (réciproque > classe autonome)
    _sc_ok = (not root_tok
              or root_tok.get('semantic_class', '') not in _refl_abs_classes
              or (_refl_tok and _refl_tok.get('dep') == 'expl:comp')
              or _recip_signal)
    # Passé composé avec participe passé (on s'est battu) : autoriser VerbForm=Part
    _has_aux_tense = any(x.get('dep') == 'aux:tense' for x in T)
    # Nomination / voix moyenne : un verbe réfléchi SUIVI d'un NOM PROPRE
    # (je m'appelle Hawa) n'est pas un réfléchi agentif (se blesser). Le nom
    # propre est l'attribut, le verbe reste nu → passif réflexif, PAS de yɛrɛ.
    _has_propn_attr = (_refl_tok is not None and root_tok is not None and any(
            x.get('pos') == 'PROPN'
            and x.get('dep') in ('xcomp', 'obj', 'attr', 'appos', 'dep')
            and x.get('head_index') == root_tok['orig_index']
            and x['orig_index'] != (_nsubj_tok['orig_index'] if _nsubj_tok else -1)
            for x in T))
    if _has_propn_attr:
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
            # Exception datif réflexif : verbe de communication + singulier + sans attribut
            # nominal (PROPN) + pas interrogatif → "je me parle", "elle se parle"
            # → libérer le token pour la boucle iobj (is_refl_passive bloque déjà refl_absolute).
            # Protégé : "je m'appelle Hawa" (_has_propn_attr), "comment t'appelles-tu?"
            # (content_question), "nous nous parlons" (_subj_is_plural).
            if (root_tok.get('semantic_class') in ('communication', 'saying')
                    and not _has_propn_attr
                    and not _subj_is_plural
                    and not _recip_signal
                    and tree.get('clause_type') not in ('content_question',)):
                processed_indices.discard(_refl_tok['orig_index'])
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
            elif _effective_plural and tree.get('clause_type') != 'content_question':
                # Sujet PLURIEL + réfléchi 'se' → RÉCIPROQUE (ɲɔgɔn) par défaut.
                # Exception : semantic_class='preparation' est structurellement réflexif
                # même au pluriel (nous nous préparons = chacun se prépare pour soi,
                # pas "nous préparons l'un l'autre") → refl_absolute en aval.
                _nat_refl_plural = (root_tok is not None
                                    and root_tok.get('semantic_class') in ('preparation',))
                if not _nat_refl_plural:
                    if not root_tok.get('is_plural'):
                        root_tok['is_plural'] = True
            # REFLEXIVE singulier (se blesser, se laver, s'asseoir) : pas de
            # clause_type ici → traité par refl_absolute en aval (avec ou sans
            # yɛrɛ selon que le verbe est agentif transitif ou inhérent).
        elif _subj_is_plural and tree.get('clause_type') != 'content_question':
            pass  # PatternRule KG détecte via has_reciprocal
            # Réciproque pluriel → ɲɔgɔn
        # Singulier dep='obj' : traité par refl_absolute en aval (refl_yere contrôle yɛrɛ)

    # ── ARBRE DE DÉCISION RÉFLEXIF ────────────────────────────────────────────
    # Appliqué à tout verbe V associé à son pronom réflexif (se/me/te/nous/vous).
    #
    # Étape 1 : Existence autonome
    #   → Verbe sans 'se' inexistant (s'évanouir, se souvenir) = Essentiellement pronominal
    #   → Signal : is_refl_Subjective=True / is_refl_passive=True  → SKIP (verbe nu)
    #
    # Étape 2 : Animation du sujet
    #   → Sujet inanimé (objet/concept) = Sens Passif ("les voitures se vendent")
    #   → Signal : is_refl_passive=True (KG) ou _nsubj_tok.is_animate=False
    #   → SKIP : laisser le chemin passif gérer
    #
    # Étape 3 : Pluralité / Réciprocité
    #   → Sujet pluriel + action partagée (A→B et B→A) = Réciproque
    #   → Géré plus haut : clause_type='reciprocal'  → SKIP
    #
    # Étape 4 : Catégorie fine (_classify_refl_verb, 5 catégories, VerbeNet-groundé
    #   — VerbeNet = VerbNet français de Danlos et al., github.com/aymara/verbenet)
    #   → SOIN_CORPOREL (floss-41.2.1/braid-41.2.2) : se laver, se coiffer, s'habiller
    #   → POSTURE (assuming_position-50, classe pronominale dédiée) : s'asseoir, se lever
    #   → ACCIDENTEL (hurt-40.8.3)                   : se blesser, se couper, se brûler
    #   → ACTIF (reste)                              : se préparer, se déguiser
    #   → Bambara : MÊME structure S TAM refl_pron V pour les 4 catégories
    #   → yɛrɛ NON par défaut, sauf ACCIDENTEL (True) ou emphase explicite ('lui même')
    #
    # Résultat : clause_type='refl_absolute' pour les 4 catégories ci-dessus

    # Étape 1 LLM : existence autonome + intentionnalité
    # Appeler _classify_refl_verb si disponible (injecté par TranslationEngine)
    _classify_refl_fn = G_kg.get('_classify_refl_verb')
    _refl_cat = 'actif'   # défaut si classifieur absent
    if _refl_tok and root_tok and _classify_refl_fn:
        # Exception : 'se mettre à V' et similaires → 'se' est un COD réfléchi
        # (mettre est purement transitif ; le sujet s'applique l'action à lui-même)
        # Signal fiable : mark locatif ('à') sur l'xcomp infinitif.
        _locative_xcomp_mark = (
            xcomp_verb_tok is not None
            and any(x.get('dep') == 'mark' and x.get('role') == 'locative'
                    and x.get('head_index') == xcomp_verb_tok['orig_index']
                    for x in T)
        )
        # Comportement réflexif depuis KG (annoté par kg_gateway.apply_kg_semantic_behaviors)
        # root_tok['_kg_reflexive'] = 'pronominal'/'posture'/'actif'/'accidentel' selon KG
        _kg_refl = root_tok.get('_kg_reflexive', '')
        if (root_tok.get('intransitive_type') == 'ABSOLU'
                or (root_tok.get('is_refl_Subjective') and not _locative_xcomp_mark)
                or _kg_refl == 'pronominal'):
            # KG: biological/pronominal → verbe nu (se réveiller, s'évanouir)
            _refl_cat = 'pronominal'
        else:
            _refl_lemma = root_tok.get('lemma', '')
            _sens_fr = root_tok.get('sens_fr', '')
            if _sens_fr:
                _inf_candidate = _sens_fr.split('.')[0].strip().split(',')[0].strip()
                if _inf_candidate.startswith('se '):
                    _inf_candidate = _inf_candidate[3:]
                if _inf_candidate and ' ' not in _inf_candidate:
                    _refl_lemma = _inf_candidate
            # COI bénéfactif : 'se' = bénéficiaire + COD explicite présent
            # Ex: "elle s'est acheté une voiture" → a bɛ [watiri] sàn a yɛrɛ yé
            # Détection structurelle (pas LLM) : expl:comp/iobj + obj dans T
            _cod_for_refl = next((x for x in T
                                  if x.get('dep') == 'obj'
                                  and x.get('pos') in ('NOUN', 'PROPN', 'PRON')
                                  and x['orig_index'] not in processed_indices), None)
            if _cod_for_refl and _refl_tok.get('dep') in ('expl:comp', 'iobj'):
                _refl_cat = 'coi_benefactif'
            else:
                _refl_cat = _classify_refl_fn(_refl_lemma)
            # Override KG : si KG a annoté un comportement réflexif spécifique, l'utiliser
            if _kg_refl and _kg_refl != _refl_cat:
                # KG: posture='posture', perception='actif', spontaneous='accidentel'
                _refl_cat = _kg_refl
            elif (root_tok.get('semantic_class') == 'posture'
                    and _refl_cat not in ('pronominal', 'posture')):
                # Fallback Python si KG n'a pas annotée cette posture
                _refl_cat = 'posture'
        # Override KG perception (regarder, voir…) → actif + yɛrɛ
        _kg_refl_perception = root_tok.get('_kg_reflexive', '')
        if _kg_refl_perception == 'actif' and _refl_cat in ('soin_corporel', 'posture'):
            _refl_cat = 'actif'
        elif (_refl_cat in ('soin_corporel', 'posture')
                and root_tok.get('semantic_class') == 'perception'):
            # Fallback Python si KG n'a pas de règle perception
            _refl_cat = 'actif'

    # Étape 2 : animation du sujet (complément de is_refl_passive)
    # Si le KG marque explicitement le nom-sujet comme inanimé → sens passif → SKIP
    _nsubj_animate = (
        not _nsubj_tok                                          # pas de sujet nsubj
        or _nsubj_tok.get('pos') in ('PRON', 'PROPN')          # pronoms/noms propres = animés
        or _nsubj_tok.get('is_animate', True) is not False     # flag KG (défaut=animé)
    )

    # Pronominal idiomatique (se tromper, se souvenir…) : verbe intransitif
    # au sens propre — au passé positif, doit produire V+ra (résultatif), pas V+li kɛ.
    # Le flag bypasse la nominalisation de step7 et active le résultatif F6.
    if (_refl_tok and root_tok and root_tok.get('pos') == 'VERB'
            and _refl_cat == 'pronominal'
            and _nsubj_animate
            and tree.get('clause_type') != 'reciprocal'):
        tree['_refl_pronominal_bypass'] = True
        tree['is_transitive'] = False

    if (_refl_tok and root_tok and root_tok.get('pos') == 'VERB'
            and not root_tok.get('is_statif')
            and not root_tok.get('is_refl_passive')   # Étape 1+2 (KG passif)
            and _refl_cat != 'pronominal'              # Étape 1 LLM (essentiellement pronominal)
            and _refl_cat != 'coi_benefactif'          # COI bénéfactif géré séparément ci-dessous
            and _nsubj_animate                         # Étape 2 (animacy heuristic)
            and tree.get('clause_type') != 'reciprocal'):   # Étape 3 déjà géré
        print(f"DEBUG [REFL_ABSOLUTE SET] _refl_tok={_refl_tok.get('surface')}, root_tok={root_tok.get('surface')}, refl_cat={_refl_cat}, is_statif={root_tok.get('is_statif')}, is_refl_passive={root_tok.get('is_refl_passive')}, clause_type={tree.get('clause_type')}")

        # Étape 4 : annotation Cat. 1 / Cat. 2 (même structure Bambara pour les deux)
        tree['refl_category'] = _refl_cat   # 'actif' ou 'accidentel' (LLM)

        tree['_is_refl_absolute'] = True  # signal → PatternRule KG
        tree['is_transitive'] = True   # évite le F6 résultatif (V+ra)
        # yɛrɛ uniquement si:
        #   1. Pronom emphatique explicite ('lui même', 'soi même', 'moi même') → True
        #   2. Idiomatique + marque locative xcomp ('se mettre à V') → True
        #   3. _refl_cat == 'accidentel' (≈ VerbeNet hurt-40.8.3 : se blesser, se
        #      couper, se brûler) → True, l'évènement atteint le sujet comme un
        #      patient distinct, contrairement à soin_corporel/posture/actif.
        #   4. Sinon → False (réflexifs inhérents : se laver, s'asseoir, etc.)
        _emph_meme = next((x for x in T
                           if x.get('role') == 'reflexive'
                           and str(x.get('surface', '')).lower() in (G_kg.get('refl_emphasis_surfaces') or {'même', 'meme', 'mêmes', 'memes', 'soi'})
                           and x['orig_index'] not in processed_indices), None)
        _xcomp_locative_mark = (
            xcomp_verb_tok and next((
                x for x in T
                if x.get('dep') == 'mark'
                and x.get('role') == 'locative'
                and x.get('head_index') == xcomp_verb_tok['orig_index']
            ), None)
        ) if root_tok.get('is_refl_Subjective') and xcomp_verb_tok else None
        if _emph_meme:
            tree['refl_yere'] = True
            processed_indices.add(_emph_meme['orig_index'])
            # Consommer aussi le pronom adjacent ('lui', 'soi', 'moi', 'toi', etc.)
            _emph_pron = next((x for x in T
                               if x.get('pos') == 'PRON'
                               and x['orig_index'] not in processed_indices
                               and abs(x['orig_index'] - _emph_meme['orig_index']) <= 1), None)
            if _emph_pron:
                processed_indices.add(_emph_pron['orig_index'])
        elif _xcomp_locative_mark or _locative_xcomp_mark:
            # _xcomp_locative_mark: locatif xcomp + is_refl_Subjective (se mettre à V)
            # _locative_xcomp_mark: locatif xcomp seul (fallback sans is_refl_Subjective)
            tree['refl_yere'] = True
            tree['refl_serial_postpos'] = G_kg.get('refl_serial_postpos', '')
        elif _refl_cat in ('accidentel', 'actif'):
            tree['refl_yere'] = True
        else:
            tree['refl_yere'] = False
        processed_indices.add(_refl_tok['orig_index'])
        # Capturer le verbe NU ici : step6 (participe résultatif du passé composé)
        # écraserait sinon m['V'] en V+ra et viderait le TAM.
        tree['refl_verb'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
        tree['refl_semantic_class'] = root_tok.get('semantic_class', '')
        # Pronom de reprise = pronom sujet si présent, sinon 'a' (3sg)
        _rsubj = next((x for x in T
                       if x.get('dep') in ('nsubj', 'nsubj:pass')), None)
        if _rsubj and _rsubj.get('pos') == 'PRON' and _rsubj.get('bm'):
            tree['refl_pron'] = _rsubj.get('bm')
        else:
            tree['refl_pron'] = 'a'

        # Resolve TAM for reflexive clauses (ne pas laisser TAM vide)
        tree['tense'] = root_tok.get('tense', 'pres')
        tree['neg'] = tree.get('neg', False) or root_tok.get('is_neg', False)
        tree['tam'] = _resolve_tam(tree['tense'], tree['neg'], G_kg)

        print(f"DEBUG [REFL_ABSOLUTE DONE] cat={tree.get('refl_category')} refl_verb={tree.get('refl_verb')}, refl_pron={tree.get('refl_pron')}, refl_yere={tree.get('refl_yere')}, tam={tree.get('tam')}")

    # ── COI BÉNÉFACTIF : se = bénéficiaire + COD ─────────────────────────────
    # "elle s'est acheté une voiture" → a bɛ [watiri] sàn a yɛrɛ yé
    # COD géré normalement par step4 (m['O']). On ajoute OBL "S yɛrɛ yé".
    elif (_refl_tok and root_tok and root_tok.get('pos') == 'VERB'
          and _refl_cat == 'coi_benefactif'
          and _nsubj_animate):
        tree['is_transitive'] = True          # bloque F6 résultatif
        processed_indices.add(_refl_tok['orig_index'])
        tree['tense'] = root_tok.get('tense', 'pres')
        tree['neg']   = tree.get('neg', False) or root_tok.get('is_neg', False)
        tree['tam']   = _resolve_tam(tree['tense'], tree['neg'], G_kg)
        # Pronom sujet pour l'OBL (m['S'] posé par step2 avant step3)
        _coi_subj_bm = m.get('S', '')
        if not _coi_subj_bm:
            _rsubj_c = next((x for x in T
                             if x.get('dep') in ('nsubj', 'nsubj:pass') and x.get('bm')), None)
            _coi_subj_bm = (_rsubj_c.get('bm', '') if _rsubj_c
                            else G_kg.get('pronoun_3sg_default', 'a'))
        _yere = G_kg.get('reflexive_self_marker', 'yɛrɛ') or 'yɛrɛ'
        _yé   = G_kg.get('comitative_end_marker', 'yé') or 'yé'
        m['OBL_ALL'].append({
            'HEAD': j(_coi_subj_bm, _yere),
            'MARKER': _yé,
            'local_clause_type': 'comitative',
            'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
            'DEP_TYPE': 'coi_benefactif', 'COMPOUND_IS_QUANTIFIER': False,
            'MARKER_IS_PREFIX': False,
        })

    # ── CLITIQUES PRONOMINAUX IOBJ SANS TRADUCTION ───────────────────────────
    # PRON dep='iobj' sans bm valide = clitique adverbial (en, y…) partie du
    # groupe verbal, sans équivalent bambara direct.
    # Ajouté à processed_indices pour empêcher step5 de créer un oblique parasite.
    _dative_mk = G_kg.get('dative_marker', 'ma') or 'ma'
    # Verbes de communication indirecte (dire, parler…) → datif 'yé' plutôt que 'ma'
    _root_sc_step3 = root_tok.get('semantic_class', '') if root_tok else ''
    if _root_sc_step3 in ('saying', 'communication'):
        _dative_mk = 'yé'
    # communication_transitive (appeler, voir…) → objet DIRECT en m['O'], pas de marqueur datif
    for _iobj_tok in T:
        # Also catch mislabeled clitics: dep='dep' + role='object' (e.g. "Dis lui")
        _is_dative_dep = (_iobj_tok.get('dep') == 'dep'
                          and _iobj_tok.get('role') == 'object')
        # Impératif + verbe NON-communication : clitique postverbal = COD, pas COI
        # (aide-moi → moi = COD géré par step4 ; dis-lui → lui = COI, chemin iobj normal)
        if (_is_dative_dep
                and tree.get('clause_type') == 'imperative'
                and root_tok
                and root_tok.get('semantic_class') not in ('communication', 'saying')):
            continue
        if ((_iobj_tok.get('dep') == 'iobj' or _is_dative_dep)
                and _iobj_tok.get('pos') == 'PRON'
                and _iobj_tok['orig_index'] not in processed_indices):
            _iobj_bm = str(_iobj_tok.get('bm', ''))
            # Réflexif COI : détecter si le clitique datif est coréférent au sujet
            _iobj_bm_test  = _iobj_tok.get('bm', '')
            _iobj_morph    = str(_iobj_tok.get('morph', ''))
            _iobj_pers     = next((p.split('=')[1] for p in _iobj_morph.split('|')
                                   if p.startswith('Person=')), '')
            _subj_iobj_tok = next((x for x in T
                                   if x.get('dep') in ('nsubj', 'nsubj:pass')), None)
            _subj_bm_dat   = _subj_iobj_tok.get('bm', '') if _subj_iobj_tok else ''
            _subj_morph_dat = str(_subj_iobj_tok.get('morph', '')) if _subj_iobj_tok else ''
            _subj_pers_dat  = next((p.split('=')[1] for p in _subj_morph_dat.split('|')
                                    if p.startswith('Person=')), '')
            _is_refl_iobj = (
                # Réflexif 1re/2e personne : bm identique garantit la coréférence
                # En 3e personne, a == a (elle/lui) NE SUFFIT PAS : "elle lui parle" ≠ réflexif
                (_iobj_bm_test and _subj_bm_dat and _iobj_bm_test == _subj_bm_dat
                 and _iobj_pers in ('1', '2'))
                or (_iobj_pers and _subj_pers_dat and _iobj_pers == _subj_pers_dat
                    and _iobj_pers in ('1', '2'))
                # 3e personne : 'se'/'s'' = TOUJOURS réflexif en français (≠ 'lui'/'leur')
                # → n'atteint cette boucle que via le datif réflexif communication (fix ci-dessus)
                or str(_iobj_tok.get('surface', '')).lower().lstrip('-') in ('se', "s'", 's')
            )
            if _is_refl_iobj:
                # Réflexif COI (tu te parles, je me parle) → réciprocité sur soi-même
                # → S yɛrɛ DATIVE_MK (i yɛrɛ yé, n yɛrɛ yé)
                # NB: ne s'applique PAS à me/se expl:comp (je me lève, je me réveille)
                _subj_bm_refl = _iobj_bm_test
                if _subj_bm_refl:
                    m['OBL_ALL'].append({
                        'HEAD': j(_subj_bm_refl, G_kg.get('reflexive_self_marker', ''), _dative_mk), 'MARKER': '',
                        'local_clause_type': 'simple', 'COMPOUND': '', 'MOD': '',
                        'DEM_PREF': '', 'DEM_SUFF': '', 'DEP_TYPE': 'iobj',
                        'COMPOUND_IS_QUANTIFIER': False, 'MARKER_IS_PREFIX': False,
                    })
                processed_indices.add(_iobj_tok['orig_index'])
                continue
            # Réflexif lexicalisé ("s'appeler") : main check n'a trouvé aucun _refl_tok
            # ET l'iobj est coréférent au sujet (même bm, sujet inclusif dep='dep') → supprimer datif
            # Guard : 'lui'/'leur' ne sont JAMAIS réflexifs en français (réflexif 3e pers = 'se').
            # Sans ce guard, bm(elle)=bm(lui)='a' → faux positif ("elle lui parle" supprimé).
            _REFL_SURFS = {'me', "m'", 'm', 'te', "t'", 't', 'se', "s'", 's', 'nous', 'vous'}
            _iobj_surf_lower = str(_iobj_tok.get('surface', '')).lower().lstrip('-')
            if (_refl_tok is None and _iobj_bm_test
                    and _iobj_surf_lower in _REFL_SURFS):
                _subj_ext = next((x for x in T
                                  if x.get('dep') in ('nsubj', 'nsubj:pass')
                                  or (x.get('dep') == 'dep' and x.get('role') == 'subject')), None)
                if _subj_ext and _subj_ext.get('bm') == _iobj_bm_test:
                    processed_indices.add(_iobj_tok['orig_index'])
                    continue
            if _iobj_bm and not _iobj_bm.startswith('['):
                # communication_transitive (appeler, voir) : objet DIRECT → m['O'] sans marqueur
                if _root_sc_step3 == 'communication_transitive' and not m.get('O'):
                    m['O'] = _iobj_bm
                    processed_indices.add(_iobj_tok['orig_index'])
                else:
                    # Clitique datif pronominal (te→i, me→n, lui→a) → "bm ma/yé"
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

    # Clitique accusatif mislabeled (dep='dep' + role='object_pronoun', e.g. "prends le")
    for _acc_tok in T:
        if (_acc_tok.get('dep') == 'dep'
                and _acc_tok.get('pos') == 'PRON'
                and _acc_tok.get('role') == 'object_pronoun'
                and _acc_tok.get('bm')
                and not str(_acc_tok.get('bm', '')).startswith('[')
                and _acc_tok['orig_index'] not in processed_indices
                and not m.get('O')):
            m['O'] = _acc_tok.get('bm')
            processed_indices.add(_acc_tok['orig_index'])

    # ── INFINITIF ─────────────────────────────────────────────────────────────
    _is_infinitive = (root_tok and root_tok.get('pos') == 'VERB'
                      and 'Inf' in str(root_tok.get('morph', ''))
                      and not m.get('S'))
    if _is_infinitive:
        pass  # body removed → PatternRule KG

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
        if tree.get('clause_type') != 'noun_phrase_have':
            pass  # body removed → PatternRule KG
        processed_indices.add(_cond_mark['orig_index'])

    # ── TEMPOREL : quand / lorsque → marqueur postposé (ex: tuma min) ───────
    _tmp_mark = next((x for x in T
                      if x.get('dep') == 'mark'
                      and x.get('role') == 'temporal'
                      and x.get('bm')
                      and x['orig_index'] not in processed_indices), None)
    if _tmp_mark and tree.get('clause_type') not in ('conditional', 'infinitive'):
        # Guard : si le marqueur temporal appartient à un advcl (pas à la clause
        # principale), il est géré par advcl.handle dans step5 → ne pas le poser ici.
        # Ex: "Malheur quand il cligne" → quand est mark de cligne(advcl de malheur).
        _mark_head = next((x for x in T if x['orig_index'] == _tmp_mark.get('head_index')), None)
        _mark_is_advcl = (_mark_head
                          and _mark_head.get('dep') == 'advcl'
                          and (_mark_head.get('head_index') != _mark_head.get('orig_index')))
        if not _mark_is_advcl:
            tree['temporal_marker'] = _tmp_mark.get('bm')
            processed_indices.add(_tmp_mark['orig_index'])

    # ── F1 : TENSE TRANSFER ───────────────────────────────────────────────────
    aux_tense_tok = next((x for x in T
                          if x.get('dep') in ('aux', 'aux:tense', 'aux:pass', 'cop')
                          and x.get('tense') in ('past', 'hab', 'plup', 'imp')), None)

    if not aux_tense_tok and root_tok and root_tok.get('tense') == 'past':
        _aux_pass = next((x for x in T if x.get('dep') == 'aux:pass'), None)
        if _aux_pass and not root_tok.get('is_statif'):
            aux_tense_tok = root_tok

    if aux_tense_tok and root_tok:
        _tn = aux_tense_tok.get('tense', 'pres')
        # Plus-que-parfait : avoir/être imparfait (hab) + participe passé → plup
        # (tùn yé / tùn ma), distinct du passé simple (yé / ma).
        if (_tn == 'hab'
                and root_tok.get('tense') == 'past'
                and 'VerbForm=Part' in str(root_tok.get('morph', ''))):
            _tn = 'plup'
        tree['tense'] = _tn
        tree['neg']   = tree['neg'] or aux_tense_tok.get('is_neg', False)
        tree['tam']   = _resolve_tam(tree['tense'], tree['neg'], G_kg)
        # Imparfait progressif : "j'étais en train de V" → aux porte hab, root
        # porte prog (posé par _detect_progressive). tree['tense'] reste 'hab'
        # (pour les listes past/hab/plup ailleurs) ; seul le TAM gagne 'kà'.
        if _tn == 'hab' and root_tok.get('tense') == 'prog':
            tree['tam'] = j(tree['tam'], 'kà')
    elif (root_tok and root_tok.get('tense') in ('past', 'fut', 'cond', 'hab', 'prog')
          and not root_tok.get('is_statif')):
        tree['tense'] = root_tok['tense']
        tree['neg']   = tree['neg'] or root_tok.get('is_neg', False)
        tree['tam']   = _resolve_tam(tree['tense'], tree['neg'], G_kg)
        # Présent progressif sans aux_tense_tok : "en train de V" → TAM_pres + TAM_prog
        # (l'imparfait progressif "était en train de" est géré par j(tree['tam'], 'kà')
        # dans le bloc aux_tense_tok ci-dessus via _resolve_tam('hab'…))
        if root_tok.get('tense') == 'prog':
            # Si le KG fournit déjà le TAM progressif complet ('bɛ kà'), le garder.
            # Sinon, construire depuis TAM présent + particule infinitive.
            if not tree.get('tam'):
                _pres_tam  = _resolve_tam('pres', tree['neg'], G_kg)
                _prog_part = G_kg.get('infinitive_marker', '')
                if _pres_tam:
                    tree['tam'] = j(_pres_tam, _prog_part)
                elif _prog_part:
                    tree['tam'] = _prog_part
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
        elif (tree.get('clause_type') not in ('optative', 'content_question',
                                               'imperative', 'prohibitive',
                                               'quest_ce_que_modal', 'quest_ce_que')
              and not tree.get('_imp_tam_done')):
            # 'optative' exclu : tam='ka' déjà posé plus haut
            # 'content_question' exclu : tam déjà posé depuis le tense du verbe
            # 'imperative'/'prohibitive' exclus : verbe nu, pas de TAM présent.
            # _imp_tam_done exclu : impératif détecté structurellement (pas par PatternRule).
            # _resolve_tam en premier : prend en compte tree['neg'] (négatif → tɛ)
            # tam_default en fallback seulement si _resolve_tam retourne ''.
            tree['tam'] = (_resolve_tam('pres', tree['neg'], G_kg)
                           or G_kg.get('tam_default', ''))

    # Override TAM pour content_question quand le verbe acl:relcl n'est pas au présent
    # (ex: pourrait→fut → bɛ na, devait→hab → bɛ). Le bloc else ligne ~1225
    # utilisait le tense du ROOT PRON (toujours pres), écrasant le tense réel.
    if tree.get('_cq_acl_tense') and tree.get('clause_type') == 'content_question':
        tree['tense'] = tree['_cq_acl_tense']
        tree['tam']   = _resolve_tam(tree['_cq_acl_tense'], tree.get('neg', False), G_kg) or 'bɛ'

    # OBLIGATION : devoir/falloir → TAM spécial + V='kan'
    # Positif : S kan ka V_nom kɛ  |  Négatif : S man kan ka V_nom kɛ
    if (root_tok and root_tok.get('semantic_class') == 'obligation'
            and tree.get('clause_type') not in ('prohibitive', 'imperative')):
        root_tok['bm'] = 'kan'
        m['V'] = 'kan'
        tree['tam'] = 'man' if tree.get('neg', False) else ''
        tree['_obligation'] = True   # bloque le ré-écrasement TAM dans step6

    # NB: l'application des suffixes verbaux (li kɛ / la / kɛ) est centralisée
    # dans step6_copule ("Capturer root non encore traité comme V"). On ne fait
    # rien ici pour éviter le double-suffixage.

    # ── COORDINATION VERBALE : dep=conj VERB → clause2 jointe par 'wa' ──────
    # "S V1 et V2 O" → clause1 = S TAM V1 (V_ACT), clause2 = S TAM O V2
    # Stocké dans tree['conj_clauses'] ; consommé par le renderer (renderers/__init__.py).
    # En verb_serial (motion + xcomp), inclure aussi les conj de l'xcomp :
    # "elle alla trouver et demander X" → demander est conj de trouver (pas de root)
    _xcomp_v_idx = next((x.get('orig_index') for x in T
                         if x.get('dep') == 'xcomp' and x.get('pos') == 'VERB'
                         and x.get('head_index') == root_tok.get('orig_index')), None) if root_tok else None
    _cv_head_indices = {root_tok.get('orig_index')} if root_tok else set()
    if (tree.get('clause_type') in ('verb_serial', 'serial_motion_pres', 'serial_futur_proche')
            and _xcomp_v_idx is not None):
        _cv_head_indices.add(_xcomp_v_idx)
    _cv_toks = [x for x in T
                if x.get('dep') == 'conj' and x.get('pos') == 'VERB'
                and x['orig_index'] not in processed_indices
                and x.get('head_index') in _cv_head_indices]
    if _cv_toks:
        _conj_clauses = []
        for _cv in _cv_toks:
            _cv_bm  = _cv.get('bm') or f"[{_cv.get('lemma', '')}]"
            _cv_obj = next((x for x in T
                            if x.get('dep') == 'obj'
                            and x.get('head_index') == _cv['orig_index']
                            and x['orig_index'] not in processed_indices), None)
            _cv_obj_bm = ((_cv_obj.get('bm') or f"[{_cv_obj.get('lemma', '')}]")
                          if _cv_obj else '')
            _cv_tense = _cv.get('tense', tree.get('tense', 'pres'))
            _cv_tam   = _resolve_tam(_cv_tense, tree.get('neg', False), G_kg) or 'bɛ'
            _cc_tok   = next((x for x in T if x.get('dep') == 'cc'
                              and x.get('head_index') == _cv['orig_index']), None)
            _cc_role  = _cc_tok.get('role', '') if _cc_tok else ''
            _conj_clauses.append({
                'V':                _cv_bm,
                'O':                _cv_obj_bm,
                'tam':              _cv_tam,
                'intransitive_type': _cv.get('intransitive_type', ''),
                'semantic_class':   _cv.get('semantic_class', ''),
                'is_liquid':        _cv.get('is_liquid', False),
                'cc_role':          _cc_role,
                'cc_bm':            (_cc_tok.get('bm', '') if _cc_tok else ''),
            })
            processed_indices.add(_cv['orig_index'])
            if _cv_obj:
                processed_indices.add(_cv_obj['orig_index'])
            if _cc_tok:
                processed_indices.add(_cc_tok['orig_index'])
        if _conj_clauses:
            tree['conj_clauses'] = _conj_clauses

    # ── DÉJÀ ──────────────────────────────────────────────────────────────────
    _deja_tok = next((x for x in T if x.get('role') == 'temporal_already'), None)
    if _deja_tok:
        _deja_modifies_state = (root_tok and root_tok.get('pos') in ('NOUN', 'ADJ'))
        if tree.get('neg'):
            tree['already_marker'] = _deja_tok.get('bm_neg') or G_kg.get('already_neg_marker', '')
        elif tree.get('tense') == 'past':
            tree['already_marker'] = _deja_tok.get('bm') or G_kg.get('already_past_marker', '')
        elif _deja_modifies_state:
            tree['already_marker'] = G_kg.get('already_past_marker', '')
        else:
            tree['already_marker'] = G_kg.get('already_pres_marker', '')
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
        m['ADV'] = j(s_bm, part_to_tok.get('bm', '') + G_kg.get('participial_to_suffix', ''))
        processed_indices.add(part_to_tok['orig_index'])

    # ── PRÉDICAT PRIVATIF NOUN ROOT ───────────────────────────────────────────
    _priv_case = next((x for x in T
                       if x.get('dep') == 'case' and x.get('role') == 'privative'
                       and root_tok and x.get('head_index') == root_tok['orig_index']
                       and root_tok.get('pos') == 'NOUN'), None)
    if _priv_case:
        _priv_marker = _priv_case.get('bm_marker') or G_kg.get('privative_noun_pred', '')
        # Morphologie bambara : nom verbal (bm se termine en '-li') → 'bali'.
        # Ex: tóbili (cuisson) → tóbilibali ; kɔ̀gɔ (sel) → kɔ̀gɔtan.
        _bm_rt = root_tok.get('bm', '')
        if _bm_rt and _bm_rt.rstrip('.').endswith('li'):
            _priv_marker = G_kg.get('privative_verb_marker_suffix', 'bali') or 'bali'
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
        _priv_marker = (G_kg.get('privative_pron_marker', 'kɔ')
                        if root_tok.get('pos') == 'PRON'
                        else _priv_on_root_pron.get('bm_marker') or G_kg.get('privative_noun_pred', ''))
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
                    _amod_bm = adj_man(_amod_bm, is_classifying=_amod_on_root.get('is_classifying_adj', False))
                _root_bm = j(_root_bm, _amod_bm)
                processed_indices.add(_amod_on_root['orig_index'])
        m['O'] = _root_bm
        processed_indices.add(root_tok['orig_index'])

    elif _root_has_loc_case and root_tok and root_tok['orig_index'] not in processed_indices:
        m['S'] = next((x.get('bm', '') for x in T
                       if x.get('dep') in ('nsubj', 'nsubj:pass') and x.get('bm')),
                      m.get('S', ''))
        _loc_case_root = next((x for x in T
                               if x.get('dep') == 'case' and x.get('role') in _loc_case_roles
                               and x.get('head_index') == root_tok['orig_index']), None)
        _loc_marker = _loc_case_root.get('bm_marker') or G_kg.get('locative_suffix', '') if _loc_case_root else G_kg.get('locative_suffix', '')
        if not _loc_marker:
            _loc_marker = G_kg.get('locative_suffix', '')
        # Construire la chaîne génitive pour le HEAD si le nom locatif a un nmod
        # (sous le poids de la neige → nɛzi gírinya kɔrɔ : possesseur en tête).
        # Sinon le nmod resterait orphelin et deviendrait un oblique séparé.
        _has_nmod_on_loc = any(x.get('dep') == 'nmod'
                               and x.get('head_index') == root_tok['orig_index']
                               for x in T)
        if _has_nmod_on_loc:
            from rules.steps.step4_objet.objet_standard import _build_genitive_chain
            from rules.steps.step5_obliques.comitative import _collect_genitive_tokens
            _loc_head = _build_genitive_chain(root_tok, T, G_kg)
            processed_indices.update(_collect_genitive_tokens(root_tok, T))
        else:
            _loc_head = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
        m['OBL_ALL'].append({
            'HEAD': _loc_head,
            'MARKER': _loc_marker, 'local_clause_type': 'locative',
            'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
            'DEP_TYPE': 'case', 'COMPOUND_IS_QUANTIFIER': False,
            'MARKER_IS_PREFIX': False,
        })
        processed_indices.add(root_tok['orig_index'])
        if _loc_case_root:
            processed_indices.add(_loc_case_root['orig_index'])

    # ── VERBE MÉTÉOROLOGIQUE (impersonnel) ───────────────────────────────────
    # Signal : semantic_class='meteorological' (LLM fiable pour pleuvoir, neiger…)
    # Phénomène : bm du verbe (san pour pleuvoir, etc.) ou placeholder
    # Verbe bambara : 'na' (venir) par défaut
    # NE TOUCHE À AUCUNE AUTRE STRUCTURE.
    if (root_tok
            and root_tok.get('semantic_class') == 'meteorological'
            and tree.get('clause_type') not in (
                'existential_absolute', 'existential_localized',
                'interrogative', 'content_question')):
        _meteo_bm = (root_tok.get('bm')
                     or f"[{root_tok.get('lemma', '')}]")
        tree['meteo_bm']     = _meteo_bm
        tree['meteo_motion'] = G_kg.get('meteo_motion_default', '') or tree.get('meteo_motion', '')
        m['S'] = ''; m['V'] = ''; m['O'] = ''
        processed_indices.add(root_tok['orig_index'])
        for _il in T:
            if (str(_il.get('surface', '')).lower() in ('il', 'ce', 'ça')
                    and _il.get('dep') in ('nsubj', 'nsubj:pass', 'expl:subj', 'expl')):
                processed_indices.add(_il['orig_index'])

    return aux_tense_tok