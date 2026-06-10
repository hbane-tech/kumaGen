"""
rules/renderers/interrogative.py
Clause types : interrogative, content_question, noun_phrase_have
"""
from rules.core import j, INTRANS_SC


def render_interrogative(tree, m, S, O, V, TAM, obl_strings, G):
    _tokens = tree.get('_tokens', [])

    # Est-ce que X ? → Yala X TAM O V [obliques] wà ?
    if tree.get('est_ce_que'):
        _yala = G.get('question_marker_yala', 'Yala') or 'Yala'
        # Rebuild subject from tokens to capture possessive + coordinated NPs
        _subj_tok_i = next((t for t in _tokens
                            if t.get('dep') == 'nsubj'
                            and t.get('pos') in ('NOUN', 'PROPN')
                            and t.get('bm')), None)
        _conj_tok_i = next((t for t in _tokens
                            if t.get('dep') == 'conj'
                            and t.get('pos') in ('NOUN', 'PROPN')
                            and t.get('bm')), None)
        if _subj_tok_i:
            _poss_s_i = next((t for t in _tokens
                              if t.get('dep') == 'det'
                              and t.get('role') in ('pronoun', 'possessive')
                              and t.get('head_index') == _subj_tok_i['orig_index']
                              and t.get('bm')), None)
            _s_bm_i = _subj_tok_i.get('bm', '')
            if _poss_s_i:
                _s_bm_i = j(_poss_s_i.get('bm'), _s_bm_i)
            if _conj_tok_i:
                _poss_c_i = next((t for t in _tokens
                                  if t.get('dep') == 'det'
                                  and t.get('role') in ('pronoun', 'possessive')
                                  and t.get('head_index') == _conj_tok_i['orig_index']
                                  and t.get('bm')), None)
                _c_bm_i = _conj_tok_i.get('bm', '')
                if _conj_tok_i.get('is_plural') and not _c_bm_i.endswith('w'):
                    _c_bm_i += 'w'
                if _poss_c_i:
                    _c_bm_i = j(_poss_c_i.get('bm'), _c_bm_i)
                elif _poss_s_i:
                    _c_bm_i = j(_poss_s_i.get('bm'), _c_bm_i)
                if _subj_tok_i.get('is_plural') and not _s_bm_i.endswith('w'):
                    _s_bm_i += 'w'
                _real_s_i = j(_s_bm_i, 'ni', _c_bm_i)
            else:
                if _subj_tok_i.get('is_plural') and not _s_bm_i.endswith('w'):
                    _s_bm_i += 'w'
                _real_s_i = _s_bm_i
            S = _real_s_i
        if obl_strings and not O and not V:
            return j(_yala, S, TAM, *obl_strings, 'wà ?')
        return j(_yala, S, TAM, O, V, *obl_strings, 'wà ?')

    _question_marker = next((t.get('bm', '') for t in _tokens
                             if t.get('role') == 'question_marker' and t.get('bm')), '')
    _alt_tok = next((t for t in _tokens if t.get('role') == 'alternative' and t.get('bm')), None)
    _interrog_end = next((t.get('bm', '') for t in _tokens
                          if t.get('role') == 'interrogative_end' and t.get('bm')), 'wà ?')
    _clean_obls = [_xv for _ci, _xv in enumerate(obl_strings)
                   if _ci < len(m.get('OBL_ALL', []))
                   and m['OBL_ALL'][_ci].get('DEP_TYPE') != 'fixed'
                   and not any(t.get('dep') == 'fixed'
                               and t.get('bm') == m['OBL_ALL'][_ci].get('HEAD')
                               for t in _tokens)]
    _v_display = V if (V and V.rstrip('́') != TAM.rstrip('́')) else ''
    # Verbe sériel / xcomp (trouveras-tu le temps DE VENIR) → ka V_ACT.
    # Vide si pas d'xcomp → no-op pour les interrogatives simples.
    _v_act = m.get('V_ACTION', '') or ''
    _vact_part = j('ka', _v_act) if _v_act else ''

    if _alt_tok:
        _conj_v = next((t for t in _tokens
                        if t.get('dep') == 'conj' and t.get('pos') == 'VERB' and t.get('bm')), None)
        _conj_o = next((t for t in _tokens
                        if t.get('dep') == 'obj' and _conj_v
                        and t.get('head_index') == _conj_v.get('orig_index')
                        and t.get('bm')), None)
        _conj_o_amod = next((t for t in _tokens
                             if t.get('dep') == 'amod' and _conj_o
                             and t.get('head_index') == _conj_o.get('orig_index')
                             and t.get('bm')), None)
        _conj_o_bm = (j(_conj_o.get('bm', ''),
                        _conj_o_amod.get('bm', '') if _conj_o_amod else '')
                      if _conj_o else '')
        _conj_v_bm = _conj_v.get('bm', '') if _conj_v else ''
        return j(S, TAM, O, _v_display, *_clean_obls,
                 _alt_tok.get('bm', ''), S, TAM, _conj_o_bm, _conj_v_bm, '?')

    _v_is_motion = next((t for t in _tokens
                         if t.get('is_root') and t.get('semantic_class') == 'motion'), None)
    if _v_is_motion and _v_display:
        _loc_marker = next((t.get('bm_marker', '') for t in _tokens
                            if t.get('role') == 'locative'
                            and t.get('dep') == 'case'
                            and t.get('bm_marker')), '')
        _o_loc = j(O, _loc_marker) if _loc_marker else O
        return j(_question_marker, S, TAM, _v_display, _o_loc, _vact_part,
                 *_clean_obls, _interrog_end)

    # Action verb sans objet → V+li kɛ
    _root_ref = next((t for t in _tokens if t.get('is_root')), None)
    _is_action = (_v_display and not O and _root_ref
                  and _root_ref.get('intransitive_type') in ('ACTION', 'nominalized', 'support')
                  and _root_ref.get('semantic_class', '') not in INTRANS_SC
                  and not _root_ref.get('is_refl_passive'))
    if _is_action:
        return j(_question_marker, S, TAM, _v_display + 'li', 'kɛ', _vact_part,
                 *_clean_obls, _interrog_end)
    return j(_question_marker, S, TAM, O, _v_display, _vact_part,
             *_clean_obls, _interrog_end)


def render_content_question(tree, m, S, O, V, TAM, obl_strings, G):
    _tokens = tree.get('_tokens', [])

    # Nom seul + det interrogatif (Quelle femme ?)
    _interrog_det = next((t for t in _tokens
                          if t.get('role') == 'interrogative'
                          and t.get('dep') == 'det'
                          and t.get('bm')), None)
    if _interrog_det and S and not O and not V:
        return j(S, _interrog_det.get('bm', 'jùmɛn'), '?')

    # Cas : det interrogatif sur l'objet (Tu veux quelle maison ?)
    # → S bɛ O jùmɛn V ?
    _interrog_det_on_o = next((t for t in _tokens
                               if t.get('role') == 'interrogative'
                               and t.get('dep') == 'det'
                               and t.get('bm')), None)
    if _interrog_det_on_o and O and V:
        return j(S, TAM, O, _interrog_det_on_o.get('bm', ''), V, '?')
    
    # Détecter si S est un pronom interrogatif
    # Un pronom interrogatif est soit role='interrogative', soit role='relative'
    # Les pronoms personnels (je/tu/il/ils) ont role='pronoun' mais bm personnel
    # On exclut les pronoms qui sont aussi le sujet grammatical réel
    _s_is_interrog = any(
        t.get('bm') == S
        and t.get('role') in ('interrogative', 'relative')
        for t in _tokens)

    if _s_is_interrog:
        # Vérifier si l'interrogatif EST le sujet grammatical (ex: "Qui fait X ?")
        _interrog_tok_ref = next((t for t in _tokens
                                  if t.get('bm') == S
                                  and t.get('role') in ('interrogative', 'relative')), None)
        _interrog_is_subj = bool(_interrog_tok_ref
                                 and _interrog_tok_ref.get('dep') in ('nsubj', 'nsubj:pass'))
        if _interrog_is_subj:
            # Question-sujet : "Qui fait la bagarre ?" → S='jɔn', O=objet déjà correct
            pass
        else:
            # Question-objet : "Qu'est-ce que Marie aime ?" → déplacer interrog en O
            if not O:
                O = S
            _real_subj = next((t for t in _tokens
                               if t.get('pos') == 'PRON'
                               and t.get('bm')
                               and t.get('bm') != O
                               and t.get('role') not in {'interrogative', 'expletive',
                                                          'clitic', 'relative'}
                               and 'Int' not in str(t.get('morph', ''))), None)
            if _real_subj:
                S = _real_subj.get('bm', '')
            else:
                _root_ref = next((t for t in _tokens if t.get('is_root')), None)
                _morph = str(_root_ref.get('morph', '')) if _root_ref else ''
                _person = next((p.split('=')[1] for p in _morph.split('|')
                                if p.startswith('Person=')), '')
                _number = next((p.split('=')[1] for p in _morph.split('|')
                                if p.startswith('Number=')), '')
                _match = next((t for t in _tokens
                               if t.get('pos') == 'PRON'
                               and t.get('bm')
                               and t.get('bm') != O
                               and f'Person={_person}' in str(t.get('morph', ''))
                               and f'Number={_number}' in str(t.get('morph', ''))), None)
                S = _match.get('bm', '') if _match else ''
    else:
            # S est déjà le sujet (ex: tu→i) — récupérer O depuis tokens interrogatifs
            if not O:
                _interrog_tok = next((t for t in _tokens
                                      if t.get('bm')
                                      and t.get('role') in ('interrogative', 'relative', 'pronoun')
                                      and t.get('dep') in ('nsubj', 'obj', 'dep')
                                      and t.get('bm') != S), None)
                if _interrog_tok:
                    O = _interrog_tok.get('bm')

    _interrog_adv = next((t for t in _tokens
                          if t.get('role') == 'interrogative'
                          and t.get('dep') in ('advmod', 'dep')
                          and t.get('bm')), None)
    _interrog_adv_bm  = _interrog_adv.get('bm', '') if _interrog_adv else ''
    _interrog_noun    = tree.get('interrog_noun')
    _interrog_noun_bm = _interrog_noun.get('bm', '') if _interrog_noun else ''
    if _interrog_noun_bm and O:
        O = j(_interrog_noun_bm, O)
    elif _interrog_noun_bm:
        O = _interrog_noun_bm

    if TAM == 'yé' and not _interrog_noun_bm:
        return j(S, TAM, V or O, 'yé', _interrog_adv_bm, *obl_strings)
    if TAM == 'yé' and _interrog_noun_bm:
        return j(S, TAM, O, V, '?')
    # Question adverbiale (quand/où/pourquoi/comment) : déterminer tôt pour éviter
    # la nominalization incorrecte des verbes intransitifs
    _has_adv_question = any(
        t.get('dep') in ('advmod', 'dep')
        and t.get('role') in ('interrogative', 'temporal', 'locative')
        for t in _tokens)

    # Appliquer VERBli kɛ si verbe transitif sans COD réel
    # (N'applique PAS si c'est une question adverbiale : le verbe intransitif ne se nominalise pas)
    _root_ref_cq = next((t for t in _tokens if t.get('is_root')), None)
    _needs_li_cq = (
        V and _root_ref_cq
        and not _has_adv_question
        and _root_ref_cq.get('intransitive_type') in ('ACTION', 'nominalized', 'support')
        and _root_ref_cq.get('semantic_class', '') not in INTRANS_SC
        and not V.endswith('li')
        and not V.endswith('kɛ')
        and not _root_ref_cq.get('is_refl_passive'))

    _adv_in_O = _interrog_adv_bm and _interrog_adv_bm in O
    if _adv_in_O:
        _v_final = (V + 'li kɛ') if _needs_li_cq else V
        return j(S, TAM, _v_final, O, *obl_strings, '?')

    # le mot interrogatif est un adverbe (role interrogative/temporal/locative en
    # advmod), pas un participant. 'jɔn' n'est le défaut que pour les questions
    # portant sur un participant (qui/quoi) sans mot capturé.
    _o_final = O or ('' if _has_adv_question else G.get('interrogative_who', 'jɔn'))
    _v_final = (V + 'li kɛ') if (_needs_li_cq and not _o_final) else V
    return j(S, TAM, _o_final, _v_final, _interrog_adv_bm, *obl_strings, '?')


def render_noun_phrase_have(tree, S, O, neg, obl_strings, G):
    # Cas AGE interrogatif : i bɛ saan jóli la ?
    _have_type = tree.get('have_type', '') or tree.get('possession_type', '')
    if _have_type == 'AGE':
        _qty_interrog = next((t for t in tree.get('_tokens', [])
                              if t.get('role') == 'interrogative'
                              and t.get('bm')
                              and t.get('dep') in ('det', 'advmod', 'dep', 'obj')), None)
        _age_bm = G.get('age_marker', 'saan') or 'saan'
        # Adverbe temporel (demain, hier...)
        _adv_tok = next((t for t in tree.get('_tokens', [])
                         if t.get('dep') == 'advmod'
                         and t.get('pos') == 'ADV'
                         and t.get('bm')
                         and t.get('role') not in ('interrogative', 'negation')), None)
        _adv_bm = _adv_tok.get('bm', '') if _adv_tok else ''
        if _qty_interrog:
            _tam = tree.get('tam', 'bɛ') or 'bɛ'
            if _tam not in ('bɛ', 'tɛ'):
                return j(S, _tam, _age_bm, _qty_interrog.get('bm'), 'sɔrɔ', _adv_bm, '?')
            return j(S, 'bɛ', _age_bm, _qty_interrog.get('bm'), 'la', _adv_bm, '?')
        return j(O, 'bɛ', S, G.get('possession_abstract_marker', 'fɛ'))
    
    _poss_type = tree.get('possession_type', '')
    if not _poss_type:
        _obj_tok = next((t for t in tree.get('_tokens', [])
                         if t.get('bm') == O and t.get('pos') in ('NOUN', 'PROPN')), None)
        if _obj_tok:
            _sc = _obj_tok.get('semantic_class', '')
            _poss_type = 'material' if _sc in ('object', 'money', 'vehicle', 'tool') else 'abstract'
    _poss_marker = (G.get('possession_material_marker', 'bóló')
                    if _poss_type == 'material'
                    else G.get('possession_abstract_marker', 'fɛ'))
    _exist_op = 'tɛ' if neg else 'bɛ'
    _qty_interrog = next((t for t in tree.get('_tokens', [])
                          if t.get('role') == 'interrogative'
                          and t.get('bm')
                          and t.get('dep') in ('det', 'advmod', 'dep', 'obj')), None)
    # if _qty_interrog and _qty_interrog.get('bm') not in (O or ''):
    #     return j(O, _qty_interrog.get('bm'), _exist_op, S, _poss_marker, '?')
    _qty_interrog_bm = tree.get('interrog_qty', '')
    if not _qty_interrog_bm:
        _qty_interrog = next((t for t in tree.get('_tokens', [])
                              if t.get('role') == 'interrogative'
                              and t.get('bm')
                              and t.get('dep') in ('det', 'advmod', 'dep', 'obj')), None)
        _qty_interrog_bm = _qty_interrog.get('bm', '') if _qty_interrog else ''
    if _qty_interrog_bm and _qty_interrog_bm not in (O or ''):
        return j(O, _qty_interrog_bm, _exist_op, S, _poss_marker, '?')
    
    _polar_end = 'wà ?' if tree.get('_has_question_mark') else ''
    return j(O, _exist_op, S, _poss_marker, *obl_strings, _polar_end)
