"""
rules/renderers/copula.py
Clause types : statif, statif_past, equative, identificatory,
               presentative, qualitative, locative, existential
"""
from rules.core import j, _resolve_tam


def render_statif(tree, m, S, TAM, neg, obl_strings):
    ct = tree['clause_type']
    if ct == 'statif_past':
        _assert = 'tɛ' if neg else 'dòn'
        return j(S, 'tùn', m.get('QUAL', ''), _assert, *obl_strings)
    return j(S, m.get('QUAL', ''), TAM, *obl_strings)


def render_equative(tree, m, S, O, V, TAM, neg, tn, obl_strings, _ccomp_str, G,
                    _tokens_ref):
    # Est-ce que X est [loc] ? → Yala X bɛ [loc] wà ?
    if tree.get('est_ce_que') and obl_strings:
        _yala = G.get('question_marker_yala', 'Yala') or 'Yala'
        _subj_tok = next((t for t in _tokens_ref
                          if t.get('dep') == 'nsubj'
                          and t.get('pos') in ('NOUN', 'PROPN')
                          and t.get('bm')), None)
        _conj_tok = next((t for t in _tokens_ref
                          if t.get('dep') == 'conj'
                          and t.get('pos') in ('NOUN', 'PROPN')
                          and t.get('bm')), None)
        if _subj_tok:
            _poss_s = next((t for t in _tokens_ref
                            if t.get('dep') == 'det'
                            and t.get('role') in ('pronoun', 'possessive')
                            and t.get('head_index') == _subj_tok['orig_index']
                            and t.get('bm')), None)
            _s_bm = _subj_tok.get('bm', '')
            if _poss_s:
                _s_bm = j(_poss_s.get('bm'), _s_bm)
            if _conj_tok:
                _poss_c = next((t for t in _tokens_ref
                               if t.get('dep') == 'det'
                               and t.get('role') in ('pronoun', 'possessive')
                               and t.get('head_index') == _conj_tok['orig_index']
                               and t.get('bm')), None)
                _c_bm = _conj_tok.get('bm', '')
                if _conj_tok.get('is_plural') and not _c_bm.endswith('w'):
                    _c_bm += 'w'
                if _poss_c:
                    _c_bm = j(_poss_c.get('bm'), _c_bm)
                if _subj_tok.get('is_plural') and not _s_bm.endswith('w'):
                    _s_bm += 'w'
                _real_s = j(_s_bm, 'ni', _c_bm)
            else:
                _real_s = _s_bm
        else:
            _real_s = O or S
        return j(_yala, _real_s, 'bɛ', *obl_strings, 'wà ?')

    # Cas rien
    if not O and not V:
        _rien = next((t for t in tree.get('_tokens', [])
                      if t.get('role') == 'negation'
                      and t.get('dep') == 'ROOT'
                      and t.get('bm')), None)
        if _rien:
            O = _rien.get('bm', '')

    # Détecter cop futur
    if not tree.get('cop_tense'):
        _cop_fut = next((t for t in _tokens_ref
                         if t.get('dep') == 'cop'
                         and (t.get('tense') == 'fut'
                              or 'Tense=Fut' in str(t.get('morph', '')))), None)
        if _cop_fut:
            tree['cop_tense'] = 'fut'
            tree['cop_bm']    = _cop_fut.get('bm', '')
            tree['cop_lemma'] = _cop_fut.get('lemma', 'être')

    if not O and not V:
        _conj_attr = next((t for t in _tokens_ref
                           if t.get('dep') in ('conj', 'attr', 'appos')
                           and t.get('pos') in ('NOUN', 'PROPN', 'ADJ')
                           and t.get('bm')), None)
        if _conj_attr:
            O = _conj_attr.get('bm', '')

    if tree.get('cop_tense') == 'fut':
        _cop_bm = tree.get('cop_bm') or f"[{tree.get('cop_lemma', 'être')}]"
        TAM = 'tɛ na' if neg else 'bɛ na'
        O = j(_cop_bm, O) if O else _cop_bm

    if neg and tree.get('tense') in ('past', 'hab', 'plup'):
        _tun_base = TAM.split()[0] if TAM and ' ' in TAM else 'tùn'
        return j(S, _tun_base, 'tɛ', O or V, 'yé', *obl_strings)
    elif neg:
        return j(S, TAM, O or V, 'yé', *obl_strings)
    elif O and str(O).endswith('yé'):
        return j(S, tree.get('tam', 'yé'), O, *obl_strings)
    elif V and not O and tn == 'past':
        _already = tree.get('already_marker', '')
        _v_past = V
        if not _v_past.endswith('ra') and not _v_past.endswith('na'):
            _v_past = _v_past + 'na' if _v_past.endswith('n') else _v_past + 'ra'
        return j(S, _v_past, _already, *obl_strings)
    else:
        _why_tok = next((t for t in tree.get('_tokens', [])
                         if t.get('role') == 'interrogative'
                         and t.get('dep') == 'advmod'
                         and t.get('bm')), None)
        if _why_tok:
            _adj_bm = V or next((t.get('bm', '') for t in tree.get('_tokens', [])
                                 if t.get('pos') == 'ADJ' and t.get('is_root')), '')
            return j(S, _adj_bm, 'lendòn', _why_tok.get('bm', ''), '?')
        tam_eq = TAM if TAM not in ('bɛ', 'tɛ', '') else tree.get('tam', 'yé')
        return j(S, tam_eq, O or V, 'yé', *obl_strings, _ccomp_str)


def render_identificatory(tree, S, neg, G):
    _appos = next((t for t in tree.get('_tokens', [])
                   if t.get('pos') == 'PROPN'
                   and t.get('dep') in ('ROOT', 'appos', 'flat', 'flat:name')), None)
    _appos_bm = (_appos.get('bm') or _appos.get('surface', '')) if _appos else ''
    _is_interrog = any(str(t.get('surface', '')).strip() == '?'
                       for t in tree.get('_tokens', []))
    _end = 'wà ?' if _is_interrog else ''
    if _appos_bm:
        return j(_appos_bm, 'tɛ' if neg else 'dòn', _end)
    return j(S, 'tɛ') if neg else j(S, 'dòn', _end)


def render_presentative(tree, S, O, neg, obl_strings, G):
    if neg:
        return j(S, 'tɛ', *obl_strings)
    _is_deictique = any(t.get('role') == 'deictique' for t in tree.get('_tokens', []))
    if _is_deictique:
        return j(O or S, G.get('deictique_marker', 'félé'))
    o_clean = O if O != S else ''
    if o_clean:
        return j(S, 'ni', o_clean, 'dòn', *obl_strings)
    return j(S, 'dòn', *obl_strings)


def render_qualitative(tree, m, S, O, V, TAM, neg, tn, obl_strings, _ccomp_str, G):
    _qual_content = m.get('QUAL', '') or O or V
    _tokens_ref = tree.get('_tokens', [])
    _root_is_participe = any(
        t.get('is_participe_passe') is True
        for t in _tokens_ref if t.get('is_root'))
    if _root_is_participe:
        _v = _qual_content
        if _v and not _v.startswith('['):
            if _v.endswith('n'):
                _v += 'na'
            elif _v[-1] in ('o', 'u', 'ɔ'):
                _v += 'la'
            else:
                _v += 'ra'
        return j(S, _v, *obl_strings)
    _qual_tam = TAM if TAM in ('man', 'ma') else ('man' if neg else 'ka')
    if tn in ('past', 'hab'):
        _tun = _resolve_tam(tn, neg, G) or ('tùn tɛ' if neg else 'tùn bɛ')
        _tun_base = _tun.split()[0] if _tun else 'tùn'
        _qual_tam = j(_tun_base, 'ma' if neg else 'ka')
    elif tree.get('cop_tense') == 'fut':
        _qual_tam = 'tɛ na' if neg else 'bɛ na'
        _qual_content = _qual_content + 'ya'
    return j(S, _qual_tam, _qual_content, *obl_strings, _ccomp_str)


def render_locative_existential(S, neg, obl_strings):
    return j(S, 'tɛ' if neg else 'bɛ', *obl_strings)
