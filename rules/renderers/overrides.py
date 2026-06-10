"""
rules/renderers/overrides.py
Cas spéciaux détectés AVANT le dispatch par clause_type.
Cas -1 : NOUN ROOT + conj NOUN + cop + ce → présentatif ni
Cas -2 : cop futur + NOUN → équatif futur
Cas  0 : comitative + cop → présentatif ni
Cas  1 : PROPN ROOT/conj → identificatoire/équatif
Cas  2 : ADJ is_valeur + expletif → présentatif
Cas  3 : ADJ is_participe_passe + cop → résultatif
"""
from rules.core import j


def apply_overrides(tree, m, S, O, neg, _tokens_ref,
                    _has_real_subj_ttb, _has_expletif_ttb):
    """
    Tente de résoudre la clause via les cas d'override.
    Retourne (result_str, handled) — handled=True si un cas a matché.
    """

    # ── Cas -1 : Ce sont mes frères et sœurs → n bálimakɛ ni n dúsukɛ dòn ──
    _ce_subj = any(
        str(t.get('surface', '')).lower().rstrip("'").rstrip('\u2019') in ('ce', 'c')
        and t.get('dep') in ('nsubj', 'expl:subj', 'expl')
        for t in _tokens_ref)
    _root_noun = next((t for t in _tokens_ref
                       if t.get('dep') == 'ROOT'
                       and t.get('pos') in ('NOUN', 'PROPN')
                       and t.get('bm')), None)
    _conj_noun = next((t for t in _tokens_ref
                       if t.get('dep') == 'conj'
                       and t.get('pos') in ('NOUN', 'PROPN')
                       and t.get('bm')), None)
    _has_cop = any(t.get('dep') == 'cop' for t in _tokens_ref)
    if _ce_subj and _root_noun and _conj_noun and _has_cop:
        _poss_r = next((t for t in _tokens_ref
                        if t.get('dep') == 'det'
                        and t.get('role') in ('pronoun', 'possessive')
                        and t.get('head_index') == _root_noun['orig_index']), None)
        _poss_c = next((t for t in _tokens_ref
                        if t.get('dep') == 'det'
                        and t.get('role') in ('pronoun', 'possessive')
                        and t.get('head_index') == _conj_noun['orig_index']), None)
        _r_bm = _root_noun.get('bm', '')
        _c_bm = _conj_noun.get('bm', '')
        if _poss_r:
            _pb = _poss_r.get('bm', '')
            _r_bm = j('n', _r_bm) if _pb == 'n' else j(_pb, _r_bm)
        if _poss_c:
            _pb2 = _poss_c.get('bm', '')
            _c_bm = j('n', _c_bm) if _pb2 == 'n' else j(_pb2, _c_bm)
        return j(_r_bm, 'ni', _c_bm, 'dòn'), True

    # ── Cas -2 : cop futur + NOUN → équatif futur ────────────────────────────
    _cop_fut = next((t for t in _tokens_ref
                     if t.get('dep') in ('cop', 'aux', 'aux:tense')
                     and t.get('role') in ('copula', 'auxiliary', 'content', '')
                     and (t.get('tense') == 'fut'
                          or 'Tense=Fut' in str(t.get('morph', ''))
                          or tree.get('cop_tense') == 'fut')), None)
    _noun_conj = next((t for t in _tokens_ref
                       if t.get('dep') in ('conj', 'attr', 'appos', 'ROOT')
                       and t.get('pos') in ('NOUN', 'PROPN', 'ADJ')
                       and t.get('bm')
                       and not t.get('is_root', False)), None)
    if not _noun_conj and _cop_fut:
        _noun_conj = next((t for t in _tokens_ref
                           if t.get('dep') == 'ROOT'
                           and t.get('pos') in ('NOUN', 'PROPN')
                           and t.get('bm')), None)
    if not _cop_fut and tree.get('cop_tense') == 'fut':
        _cop_fut = {'dep': 'cop', 'tense': 'fut',
                    'bm': tree.get('cop_bm', ''),
                    'lemma': tree.get('cop_lemma', 'être')}
    if _cop_fut and _noun_conj and S:
        _cop_bm = _cop_fut.get('bm') or f"[{_cop_fut.get('lemma', 'être')}]"
        _obj_bm = O if O else _noun_conj.get('bm', '')
        _tam = 'tɛ na' if neg else 'bɛ na'
        return j(S, _tam, _cop_bm, _obj_bm, 'yé'), True

    # ── Cas -0.5 : expletif PRON + ADJ/ADV ROOT → qualitatif o ka ADJ ──────────
    # c'est bon → o ka ànháàn   /   c'est bien → o ka kóɲuman
    _adj_adv_root = next((t for t in _tokens_ref
                          if t.get('dep') == 'ROOT'
                          and t.get('pos') in ('ADJ', 'ADV')
                          and t.get('bm')
                          and not t.get('is_loc')
                          and not t.get('is_statif')
                          and not t.get('is_participe_passe')
                          and not t.get('is_valeur')
                          and not t.get('is_nominal_adj')), None)
    _has_expl_pron = any(t.get('role') == 'expletive' and t.get('pos') == 'PRON'
                         for t in _tokens_ref)
    if _adj_adv_root and _has_expl_pron and S:
        _qual_tam = 'man' if neg else 'ka'
        return j(S, _qual_tam, _adj_adv_root.get('bm', '')), True

    # ── Cas 0 : comitative + cop → présentatif ni ────────────────────────────
    _com_case = any(t.get('dep') == 'case' and t.get('role') == 'comitative'
                    for t in _tokens_ref)
    _com_conj = next((t for t in _tokens_ref
                      if t.get('dep') in ('conj', 'nmod')
                      and t.get('pos') in ('NOUN', 'PROPN')
                      and t.get('bm')
                      and _com_case), None)
    _has_verb_root = any(t.get('dep') == 'ROOT' and t.get('pos') == 'VERB'
                         for t in _tokens_ref)
    if (_com_conj and _com_case and S
            and not _has_verb_root
            and not any(t.get('dep') == 'obj' for t in _tokens_ref)):
        # Utiliser O du slot si déjà construit (chaîne génitive complète)
        # Sinon fallback sur le premier nmod/conj trouvé
        if O:
            _companion = O
        else:
            _c_bm = _com_conj.get('bm') or _com_conj.get('surface', '')
            _poss_c = next((t for t in _tokens_ref
                            if t.get('dep') == 'det'
                            and t.get('role') in ('pronoun', 'possessive')
                            and t.get('head_index') == _com_conj['orig_index']), None)
            if _poss_c:
                _pb = _poss_c.get('bm', '')
                _c_bm = j('n', _c_bm) if _pb == 'n' else j(_pb, 'ka', _c_bm)
            _companion = _c_bm
        return j(S, 'ni', _companion, 'dòn'), True

    # ── Cas 1 : PROPN ROOT ou conj+cop ───────────────────────────────────────
    _propn_root = next((t for t in _tokens_ref
                        if t.get('pos') == 'PROPN'
                        and t.get('dep') == 'ROOT'), None)
    _propn_conj = next((t for t in _tokens_ref
                        if t.get('pos') == 'PROPN'
                        and t.get('dep') in ('conj', 'attr', 'xcomp')
                        and any(x.get('dep') == 'cop' for x in _tokens_ref)), None)
    if _propn_conj and S:
        _propn_bm = _propn_conj.get('bm') or _propn_conj.get('surface', '')
        result = j(S, 'tɛ', _propn_bm, 'yé') if neg else j(S, 'yé', _propn_bm, 'yé')
        return result, True
    if _propn_root and not _has_real_subj_ttb:
        _propn_bm = _propn_root.get('bm') or _propn_root.get('surface', '')
        return j(_propn_bm, 'tɛ' if neg else 'dòn'), True

    # ── Cas 2 : ADJ is_valeur + expletif ─────────────────────────────────────
    _valeur_adj = next((t for t in _tokens_ref
                        if t.get('pos') == 'ADJ'
                        and t.get('is_valeur') is True
                        and not t.get('is_participe_passe')
                        and not t.get('is_statif')
                        and t.get('bm')), None)
    if _valeur_adj and _has_expletif_ttb and not _has_real_subj_ttb:
        return j(_valeur_adj.get('bm'), 'tɛ' if neg else 'dòn'), True

    # ── Cas 3 : ADJ is_participe_passe + cop → résultatif ────────────────────
    _part_adj = next((t for t in _tokens_ref
                      if t.get('is_participe_passe') is True
                      and t.get('bm')), None)
    if not _part_adj and tree.get('is_participe_passe') and tree.get('participe_bm'):
        _part_adj = {'bm': tree['participe_bm'], 'is_participe_passe': True}
    _has_cop = any(t.get('dep') == 'cop' for t in _tokens_ref)
    if _part_adj and _has_cop:
        _part_bm = _part_adj.get('bm', '')
        _subj = S if S else 'o'
        if neg:
            return j(_subj, 'ma', _part_bm), True
        _v = _part_bm
        if _v and not _v.endswith('ra') and not _v.endswith('na'):
            _v += 'na' if _v.endswith('n') else 'ra'
        return j(_subj, _v), True

    return '', False
