"""
step4_objet/ownership.py
Appartenance (ownership) : purposive case + cop sur NOUN/PRON ROOT.
"""
from rules.core import j


def run(T, tree, m, processed_indices, G_kg, root_tok):
    _purp_case = next((x for x in T
                       if x.get('dep') == 'case'
                       and x.get('role') in ('purposive', 'benefactive')
                       and root_tok and x.get('head_index') == root_tok['orig_index']), None)
    _has_cop = any(x.get('dep') == 'cop' for x in T)

    if not (_purp_case and _has_cop and root_tok
            and root_tok.get('pos') in ('NOUN', 'PRON')):
        return


    _ta     = G_kg.get('ownership_marker') or 'ta'  # marqueur possessif ('ta')
    _focus  = G_kg.get('focus_marker', '')           # particule d'emphase ('de')
    if root_tok.get('pos') == 'PRON':
        # Structure bambara : {O} {COMPANION} dòn → 'i de ta dòn'
        # COMPANION = focus_marker + ownership_marker (ex: 'de ta')
        m['O'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
        if not m.get('COMPANION'):
            m['COMPANION'] = j(_focus, _ta)
        tree['ownership_o_is_pron'] = True
    else:
        tree['ownership_o_is_pron'] = False
        _poss_det = next((x for x in T
                          if x.get('dep') == 'det'
                          and x.get('role') in ('pronoun', 'possessive')
                          and x.get('head_index') == root_tok['orig_index']), None)
        _root_bm = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
        if _poss_det:
            _poss_bm = _poss_det.get('bm', '')
            _noun_phrase = (j('n', _root_bm) if _poss_bm == 'n'
                            else j(_poss_bm, 'ka', _root_bm))
            m['O'] = j(_noun_phrase, _ta)
            processed_indices.add(_poss_det['orig_index'])
        else:
            m['O'] = j(_root_bm, _ta)

    processed_indices.add(root_tok['orig_index'])
    processed_indices.add(_purp_case['orig_index'])

    # Démonstratif sur le sujet
    _subj_for_ownership = next((x for x in T
                                if x.get('dep') in ('nsubj', 'nsubj:pass')
                                and x.get('pos') == 'NOUN'), None)
    if _subj_for_ownership:
        _demo_surfaces = G_kg.get('demonstrative_surfaces', set())
        _demo_on_subj  = next((x for x in T
                               if x.get('dep') == 'det'
                               and (x.get('role') == 'demonstrative'
                                    or str(x.get('surface', '')).lower() in _demo_surfaces)
                               and x.get('head_index') == _subj_for_ownership['orig_index']), None)
        _subj_bm = _subj_for_ownership.get('bm') or _subj_for_ownership.get('surface', '')
        if _demo_on_subj:
            _demo_suf = G_kg.get('demonstrative_suffix', '')
            m['S'] = j(_demo_on_subj.get('bm') or G_kg.get('demonstrative_prefix', ''), _subj_bm, _demo_suf)
            processed_indices.add(_demo_on_subj['orig_index'])
        else:
            m['S'] = _subj_bm
        processed_indices.add(_subj_for_ownership['orig_index'])
