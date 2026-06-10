"""
Présentatif dòn :
  run_with    → has_with → ni comitative
  run_cas14   → expletif + NOUN root + cop → n bálimakɛw dòn
"""
from rules.core import j


def run_with(tree, G_kg):
    tree['clause_type'] = 'presentative'
    tree['tam']         = G_kg.get('comitative_marker', 'ni')


def run_cas14(T, tree, m, processed_indices, G_kg, root_tok):
    tree['clause_type'] = 'presentative'
    _root_bm = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
    _poss_root = next((x for x in T if x.get('dep') == 'det'
                       and x.get('role') in ('pronoun', 'possessive')
                       and x.get('head_index') == root_tok['orig_index']), None)
    if _poss_root:
        _poss_bm = _poss_root.get('bm', '')
        _root_bm = j('n', _root_bm) if _poss_bm == 'n' else j(_poss_bm, 'ka', _root_bm)
        processed_indices.add(_poss_root['orig_index'])
    if root_tok.get('is_plural') or str(root_tok.get('surface', '')).endswith('s'):
        if not _root_bm.endswith('w') and not _root_bm.endswith('w)'):
            _parts = _root_bm.split(' ')
            _parts[-1] = _parts[-1] + 'w' if not _parts[-1].endswith('w') else _parts[-1]
            _root_bm = ' '.join(_parts)
    _root_conjs = [x for x in T if x.get('dep') == 'conj'
                   and x.get('head_index') == root_tok['orig_index'] and x.get('bm')]
    for _rc in _root_conjs:
        _rc_bm = _rc.get('bm')
        if _poss_root:
            _poss_bm = _poss_root.get('bm', '')
            _rc_bm = j('n', _rc_bm) if _poss_bm == 'n' else j(_poss_bm, 'ka', _rc_bm)
        _cc_rc = next((x for x in T if x.get('dep') == 'cc'
                       and x.get('head_index') == root_tok['orig_index']), None)
        _cc_bm_rc = _cc_rc.get('bm', '') if _cc_rc and _cc_rc.get('bm') else ''
        _root_bm = j(_root_bm, _cc_bm_rc, _rc_bm)
        processed_indices.add(_rc['orig_index'])
        if _cc_rc: processed_indices.add(_cc_rc['orig_index'])
    m['O'] = _root_bm
    m['S'] = ''
    processed_indices.add(root_tok['orig_index'])
