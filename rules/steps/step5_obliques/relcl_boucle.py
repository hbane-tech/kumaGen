"""
step5_obliques/relcl_boucle.py
Relatives acl:relcl rencontrées dans la boucle obliques (sur conj, sur obl...).
Forme : minw TAM V ka O xcomp
"""
import networkx as nx
from rules.core import j, _resolve_tam


def handle(tok_item, T, m, processed_indices, G_kg, NX_G):
    _rel_subj  = next((x for x in T if x.get('dep') == 'nsubj'
                       and x.get('head_index') == tok_item['orig_index']), None)
    _rel_xcomp = next((x for x in T if x.get('dep') == 'xcomp'
                       and x.get('head_index') == tok_item['orig_index']), None)
    _rel_obj   = next((x for x in T if x.get('dep') == 'obj'
                       and x.get('head_index') == tok_item['orig_index']), None)
    _rel_obj2  = next((x for x in T if x.get('dep') == 'obj'
                       and _rel_xcomp
                       and x.get('head_index') == _rel_xcomp['orig_index']), None)
    _rel_o2_amod = next((x for x in T if x.get('dep') == 'amod'
                         and _rel_obj2
                         and x.get('head_index') == _rel_obj2['orig_index']), None)
    _rel_v_bm  = tok_item.get('bm', '')
    _rel_xc_bm = _rel_xcomp.get('bm', '') if _rel_xcomp else ''
    _rel_o_bm  = _rel_obj.get('bm', '') if _rel_obj else ''
    _rel_o2_bm = _rel_obj2.get('bm', '') if _rel_obj2 else ''
    if _rel_o2_amod:
        _rel_o2_bm = j(_rel_o2_bm, _rel_o2_amod.get('bm', ''))
    _rel_tam = _resolve_tam(tok_item.get('tense', 'pres'), False, G_kg)
    _rel_str = j('minw', _rel_tam, _rel_v_bm,
                 'ka' if _rel_xc_bm else '',
                 _rel_xc_bm, _rel_o2_bm or _rel_o_bm)
    m['OBL_ALL'].append({
        'HEAD': _rel_str, 'MARKER': '', 'local_clause_type': 'simple',
        'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
        'DEP_TYPE': 'acl:relcl', 'COMPOUND_IS_QUANTIFIER': False,
        'MARKER_IS_PREFIX': False,
    })
    processed_indices.update(nx.descendants(NX_G, tok_item['orig_index']) | {tok_item['orig_index']})
