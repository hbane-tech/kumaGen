"""
step4_objet/relcl_post.py
Post-traitement acl:relcl sur l'objet (après la boucle objet principale)
+ pré-marquage ccomp.
"""
import networkx as nx
from rules.core import j, _resolve_tam


def run(T, tree, m, processed_indices, G_kg, NX_G):
    for _rt in sorted(T, key=lambda x: x['orig_index']):
        if _rt.get('dep') != 'acl:relcl':
            continue
        if _rt['orig_index'] in processed_indices:
            continue
        # Skip si tête = conj d'un obl comitative (traité step5)
        _rt_head = next((x for x in T if x['orig_index'] == _rt.get('head_index')), None)
        if (_rt_head and _rt_head.get('dep') == 'conj'
                and any(x.get('dep') == 'case' and x.get('role') == 'comitative'
                        and x.get('head_index') == _rt_head.get('head_index') for x in T)):
            continue

        _rel_subj  = next((x for x in T if x.get('dep') == 'nsubj'
                           and x.get('head_index') == _rt['orig_index']), None)
        _rel_xcomp = next((x for x in T if x.get('dep') == 'xcomp'
                           and x.get('head_index') == _rt['orig_index']), None)
        _rel_obj   = next((x for x in T if x.get('dep') == 'obj'
                           and x.get('head_index') == _rt['orig_index']), None)
        _rel_obj2  = next((x for x in T if x.get('dep') == 'obj'
                           and _rel_xcomp
                           and x.get('head_index') == _rel_xcomp['orig_index']), None)
        _rel_o2_amod = next((x for x in T if x.get('dep') == 'amod'
                             and _rel_obj2
                             and x.get('head_index') == _rel_obj2['orig_index']), None)
        _rel_v_bm  = _rt.get('bm', '')
        _rel_xc_bm = _rel_xcomp.get('bm', '') if _rel_xcomp else ''
        _rel_o_bm  = _rel_obj.get('bm', '') if _rel_obj else ''
        _rel_o2_bm = _rel_obj2.get('bm', '') if _rel_obj2 else ''
        if _rel_o2_amod:
            _rel_o2_bm = j(_rel_o2_bm, _rel_o2_amod.get('bm', ''))
        _rel_tam = _resolve_tam(_rt.get('tense', 'pres'), False, G_kg)
        _rel_obls = []
        if _rel_xcomp:
            for _robl in sorted(T, key=lambda x: x['orig_index']):
                if _robl['orig_index'] in processed_indices:
                    continue
                if _robl.get('dep') not in ('obl:mod', 'obl:arg', 'obl'):
                    continue
                if not nx.has_path(NX_G, _rel_xcomp['orig_index'], _robl['orig_index']):
                    continue
                _robl_bm   = _robl.get('bm') or f"[{_robl.get('lemma')}]"
                _robl_case = next((x for x in T if x.get('dep') == 'case'
                                   and x.get('head_index') == _robl['orig_index']), None)
                _robl_nmod = next((x for x in T if x.get('dep') == 'nmod'
                                   and x.get('head_index') == _robl['orig_index']), None)
                _robl_marker = _robl_case.get('bm_marker', 'la') if _robl_case else 'la'
                if _robl_nmod:
                    _rel_obls.append(j(_robl_nmod.get('bm') or f"[{_robl_nmod.get('lemma')}]",
                                       _robl_bm, _robl_marker))
                else:
                    _rel_obls.append(j(_robl_bm, _robl_marker))

        _rel_o_final = _rel_o2_bm or _rel_o_bm
        if _rel_xc_bm:
            _rel_str = j('min', _rel_tam, _rel_v_bm, 'ka', _rel_o_final, _rel_xc_bm, *_rel_obls)
        else:
            _rel_str = j('min', _rel_tam, _rel_o_final, _rel_v_bm, *_rel_obls)

        m['OBL_ALL'].append({
            'HEAD': _rel_str, 'MARKER': '', 'local_clause_type': 'simple',
            'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
            'DEP_TYPE': 'acl:relcl', 'COMPOUND_IS_QUANTIFIER': False,
            'MARKER_IS_PREFIX': False,
        })
        processed_indices.update(nx.descendants(NX_G, _rt['orig_index']) | {_rt['orig_index']})

    # Pré-marquage ccomp
    for _cc in T:
        if _cc.get('dep') == 'ccomp' and _cc['orig_index'] not in processed_indices:
            processed_indices.update(nx.descendants(NX_G, _cc['orig_index']) | {_cc['orig_index']})
