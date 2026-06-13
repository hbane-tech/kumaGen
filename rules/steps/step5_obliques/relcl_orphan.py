"""
step5_obliques/relcl_orphan.py
Catch acl:relcl tokens that weren't handled by relcl_boucle (e.g., relatives on direct objects).
Only appends if the relative hasn't been processed yet, and only if no similar entry already exists in OBL_ALL.
This is a fallback to ensure complex structures like reflexive + relative don't lose the relative clause.
"""
import networkx as nx
from rules.core import j, _resolve_tam


def handle_orphan_relcl(T, m, processed_indices, G_kg, NX_G):
    """
    Scan for acl:relcl tokens not yet in processed_indices.
    Add them to OBL_ALL as separate relative clauses (minw TAM V ...).
    """
    for tok_item in T:
        if tok_item.get('dep') != 'acl:relcl':
            continue
        if tok_item['orig_index'] in processed_indices:
            continue

        # Check if this relative is already in OBL_ALL (avoid duplicates)
        _tok_bm = tok_item.get('bm', '')
        _already_added = any(
            _tok_bm in obl.get('HEAD', '')
            for obl in m.get('OBL_ALL', [])
            if obl.get('DEP_TYPE') == 'acl:relcl'
        )
        if _already_added:
            processed_indices.add(tok_item['orig_index'])
            continue

        # Build relative clause: minw TAM V [ka xcomp_obj] [obl...]
        _rel_subj = next((x for x in T if x.get('dep') == 'nsubj'
                          and x.get('head_index') == tok_item['orig_index']), None)
        _rel_xcomp = next((x for x in T if x.get('dep') == 'xcomp'
                           and x.get('head_index') == tok_item['orig_index']), None)
        _rel_obj = next((x for x in T if x.get('dep') == 'obj'
                         and x.get('head_index') == tok_item['orig_index']), None)

        _rel_v_bm = tok_item.get('bm', '')
        _rel_xc_bm = _rel_xcomp.get('bm', '') if _rel_xcomp else ''
        _rel_o_bm = _rel_obj.get('bm', '') if _rel_obj else ''

        _rel_tam = _resolve_tam(tok_item.get('tense', 'pres'), False, G_kg)
        _rel_str = j('minw', _rel_tam, _rel_v_bm,
                     'ka' if _rel_xc_bm else '',
                     _rel_xc_bm, _rel_o_bm)

        m['OBL_ALL'].append({
            'HEAD': _rel_str,
            'MARKER': '',
            'local_clause_type': 'simple',
            'COMPOUND': '',
            'MOD': '',
            'DEM_PREF': '',
            'DEM_SUFF': '',
            'DEP_TYPE': 'acl:relcl',
            'COMPOUND_IS_QUANTIFIER': False,
            'MARKER_IS_PREFIX': False,
        })

        # Mark all descendants of the relative as processed to avoid re-processing
        processed_indices.update(
            nx.descendants(NX_G, tok_item['orig_index']) | {tok_item['orig_index']}
        )
