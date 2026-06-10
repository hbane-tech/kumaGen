"""
step5_obliques/comitative.py
Oblique comitative : role='comitative' → ni + conj + acl:relcl du conj.
"""
import networkx as nx
from rules.core import j, _resolve_tam
from rules.steps.step4_objet.objet_standard import _build_genitive_chain


def _collect_genitive_tokens(tok, all_toks):
    """Collect orig_indices of all tokens consumed by _build_genitive_chain."""
    result = set()
    nmod = next((t for t in all_toks if t.get('dep') == 'nmod'
                 and t.get('head_index') == tok['orig_index']), None)
    poss = next((t for t in all_toks if t.get('dep') == 'det'
                 and t.get('role') in ('pronoun', 'possessive')
                 and t.get('head_index') == tok['orig_index']
                 and t.get('bm')), None)
    for case_t in all_toks:
        if case_t.get('dep') == 'case' and case_t.get('head_index') == tok['orig_index']:
            result.add(case_t['orig_index'])
    if poss:
        result.add(poss['orig_index'])
    if nmod:
        result.add(nmod['orig_index'])
        result |= _collect_genitive_tokens(nmod, all_toks)
    return result


def handle(tok_item, T, m, processed_indices, G_kg, NX_G):
    marker_val = G_kg.get('comitative_marker', 'ni') or 'ni'

    _has_nmod_chain = any(t.get('dep') == 'nmod'
                          and t.get('head_index') == tok_item['orig_index']
                          for t in T)

    if _has_nmod_chain:
        head_base = _build_genitive_chain(tok_item, T, G_kg)
        _chain_toks = _collect_genitive_tokens(tok_item, T)
        processed_indices.update(_chain_toks)
    else:
        head_base = tok_item.get('bm') or f"[{tok_item.get('lemma')}]"
        if tok_item.get('bm_suffix'):
            head_base += tok_item['bm_suffix']
        # Possessif sur tok_item lui-même (ex: mon fils → n dénkɛ)
        _poss_main = next((x for x in T if x.get('dep') == 'det'
                           and x.get('role') in ('pronoun', 'possessive')
                           and x.get('head_index') == tok_item['orig_index']), None)
        if _poss_main:
            _poss_bm = _poss_main.get('bm', '')
            head_base = j('n', head_base) if _poss_bm == 'n' else j(_poss_bm, 'ka', head_base)
            processed_indices.add(_poss_main['orig_index'])

    _com_conjs = [x for x in T if x.get('dep') == 'conj'
                  and x.get('head_index') == tok_item['orig_index']
                  and x['orig_index'] not in processed_indices]

    for _cc in _com_conjs:
        _cc_bm = _cc.get('bm') or f"[{_cc.get('lemma')}]"
        # Possessif du conj
        _cc_poss = next((x for x in T if x.get('dep') == 'det'
                         and x.get('role') in ('pronoun', 'possessive')
                         and x.get('head_index') == _cc['orig_index']), None)
        if _cc_poss:
            _poss_bm = _cc_poss.get('bm', '')
            _cc_bm = j('n', _cc_bm) if _poss_bm == 'n' else j(_poss_bm, 'ka', _cc_bm)
            processed_indices.add(_cc_poss['orig_index'])
        # Relative du conj
        _cc_relcl = next((x for x in T if x.get('dep') == 'acl:relcl'
                          and x.get('head_index') == _cc['orig_index']), None)
        if _cc_relcl:
            _rel_marker = G_kg.get('relative_marker', 'mìn') or 'mìn'
            _rel_adj    = next((x for x in T if x.get('head_index') == _cc_relcl['orig_index']
                                and x.get('pos') == 'ADJ'), None)
            _rel_v_bm   = _cc_relcl.get('bm', '')
            _rel_adj_bm = _rel_adj.get('bm', '') if _rel_adj else ''
            _relcl_self_adj = _cc_relcl.get('pos') == 'ADJ'
            if _relcl_self_adj or (_rel_adj and not _rel_v_bm):
                _rel_tam = 'ka'
            else:
                _rel_tam = _resolve_tam(_cc_relcl.get('tense', 'pres'), False, G_kg)
            _cc_bm = j(_cc_bm, _rel_marker, _rel_tam, _rel_v_bm or _rel_adj_bm)
            processed_indices.update(
                nx.descendants(NX_G, _cc_relcl['orig_index']) | {_cc_relcl['orig_index']})
        head_base = j(head_base, marker_val, _cc_bm)
        processed_indices.add(_cc['orig_index'])
        _cc_tok_com = next((x for x in T if x.get('dep') == 'cc'
                            and x.get('head_index') == tok_item['orig_index']), None)
        if _cc_tok_com:
            processed_indices.add(_cc_tok_com['orig_index'])

    # ni déjà intégré dans head_base si conj → MARKER vide
    # _effective_marker = '' if _com_conjs else marker_val

    # dep_case = next((x for x in T if x.get('dep') == 'case'
    #                  and x.get('head_index') == tok_item['orig_index']), None)

    # m['OBL_ALL'].append({
    #     'HEAD': head_base, 'MARKER': _effective_marker,
    #     'DEM_SUFF': 'yé',   # ← marker final yé
    #     'local_clause_type': 'comitative',
    #     'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
    #     'DEP_TYPE': 'case' if dep_case else '',
    #     'COMPOUND_IS_QUANTIFIER': False, 'MARKER_IS_PREFIX': False,
    # })
    # Construire ni HEAD yé directement
    dep_case = next((x for x in T if x.get('dep') == 'case'
                     and x.get('head_index') == tok_item['orig_index']), None)
    if not _com_conjs:
        # ni + head + yé intégrés dans HEAD, MARKER vide
        head_base = j(marker_val, head_base, 'yé')
    m['OBL_ALL'].append({
        'HEAD': head_base, 'MARKER': '',
        'local_clause_type': 'comitative',
        'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
        'DEP_TYPE': 'case' if dep_case else '',
        'COMPOUND_IS_QUANTIFIER': False, 'MARKER_IS_PREFIX': False,
    })
    processed_indices.add(tok_item['orig_index'])
    if dep_case:
        processed_indices.add(dep_case['orig_index'])
