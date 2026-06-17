"""
step5_obliques/wagon.py
Assemblage générique d'un wagon OBL_ALL :
compound, amod, démonstratif, loc_nmod, MARKER_IS_PREFIX, fusion week-end.
Utilisé par locatif, temporel, simple.
"""
import networkx as nx
from rules.core import j, adj_man


def _nmod_has_loc_adp(nmod_t, T, loc_markers, tmp_markers):
    own_adp = next((p for p in T if p.get('dep') == 'case'
                    and p.get('head_index') == nmod_t['orig_index']), None)
    if not own_adp:
        return False
    mk = own_adp.get('bm_marker', '')
    return mk in loc_markers or mk in tmp_markers


def fuse_weekend(tok_item, T, processed_indices, G_kg):
    """Fusionne les tokens week-end (NOUN-obl:mod + '-' + NOUN-obl:mod)."""
    if tok_item.get('dep') == 'obl:mod' and tok_item.get('pos') == 'NOUN':
        _next_toks = [x for x in T
                      if x['orig_index'] > tok_item['orig_index']
                      and x.get('dep') == 'obl:mod'
                      and x['orig_index'] not in processed_indices]
        _dash = next((x for x in _next_toks
                      if str(x.get('surface', '')).strip() == '-'), None)
        if _dash:
            _after_dash = next((x for x in _next_toks
                                if x['orig_index'] > _dash['orig_index']), None)
            if _after_dash:
                _fused_bm = j(tok_item.get('bm') or f"[{tok_item.get('lemma')}]",
                              _after_dash.get('bm') or f"[{_after_dash.get('lemma')}]")
                tok_item = dict(tok_item)
                tok_item['bm'] = _fused_bm
                processed_indices.add(_dash['orig_index'])
                processed_indices.add(_after_dash['orig_index'])
    return tok_item


def append(tok_item, T, m, processed_indices, G_kg, NX_G,
           clause_type_val, marker_val, dep_case):
    """Construit et ajoute un wagon générique à m['OBL_ALL']."""
    from rules.core import get_bounded_chunk_tokens
    _loc_markers = G_kg.get('locative_markers', set())
    _tmp_markers = G_kg.get('temporal_markers', set())

    obl_chunk = get_bounded_chunk_tokens(tok_item['orig_index'], NX_G, processed_indices)
    obl_chunk = [x for x in obl_chunk
                 if x.get('pos') not in ('PUNCT', 'SYM')
                 and str(x.get('surface', '')).strip() not in ('-', '–', '—')]
    if not obl_chunk:
        return

    amod_toks = sorted([x for x in obl_chunk if x.get('dep') == 'amod'],
                       key=lambda x: x['orig_index'])

    compound_toks = sorted(
        [x for x in obl_chunk
         if x.get('dep') in ('nmod', 'nummod', 'det')
         and x != tok_item
         and x.get('role') not in ('article', 'pronoun', 'demonstrative')
         and not (x.get('pos') == 'DET' and not x.get('bm'))
         and not _nmod_has_loc_adp(x, T, _loc_markers, _tmp_markers)],
        key=lambda x: x['orig_index'])

    demo_tok = next((x for x in obl_chunk if x.get('role') == 'demonstrative'), None)

    head_base = tok_item.get('bm') or f"[{tok_item.get('lemma')}]"
    if tok_item.get('bm_suffix'):
        head_base += tok_item['bm_suffix']
    if (tok_item.get('pos') not in ('PROPN', 'PRON')
            and not head_base.startswith('[')
            and (tok_item.get('is_plural') or str(tok_item.get('surface', '')).endswith('s'))
            and not head_base.endswith('w')):
        head_base = f"{head_base}w"

    # Démonstratif det
    _demo_surfaces_set = G_kg.get('demonstrative_surfaces', set())
    _demo_det = next((x for x in T
                      if x.get('dep') == 'det'
                      and (x.get('role') == 'demonstrative'
                           or str(x.get('surface', '')).lower() in _demo_surfaces_set)
                      and x.get('head_index') == tok_item['orig_index']), None)
    if not _demo_det:
        _demo_det = next((x for x in obl_chunk
                          if x.get('dep') == 'det'
                          and (x.get('role') == 'demonstrative'
                               or str(x.get('surface', '')).lower() in _demo_surfaces_set)), None)

    pref_val = ''
    suff_dict_val = ''
    if _demo_det and not demo_tok:
        pref_val      = 'nin'
        suff_dict_val = _demo_det.get('bm_suffix') or G_kg.get('demonstrative_suffix', 'in') or 'in'
        processed_indices.add(_demo_det['orig_index'])

    # purposive sur nom → équatif yé
    if (dep_case and dep_case.get('role') == 'purposive'
            and tok_item.get('pos') in ('NOUN', 'PROPN', 'NUM')):
        marker_val      = G_kg.get('equative_marker', 'yé') or 'yé'
        clause_type_val = 'simple'

    # bm_suffix du verbe ROOT sur le marqueur
    _root_verb_suffix = None  # root_tok non disponible ici — géré en amont si nécessaire

    # Compound
    compound_elements = []
    absorbed_amods    = set()
    for c_tok in compound_toks:
        compound_elements.append(c_tok.get('bm') or f"[{c_tok.get('lemma')}]")
    compound_base = j(*compound_elements) if compound_elements else ''

    # mod_compiled
    _compound_amods = []
    for c_tok in compound_toks:
        c_amod = next((x for x in obl_chunk if x.get('dep') == 'amod'
                       and x.get('head_index') == c_tok['orig_index']), None)
        if c_amod and c_amod['orig_index'] not in absorbed_amods:
            _compound_amods.append(c_amod)
            absorbed_amods.add(c_amod['orig_index'])
    clean_amod_toks = [a for a in amod_toks if a['orig_index'] not in absorbed_amods]
    # Quantifier amods (e.g. tous→bɛɛ) postposés directement sur head_base,
    # sans adj_man. Les autres amods ADJ passent par adj_man normalement.
    _quant_amods  = [a for a in clean_amod_toks if a.get('role') == 'quantifier']
    _distrib_dets = [x for x in obl_chunk
                     if x.get('dep') == 'det' and x.get('role') == 'distributive_each']
    _regular_amods = [a for a in clean_amod_toks if a.get('role') != 'quantifier']
    mod_compiled = j(*[
        adj_man(a.get('bm') or f"[{a.get('lemma')}]",
                is_classifying=a.get('is_classifying_adj', False)) if a.get('pos') == 'ADJ'
        else (a.get('bm') or f"[{a.get('lemma')}]")
        for a in _regular_amods + _compound_amods
    ])
    for a in _quant_amods:
        head_base = j(head_base, a.get('bm') or f"[{a.get('lemma')}]")
    for d in _distrib_dets:
        _d_bm = d.get('bm') or 'ò'
        head_base = j(head_base, _d_bm, head_base)

    if demo_tok and compound_base:
        suff_val = G_kg.get('demonstrative_suffix', 'in') or 'in'
        if not compound_base.endswith(suff_val):
            compound_base = f"{compound_base} {suff_val}"
        pref_val      = demo_tok.get('bm') or 'nin'
        suff_dict_val = ''
    elif demo_tok:
        pref_val      = demo_tok.get('bm') or ''
        suff_dict_val = G_kg.get('demonstrative_suffix', 'in') or ''

    # loc_nmod imbriqués
    loc_nmod_toks = sorted(
        [x for x in obl_chunk if x.get('dep') == 'nmod' and x != tok_item
         and _nmod_has_loc_adp(x, T, _loc_markers, _tmp_markers)],
        key=lambda x: x['orig_index'])
    for loc_nmod in loc_nmod_toks:
        loc_adp    = next((p for p in T if p.get('dep') == 'case'
                           and p.get('head_index') == loc_nmod['orig_index']), None)
        loc_marker = loc_adp.get('bm_marker', '') if loc_adp else ''
        loc_lct    = ('locative' if loc_marker in _loc_markers
                      else 'temporal' if loc_marker in _tmp_markers else 'simple')
        m['OBL_ALL'].append({
            'HEAD': loc_nmod.get('bm') or f"[{loc_nmod.get('lemma')}]",
            'MARKER': loc_marker, 'local_clause_type': loc_lct,
            'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
            'DEP_TYPE': 'case', 'COMPOUND_IS_QUANTIFIER': False,
            'MARKER_IS_PREFIX': False,
        })
        processed_indices.add(loc_nmod['orig_index'])
        if loc_adp: processed_indices.add(loc_adp['orig_index'])

    _has_quantifier_compound = any(x.get('role') == 'quantifier' for x in compound_toks)

    m['OBL_ALL'].append({
        'HEAD':               head_base,
        'MARKER':             marker_val or '',
        'local_clause_type':  clause_type_val,
        'COMPOUND':           compound_base,
        'MOD':                mod_compiled,
        'DEM_PREF':           pref_val,
        'DEM_SUFF':           suff_dict_val,
        'DEP_TYPE':           'case' if dep_case else '',
        'COMPOUND_IS_QUANTIFIER': _has_quantifier_compound,
        'MARKER_IS_PREFIX': (
            clause_type_val == 'temporal'
            and dep_case
            and (
                (dep_case.get('role') == 'temporal'
                 and dep_case.get('bm_marker', '') in {'Kabini', "k'an bɔ"})
                or bool(marker_val and marker_val.startswith('['))
            )
        ),
    })
    processed_indices.update([t['orig_index'] for t in obl_chunk])
