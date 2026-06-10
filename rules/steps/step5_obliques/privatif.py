"""
step5_obliques/privatif.py
Oblique privatif : role='privative' → tan (NOUN) / bali (VERB) / kɔ (PRON/poss).
"""
from rules.core import j


def handle(tok_item, T, m, processed_indices, G_kg, dep_case):
    if tok_item.get('pos') == 'VERB':
        marker_val = 'bali'
    elif any(x.get('dep') == 'det' and x.get('role') in ('pronoun', 'possessive')
             and x.get('head_index') == tok_item['orig_index'] for x in T):
        marker_val = 'kɔ'
    elif tok_item.get('pos') == 'PRON':
        marker_val = 'kɔ'
    else:
        marker_val = 'tan'

    _head = tok_item.get('bm') or f"[{tok_item.get('lemma')}]"
    m['OBL_ALL'].append({
        'HEAD': _head, 'MARKER': marker_val, 'local_clause_type': 'privative',
        'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
        'DEP_TYPE': 'case', 'COMPOUND_IS_QUANTIFIER': False,
        'MARKER_IS_PREFIX': False,
    })
    processed_indices.add(tok_item['orig_index'])
    if dep_case:
        processed_indices.add(dep_case['orig_index'])
