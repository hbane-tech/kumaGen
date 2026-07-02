"""
step5_obliques/privatif.py
Oblique privatif : role='privative' → marqueur lu depuis PrivativeRule KG.
"""
from rules.core import j


def _get_marker(tok_item, T, G_kg):
    """Lit le marqueur privatif selon le POS du token.

    Règles Bambara :
    - sans + VERB nominalisé (bm en -li) → bali
    - sans + NOUN → tan
    - sans + PROPN → kɔ
    - sans + PRON → kɔ
    """
    priv_rules = G_kg.get('kg_privative_rules', [])
    pos = tok_item.get('pos', '')
    has_poss = any(
        x.get('dep') == 'det' and x.get('role') in ('pronoun', 'possessive')
        and x.get('head_index') == tok_item['orig_index'] for x in T)

    # PROPN : toujours kɔ
    if pos == 'PROPN':
        return G_kg.get('privative_propn_marker', 'kɔ') or 'kɔ'

    # PRON ou possession : kɔ
    if pos == 'PRON' or has_poss:
        return G_kg.get('privative_pron_marker', 'kɔ') or 'kɔ'

    # VERB (rarement en position de ROOT, mais possible) : bali
    if pos == 'VERB':
        return G_kg.get('privative_verb_marker_suffix', 'bali') or 'bali'

    # NOUN : détecte si c'est nom d'action/déverbatif (semantic_class='action') → bali
    # ou verbe nominalisé morphologically (bm en -li) → bali ; sinon tan
    if pos == 'NOUN':
        sc = tok_item.get('semantic_class', '')
        bm = tok_item.get('bm', '')

        # Nom d'action (cuisson=action, travail=action, etc.)
        if sc == 'action':
            return G_kg.get('privative_verb_marker_suffix', 'bali') or 'bali'

        # Verbe nominalisé morphologically (dúnli, báarali, etc.)
        if bm and bm.rstrip('.').endswith('li'):
            return G_kg.get('privative_verb_marker_suffix', 'bali') or 'bali'

        # Nom simple
        return G_kg.get('privative_noun_marker', 'tan') or 'tan'

    # Fallback
    return 'tan'


def handle(tok_item, T, m, processed_indices, G_kg, dep_case):
    marker_val = _get_marker(tok_item, T, G_kg)
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
