"""
step4_objet/noun_phrase.py
Cas noun_phrase : NOUN ROOT + acl participial → -len (résultatif adnominal).
Détecte aussi existential_nominal.
"""
from rules.core import j


def run(T, tree, m, processed_indices, G_kg, root_noun, has_acl, _has_relcl):
    _has_acl_only   = any(x.get('dep') == 'acl' for x in T)

    # On délègue ici dès qu'il y a un participe 'acl' sur le nom (frein … lié …),
    # MÊME si une relative 'acl:relcl' existe ailleurs (sur un nmod profond) : la
    # relative est gérée séparément (step5 relcl_boucle). Sans ça, l'objet
    # existentiel 'frein politique lié' était entièrement perdu quand la relative
    # était présente. On n'agit que sur l'acl rattaché au NOM racine.
    _acl_on_root = any(x.get('dep') == 'acl'
                       and x.get('head_index') == root_noun['orig_index']
                       for x in T) if root_noun else False
    if not (root_noun and _has_acl_only and _acl_on_root):
        return

    r_idx     = root_noun['orig_index']
    child_adj = next((x for x in T if x.get('dep') == 'amod'
                      and x.get('head_index') == r_idx), None)
    acl_tok   = next((x for x in T if x.get('dep') == 'acl'
                      and x.get('head_index') == r_idx), None)

    # Finite temporal clause (quand/lorsque + nsubj): treat as existential topic,
    # not as a participial adjective. step5 will build the temporal OBL.
    if acl_tok:
        _temporal_mark = next((x for x in T
                               if x.get('dep') == 'mark' and x.get('pos') == 'SCONJ'
                               and x.get('head_index') == acl_tok['orig_index']), None)
        if _temporal_mark:
            m['S'] = root_noun.get('bm') or f"[{root_noun.get('lemma')}]"
            processed_indices.add(root_noun['orig_index'])
            return

    acl_val = acl_tok.get('bm', '') if acl_tok else ''
    if acl_val:
        _sfx = G_kg.get('morpho_rules', {}).get('statif', {}).get('suffix', 'len')
        if not acl_val.endswith(_sfx):
            acl_val += _sfx

    head_block = j(
        root_noun.get('bm', root_noun.get('surface', '')),
        child_adj.get('bm', '') if child_adj else '',
        acl_val)

    if acl_tok and acl_tok.get('bm_suffix'):
        m['V_SUFFIX'] = acl_tok.get('bm_suffix', '')

    m['O'] = head_block
    m['V'] = ''

    _noyau = {root_noun['orig_index']}
    if acl_tok:   _noyau.add(acl_tok['orig_index'])
    if child_adj: _noyau.add(child_adj['orig_index'])
    processed_indices.update(_noyau)
