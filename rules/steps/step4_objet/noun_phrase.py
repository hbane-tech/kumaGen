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

    if tree.get('clause_type') != 'existential_nominal':
        tree['clause_type'] = 'noun_phrase'

    r_idx     = root_noun['orig_index']
    child_adj = next((x for x in T if x.get('dep') == 'amod'
                      and x.get('head_index') == r_idx), None)
    acl_tok   = next((x for x in T if x.get('dep') == 'acl'
                      and x.get('head_index') == r_idx), None)

    acl_val = acl_tok.get('bm', '') if acl_tok else ''
    if acl_val and not acl_val.endswith('len'):
        acl_val += 'len'

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
