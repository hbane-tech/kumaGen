"""
step4_objet/noun_phrase.py
Cas noun_phrase : NOUN ROOT + acl participial → -len (résultatif adnominal).
Détecte aussi existential_nominal.
"""
from rules.core import j, render_obl_entry, _resolve_tam
from rules.steps.step4_objet import relcl_post
from rules.steps.step5_obliques import wagon


def run(T, tree, m, processed_indices, G_kg, NX_G, root_noun, has_acl, _has_relcl):
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

    # existential_* (absolute/localized/nominal) : le template KG "{O} {TAM}"
    # place le TAM après TOUT O — correct pour un O simple, mais quand un
    # complément/relative se replie ensuite dans O (ci-dessous, "lié aux
    # logiques... QUI VEULENT..."), le TAM se retrouve après une clause déjà
    # complète (son propre TAM interne), à l'autre bout de la phrase, au lieu
    # de juste après le noyau nominal (bug trouvé 2026-07-19 : "frein... à
    # l'écart de la gouvernance bɛ" — 'bɛ' orphelin en toute fin au lieu de
    # "frein politique bɛ, lié à..."). On insère donc le TAM ICI, juste après
    # le noyau, et on bloque son refill en aval pour que le template ne le
    # rajoute pas une seconde fois.
    if tree.get('clause_type') in ('existential_absolute', 'existential_localized',
                                    'existential_nominal') and acl_tok:
        _tam_here = _resolve_tam(tree.get('tense', 'pres'), bool(tree.get('neg')), G_kg)
        if _tam_here:
            m['O'] = j(m['O'], _tam_here)
            tree['_tam_embedded_in_o'] = True
            # step3_verbe.py pose déjà tree['tam']='bɛ' par défaut AVANT que
            # step4 ne s'exécute — sans le vider ici, cette valeur pré-posée
            # survit telle quelle jusqu'au rendu (le refill-guard ne protège
            # que contre un NOUVEAU calcul, pas contre une valeur déjà en
            # place), et le template "{O} {TAM}" la rajoute quand même.
            tree['tam'] = ''

    _noyau = {root_noun['orig_index']}
    if acl_tok:   _noyau.add(acl_tok['orig_index'])
    if child_adj: _noyau.add(child_adj['orig_index'])
    processed_indices.update(_noyau)

    # Complément propre de l'acl (ex: "lié AUX LOGIQUES DE POUVOIR DES
    # ÉLITES") : rattaché structurellement au même syntagme nominal que le
    # nom racine, il doit faire partie du même bloc O — sinon il finit dans
    # OBL_ALL, rendu après le TAM de la clause plutôt qu'avec son nom
    # (bug trouvé 2026-07-19 : "il existe un frein... lié à X" produisait
    # "frein... bɛ X" au lieu de "frein... X bɛ"). On réutilise TEL QUEL le
    # bâtisseur de wagon générique (step5_obliques.wagon, mêmes règles KG
    # que pour un oblique de fin de clause) puis on récupère sa sortie dans
    # O au lieu de OBL_ALL.
    if acl_tok:
        _obl_arg = next((x for x in T if x.get('dep') == 'obl:arg'
                         and x.get('head_index') == acl_tok['orig_index']
                         and x['orig_index'] not in processed_indices), None)
        if _obl_arg:
            _dep_case = next((x for x in T if x.get('dep') == 'case'
                              and x.get('head_index') == _obl_arg['orig_index']), None)
            _loc_markers = G_kg.get('locative_markers', set())
            _marker_val = (_dep_case.get('bm_marker') or _dep_case.get('bm') or '') if _dep_case else ''
            _lct = 'locative' if _marker_val in _loc_markers else 'simple'
            _len_before = len(m.get('OBL_ALL', []))
            wagon.append(_obl_arg, T, m, processed_indices, G_kg, NX_G,
                        _lct, _marker_val, _dep_case)
            _consumed = set(processed_indices)
            _new = m.get('OBL_ALL', [])[_len_before:]
            del m['OBL_ALL'][_len_before:]
            for _entry in _new:
                m['O'] = j(m['O'], render_obl_entry(_entry, G_kg))

            # Relative éventuellement rattachée à un token du complément
            # qu'on vient d'incorporer (ex: "...des élites QUI VEULENT...") :
            # même logique — repliée dans O si son antécédent fait partie de
            # ce complément, laissée à OBL_ALL sinon (relative sans rapport
            # ailleurs dans la phrase).
            if any(x.get('dep') == 'acl:relcl'
                   and x.get('head_index') in _consumed
                   and x['orig_index'] not in processed_indices
                   for x in T):
                _len_before2 = len(m.get('OBL_ALL', []))
                relcl_post.run(T, tree, m, processed_indices, G_kg, NX_G)
                _new2 = m.get('OBL_ALL', [])[_len_before2:]
                del m['OBL_ALL'][_len_before2:]
                for _entry in _new2:
                    if _entry.get('_HEAD_IDX') in _consumed:
                        m['O'] = j(m['O'], render_obl_entry(_entry, G_kg))
                    else:
                        m['OBL_ALL'].append(_entry)
