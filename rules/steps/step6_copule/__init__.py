"""rules/steps/step6_copule/__init__.py
Extraction structurelle uniquement : copule → slots S / O / V / QUAL / OBL_ALL / ADV.
Aucune décision de clause_type ou TAM — délégué à kg_gateway après step7.
"""
from rules.core import j, apply_affixes, _is_copula


def _extract_cop_tense(tok):
    t = tok.get('tense', '')
    if not t:
        m = str(tok.get('morph', ''))
        if 'Tense=Fut'  in m: return 'fut'
        if 'Tense=Imp'  in m: return 'hab'
        if 'Tense=Past' in m: return 'past'
    return t


def run(T, tree, m, processed_indices, G_kg, root_tok,
        _has_expletive, aux_tense_tok, clause_type_init):

    # ── 0. Transitivité structurelle (extraction, pas une règle) ─────────
    # Utilisé par le bloc F6 du renderer pour le résultatif passé intransitif
    # Guard head_index : ne regarder QUE les auxiliaires directs du ROOT principal.
    # Sans ça, un copule dans un ccomp subordonné ("elle a dit que Musa est parti")
    # déclenchait _has_aux_cop=True → is_transitive=False → fɔla au lieu de yé fɔ.
    _root_idx_s6 = root_tok['orig_index'] if root_tok else -1
    _has_obj      = any(x.get('dep') == 'obj' for x in T)
    _has_aux_pass = any(x.get('dep') in ('aux:pass',) and x.get('pos') == 'AUX'
                        and x.get('head_index') == _root_idx_s6 for x in T)
    _has_aux_cop  = any(x.get('dep') in ('aux', 'aux:tense') and _is_copula(x)
                        and x.get('head_index') == _root_idx_s6 for x in T)
    if 'is_transitive' not in tree:
        tree['is_transitive'] = True if _has_obj else (
            False if (_has_aux_pass or _has_aux_cop) else True)

    # ── 1. Participe simultané (-tɔ) : extraction pure par rôle ──────────
    _part_to = next((x for x in T
                     if x.get('role') == 'participial_to' and x.get('bm')), None)
    if _part_to:
        _s = (next((x.get('bm') for x in T if x.get('dep') == 'nsubj'), None)
              or next((x.get('bm') for x in T if x.get('role') == 'pronoun'), None)
              or '')
        _participial_to_sfx = G_kg.get('participial_to_suffix', 'tɔ') or 'tɔ'
        m['ADV'] = j(_s, _part_to.get('bm', '') + _participial_to_sfx)
        processed_indices.add(_part_to['orig_index'])

    _root_idx = root_tok['orig_index'] if root_tok else -1
    _root_bm  = (root_tok.get('bm') or f"[{root_tok.get('lemma', '')}]") if root_tok else ''
    _root_pos = root_tok.get('pos', '') if root_tok else ''
    _morph    = str(root_tok.get('morph', '')) if root_tok else ''

    # ── 2. Trouver la copule dans la clause principale ────────────────────
    copula_tok = next(
        (x for x in T
         if x.get('dep') in ('cop', 'aux:pass') and _is_copula(x)
         and not any(y.get('orig_index') == x.get('head_index')
                     and y.get('dep') in ('ccomp', 'advcl', 'acl:relcl', 'xcomp', 'parataxis')
                     for y in T)),
        None)

    # Être ROOT lui-même = copule (pas de cop séparée, pas d'interrogatif ni locatif)
    if (not copula_tok and root_tok and _is_copula(root_tok)
            and _root_pos in ('VERB', 'AUX')
            and not any(x.get('role') == 'interrogative' for x in T)
            and not any(x.get('dep') == 'case' and x.get('role') == 'locative' for x in T)
            and not any(x.get('is_loc') for x in T)):
        copula_tok = root_tok

    # aux:pass sans être-cop (passif synthétique)
    _aux_pass = next((x for x in T
                      if x.get('dep') == 'aux:pass'
                      and x.get('head_index') == _root_idx), None)

    if not copula_tok and not _aux_pass:
        return

    # ── 3. Signaux structurels de la copule ──────────────────────────────
    if copula_tok:
        tree['cop_tense'] = _extract_cop_tense(copula_tok)
        tree['_has_cop']  = True
        if copula_tok != root_tok:
            processed_indices.add(copula_tok['orig_index'])
    if _aux_pass:
        processed_indices.add(_aux_pass['orig_index'])

    # ── 4. Passif (être + participe passé) ────────────────────────────────
    # Slot : m['V'] = bm du verbe passif (kg_gateway ajoutera -len ou -ra)
    _is_pass_participle = (root_tok
                           and root_tok.get('is_passive')
                           and ('VerbForm=Part' in _morph or 'Voice=Pass' in _morph))
    _is_aux_pass_on_root = bool(_aux_pass and _aux_pass.get('head_index') == _root_idx)

    if _is_pass_participle or _is_aux_pass_on_root:
        # Participe passif sans traduction lexicale (compound Sense non
        # trouvé, ex: "faire" seul face à "faire peur"/"faire voeu"...) :
        # repli sur le verbe support générique (kɛ = "se faire, arriver",
        # sens naturel du passif impersonnel "ce fut fait" = "cela arriva").
        _root_bm_pass = (G_kg.get('coord_action_suffix', 'kɛ') or 'kɛ') \
            if _root_bm.startswith('[') else _root_bm
        m['V'] = _root_bm_pass
        tree['_is_passive']    = True
        tree['is_participe_passe'] = True
        tree['participe_bm']   = _root_bm_pass
        processed_indices.add(_root_idx)
        return

    # ── 5. Statif / potential : m['QUAL'] = base (sans suffixe) ──────────
    # kg_gateway ajoute 'len' (statif) ou 'ta' (potential) selon clause_type
    _is_statif = bool(root_tok and (
        root_tok.get('is_statif')
        or root_tok.get('is_potential')
        or root_tok.get('statif_root')
        or root_tok.get('semantic_class') in ('statif', 'state', 'physical_state', 'color')
        or ('VerbForm=Part' in _morph and root_tok.get('is_participe_passe'))))

    if _is_statif and _root_pos == 'ADJ':
        _base = root_tok.get('statif_root') or _root_bm
        m['QUAL'] = _base
        tree['_is_statif'] = True
        root_tok['is_statif'] = True
        processed_indices.add(_root_idx)
        return

    # ── 5b. Privative ROOT : "ce légume est sans cuisson" → [légume] [cuisson]tan dòn ──
    # Détecté après step5 (du fait qu'on a exclu privative ROOT de step5_obliques)
    _priv_case = next((x for x in T if x.get('dep') == 'case'
                       and x.get('role') == 'privative'
                       and x.get('head_index') == _root_idx), None)
    if _priv_case and root_tok and not _is_statif:
        # Construire le prédicat privatif avec suffixe -tan (ou -bali pour verbes)
        # Suffixe collé sans espace (c'est une morphologie bambara)
        from rules.steps.step5_obliques.privatif import _get_marker
        _priv_marker = _get_marker(root_tok, T, G_kg)
        m['O'] = (_root_bm + _priv_marker) if _priv_marker else _root_bm

        # Cherche le sujet : peut être nsubj du ROOT ou de la copule (être)
        # Ex: "Ce légume est sans cuisson" → sujet='légume' est nsubj de 'être', pas de 'cuisson'
        _subj_priv = next((x for x in T
                          if x.get('dep') == 'nsubj'
                          and (x.get('head_index') == _root_idx
                               or (copula_tok and x.get('head_index') == copula_tok.get('orig_index')))), None)
        if _subj_priv and _subj_priv.get('bm'):
            _subj_priv_bm = _subj_priv.get('bm')
            _subj_priv_demo = next((x for x in T
                                    if x.get('dep') == 'det'
                                    and x.get('role') == 'demonstrative'
                                    and x.get('head_index') == _subj_priv['orig_index']), None)
            if _subj_priv_demo:
                _subj_priv_bm = apply_affixes(
                    _subj_priv_demo.get('bm'), _subj_priv_bm, _subj_priv_demo.get('bm_suffix'))
                processed_indices.add(_subj_priv_demo['orig_index'])
            m['S'] = _subj_priv_bm
            processed_indices.add(_subj_priv['orig_index'])

        tree['_is_privative'] = True
        tree['_has_cop']  = True
        # Privatif utilise la copule équative du KG
        # Rendu: "[S] [O_privatif] equative_marker"
        m['V'] = G_kg.get('equative_marker', 'yé') or 'yé'
        processed_indices.add(_root_idx)
        if _priv_case:
            processed_indices.add(_priv_case['orig_index'])
        return

    # ── 6. Prédicat nominal / adjectival ──────────────────────────────────
    if root_tok and _root_pos in ('NOUN', 'ADJ', 'PROPN'):
        # Advmods du prédicat ADJ → qualitative (ka bòn, trop grand…)
        _advs = [x for x in T
                 if x.get('dep') == 'advmod'
                 and x.get('head_index') == _root_idx
                 and x.get('bm')
                 and x['orig_index'] not in processed_indices]
        if _advs:
            m['QUAL'] = _root_bm
            for _a in _advs:
                m['QUAL'] = j(m['QUAL'], _a.get('bm'))
                processed_indices.add(_a['orig_index'])
        elif not m.get('O'):
            # step4_objet a peut-être déjà extrait m['O'] (avec possessif) → ne pas écraser.
            # Possessif DET sur le prédicat (mon ami, ta maison…)
            _poss6 = next((x for x in T
                           if x.get('dep') == 'det'
                           and x.get('role') in ('pronoun', 'possessive')
                           and x.get('head_index') == _root_idx), None)
            if _poss6 and _poss6['orig_index'] not in processed_indices:
                _pb6  = _poss6.get('bm', '')
                _p1sg = G_kg.get('pron_1sg', '')
                _gm   = G_kg.get('genitive_marker', '')
                _obj_bm = j(_p1sg, _root_bm) if _pb6 == _p1sg else j(_pb6, _gm, _root_bm)
                processed_indices.add(_poss6['orig_index'])
            else:
                _poss6 = None
                _obj_bm = _root_bm
            # Pluriel du nom prédicat (frères → bálimakɛw)
            _plur_s6 = G_kg.get('plural_noun_suffix', '') or 'w'
            if (root_tok and root_tok.get('is_plural')
                    and _obj_bm and not _obj_bm.endswith(_plur_s6)
                    and root_tok.get('pos') not in ('PRON', 'PROPN')):
                _obj_bm += _plur_s6
            m['O'] = _obj_bm

        # Pluriel sur m['O'] pré-posé par step4 (frères → bálimakɛ → bálimakɛw)
        if m.get('O') and root_tok and root_tok.get('is_plural'):
            _plur_s6b = G_kg.get('plural_noun_suffix', '') or 'w'
            if (not m['O'].endswith(_plur_s6b)
                    and root_tok.get('pos') not in ('PRON', 'PROPN')):
                m['O'] = m['O'] + _plur_s6b

        # Coordonnés du ROOT NOUN (et sœurs, et amis, …) : exécuté que m['O']
        # ait été posé ici ou par step4 (qui ne traite pas les conj du ROOT NOUN).
        if m.get('O'):
            _poss6_conj = next((x for x in T
                                if x.get('dep') == 'det'
                                and x.get('role') in ('pronoun', 'possessive')
                                and x.get('head_index') == _root_idx), None)
            for _rc6 in T:
                if (_rc6.get('dep') == 'conj'
                        and _rc6.get('head_index') == _root_idx
                        and _rc6.get('bm')
                        and _rc6['orig_index'] not in processed_indices):
                    _cc6 = (next((x for x in T if x.get('dep') == 'cc'
                                  and x.get('head_index') == _root_idx), None)
                            or next((x for x in T if x.get('dep') == 'cc'
                                     and x.get('head_index') == _rc6['orig_index']), None))
                    _rc6_bm = _rc6.get('bm', '')
                    # Pluriel du conjoint (sœurs → bálimamuso+w)
                    _plur_rc6 = G_kg.get('plural_noun_suffix', '') or 'w'
                    if (_rc6.get('is_plural') and _rc6_bm
                            and not _rc6_bm.endswith(_plur_rc6)
                            and _rc6.get('pos') not in ('PRON', 'PROPN')):
                        _rc6_bm += _plur_rc6
                    if _poss6_conj:
                        _pb6c  = _poss6_conj.get('bm', '')
                        _p1sg6 = G_kg.get('pron_1sg', '')
                        _gm6   = G_kg.get('genitive_marker', '')
                        _rc6_bm = (j(_p1sg6, _rc6_bm) if _pb6c == _p1sg6
                                   else j(_pb6c, _gm6, _rc6_bm))
                    _conj_mk6 = (_cc6.get('bm', '') if _cc6 and _cc6.get('bm') else
                                 G_kg.get('comitative_marker', ''))
                    m['O'] = j(m['O'], _conj_mk6, _rc6_bm)
                    processed_indices.add(_rc6['orig_index'])
                    if _cc6:
                        processed_indices.add(_cc6['orig_index'])
        processed_indices.add(_root_idx)

    # ADV prédicatif (locatif ici/là → OBL_ALL ; qualitatif loin/proche → QUAL)
    elif root_tok and _root_pos == 'ADV':
        if root_tok.get('is_loc') or root_tok.get('role') == 'locative':
            m['OBL_ALL'].append({
                'HEAD': _root_bm, 'MARKER': '',
                'local_clause_type': 'locative',
                'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
                'DEP_TYPE': '', 'COMPOUND_IS_QUANTIFIER': False,
                'MARKER_IS_PREFIX': False,
            })
        else:
            m['QUAL'] = _root_bm
        processed_indices.add(_root_idx)

    # Être ROOT : chercher l'attribut parmi les dépendants directs
    elif copula_tok and copula_tok == root_tok:
        _attr = next(
            (x for x in T
             if x.get('dep') in ('attr', 'xcomp', 'conj', 'appos')
             and x.get('pos') in ('NOUN', 'PROPN', 'ADJ', 'PRON')
             and x['orig_index'] != _root_idx
             and x['orig_index'] not in processed_indices),
            None)
        if _attr:
            _attr_bm = _attr.get('bm') or f"[{_attr.get('lemma', '')}]"
            # Possessif sur l'attribut
            _poss = next((x for x in T
                          if x.get('dep') == 'det'
                          and x.get('role') in ('pronoun', 'possessive')
                          and x.get('head_index') == _attr['orig_index']), None)
            if _poss:
                _pb   = _poss.get('bm', '')
                _p1sg = G_kg.get('pron_1sg', '')
                _gm   = G_kg.get('genitive_marker', '')
                _attr_bm = j(_p1sg, _attr_bm) if _pb == _p1sg else j(_pb, _gm, _attr_bm)
                processed_indices.add(_poss['orig_index'])
            # Pluriel
            if (_attr.get('is_plural') or str(_attr.get('surface', '')).endswith('s')):
                if not _attr_bm.endswith('w'):
                    _attr_bm += 'w'
            # Coordonnés de l'attribut
            for _rc in T:
                if (_rc.get('dep') == 'conj'
                        and _rc.get('head_index') == _attr['orig_index']
                        and _rc.get('bm')
                        and _rc['orig_index'] not in processed_indices):
                    _cc = next((x for x in T if x.get('dep') == 'cc'
                                and x.get('head_index') == _attr['orig_index']), None)
                    _rc_bm = _rc.get('bm', '')
                    if _poss:
                        _pb2 = _poss.get('bm', '')
                        _rc_bm = j('n', _rc_bm) if _pb2 == 'n' else j(_pb2, 'ka', _rc_bm)
                    _attr_bm = j(_attr_bm, _cc.get('bm', '') if _cc else '', _rc_bm)
                    processed_indices.add(_rc['orig_index'])
                    if _cc:
                        processed_indices.add(_cc['orig_index'])
            m['O'] = _attr_bm
            processed_indices.add(_attr['orig_index'])

    # ── 7. Locatif (advmod/obl locatif non encore traité) ─────────────────
    _loc = next(
        (t for t in T
         if (t.get('role') == 'locative' or t.get('is_loc'))
         and t.get('dep') in ('advmod', 'obl', 'obl:mod', 'obl:arg')
         and t['orig_index'] not in processed_indices),
        None)
    if _loc:
        _loc_bm = _loc.get('bm', '')
        if _loc_bm:
            m['OBL_ALL'].append({
                'HEAD': _loc_bm, 'MARKER': '',
                'local_clause_type': 'locative',
                'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
                'DEP_TYPE': '', 'COMPOUND_IS_QUANTIFIER': False,
                'MARKER_IS_PREFIX': False,
            })
            processed_indices.add(_loc['orig_index'])

    # ── 8. Sujet expletif (c'est…, il y a…) ──────────────────────────────
    if not m.get('S') and _has_expletive:
        _expl = next((x for x in T if x.get('role') == 'expletive' and x.get('bm')), None)
        if _expl:
            m['S'] = _expl.get('bm')
            processed_indices.add(_expl['orig_index'])
