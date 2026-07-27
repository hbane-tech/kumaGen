"""
step5_obliques/comitative.py
Oblique comitative : role='comitative' → ni + conj + acl:relcl du conj.
"""
import networkx as nx
from rules.core import j, _resolve_tam, adj_man
from rules.steps.step4_objet.objet_standard import (
    _build_genitive_chain, _collect_genitive_tokens)


def handle(tok_item, T, m, processed_indices, G_kg, NX_G):
    marker_val = G_kg.get('comitative_marker', '') or 'ni'

    _has_nmod_chain = any(t.get('dep') == 'nmod'
                          and t.get('head_index') == tok_item['orig_index']
                          for t in T)

    # Calculer _com_conjs AVANT d'ajouter obl_chunk à processed_indices
    # pour éviter que les conjoncts (dep='conj') soient consommés par le chunk
    _com_conjs = [x for x in T if x.get('dep') == 'conj'
                  and x.get('head_index') == tok_item['orig_index']
                  and x['orig_index'] not in processed_indices]
    _com_conj_idx = {cc['orig_index'] for cc in _com_conjs}

    if _has_nmod_chain:
        head_base = _build_genitive_chain(tok_item, T, G_kg)
        _chain_toks = _collect_genitive_tokens(tok_item, T)
        processed_indices.update(_chain_toks)
    else:
        from rules.core import get_bounded_chunk_tokens
        obl_chunk = get_bounded_chunk_tokens(tok_item['orig_index'], NX_G, processed_indices)
        obl_chunk = [x for x in obl_chunk
                     if x.get('pos') not in ('PUNCT', 'SYM')
                     and str(x.get('surface', '')).strip() not in ('-', '–', '—')
                     and x['orig_index'] not in _com_conj_idx]  # exclure les conjoncts

        head_base = tok_item.get('bm') or f"[{tok_item.get('lemma')}]"
        if tok_item.get('bm_suffix'):
            head_base += tok_item['bm_suffix']

        # Pluriel ("avec ses amis" → tériw, pas téri) : même règle que
        # step5_obliques/wagon.py::append — sans ce garde, un possédé
        # pluriel perdait sa marque 'w' dans cette construction comitative
        # spécifique (bug trouvé 2026-07-20).
        if (tok_item.get('pos') not in ('PROPN', 'PRON')
                and not head_base.startswith('[')
                and (tok_item.get('is_plural') or str(tok_item.get('surface', '')).endswith('s'))
                and not head_base.endswith('w')):
            head_base = f"{head_base}w"

        # Possessif (mon/ton/son…) : 'ka' seulement si le possédé est
        # ALIÉNABLE (is_relational déjà calculé lors de la traduction du
        # token — cf. _detect_relational_noun). Sans ce garde, 'ka' était
        # ajouté aveuglément pour tout possesseur ≠ 'n', y compris pour un
        # possédé inaliénable comme une personne ("ses amis" → PER×PER,
        # inaliénable) (bug trouvé 2026-07-20 : "avec ses amis" → "ni a ka
        # téri yé" au lieu de "ni a téri yé").
        _poss_main = next((x for x in obl_chunk if x.get('dep') == 'det'
                           and x.get('role') in ('pronoun', 'possessive')
                           and x.get('bm')), None)
        if _poss_main:
            _poss_bm = _poss_main.get('bm', '')
            _tok_is_rel = tok_item.get('is_relational', False)
            # Aucune personne (y compris 1sg) ne dispense de 'ka' quand le
            # possédé est ALIÉNABLE — seule l'inaliénabilité du possédé
            # supprime le marqueur (cf. _classify_noun_is_object : "n ka
            # bìlakun" = mes devoirs, aliénable même à la 1sg). Une
            # précédente version court-circuitait 'ka' dès que le
            # possesseur était la 1sg, quelle que soit l'aliénabilité (bug
            # trouvé 2026-07-20 : "avec ma famille", pourtant aliénable/
            # collectif, perdait 'ka' → "ni n kɔ́lɛ yé" au lieu de "ni n ka
            # kɔ́lɛ yé").
            if _tok_is_rel:
                head_base = j(_poss_bm, head_base)
            else:
                head_base = j(_poss_bm, 'ka', head_base)

        # Adjectifs qualifiants (amod) : "dentition complète" → dákolon [adj]man
        for _a in sorted([x for x in obl_chunk if x.get('dep') == 'amod'],
                         key=lambda x: x['orig_index']):
            _a_bm = _a.get('bm') or f"[{_a.get('lemma')}]"
            if _a.get('pos') == 'ADJ':
                _a_bm = adj_man(_a_bm, is_classifying=_a.get('is_classifying_adj', False))
            head_base = j(head_base, _a_bm)

        # Nombre cardinal (nummod) : "avec trois enfants" → denmisɛnw saba
        _nummod = next((x for x in obl_chunk
                        if x.get('dep') == 'nummod' and x.get('bm')), None)
        if _nummod:
            head_base = j(head_base, _nummod.get('bm'))

        # Marquer les tokens du chunk (hors conjoncts) comme traités
        for _ct in obl_chunk:
            processed_indices.add(_ct['orig_index'])

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
            _relcl_is_statif = (
                _cc_relcl.get('is_valeur') is True
                or _cc_relcl.get('is_statif') is True)
            if _relcl_self_adj and _relcl_is_statif:
                # Statif : N yé min ADJlen dòn / dòn|tɛ selon négation de la relative
                _relcl_neg = any(
                    x.get('dep') in ('advmod', 'mark')
                    and x.get('role') == 'negation'
                    and x.get('head_index') == _cc_relcl['orig_index']
                    for x in T)
                _statif_neg = G_kg.get('statif_neg_marker', '') or 'tɛ'
                _statif_pos = G_kg.get('statif_pos_marker', '') or 'dòn'
                _assert = _statif_neg if _relcl_neg else _statif_pos
                _equative = G_kg.get('equative_marker', '') or 'yé'
                _cc_bm = j(_cc_bm, _equative, _rel_marker, _rel_v_bm, _assert)
            elif _relcl_self_adj or (_rel_adj and not _rel_v_bm):
                # Qualitatif : N min ka ADJ
                _rel_tam = G_kg.get('genitive_marker', '') or 'ka'
                _cc_bm = j(_cc_bm, _rel_marker, _rel_tam, _rel_v_bm or _rel_adj_bm)
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

    # Construire ni HEAD [ni HEAD2 ...] yé
    dep_case = next((x for x in T if x.get('dep') == 'case'
                     and x.get('head_index') == tok_item['orig_index']), None)
    # Utiliser le bm du case token s'il existe (ex: avec→ni), sinon KG default
    _actual_marker = (dep_case.get('bm') or marker_val) if dep_case else marker_val
    if not _com_conjs:
        # Cas simple : ni head yé
        head_base = j(_actual_marker, head_base, 'yé')
    else:
        # Cas conjoint : ni head1 ni head2 [yé]
        # Le 'ni' initial est toujours nécessaire.
        # Le 'yé' final est ajouté sauf si le dernier conj porte déjà une
        # acl:relcl statif dont la forme intègre son propre 'yé' équatif.
        _last_cc_relcl = next((x for x in T
                               if x.get('dep') == 'acl:relcl'
                               and x.get('head_index') == _com_conjs[-1]['orig_index']), None)
        head_base = j(_actual_marker, head_base)
        if not _last_cc_relcl:
            head_base = j(head_base, 'yé')
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
