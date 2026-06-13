"""
step4_objet/objet_standard.py
Objet standard : obj_tok, _build_genitive_chain, o_chunk,
amod/conj, démonstratif, pluriel, xcomp_adj.
"""
import networkx as nx
from rules.core import j, get_bounded_chunk_tokens, adj_man


def _build_genitive_chain(tok, all_toks, G_kg=None):
    nmod  = next((t for t in all_toks if t.get('dep') == 'nmod'
                  and t.get('head_index') == tok['orig_index']), None)
    amods = [t for t in all_toks if t.get('dep') == 'amod'
             and t.get('head_index') == tok['orig_index'] and t.get('bm')]
    poss  = next((t for t in all_toks if t.get('dep') == 'det'
                  and t.get('role') in ('pronoun', 'possessive')
                  and t.get('head_index') == tok['orig_index']
                  and t.get('bm')), None)
    tok_bm = tok.get('bm') or tok.get('surface') or f"[{tok.get('lemma')}]"
    if poss:
        tok_bm = j(poss.get('bm'), tok_bm)
    _classifiants = [a for a in amods if a['orig_index'] > tok['orig_index']]
    _qualifiants  = [a for a in amods if a['orig_index'] < tok['orig_index']]
    if nmod:
        nmod_bm = _build_genitive_chain(nmod, all_toks, G_kg)
        _relational = (G_kg or {}).get('relational_bms', set())
        _tok_is_rel = tok.get('is_relational', False) or tok_bm in _relational
        _gen = '' if _tok_is_rel else 'ka'
        # Bambara : adjectifs toujours après le nom (qualifiants et classifiants)
        return j(nmod_bm, _gen, tok_bm,
                 *[a.get('bm','') for a in _classifiants],
                 *[a.get('bm','') for a in _qualifiants])
    return j(tok_bm,
             *[a.get('bm','') for a in _classifiants],
             *[a.get('bm','') for a in _qualifiants])


def run(T, tree, m, processed_indices, G_kg, NX_G,
        root_tok, root_noun, has_acl,
        xcomp_adj_tok, clause_type_init, _has_question_mark):

    _has_acl_only   = any(x.get('dep') == 'acl' for x in T)

    # Déjà traité par noun_phrase.py (participe 'acl' sur le nom racine), même si
    # une relative 'acl:relcl' coexiste ailleurs (gérée séparément par step5).
    _acl_on_root = any(x.get('dep') == 'acl'
                       and x.get('head_index') == root_noun['orig_index']
                       for x in T) if root_noun else False
    if root_noun and _has_acl_only and _acl_on_root:
        return

    # ── XCOMP ADJ ─────────────────────────────────────────────────────────────
    if xcomp_adj_tok:
        m['O'] = xcomp_adj_tok.get('bm') or f"[{xcomp_adj_tok.get('lemma')}]"
        _xcomp_advs = [x for x in T if x.get('dep') == 'advmod'
                       and x.get('head_index') == xcomp_adj_tok['orig_index']
                       and x.get('bm')]
        if _xcomp_advs:
            m['O'] = j(m['O'], j(*[x.get('bm') for x in _xcomp_advs]))
            for _xa in _xcomp_advs:
                processed_indices.add(_xa['orig_index'])
        m['O_IS_XCOMP'] = True
        processed_indices.add(xcomp_adj_tok['orig_index'])
        return

    # ── OBJET STANDARD ────────────────────────────────────────────────────────
    if root_noun and has_acl:
        return  # géré ailleurs

    _phrase_interrog = any(
        '?' in str(x.get('surface', ''))
        or (x.get('dep') == 'punct' and str(x.get('surface', '')).strip() == '?')
        for x in T)

    # Exclure les tokens internes à une relative (acl:relcl) : l'objet d'un verbe
    # DANS une relative (ex: 'masse' dans 'qui veulent maintenir la masse…') n'est
    # pas l'objet de la clause principale.
    from rules.core import relcl_subtree_indices
    _relcl_idx = relcl_subtree_indices(T, NX_G)

    obj_tok = next((x for x in T if x.get('dep') in ('obj', 'xcomp')
                    and x.get('pos') in ('NOUN', 'PROPN', 'ADJ')
                    and x.get('orig_index') not in _relcl_idx), None)
    if not obj_tok:
        obj_tok = next((x for x in T if x.get('dep') in ('obj', 'advmod', 'dep')
                        and x.get('role') == 'interrogative' and x.get('bm')), None)
    if not obj_tok and _phrase_interrog:
        obj_tok = next((x for x in T if x.get('dep') == 'dep'
                        and x.get('pos') == 'PROPN' and x.get('bm')), None)

    # Objet PRONOM clitique direct (me/te/le/la/nous/vous/les → n/i/a/.../u).
    # Non capturé par la recherche NOUN/PROPN/ADJ ci-dessus. Exclut les pronoms
    # relatifs/interrogatifs/réflexifs (gérés ailleurs). Si le KG ne fournit pas
    # de bm (le/les collisionnent avec l'article → bm vide), on garde le pronom
    # TEL QUEL (surface) plutôt que de l'ignorer.
    if not obj_tok:
        _pron_obj = next((x for x in T
                          if x.get('dep') == 'obj'
                          and x.get('pos') == 'PRON'
                          and x.get('role') not in ('relative', 'interrogative',
                                                    'reflexive', 'expletive')
                          and x.get('orig_index') not in _relcl_idx
                          and x['orig_index'] not in processed_indices), None)
        if _pron_obj:
            if not _pron_obj.get('bm'):
                _pron_obj['bm'] = _pron_obj.get('surface', '')
            obj_tok = _pron_obj

    if obj_tok and obj_tok.get('role') == 'interrogative':
        _interrog_noun = next((x for x in T
                               if x.get('dep') in ('obl:arg', 'nmod')
                               and x.get('head_index') == obj_tok['orig_index']
                               and x.get('pos') in ('NOUN', 'PROPN')), None)
        if not _interrog_noun:
            _case_d = next((x for x in T if x.get('dep') == 'case'
                            and x.get('head_index') == obj_tok['orig_index']), None)
            if _case_d:
                _interrog_noun = next((x for x in T if x.get('dep') == 'obl:arg'
                                       and x.get('pos') in ('NOUN', 'PROPN')), None)
        if _interrog_noun:
            tree['interrog_noun'] = _interrog_noun
            processed_indices.add(_interrog_noun['orig_index'])

    if not obj_tok and clause_type_init == 'content_question':
        obj_tok = next((x for x in T if x.get('pos') == 'NOUN'
                        and any(d.get('role') == 'interrogative'
                                for d in T if d.get('head_index') == x['orig_index'])), None)
    if not obj_tok and _phrase_interrog:
        # Exclure les sujets grammaticaux (nsubj) du fallback objet
        obj_tok = next((x for x in T
                        if x.get('pos') == 'NOUN'
                        and x.get('dep') not in ('nsubj', 'nsubj:pass')), None)

    # PROPN/NOUN dep='ROOT' secondaire (spaCy double ROOT) → traiter comme objet
    # ex: elle ne parle pas bambara → bambara dep='ROOT' is_root=False
    if not obj_tok:
        _second_root = next((x for x in T
                             if x.get('dep') == 'ROOT'
                             and x.get('pos') in ('NOUN', 'PROPN')
                             and x.get('role') != 'negation'
                             and x.get('orig_index') != (root_tok['orig_index'] if root_tok else -1)
                             and x['orig_index'] not in processed_indices), None)
        if _second_root:
            obj_tok = _second_root

    # nmod direct du ROOT VERB sans sujet → objet impératif
    # ex: Parle bambara → bambara dep='nmod' head=ROOT
    if not obj_tok and root_tok and root_tok.get('pos') == 'VERB':
        _nmod_obj = next((x for x in T
                          if x.get('dep') == 'nmod'
                          and x.get('head_index') == root_tok['orig_index']
                          and x.get('pos') in ('NOUN', 'PROPN')
                          and x['orig_index'] not in processed_indices), None)
        if _nmod_obj:
            obj_tok = _nmod_obj

    # Copule + vrai sujet (Je suis Hawa, mon nom est Hawa) → le PROPN/NOUN racine
    # est le PRÉDICAT d'une ÉQUATIVE (S yé PROPN yé), pas un noun_phrase. On ne
    # transforme donc PAS ce root en objet/noun_phrase dans ce cas. Sans sujet réel
    # (C'est Musa : 'ce' explétif) → comportement inchangé (présentatif/identif).
    _has_cop_o = any(x.get('dep') == 'cop' for x in T)
    _real_subj_o = any(
        x.get('dep') in ('nsubj', 'nsubj:pass')
        and x.get('role') not in ('expletive', 'clitic')
        and x.get('pos') in ('PRON', 'NOUN', 'PROPN')
        and str(x.get('surface', '')).lower().rstrip("'").rstrip('’')
            not in ('ce', 'c', 'ca', 'ça')
        and x.get('orig_index') != (root_tok['orig_index'] if root_tok else -2)
        for x in T)
    if (not obj_tok and root_tok and root_tok.get('pos') in ('NOUN', 'PROPN')
            and not has_acl
            and not (_has_cop_o and _real_subj_o)):
        obj_tok = root_tok
        _has_interrog_det = any(x.get('role') == 'interrogative'
                                and x.get('head_index') == root_tok['orig_index'] for x in T)
        if _has_interrog_det and _has_question_mark:
            tree['clause_type'] = 'content_question'
        elif tree.get('clause_type') not in ('existential_nominal', 'locative'):
            tree['clause_type'] = 'noun_phrase'

    if not obj_tok or obj_tok['orig_index'] in processed_indices:
        return

    o_chunk = get_bounded_chunk_tokens(obj_tok['orig_index'], NX_G, processed_indices)
    print(f"DEBUG o_chunk={[(t.get('surface'), t.get('dep'), t.get('orig_index')) for t in o_chunk]}")

    if tree.get('clause_type') == 'noun_phrase':
        _has_any_nmod = any(x.get('dep') == 'nmod' for x in o_chunk)
        o_chunk = [x for x in o_chunk if x.get('dep') != 'obl:mod']
        if not _has_any_nmod:
            o_chunk = [x for x in o_chunk if x.get('dep') not in ('nmod', 'obl', 'case')]

    o_chunk = sorted([x for x in o_chunk if x.get('pos') not in ('PUNCT', 'SYM')],
                     key=lambda x: x['orig_index'])

    if obj_tok.get('dep') == 'xcomp':
        m['O_IS_XCOMP'] = True

    tete_tok = next((x for x in o_chunk if x == obj_tok or x.get('dep') == 'ROOT'), obj_tok)
    _has_nmod_chain = any(x.get('dep') == 'nmod' for x in o_chunk)
    objet_elements  = []

    if _has_nmod_chain and tete_tok:
        _resolved = _build_genitive_chain(tete_tok, T, G_kg)
        _top_conjs = [x for x in o_chunk if x.get('dep') == 'conj'
                      and x.get('head_index') == tete_tok['orig_index']]
        for _tc in _top_conjs:
            _tc_bm = _tc.get('bm') or f"[{_tc.get('lemma')}]"
            _tc_cc = next((x for x in T if x.get('dep') == 'cc'
                           and x.get('head_index') == tete_tok['orig_index']), None)
            _cc_bm_tc = _tc_cc.get('bm', '') if _tc_cc and _tc_cc.get('bm') else ''
            _resolved = j(_resolved, _cc_bm_tc, _tc_bm)
            processed_indices.add(_tc['orig_index'])
            if _tc_cc: processed_indices.add(_tc_cc['orig_index'])
        objet_elements = [_resolved] if _resolved else []

    elif tete_tok:
        _det_int = next((x for x in o_chunk if x.get('role') == 'interrogative'), None)
        if ((clause_type_init == 'content_question' or _phrase_interrog)
                and _det_int and tete_tok.get('pos') == 'NOUN'):
            nom_val = tete_tok.get('bm') or f"[{tete_tok.get('lemma')}]"
            int_val = (_det_int.get('bm') if _det_int.get('bm')
                       else G_kg.get('interrogative_default', 'jùmɛn'))
            objet_elements = [nom_val, int_val]
        else:
            tete_bm = tete_tok.get('bm') or f"[{tete_tok.get('lemma')}]"
            _global_postpos = []
            _amods      = [x for x in o_chunk if x.get('dep') == 'amod'
                           and x.get('head_index') == tete_tok['orig_index']]
            _postpos_amods = [x for x in _amods if x.get('role') == 'quantifier']
            _other_amods   = [x for x in _amods if x not in _postpos_amods]
            _pre_amods  = [a for a in _other_amods if a['orig_index'] < tete_tok['orig_index']]
            _post_amods = [a for a in _other_amods if a['orig_index'] > tete_tok['orig_index']]

            # En bambara, ADJ toujours après le nom — forme épithète : ADJ+man
            for _a in _post_amods + _pre_amods:
                _a_bm = _a.get('bm') or f"[{_a.get('lemma')}]"
                if _a.get('pos') == 'ADJ':
                    _a_bm = adj_man(_a_bm, is_classifying=_a.get('is_classifying_adj', False))
                tete_bm = j(tete_bm, _a_bm)
                processed_indices.add(_a['orig_index'])

            # Participe passé adjectival (acl, VerbForm=Part) → VERB+len
            # ex: riz cuit → iri tobilen
            _acl_parts = [x for x in o_chunk
                          if x.get('dep') == 'acl'
                          and x.get('head_index') == tete_tok['orig_index']
                          and ('VerbForm=Part' in str(x.get('morph', ''))
                               or 'Tense=Past' in str(x.get('morph', '')))]
            for _acl in _acl_parts:
                _acl_bm = _acl.get('bm') or f"[{_acl.get('lemma')}]"
                _len_form = _acl_bm + 'len' if not _acl_bm.endswith('len') else _acl_bm
                tete_bm = j(tete_bm, _len_form)
                processed_indices.add(_acl['orig_index'])

            # Déterminant possessif (mon/ton/son…) sur la tête de l'objet :
            # mon cahier → n ka kàye (aliénable) / mon père → n fa (inaliénable).
            # 'ka' selon is_relational, comme _build_genitive_chain.
            _poss_obj = next((x for x in T
                              if x.get('dep') == 'det'
                              and x.get('role') in ('pronoun', 'possessive')
                              and x.get('head_index') == tete_tok['orig_index']), None)
            if _poss_obj:
                # Use BM if available; fallback to surface form (ton, mon, son...)
                _poss_bm = _poss_obj.get('bm') or _poss_obj.get('surface', '')
                if _poss_bm:
                    _is_rel  = (tete_tok.get('is_relational', False)
                                or tete_bm in G_kg.get('relational_bms', set()))
                    if _is_rel:
                        tete_bm = j(_poss_bm, tete_bm)
                    else:
                        _gen_mk = G_kg.get('genitive_marker', 'ka') or 'ka'
                        tete_bm = j(_poss_bm, _gen_mk, tete_bm)
                    processed_indices.add(_poss_obj['orig_index'])

            _global_postpos = _postpos_amods

            _amod_indices = {x['orig_index'] for x in _amods} | {tete_tok['orig_index']}
            _root_orig_o = root_tok['orig_index'] if root_tok else -1
            _conjs = [x for x in T
                      if x.get('dep') == 'conj'
                      and x['orig_index'] not in processed_indices
                      and (x.get('head_index') in _amod_indices
                           or (x.get('head_index') == _root_orig_o
                               and x.get('pos') in ('NOUN', 'PROPN')
                               and x['orig_index'] > tete_tok['orig_index']))]
            for _c in _conjs:
                _c_bm = _c.get('bm') or f"[{_c.get('lemma')}]"
                _c_relcl_tok = next((x for x in T if x.get('dep') == 'acl:relcl'
                                     and x.get('head_index') == _c['orig_index']), None)
                if _c_relcl_tok:
                    processed_indices.update(
                        nx.descendants(NX_G, _c_relcl_tok['orig_index']) | {_c_relcl_tok['orig_index']})
                else:
                    _c_amods = [x for x in o_chunk if x.get('dep') == 'amod'
                                and x.get('head_index') == _c['orig_index']]
                    _c_post  = [a for a in _c_amods if a['orig_index'] > _c['orig_index']]
                    _c_pre   = [a for a in _c_amods if a['orig_index'] < _c['orig_index']]
                    for _ca in _c_post:
                        _c_bm = j(_ca.get('bm') or f"[{_ca.get('lemma')}]", _c_bm)
                        processed_indices.add(_ca['orig_index'])
                    for _ca in _c_pre:
                        _c_bm = j(_c_bm, _ca.get('bm') or f"[{_ca.get('lemma')}]")
                        processed_indices.add(_ca['orig_index'])
                    _poss_on_head = next((x for x in T
                                          if x.get('dep') == 'det'
                                          and x.get('role') in ('pronoun', 'possessive')
                                          and x.get('head_index') == tete_tok['orig_index']
                                          and x.get('bm')), None)
                    if _poss_on_head:
                        _poss_bm = _poss_on_head.get('bm', '')
                        _c_bm = (j('n', _c_bm) if _poss_bm == 'n'
                                 else j(_poss_bm, 'ka', _c_bm))
                _cc_tok = next((x for x in T if x.get('dep') == 'cc'
                                and x.get('head_index') == _c['orig_index']), None) or \
                          next((x for x in T if x.get('dep') == 'cc'
                                and x.get('head_index') == tete_tok['orig_index']), None)
                _conj_marker = _cc_tok.get('bm', '') if _cc_tok and _cc_tok.get('bm') else ''
                if _cc_tok: processed_indices.add(_cc_tok['orig_index'])
                tete_bm = j(tete_bm, _conj_marker, _c_bm)
                processed_indices.add(_c['orig_index'])

            for _a in _global_postpos:
                tete_bm = j(tete_bm, _a.get('bm') or f"[{_a.get('lemma')}]")
                processed_indices.add(_a['orig_index'])
            objet_elements = [tete_bm]

    for _ct in o_chunk:
        if _ct.get('dep') in ('nmod', 'det', 'case'):
            processed_indices.add(_ct['orig_index'])

    if not objet_elements:
        objet_elements = [t.get('bm') or f"[{t.get('lemma')}]"
                          for t in o_chunk
                          if t.get('dep') not in ('det', 'case', 'cop', 'aux', 'aux:tense', 'fixed')
                          and t.get('pos') not in ('PUNCT', 'SYM')
                          and str(t.get('surface', '')).strip() not in ("'", "\u2019", "l", "L")]

    _interrog_det = next((x for x in o_chunk if x.get('role') == 'interrogative'
                          and x.get('dep') == 'det'), None)
    if _interrog_det and _interrog_det.get('bm'):
        if _interrog_det.get('bm') not in objet_elements:
            objet_elements.append(_interrog_det.get('bm'))
        processed_indices.add(_interrog_det['orig_index'])

    m['O'] = j(*[x for x in objet_elements if str(x).strip() != 'ni'])

    _is_deictique_pres = (tree.get('clause_type') == 'presentative'
                          and any(t.get('role') == 'deictique'
                                  for t in tree.get('_tokens', [])))
    # Pluriel : is_plural (Number=Plur spaCy) OU déterminant pluriel (les/des).
    # PAS le heuristique surface 'endswith(s)' qui pluralisait à tort les noms
    # invariables en -s (le temps, la fois, le corps → temps/fois/corps).
    _obj_det = next((x for x in T if x.get('dep') == 'det'
                     and x.get('head_index') == obj_tok['orig_index']), None)
    _det_is_plural = bool(_obj_det and (
        'Number=Plur' in str(_obj_det.get('morph', ''))
        or str(_obj_det.get('surface', '')).lower() in ('les', 'des')))
    if ((obj_tok.get('is_plural') or _det_is_plural)
            and not str(m['O']).endswith('w')
            and (tree.get('clause_type') != 'presentative' or _is_deictique_pres)):
        if obj_tok.get('pos') not in ('PRON', 'PROPN'):
            m['O'] = f"{m['O']}w"

    # Nombre cardinal (nummod) → après nom+pluriel en bambara : màlow saba
    if tete_tok:
        _o_nummod = next((x for x in o_chunk
                          if x.get('dep') == 'nummod'
                          and x.get('head_index') == tete_tok['orig_index']
                          and x.get('bm')), None)
        if _o_nummod:
            m['O'] = j(m['O'], _o_nummod.get('bm'))

    processed_indices.update([t['orig_index'] for t in o_chunk])

    # existential_nominal : récupérer tout le sous-arbre obj
    if tree.get('clause_type') == 'existential_nominal':
        _all_desc = nx.descendants(NX_G, obj_tok['orig_index']) | {obj_tok['orig_index']}
        _all_desc -= processed_indices
        _ = sorted([NX_G.nodes[idx]['token'] for idx in _all_desc
                    if idx in NX_G.nodes
                    and NX_G.nodes[idx]['token'].get('pos') not in ('PUNCT', 'SYM')
                    and NX_G.nodes[idx]['token'].get('dep') not in ('det',)
                    and NX_G.nodes[idx]['token'].get('role') not in ('article', 'pronoun')],
                   key=lambda x: x['orig_index'])