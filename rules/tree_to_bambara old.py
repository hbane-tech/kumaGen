"""
rules/tree_to_bambara.py
Convertit un tree dict en string bambara selon clause_type.
"""
from rules.core import j, _GENITIVE_FALLBACK, _resolve_tam, INTRANS_SC

def tree_to_bambara(tree, G=None, grammar=None):
    G   = grammar or G or {}
    m   = tree['main']
    ct  = tree['clause_type']
    S   = m.get('S', '') or ''
    O   = m.get('O', '') or ''
    V   = m.get('V', '') or ''
    V_ACT = m.get('V_ACTION', '') or ''
    V_SUF = m.get('V_SUFFIX', '') or ''
    ADV   = m.get('ADV', '')   or ''
    neg   = tree.get('neg', False)
    tn    = tree.get('tense', 'pres')

    tam_val = tree.get('tam', '')
    if not tam_val or tam_val.strip() == '':
        tam_val = 'bɛ'
    TAM = tam_val

    # Ccomp
    _ccomp_data = m.get('CCOMP')
    _ccomp_str  = ''
    if isinstance(_ccomp_data, dict) and _ccomp_data.get('O'):
        _ko        = G.get('reported_intro', 'ko') or 'ko'
        _ccomp_str = j(_ko, _ccomp_data.get('S', ''),
                       _ccomp_data.get('tam', 'yé'),
                       _ccomp_data.get('O', ''),
                       _ccomp_data.get('tam', 'yé'))

    # Wagons obliques
    obl_strings = []
    for c in (m.get('OBL_ALL') or []):
        comp   = c.get('COMPOUND', '')
        head   = c.get('HEAD', '')
        mod    = c.get('MOD', '')
        pref   = c.get('DEM_PREF', '')
        suff   = c.get('DEM_SUFF', '')
        marker = c.get('MARKER', '')
        lct    = c.get('local_clause_type', 'simple')
        gen_marker = G.get('genitive_marker', '') or _GENITIVE_FALLBACK
        if comp:
            if c.get('COMPOUND_IS_QUANTIFIER'):
                noun_base = j(head, comp)
            # elif lct in ('locative', 'temporal'):
            #     noun_base = j(comp, head)
            elif lct in ('locative', 'temporal'):
                if pref or suff:
                    # nin + [compound + head] + in
                    noun_base = j(pref, comp, head, suff)
                    pref = ''
                    suff = ''
                else:
                    noun_base = j(comp, head)
            else:
                noun_base = j(comp, gen_marker, head)
        else:
            noun_base = head
        if mod:
            noun_base = j(noun_base, mod)
        full_chunk = j(pref, noun_base, suff) if (pref or suff) else noun_base
        if marker:
            if (lct == 'temporal'
                    and (c.get('MARKER_IS_PREFIX')
                         or str(marker).strip().lower() in ('kabini', "k'an bɔ"))):
                full_chunk = j(marker, full_chunk)
            elif lct == 'privative' and marker in ('tan', 'bali'):
                full_chunk = full_chunk + marker
            else:
                full_chunk = j(full_chunk, marker)
        obl_strings.append(full_chunk)

    # F6 : past intransitif
    _o_is_xcomp = m.get('O_IS_XCOMP', False)
    if tn == 'past' and not tree.get('is_transitive', True) and not _o_is_xcomp:
        if neg:
            # Négatif résultatif : S ma V (TAM='ma' devant le verbe nu)
            # ex: o ma [cuit]  — pas de suffixe -ra
            TAM = 'ma'
        else:
            # Affirmatif résultatif : S V+ra (TAM vide)
            # ex: o [cuit]ra
            if V:
                _v_parts = V.split(' ')
                _root_v  = _v_parts[0]
                _rest_v  = _v_parts[1:]
                if not _root_v.endswith('ra') and not _root_v.endswith('na'):
                    _root_v += 'na' if _root_v.endswith('n') else 'ra'
                V = j(_root_v, *_rest_v)
            TAM = ''
    else:
        TAM = tam_val
    print(f"DEBUG F6: V={V!r}, TAM={TAM!r}, S={S!r}")
    # Affichage slots
    print('\n  📦 SLOTS STRUCTURELS FINAUX :')
    if S:        print(f'     [ S     ] → {S}')
    if TAM and ct != 'noun_phrase':
                 print(f'     [ TAM   ] → {TAM}')
    if O:        print(f'     [ O     ] → {O}')
    if V:        print(f'     [ V     ] → {V}')
    if V_ACT:    print(f'     [ V_ACT ] → {V_ACT}')
    if ADV:      print(f'     [ ADV   ] → {ADV}')
    if V_SUF:    print(f'    [V_SUF  ] -> {V_SUF}')
    if _ccomp_str: print(f'    [ccomp  ] -> {_ccomp_str}')
    for _i, _xv in enumerate(obl_strings):
        lct = (m['OBL_ALL'][_i].get('local_clause_type', '')
               if _i < len(m.get('OBL_ALL', [])) else '')
        print(f'     [ X{_i+1:<3d}   ] → {_xv}  ({lct})')
    print(f"  🏷️  clause_type = {ct}")

    result = ''

    # ── OVERRIDE identificatoire / présentatif-valeur ──────────────────
    # C'est Moussa → Moussa dòn  (PROPN ROOT + pas vrai sujet)
    # C'est vrai   → bɛ́rɛ dòn   (ADJ is_valeur + expletif)
    _tokens_ref = tree.get('_tokens', [])
    # Vrai sujet = PRON personnel (je/tu/il...) — pas ce/ça, pas NOUN
    # Pour les résultatifs, NOUN sujet (riz, pain) → repris par 'o'
    _has_real_subj_ttb = any(
        t.get('dep') in ('nsubj', 'nsubj:pass')
        and t.get('role') not in ('expletive', 'clitic')
        and t.get('pos') == 'PRON'
        and str(t.get('surface', '')).lower().rstrip("'").rstrip('\u2019')
        not in ('ce', 'c', 'ca', 'ça')
        for t in _tokens_ref)
    _has_expletif_ttb = any(
        t.get('role') == 'expletive'
        or t.get('dep') in ('expl:subj', 'expl:comp')
        for t in _tokens_ref)

    # Cas -1 : NOUN ROOT + conj NOUN + cop + ce démonstratif → présentatif ni
    # Ce sont mes frères et sœurs → n bálimakɛ ni n dúsukɛ dòn
    _ce_subj = any(
        str(t.get('surface', '')).lower().rstrip("'").rstrip('\u2019') in ('ce', 'c')
        and t.get('dep') in ('nsubj', 'expl:subj', 'expl')
        for t in _tokens_ref)
    _root_noun_ttb = next((t for t in _tokens_ref
                           if t.get('dep') == 'ROOT'
                           and t.get('pos') in ('NOUN', 'PROPN')
                           and t.get('bm')), None)
    _conj_noun_ttb = next((t for t in _tokens_ref
                           if t.get('dep') == 'conj'
                           and t.get('pos') in ('NOUN', 'PROPN')
                           and t.get('bm')), None)
    _has_cop_ce = any(t.get('dep') == 'cop' for t in _tokens_ref)
    if _ce_subj and _root_noun_ttb and _conj_noun_ttb and _has_cop_ce:
        _poss_root = next((t for t in _tokens_ref
                           if t.get('dep') == 'det'
                           and t.get('role') in ('pronoun', 'possessive')
                           and t.get('head_index') == _root_noun_ttb['orig_index']), None)
        _poss_conj = next((t for t in _tokens_ref
                           if t.get('dep') == 'det'
                           and t.get('role') in ('pronoun', 'possessive')
                           and t.get('head_index') == _conj_noun_ttb['orig_index']), None)
        _r_bm = _root_noun_ttb.get('bm', '')
        _c_bm = _conj_noun_ttb.get('bm', '')
        if _poss_root:
            _pb = _poss_root.get('bm', '')
            _r_bm = j('n', _r_bm) if _pb == 'n' else j(_pb, _r_bm)
        if _poss_conj:
            _pb2 = _poss_conj.get('bm', '')
            _c_bm = j('n', _c_bm) if _pb2 == 'n' else j(_pb2, _c_bm)
        result = j(_r_bm, 'ni', _c_bm, 'dòn')
        print(f"  ✂️  Clause 1 -> '{result}'")
        return result.strip()

    # Cas -2 : PRON sujet + NOUN conj + cop futur → équatif futur
    # il ne sera pas président → a tɛ na [être] pèresidan yé
    _cop_fut_ref = next((t for t in _tokens_ref
                         if t.get('dep') in ('cop', 'aux', 'aux:tense')
                         and t.get('role') in ('copula', 'auxiliary', 'content', '')
                         and (t.get('tense') == 'fut'
                              or 'Tense=Fut' in str(t.get('morph', ''))
                              or tree.get('cop_tense') == 'fut')), None)
    # NOUN ROOT avec cop futur → aussi éligible (ex: je ne serai pas professeur)
    _noun_conj_ref = next((t for t in _tokens_ref
                           if t.get('dep') in ('conj', 'attr', 'appos', 'ROOT')
                           and t.get('pos') in ('NOUN', 'PROPN', 'ADJ')
                           and t.get('bm')
                           and not t.get('is_root', False)), None)
    # Fallback : si tous les NOUN sont is_root, prendre quand même le ROOT NOUN
    if not _noun_conj_ref and _cop_fut_ref:
        _noun_conj_ref = next((t for t in _tokens_ref
                               if t.get('dep') == 'ROOT'
                               and t.get('pos') in ('NOUN', 'PROPN')
                               and t.get('bm')), None)
    # Aussi déclencher via tree['cop_tense'] stocké par step6_copule
    if not _cop_fut_ref and tree.get('cop_tense') == 'fut':
        _cop_fut_ref = {'dep': 'cop', 'tense': 'fut',
                        'bm': tree.get('cop_bm', ''),
                        'lemma': tree.get('cop_lemma', 'être')}
    if _cop_fut_ref and _noun_conj_ref and S:
        _cop_bm_ref = _cop_fut_ref.get('bm') or f"[{_cop_fut_ref.get('lemma', 'être')}]"
        # Utiliser O si déjà construit (inclut amod), sinon bm du token
        _obj_bm_ref = O if O else _noun_conj_ref.get('bm', '')
        _tam_ref = 'tɛ na' if neg else 'bɛ na'
        result = j(S, _tam_ref, _cop_bm_ref, _obj_bm_ref, 'yé')
        print(f"  ✂️  Clause 1 -> '{result}'")
        return result.strip()

    # Cas 0 : NOUN/PROPN conj + case comitative + cop → présentatif ni
    # Je suis avec mon mari → n ni n fúrucɛ dòn
    _com_case = any(t.get('dep') == 'case' and t.get('role') == 'comitative'
                    for t in _tokens_ref)
    _com_conj_ttb = next((t for t in _tokens_ref
                          if t.get('dep') in ('conj', 'nmod')
                          and t.get('pos') in ('NOUN', 'PROPN')
                          and t.get('bm')
                          and _com_case), None)
    # S non-vide suffit comme preuve de sujet (même si dep='ROOT')
    if (_com_conj_ttb and _com_case and S
            and not any(t.get('dep') == 'obj' for t in _tokens_ref)):
        # Utiliser O du slot si déjà construit (chaîne génitive complète)
        # Sinon fallback sur le premier nmod/conj trouvé
        if O:
            _companion = O
        else:
            _c_bm = _com_conj_ttb.get('bm') or _com_conj_ttb.get('surface', '')
            _poss_c = next((t for t in _tokens_ref
                            if t.get('dep') == 'det'
                            and t.get('role') in ('pronoun', 'possessive')
                            and t.get('head_index') == _com_conj_ttb['orig_index']), None)
            if _poss_c:
                _pb = _poss_c.get('bm', '')
                _c_bm = j('n', _c_bm) if _pb == 'n' else j(_pb, 'ka', _c_bm)
            _companion = _c_bm
        result = j(S, 'ni', _companion, 'dòn')
        print(f"  ✂️  Clause 1 -> '{result}'")
        return result.strip()

    # Cas 1 : PROPN ROOT ou PROPN conj+cop → identificatoire/équatif
    # C'est Moussa → Moussa dòn
    # Je ne suis pas Hawa → n tɛ Hawa yé
    _propn_root = next((t for t in _tokens_ref
                        if t.get('pos') == 'PROPN'
                        and t.get('dep') == 'ROOT'), None)
    # PROPN conj avec cop → équatif (Je suis/ne suis pas Hawa)
    _propn_conj = next((t for t in _tokens_ref
                        if t.get('pos') == 'PROPN'
                        and t.get('dep') in ('conj', 'attr', 'xcomp')
                        and any(x.get('dep') == 'cop' for x in _tokens_ref)), None)
    if _propn_conj and S:
        _propn_bm = _propn_conj.get('bm') or _propn_conj.get('surface', '')
        if neg:
            result = j(S, 'tɛ', _propn_bm, 'yé')
        else:
            result = j(S, 'yé', _propn_bm, 'yé')
        print(f"  ✂️  Clause 1 -> '{result}'")
        return result.strip()
    if _propn_root and not _has_real_subj_ttb:
        _propn_bm = _propn_root.get('bm') or _propn_root.get('surface', '')
        result = j(_propn_bm, 'tɛ' if neg else 'dòn')
        print(f"  ✂️  Clause 1 -> '{result}'")
        return result.strip()

    # Cas 2 : ADJ is_valeur + expletif + pas vrai sujet → présentatif
    # C'est vrai → bɛ́rɛ dòn  /  ce n'est pas vrai → bɛ́rɛ tɛ
    # Exclure les participes (is_participe_passe) → traités cas 3
    _valeur_adj_ttb = next((t for t in _tokens_ref
                            if t.get('pos') == 'ADJ'
                            and t.get('is_valeur') is True
                            and not t.get('is_participe_passe')
                            and not t.get('is_statif')
                            and t.get('bm')), None)
    if _valeur_adj_ttb and _has_expletif_ttb and not _has_real_subj_ttb:
        _val_bm = _valeur_adj_ttb.get('bm')
        result = j(_val_bm, 'tɛ' if neg else 'dòn')
        print(f"  ✂️  Clause 1 -> '{result}'")
        return result.strip()

    # Cas 3 : ADJ is_participe_passe + cop → résultatif
    # Sujet repris par 'o' (anaphorique bambara)
    # Le riz est cuit    → o [cuit]ra
    # c'est cuit         → o [cuit]ra
    # ce n'est pas cuit  → o ma [cuit]
    # Le riz n'est pas cuit → o ma [cuit]
    # Cas 3 : participe résultatif — détecté via tree['is_participe_passe']
    # ou via tokens (selon le chemin de détection)
    _part_adj_ttb = next((t for t in _tokens_ref
                          if t.get('is_participe_passe') is True
                          and t.get('bm')), None)
    _has_cop_ttb = any(t.get('dep') == 'cop' for t in _tokens_ref)
    # Fallback : utiliser tree['participe_bm'] si tokens ne contiennent pas le flag
    if not _part_adj_ttb and tree.get('is_participe_passe') and tree.get('participe_bm'):
        _part_adj_ttb = {'bm': tree['participe_bm'], 'is_participe_passe': True}
    if _part_adj_ttb and _has_cop_ttb:
        _part_bm = _part_adj_ttb.get('bm', '')
        # Sujet : garder S si traduit (iri, dén...), sinon 'o' anaphorique
        # Expletif (c', ce, ça) → S='o' déjà mis par le moteur
        _subj = S if S else 'o'
        if neg:
            result = j(_subj, 'ma', _part_bm)
        else:
            _v = _part_bm
            if _v and not _v.endswith('ra') and not _v.endswith('na'):
                _v += 'na' if _v.endswith('n') else 'ra'
            result = j(_subj, _v)
        print(f"  ✂️  Clause 1 -> '{result}'")
        return result.strip()

    # F3 : résultatif passé
    if (ct in ('simple', 'complex', 'conditional', 'temporal', 'relative_post')
            and _o_is_xcomp and O and tn == 'past'
            and V and not V.endswith('ra') and not V.endswith('na')):
        V_past = V + 'ra'
        if obl_strings or ' ' in S:
            result = j(S, *obl_strings) + ', o ' + j(V_past, O, 'ye')
        else:
            result = j(S, V_past, O, 'ye')

    elif ct in ('statif', 'statif_past'):
        if ct == 'statif_past':
            # ST_03/ST_04 : S tùn V-len dòn/tɛ (pas de bɛ)
            _tun = 'tùn'
            _assert = 'tɛ' if neg else 'dòn'
            result = j(S, _tun, m.get('QUAL', ''), _assert, *obl_strings)
        else:
            # ST_01/ST_02 : S V-len dòn/tɛ
            result = j(S, m.get('QUAL', ''), TAM, *obl_strings)

    elif ct == 'existential_nominal':
        _exist_op    = 'tɛ' if neg else 'bɛ'
        _normal_obls = [obl_strings[i] for i, c in enumerate(m.get('OBL_ALL', []))
                        if i < len(obl_strings) and c.get('DEP_TYPE') != 'acl:relcl']
        _relcl_obls  = [obl_strings[i] for i, c in enumerate(m.get('OBL_ALL', []))
                        if i < len(obl_strings) and c.get('DEP_TYPE') == 'acl:relcl']
        result = j(O, *_normal_obls, _exist_op, ',', *_relcl_obls)

    elif ct == 'existential_absolute':
        # Matrice cas 13 : [Nom] + bɛ (existence pure)
        # Le sujet bambara est le nom lui-même, pas le pronom français 'il'
        # 'Il y a du pain' → búuru bɛ (pas a bɛ búuru)
        # 'Il y a du pain ici' → búuru bɛ yàn
        _exist_op   = 'tɛ' if neg else 'bɛ'
        _exist_loc  = G.get('existence_loc_marker', 'yàn') or 'yàn'
        _exist_subj = O or S
        # Capturer advmod ET ROOT ADV (ex: ici dep=ROOT)
        _adv_strs = [t.get('bm', '') for t in tree.get('_tokens', [])
                     if t.get('dep') in ('advmod', 'ROOT')
                     and t.get('bm')
                     and t.get('role') not in ('negation',)
                     and t.get('pos') == 'ADV']
        if obl_strings or _adv_strs:
            result = j(_exist_subj, _exist_op, *obl_strings, *_adv_strs)
        else:
            result = j(_exist_subj, _exist_op)

    elif ct == 'existential_localized':
        # Matrice cas 13 : [Nom] + bɛ + [Lieu]
        # 'Il y a de l'eau dans la bouteille' → jí bɛ bútèli kɔnɔ
        _exist_op   = 'tɛ' if neg else 'bɛ'
        _exist_subj = O or S
        result      = j(_exist_subj, _exist_op, *obl_strings)

    elif ct == 'infinitive':
        # Si O vide mais S présent → S est l'objet de l'infinitif
        _inf_o = O if O else S
        result = j('ka', _inf_o, V, *obl_strings)

    elif ct == 'ownership':
        result = j(S, 'yé', O, 'de', 'ta', 'ye')

    elif ct == 'verb_serial':
        _com_marker = G.get('comitative_marker', 'ni') or 'ni'
        _com_heads, _other_obls = [], []
        _raw_obls = m.get('OBL_ALL', [])
        for _ci, _obl in enumerate(_raw_obls):
            if isinstance(_obl, dict) and _obl.get('local_clause_type') == 'comitative':
                _head_only = _obl.get('HEAD', '')
                if _obl.get('COMPOUND'):
                    _head_only = _obl['COMPOUND'] + ' ' + _head_only
                if _obl.get('MOD'):
                    _head_only = j(_head_only, _obl['MOD'])
                _com_heads.append(_head_only)
                _tokens_ref = tree.get('_tokens', [])
                _head_orig  = next((t.get('orig_index') for t in _tokens_ref
                                    if t.get('bm') == _obl.get('HEAD')
                                    or (t.get('bm') and t.get('bm') in _head_only)), None)
                if _head_orig is not None:
                    for _conj_tok in _tokens_ref:
                        if (_conj_tok.get('dep') == 'conj'
                                and _conj_tok.get('head_index') == _head_orig):
                            _conj_bm = _conj_tok.get('bm') or ''
                            _conj_has_relcl = any(
                                t.get('dep') == 'acl:relcl'
                                and t.get('head_index') == _conj_tok['orig_index']
                                for t in _tokens_ref)
                            if not _conj_has_relcl and _conj_bm:
                                _com_heads.append(_conj_bm)
            else:
                _other_obls.append(obl_strings[_ci] if _ci < len(obl_strings) else '')
        # _com_str : si les conj sont déjà dans head, utiliser directement
        if _com_heads:
            _com_str = j(_com_marker, (' ' + _com_marker + ' ').join(_com_heads))
        else:
            _com_str = ''
        _xcomp_tok = next((t for t in _tokens_ref
                           if t.get('dep') == 'xcomp'
                           and t.get('pos') == 'VERB'
                           and t.get('bm') == V_ACT), None)
        _xcomp_is_action = (_xcomp_tok
                            and _xcomp_tok.get('intransitive_type') in ('ACTION', 'nominalized', 'support')
                            and _xcomp_tok.get('semantic_class', '') not in INTRANS_SC)
        if O:
            result = j(S, TAM, V, 'ka', O, _com_str, V_ACT, *_other_obls)
        elif _xcomp_is_action:
            _v_act_nom = V_ACT + 'li'
            result = j(S, TAM, V, 'ka', _v_act_nom, 'kɛ', _com_str, *_other_obls)
        else:
            result = j(S, TAM, V, 'ka', V_ACT, _com_str, *_other_obls)

    elif ct == 'interrogative':
        _question_marker = next((t.get('bm', '') for t in tree.get('_tokens', [])
                                 if t.get('role') == 'question_marker'
                                 and t.get('bm')), '')
        _alt_tok = next((t for t in tree.get('_tokens', [])
                         if t.get('role') == 'alternative' and t.get('bm')), None)
        _interrog_end = next((t.get('bm', '') for t in tree.get('_tokens', [])
                              if t.get('role') == 'interrogative_end'
                              and t.get('bm')), 'wà ?')
        _clean_obls = [_xv for _ci, _xv in enumerate(obl_strings)
                       if _ci < len(m.get('OBL_ALL', []))
                       and m['OBL_ALL'][_ci].get('DEP_TYPE') != 'fixed'
                       and not any(t.get('dep') == 'fixed'
                                   and t.get('bm') == m['OBL_ALL'][_ci].get('HEAD')
                                   for t in tree.get('_tokens', []))]
        # Éviter doublon TAM == V
        _v_display = V if (V and V.rstrip('́') != TAM.rstrip('́')) else ''

        if _alt_tok:
            _conj_v = next((t for t in tree.get('_tokens', [])
                            if t.get('dep') == 'conj'
                            and t.get('pos') == 'VERB'
                            and t.get('bm')), None)
            _conj_o = next((t for t in tree.get('_tokens', [])
                            if t.get('dep') == 'obj'
                            and _conj_v
                            and t.get('head_index') == _conj_v.get('orig_index')
                            and t.get('bm')), None)
            _conj_o_amod = next((t for t in tree.get('_tokens', [])
                                 if t.get('dep') == 'amod'
                                 and _conj_o
                                 and t.get('head_index') == _conj_o.get('orig_index')
                                 and t.get('bm')), None)
            _conj_o_bm = (j(_conj_o.get('bm', ''),
                            _conj_o_amod.get('bm', '') if _conj_o_amod else '')
                          if _conj_o else '')
            _conj_v_bm = _conj_v.get('bm', '') if _conj_v else ''
            result = j(S, TAM, O, _v_display, *_clean_obls,
                       _alt_tok.get('bm', ''), S, TAM, _conj_o_bm, _conj_v_bm, '?')
        else:
            _v_is_motion = next((t for t in tree.get('_tokens', [])
                                 if t.get('is_root')
                                 and t.get('semantic_class') == 'motion'), None)
            if _v_is_motion and _v_display:
                _loc_marker = next((t.get('bm_marker', '')
                                    for t in tree.get('_tokens', [])
                                    if t.get('role') == 'locative'
                                    and t.get('dep') == 'case'
                                    and t.get('bm_marker')), '')
                _o_loc = j(O, _loc_marker) if _loc_marker else O
                result = j(_question_marker, S, TAM, _v_display, _o_loc,
                           *_clean_obls, _interrog_end)
            else:
                # Action verb sans objet → V+li kɛ
                _root_tok_ref2 = next((t for t in tree.get('_tokens', [])
                                       if t.get('is_root')), None)
                _is_action_interrog = (
                    _v_display and not O
                    and _root_tok_ref2
                    and _root_tok_ref2.get('intransitive_type') == 'ACTION'
                    and _root_tok_ref2.get('semantic_class', '')
                    not in {'saying', 'motion', 'perception',
                            'sound', 'communication', 'eTu mangmission'})
                if _is_action_interrog:
                    _v_nom = _v_display + 'li'
                    result = j(_question_marker, S, TAM, _v_nom, 'kɛ',
                               *_clean_obls, _interrog_end)
                else:
                    result = j(_question_marker, S, TAM, O, _v_display,
                               *_clean_obls, _interrog_end)

    elif ct == 'equative':
        # Est-ce que X est [loc] ? → Yala X bɛ [loc] wà ?
        if tree.get('est_ce_que') and obl_strings:
            _yala = G.get('question_marker_yala', 'Yala') or 'Yala'
            # Reconstruire le sujet réel depuis les tokens
            # frères (nsubj) + possessif (tes→i) + conj sœurs
            _subj_tok = next((t for t in _tokens_ref
                              if t.get('dep') == 'nsubj'
                              and t.get('pos') in ('NOUN', 'PROPN')
                              and t.get('bm')), None)
            _conj_tok = next((t for t in _tokens_ref
                              if t.get('dep') == 'conj'
                              and t.get('pos') in ('NOUN', 'PROPN')
                              and t.get('bm')), None)
            if _subj_tok:
                _poss_s = next((t for t in _tokens_ref
                                if t.get('dep') == 'det'
                                and t.get('role') in ('pronoun', 'possessive')
                                and t.get('head_index') == _subj_tok['orig_index']
                                and t.get('bm')), None)
                _s_bm = _subj_tok.get('bm', '')
                if _poss_s:
                    _s_bm = j(_poss_s.get('bm'), _s_bm)
                if _conj_tok:
                    _poss_c = next((t for t in _tokens_ref
                                   if t.get('dep') == 'det'
                                   and t.get('role') in ('pronoun', 'possessive')
                                   and t.get('head_index') == _conj_tok['orig_index']
                                   and t.get('bm')), None)
                    _c_bm = _conj_tok.get('bm', '')
                    # Pluriel sur conj
                    if _conj_tok.get('is_plural') and not _c_bm.endswith('w'):
                        _c_bm = _c_bm + 'w'
                    if _poss_c:
                        _c_bm = j(_poss_c.get('bm'), _c_bm)
                    # Pluriel sur sujet principal
                    if _subj_tok.get('is_plural') and not _s_bm.endswith('w'):
                        _s_bm = _s_bm + 'w'
                    _real_s = j(_s_bm, 'ni', _c_bm)
                else:
                    _real_s = _s_bm
            else:
                _real_s = O or S
            result = j(_yala, _real_s, 'bɛ', *obl_strings, 'wà ?')
            print(f"  ✂️  Clause 1 -> '{result}'")
            return result.strip()
        # Cas spécial : rien (role=negation + dep=ROOT) → a tɛ foyi yé
        if not O and not V:
            _rien_tok = next((t for t in tree.get('_tokens', [])
                              if t.get('role') == 'negation'
                              and t.get('dep') == 'ROOT'
                              and t.get('bm')), None)
            if _rien_tok:
                O = _rien_tok.get('bm', '')
        # Futur via cop_tense OU via tokens directement
        # Fallback : chercher cop futur dans _tokens_ref
        if not tree.get('cop_tense'):
            _cop_fut_tok = next((t for t in _tokens_ref
                                 if t.get('dep') == 'cop'
                                 and (t.get('tense') == 'fut'
                                      or 'Tense=Fut' in str(t.get('morph', '')))), None)
            if _cop_fut_tok:
                tree['cop_tense'] = 'fut'
                tree['cop_bm']    = _cop_fut_tok.get('bm', '')
                tree['cop_lemma'] = _cop_fut_tok.get('lemma', 'être')
        # Récupérer NOUN conj si O vide
        if not O and not V:
            _conj_attr = next((t for t in _tokens_ref
                               if t.get('dep') in ('conj', 'attr', 'appos')
                               and t.get('pos') in ('NOUN', 'PROPN', 'ADJ')
                               and t.get('bm')), None)
            if _conj_attr:
                O = _conj_attr.get('bm', '')
        # Futur via cop_tense stocké dans tree par step6_copule
        # a tɛ na [être] foyi yé
        if tree.get('cop_tense') == 'fut':
            _cop_bm = tree.get('cop_bm') or f"[{tree.get('cop_lemma', 'être')}]"
            TAM = 'tɛ na' if neg else 'bɛ na'
            O = j(_cop_bm, O) if O else _cop_bm
        if neg and tree.get('tense') in ('past', 'hab', 'plup'):
            _tun_base = TAM.split()[0] if TAM and ' ' in TAM else 'tùn'
            result = j(S, _tun_base, 'tɛ', O or V, 'yé', *obl_strings)
        elif neg:
            # Utiliser TAM (peut être 'tɛ na' pour le futur)
            result = j(S, TAM, O or V, 'yé', *obl_strings)
        elif O and str(O).endswith('yé'):
            result = j(S, tree.get('tam', 'yé'), O, *obl_strings)
        elif V and not O and tn == 'past':
            _already = tree.get('already_marker', '')
            _v_past = V
            if not _v_past.endswith('ra') and not _v_past.endswith('na'):
                _v_past = _v_past + 'na' if _v_past.endswith('n') else _v_past + 'ra'
            result = j(S, _v_past, _already, *obl_strings)
        else:
            _why_tok = next((t for t in tree.get('_tokens', [])
                             if t.get('role') == 'interrogative'
                             and t.get('dep') == 'advmod'
                             and t.get('bm')), None)
            if _why_tok:
                _adj_bm = V or next((t.get('bm', '') for t in tree.get('_tokens', [])
                                     if t.get('pos') == 'ADJ'
                                     and t.get('is_root')), '')
                result = j(S, _adj_bm, 'lendòn', _why_tok.get('bm', ''), '?')
            else:
                # Utiliser TAM (peut être 'bɛ na' pour le futur)
                tam_equatif = TAM if TAM not in ('bɛ', 'tɛ', '') else tree.get('tam', 'yé')
                result = j(S, tam_equatif, O or V, 'yé', *obl_strings, _ccomp_str)
    elif ct == 'identificatory':
        _focus_marker = G.get('focus_marker', 'de')
        _appos = next((t for t in tree.get('_tokens', [])
                       if t.get('pos') == 'PROPN'
                       and t.get('dep') in ('ROOT', 'appos', 'flat', 'flat:name')), None)
        _appos_bm = (_appos.get('bm') or _appos.get('surface', '')) if _appos else ''
        _is_interrog_idnt = any(
            str(t.get('surface', '')).strip() == '?'
            for t in tree.get('_tokens', []))
        _end = 'wà ?' if _is_interrog_idnt else ''
        if _appos_bm:
            result = j(_appos_bm, 'tɛ' if neg else 'dòn', _end)
        else:
            result = j(S, 'tɛ') if neg else j(S, 'dòn', _end)

    elif ct == 'presentative':
        if neg:
            result = j(S, 'tɛ', *obl_strings)
        else:
            # Déictique (voilà) → [Nom] félé — role='deictique' depuis KG
            _is_deictique = any(t.get('role') == 'deictique' for t in tree.get('_tokens', []))
            if _is_deictique:
                result = j(O or S, G.get('deictique_marker', 'félé'))
            else:
                o_clean = O if O != S else ''
                if o_clean:
                    result = j(S, 'ni', o_clean, 'dòn', *obl_strings)
                else:
                    result = j(S, 'dòn', *obl_strings)

    elif ct == 'noun_phrase':
        # Ne pas appliquer le suffixe article (ex: 'yin' de 'le') quand le nom
        # fait partie d'une chaîne génitive (nmod présent) : le résultat serait
        # "n díyanyemɔgɔ ka dénkɛw yin" au lieu de "n díyanyemɔgɔ ka dénkɛ".
        _has_nmod_chain = any(t.get('dep') == 'nmod'
                              for t in tree.get('_tokens', []))
        _conn_suf = (m.get('SLOTS', {}).get('conn_suffix', '')
                     or next((c.get('DEM_SUFF', '')
                               for c in m.get('OBL_ALL', [])
                               if c.get('DEM_SUFF')), ''))
        # Supprimer le suffixe quand une chaîne génitive est présente
        if _has_nmod_chain:
            _conn_suf = ''
        if not _conn_suf and not _has_nmod_chain and tree.get('_tokens'):
            _conn_suf = next((t.get('bm_suffix', '')
                              for t in tree['_tokens']
                              if t.get('bm_suffix')), '')
            
        _already     = tree.get('already_marker', '')
        _already_pos = tree.get('already_position', 'END')
        if _conn_suf and _conn_suf.strip():
            if not str(O).endswith(_conn_suf) and not any(
                    str(xv).endswith(_conn_suf) for xv in obl_strings):
                result = (j(_already, O, *obl_strings, _conn_suf)
                          if _already_pos == 'HEAD'
                          else j(O, _already, *obl_strings, _conn_suf))
            else:
                result = (j(_already, O, *obl_strings)
                          if _already_pos == 'HEAD'
                          else j(O, _already, *obl_strings))
        else:
            if _already_pos == 'HEAD':
                result = j(_already, O, *obl_strings)
            else:
                _tmp_obls   = [obl_strings[i]
                               for i, c in enumerate(m.get('OBL_ALL', []))
                               if i < len(obl_strings)
                               and isinstance(c, dict)
                               and c.get('local_clause_type') == 'temporal']
                _other_obls = [obl_strings[i]
                               for i, c in enumerate(m.get('OBL_ALL', []))
                               if i < len(obl_strings)
                               and isinstance(c, dict)
                               and c.get('local_clause_type') != 'temporal']
                result = j(O, *_other_obls, _already, *_tmp_obls)

    elif ct == 'privative_pred':
        if not S:
            result = j(O, *obl_strings)
        else:
            result = j(S, O, 'dòn', *obl_strings)

    elif ct == 'relative_topic':
        _rel_neg = tree.get('neg', False)
        _rel_tam = 'tɛ' if _rel_neg else 'bɛ'
        if V_ACT:
            # SOV bambara : S TAM V ka O V_ACT
            result = j(S, ',', 'o', _rel_tam, V, 'ka', O, V_ACT, *obl_strings)
        else:
            V_past = V if V.endswith('ra') or V.endswith('na') else (
                V + 'na' if V.endswith('n') else V + 'ra')
            result = j(S, ',', 'o', V_past, *obl_strings)

    elif ct in ('simple', 'complex', 'conditional', 'temporal',
                'relative_post', 'reported_comp', 'comitative'):
        _contrast_tok = next((t for t in tree.get('_tokens', [])
                              if t.get('role') == 'contrast' and t.get('bm')), None)
        _contrast_str = _contrast_tok.get('bm', '') if _contrast_tok else ''
        _raw_obls     = m.get('OBL_ALL', [])
        _tmp_strs, _other_strs = [], []
        _tokens_ref = tree.get('_tokens', [])
        for _ci, _obl in enumerate(_raw_obls):
            _xv = obl_strings[_ci] if _ci < len(obl_strings) else ''
            if isinstance(_obl, dict) and _obl.get('local_clause_type') == 'temporal':
                _obl_marker  = _obl.get('MARKER', '')
                _suffix_mkrs = G.get('temporal_suffix_markers', set())
                if _obl_marker and _obl_marker in _suffix_mkrs:
                    _other_strs.append(_xv)
                    continue
                _obl_head = _obl.get('HEAD', '')
                _src_tok  = next((t for t in _tokens_ref
                                  if (t.get('bm') == _obl_head
                                      or t.get('surface') == _obl_head)
                                  and t.get('dep') == 'advmod'
                                  and t.get('pos') == 'ADV'), None)
                if _src_tok:
                    _tmp_strs.append(_xv)
                else:
                    _other_strs.append(_xv)
            else:
                _other_strs.append(_xv)

        _already     = tree.get('already_marker', '')
        _already_pos = tree.get('already_position', 'END')

        # VERB intransitif + comitative → V+li kɛ ni X yé
        # ex: Je mange avec mon mari → n bɛ dúnli kɛ ni n fúrucɛ yé
        _com_obls = [(_ci, _obl) for _ci, _obl in enumerate(_raw_obls)
                     if isinstance(_obl, dict)
                     and _obl.get('local_clause_type') == 'comitative']
        _has_com_obl = bool(_com_obls)
        _root_tok_ref = next((t for t in _tokens_ref if t.get('is_root')), None)
        _root_tok_sc  = (_root_tok_ref.get('semantic_class', '') if _root_tok_ref else '')
        _root_intrans = (_root_tok_ref.get('intransitive_type', '') if _root_tok_ref else '')
        _root_morph = str(_root_tok_ref.get('morph', '')) if _root_tok_ref else ''
        _root_is_resultative = (
            bool(_root_tok_ref)
            and (
                _root_tok_ref.get('is_participe_passe') is True
                or _root_tok_ref.get('is_passive') is True
                or 'VerbForm=Part' in _root_morph
                or 'Voice=Pass' in _root_morph
            )
        )
        _abs_intrans_classes = {'saying', 'motion', 'perception', 'sound',
                                'communication', 'emission'}
        _is_action_verb = (V and not O
                           and ct == 'simple'
                           and not _root_is_resultative
                           and _root_tok_sc not in _abs_intrans_classes
                           and _root_intrans == 'ACTION')
        _is_intrans_com = (V and not O and _has_com_obl
                           and ct == 'simple')
        if _is_action_verb:
            _v_nom = V + 'li'
            if _has_com_obl:
                _com_strs = [obl_strings[_ci]
                             for _ci, _obl in enumerate(_raw_obls)
                             if isinstance(_obl, dict)
                             and _obl.get('local_clause_type') == 'comitative'
                             and _ci < len(obl_strings)]
                _non_com_strs = [obl_strings[_ci]
                                 for _ci, _obl in enumerate(_raw_obls)
                                 if isinstance(_obl, dict)
                                 and _obl.get('local_clause_type') != 'comitative'
                                 and _ci < len(obl_strings)]
                result = j(S, TAM, _v_nom, 'kɛ', *_com_strs, *_non_com_strs)
            else:
                result = j(S, TAM, _v_nom, 'kɛ', *obl_strings)
        elif _is_intrans_com:
            # Nominaliser le verbe : V + li
            _v_nom = V + 'li'
            # Construire ni X yé pour chaque comitative
            _com_str_parts = []
            for _ci, _obl in _com_obls:
                _obl_head = _obl.get('HEAD', '')
                _obl_comp = _obl.get('COMPOUND', '')
                _obl_bm = j(_obl_comp, _obl_head) if _obl_comp else _obl_head
                _com_str_parts.append(j('ni', _obl_bm, 'yé'))
            _non_com_strs = [obl_strings[_ci]
                             for _ci, _obl in enumerate(_raw_obls)
                             if isinstance(_obl, dict)
                             and _obl.get('local_clause_type') != 'comitative'
                             and _ci < len(obl_strings)]
            result = j(S, TAM, _v_nom, 'kɛ', *_com_str_parts, *_non_com_strs)
        elif TAM in ('bɛ kà', 'tɛ kà'):
            result = j(*_tmp_strs, S, TAM, O, V, ADV, *_other_strs)
        elif _o_is_xcomp and O:
            if _already_pos == 'HEAD':
                result = j(_already, _contrast_str, S, TAM, V, O, V_ACT, V_SUF,
                           *_tmp_strs, ADV, *_other_strs, _ccomp_str)
            else:
                result = j(_contrast_str, S, TAM, V, O, V_ACT, V_SUF,
                           *_tmp_strs, ADV, *_other_strs, _ccomp_str, _already)
        else:
            if _already_pos == 'HEAD':
                result = j(_already, _contrast_str, S, TAM, O, V, V_ACT, V_SUF,
                           *_tmp_strs, ADV, *_other_strs, _ccomp_str)
            
            else:
                result = j(_contrast_str, S, TAM, O, V, V_ACT, V_SUF,
                           *_tmp_strs, ADV, *_other_strs, _ccomp_str, _already)

    elif ct == 'qualitative':
        _qual_content = m.get('QUAL', '') or O or V
        # Participe passé résultatif → V+ra/la/na (pas de ka)
        _tokens_ref = tree.get('_tokens', [])
        _root_is_participe = any(
            t.get('is_participe_passe') is True
            for t in _tokens_ref if t.get('is_root'))
        if _root_is_participe:
            _v = _qual_content
            if _v and not _v.startswith('['):
                if _v.endswith('n'):
                    _v += 'na'
                elif _v[-1] in ('o', 'u', 'ɔ'):
                    _v += 'la'
                else:
                    _v += 'ra'
            result = j(S, _v, *obl_strings)
        else:
            _qual_tam = TAM if TAM in ('man', 'ma') else ('man' if neg else 'ka')
            if tn == 'past' or tn == 'hab':
                _tun = _resolve_tam(tn, neg, G) or ('tùn tɛ' if neg else 'tùn bɛ')
                _tun_base = _tun.split()[0] if _tun else 'tùn'
                _qual_tam = j(_tun_base, 'ma' if neg else 'ka')
            # Futur qualitatif : a bɛ na bònya (ADJ + ya nominalisé)
            elif tree.get('cop_tense') == 'fut':
                _qual_tam = 'tɛ na' if neg else 'bɛ na'
                _qual_content = _qual_content + 'ya'
            result = j(S, _qual_tam, _qual_content, *obl_strings, _ccomp_str)

    elif ct == 'locative':
        result = j(S, 'tɛ' if neg else 'bɛ', *obl_strings)

    elif ct == 'existential':
        result = j(S, 'tɛ' if neg else 'bɛ', *obl_strings)

    elif ct == 'passive':
        result = j(S, 'bɛ ka', V, *obl_strings)

    elif ct == 'imperative':
        result = j(O, V, *obl_strings)

    elif ct == 'prohibitive':
        result = j('kàna', O, V, *obl_strings)

    elif ct == 'participial_to':
        result = j(S, V, ADV, *obl_strings)

    elif ct == 'focus':
        result = j(S, 'de', TAM, O, V, *obl_strings)

    elif ct == 'exclamative':
        result = j(S, TAM, O, V, *obl_strings, 'dɛ !')

    elif ct == 'reciprocal':
        result = j(S, TAM, 'ɲɔgɔn', V, *obl_strings)

    elif ct == 'concessive':
        result = j(S, TAM, O, V, ADV, *obl_strings)

    elif ct == 'causal':
        result = j(S, TAM, O, V, ADV, *obl_strings)

    elif ct == 'relative_min':
        result = j(S, 'mìn', TAM, O, V, ',', 'ò', TAM, V, *obl_strings)

    elif ct == 'topicalised':
        result = j(O, ',', S, TAM, V, *obl_strings)

    elif ct in ('participial_len', 'participial_ta', 'participial_bali'):
        result = j(S, V, *obl_strings)

    elif ct == 'relative_post':
        result = j(S, 'mìn', TAM, O, V, *obl_strings)

    elif ct == 'refl_past':
        result = j(S, 'tun ye', V, O, *obl_strings, ADV)

    elif ct in ('reported_verb', 'reported_comp', 'reported'):
        if neg:
            result = j(S, 'ma', O, 'fɔ', ADV, *obl_strings)
        else:
            result = j(S, TAM, O, V, ADV, *obl_strings)

    elif ct == 'content_question':
        # Nom seul + det interrogatif (Quelle femme ?)
        _interrog_det_on_s = next((t for t in tree.get('_tokens', [])
                                   if t.get('role') == 'interrogative'
                                   and t.get('dep') == 'det'
                                   and t.get('bm')), None)
        if _interrog_det_on_s and S and not O and not V:
            result = j(S, _interrog_det_on_s.get('bm', 'jùmɛn'), '?')
            print(f"  ✂️  Clause 1 -> '{result}'")
            return result.strip()

        # S contient l'interrogatif → déplacer en O, récupérer vrai sujet
        _interrog_bms = {v.get('bm') for (surf, l), v in G.get('funcs', {}).items()
                         if v.get('role') in ('interrogative', 'relative')
                         and v.get('bm')}
        _s_is_interrog = any(
            (t.get('role') in ('interrogative', 'relative', 'pronoun')
             and t.get('bm') == S
             and t.get('bm') in _interrog_bms)
            for t in tree.get('_tokens', []))
        if _s_is_interrog:
            if not O:
                O = S
            _real_subj = next((t for t in tree.get('_tokens', [])
                               if t.get('pos') == 'PRON'
                               and t.get('dep') in ('nsubj', 'nsubj:pass', 'dep')
                               and t.get('bm')
                               and t.get('bm') != S
                               and t.get('role') not in {'interrogative', 'expletive', 'clitic'}
                               and 'Int' not in str(t.get('morph', ''))
                               and not str(t.get('surface', '')).startswith('-')), None)

            # Fallback : chercher pronom inversé (-ils, -vous) avec bm valide
            if not _real_subj:
                _root_ref = next((t for t in tree.get('_tokens', [])
                                  if t.get('is_root')), None)
                _root_is_plural = _root_ref.get('is_plural', False) if _root_ref else False
                _real_subj = next((t for t in tree.get('_tokens', [])
                                   if t.get('dep') in ('nsubj', 'nsubj:pass')
                                   and t.get('bm')
                                   and t.get('bm') != S
                                   and 'Int' not in str(t.get('morph', ''))
                                   and t.get('pos') == 'PRON'
                                   and t.get('is_plural') == _root_is_plural), None)
                
            if _real_subj:
                print(f"DEBUG _real_subj: {_real_subj.get('surface')} bm={_real_subj.get('bm')} plural={_real_subj.get('is_plural')}")
                S = _real_subj.get('bm', '')
            else:
                _funcs = G.get('funcs', {})
                _lang_ref = next((t.get('lang', 'fr') for t in tree.get('_tokens', [])
                                  if t.get('lang')), 'fr')
                _s_from_funcs = next(
                    (v.get('bm') for (surf, l), v in _funcs.items()
                     if l == _lang_ref
                     and v.get('role') == 'subject'
                     and v.get('bm')
                     and surf in G.get('pron', set())
                     and surf not in G.get('sing', set())
                     and _root_is_plural), None)
                if _s_from_funcs:
                    S = _s_from_funcs

        elif not S:
            _subj_tok = next((t for t in tree.get('_tokens', [])
                              if t.get('pos') == 'PRON'
                              and t.get('dep') in ('nsubj', 'nsubj:pass')
                              and t.get('bm')), None)
            if _subj_tok:
                S = _subj_tok.get('bm', '')

            # Récupérer le mot interrogatif en O s'il est absent
            if not O:
                _interrog_tok = next((t for t in tree.get('_tokens', [])
                                    if t.get('role') == 'interrogative'
                                    and t.get('dep') in ('obj', 'nsubj', 'dep')), None)
                if _interrog_tok:
                    O = _interrog_tok.get('bm') or G.get('interrogative_who', 'jɔn')

        _interrog_adv = next((t for t in tree.get('_tokens', [])
                              if t.get('role') == 'interrogative'
                              and t.get('dep') in ('advmod', 'dep')
                              and t.get('bm')), None)
        _interrog_adv_bm  = _interrog_adv.get('bm', '') if _interrog_adv else ''
        _interrog_noun    = tree.get('interrog_noun')
        _interrog_noun_bm = _interrog_noun.get('bm', '') if _interrog_noun else ''
        if _interrog_noun_bm and O:
            O = j(_interrog_noun_bm, O)
        elif _interrog_noun_bm:
            O = _interrog_noun_bm

        if TAM == 'yé' and not _interrog_noun_bm:
            result = j(S, TAM, V or O, 'yé', _interrog_adv_bm, *obl_strings)
        elif TAM == 'yé' and _interrog_noun_bm:
            result = j(S, TAM, O, V, '?')
        else:
            _adv_already_in_O = _interrog_adv_bm and _interrog_adv_bm in O
            if _adv_already_in_O:
                result = j(S, TAM, V, O, *obl_strings, '?')
            else:
                _o_final = O or G.get('interrogative_who', 'jɔn')
                result = j(S, TAM, _o_final, V, _interrog_adv_bm, *obl_strings, '?')

    elif ct == 'noun_phrase_have':
        # Matrice document :
        # Possession matérielle (objet physique, argent, voiture) → bóló
        # Possession abstraite (famille, sentiment, âge, lien) → fɛ
        # Détection depuis KG via possession_type du token objet
        _poss_type   = tree.get('possession_type', '')
        if not _poss_type:
            # Fallback : détecter via semantic_class de l'objet
            _obj_tok_have = next((t for t in tree.get('_tokens', [])
                                  if t.get('bm') == O and t.get('pos') in ('NOUN', 'PROPN')), None)
            if _obj_tok_have:
                _sc = _obj_tok_have.get('semantic_class', '')
                _poss_type = 'material' if _sc in ('object', 'money', 'vehicle', 'tool') else 'abstract'
        _poss_marker = G.get('possession_material_marker', 'bóló') if _poss_type == 'material'                        else G.get('possession_abstract_marker', 'fɛ')
        _exist_op    = 'tɛ' if neg else 'bɛ'
        _qty_interrog = next((t for t in tree.get('_tokens', [])
                              if t.get('role') == 'interrogative'
                              and t.get('bm')
                              and t.get('dep') in ('det', 'advmod', 'dep', 'obj')), None)
        if _qty_interrog and _qty_interrog.get('bm') not in (O or ''):
            result = j(O, _exist_op, S, _poss_marker, _qty_interrog.get('bm'), '?')
        else:
            result = j(O, _exist_op, S, _poss_marker, *obl_strings)

    elif ct == 'noun_phrase_inh':
        result = j(O, *obl_strings) if obl_strings else j(S, O)

    elif ct == 'comitative':
        _tokens = tree.get('_tokens', [])
        _avec   = next((t for t in _tokens
                        if isinstance(t, dict)
                        and t.get('role') == 'preposition'
                        and t.get('dep') == 'case'), None)
        _comp   = None
        if _avec:
            av_i = _avec.get('orig_index', -1)
            _cn  = next((t for t in _tokens
                         if isinstance(t, dict)
                         and t.get('pos') in ('NOUN', 'PROPN', 'PRON')
                         and (t.get('head_index', -1) == av_i
                              or _avec.get('head_index', -1) == t.get('orig_index', -1))),
                        None)
            if _cn:
                _cp = next((t for t in _tokens
                             if isinstance(t, dict)
                             and t.get('role') == 'possessive'
                             and t.get('head_index', -1) == _cn.get('orig_index', -1)), None)
                _bm  = _cn.get('bm', '')
                if _cp:
                    _p    = _cp.get('bm', '')
                    _comp = j('n', _bm) if _p == 'n' else j(_p, 'ka', _bm)
                else:
                    _comp = _bm
        companion = _comp or V
        if companion and V and V != S:
            result = j(S, 'ni', companion, TAM, V, *obl_strings)
        elif companion:
            result = j(S, 'ni', companion, 'dòn', *obl_strings)
        else:
            result = j(S, 'dòn', *obl_strings)

    elif ct == 'relative_nominal':
        result = tree.get('final_string', '')
    else:
        result = j(S, TAM, O, V, V_ACT, V_SUF, *obl_strings, ADV)

    print(f"  ✂️  Clause 1 -> '{result}'")
    return result.strip().replace(' ,', ',').replace('« ', '«').replace(' »', '»')


# ── RULE ENGINE ───────────────────────────────────────────────────────────────
