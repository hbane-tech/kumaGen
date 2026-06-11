"""
rules/renderers/__init__.py
Point d'entrée unique : tree_to_bambara()
Remplace rules/tree_to_bambara.py — même interface, même comportement.

Usage (identique à l'ancien fichier) :
    from rules.renderers import tree_to_bambara
"""
from rules.core import j, _GENITIVE_FALLBACK, _resolve_tam

from rules.renderers.overrides      import apply_overrides
from rules.renderers.existential    import (render_existential_nominal,
                                            render_existential_absolute,
                                            render_existential_localized)
from rules.renderers.copula         import (render_statif, render_equative,
                                            render_identificatory,
                                            render_presentative, render_qualitative,
                                            render_locative_existential)
from rules.renderers.verbal         import (render_verb_serial, render_simple,
                                            render_conditional,
                                            render_relative_topic,
                                            render_comitative_clause, render_misc)
from rules.renderers.interrogative  import (render_interrogative,
                                            render_content_question,
                                            render_noun_phrase_have)
from rules.renderers.nominal        import render_noun_phrase


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
    # REFLEXIF ABSOLU: preserve TAM from step3 even if lost in step6
    if tree.get('clause_type') == 'refl_absolute' and (not tam_val or tam_val.strip() == ''):
        # Recover TAM based on tense for reflexive clauses
        tam_val = _resolve_tam(tn, tree.get('neg', False), G)
    if not tam_val or tam_val.strip() == '':
        tam_val = 'bɛ'
    TAM = tam_val

    # ── Ccomp ────────────────────────────────────────────────────────────────
    _ccomp_data = m.get('CCOMP')
    _ccomp_str  = ''
    if isinstance(_ccomp_data, dict):
        _ko = G.get('reported_intro', 'ko') or 'ko'
        if _ccomp_data.get('type') == 'comparative':
            _cs = _ccomp_data.get('S', '')
            _ca = _ccomp_data.get('adj_bm', '')
            _cp = _ccomp_data.get('particle', 'ka tɛmɛ')
            _cr = _ccomp_data.get('ref_bm', '')
            _cn = 'man' if _ccomp_data.get('neg') else 'ka'
            _ccomp_str = j(_ko, _cs, _cn, _ca, _cp, _cr, 'kan')
        elif _ccomp_data.get('type') == 'verbal':
            # Clause verbale rapportée : ko S TAM (O) V
            _ccomp_str = j(_ko, _ccomp_data.get('S', ''),
                           _ccomp_data.get('tam', ''),
                           _ccomp_data.get('O', ''),
                           _ccomp_data.get('V', ''))
        elif _ccomp_data.get('O'):
            _ccomp_str = j(_ko, _ccomp_data.get('S', ''),
                           _ccomp_data.get('tam', 'yé'),
                           _ccomp_data.get('O', ''),
                           _ccomp_data.get('tam', 'yé'))

    # ── Wagons obliques ───────────────────────────────────────────────────────
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
            elif lct in ('locative', 'temporal'):
                if pref or suff:
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

    # ── F6 : past intransitif ─────────────────────────────────────────────────
    _o_is_xcomp = m.get('O_IS_XCOMP', False)
    if tn == 'past' and not tree.get('is_transitive', True) and not _o_is_xcomp and ct != 'reciprocal':
        if neg:
            TAM = 'ma'
        else:
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

    # ── Debug slots ───────────────────────────────────────────────────────────
    print(f"DEBUG F6: V={V!r}, TAM={TAM!r}, S={S!r}")
    print('\n  📦 SLOTS STRUCTURELS FINAUX :')
    if S:      print(f'     [ S     ] → {S}')
    if TAM and ct != 'noun_phrase':
               print(f'     [ TAM   ] → {TAM}')
    if O:      print(f'     [ O     ] → {O}')
    if V:      print(f'     [ V     ] → {V}')
    if V_ACT:  print(f'     [ V_ACT ] → {V_ACT}')
    if ADV:    print(f'     [ ADV   ] → {ADV}')
    if V_SUF:  print(f'    [V_SUF  ] -> {V_SUF}')
    if _ccomp_str: print(f'    [ccomp  ] -> {_ccomp_str}')
    for _i, _xv in enumerate(obl_strings):
        _lct = (m['OBL_ALL'][_i].get('local_clause_type', '')
                if _i < len(m.get('OBL_ALL', [])) else '')
        print(f'     [ X{_i+1:<3d}   ] → {_xv}  ({_lct})')
    print(f"  🏷️  clause_type = {ct}")

    # ── Contexte partagé pour overrides ──────────────────────────────────────
    _tokens_ref = tree.get('_tokens', [])
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

    # ── OVERRIDES (cas -1 à 3) ────────────────────────────────────────────────
    _override_result, _handled = apply_overrides(
        tree, m, S, O, neg, _tokens_ref,
        _has_real_subj_ttb, _has_expletif_ttb)
    if _handled:
        print(f"  ✂️  Clause 1 -> '{_override_result}'")
        return _override_result.strip()

    # ── F3 : résultatif passé ─────────────────────────────────────────────────
    result = ''
    if (ct in ('simple', 'complex', 'conditional', 'temporal', 'relative_post')
            and _o_is_xcomp and O and tn == 'past'
            and V and not V.endswith('ra') and not V.endswith('na')):
        V_past = V + 'ra'
        if obl_strings or ' ' in S:
            result = j(S, *obl_strings) + ', o ' + j(V_past, O, 'ye')
        else:
            result = j(S, V_past, O, 'ye')

    # ── DISPATCH ──────────────────────────────────────────────────────────────
    elif ct in ('statif', 'statif_past'):
        result = render_statif(tree, m, S, TAM, neg, obl_strings)

    elif ct == 'existential_nominal':
        result = render_existential_nominal(tree, m, O, neg, obl_strings)

    elif ct == 'existential_absolute':
        result = render_existential_absolute(tree, S, O, neg, obl_strings, G)

    elif ct == 'existential_localized':
        result = render_existential_localized(S, O, neg, obl_strings)

    elif ct == 'infinitive':
        result = render_misc(ct, S, O, V, V_ACT, V_SUF, ADV, TAM, neg,
                             obl_strings, tree, m)

    elif ct == 'ownership':
        result = render_misc(ct, S, O, V, V_ACT, V_SUF, ADV, TAM, neg,
                             obl_strings, tree, m)

    elif ct == 'verb_serial':
        result = render_verb_serial(tree, m, S, O, V, V_ACT, TAM, obl_strings, G)

    elif ct == 'interrogative':
        result = render_interrogative(tree, m, S, O, V, TAM, obl_strings, G)

    elif ct == 'equative':
        result = render_equative(tree, m, S, O, V, TAM, neg, tn,
                                 obl_strings, _ccomp_str, G, _tokens_ref)

    elif ct == 'identificatory':
        result = render_identificatory(tree, S, neg, G)

    elif ct == 'presentative':
        result = render_presentative(tree, S, O, neg, obl_strings, G)

    elif ct == 'noun_phrase':
        result = render_noun_phrase(tree, m, O, obl_strings)

    elif ct == 'privative_pred':
        result = render_misc(ct, S, O, V, V_ACT, V_SUF, ADV, TAM, neg,
                             obl_strings, tree, m)

    elif ct == 'relative_topic':
        result = render_relative_topic(tree, m, S, O, V, V_ACT, obl_strings)

    elif ct == 'conditional':
        result = render_conditional(tree, m, S, O, V, V_ACT, TAM, obl_strings, G)

    elif ct in ('simple', 'complex', 'temporal',
                'relative_post', 'reported_comp', 'comitative'):
        result = render_simple(tree, m, S, O, V, V_ACT, V_SUF, ADV, TAM, ct,
                               neg, obl_strings, _ccomp_str, _o_is_xcomp, G)

    elif ct == 'comparative':
        # ka [ADJ] ka tɛmɛ [référent] kan
        _comp_particle = m.get('comparative_particle', 'ka tɛmɛ')
        _comp_ref      = m.get('comparative_ref', '')
        _comp_neg      = 'man' if neg else 'ka'
        result = j(S, _comp_neg, V, _comp_particle, _comp_ref, 'kan')

    elif ct == 'refl_absolute':
        # Réflexif → S TAM S [yɛrɛ] V (forme transitive, pas le passif V+ra).
        # TAM transitif : passé yé/ma, présent bɛ/tɛ. Verbe NU (pas V+ra).
        # yɛrɛ (soi-même) seulement pour les agentifs transitifs (se blesser),
        # pas pour les inhérents posture/soin (s'asseoir, se laver).
        _refl_pron = tree.get('refl_pron', 'a')
        _refl_v    = tree.get('refl_verb') or V
        _refl_tam  = TAM  # respecte progressif (bɛ kà), passé (yé/ma), négatif (tɛ/ma)
        _refl_self = 'yɛrɛ' if tree.get('refl_yere') else ''
        result = j(S, _refl_tam, _refl_pron, _refl_self, _refl_v, *obl_strings, ADV)

    elif ct == 'qualitative':
        result = render_qualitative(tree, m, S, O, V, TAM, neg, tn,
                                    obl_strings, _ccomp_str, G)

    elif ct in ('locative', 'existential'):
        result = render_locative_existential(S, neg, obl_strings)

    elif ct == 'passive':
        result = render_misc(ct, S, O, V, V_ACT, V_SUF, ADV, TAM, neg,
                             obl_strings, tree, m)

    elif ct in ('imperative', 'prohibitive', 'participial_to', 'focus',
                'exclamative', 'reciprocal', 'concessive', 'causal',
                'relative_min', 'topicalised',
                'participial_len', 'participial_ta', 'participial_bali',
                'relative_post', 'refl_past',
                'reported_verb', 'reported_comp', 'reported',
                'noun_phrase_inh'):
        result = render_misc(ct, S, O, V, V_ACT, V_SUF, ADV, TAM, neg,
                             obl_strings, tree, m)

    elif ct == 'content_question':
        result = render_content_question(tree, m, S, O, V, TAM, obl_strings, G)

    elif ct == 'noun_phrase_have':
        result = render_noun_phrase_have(tree, S, O, neg, obl_strings, G)

    elif ct == 'comitative':
        result = render_comitative_clause(tree, S, V, TAM, obl_strings)

    elif ct == 'relative_nominal':
        result = tree.get('final_string', '')

    else:
        result = j(S, TAM, O, V, V_ACT, V_SUF, *obl_strings, ADV)

    # ── Marqueur temporel antéposé (quand/lorsque → tuma min, position FR) ────
    if ct == 'temporal' and tree.get('temporal_marker') and result:
        result = j(tree['temporal_marker'], result)

    print(f"  ✂️  Clause 1 -> '{result}'")
    return result.strip().replace(' ,', ',').replace('« ', '«').replace(' »', '»')
