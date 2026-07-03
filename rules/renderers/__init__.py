"""
rules/renderers/__init__.py
Point d'entrée unique : tree_to_bambara()

Moteur de rendu pur — zéro règle linguistique hardcodée.
Toutes les valeurs bambara viennent du KG :
  - ClauseTemplate  → ordre des mots
  - FunctionWord    → marqueurs fixes (ko, wa, kɛ…)
  - MorphoRule      → suffixes morphologiques
  - G_kg            → marqueurs dynamiques (genitive_marker, equative_marker…)
"""
from rules.core import j, _GENITIVE_FALLBACK, _resolve_tam
from rules.kg_rule_engine import (lookup_clause_template,
                                   fill_template, apply_morpho_suffix)


def _fw(G, role, fallback=''):
    """Lit un FunctionWord depuis G_kg par son rôle."""
    for fw in (G.get('function_words') or []):
        if fw.get('role') == role:
            return fw.get('bm', fallback)
    return fallback


def tree_to_bambara(tree, G=None, grammar=None):
    G   = grammar or G or {}
    m   = tree['main']
    ct  = tree['clause_type']
    S   = m.get('S', '') or ''
    O   = m.get('O', '') or ''
    V   = m.get('V', '') or ''
    # Privative ROOT : tree['final_string'] est la bonne traduction (construite en step7)
    if tree.get('_is_privative'):
        return tree.get('final_string', '')

    V_ACT = m.get('V_ACTION', '') or ''
    V_SUF = m.get('V_SUFFIX', '') or ''
    ADV   = m.get('ADV', '')   or ''
    neg   = tree.get('neg', False)
    tn    = tree.get('tense', 'pres')

    # TAM posé par kg_gateway TransformRule — lu depuis tree
    tam_val = tree.get('tam', '')
    if (not tam_val or tam_val.strip() == '') and not tree.get('_obligation'):
        # Impératif/prohibitif : TAM vide est intentionnel (verbe nu) — ne pas refill
        if ct not in ('imperative', 'prohibitive'):
            tam_val = _resolve_tam(tn, neg, G) or G.get('tam_default', '')
    TAM = tam_val

    # ── Ccomp : template lu depuis le KG ─────────────────────────────────────
    _ccomp_data = m.get('CCOMP')
    _ccomp_str  = ''
    if isinstance(_ccomp_data, dict):
        _ko  = G.get('reported_intro', '')
        _typ = _ccomp_data.get('type', '')
        _ccomp_slots = {
            'S':   _ccomp_data.get('S', ''),
            'O':   _ccomp_data.get('O', ''),
            'V':   _ccomp_data.get('V', ''),
            'TAM': _ccomp_data.get('tam', ''),
            'ADJ': _ccomp_data.get('adj_bm', ''),
            'REF': _ccomp_data.get('ref_bm', ''),
            'PARTICLE': _ccomp_data.get('particle',
                        G.get('comparative_particle') or _fw(G, 'comparative_particle', '')),
        }
        if _typ == 'comparative':
            _tpl_key = 'ccomp_comparative_neg' if _ccomp_data.get('neg') else 'ccomp_comparative_pos'
            _tpl = lookup_clause_template(_tpl_key, G) or ''
            _ccomp_str = j(_ko, fill_template(_tpl, _ccomp_slots)) if _tpl else ''
        elif _typ == 'verbal':
            _tpl = lookup_clause_template('ccomp_verbal', G) or ''
            _ccomp_str = j(_ko, fill_template(_tpl, _ccomp_slots)) if _tpl else ''
        elif _typ == 'qualite':
            _tpl = lookup_clause_template('ccomp_qualite', G) or ''
            _ccomp_str = j(_ko, fill_template(_tpl, _ccomp_slots)) if _tpl else ''
        elif _ccomp_data.get('O'):
            _eq_mk = G.get('equative_marker') or _fw(G, 'f3_equative', '')
            _ccomp_slots['TAM'] = _eq_mk
            _tpl = lookup_clause_template('ccomp_equative', G) or ''
            _ccomp_str = j(_ko, fill_template(_tpl, _ccomp_slots)) if _tpl else ''

    # ── Wagons obliques : lecture depuis OBL_ALL, marqueurs depuis KG ────────
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
                    pref = suff = ''
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
            if c.get('MARKER_IS_PREFIX'):
                # Marqueur préfixe (kabini, k'an bɔ…) : lu depuis OBL_ALL
                full_chunk = j(marker, full_chunk)
            elif lct == 'privative':
                # Privatif : suffixe collé sans espace (tanli kɛ → tan)
                full_chunk = full_chunk + marker
            else:
                full_chunk = j(full_chunk, marker)
        obl_strings.append(full_chunk)

    # ── F6 : passé intransitif → résultatif ──────────────────────────────────
    _o_is_xcomp = m.get('O_IS_XCOMP', False)
    if tree.get('_passive_statif'):
        # Passif statif : S V+len TAM — délégué au dispatch KG générique
        ct = 'passive_statif'
        tree['clause_type'] = 'passive_statif'

    if (tn == 'past'
            and (not tree.get('is_transitive', True) or tree.get('_refl_pronominal_bypass'))
            and not _o_is_xcomp
            and ct not in ('reciprocal', 'refl_absolute', 'passive_statif', 'passive_statif_question')):
        if neg:
            TAM = _resolve_tam('past', True, G) or ''
        else:
            if V:
                _v_parts = V.split(' ')
                _root_v  = _v_parts[0]
                _rest_v  = _v_parts[1:]
                _morpho_res = G.get('morpho_rules', {}).get('resultative', {})
                if _morpho_res:
                    _root_v = apply_morpho_suffix(_root_v, _morpho_res)
                V = j(_root_v, *_rest_v)
            # Subordonnée temporelle ("quand X ...") : le résultatif prend le
            # préfixe d'antériorité 'tùn' (l'événement précède la référence
            # temporelle implicite), contrairement au résultatif d'une
            # principale simple qui reste sans TAM.
            TAM = (G.get('statif_hab_prefix', '') or 'tùn') if ct == 'temporal' else ''
    else:
        TAM = tam_val

    _tokens_ref = tree.get('_tokens', [])

    # ── F3 : résultatif passé (O_IS_XCOMP) ──────────────────────────────────
    result = ''
    _f3_cts = G.get('f3_clause_types', set()) or {'simple', 'complex', 'conditional', 'temporal', 'relative_post'}
    if (ct in _f3_cts
            and _o_is_xcomp and O and tn == 'past'
            and V and not V.endswith('ra') and not V.endswith('na')):
        _morpho_res = G.get('morpho_rules', {}).get('resultative', {})
        _f3_sep  = G.get('f3_separator', '')
        _eq_mk   = G.get('equative_marker', '')
        _f3_vres = apply_morpho_suffix(V, _morpho_res) if _morpho_res else V
        _f3_slots = {'S': S, 'V_RES': _f3_vres, 'O': O, 'EQ_MK': _eq_mk,
                     'OBL': obl_strings[0] if obl_strings else '', 'ADV': ADV}
        if obl_strings or ' ' in S:
            _tpl_f3 = lookup_clause_template('f3_resultative_obl', G)
            result  = fill_template(_tpl_f3, _f3_slots) if _tpl_f3 else j(S, *obl_strings) + _f3_sep + j(_f3_vres, O, _eq_mk)
        else:
            _tpl_f3 = lookup_clause_template('f3_resultative', G)
            result  = fill_template(_tpl_f3, _f3_slots) if _tpl_f3 else j(S, _f3_vres, O, _eq_mk)

    # ── Slots pour fill_template ──────────────────────────────────────────────
    QUAL = m.get('QUAL', '') or ''
    _obl0 = obl_strings[0] if obl_strings else ''
    _all_slots = {
        'S': S, 'O': O, 'V': V, 'TAM': TAM, 'ADV': ADV,
        'QUAL': QUAL, 'OBL': _obl0, 'V_ACT': V_ACT,
        'PHENOMENON': tree.get('meteo_bm', ''),
        'MOTION_VERB': tree.get('meteo_motion', ''),
        'PROPN':    m.get('PROPN', ''),
        'DEICTIC_MK':m.get('DEICTIC_MK') or G.get('deictique_marker', ''),
        'R':        m.get('R', ''),
        'C':        m.get('C', ''),
        'V_RES':    m.get('V_RES', ''),
        'V_NOM':    m.get('V_NOM', ''),
        'TAM_BASE': m.get('TAM_BASE', ''),
        'END':      m.get('END', ''),
        'ADJ':      m.get('ADJ', ''),
        'MARKER':   m.get('MARKER', ''),
        'SRC':      m.get('SRC', ''),
        'RESTRICT_MK': m.get('RESTRICT_MK', '') or G.get('restrictive_exclusive_marker', ''),
        'ATTR':     m.get('ATTR', ''),
        'COMPANION':m.get('COMPANION', ''),
        'CONTRAST': m.get('CONTRAST', ''),
        'ALREADY':  m.get('ALREADY', ''),
        'PAIN':     m.get('PAIN', ''),
        'INTERROG_QTY': m.get('INTERROG_QTY', ''),
        'O2':       m.get('O_COORD', ''),
        'Q_INTRO':  m.get('Q_INTRO') or G.get('question_marker_yala', ''),
        'INTERROG': m.get('INTERROG', ''),
        'ALT':      m.get('ALT', ''),
        'O_LOC':    m.get('O_LOC', ''),
        'REFL_PRON': tree.get('refl_pron', ''),
        'REFL_SELF': G.get('reflexive_self_marker', ''),
    }

    # ── Dispatch KG générique ─────────────────────────────────────────────────
    # Résidus complexes (logique Python non-triviale)
    _COMPLEX_CT = {
        'refl_absolute', 'relative_nominal', 'impersonal',
    }

    # ── Relative topic sans prédicat principal (Toi qui...) → juste {S} ────────────
    # "Toi qui prends l'ennemi vivant" = i mìn bɛ júgu ɲɛ́nama mɔ́n (vocatif/topique)
    if not result and ct == 'relative_topic' and not V and not O and not V_ACT:
        result = S

    # ── Relative topic + V_ACT (modal + xcomp : ne peut pas abandonner…) ───────
    # S contient déjà la relative → utiliser relative_topic_vact ({S}, o TAM V ka O V_ACT)
    # et non le template générique {S} mìn TAM O V qui doublerait le mìn de S.
    if not result and ct == 'relative_topic' and V_ACT:
        _tpl_vact = lookup_clause_template('relative_topic_vact', G) or ''
        if _tpl_vact and '{' in _tpl_vact:
            result = fill_template(_tpl_vact, _all_slots)

    # ── Relative topic : verbe principal intransitif passé → relative_topic_intrans ──
    if not result and ct == 'relative_topic' and tn == 'past' and not neg:
        if not tree.get('is_transitive', True) and V:
            _morpho_res = G.get('morpho_rules', {}).get('resultative', {})
            _v_res = apply_morpho_suffix(V, _morpho_res) if _morpho_res else V
            _all_slots['V_RES'] = _v_res
            _tpl_rel = lookup_clause_template('relative_topic_intrans', G) or ''
            if _tpl_rel and '{' in _tpl_rel:
                result = fill_template(_tpl_rel, _all_slots)

    if not result and ct not in _COMPLEX_CT:
        _polarity = '_neg' if neg else '_pos'
        # Essayer d'abord une variante tense-spécifique : equative_past_pos, qualitative_hab_neg…
        _tpl_key = ct + '_' + tn + _polarity
        _tpl = (lookup_clause_template(_tpl_key, G)
                or lookup_clause_template(ct + _polarity, G)
                or lookup_clause_template(ct, G)
                # quest_ce_que_modal : template pres comme base universelle (TAM slot rempli)
                or (lookup_clause_template(ct + '_pres' + _polarity, G)
                    if ct == 'quest_ce_que_modal' else '')
                or '')
        _tpl_placeholder = G.get('template_placeholder_prefix', '')
        _tpl_slot_open   = G.get('template_slot_open', '')
        if _tpl and (not _tpl_placeholder or not _tpl.startswith(_tpl_placeholder)) and (_tpl_slot_open in _tpl if _tpl_slot_open else '{' in _tpl):
            result = fill_template(_tpl, _all_slots)
            # Éviter duplication : obl_strings déjà présents dans le résultat via {ADV}/{OBL}
            _extra_obls = (obl_strings[1:]) if '{OBL}' in _tpl else obl_strings
            _extra_obls = [o for o in _extra_obls if o and o not in result]
            if _extra_obls:
                # Insérer avant le marqueur de question si présent
                _q_sfx_obl = _fw(G, 'question_suffix', '')
                _res_obl = result.rstrip()
                if _q_sfx_obl and _res_obl.endswith(_q_sfx_obl.rstrip()):
                    _body_obl = _res_obl[:-(len(_q_sfx_obl.rstrip()))].rstrip()
                    result = j(_body_obl, *_extra_obls, _q_sfx_obl)
                elif _res_obl.endswith('?'):
                    # Template avec '?' hardcodé (quest_ce_que_modal…) : insérer OBL avant ?
                    _body_obl = _res_obl[:-1].rstrip()
                    result = j(_body_obl, *_extra_obls, '?')
                else:
                    result = j(result, *_extra_obls)
            # ADV non inclus dans le template → injecter avant le marqueur de question
            if ADV and '{ADV}' not in _tpl:
                _q_sfx_adv = _fw(G, 'question_suffix', '')
                _res_adv = result.rstrip()
                if _q_sfx_adv and _res_adv.endswith(_q_sfx_adv.rstrip()):
                    _body_adv = _res_adv[:-(len(_q_sfx_adv.rstrip()))].rstrip()
                    result = j(_body_adv, ADV, _q_sfx_adv)
                elif _res_adv.endswith('?'):
                    result = j(_res_adv[:-1].rstrip(), ADV, '?')
                else:
                    result = j(result, ADV)
            if _ccomp_str:
                # Insérer ccomp AVANT le marqueur de question (wà ?) si présent
                _q_sfx = _fw(G, 'question_suffix', '')
                _res_stripped = result.rstrip()
                if _q_sfx and _res_stripped.endswith(_q_sfx.rstrip()):
                    _body = _res_stripped[:-(len(_q_sfx.rstrip()))].rstrip()
                    result = j(_body, _ccomp_str, _q_sfx)
                else:
                    result = j(result, _ccomp_str)
        else:
            result = j(S, TAM, O, V, V_ACT, V_SUF, *obl_strings, ADV)
            result = j(result, _ccomp_str)

    elif not result:
        if ct == 'refl_absolute':
            _refl_pron_default = G.get('reflexive_pron_default', '')
            _refl_self_marker  = G.get('reflexive_self_marker', '')
            _morpho_res = G.get('morpho_rules', {}).get('resultative', {})
            _refl_pron  = '' if tree.get('refl_semantic_class') == G.get('biological_sc_name', 'biological') else tree.get('refl_pron', _refl_pron_default)
            _refl_v     = tree.get('refl_verb') or V
            _refl_tam   = TAM
            _refl_self  = _refl_self_marker if tree.get('refl_yere') else ''
            _serial_pp  = tree.get('refl_serial_postpos', '')
            _is_bio_past = (tree.get('refl_semantic_class') == G.get('biological_sc_name', 'biological')
                            and _refl_tam == _resolve_tam('past', False, G))
            _all_slots.update({'REFL_PRON': _refl_pron, 'REFL_SELF': _refl_self,
                               'SERIAL_PP': _serial_pp,
                               'V_RES': apply_morpho_suffix(_refl_v, _morpho_res) if _morpho_res else _refl_v,
                               'V': _refl_v})
            if _is_bio_past:
                _tpl = lookup_clause_template('refl_absolute_bio_past', G)
                result = fill_template(_tpl, _all_slots) if _tpl else j(S, _all_slots['V_RES'], *obl_strings, ADV)
            elif V_ACT:
                _tpl = lookup_clause_template('refl_absolute_vact', G)
                result = fill_template(_tpl, _all_slots) if _tpl else j(S, _refl_tam, _refl_pron, _refl_self, _refl_v, V_ACT, _serial_pp, *obl_strings, ADV)
            else:
                _tpl = lookup_clause_template('refl_absolute_default', G)
                result = fill_template(_tpl, _all_slots) if _tpl else j(S, _refl_tam, _refl_pron, _refl_self, _refl_v, *obl_strings, ADV)

        elif ct in ('relative_nominal', 'impersonal'):
            result = tree.get('final_string', '')

        else:
            result = j(S, TAM, O, V, V_ACT, V_SUF, *obl_strings, ADV)

    # ── Coordination verbale → résultat {wa} clause2 ─────────────────────────
    _coord_marker = G.get('coord_verb_marker') or _fw(G, 'coord_verb', '')
    _action_sfx   = G.get('coord_action_suffix') or _fw(G, 'coord_action_suffix', '')
    if tree.get('conj_clauses') and result:
        for _cc in tree['conj_clauses']:
            if _cc.get('cc_role') == 'alternative':
                # Disjonctive "ou/ou bien" → wàlima S TAM O V, en préservant wà ? final
                _alt_marker = _cc.get('cc_bm') or 'wàlima'
                _cc_tam_alt = _cc.get('tam', TAM)
                _alt_str = j(S, _cc_tam_alt, _cc.get('O', ''), _cc.get('V', ''))
                if _alt_str:
                    _q_sfx_alt = G.get('question_suffix', '')
                    if _q_sfx_alt and result.endswith(_q_sfx_alt.rstrip()):
                        _body_alt = result[:-(len(_q_sfx_alt.rstrip()))].rstrip()
                        result = j(_body_alt, _alt_marker, _alt_str, _q_sfx_alt)
                    else:
                        result = j(result, _alt_marker, _alt_str)
                continue
            _cc_tam = _cc.get('tam', TAM)
            _cc_o   = _cc.get('O', '')
            _cc_v   = _cc.get('V', '')
            _cc_it  = _cc.get('intransitive_type', '')
            _cc_sc  = _cc.get('semantic_class', '')
            _cc_liq = _cc.get('is_liquid', False)
            _coord_nom_types = G.get('coord_intrans_types', set())
            _coord_excl_sc   = G.get('coord_exclude_sc', set())
            if (not _cc_o and _cc_v
                    and _cc_it in _coord_nom_types
                    and _cc_sc not in G.get('coord_intrans_sc', set())
                    and not (_cc_sc in _coord_excl_sc and _cc_liq)
                    and not _cc_v.endswith(tuple(G.get('coord_excl_verb_sfx', ())) + (_action_sfx,) if _action_sfx else tuple(G.get('coord_excl_verb_sfx', ())))):
                _nom_sfx = G.get('nominalization_verb_suffix', '')
                _cc_str = j(S, _cc_tam, _cc_v + _nom_sfx, _action_sfx)
            else:
                _cc_str = j(S, _cc_tam, _cc_o, _cc_v)
            if _cc_str:
                result = j(result, _coord_marker, _cc_str)

    # ── Marqueur conditionnel ─────────────────────────────────────────────────
    _cond_tok = next((t for t in _tokens_ref
                      if t.get('dep') == 'mark' and t.get('role') == 'conditional'), None)
    if _cond_tok and result:
        _cond_bm = tree.get('conditional_marker') or _cond_tok.get('bm') or G.get('conditional_marker', '')
        if not result.startswith(_cond_bm):
            result = j(_cond_bm, result)

    # ── Marqueur temporel ─────────────────────────────────────────────────────
    if tree.get('temporal_marker') and result:
        result = j(tree['temporal_marker'], result)

    # ── Négateur porteur de contenu ───────────────────────────────────────────
    _cont = tree.get('neg_continuative_bm')
    if _cont and result and _cont not in result.split():
        _q_sfx   = G.get('question_suffix', '')
        _q_sfx_b = G.get('question_suffix_bare', '')
        _suffixes = ((_q_sfx, ' ' + _q_sfx_b, _q_sfx_b) if _q_sfx
                     else ())
        for _suf in _suffixes:
            if _suf and result.endswith(_suf):
                result = result[:-len(_suf)] + ' ' + _cont + _suf
                break
        else:
            result = j(result, _cont)

    result = result.strip()
    for _from, _to in (G.get('typo_rules') or []):
        result = result.replace(_from, _to)
    return result
