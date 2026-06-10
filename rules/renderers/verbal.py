"""
rules/renderers/verbal.py
Clause types : simple, complex, conditional, temporal, relative_post,
               reported_comp, comitative (verbal), verb_serial,
               relative_topic, refl_past, reported_verb, reported,
               passive, imperative, prohibitive, participial_*,
               focus, exclamative, reciprocal, concessive, causal,
               relative_min, topicalised, ownership, infinitive,
               privative_pred, noun_phrase_inh, comitative (clause)
"""
from rules.core import j, INTRANS_SC
from rules.steps.step5_obliques.advcl import _purp_verb_bm


def render_conditional(tree, m, S, O, V, V_ACT, TAM, obl_strings, G):
    """
    ní  : condition simple  → ní S TAM O V [obliques]
    mána: éventualité       → S mána O V   [obliques]  (mána remplace TAM)
    """
    _marker = tree.get('conditional_marker', 'ní')
    if _marker == 'mána':
        # S mána [O] V [V_ACT] [obliques]
        if V_ACT:
            return j(S, 'mána', V, 'ka', V_ACT, *obl_strings)
        return j(S, 'mána', O, V, *obl_strings)
    else:
        # ní S TAM [O] V [V_ACT] [obliques]
        if V_ACT:
            return j('ní', S, TAM, V, 'ka', V_ACT, *obl_strings)
        return j('ní', S, TAM, O, V, *obl_strings)


def render_verb_serial(tree, m, S, O, V, V_ACT, TAM, obl_strings, G):
    _tokens_ref = tree.get('_tokens', [])
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
            _head_orig = next((t.get('orig_index') for t in _tokens_ref
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

    _com_str = j(_com_marker, (' ' + _com_marker + ' ').join(_com_heads)) if _com_heads else ''

    _xcomp_tok = next((t for t in _tokens_ref
                       if t.get('dep') == 'xcomp'
                       and t.get('pos') == 'VERB'
                       and t.get('bm') == V_ACT), None)
    _xcomp_needs_li = (_xcomp_tok
                       and _xcomp_tok.get('intransitive_type') in ('ACTION', 'nominalized', 'support')
                       and _xcomp_tok.get('semantic_class', '') not in INTRANS_SC)
    if O:
        return j(S, TAM, V, 'ka', O, _com_str, V_ACT, *_other_obls)
    elif _xcomp_needs_li:
        # Pas de COD → forme nominalisée Vli kɛ (même avec comitative)
        _xv = V_ACT + 'li' if (V_ACT and not V_ACT.endswith('li')) else V_ACT
        return j(S, TAM, V, 'ka', _xv, 'kɛ', _com_str, *_other_obls)
    return j(S, TAM, V, 'ka', V_ACT, _com_str, *_other_obls)


def render_simple(tree, m, S, O, V, V_ACT, V_SUF, ADV, TAM, ct,
                  neg, obl_strings, _ccomp_str, _o_is_xcomp, G):
    _tokens_ref = tree.get('_tokens', [])
    _contrast_tok = next((t for t in _tokens_ref
                          if t.get('role') == 'contrast' and t.get('bm')), None)
    _contrast_str = _contrast_tok.get('bm', '') if _contrast_tok else ''
    _raw_obls = m.get('OBL_ALL', [])
    _tmp_strs, _other_strs = [], []

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

    _com_obls = [(_ci, _obl) for _ci, _obl in enumerate(_raw_obls)
                 if isinstance(_obl, dict)
                 and _obl.get('local_clause_type') == 'comitative']
    _has_com_obl = bool(_com_obls)
    _root_tok_ref = next((t for t in _tokens_ref if t.get('is_root')), None)
    _root_sc  = (_root_tok_ref.get('semantic_class', '') if _root_tok_ref else '')
    _root_it  = (_root_tok_ref.get('intransitive_type', '') if _root_tok_ref else '')
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


    _is_pres      = (tree.get('tense', 'pres') in ('pres', 'hab'))
    _end_marker   = 'la' if _is_pres else 'kɛ'
    _v_already_la = V and V.endswith(' la')

    # Passé composé avec avoir : VerbForm=Part n'est PAS résultatif (≠ être+participe)
    _has_avoir_aux = any(
        t.get('dep') in ('aux', 'aux:tense')
        and str(t.get('lemma', '')).lower() == 'avoir'
        for t in _tokens_ref
    )

    # not _ccomp_str : un ccomp (ko…) tient lieu d'objet → pas de nominalisation
    # Vli la (sinon montrer QUE… donnerait 'ɲásili la' et larguerait le ko…).
    _is_action_verb = (V and not _v_already_la and not O and not _ccomp_str
                       and ct == 'simple'
                       and not _root_is_resultative
                       and _root_sc not in INTRANS_SC
                       and _root_sc != 'other'
                       and _root_it in ('ACTION', 'nominalized', 'support'))
    _is_intrans_com = (V and not _v_already_la and not O and _has_com_obl and ct == 'simple')
    # Verbe intransitif ACTION (∈ INTRANS_SC, ex: travailler=having) :
    #   présent    → V la   (n bɛ báara la)
    #   progressif → V kɛ   (n bɛ kà báara kɛ)
    #   perfectif  → V kɛ   (n yé/ma báara kɛ)
    # Les verbes ABSOLU (dormir) ne reçoivent ni 'la' ni 'kɛ'.
    # _ccomp_str non vide → le verbe a un complément de phrase (ko...) = objet implicite
    # → pas de suffixe -la (ex: dire que... → fɔ, pas fɔ la)
    _is_intrans_action = (V and not _v_already_la and not O and not _ccomp_str
                          and ct == 'simple'
                          and (not _root_is_resultative or _has_avoir_aux)
                          and _root_sc in INTRANS_SC
                          and _root_it == 'ACTION')

    # Verbe NOM-ACTION (báara=travailler…) : le bambara est un nom → support
    # 'la' (présent) / 'kɛ' (progressif/perfectif). Gaté sur le semantic_class
    # (KG, stable, indépendant de la transitivité française et des timeouts Ollama).
    # Les vrais verbes (saying=kúma…) ne sont PAS ici → restent nus.
    _is_support_nom = (V and not _v_already_la and not O and not _ccomp_str
                       and ct == 'simple'
                       and (not _root_is_resultative or _has_avoir_aux)
                       and _root_sc in ('having', 'consumption', 'preparation'))

    if _is_action_verb:
            _v_nom = V + 'li'
            if _has_com_obl:
                _com_strs = [obl_strings[_ci] for _ci, _obl in enumerate(_raw_obls)
                             if isinstance(_obl, dict)
                             and _obl.get('local_clause_type') == 'comitative'
                             and _ci < len(obl_strings)]
                _non_com  = [obl_strings[_ci] for _ci, _obl in enumerate(_raw_obls)
                             if isinstance(_obl, dict)
                             and _obl.get('local_clause_type') != 'comitative'
                             and _ci < len(obl_strings)]
                return j(S, TAM, _v_nom, _end_marker, *_com_strs, *_non_com, ADV)
            return j(S, TAM, _v_nom, _end_marker, *obl_strings, ADV)

    if _is_intrans_com:
        _v_nom = V + 'li'
        _com_parts = []
        for _ci, _obl in _com_obls:
            _bm = j(_obl.get('COMPOUND', ''), _obl.get('HEAD', '')) if _obl.get('COMPOUND') else _obl.get('HEAD', '')
            _com_parts.append(j('ni', _bm, 'yé'))
        _non_com = [obl_strings[_ci] for _ci, _obl in enumerate(_raw_obls)
                    if isinstance(_obl, dict)
                    and _obl.get('local_clause_type') != 'comitative'
                    and _ci < len(obl_strings)]
        return j(S, TAM, _v_nom, _end_marker, *_com_parts, *_non_com)

    if TAM in ('bɛ kà', 'tɛ kà'):
        # Progressif : 'kɛ' seulement pour les classes NOM-ACTION (having=báara…),
        # car le bambara est un nom (kà báara kɛ = en train de faire le travail).
        # Les vrais verbes (saying=kúma…) restent nus : a bɛ kà kúma.
        # Gardé sur le semantic_class (KG, stable) — robuste aux timeouts Ollama
        # (n'exige pas l'intransitive_type='ACTION' qui dépend du LLM).
        if _is_support_nom:
            return j(*_tmp_strs, S, TAM, O, V, 'kɛ', ADV, *_other_strs)
        return j(*_tmp_strs, S, TAM, O, V, ADV, *_other_strs)
    if _is_support_nom and TAM in ('bɛ', 'tɛ', 'tùn bɛ', 'tùn tɛ'):
        # Présent/habituel nom-action → n bɛ báara la / n tùn bɛ báara la
        return j(_contrast_str, S, TAM, V, 'la', V_ACT, V_SUF,
                 *_tmp_strs, ADV, *_other_strs, _ccomp_str, _already)
    # Perfectif intransitif ACTION : _root_is_resultative bloque _is_intrans_action
    # pour VerbForm=Part (passé composé). Le passé composé avec avoir n'est PAS
    # résultatif → bypasser le guard via _has_avoir_aux.
    # Futur simple : pas d'auxiliaire avoir → bypasser _has_avoir_aux
    # semantic_class (KG) est fiable même sans Ollama
    if _is_support_nom and TAM in ('yé', 'ma', 'tùn yé', 'tùn ma', 'bɛ na', 'tɛ na'):
        # Perfectif/futur nom-action → n yé báara kɛ / n ma báara kɛ / n bɛ na báara kɛ
        # (gaté sur semantic_class : robuste au passé composé et aux timeouts Ollama)
        return j(_contrast_str, S, TAM, V, 'kɛ', V_ACT, V_SUF,
                 *_tmp_strs, ADV, *_other_strs, _ccomp_str, _already)
    if _o_is_xcomp and O:
        if _already_pos == 'HEAD':
            return j(_already, _contrast_str, S, TAM, V, O, V_ACT, V_SUF,
                     *_tmp_strs, ADV, *_other_strs, _ccomp_str)
        return j(_contrast_str, S, TAM, V, O, V_ACT, V_SUF,
                 *_tmp_strs, ADV, *_other_strs, _ccomp_str, _already)
    if _already_pos == 'HEAD':
        return j(_already, _contrast_str, S, TAM, O, V, V_ACT, V_SUF,
                 *_tmp_strs, ADV, *_other_strs, _ccomp_str)
    return j(_contrast_str, S, TAM, O, V, V_ACT, V_SUF,
             *_tmp_strs, ADV, *_other_strs, _ccomp_str, _already)


def render_relative_topic(tree, m, S, O, V, V_ACT, obl_strings):
    _rel_neg = tree.get('neg', False)
    _rel_tam = 'tɛ' if _rel_neg else 'bɛ'
    if V_ACT:
        return j(S, ',', 'o', _rel_tam, V, 'ka', O, V_ACT, *obl_strings)
    V_past = V if V.endswith('ra') or V.endswith('na') else (
        V + 'na' if V.endswith('n') else V + 'ra')
    return j(S, ',', 'o', V_past, *obl_strings)


def render_comitative_clause(tree, S, V, TAM, obl_strings):
    _tokens = tree.get('_tokens', [])
    _avec = next((t for t in _tokens
                  if isinstance(t, dict)
                  and t.get('role') == 'preposition'
                  and t.get('dep') == 'case'), None)
    _comp = None
    if _avec:
        av_i = _avec.get('orig_index', -1)
        _cn = next((t for t in _tokens
                    if isinstance(t, dict)
                    and t.get('pos') in ('NOUN', 'PROPN', 'PRON')
                    and (t.get('head_index', -1) == av_i
                         or _avec.get('head_index', -1) == t.get('orig_index', -1))), None)
        if _cn:
            _cp = next((t for t in _tokens
                        if isinstance(t, dict)
                        and t.get('role') == 'possessive'
                        and t.get('head_index', -1) == _cn.get('orig_index', -1)), None)
            _bm = _cn.get('bm', '')
            if _cp:
                _p = _cp.get('bm', '')
                _comp = j('n', _bm) if _p == 'n' else j(_p, 'ka', _bm)
            else:
                _comp = _bm
    companion = _comp or V
    if companion and V and V != S:
        return j(S, 'ni', companion, TAM, V, *obl_strings)
    if companion:
        return j(S, 'ni', companion, 'dòn', *obl_strings)
    return j(S, 'dòn', *obl_strings)


def render_misc(ct, S, O, V, V_ACT, V_SUF, ADV, TAM, neg, obl_strings, tree, m):
    if ct == 'infinitive':
        # Comitatives go before V (part of root-verb argument group); others after.
        # Strip trailing ' yé' from comitatives: in infinitive context the equative
        # marker is wrong — food/object comitative is just "ni X", not "ni X yé".
        def _strip_ye(s):
            return s[:-len(' yé')] if s.endswith(' yé') else s
        _pre_v  = [_strip_ye(s) for s, c in zip(obl_strings, m.get('OBL_ALL', []))
                   if c.get('local_clause_type') == 'comitative']
        _post_v = [s for s, c in zip(obl_strings, m.get('OBL_ALL', []))
                   if c.get('local_clause_type') != 'comitative']
        # ADV (ex. participe '-tɔ' : en secouant → júnjuntɔ) après le verbe.
        return j('ka', O if O else S, *_pre_v, V, ADV, *_post_v)
    if ct == 'ownership':
        return j(S, 'yé', O, 'de', 'ta', 'ye')
    if ct == 'passive':
        return j(S, 'bɛ ka', V, *obl_strings)
    if ct in ('imperative', 'prohibitive'):
        # Transitivité : un verbe transitif sans COD se nominalise → Vli kɛ
        # (ne mange pas → kàna dúnli kɛ, pas kàna dún). Avec COD, verbe nu.
        _cmd_v = V
        if not O:
            _r = next((t for t in tree.get('_tokens', []) if t.get('is_root')), None)
            if _r:
                _cmd_v = _purp_verb_bm(_r, tree.get('_tokens', []), set())
        if ct == 'imperative':
            return j(O, _cmd_v, *obl_strings)
        return j('kàna', O, _cmd_v, *obl_strings)
    if ct == 'participial_to':
        return j(S, V, ADV, *obl_strings)
    if ct == 'focus':
        return j(S, 'de', TAM, O, V, *obl_strings)
    if ct == 'exclamative':
        return j(S, TAM, O, V, *obl_strings, 'dɛ !')
    if ct == 'reciprocal':
        # Pluriel → ɲɔgɔn (l'un l'autre) / Singulier → yɛrɛ (soi-même)
        _root_ref = next((t for t in tree.get('_tokens', []) if t.get('is_root')), None)
        _is_plural = _root_ref.get('is_plural', False) if _root_ref else False
        _refl_marker = 'ɲɔgɔn' if _is_plural else S + ' yɛrɛ'
        # Passé réciproque : S yé ɲɔgɔn V  (transitif → yé, pas -ra)
        if tree.get('tense') == 'past':
            _rec_tam = 'ma' if neg else 'yé'
            return j(S, _rec_tam, _refl_marker, V, *obl_strings)
        return j(S, TAM, _refl_marker, V, *obl_strings)
    if ct == 'concessive':
        return j(S, TAM, O, V, ADV, *obl_strings)
    if ct == 'causal':
        return j(S, TAM, O, V, ADV, *obl_strings)
    if ct == 'relative_min':
        return j(S, 'mìn', TAM, O, V, ',', 'ò', TAM, V, *obl_strings)
    if ct == 'topicalised':
        return j(O, ',', S, TAM, V, *obl_strings)
    if ct in ('participial_len', 'participial_ta', 'participial_bali'):
        return j(S, V, *obl_strings)
    if ct == 'relative_post':
        return j(S, 'mìn', TAM, O, V, *obl_strings)
    if ct == 'refl_past':
        return j(S, 'tun ye', V, O, *obl_strings, ADV)
    if ct in ('reported_verb', 'reported_comp', 'reported'):
        if neg:
            return j(S, 'ma', O, 'fɔ', ADV, *obl_strings)
        return j(S, TAM, O, V, ADV, *obl_strings)
    if ct == 'privative_pred':
        if not S:
            return j(O, *obl_strings)
        return j(S, O, 'dòn', *obl_strings)
    if ct == 'noun_phrase_inh':
        return j(O, *obl_strings) if obl_strings else j(S, O)
    # fallback
    return j(S, TAM, O, V, V_ACT, V_SUF, *obl_strings, ADV)
