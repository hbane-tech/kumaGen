"""
step5_obliques/advcl.py
Clauses adverbiales subordonnées :
  - purposive  : walasa ka V  (pour faire...)
  - privative  : V + bali     (sans faire...)
"""
from rules.core import j, _resolve_tam


# _AUTONOMOUS_SC chargé depuis KG via G_kg.get('autonomous_sc') dans chaque appel


def _purp_verb_bm(verb_tok, T, processed_indices, G_kg=None):
    """Forme du verbe dans une clause subordonnée (purposive/privative).
    Les verbes transitifs sans COD → VERBli kɛ (depuis KG).
    Les verbes intransitifs/autonomes → V nu.
    """
    G_kg = G_kg or {}
    _nom_sfx = G_kg.get('nominalization_verb_suffix', '')
    _act_sfx = G_kg.get('coord_action_suffix', '')
    _auto_sc = G_kg.get('autonomous_sc', set())

    base_bm = verb_tok.get('bm') or f"[{verb_tok.get('lemma')}]"
    _is_unknown = base_bm.startswith('[')

    _has_obj = any(
        x.get('dep') == 'obj'
        and x.get('head_index') == verb_tok['orig_index']
        and x['orig_index'] not in processed_indices
        for x in T
    )
    if _has_obj:
        return base_bm

    _it = verb_tok.get('intransitive_type', '')
    _sc = verb_tok.get('semantic_class', '')

    # Intransitif absolu → V nu
    if _it == 'ABSOLU':
        return base_bm

    # Classe autonome → V nu
    if _sc in _auto_sc:
        return base_bm

    # Liquide → V nu
    if _sc == 'consumption_liquid':
        return base_bm

    # Transitif sans COD → V+nominalization+kɛ (depuis KG MorphoRule)
    if _it in ('support', 'nominalized'):
        if _is_unknown:
            return base_bm + (' ' + _act_sfx if _act_sfx else '')
        return base_bm + _nom_sfx + (' ' + _act_sfx if _act_sfx else '')

    # ACTION + classe autonome → V nu
    if _it == 'ACTION' and _sc in _auto_sc:
        return base_bm

    # ACTION générique → Vli kɛ
    if _it == 'ACTION':
        if _is_unknown:
            return base_bm + (' ' + _act_sfx if _act_sfx else '')
        return base_bm + _nom_sfx + (' ' + _act_sfx if _act_sfx else '')

    # Classe non-autonome → Vli kɛ
    if _sc and _sc not in _auto_sc:
        if _is_unknown:
            return base_bm + (' ' + _act_sfx if _act_sfx else '')
        return base_bm + _nom_sfx + (' ' + _act_sfx if _act_sfx else '')

    # Défaut : verbe connu → Vli kɛ, inconnu → [V] kɛ
    if _is_unknown:
        return base_bm + (' ' + _act_sfx if _act_sfx else '')
    return base_bm + _nom_sfx + (' ' + _act_sfx if _act_sfx else '')



def handle(tok_item, T, m, processed_indices, G_kg):
    # ── ADVCL DE PAROLE + ccomp → discours rapporté "ko ..." ────────────────
    # "disant qu'elle partait" → ko a bɛ fáɲi  (le verbe de parole est absorbé
    # par 'ko' ; on ne traduit pas 'disant' lui-même).
    if tok_item.get('semantic_class') in ('saying', 'communication'):
        _ccomp = next((x for x in T if x.get('dep') == 'ccomp'
                       and x.get('head_index') == tok_item['orig_index']), None)
        if _ccomp:
            _ko = G_kg.get('reported_intro', '')
            _cc_subj = next((x for x in T
                             if x.get('dep') in ('nsubj', 'nsubj:pass')
                             and x.get('head_index') == _ccomp['orig_index']
                             and x.get('bm')), None)
            _neg_surfs = G_kg.get('neg_surfaces', set())
            _cc_neg = any(
                x.get('head_index') == _ccomp['orig_index']
                and x.get('dep') in ('advmod', 'fixed', 'mark')
                and (str(x.get('surface', '')).lower().rstrip("'") in _neg_surfs
                     or x.get('role') == 'negation')
                for x in T)
            _cc_tam = _resolve_tam(_ccomp.get('tense', 'pres'), _cc_neg, G_kg) or G_kg.get('tam_default', '')
            _cc_obj = next((x for x in T
                            if x.get('dep') in ('obj', 'iobj')
                            and x.get('head_index') == _ccomp['orig_index']
                            and x.get('bm')), None)
            _cc_v = _ccomp.get('bm') or f"[{_ccomp.get('lemma')}]"
            _reported = j(_ko,
                          _cc_subj.get('bm', '') if _cc_subj else '',
                          _cc_tam,
                          _cc_obj.get('bm', '') if _cc_obj else '',
                          _cc_v)
            m['OBL_ALL'].append({
                'HEAD': _reported, 'MARKER': '',
                'local_clause_type': 'simple', 'COMPOUND': '', 'MOD': '',
                'DEM_PREF': '', 'DEM_SUFF': '', 'DEP_TYPE': 'advcl',
                'COMPOUND_IS_QUANTIFIER': False, 'MARKER_IS_PREFIX': False,
            })
            # Marquer disant + toute la sous-arbre ccomp comme traités
            processed_indices.add(tok_item['orig_index'])
            processed_indices.add(_ccomp['orig_index'])
            for x in T:
                if x.get('head_index') == _ccomp['orig_index']:
                    processed_indices.add(x['orig_index'])
            return

    _mark = next((x for x in T if x.get('dep') == 'mark'
                  and x.get('head_index') == tok_item['orig_index']), None)
    if not _mark:
        return

    if _mark.get('role') == 'purposive':
        _purp_marker = G_kg.get('purposive_marker', '')
        _verb_bm     = _purp_verb_bm(tok_item, T, processed_indices, G_kg)
        _conj_verbs  = [x for x in T if x.get('dep') == 'conj'
                        and x.get('head_index') == tok_item['orig_index']
                        and x.get('pos') == 'VERB']
        for _cv in _conj_verbs:
            # Coordination de VERBES → connecteur 'ani' (pas 'ni' qui est pour les
            # noms), comme step3 pour les infinitifs coordonnés.
            # Le 'cc' (et) a souvent pour tête le 2e conjoint, pas le 1er : on le
            # cherche sur les deux têtes pour le marquer traité.
            _cc_tok = (next((x for x in T if x.get('dep') == 'cc'
                             and x.get('head_index') == _cv['orig_index']), None)
                       or next((x for x in T if x.get('dep') == 'cc'
                                and x.get('head_index') == tok_item['orig_index']), None))
            _cv_bm = _purp_verb_bm(_cv, T, processed_indices, G_kg)
            _verbal_coord = G_kg.get('verbal_coord_inf', '')
            _verb_bm = j(_verb_bm, _verbal_coord, _cv_bm)
            processed_indices.add(_cv['orig_index'])
            if _cc_tok:
                processed_indices.add(_cc_tok['orig_index'])
        m['OBL_ALL'].append({
            'HEAD': j(_purp_marker, _verb_bm), 'MARKER': '',
            'local_clause_type': 'simple', 'COMPOUND': '', 'MOD': '',
            'DEM_PREF': '', 'DEM_SUFF': '', 'DEP_TYPE': 'advcl',
            'COMPOUND_IS_QUANTIFIER': False, 'MARKER_IS_PREFIX': False,
        })
        processed_indices.add(tok_item['orig_index'])
        processed_indices.add(_mark['orig_index'])

    elif _mark.get('role') == 'comparative':
        # Clause comparative similitude/équative : "A est mortel comme B est mortel"
        # → "[main clause] i ko [B] yé [ADJ] yé"
        # Utilise le bm du marqueur (ex. 'i ko' pour 'comme') plutôt que comparative_particle
        # ('ka tɛmɛ' qui est réservé aux comparatifs d'inégalité plus/moins que).
        _comp_marker = _mark.get('bm') or G_kg.get('comparative_particle', '')
        _eq_tam = G_kg.get('equative_marker', '') or G_kg.get('tam_default', '')
        _advcl_subj = next((x for x in T
                            if x.get('dep') in ('nsubj', 'nsubj:pass')
                            and x.get('head_index') == tok_item['orig_index']
                            and x.get('bm')), None)
        _subj_bm = _advcl_subj.get('bm', '') if _advcl_subj else m.get('S', '')
        if _advcl_subj:
            _amod_on_subj = next((x for x in T
                                  if x.get('dep') == 'amod'
                                  and x.get('head_index') == _advcl_subj['orig_index']
                                  and x.get('bm')), None)
            if _amod_on_subj:
                # Ne pas appendre l'amod si le head encapsule déjà son sens
                # (modifier lemma dans sens_fr du head → composé directionnel)
                # OU si l'amod est un mauvais match embedding (son propre lemme
                # n'apparaît pas dans son sens_fr → traduction peu fiable).
                _head_sf = (_advcl_subj.get('sens_fr') or '').lower()
                _am_lem  = (_amod_on_subj.get('lemma') or '').lower()
                _am_surf = (_amod_on_subj.get('surface') or '').lower()
                _am_sf   = (_amod_on_subj.get('sens_fr') or '').lower()
                _head_encodes   = _am_lem in _head_sf or _am_surf in _head_sf
                _reliable_trans = _am_lem in _am_sf or _am_surf in _am_sf
                if not _head_encodes and _reliable_trans:
                    _subj_bm = j(_subj_bm, _amod_on_subj.get('bm', ''))
                processed_indices.add(_amod_on_subj['orig_index'])
        _adj_bm = tok_item.get('bm') or f"[{tok_item.get('lemma')}]"
        # Vérifier si prédicat adjectival : pos=ADJ OU présence d'un cop enfant
        _has_cop = any(x.get('dep') == 'cop'
                       and x.get('head_index') == tok_item['orig_index']
                       for x in T)
        if tok_item.get('pos') == 'ADJ' or _has_cop:
            _obl_str = j(_comp_marker, _subj_bm, _eq_tam, _adj_bm, _eq_tam)
        else:
            _neg = any(x.get('head_index') == tok_item['orig_index']
                       and (x.get('role') == 'negation'
                            or str(x.get('surface', '')).lower().rstrip("'")
                            in (G_kg.get('neg_surfaces') or set()))
                       for x in T)
            _tam = _resolve_tam(tok_item.get('tense', 'pres'), _neg, G_kg) or G_kg.get('tam_default', '')
            _v_obj = next((x for x in T if x.get('dep') in ('obj', 'iobj')
                           and x.get('head_index') == tok_item['orig_index']
                           and x.get('bm')), None)
            _obl_str = j(_comp_marker, _subj_bm, _tam,
                         _v_obj.get('bm', '') if _v_obj else '', _adj_bm)
            if _v_obj:
                processed_indices.add(_v_obj['orig_index'])
        m['OBL_ALL'].append({
            'HEAD': _obl_str, 'MARKER': '',
            'local_clause_type': 'simple', 'COMPOUND': '', 'MOD': '',
            'DEM_PREF': '', 'DEM_SUFF': '', 'DEP_TYPE': 'advcl',
            'COMPOUND_IS_QUANTIFIER': False, 'MARKER_IS_PREFIX': False,
        })
        processed_indices.add(tok_item['orig_index'])
        processed_indices.add(_mark['orig_index'])
        if _advcl_subj:
            processed_indices.add(_advcl_subj['orig_index'])

    elif (_mark.get('role') == 'temporal'
          or (_mark.get('pos') == 'SCONJ'
              and _mark.get('role') not in ('purposive', 'privative', 'comparative'))):
        # Clause temporelle : "quand il V O" → "tuma min S TAM O V"
        # Le marqueur temporal (bm de quand/lorsque) vient du token mark lui-même.
        _temp_marker = _mark.get('bm') or ''
        _advcl_subj = next((x for x in T
                            if x.get('dep') in ('nsubj', 'nsubj:pass')
                            and x.get('head_index') == tok_item['orig_index']
                            and x.get('bm')), None)
        _subj_bm = _advcl_subj.get('bm') if _advcl_subj else m.get('S', '')
        _neg = any(
            x.get('head_index') == tok_item['orig_index']
            and (x.get('role') == 'negation'
                 or (x.get('dep') in ('advmod', 'fixed', 'mark')
                     and str(x.get('surface', '')).lower().rstrip("'")
                     in (G_kg.get('neg_surfaces') or set())))
            for x in T)
        _tense = tok_item.get('tense', 'pres')
        _tam = _resolve_tam(_tense, _neg, G_kg) or G_kg.get('tam_default', '')
        _advcl_obj = next((x for x in T
                           if x.get('dep') in ('obj', 'iobj')
                           and x.get('head_index') == tok_item['orig_index']
                           and x.get('bm')
                           and x['orig_index'] not in processed_indices), None)
        _obj_bm = _advcl_obj.get('bm') if _advcl_obj else ''
        _verb_bm = tok_item.get('bm') or f"[{tok_item.get('lemma')}]"
        _obl_str = j(_temp_marker, _subj_bm, _tam, _obj_bm, _verb_bm)
        m['OBL_ALL'].append({
            'HEAD': _obl_str, 'MARKER': '',
            'local_clause_type': 'simple', 'COMPOUND': '', 'MOD': '',
            'DEM_PREF': '', 'DEM_SUFF': '', 'DEP_TYPE': 'advcl',
            'COMPOUND_IS_QUANTIFIER': False, 'MARKER_IS_PREFIX': False,
        })
        processed_indices.add(tok_item['orig_index'])
        processed_indices.add(_mark['orig_index'])
        if _advcl_subj:
            processed_indices.add(_advcl_subj['orig_index'])
        if _advcl_obj:
            processed_indices.add(_advcl_obj['orig_index'])

    elif _mark.get('role') == 'privative':
        # 'sans + VERBE' (advcl) → clause subordonnée 'k'a sɔrɔ' NÉGATIVE :
        #   S VERB1 k'a sɔrɔ S ma [O] VERB2   (il est parti sans avertir)
        # Le 'sans' = non-réalisation → négation perfective 'ma'.
        # La forme verbale et l'objet suivent les règles standard.
        # ≠ 'sans + nom/pronom' qui garde le marqueur kɔ/tan (privatif.py).
        _priv_marker = G_kg.get('privative_verb_marker', '')
        _neg_marker  = _resolve_tam('past', True, G_kg) or ''
        # Sujet de la subordonnée : son nsubj propre sinon le sujet principal
        # (infinitif partagé : 'sans avertir').
        _advcl_subj = next((x for x in T
                            if x.get('dep') in ('nsubj', 'nsubj:pass')
                            and x.get('head_index') == tok_item['orig_index']
                            and x.get('bm')), None)
        _subj_bm = _advcl_subj.get('bm') if _advcl_subj else m.get('S', '')
        # Objet éventuel du verbe subordonné → placé avant le verbe (S ma O V).
        _advcl_obj = next((x for x in T
                           if x.get('dep') in ('obj', 'iobj')
                           and x.get('head_index') == tok_item['orig_index']
                           and x.get('bm')
                           and x['orig_index'] not in processed_indices), None)
        _obj_bm = _advcl_obj.get('bm') if _advcl_obj else ''
        # Forme verbale standard (transitif sans COD → Vli kɛ, intransitif → V…)
        # Appelée AVANT de marquer l'objet traité pour que la détection
        # transitif/intransitif voie le COD.
        _verb_bm = _purp_verb_bm(tok_item, T, processed_indices, G_kg)
        m['OBL_ALL'].append({
            'HEAD': j(_priv_marker, _subj_bm, _neg_marker, _obj_bm, _verb_bm),
            'MARKER': '',
            'local_clause_type': 'simple', 'COMPOUND': '', 'MOD': '',
            'DEM_PREF': '', 'DEM_SUFF': '', 'DEP_TYPE': 'advcl',
            'COMPOUND_IS_QUANTIFIER': False, 'MARKER_IS_PREFIX': False,
        })
        processed_indices.add(tok_item['orig_index'])
        processed_indices.add(_mark['orig_index'])
        if _advcl_subj:
            processed_indices.add(_advcl_subj['orig_index'])
        if _advcl_obj:
            processed_indices.add(_advcl_obj['orig_index'])
