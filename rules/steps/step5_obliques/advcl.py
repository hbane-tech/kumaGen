"""
step5_obliques/advcl.py
Clauses adverbiales subordonnées :
  - purposive  : walasa ka V  (pour faire...)
  - privative  : V + bali     (sans faire...)
"""
from rules.core import j, _resolve_tam


_AUTONOMOUS_SC = {'saying', 'motion', 'perception', 'sound', 'communication',
                  'emission', 'having', 'copula', 'other',
                  'biological', 'posture', 'spontaneous', 'meteorological'}


def _purp_verb_bm(verb_tok, T, processed_indices):
    """Forme du verbe dans une clause subordonnée (purposive/privative).
    Les verbes transitifs sans COD → VERBli kɛ.
    Les verbes intransitifs → V (pas li kɛ).
    Les verbes autonomes → V (pas li kɛ).
    Les inconnus [V] → [V] kɛ.
    """
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
    _lemma = verb_tok.get('lemma', '').lower()

    # ── Verbe intransitif confirmé (ABSOLU) → V nu (pas li kɛ) ───────────────
    # (ex: travailler=báara, ABSOLU → walasa ka báara, sans li kɛ)
    if _it == 'ABSOLU':
        return base_bm

    # ── Classe sémantique autonome → V nu (pas li kɛ) ──────────────────────────
    # (motion, copula, perception, etc. sont intrinsèquement intransitifs)
    if _sc in _AUTONOMOUS_SC:
        return base_bm

    # ── Transitivité : verbe transitif sans COD → Vli kɛ ──────────────────────
    # (support=avoir, nominalized=certain nouns, ACTION=generic action verb)
    # MAIS : ne pas forcer li kɛ pour ACTION si semantic_class est autonome
    if _it in ('support', 'nominalized'):
        if _is_unknown:
            return base_bm + ' kɛ'
        return base_bm + 'li kɛ'

    # ── ACTION avec classe autonome = verbe intransitif → V nu ────────────────
    # (ex: 'venir'=motion type, classé ACTION par LLM, mais intransitif)
    if _it == 'ACTION' and _sc in _AUTONOMOUS_SC:
        return base_bm

    # ── ACTION générique : par défaut transitif → Vli kɛ ───────────────────────
    if _it == 'ACTION':
        if _is_unknown:
            return base_bm + ' kɛ'
        return base_bm + 'li kɛ'

    # ── Pas d'intransitive_type : défaut par classe sémantique ────────────────
    # Si classe non-autonome → Vli kɛ (prudence : suppose transitif)
    if _sc and _sc not in _AUTONOMOUS_SC:
        if _is_unknown:
            return base_bm + ' kɛ'
        return base_bm + 'li kɛ'

    # ── Classe inconnue : verbe connu → Vli kɛ par défaut ────────────────────
    #                      verbe inconnu → [V] kɛ
    if _is_unknown:
        return base_bm + ' kɛ'
    return base_bm + 'li kɛ'


def handle(tok_item, T, m, processed_indices, G_kg):
    # ── ADVCL DE PAROLE + ccomp → discours rapporté "ko ..." ────────────────
    # "disant qu'elle partait" → ko a bɛ fáɲi  (le verbe de parole est absorbé
    # par 'ko' ; on ne traduit pas 'disant' lui-même).
    if tok_item.get('semantic_class') in ('saying', 'communication'):
        _ccomp = next((x for x in T if x.get('dep') == 'ccomp'
                       and x.get('head_index') == tok_item['orig_index']), None)
        if _ccomp:
            _ko = G_kg.get('reported_intro', 'ko') or 'ko'
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
            _cc_tam = _resolve_tam(_ccomp.get('tense', 'pres'), _cc_neg, G_kg) or 'bɛ'
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
        _purp_marker = G_kg.get('purposive_marker', 'walasa ka') or 'walasa ka'
        _verb_bm     = _purp_verb_bm(tok_item, T, processed_indices)
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
            _cv_bm = _purp_verb_bm(_cv, T, processed_indices)
            _verb_bm = j(_verb_bm, 'ani', 'ka', _cv_bm)
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

    elif _mark.get('role') == 'privative':
        # 'sans + VERBE' (advcl) → clause subordonnée 'k'a sɔrɔ' NÉGATIVE :
        #   S VERB1 k'a sɔrɔ S ma [O] VERB2   (il est parti sans avertir)
        # Le 'sans' = non-réalisation → négation perfective 'ma'.
        # La forme verbale et l'objet suivent les règles standard.
        # ≠ 'sans + nom/pronom' qui garde le marqueur kɔ/tan (privatif.py).
        _priv_marker = G_kg.get('privative_verb_marker', "k'a sɔrɔ") or "k'a sɔrɔ"
        _neg_marker  = _resolve_tam('past', True, G_kg) or 'ma'
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
        _verb_bm = _purp_verb_bm(tok_item, T, processed_indices)
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
