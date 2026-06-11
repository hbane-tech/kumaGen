"""
rules/steps/step1_avoir.py
Étape 1-avoir : détection et traitement du verbe AVOIR (possession + il y a + existential).
Ne touche pas à ÊTRE — géré dans step6_copule.py.
"""
from rules.core import j, _is_avoir


def _resolve_poss_type(obj_tok):
    """Type de possession pour le marqueur (material→bóló vs abstract→fɛ).

    Basé UNIQUEMENT sur le possession_type posé sur le token objet par
    _detect_possession_type (LLM) : AGE / MATERIAL / ABSTRACT / PAIN.
    Aucune liste de mots codée en dur. Marche pour les NOMS (qui n'ont pas de
    semantic_class). Défaut (non détecté) → abstract (marqueur fɛ).
    """
    _pt = str(obj_tok.get('possession_type', '')).upper()
    if _pt == 'MATERIAL':
        return 'material'
    if _pt == 'AGE':
        return 'AGE'
    return 'abstract'  # ABSTRACT, PAIN, ou non détecté → fɛ


def run(T, tree, m, processed_indices, G_kg, root_tok):
    """
    Modifie tree/m/processed_indices en place si avoir détecté.
    Retourne root_tok (inchangé).
    """

    # ── IL Y A (semantic_class='having' hérité, avant fix _is_avoir) ──────────
    _is_il_ya_legacy = (
        root_tok
        and _is_avoir(root_tok)
        and root_tok.get('dep') == 'ROOT'
        and any(x.get('dep') in ('expl:subj', 'expl:comp') for x in T)
    )
    if _is_il_ya_legacy:
        _vrai_subj = next((x for x in T
                           if x.get('dep') == 'obj'
                           and x.get('pos') in ('NOUN', 'PROPN')), None)
        if _vrai_subj:
            _has_loc = any(
                x.get('dep') in ('obl', 'obl:mod', 'obl:arg')
                and any(p.get('dep') == 'case' and p.get('role') == 'locative'
                        for p in T if p.get('head_index') == x['orig_index'])
                for x in T)
            tree['clause_type'] = ('existential_localized' if _has_loc
                                   else 'existential_absolute')
            _es_bm = _vrai_subj.get('bm') or f"[{_vrai_subj.get('lemma')}]"
            if (_vrai_subj.get('is_plural')
                    or str(_vrai_subj.get('surface', '')).endswith('s')):
                if not _es_bm.endswith('w'):
                    _es_bm += 'w'
            m['S'] = ''
            m['O'] = _es_bm
            processed_indices.add(root_tok['orig_index'])
            processed_indices.add(_vrai_subj['orig_index'])
            for _expl in T:
                if _expl.get('dep') in ('expl:subj', 'expl:comp'):
                    processed_indices.add(_expl['orig_index'])
            for _dt in T:
                if (_dt.get('dep') in ('det', 'fixed')
                        and _dt.get('head_index') == _vrai_subj['orig_index']):
                    processed_indices.add(_dt['orig_index'])

    # ── AVOIR POSSESSION (_is_avoir = semantic_class='having') ────────────────
    _avoir_possession = (
        root_tok
        and _is_avoir(root_tok)
        and root_tok.get('pos') in ('VERB', 'AUX')
        and root_tok.get('dep') == 'ROOT'
        and not any(x.get('dep') in ('expl:comp', 'expl:subj') for x in T)
        and any(x.get('dep') == 'obj' for x in T)
        and not any(x.get('dep') == 'xcomp' for x in T)
    )
    if _avoir_possession:
        _obj_poss  = next((x for x in T
                           if x.get('dep') == 'obj'
                           and x.get('pos') in ('NOUN', 'PROPN')
                           and x.get('bm')), None)
        _subj_poss = next((x for x in T
                           if x.get('dep') == 'nsubj'
                           and x.get('pos') == 'PRON'
                           and x.get('bm')), None)
        if _obj_poss and _subj_poss:
            _poss_type = _resolve_poss_type(_obj_poss)
            tree['clause_type']    = 'noun_phrase_have'
            tree['possession_type'] = _poss_type
            _obj_bm = _obj_poss.get('bm')
            if (_obj_poss.get('is_plural')
                    or str(_obj_poss.get('surface', '')).endswith('s')):
                if not _obj_bm.endswith('w'):
                    _obj_bm += 'w'
            # Conj de l'objet
            _root_orig_av = root_tok['orig_index'] if root_tok else -1
            _obj_conjs = [x for x in T
                          if x.get('dep') == 'conj'
                          and x['orig_index'] not in processed_indices
                          and (x.get('head_index') == _obj_poss['orig_index']
                               or (x.get('head_index') == _root_orig_av
                                   and x.get('pos') in ('NOUN', 'PROPN')
                                   and x['orig_index'] > _obj_poss['orig_index']))]
            for _oc in _obj_conjs:
                _cc_oc    = (next((x for x in T if x.get('dep') == 'cc'
                                   and x.get('head_index') == _obj_poss['orig_index']), None)
                             or next((x for x in T if x.get('dep') == 'cc'
                                      and x.get('head_index') == _oc['orig_index']), None))
                _cc_bm_oc = _cc_oc.get('bm', '') if _cc_oc and _cc_oc.get('bm') else ''
                _oc_bm    = _oc.get('bm') or f"[{_oc.get('lemma', '')}]"
                _obj_bm   = j(_obj_bm, _cc_bm_oc, _oc_bm)
                processed_indices.add(_oc['orig_index'])
                if _cc_oc:
                    processed_indices.add(_cc_oc['orig_index'])
            m['S'] = _subj_poss.get('bm')
            m['O'] = _obj_bm
            m['V'] = ''
            processed_indices.add(_obj_poss['orig_index'])
            processed_indices.add(_subj_poss['orig_index'])
            processed_indices.add(root_tok['orig_index'])
            for _dt in T:
                if (_dt.get('dep') in ('det', 'fixed')
                        and _dt.get('head_index') == _obj_poss['orig_index']):
                    processed_indices.add(_dt['orig_index'])

    # ── IL Y A (via _is_avoir) ────────────────────────────────────────────────
    _is_il_ya = (
        root_tok
        and _is_avoir(root_tok)
        and root_tok.get('dep') == 'ROOT'
        and any(x.get('dep') in ('expl:subj', 'expl:comp') for x in T)
    )
    if _is_il_ya:
        _vrai_subj2 = next((x for x in T
                            if x.get('dep') == 'obj'
                            and x.get('pos') in ('NOUN', 'PROPN')), None)
        if _vrai_subj2:
            _has_loc2 = any(
                x.get('dep') in ('obl', 'obl:mod', 'obl:arg')
                and any(p.get('dep') == 'case' and p.get('role') == 'locative'
                        for p in T if p.get('head_index') == x['orig_index'])
                for x in T)
            tree['clause_type'] = ('existential_localized' if _has_loc2
                                   else 'existential_absolute')
            _es2_bm = _vrai_subj2.get('bm') or f"[{_vrai_subj2.get('lemma')}]"
            if (_vrai_subj2.get('is_plural')
                    or str(_vrai_subj2.get('surface', '')).endswith('s')):
                if not _es2_bm.endswith('w'):
                    _es2_bm += 'w'
            m['S'] = ''
            m['O'] = _es2_bm
            processed_indices.add(root_tok['orig_index'])
            processed_indices.add(_vrai_subj2['orig_index'])
            for _expl in T:
                if _expl.get('dep') in ('expl:subj', 'expl:comp'):
                    processed_indices.add(_expl['orig_index'])
            for _dt in T:
                if (_dt.get('dep') in ('det', 'fixed')
                        and _dt.get('head_index') == _vrai_subj2['orig_index']):
                    processed_indices.add(_dt['orig_index'])

    # ── AVOIR ROOT possession (fallback semantic_class) ───────────────────────
    _avoir_possession2 = (
        root_tok
        and _is_avoir(root_tok)
        and root_tok.get('pos') in ('VERB', 'AUX')
        and root_tok.get('dep') == 'ROOT'
        and not any(x.get('dep') in ('expl:comp', 'expl:subj') for x in T)
        and any(x.get('dep') == 'obj' for x in T)
        and not any(x.get('dep') == 'xcomp' for x in T)
    )
    if _avoir_possession2:
        _obj2  = next((x for x in T
                       if x.get('dep') == 'obj'
                       and x.get('pos') in ('NOUN', 'PROPN')
                       and x.get('bm')), None)
        _subj2 = next((x for x in T
                       if x.get('dep') in ('nsubj', 'nsubj:pass')
                       and x.get('bm')), None)
        if _obj2 and _subj2:
            _poss_type2 = _resolve_poss_type(_obj2)
            tree['clause_type']     = 'noun_phrase_have'
            tree['possession_type'] = _poss_type2
            _obj_bm2 = _obj2.get('bm')
            if (_obj2.get('is_plural')
                    or str(_obj2.get('surface', '')).endswith('s')):
                if not _obj_bm2.endswith('w'):
                    _obj_bm2 += 'w'
            _root_orig2 = root_tok['orig_index'] if root_tok else -1
            _obj_conjs2 = [x for x in T
                           if x.get('dep') == 'conj'
                           and x.get('pos') in ('NOUN', 'PROPN')
                           and (x.get('head_index') == _obj2['orig_index']
                                or (x.get('head_index') == _root_orig2
                                    and x['orig_index'] > _obj2['orig_index']))]
            for _oc2 in _obj_conjs2:
                _cc2    = next((x for x in T
                                if x.get('dep') == 'cc'
                                and (x.get('head_index') == _obj2['orig_index']
                                     or x.get('head_index') == _oc2['orig_index'])), None)
                _cc_bm2 = _cc2.get('bm', '') if _cc2 and _cc2.get('bm') else ''
                _oc2_bm = _oc2.get('bm') or f"[{_oc2.get('lemma', '')}]"
                if ((_oc2.get('is_plural') or str(_oc2.get('surface', '')).endswith('s'))
                        and not _oc2_bm.endswith('w')):
                    _oc2_bm += 'w'
                _obj_bm2 = j(_obj_bm2, _cc_bm2, _oc2_bm)
                processed_indices.add(_oc2['orig_index'])
                if _cc2:
                    processed_indices.add(_cc2['orig_index'])
            m['S'] = _subj2.get('bm')
            m['O'] = _obj_bm2
            m['V'] = ''
            processed_indices.add(_obj2['orig_index'])
            processed_indices.add(_subj2['orig_index'])
            processed_indices.add(root_tok['orig_index'])
            for _dt in T:
                if (_dt.get('dep') in ('det', 'fixed')
                        and _dt.get('head_index') == _obj2['orig_index']):
                    processed_indices.add(_dt['orig_index'])

    return root_tok