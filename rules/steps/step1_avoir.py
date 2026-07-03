"""
rules/steps/step1_avoir.py
Extraction des slots AVOIR : possession, il y a, existential.

Le clause_type est déjà posé par apply_kg_patterns_early (PatternRule KG).
Ce step remplit uniquement les slots S / O / V et marque processed_indices.
"""
from rules.core import j, _is_avoir


def _resolve_poss_type(obj_tok):
    """Lit le possession_type LLM du token objet → type de marqueur bambara (toujours MAJUSCULE)."""
    _pt = str(obj_tok.get('possession_type', '')).upper()
    if _pt in ('MATERIAL', 'AGE', 'EXPERIENCER', 'STATIF', 'STATIF_HAB', 'EXPERIENCER_PAIN'):
        return _pt
    return 'ABSTRACT'


def _mark_aux(T, processed_indices):
    for x in T:
        if x.get('dep') in ('aux:tense', 'aux:pass', 'det', 'fixed'):
            processed_indices.add(x['orig_index'])


def _mark_expletives(T, processed_indices):
    for x in T:
        if x.get('dep') in ('expl:subj', 'expl:comp'):
            processed_indices.add(x['orig_index'])


def _mark_det(T, head_idx, processed_indices, deps=('det', 'fixed')):
    for x in T:
        if x.get('dep') in deps and x.get('head_index') == head_idx:
            processed_indices.add(x['orig_index'])


def run(T, tree, m, processed_indices, G_kg, root_tok):
    """Remplit S/O/V et possession_type selon le clause_type déjà décidé par KG."""

    ct = tree.get('clause_type', '')

    # ── Passé composé avoir + sensation/émotion : clause spéciale ─────────
    # (non couvert par PatternRule car condition composite aux + possession_type token)
    _has_composite_aux = any(
        x.get('dep') in ('aux:tense', 'aux:pass') and x.get('pos') == 'AUX'
        for x in T)
    # Structural guard : 'avoir' comme aux:tense d'un AUTRE verbe promu root_tok
    # (ex: root_tok redirigé vers 'faire' derrière "il a fait") ne fait pas de
    # root_tok le verbe "avoir" lui-même — ignorer la classification
    # semantic_class='having' potentiellement erronée du LLM dans ce cas.
    _root_is_other_verb_with_avoir_aux_0 = (
        str(root_tok.get('lemma', '')).lower() != 'avoir'
        and any(x.get('dep') in ('aux:tense', 'aux:pass')
                and str(x.get('lemma', '')).lower() == 'avoir'
                and x.get('head_index') == root_tok.get('orig_index')
                for x in T))
    if (_has_composite_aux and _is_avoir(root_tok)
            and not _root_is_other_verb_with_avoir_aux_0):
        _obj_sens = next((x for x in T
                          if x.get('dep') == 'obj'
                          and x.get('possession_type') in ('EXPERIENCER', 'STATIF')
                          and x.get('bm')), None)
        if _obj_sens:
            _subj = next((x for x in T
                          if x.get('dep') in ('nsubj', 'nsubj:pass') and x.get('bm')), None)
            m['O'] = _obj_sens.get('bm', '')
            m['S'] = _subj.get('bm', 'a') if _subj else 'a'
            m['V'] = ''
            tree['possession_type'] = str(_obj_sens.get('possession_type', '')).upper()
            tree['tense'] = 'past'
            processed_indices.add(root_tok['orig_index'])
            processed_indices.add(_obj_sens['orig_index'])
            if _subj:
                processed_indices.add(_subj['orig_index'])
            _mark_aux(T, processed_indices)
            return root_tok
        # Pas de sensation/émotion EXPERIENCER/STATIF (ex: "j'ai eu un mari" =
        # possession ABSTRACT/MATERIAL/AGE) : ne pas retourner, laisser la
        # logique de possession standard ci-dessous traiter l'objet.

    # ── Existential (il y a) — slots : S='', O=nom réel ──────────────────
    if ct in ('existential_absolute', 'existential_localized'):
        _real = next((x for x in T
                      if x.get('dep') == 'obj'
                      and x.get('pos') in ('NOUN', 'PROPN')), None)
        if _real:
            _bm = _real.get('bm') or f"[{_real.get('lemma')}]"
            if _real.get('is_plural') or str(_real.get('surface', '')).endswith('s'):
                if not _bm.endswith('w'):
                    _bm += 'w'
            m['S'] = ''
            m['O'] = _bm
            processed_indices.add(root_tok['orig_index'])
            processed_indices.add(_real['orig_index'])
            _mark_expletives(T, processed_indices)
            _mark_det(T, _real['orig_index'], processed_indices)
        return root_tok

    # ── Possession avoir — slots : S=sujet, O=objet possédé ──────────────
    if ct != 'noun_phrase_have':
        # LLM classified root as 'having' (posséder, détenir…) but no PatternRule fired.
        # Structural guard: a reflexive root ("se faire", "s'occuper"…) can never be
        # genuine possession — French has no "s'avoir" — so a reflexive marker on the
        # root is a strong signal the LLM 'having' classification is a misfire (e.g.
        # idiomatic "se faire" = "to happen", not "to have").
        _root_is_reflexive = any(
            x.get('dep') in ('expl:comp', 'expl:pass')
            and x.get('head_index') == root_tok.get('orig_index')
            for x in T)
        # Structural guard : si 'avoir' apparaît comme AUXILIAIRE DE TEMPS d'un
        # AUTRE verbe promu root_tok (ex: "Qu'est-ce qu'il a fait ?" → root_tok
        # redirigé vers 'faire', 'avoir' reste aux:tense enfant de 'faire'),
        # alors root_tok n'est PAS lexicalement "avoir" — la classification
        # semantic_class='having' du LLM sur root_tok est un misfire (même
        # défaut que le LLM de transitivité : reclassification non-déterministe
        # d'un verbe qui n'a rien à voir avec la possession).
        _root_lemma_is_avoir = str(root_tok.get('lemma', '')).lower() == 'avoir'
        _avoir_is_aux_of_root = any(
            x.get('dep') in ('aux:tense', 'aux:pass')
            and str(x.get('lemma', '')).lower() == 'avoir'
            and x.get('head_index') == root_tok.get('orig_index')
            for x in T)
        _root_is_other_verb_with_avoir_aux = (
            not _root_lemma_is_avoir and _avoir_is_aux_of_root)
        if (_is_avoir(root_tok) and not _root_is_reflexive
                and not _root_is_other_verb_with_avoir_aux):
            tree['clause_type'] = 'noun_phrase_have'
        else:
            return root_tok

    _obj = next((x for x in T
                 if x.get('dep') == 'obj'
                 and x.get('pos') in ('NOUN', 'PROPN', 'ADJ')
                 and x.get('bm')), None)
    if not _obj:
        # "tu as combien d'enfants ?" : combien (ADV, dep=obj) + enfant (NOUN, obl:arg)
        # → utiliser le NOUN pour déterminer possession_type
        _interrog_qty_tok = next((x for x in T
                                  if x.get('dep') == 'obj'
                                  and x.get('role') == 'interrogative'
                                  and x.get('pos') == 'ADV'), None)
        if _interrog_qty_tok:
            _obj = next((x for x in T
                         if x.get('dep') in ('obl:arg', 'nmod')
                         and x.get('head_index') == _interrog_qty_tok['orig_index']
                         and x.get('pos') in ('NOUN', 'PROPN')), None)
    _subj = next((x for x in T
                  if x.get('dep') in ('nsubj', 'nsubj:pass')
                  and x.get('bm')), None)
    if not (_obj and _subj):
        return root_tok

    # "avoir mal" → surface 'mal' lu depuis KG
    _avoir_mal_surf = G_kg.get('avoir_mal_fr', '')
    _is_avoir_mal = bool(_avoir_mal_surf and (
        str(_obj.get('surface', '')).lower() == _avoir_mal_surf.lower()
        or str(_obj.get('lemma', '')).lower() == _avoir_mal_surf.lower()))
    _poss_type = 'EXPERIENCER' if _is_avoir_mal else _resolve_poss_type(_obj)
    tree['possession_type'] = _poss_type

    _state_bm = _obj.get('bm') or f"[{_obj.get('lemma', '')}]"
    _subj_bm  = _subj.get('bm', '')

    if _poss_type == 'STATIF':
        m['O'] = _state_bm
        m['S'] = _subj_bm
        m['V'] = ''
        processed_indices.add(root_tok['orig_index'])
        processed_indices.add(_obj['orig_index'])
        processed_indices.add(_subj['orig_index'])
        _mark_det(T, _obj['orig_index'], processed_indices, ('det', 'fixed', 'case'))
        return root_tok

    if _poss_type == 'EXPERIENCER':
        _body = next((x for x in T
                      if x.get('dep') in ('obl', 'obl:arg', 'obl:mod', 'iobj')
                      and x.get('pos') in ('NOUN', 'PROPN')
                      and x.get('head_index') == root_tok['orig_index']
                      and x['orig_index'] not in processed_indices), None)
        processed_indices.add(root_tok['orig_index'])
        processed_indices.add(_obj['orig_index'])
        processed_indices.add(_subj['orig_index'])
        _mark_det(T, _obj['orig_index'], processed_indices, ('det', 'fixed', 'case'))
        if _body:
            _body_bm = _body.get('bm') or f"[{_body.get('lemma', '')}]"
            m['O'] = j(_subj_bm, _body_bm)
            m['S'] = _subj_bm
            m['V'] = ''
            tree['pain_bm'] = _state_bm
            tree['possession_type'] = 'EXPERIENCER_PAIN'
            _mark_det(T, _body['orig_index'], processed_indices, ('det', 'fixed', 'case'))
            processed_indices.add(_body['orig_index'])
        else:
            m['O'] = _state_bm
            m['S'] = _subj_bm
            m['V'] = ''
        return root_tok

    # AGE : O = nummod (le nombre), pas l'unité ("ans/saan" est dans le template)
    _root_idx = root_tok['orig_index'] if root_tok else -1
    if _poss_type == 'AGE':
        _nummod = next((x for x in T
                        if x.get('dep') == 'nummod'
                        and x.get('head_index') == _obj['orig_index']), None)
        if _nummod:
            _num_bm = _nummod.get('bm') or _nummod.get('surface', '')
            m['S'] = _subj_bm
            m['O'] = _num_bm
            m['V'] = ''
            processed_indices.add(_obj['orig_index'])
            processed_indices.add(_subj['orig_index'])
            processed_indices.add(_root_idx)
            processed_indices.add(_nummod['orig_index'])
            return root_tok
        # Pas de nummod → O = bm de l'unité (âge interrogatif: O vide, INTERROG rempli ailleurs)
        m['S'] = _subj_bm
        m['O'] = _obj.get('bm', '')
        m['V'] = ''
        processed_indices.add(_obj['orig_index'])
        processed_indices.add(_subj['orig_index'])
        processed_indices.add(_root_idx)
        return root_tok

    # MATERIAL / ABSTRACT : O = objet possédé (+ conjoints)
    _obj_bm = _obj.get('bm', '')
    if _obj.get('is_plural') or str(_obj.get('surface', '')).endswith('s'):
        if not _obj_bm.endswith('w'):
            _obj_bm += 'w'
    # Nummod sur l'objet (ex: deux haches → jélew fila)
    _nummod_on_obj = next((x for x in T
                           if x.get('dep') == 'nummod'
                           and x.get('head_index') == _obj['orig_index']
                           and x.get('bm')), None)
    if _nummod_on_obj:
        _obj_bm = j(_obj_bm, _nummod_on_obj.get('bm', ''))
        processed_indices.add(_nummod_on_obj['orig_index'])
    # Déterminant possessif (ton/mon/son…) sur l'objet haver :
    # "il eut ton appel" → _obj_bm = 'i ka wéle' (alienable) / 'i wéle' (inalienable)
    # _mark_det() marque le DET comme traité → step4 ne le verrait plus sinon.
    _poss_det_avoir = next((x for x in T
                            if x.get('dep') == 'det'
                            and x.get('role') in ('pronoun', 'possessive')
                            and x.get('head_index') == _obj['orig_index']
                            and x.get('bm')), None)
    if _poss_det_avoir:
        _poss_bm_av = _poss_det_avoir.get('bm', '')
        if _poss_bm_av:
            _is_rel_av = (_obj.get('is_relational', False)
                          or _obj_bm in G_kg.get('relational_bms', set()))
            _gen_av = '' if _is_rel_av else (G_kg.get('genitive_marker', '') or 'ka')
            _obj_bm = j(_poss_bm_av, _gen_av, _obj_bm)
    for _oc in T:
        if (_oc.get('dep') == 'conj'
                and _oc['orig_index'] not in processed_indices
                and (_oc.get('head_index') == _obj['orig_index']
                     or (_oc.get('head_index') == _root_idx
                         and _oc.get('pos') in ('NOUN', 'PROPN')
                         and _oc['orig_index'] > _obj['orig_index']))):
            _cc = (next((x for x in T if x.get('dep') == 'cc'
                         and x.get('head_index') == _obj['orig_index']), None)
                   or next((x for x in T if x.get('dep') == 'cc'
                            and x.get('head_index') == _oc['orig_index']), None))
            _cc_bm = _cc.get('bm', '') if _cc and _cc.get('bm') else ''
            _oc_bm = _oc.get('bm') or f"[{_oc.get('lemma', '')}]"
            _obj_bm = j(_obj_bm, _cc_bm, _oc_bm)
            processed_indices.add(_oc['orig_index'])
            if _cc:
                processed_indices.add(_cc['orig_index'])

    m['S'] = _subj_bm
    m['O'] = _obj_bm
    m['V'] = ''
    processed_indices.add(_obj['orig_index'])
    processed_indices.add(_subj['orig_index'])
    processed_indices.add(_root_idx)
    _mark_det(T, _obj['orig_index'], processed_indices)
    return root_tok
