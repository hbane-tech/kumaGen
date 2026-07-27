"""
rules/steps/step4_objet/free_relative.py
Relative libre ("tout ce que S V(O)...") : ROOT = PRON démonstratif (ce/ça)
avec un acl:relcl rattaché DIRECTEMENT dessus (pas à un NOUN antécédent).
Contrairement à une relative adjointe classique (relcl_post.py, antécédent =
un NOM ailleurs dans la phrase), ici la relative EST le contenu principal de
l'énoncé : son propre sujet ("mes pères et grands-pères") est distinct de
l'antécédent démonstratif, et doit être construit comme un sujet normal
(avec coordination + possessif distribué), pas laissé implicite comme dans
une relative sur antécédent nominal (bug trouvé 2026-07-26 : "tout ce que mes
pères... m'avaient enseigné..." perdait entièrement le sujet coordonné et
produisait un 'min' dupliqué + un objet orphelin, faute de toute prise en
charge dédiée — ni le chemin content_question (exige ROOT interrogatif), ni
la relative nominale (exige ROOT=NOUN), ni relcl_post (traite l'acl:relcl
comme adjonction oblique d'un antécédent nominal absent ici) ne matchaient).
"""
from rules.core import j


def _build_coord_subj(head_tok, T, G_kg):
    """Sujet éventuellement coordonné ('père et grand-père'), possessif du
    premier conjoint distribué sur les suivants s'ils n'ont pas leur propre
    déterminant (spaCy n'attache 'mon' qu'au premier nom coordonné)."""
    # Import tardif : step2_sujet importe step5_obliques qui importe
    # step4_objet (comitative.py) → import circulaire si posé en tête de module.
    from rules.steps.step2_sujet import _build_subj_chain
    _head_bm = _build_subj_chain(head_tok, T, G_kg)
    _poss_det = next((x for x in T if x.get('dep') == 'det'
                      and x.get('role') in ('pronoun', 'possessive')
                      and x.get('head_index') == head_tok['orig_index']
                      and x.get('bm')), None)
    _consumed = {head_tok['orig_index']}
    if _poss_det:
        _consumed.add(_poss_det['orig_index'])
    _parts = [_head_bm]
    _relational_bms = (G_kg or {}).get('relational_bms', set())
    for _c in sorted((x for x in T if x.get('dep') == 'conj'
                      and x.get('head_index') == head_tok['orig_index']),
                     key=lambda x: x['orig_index']):
        _cc = next((x for x in T if x.get('dep') == 'cc'
                    and x.get('head_index') in (head_tok['orig_index'], _c['orig_index'])), None)
        if _cc:
            _consumed.add(_cc['orig_index'])
        _c_bm = _c.get('bm') or f"[{_c.get('lemma')}]"
        _c_has_own_det = any(x.get('dep') == 'det' and x.get('head_index') == _c['orig_index']
                             for x in T)
        if _poss_det and not _c_has_own_det:
            _c_is_rel = _c.get('is_relational', False) or _c_bm in _relational_bms
            _gen_mk = '' if _c_is_rel else (G_kg.get('genitive_marker', '') or '')
            _c_bm = j(_poss_det.get('bm', ''), _gen_mk, _c_bm)
        _parts.append(_cc.get('bm', '') if _cc else '')
        _parts.append(_c_bm)
        _consumed.add(_c['orig_index'])
    return j(*_parts), _consumed


def run(T, tree, m, processed_indices, G_kg, NX_G, root_tok):
    if not (root_tok and root_tok.get('pos') == 'PRON'
            and root_tok.get('role') == 'demonstrative'):
        return
    _rel_v = next((t for t in T
                   if t.get('dep') == 'acl:relcl' and t.get('pos') == 'VERB'
                   and t.get('head_index') == root_tok.get('orig_index')), None)
    if not _rel_v:
        return

    _consumed = {root_tok['orig_index'], _rel_v['orig_index']}

    # Antécédent ("ce") + quantifieur amod ("tout" → bɛɛ)
    _antecedent_bm = root_tok.get('bm') or root_tok.get('surface', '')
    _tout_amod = next((t for t in T if t.get('dep') == 'amod'
                       and t.get('head_index') == root_tok['orig_index']), None)
    if _tout_amod:
        _antecedent_bm = j(_antecedent_bm, _tout_amod.get('bm') or f"[{_tout_amod.get('lemma')}]")
        _consumed.add(_tout_amod['orig_index'])

    # 'que'/'qui' relativiseur : représente déjà l'antécédent au sein de la
    # relative (objet direct relativisé) — ne doit pas être retraduit à part.
    _relativizer = next((t for t in T if t.get('dep') in ('obj', 'nsubj')
                         and t.get('role') in ('relative', 'interrogative')
                         and t.get('head_index') == _rel_v['orig_index']), None)
    if _relativizer:
        _consumed.add(_relativizer['orig_index'])

    # Sujet propre de la relative (distinct de l'antécédent)
    _rel_subj = next((t for t in T if t.get('dep') == 'nsubj'
                      and t.get('head_index') == _rel_v['orig_index']), None)
    _subj_bm = ''
    if _rel_subj:
        _subj_bm, _subj_consumed = _build_coord_subj(_rel_subj, T, G_kg)
        _consumed |= _subj_consumed

    # Auxiliaire(s) de temps composé (avait/avaient...)
    for _aux in T:
        if _aux.get('dep') in ('aux', 'aux:tense', 'aux:pass') and _aux.get('head_index') == _rel_v['orig_index']:
            _consumed.add(_aux['orig_index'])

    # Complément datif de la relative (iobj : "m'avaient enseigné" → à moi)
    _rel_iobj = next((t for t in T if t.get('dep') == 'iobj'
                      and t.get('head_index') == _rel_v['orig_index']), None)
    _dative_str = ''
    if _rel_iobj and _rel_iobj.get('bm'):
        _dative_mk = G_kg.get('dative_marker', 'ma') or 'ma'
        _dative_str = j(_rel_iobj.get('bm', ''), _dative_mk)
        _consumed.add(_rel_iobj['orig_index'])

    from rules.core import _resolve_tam
    _tense = _rel_v.get('tense', 'pres')
    _tam = _resolve_tam(_tense, bool(_rel_v.get('is_neg')), G_kg)
    _v_bm = _rel_v.get('bm') or f"[{_rel_v.get('lemma')}]"

    _rel_marker = G_kg.get('relative_marker', '') or 'min'
    _prefix = j(_antecedent_bm, _rel_marker, _subj_bm, _tam, _dative_str, _v_bm)

    processed_indices.update(_consumed)
    tree['_is_free_relative'] = True
    tree['_free_rel_prefix'] = _prefix
