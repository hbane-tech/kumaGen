"""
rules/steps/step2_sujet.py
Étape 2 : détection du sujet, _build_subj_chain, relatives sur le sujet,
verrou interrogatif, attribut négatif, expletif pronominal.
"""
import networkx as nx
from rules.core import j, apply_affixes, _is_avoir, _is_copula, _resolve_tam, get_bounded_chunk_tokens, adj_man, INTRANS_SC
from rules.steps.step5_obliques.advcl import _purp_verb_bm
from rules.kg_rule_engine import apply_morpho_suffix


def _build_subj_chain(tok, all_toks, G_kg=None):
    nmod = next((t for t in all_toks
                 if t.get('dep') == 'nmod'
                 and t.get('head_index') == tok['orig_index']), None)
    amod = next((t for t in all_toks
                 if t.get('dep') == 'amod'
                 and t.get('head_index') == tok['orig_index']
                 and t.get('bm')), None)
    poss_det = next((t for t in all_toks
                     if t.get('dep') == 'det'
                     and t.get('role') in ('pronoun', 'possessive')
                     and t.get('head_index') == tok['orig_index']
                     and t.get('bm')), None)
    tok_bm = tok.get('bm') or tok.get('surface') or f"[{tok.get('lemma')}]"
    _relational = (G_kg or {}).get('relational_bms', set())
    if amod:
        tok_bm = j(tok_bm, amod.get('bm'))
    if nmod:
        _nmod_priv = next((x for x in all_toks
                           if x.get('dep') == 'case'
                           and x.get('role') == 'privative'
                           and x.get('head_index') == nmod['orig_index']), None)
        if _nmod_priv:
            nmod_bm = nmod.get('bm') or nmod.get('surface') or f"[{nmod.get('lemma')}]"
            tok_bm  = j(tok_bm, nmod_bm + 'tan')
        else:
            nmod_bm = _build_subj_chain(nmod, all_toks, G_kg)
            # Priorité 1 : le sens_fr du HEAD contient déjà le lemme du nmod
            # → le HEAD encode le sens composé → garder tok_bm (ex: patte→ɲɛ́sen
            # avec sens_fr="patte de devant" alors que nmod=devant)
            _head_sensfr = tok.get('sens_fr', '').lower()
            _nmod_lemma  = nmod.get('lemma', '').lower()
            # Si le sens_fr du nmod contient déjà le lemme du parent
            # → le nmod encode un concept composé qui subsume le parent → pas de doublon
            # ex: venue→jɔ̀kun (sens_fr="raison de la venue") + raison→jó → doublon
            _nmod_sensfr = nmod.get('sens_fr', '').lower()
            _parent_lemma = tok.get('lemma', '').lower()
            if _head_sensfr and _nmod_lemma and _nmod_lemma in _head_sensfr:
                pass  # tok_bm déjà le bon sens composé → ne pas écraser avec nmod_bm
            elif _parent_lemma and _nmod_sensfr and _parent_lemma in _nmod_sensfr:
                tok_bm = nmod_bm  # nmod subsume le parent → utiliser nmod_bm seul
            else:
                _tok_is_rel = tok.get('is_relational', False) or tok_bm in _relational
                _gen = '' if _tok_is_rel else (G_kg.get('genitive_marker', '') or '')
                tok_bm = j(nmod_bm, _gen, tok_bm)
    if poss_det:
        poss_bm = poss_det.get('bm', '')
        _gen_mk = G_kg.get('genitive_marker', '') or ''
        tok_bm = j(G_kg.get('pron_1sg', 'n'), tok_bm) if poss_bm == G_kg.get('pron_1sg', 'n') else j(poss_bm, _gen_mk, tok_bm)
    return tok_bm


def run(T, tree, m, processed_indices, G_kg, NX_G, root_tok,
        clause_type_init, _has_expletive,
        _expletive_roles, _clitic_roles, _relative_roles):
    """
    Retourne (subj_tok, root_noun, has_acl, _has_relcl).
    Modifie tree/m/processed_indices en place.
    """

    # ── ATTRIBUT NÉGATIF ROOT ─────────────────────────────────────────────────
    _neg_attr_root = (root_tok
                      and root_tok.get('role') == 'negation'
                      and root_tok.get('pos') in ('PRON', 'NOUN', 'ADJ'))
    if _neg_attr_root:
        _real_s = next((x for x in T
                        if x.get('dep') in ('nsubj', 'expl:subj')
                        and x.get('pos') == 'PRON'
                        and x.get('bm')), None)
        if _real_s:
            m['S'] = _real_s.get('bm')
            processed_indices.add(_real_s['orig_index'])
        m['O'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
        processed_indices.add(root_tok['orig_index'])

    # ── EXPLETIF PRONOMINAL AVEC COPULE ──────────────────────────────────────
    _expl_pron_cop = next((x for x in T
                           if x.get('dep') in ('expl:subj', 'ROOT')
                           and x.get('role') in ('expletive',)
                           and x.get('pos') == 'PRON'
                           and x.get('bm')
                           and any(c.get('dep') in ('cop', 'dep') for c in T)
                           and root_tok
                           and root_tok.get('pos') in ('NOUN', 'ADJ', 'PRON')), None)
    if _expl_pron_cop and not m.get('S'):
        m['S'] = _expl_pron_cop.get('bm')
        processed_indices.add(_expl_pron_cop['orig_index'])

    # ── SUJET ─────────────────────────────────────────────────────────────────
    # Préférer un sujet non-demonstratif (évite que '-ce' vole la place de 'frères')
    # Guard impersonnel : quand expl:subj est présent ('il' impersonnel), les
    # pronoms clitiques datifs/accusatifs (role='object') mislabélés dep='nsubj'
    # par spaCy ne sont PAS le sujet réel ("il pourrait t'arriver" → t'=datif).
    _has_expl_subj_early = any(x.get('dep') == 'expl:subj' for x in T)
    subj_tok = (
        next((x for x in T if x.get('dep') in ('nsubj', 'nsubj:pass')
              and x.get('role') != 'demonstrative'
              and not (_has_expl_subj_early and x.get('role') == 'object')), None)
        or next((x for x in T if x.get('dep') in ('nsubj', 'nsubj:pass')
                 and not (_has_expl_subj_early and x.get('role') == 'object')), None)
    )

    # Correction ciblée : "Toi QUI prends..." → la recherche ci-dessus
    # attrape 'qui' (nsubj de la relative 'prends') au lieu du vrai sujet
    # 'Toi' (root_tok), parce que 'qui' a aussi dep='nsubj'. Ne corriger QUE
    # ce cas précis (subj_tok est le nsubj d'un acl:relcl qui a pour tête
    # root_tok lui-même) sans toucher à la recherche générale ci-dessus.
    if (subj_tok and root_tok and subj_tok is not root_tok
            and root_tok.get('pos') in ('PRON', 'NOUN')):
        _subj_tok_head = next((x for x in T
                               if x['orig_index'] == subj_tok.get('head_index')), None)
        if (_subj_tok_head and _subj_tok_head.get('dep') == 'acl:relcl'
                and _subj_tok_head.get('head_index') == root_tok['orig_index']):
            subj_tok = root_tok

    # ROOT NOUN/ADJ avec nsubj issu d'une clause adverbiale (advcl)
    # "Malheur quand il cligne les yeux" → root=malheur(NOUN), subj_tok=il
    # (nsubj de cligner dep=advcl). Le nsubj appartient à la subordonnée.
    # La clause principale a pour sujet le ROOT NOUN lui-même.
    if (subj_tok and root_tok and subj_tok is not root_tok
            and root_tok.get('pos') in ('NOUN', 'ADJ')
            and not _is_copula(root_tok)
            and not _is_avoir(root_tok)):
        _head_of_subj = next((x for x in T
                               if x['orig_index'] == subj_tok.get('head_index')), None)
        if (_head_of_subj and _head_of_subj.get('dep') in ('advcl', 'acl')
                and _head_of_subj.get('head_index') == root_tok['orig_index']):
            subj_tok = root_tok

    _expl_subj_tok = next((x for x in T if x.get('dep') == 'expl:subj'), None)
    _expl_comp_tok = next((x for x in T if x.get('dep') == 'expl:comp'), None)
    _avoir_root    = (root_tok and _is_avoir(root_tok)
                      and root_tok.get('pos') in ('VERB', 'AUX'))

    root_noun = None
    has_acl   = False

    if not _expl_subj_tok and _expl_comp_tok and _avoir_root:
        _vrai_subj_ya = next((x for x in T
                              if x.get('dep') in ('obj', 'nsubj')
                              and x.get('pos') in ('NOUN', 'PROPN', 'PRON')), None)
        if _vrai_subj_ya:
            subj_tok = None
            processed_indices.add(_expl_comp_tok['orig_index'])
            processed_indices.add(root_tok['orig_index'])
            _es_bm  = _vrai_subj_ya.get('bm') or f"[{_vrai_subj_ya.get('lemma')}]"
            _es_amod = next((x for x in T if x.get('dep') == 'amod'
                             and x.get('head_index') == _vrai_subj_ya['orig_index']), None)
            if _es_amod:
                _es_bm = j(_es_bm, _es_amod.get('bm') or f"[{_es_amod.get('lemma')}]")
                processed_indices.add(_es_amod['orig_index'])
            m['S'] = _es_bm
            processed_indices.add(_vrai_subj_ya['orig_index'])

    if _expl_subj_tok and not _expl_comp_tok:
        # expl:subj + cop + NOUN ROOT = faux positif spaCy : 'il' est le vrai sujet
        # Ex: "il n'est pas professeur" → expl:subj mais sujet réel (pas il-y-a)
        _has_cop_tok = any(x.get('dep') in ('cop', 'aux:pass') for x in T)
        if (_has_cop_tok and not _avoir_root
                and root_tok and root_tok.get('pos') in ('NOUN', 'ADJ', 'PROPN')
                and _expl_subj_tok.get('bm')):
            # Utiliser l'expl:subj comme sujet réel
            if not m.get('S'):
                m['S'] = _expl_subj_tok.get('bm', '')
            processed_indices.add(_expl_subj_tok['orig_index'])
        else:
            _vrai_subj = next((x for x in T
                               if x.get('dep') == 'obj'
                               and x.get('pos') in ('NOUN', 'PROPN')), None)
            if _vrai_subj:
                subj_tok = None
                _has_acl_exist = any(x.get('dep') == 'acl'
                                     and x.get('head_index') == _vrai_subj['orig_index']
                                     for x in T)
                if _has_acl_exist:
                    for _t in T:
                        if _t['orig_index'] == _vrai_subj['orig_index']:
                            _t['dep'] = 'ROOT'; _t['is_root'] = True
                    root_noun = _vrai_subj
                    has_acl   = True
                else:
                    _es_bm  = _vrai_subj.get('bm') or f"[{_vrai_subj.get('lemma')}]"
                    _es_amod = next((x for x in T if x.get('dep') == 'amod'
                                     and x.get('head_index') == _vrai_subj['orig_index']), None)
                    if _es_amod:
                        _es_bm = j(_es_bm, _es_amod.get('bm') or f"[{_es_amod.get('lemma')}]")
                        processed_indices.add(_es_amod['orig_index'])
                    m['S'] = _es_bm
                    processed_indices.add(_vrai_subj['orig_index'])
                    subj_tok = None
            processed_indices.add(_expl_subj_tok['orig_index'])
        if root_tok and root_tok.get('semantic_class') == 'having':
            processed_indices.add(root_tok['orig_index'])
        for _ec in T:
            if _ec.get('dep') == 'expl:comp':
                processed_indices.add(_ec['orig_index'])

    # Détrônement faux sujet nominal en question
    _a_un_point_interrog = any(
        '?' in str(x.get('surface', ''))
        or (x.get('dep') == 'punct' and str(x.get('surface', '')).strip() == '?')
        for x in T)
    if _a_un_point_interrog and subj_tok and subj_tok.get('pos') == 'NOUN':
        vrai_pron_sujet = next(
            (x for x in T if x.get('pos') == 'PRON'
             and x.get('role') not in _expletive_roles | _clitic_roles | {'demonstrative'}
             and str(x.get('surface', '')).strip() not in ('?', '.', '-')), None)
        if vrai_pron_sujet:
            subj_tok = vrai_pron_sujet

    if clause_type_init == 'content_question' and subj_tok and subj_tok.get('pos') == 'NOUN':
        vrai_pron = next(
            (x for x in T if x.get('pos') == 'PRON'
             and x.get('role') not in _expletive_roles | _clitic_roles
             and str(x.get('surface', '')).strip() not in ('?', '.', '-')), None)
        if vrai_pron:
            subj_tok = vrai_pron

    if subj_tok and str(subj_tok.get('surface', '')).strip() == '-':
        subj_tok = next((x for x in T if x.get('pos') == 'PRON'
                         and x.get('role') not in _expletive_roles | _clitic_roles
                         and x != subj_tok), None)

    if subj_tok and subj_tok.get('role') in _expletive_roles:
        _root_pron = next((x for x in T if x.get('pos') == 'PRON'
                           and x.get('dep') == 'ROOT'), None)
        if _root_pron:
            subj_tok = _root_pron

    if not subj_tok and not m.get('S'):
        _expl_subj_pron = next((x for x in T if x.get('dep') == 'expl:subj'
                                and x.get('pos') == 'PRON' and x.get('bm')), None)
        if _expl_subj_pron:
            m['S'] = _expl_subj_pron.get('bm')
            processed_indices.add(_expl_subj_pron['orig_index'])

    if not subj_tok and root_tok and root_tok.get('pos') in ('PRON', 'NOUN'):
        _root_has_nmod = any(x.get('dep') == 'nmod'
                             and x.get('head_index') == root_tok['orig_index'] for x in T)
        _root_has_amod = any(x.get('dep') in ('amod', 'conj')
                             and x.get('head_index') == root_tok['orig_index'] for x in T)
        # Déterminant distributif/quantificateur sur la racine → syntagme nominal
        # à rendre comme O (pas sujet) : 'chaque jour', 'tous les jours'
        _root_has_distrib_det = any(
            x.get('dep') == 'det'
            and x.get('head_index') == root_tok['orig_index']
            and x.get('role') in ('distributive_each', 'distributive_one', 'quantifier')
            for x in T)
        # Ne pas utiliser root_tok comme sujet si expletif + cop + NOUN :
        # root_tok est l'attribut (O), pas le sujet (ex: c'est la vérité)
        _has_cop_here = any(x.get('dep') == 'cop' for x in T)
        _is_expl_attr = (_has_expletive and _has_cop_here
                         and root_tok.get('pos') == 'NOUN')
        if (not _is_expl_attr
                and not _root_has_nmod
                and not _root_has_distrib_det
                and (not _root_has_amod
                     or root_tok.get('pos') not in ('NOUN', 'PROPN'))):
            subj_tok = root_tok

    if (not subj_tok and root_tok
            and root_tok.get('pos') not in ('NOUN', 'PROPN')
            and 'VerbForm=Inf' not in str(root_tok.get('morph', ''))):
        subj_tok = next((x for x in T if x.get('pos') in ('PRON', 'NOUN')
                         and x.get('head_index') == root_tok['orig_index']
                         and x != root_tok
                         and x.get('dep') not in ('dep', 'obj', 'iobj')
                         and not str(x.get('dep', '')).startswith('obl')
                         and x.get('role') not in ('object', 'object_pronoun')
                         # Skip nmod with locative case (prep + noun = prepositional phrase)
                         and not (x.get('dep') == 'nmod'
                                 and any(c.get('dep') == 'case'
                                        and c.get('role') == 'locative'
                                        and c.get('head_index') == x['orig_index']
                                        for c in T))), None)

    # ── ROOT NOUN + has_acl + has_relcl ──────────────────────────────────────
    if root_noun is None:
        root_noun = next((x for x in T
                          if x.get('pos') == 'NOUN' and x.get('dep') == 'ROOT'), None)
    _has_relcl = any(x.get('dep') == 'acl:relcl' for x in T)
    if not has_acl:
        has_acl = any(x.get('dep') == 'acl' for x in T) and not _has_relcl

    # ── RELATIVE NOMINALE : NOUN ROOT + acl:relcl → retour anticipé (SAUF si copule ROOT) ──
    # "l'homme qui est parti" → relative nominale pure (retour anticipé)
    # "L'homme que tu vois est mon ami" → relative + clause principale (NO retour anticipé)
    _has_copula_root = any(x.get('dep') == 'ROOT' and _is_copula(x) for x in T)
    if (root_noun and _has_relcl and not has_acl
            and not any(x.get('dep') == 'ROOT' and x.get('pos') == 'VERB' for x in T)
            and not _has_copula_root):
        _relcl_tok = next((x for x in T if x.get('dep') == 'acl:relcl'), None)
        if _relcl_tok:
            # Antécédent : nom auquel acl:relcl est attaché (ex: 'homme' pas 'ami')
            _antecedent_idx = _relcl_tok.get('head_index')
            _antecedent_tok = next((x for x in T if x.get('orig_index') == _antecedent_idx), None)
            _head_bm = (_antecedent_tok.get('bm') if _antecedent_tok
                        else root_noun.get('bm')) or f"[{root_noun.get('lemma')}]"

            _rel_subj   = next((x for x in T if x.get('dep') == 'nsubj'
                                and x.get('head_index') == _relcl_tok['orig_index']), None)
            _rel_obj    = next((x for x in T if x.get('dep') == 'obj'
                                and x.get('head_index') == _relcl_tok['orig_index']), None)
            _rel_adv    = next((x for x in T if x.get('dep') == 'advmod'
                                and x.get('head_index') == _relcl_tok['orig_index']), None)
            _rel_marker = G_kg.get('relative_marker', '') or ''
            _rel_s_bm   = _rel_subj.get('bm', '') if _rel_subj else ''
            _rel_v_bm   = _relcl_tok.get('bm', '')
            # Exclure _rel_o_bm si c'est le pronom relatif/interrogatif
            # (role='relative'|'interrogative') → évite le double mìn
            _rel_o_raw  = _rel_obj.get('bm', '') if _rel_obj else ''
            _rel_o_is_pron = (_rel_obj and
                              _rel_obj.get('role') in ('relative', 'interrogative'))
            _rel_o_bm   = '' if _rel_o_is_pron else _rel_o_raw
            _rel_adv_bm = _rel_adv.get('bm', '') if _rel_adv else ''
            _rel_tense  = _relcl_tok.get('tense', 'pres')
            _rel_intrans = (_relcl_tok.get('intransitive_type') == 'ABSOLU'
                            or _relcl_tok.get('semantic_class') in INTRANS_SC)
            if _rel_tense == 'past' and _rel_intrans and not _rel_o_bm:
                _morpho_res = G_kg.get('morpho_rules', {}).get('resultative', {})
                _rel_v_display = apply_morpho_suffix(_rel_v_bm, _morpho_res) if _rel_v_bm else _rel_v_bm
                _rel_tam = ''
            else:
                _rel_tam = _resolve_tam(_rel_tense, False, G_kg) or G_kg.get('equative_marker', '')
                # Nominalization pour les verbes transitifs sans objet dans la relative :
                # "la femme qui mange" → dúnli kɛ (manger ACTION, pas d'objet)
                # Même logique que _purp_verb_bm pour les clauses purposives.
                _rel_v_display = _purp_verb_bm(_relcl_tok, T, processed_indices, G_kg) if not _rel_o_bm else _rel_v_bm
            _rel_s_is_relative = _rel_subj and _rel_subj.get('role') == 'relative'
            _rel_s_display = '' if _rel_s_is_relative else _rel_s_bm
            # Obliques temporels du verbe relatif (ex: depuis plusieurs jours)
            _rel_obl_parts = []
            _prefix_markers = G_kg.get('temporal_prefix_markers', set()) or {'kabini', "k'an bɔ"}
            for _robl in sorted(T, key=lambda x: x['orig_index']):
                if _robl.get('dep') not in ('obl:mod', 'obl', 'obl:arg'):
                    continue
                if _robl.get('head_index') != _relcl_tok['orig_index']:
                    continue
                _robl_bm   = _robl.get('bm') or f"[{_robl.get('lemma', '')}]"
                _robl_case = next((x for x in T if x.get('dep') == 'case'
                                   and x.get('head_index') == _robl['orig_index']), None)
                # Quantificateur (plusieurs, quelques…) → après le nom en bambara
                _robl_quant = next((x for x in T
                                    if x.get('dep') in ('det', 'nummod')
                                    and x.get('head_index') == _robl['orig_index']
                                    and x.get('bm')), None)
                if _robl_quant:
                    _robl_bm = j(_robl_bm, _robl_quant.get('bm', ''))
                if _robl_case:
                    _mk = (_robl_case.get('bm_marker') or _robl_case.get('bm') or '')
                    if _mk.lower() in _prefix_markers or _mk.startswith('['):
                        _rel_obl_parts.append(j(_mk, _robl_bm))
                    elif _mk:
                        _rel_obl_parts.append(j(_robl_bm, _mk))
                    else:
                        _rel_obl_parts.append(_robl_bm)
                else:
                    _rel_obl_parts.append(_robl_bm)
            if _rel_s_display:
                _rel_str = j(_rel_s_display, _rel_tam, _head_bm, _rel_marker,
                             _rel_v_display, _rel_o_bm, _rel_adv_bm, *_rel_obl_parts)
            else:
                _rel_str = j(_head_bm, _rel_marker, _rel_tam,
                             _rel_v_display, _rel_o_bm, _rel_adv_bm, *_rel_obl_parts)
            # ROOT NOUN avec cop (ex: "L'homme que tu vois est mon ami")
            # → rel_str + ", o yé [poss][PRED] yé"
            _root_has_cop = (root_noun and any(
                x.get('dep') == 'cop'
                and x.get('head_index') == root_noun.get('orig_index')
                for x in T))
            if _root_has_cop:
                _pred_bm   = root_noun.get('bm', '')
                _pred_poss = next((x for x in T
                                   if x.get('dep') == 'det'
                                   and x.get('role') in ('possessive', 'pronoun')
                                   and x.get('head_index') == root_noun.get('orig_index')
                                   and x.get('bm')), None)
                _pred_full = j(_pred_poss.get('bm', ''), _pred_bm) if _pred_poss else _pred_bm
                _equ       = G_kg.get('equative_marker', 'yé') or 'yé'
                _cop_str   = j('o', _equ, _pred_full, _equ)
                _rel_str   = j(_rel_str, ',', _cop_str)
            print(f"  ✂️  Clause 1 -> '{_rel_str}'")
            # Signaler retour anticipé via sentinel
            return subj_tok, root_noun, has_acl, _has_relcl, _rel_str

    # ── CONSTITUTION DU SUJET dans m['S'] ────────────────────────────────────
    _is_pure_noun_phrase = (
        root_noun is not None and has_acl
        and subj_tok is not None
        and subj_tok.get('orig_index') == root_noun.get('orig_index')
    )

    if subj_tok and not _is_pure_noun_phrase and not m.get('S'):
        if subj_tok.get('pos') == 'PRON':
            m['S'] = subj_tok.get('bm') or f"[{subj_tok.get('lemma')}]"
            processed_indices.add(subj_tok['orig_index'])
        else:
            s_chunk = get_bounded_chunk_tokens(subj_tok['orig_index'],
                                               NX_G, processed_indices)
            # Exclure les sous-arbres advcl du sujet : ils seront traités par
            # step5_obliques.advcl. Sans cette garde, les descendants de l'advcl
            # (verbe, nsubj, objet) sont marqués traités ici et step5 les saute.
            # Ex: "Malheur quand il cligne les yeux" → cligne+il+quand+oeil
            # sont dans le sous-arbre de malheur(ROOT), mais appartiennent à advcl.
            _subj_advcl_subtree = set()
            for _at in T:
                if (_at.get('dep') == 'advcl'
                        and _at.get('head_index') == subj_tok['orig_index']
                        and _at['orig_index'] in NX_G.nodes):
                    _subj_advcl_subtree.add(_at['orig_index'])
                    _subj_advcl_subtree.update(nx.descendants(NX_G, _at['orig_index']))
            s_chunk = [t for t in s_chunk
                       if t['orig_index'] not in _subj_advcl_subtree
                       and t.get('dep') not in ('case', 'det')
                       and t.get('pos') not in ('PUNCT', 'SYM')
                       and not (t.get('is_root') and t != subj_tok)]

            nmod_s = next((x for x in s_chunk
                           if x.get('dep') == 'nmod'
                           and x.get('head_index') == subj_tok['orig_index']), None)
            if nmod_s:
                m['S'] = _build_subj_chain(subj_tok, T, G_kg)
                for _st in s_chunk:
                    processed_indices.add(_st['orig_index'])
            else:
                subj_bm = subj_tok.get('bm') or subj_tok.get('surface', '')
                if subj_tok.get('bm_suffix'):
                    subj_bm += subj_tok['bm_suffix']
                flat = next((x for x in T
                             if x.get('dep') in ('flat', 'flat:name')
                             and x.get('head_index') == subj_tok['orig_index']), None)
                if flat:
                    subj_bm = j(subj_bm, flat.get('bm') or flat.get('surface', ''))
                    processed_indices.add(flat['orig_index'])
                _s_amods = [x for x in s_chunk
                            if x.get('dep') == 'amod'
                            and x.get('head_index') == subj_tok['orig_index']]
                for _sa in _s_amods:
                    _sa_bm = _sa.get('bm') or f"[{_sa.get('lemma')}]"
                    if _sa.get('pos') == 'ADJ':
                        _sa_bm = adj_man(_sa_bm, is_classifying=_sa.get('is_classifying_adj', False))
                    subj_bm = j(subj_bm, _sa_bm)
                    processed_indices.add(_sa['orig_index'])
                _s_amod_indices = {a['orig_index'] for a in _s_amods} | {subj_tok['orig_index']}
                _root_orig = root_tok['orig_index'] if root_tok else -1
                # Find conjoined nouns: either as children of amods, root, or directly of subject
                _s_conjs = [x for x in T
                            if x.get('dep') == 'conj'
                            and x['orig_index'] not in processed_indices
                            and (x.get('head_index') in _s_amod_indices
                                 or x.get('head_index') == subj_tok['orig_index']  # Direct child of subject
                                 or (x.get('head_index') == _root_orig
                                     and x.get('pos') in ('NOUN', 'PROPN')
                                     and x['orig_index'] < _root_orig))]
                for _sc in _s_conjs:
                    _sc_cc = (next((x for x in T if x.get('dep') == 'cc'
                                    and x.get('head_index') == subj_tok['orig_index']), None)
                              or next((x for x in T if x.get('dep') == 'cc'
                                       and x.get('head_index') == _sc['orig_index']), None))
                    _cc_bm = _sc_cc.get('bm', '') if _sc_cc and _sc_cc.get('bm') else ''
                    if _sc_cc: processed_indices.add(_sc_cc['orig_index'])
                    subj_bm = j(subj_bm, _cc_bm, _sc.get('bm') or f"[{_sc.get('lemma')}]")
                    processed_indices.add(_sc['orig_index'])
                _subj_demo = next((x for x in T
                                   if x.get('dep') == 'det'
                                   and x.get('role') == 'demonstrative'
                                   and x.get('head_index') == subj_tok['orig_index']), None)
                if _subj_demo:
                    subj_bm = apply_affixes(_subj_demo.get('bm'), subj_bm, _subj_demo.get('bm_suffix'))
                    processed_indices.add(_subj_demo['orig_index'])
                _poss_det_s = next((x for x in T
                                    if x.get('dep') == 'det'
                                    and x.get('role') in ('pronoun', 'possessive')
                                    and x.get('head_index') == subj_tok['orig_index']
                                    and x.get('bm')), None)
                if _poss_det_s:
                    _poss_bm = _poss_det_s.get('bm', '')
                    subj_bm = j(_poss_bm, subj_bm)
                    processed_indices.add(_poss_det_s['orig_index'])
                _s_morph = str(subj_tok.get('morph', ''))
                _s_is_plur = subj_tok.get('is_plural') or 'Number=Plur' in _s_morph
                if _s_is_plur and not subj_bm.endswith(G_kg.get('plural_noun_suffix', '') or 'w'):
                    if subj_tok.get('pos') == 'PRON':
                        pass  # PRON : bm KG déjà pluriel
                    elif (subj_tok.get('pos') == 'PROPN'
                            and 'Number=Plur' in _s_morph):
                        subj_bm += 'w'  # peuple/ethnie PROPN pluriel (les bambara)
                    elif subj_tok.get('pos') != 'PROPN':
                        _plur_sb = G_kg.get('plural_noun_suffix', '')
                        if _plur_sb: subj_bm += _plur_sb
                # Nombre cardinal (nummod) → après le nom en bambara : mɔ̀gɔw fila
                _s_nummod = next((x for x in T
                                  if x.get('dep') == 'nummod'
                                  and x.get('head_index') == subj_tok['orig_index']
                                  and x.get('bm')), None)
                if _s_nummod:
                    subj_bm = j(subj_bm, _s_nummod.get('bm'))
                    processed_indices.add(_s_nummod['orig_index'])
                m['S'] = subj_bm
            processed_indices.update([t['orig_index'] for t in s_chunk])

    # ── VERROU INTERROGATIF ───────────────────────────────────────────────────
    _phrase_contient_interrogation = any(
        '?' in str(x.get('surface', ''))
        or (x.get('dep') == 'punct' and str(x.get('surface', '')).strip() == '?')
        for x in T)
    print(f"DEBUG _phrase_contient_interrogation={_phrase_contient_interrogation}")

    # Étendre aussi aux mots interrogatifs sans '?' (ex: "Pourquoi ne viens-tu pas")
    # Exclure SCONJ+mark : 'que/qu'' complémenteur (ccomp) n'est pas un mot interrogatif
    _has_interrog_word = any(
        x.get('role') == 'interrogative'
        and not (x.get('pos') == 'SCONJ' and x.get('dep') == 'mark')
        for x in T
    )
    if _phrase_contient_interrogation or _has_interrog_word:

        _vrai_pron_sujet = next(
            (x for x in T if x.get('pos') == 'PRON'
             and x.get('role') not in (_expletive_roles | _clitic_roles | _relative_roles
                                       | {'interrogative', 'demonstrative', 'reflexive',
                                          'possessive'})
             and x.get('dep') != 'expl:comp'
             and 'Int' not in str(x.get('morph', ''))
             and x.get('dep') in ('nsubj', 'nsubj:pass', 'dep')
             and str(x.get('surface', '')).strip() not in ('?', '.', '-')
             and str(x.get('surface', '')).isalnum()), None)

        if not _vrai_pron_sujet:
            _vrai_pron_sujet = next(
                (x for x in T if x.get('pos') == 'PRON'
                 and x.get('role') not in (_expletive_roles | _clitic_roles | _relative_roles
                                           | {'interrogative', 'demonstrative', 'reflexive',
                                              'possessive'})
                 and x.get('dep') not in ('obj', 'expl:comp')
                 and 'Int' not in str(x.get('morph', ''))
                 and str(x.get('surface', '')).strip() not in ('?', '.', '-')
                 # Exclure le 't' euphonique de l'inversion (Mange-t-il → -t = liaison phonologique)
                 and str(x.get('surface', '')).lstrip('-').lower() != 't'
                 and (str(x.get('surface', '')).isalnum()
                      or str(x.get('surface', '')).startswith('-'))), None)
        # Résoudre bm des pronoms inversés (-ils, -elle, -tu...) si vide
        if _vrai_pron_sujet and not _vrai_pron_sujet.get('bm'):
            _surf_clean = str(_vrai_pron_sujet.get('surface', '')).lstrip('-').lower()
            _bm_resolved = G_kg.get('funcs', {}).get((_surf_clean, 'fr'), {}).get('bm', '')
            if _bm_resolved:
                _vrai_pron_sujet = {**_vrai_pron_sujet, 'bm': _bm_resolved}

        if _vrai_pron_sujet:
            m['S'] = _vrai_pron_sujet.get('bm') or f"[{_vrai_pron_sujet.get('lemma')}]"
            if subj_tok and subj_tok['orig_index'] in processed_indices:
                processed_indices.remove(subj_tok['orig_index'])
            processed_indices.add(_vrai_pron_sujet['orig_index'])
            # "Qui aiment-ils?" : qui/que parsé nsubj (relatif) mais objet réel.
            # Si subj_tok initial est un pronom relatif/interrogatif distinct du
            # vrai sujet (-ils), l'utiliser comme objet de la question.
            if (subj_tok and subj_tok is not _vrai_pron_sujet
                    and subj_tok.get('role') in ('relative', 'interrogative')
                    and not m.get('O')):
                _who_bm = (subj_tok.get('bm')
                           or G_kg.get('interrogative_who', ''))
                if _who_bm:
                    m['O'] = _who_bm
                    processed_indices.add(subj_tok['orig_index'])
            if not m.get('O') and clause_type_init == 'content_question':
                _interrog_pron = next((x for x in T
                                       if x.get('role') == 'interrogative'
                                       and x['orig_index'] not in processed_indices), None)
                # Ne pas traiter un DET interrogatif ("quelle maison") ici :
                # step4 construit "nom jùmɛn" depuis o_chunk. Si on consomme
                # le DET ici, o_chunk ne le voit plus → "só" seul sans jùmɛn.
                # Exclure les interrogatifs ADVERBIAUX (pourquoi, où, quand, comment →
                # ADV/advmod) : ils ne sont pas COD. Seuls les pronoms nominaux
                # (qui, que → PRON/dep=obj) vont dans m['O'].
                if (_interrog_pron
                        and _interrog_pron.get('pos') not in ('DET', 'ADV')
                        and _interrog_pron.get('dep') not in ('det', 'advmod')):
                    m['O'] = (_interrog_pron.get('bm')
                              or G_kg.get('interrogative_who', ''))
                    processed_indices.add(_interrog_pron['orig_index'])
        else :
            if subj_tok and subj_tok.get('pos') == 'NOUN':
                if not m.get('S'):
                    # m['S'] not yet set — build from scratch including possessive
                    _poss_det = next((x for x in T
                                      if x.get('dep') == 'det'
                                      and x.get('role') in ('pronoun', 'possessive')
                                      and x.get('head_index') == subj_tok['orig_index']), None)
                    subj_bm = subj_tok.get('bm') or subj_tok.get('surface', '')
                    if _poss_det:
                        poss_val = _poss_det.get('bm') or f"[{_poss_det.get('lemma')}]"
                        _gen     = G_kg.get('genitive_marker', '') or ''
                        m['S']   = j(poss_val, subj_bm) if poss_val == 'n' else j(poss_val, _gen, subj_bm)
                        processed_indices.add(_poss_det['orig_index'])
                    else:
                        if subj_tok.get('bm_suffix'): subj_bm += subj_tok['bm_suffix']
                        _plur_sb = G_kg.get('plural_noun_suffix', '')
                        if (subj_tok.get('is_plural') and _plur_sb
                                and not subj_bm.endswith(_plur_sb)
                                and subj_tok.get('pos') not in ('PRON', 'PROPN')):
                            subj_bm += _plur_sb
                        m['S'] = subj_bm
                processed_indices.add(subj_tok['orig_index'])

    # ── CLAUSE RELATIVE SUR LE SUJET ─────────────────────────────────────────
    _relcl_on_subj = next((x for x in T
                           if x.get('dep') == 'acl:relcl'
                           and subj_tok
                           and x.get('head_index') == subj_tok['orig_index']), None)
    if _relcl_on_subj:
        _rel_subj = next((x for x in T if x.get('dep') == 'nsubj'
                          and x.get('head_index') == _relcl_on_subj['orig_index']), None)
        _rel_adv  = next((x for x in T if x.get('dep') == 'advmod'
                          and x.get('head_index') == _relcl_on_subj['orig_index']), None)
        _rel_aux  = next((x for x in T if x.get('dep') in ('aux', 'aux:tense')
                          and x.get('head_index') == _relcl_on_subj['orig_index']), None)
        _subj_bm    = subj_tok.get('bm') or subj_tok.get('surface', '')
        _rel_s_bm   = _rel_subj.get('bm', '') if _rel_subj else ''
        # avoir + NOUN obj dans une relative → acquisition : bm = sɔrɔ
        _relcl_has_noun_obj = any(x.get('dep') == 'obj'
                                  and x.get('pos') in ('NOUN', 'PROPN')
                                  and x.get('head_index') == _relcl_on_subj['orig_index']
                                  for x in T)
        _rel_v_bm   = ('sɔrɔ'
                       if (_relcl_on_subj.get('lemma', '').lower() == 'avoir'
                           and _relcl_has_noun_obj)
                       else _relcl_on_subj.get('bm', ''))
        _rel_adv_bm = _rel_adv.get('bm', '') if _rel_adv else ''
        _rel_marker = G_kg.get('relative_marker', '') or ''
        _rel_intrans = (_relcl_on_subj.get('intransitive_type') == 'ABSOLU'
                        or _relcl_on_subj.get('semantic_class') in INTRANS_SC)
        _rel_tense   = _relcl_on_subj.get('tense', 'pres')
        if _rel_tense == 'past' and _rel_intrans:
            _rel_tam = ''
        else:
            _rel_tam = _resolve_tam(_rel_tense, False, G_kg) or G_kg.get('equative_marker', '')
        _avoir_relcl_override = (_relcl_on_subj.get('lemma', '').lower() == 'avoir'
                                 and _relcl_has_noun_obj)
        # avoir + NOUN obj en passé → passé composé transitif : mìn yé dén sɔrɔ
        # (pas de suffixe -la : sɔrɔ est transitif, le TAM yé prime sur _rel_intrans)
        if _avoir_relcl_override and _rel_tense == 'past':
            _rel_tam = _resolve_tam(_rel_tense, False, G_kg) or G_kg.get('equative_marker', '')
            _rel_v_display = _rel_v_bm
        elif _rel_tense == 'past' and _rel_intrans and _rel_v_bm:
            _morpho_res    = G_kg.get('morpho_rules', {}).get('resultative', {})
            _rel_v_display = apply_morpho_suffix(_rel_v_bm, _morpho_res)
        elif _avoir_relcl_override:
            _rel_v_display = _rel_v_bm
        else:
            _rel_v_display = _purp_verb_bm(_relcl_on_subj, T, processed_indices, G_kg)
        _rel_s_is_relative = (_rel_subj and _rel_subj.get('role') == 'relative'
                              and _rel_subj.get('dep') in ('nsubj', 'obj')
                              and _rel_subj.get('head_index') == _relcl_on_subj['orig_index']
                              and str(_rel_subj.get('surface', '')).lower()
                              in (G_kg.get('relative_pron_surfaces') or {'qui', 'que', 'which', 'whom'}))
        _rel_s_display = '' if _rel_s_is_relative else _rel_s_bm

        # ── Objet de la relative ─────────────────────────────────────────────
        _rel_obj = next((x for x in T if x.get('dep') == 'obj'
                         and x.get('head_index') == _relcl_on_subj['orig_index']), None)
        _rel_obj_bm = ''
        if _rel_obj:
            # Supprimer le bm si c'est le pronom relatif (que/qui) → évite le double mìn
            _rel_obj_is_pron = (
                _rel_obj.get('role') in ('relative', 'interrogative')
                or str(_rel_obj.get('surface', '')).lower()
                   in (G_kg.get('relative_pron_surfaces') or {'que', 'qui', 'qu', 'which', 'whom', 'who'}))
            if not _rel_obj_is_pron:
                _rel_obj_bm = _rel_obj.get('bm') or f"[{_rel_obj.get('lemma')}]"
                _poss_rel_obj = next((x for x in T
                                      if x.get('dep') == 'det'
                                      and x.get('role') in ('pronoun', 'possessive')
                                      and x.get('head_index') == _rel_obj['orig_index']
                                      and x.get('bm')), None)
                if _poss_rel_obj:
                    _rel_obj_bm = j(_poss_rel_obj.get('bm'), _rel_obj_bm)
                    processed_indices.add(_poss_rel_obj['orig_index'])
                # amod sur l'objet de la relative (ex: l'ennemi VIVANT) :
                # bambara = nom + adjectif, jamais reflété sinon (perdu).
                _rel_obj_amod = next((x for x in T if x.get('dep') == 'amod'
                                      and x.get('head_index') == _rel_obj['orig_index']
                                      and x.get('bm')), None)
                if _rel_obj_amod:
                    _rel_obj_bm = j(_rel_obj_bm, _rel_obj_amod.get('bm'))
                    processed_indices.add(_rel_obj_amod['orig_index'])
                if (_rel_obj.get('is_plural') and not _rel_obj_bm.endswith(G_kg.get('plural_noun_suffix', '') or 'w')
                        and _rel_obj.get('pos') not in ('PRON', 'PROPN')):
                    _plur_ro = G_kg.get('plural_noun_suffix', '')
                    if _plur_ro: _rel_obj_bm += _plur_ro
            processed_indices.add(_rel_obj['orig_index'])

        # Structure bambara : S TAM O+min V ADV (objet relativé suivi du marqueur min)
        if _rel_s_display:
            _topic_str = j(_rel_s_display, _rel_tam, _subj_bm, _rel_marker,
                           _rel_obj_bm, _rel_v_display, _rel_adv_bm)
        else:
            _topic_str = j(_subj_bm, _rel_marker, _rel_tam,
                           _rel_obj_bm, _rel_v_display, _rel_adv_bm)

        m['S'] = _topic_str
        tree['_is_relative_topic'] = True  # signal → PatternRule KG
        processed_indices.add(_relcl_on_subj['orig_index'])
        if _rel_subj: processed_indices.add(_rel_subj['orig_index'])
        if _rel_adv:  processed_indices.add(_rel_adv['orig_index'])
        if _rel_aux:  processed_indices.add(_rel_aux['orig_index'])

    return subj_tok, root_noun, has_acl, _has_relcl, None
