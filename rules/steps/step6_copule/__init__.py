"""rules/steps/step6_copule/__init__.py — orchestrateur copule/être."""
from rules.steps.step6_copule import etre_root
from rules.steps.step6_copule import presentative
from rules.steps.step6_copule import statif
from rules.steps.step6_copule import participes
from rules.steps.step6_copule import equatif
from rules.steps.step6_copule import qualitative
from rules.steps.step6_copule import identificatoire
from rules.steps.step6_copule import participial_to
from rules.core import _is_copula, _resolve_tam


def run(T, tree, m, processed_indices, G_kg, root_tok,
        _has_expletive, aux_tense_tok, clause_type_init):

    # Simultané -tɔ (indépendant de la copule)
    participial_to.run(T, tree, m, processed_indices, G_kg, root_tok)

    _has_obj      = any(x.get('dep') == 'obj' for x in T)
    _has_aux_pass = any(x.get('dep') == 'aux:pass' for x in T)
    _has_aux_cop  = any(x.get('dep') in ('aux', 'aux:tense') and _is_copula(x) for x in T)
    tree['is_transitive'] = True if _has_obj else (
        False if (_has_aux_pass or _has_aux_cop) else True)

    copula_tok = next((x for x in T
                       if x.get('dep') in ('cop', 'aux:pass') and _is_copula(x)), None)
    # Stocker le tense de la copule dans tree pour tree_to_bambara
    if copula_tok:
        _cop_tense = copula_tok.get('tense', '')
        # Fallback : lire depuis morph spaCy (Tense=Fut, Tense=Imp...)
        if not _cop_tense:
            _morph_cop = str(copula_tok.get('morph', ''))
            if 'Tense=Fut' in _morph_cop:
                _cop_tense = 'fut'
            elif 'Tense=Imp' in _morph_cop:
                _cop_tense = 'hab'
            elif 'Tense=Past' in _morph_cop:
                _cop_tense = 'past'
        tree['cop_tense'] = _cop_tense
        tree['cop_lemma'] = copula_tok.get('lemma', '')
        tree['cop_bm']    = copula_tok.get('bm', '')
        # Si cop_tense vide, chercher dans T_tenses directement
        if not tree.get('cop_tense'):
            for _t in T:
                if _t.get('dep') == 'cop' and _t.get('tense') == 'fut':
                    tree['cop_tense'] = 'fut'
                    break
    has_with   = any(x.get('role') == 'comitative' for x in T)
    _root_idx  = root_tok['orig_index'] if root_tok else -1

    # être ROOT sans copule séparée
    copula_tok = etre_root.detect(T, root_tok, copula_tok, m, processed_indices, G_kg)

    # Statif via aux:pass
    _cop_is_on_root = copula_tok and copula_tok.get('head_index') == _root_idx
    if not _cop_is_on_root and root_tok and (
            root_tok.get('is_statif')
            or (root_tok.get('is_passive')
                and 'VerbForm=Part' in str(root_tok.get('morph', ''))
                and 'Voice=Pass' in str(root_tok.get('morph', '')))
            or (root_tok.get('pos') == 'ADJ'
                and any(x.get('dep') == 'aux:pass'
                        and x.get('head_index') == _root_idx for x in T))):
        _aux_pass_cop = next((x for x in T if x.get('dep') == 'aux:pass'
                              and x.get('head_index') == _root_idx), None)
        if _aux_pass_cop:
            copula_tok      = _aux_pass_cop
            _cop_is_on_root = True
            # Marquer comme participe passé pour tree_to_bambara
            root_tok['is_participe_passe'] = True
            tree['is_participe_passe'] = True
            tree['participe_bm'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"

    print(f"DEBUG _statif_is_past check: aux_tense_tok={aux_tense_tok}, "
          f"tree_tense={tree.get('tense')}")

    # ── CAS SPECIAL : expletif + ADJ is_valeur → présentatif ────────────
    # C'est vrai → bɛ́rɛ dòn  (comme C'est Hawa → Hawa dòn)
    # La valeur abstraite est présentée comme un fait, pas une équation
    if not _cop_is_on_root and _has_expletive and root_tok:
        _valeur_adj = next((x for x in T
                            if x.get('pos') == 'ADJ'
                            and x.get('is_valeur') is True
                            and x.get('bm')), None)
        if _valeur_adj:
            # Identificatoire : bɛ́rɛ dòn (S + dòn, comme C'est Hawa → Hawa dòn)
            tree['clause_type'] = 'identificatory'
            m['S'] = _valeur_adj.get('bm') or f"[{_valeur_adj.get('lemma')}]"
            m['O'] = ''
            m['V'] = ''
            tree['tam'] = 'tɛ' if tree.get('neg') else 'dòn'
            processed_indices.add(_valeur_adj['orig_index'])
            for _ex in T:
                if _ex.get('role') == 'expletive':
                    processed_indices.add(_ex['orig_index'])

    if _cop_is_on_root:
        if tree.get('clause_type') in ('identificatory', 'ownership', 'locative'):
            pass

        elif (root_tok and root_tok.get('is_passive')
              and 'VerbForm=Part' in str(root_tok.get('morph', ''))
              and 'Voice=Pass' in str(root_tok.get('morph', ''))):
            # Les verbes de mouvement (sortir, partir...) → résultatif -ra
            # même s'ils sont classés statif : le mouvement donne un résultat
            # Posture/statif (asseoir, coucher, lever…) : is_statif=True prime sur
            # intransitive_type='absolute' — seuls les vrais verbes de mouvement
            # (semantic_class='motion') passent par le résultatif.
            _is_motion_verb = (
                root_tok.get('semantic_class') == 'motion'
                or (str(root_tok.get('intransitive_type', '')).lower() == 'absolute'
                    and not root_tok.get('is_statif')))
            # Statif passé (j'étais assis) → statif_past
            if root_tok.get('is_statif') and not _is_motion_verb:
                statif.run(T, tree, m, processed_indices, G_kg, root_tok, aux_tense_tok)
            else:
                # VERB passif résultatif (le riz est cuit, sorti, parti) → résultatif
                root_tok['is_participe_passe'] = True
                tree['is_participe_passe'] = True
                tree['participe_bm'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
                tree['clause_type'] = 'simple'
                tree['is_transitive'] = False
                tree['tense'] = 'past'
                tree['tam'] = ''
                m['V'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
                processed_indices.add(root_tok['orig_index'])

        elif has_with:
            # Vérifier d'abord si c'est un conj comitative (Je suis avec mon mari)
            _com_conj_with = next((x for x in T
                                  if x.get('dep') in ('conj', 'attr', 'appos')
                                  and x.get('pos') in ('NOUN', 'PROPN')
                                  and x.get('bm')), None)
            if _com_conj_with and not m.get('O'):
                _com_bm = _com_conj_with.get('bm') or _com_conj_with.get('surface', '')
                _poss = next((x for x in T if x.get('dep') == 'det'
                              and x.get('role') in ('pronoun', 'possessive')
                              and x.get('head_index') == _com_conj_with['orig_index']), None)
                if _poss:
                    from rules.core import j as _j
                    _pb = _poss.get('bm', '')
                    _com_bm = _j('n', _com_bm) if _pb == 'n' else _j(_pb, 'ka', _com_bm)
                    processed_indices.add(_poss['orig_index'])
                tree['clause_type'] = 'presentative'
                m['O'] = _com_bm
                tree['tam'] = 'dòn'
                processed_indices.add(_com_conj_with['orig_index'])
            else:
                presentative.run_with(tree, G_kg)

        elif (_has_expletive
              and root_tok and root_tok.get('pos') == 'NOUN'
              and root_tok['orig_index'] not in processed_indices):
            # Cas 14 : Ce sont mes frères → n bálimakɛw dòn
            presentative.run_cas14(T, tree, m, processed_indices, G_kg, root_tok)

        else:
            # NOUN/PROPN dep=conj + case=comitative → présentatif ni
            # Je suis avec mon fils/mari → n ni n dénkɛ dòn
            _com_conj = next((x for x in T
                             if x.get('dep') in ('conj', 'attr', 'appos')
                             and x.get('pos') in ('NOUN', 'PROPN')
                             and x.get('bm')
                             and any(c.get('dep') == 'case'
                                     and c.get('role') == 'comitative'
                                     for c in T)), None)
            if _com_conj and not m.get('O'):
                _com_bm = _com_conj.get('bm') or _com_conj.get('surface', '')
                # Possessif sur le conj
                _poss = next((x for x in T if x.get('dep') == 'det'
                              and x.get('role') in ('pronoun', 'possessive')
                              and x.get('head_index') == _com_conj['orig_index']), None)
                if _poss:
                    _pb = _poss.get('bm', '')
                    _com_bm = j('n', _com_bm) if _pb == 'n' else j(_pb, 'ka', _com_bm)
                    processed_indices.add(_poss['orig_index'])
                tree['clause_type'] = 'presentative'
                m['O'] = _com_bm
                tree['tam'] = 'dòn'
                processed_indices.add(_com_conj['orig_index'])
                return

            # PROPN dep=conj + cop → équatif (Je suis/ne suis pas Hawa)
            # spaCy parse parfois le PROPN comme conj du sujet PRON
            _propn_conj_cop = next((x for x in T
                                   if x.get('pos') in ('PROPN', 'NOUN')
                                   and x.get('dep') in ('conj', 'attr', 'appos')
                                   and x.get('bm')), None)
            if _propn_conj_cop and not m.get('O'):
                tree['clause_type'] = 'equative'
                m['O'] = _propn_conj_cop.get('bm') or _propn_conj_cop.get('surface', '')
                # TAM : tenir compte du futur (cop_tense='fut')
                if tree.get('cop_tense') == 'fut':
                    tree['tam'] = 'tɛ' if tree.get('neg') else 'yé'
                elif tree.get('neg'):
                    tree['tam'] = 'tɛ'
                else:
                    tree['tam'] = G_kg.get('equative_marker', 'yé') or 'yé'
                processed_indices.add(_propn_conj_cop['orig_index'])
                return  # ne pas continuer vers les branches ADJ

            # ── Branchement ADJ/NOUN avec copule ──────────────────────────────
            # Basé UNIQUEMENT sur les flags KG fiables :
            #   is_statif, semantic_class, VerbForm=Part, statif_root
            # is_valeur (qwen) est IGNORÉ — classification non fiable.
            # Arbre de décision :
            #   1. is_statif=True ou semantic_class='statif' → statif (-len dòn)
            #   2. is_potential=True                        → potential (-ta dòn)
            #   3. is_participe_passe ou bm=[..]+VerbForm=Part → résultatif (-ra)
            #   4. bm=[...] sans VerbForm=Part              → équatif (profession inconnue)
            #   5. bm valide (tout le reste)                → qualitative (ka/man)

            _statif_obl = next((x for x in T
                                if x.get('dep') in ('obl', 'obl:arg', 'nmod')
                                and x.get('head_index') == _root_idx
                                and any(p.get('dep') == 'case'
                                        and p.get('role') == 'locative'
                                        and p.get('head_index') == x['orig_index']
                                        for p in T)), None)
            _morph_str = str(root_tok.get('morph', '')) if root_tok else ''
            _has_aux_pass_on_root = any(
                x.get('dep') == 'aux:pass'
                and x.get('head_index') == _root_idx for x in T)
            _is_statif_adj = (root_tok
                              and root_tok.get('pos') == 'ADJ'
                              and (root_tok.get('semantic_class') in ('statif', 'state', 'physical_state')
                                   or root_tok.get('statif_root')
                                   or root_tok.get('is_statif') is True
                                   or 'VerbForm=Part' in _morph_str
                                   or 'Tense=Past' in _morph_str
                                   or _has_aux_pass_on_root))

            if root_tok and root_tok.get('pos') == 'ADJ' and (_statif_obl or _is_statif_adj):
                # Statif explicite (is_statif, semantic_class, VerbForm=Part)
                statif.run(T, tree, m, processed_indices, G_kg, root_tok, aux_tense_tok)

            elif root_tok and root_tok.get('pos') == 'ADV':
                # ADV + copule : distinguer locatif (is_loc=True) et qualitatif (is_loc=False)
                # Locatif  → S bɛ ADV  (ici, là, dehors → wátiriw bɛ yàn)
                # Qualitatif → S ka ADV  (loin, près → só ka póroo)
                if root_tok.get('is_loc') or root_tok.get('role') == 'locative':
                    _adv_bm = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
                    m['OBL_ALL'].append({
                        'HEAD': _adv_bm, 'MARKER': '',
                        'local_clause_type': 'simple',
                        'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
                        'DEP_TYPE': '', 'COMPOUND_IS_QUANTIFIER': False,
                        'MARKER_IS_PREFIX': False,
                    })
                    m['V'] = ''
                    tree['tam'] = _resolve_tam('pres', tree.get('neg', False), G_kg) or 'bɛ'
                    processed_indices.add(root_tok['orig_index'])
                else:
                    # ADV prédicatif non-locatif (loin, proche...) → qualitatif S ka ADV
                    qualitative.run(T, tree, m, processed_indices, G_kg, root_tok)
                    processed_indices.add(root_tok['orig_index'])

            elif root_tok and root_tok.get('pos') == 'ADJ':
                # ADJ prédicatif retraduit comme NOUN (profession/rôle) → équatif direct
                if root_tok.get('_adj_is_nominal_pred') and root_tok.get('bm') and not root_tok.get('bm', '').startswith('['):
                    equatif.run_default(T, tree, m, processed_indices, G_kg, root_tok, aux_tense_tok)
                    if not m.get('O'):
                        m['O'] = root_tok.get('bm')
                        processed_indices.add(root_tok['orig_index'])
                    return
                _morph_str2        = str(root_tok.get('morph', ''))
                _bm_is_fallback    = root_tok.get('bm', '').startswith('[')
                _has_verbform_part = 'VerbForm=Part' in _morph_str2
                _sc                = root_tok.get('semantic_class', '')
                _is_statif_sc      = _sc in ('statif', 'state', 'physical_state')

                # ── COMPARATIF : ADJ + advmod(neg_surf, head=ADJ) + mark(SCONJ) + ref ──
                # ex: "est plus important que lui" → ka kólogirinman ka tɛmɛ à kan
                _neg_surfs_s6 = G_kg.get('neg_surfaces', set())
                _comp_adv = next((x for x in T
                                  if x.get('dep') == 'advmod'
                                  and x.get('head_index') == root_tok['orig_index']
                                  and str(x.get('surface', '')).lower().rstrip("'")
                                      in _neg_surfs_s6), None)
                _comp_mark = next((x for x in T
                                   if x.get('dep') == 'mark'
                                   and x.get('pos') == 'SCONJ'
                                   and x.get('role') != 'temporal'), None)
                _comp_ref  = next((x for x in T
                                   if x.get('dep') in ('dep', 'obl', 'nsubj', 'obj')
                                   and x.get('bm')
                                   and x['orig_index'] not in processed_indices
                                   and x.get('orig_index') != root_tok['orig_index']), None)
                if _comp_adv and _comp_mark and _comp_ref:
                    tree['clause_type'] = 'comparative'
                    tree['tam'] = 'ka'
                    m['V'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
                    m['comparative_particle'] = 'ka tɛmɛ'
                    m['comparative_ref']      = _comp_ref.get('bm', '')
                    processed_indices.update([
                        root_tok['orig_index'],
                        _comp_adv['orig_index'],
                        _comp_mark['orig_index'],
                        _comp_ref['orig_index'],
                    ])

                elif root_tok.get('is_statif') is True or _is_statif_sc:
                    # Statif explicite via flag ou semantic_class KG
                    statif.run(T, tree, m, processed_indices, G_kg, root_tok, aux_tense_tok)

                elif root_tok.get('is_potential') is True:
                    # Participe potential : V + -ta
                    participes.run_potential(T, tree, m, processed_indices, G_kg, root_tok)

                elif root_tok.get('is_participe_passe') or (
                        _bm_is_fallback and _has_verbform_part):
                    # Vrai participe passé (VerbForm=Part) → résultatif -ra/-la/-na
                    participes.run_resultatif(T, tree, m, processed_indices, G_kg,
                                              root_tok, _has_expletive)

                elif _bm_is_fallback and not _has_verbform_part:
                    if root_tok.get('pos') == 'ADJ':
                        # ADJ sans traduction KG → qualitatif : n ka [bel]
                        qualitative.run(T, tree, m, processed_indices, G_kg, root_tok)
                    else:
                        # Profession/rôle inconnu du KG → équatif
                        # ex: [étudiant] → a tùn yé [étudiant] yé
                        equatif.run_default(T, tree, m, processed_indices,
                                            G_kg, root_tok, aux_tense_tok)
                        if not m.get('O'):
                            m['O'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
                            processed_indices.add(root_tok['orig_index'])

                else:
                    # bm valide du KG — distinguer via is_valeur (qwen) :
                    # QUALITE (grand, beau, petit) → qualitative → n ka bòn
                    # VALEUR/STATIF (sûr, certain, fatigué) → statif → n jóonalen dòn
                    # is_valeur est utilisé ICI UNIQUEMENT comme tiebreaker
                    # quand semantic_class est absent du KG
                    if root_tok.get('is_valeur') is True:
                        statif.run(T, tree, m, processed_indices, G_kg, root_tok, aux_tense_tok)
                    else:
                        qualitative.run(T, tree, m, processed_indices, G_kg, root_tok)

            else:
                # Dernier recours : NOUN/PROPN conj sans cop directe
                # ex: Il ne sera pas président → m['O']='pèresidan'
                _last_conj = next((x for x in T
                                   if x.get('dep') in ('conj', 'attr', 'appos')
                                   and x.get('pos') in ('NOUN', 'PROPN', 'ADJ')
                                   and x.get('bm')
                                   and x['orig_index'] not in processed_indices), None)
                if _last_conj and not m.get('O'):
                    tree['clause_type'] = 'equative'
                    m['O'] = _last_conj.get('bm') or _last_conj.get('surface', '')
                    # TAM : tenir compte du futur
                    if tree.get('cop_tense') == 'fut':
                        tree['tam'] = 'tɛ' if tree.get('neg') else 'yé'
                    elif tree.get('neg'):
                        tree['tam'] = 'tɛ'
                    else:
                        tree['tam'] = G_kg.get('equative_marker', 'yé') or 'yé'
                    processed_indices.add(_last_conj['orig_index'])
                else:
                    # être ROOT sans prédicat équatif + obliques présents
                    # (ici, là, yàn...) → locatif : ɲàmakalaw bɛ yàn
                    _has_obl = bool(m.get('OBL_ALL'))
                    if _has_obl and not m.get('O'):
                        tree['clause_type'] = 'locative'
                        tree['tam'] = _resolve_tam(
                            'pres', tree.get('neg', False), G_kg) or 'bɛ'
                        m['V'] = ''
                        processed_indices.add(root_tok['orig_index'])
                    else:
                        equatif.run_default(T, tree, m, processed_indices, G_kg,
                                            root_tok, aux_tense_tok)

        # Capturer root non encore traité comme V
        if root_tok and root_tok['orig_index'] not in processed_indices:
            _root_bm = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
            if root_tok.get('bm_suffix'):
                _root_bm += root_tok['bm_suffix']

            _has_obj = any(x.get('dep') == 'obj' for x in T)
            _is_verb = root_tok.get('pos') == 'VERB'
            _is_trans = tree.get('is_transitive', True)
            _intrans_type = root_tok.get('intransitive_type', '')
            _is_intrans = bool(_intrans_type)  # Si intransitive_type est assigné, le verbe est intransitif
            _tam = tree.get('tam', '')
            _is_progressive = _tam in ('bɛ kà', 'tɛ kà')
            _is_neg = bool(tree.get('neg'))

            # Verbes transitifs sans COD → ajouter 'li kɛ'
            # (pas de suffixe au négatif : "je ne mange pas" → n tɛ dún,
            #  "il n'a pas parlé" → a ma kúma)
            if (not _has_obj and _is_verb and _is_trans and not _is_intrans
                    and not root_tok.get('is_statif') and not _is_neg):
                # Pour les verbes d'action (transitifs sans COD), nominaliser : V+li kɛ
                if _root_bm.endswith('la'):
                    _root_bm = _root_bm[:-2]
                _root_bm += 'li kɛ'

            # Verbes intransitifs (explicitement marqués par intransitive_type)
            elif _is_verb and _is_intrans and not root_tok.get('is_statif'):
                if _is_progressive:
                    # Progressif intransitif → ajouter 'kɛ'
                    if not _root_bm.endswith('kɛ'):
                        _root_bm += ' kɛ'
                elif not _is_neg:
                    # Présent intransitif → ajouter '-la'
                    # (négatif : pas de suffixe — "je n'ai pas dormi" → n ma sùnɔgɔ)
                    if not _root_bm.endswith('la'):
                        _root_bm += 'la'

            m['V'] = _root_bm
            processed_indices.add(root_tok['orig_index'])

    elif not aux_tense_tok:
        if tree.get('neg'):
            tree['tam'] = _resolve_tam(tree.get('tense', 'pres'), True, G_kg)
        elif tree.get('clause_type') == 'locative':
            tree['tam'] = _resolve_tam('pres', tree.get('neg', False), G_kg) or 'bɛ'
        elif tree.get('clause_type') != 'noun_phrase':
            if not tree.get('tam'):
                tree['tam'] = (G_kg.get('tam_default', '')
                               or _resolve_tam('pres', False, G_kg))

    # c'est + ADJ ROOT + expletif → équatif ou participe
    identificatoire.run_adj_expletif(T, tree, m, processed_indices, G_kg,
                                     root_tok, _has_expletive)

    # Alignement TAM équatif interrogatif
    if clause_type_init == 'content_question':
        _has_be_copula    = any(_is_copula(x) or x.get('dep') == 'cop' for x in T)
        _has_loc_interrog = any(x.get('role') == 'interrogative'
                                and x.get('dep') in ('advmod', 'dep', 'obj') for x in T)
        if (_has_be_copula
                or (root_tok and root_tok.get('role') == 'interrogative')):
            if not _has_loc_interrog:
                tree['tam'] = G_kg.get('equative_marker', 'yé') or 'yé'