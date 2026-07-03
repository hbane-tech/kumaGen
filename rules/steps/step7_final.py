"""
rules/steps/step7_final.py
Étape 7 : harmonisation nominale, ccomp (complétive), intransitif kɛ,
déduplication wagons, slots finaux.
"""
from rules.core import j, get_bounded_chunk_tokens, INTRANS_SC, _resolve_tam, _is_copula
from rules.steps.step2_sujet import _build_subj_chain
from rules.steps.step4_objet.objet_standard import _build_genitive_chain


def run(T, tree, m, processed_indices, G_kg, NX_G, root_tok,
        get_bounded_chunk_tokens_fn=None):
    """Finalise tree/m. Retourne tree."""

    # Privative ROOT : construire final_string et retourner (skip tous autres traitements)
    if tree.get('_is_privative'):
        # Équatif bambara : S yé O yé (le marqueur encadre le prédicat)
        m['SLOTS'] = {}
        idx_s = 1
        if m.get('S'):
            m['SLOTS'][f'X{idx_s}'] = m['S']; idx_s += 1
        if m.get('V'):
            m['SLOTS'][f'X{idx_s}'] = m['V']; idx_s += 1
        if m.get('O'):
            m['SLOTS'][f'X{idx_s}'] = m['O']; idx_s += 1
        if m.get('V'):
            m['SLOTS'][f'X{idx_s}'] = m['V']; idx_s += 1
        ordered_keys  = sorted(m['SLOTS'].keys(), key=lambda x: int(x[1:]))
        tree['final_string'] = j(*[m['SLOTS'][k] for k in ordered_keys])
        tree['local_clause_type'] = tree.get('clause_type', 'simple')
        tree['_tokens'] = T
        return tree

    # Utiliser la closure fournie par build_tree si disponible
    _gbc = get_bounded_chunk_tokens_fn or (
        lambda idx: get_bounded_chunk_tokens(idx, NX_G, processed_indices))

    # ── HARMONISATION NOMINALE ────────────────────────────────────────────────
    if tree.get('clause_type') == 'noun_phrase':
        if m['S'] and not m['O']:
            m['O'] = m['S']
            m['S'] = ''
        elif m['S'] and m['O'] and m['S'] != m['O']:
            if len(str(m['O']).strip()) >= len(str(m['S']).strip()):
                m['S'] = ''
            elif m['O'] == (G_kg.get('plural_noun_suffix', '') or 'w'):
                m['O'] = m['S'] + (G_kg.get('plural_noun_suffix', '') or '')
                m['S'] = ''
            elif m['S'] in m['O']:
                m['S'] = ''
            else:
                m['O'] = j(m['S'], m['O'])
                m['S'] = ''

    # ── SUBORDONNÉE COMPLÉTIVE (ccomp) ───────────────────────────────────────
    if tree.get('clause_type') in ('simple', 'interrogative', 'content_question',
                                   'complex', 'temporal', 'interrogative_action'):
        ccomp_tok = next((x for x in T if x.get('dep') == 'ccomp'), None)
        if ccomp_tok:
            ccomp_subj = next((x for x in T
                               if x.get('dep') in ('nsubj', 'expl:subj', 'nsubj:pass')
                               and x.get('head_index') == ccomp_tok['orig_index']), None)
            ccomp_head_bm = ccomp_tok.get('bm') or f"[{ccomp_tok.get('lemma')}]"

            # Chaîne génitif pour le sujet du ccomp (ex: raison de la venue de qqu'un)
            _subj_bm = (_build_subj_chain(ccomp_subj, T, G_kg)
                        if ccomp_subj else '')
            if ccomp_subj and ccomp_subj.get('is_plural') and not _subj_bm.endswith(G_kg.get('plural_noun_suffix', '') or 'w'):
                if ccomp_subj.get('pos') not in ('PRON', 'PROPN'):
                    _plur_s = G_kg.get('plural_noun_suffix', '')
                    if _plur_s: _subj_bm += _plur_s

            # Détection du comparatif : ccomp ADJ + advmod(négation,head=ADJ) + mark(SCONJ) + ref
            _neg_surfs_c = G_kg.get('neg_surfaces', set())
            _c_comp_adv = next((x for x in T
                                if x.get('dep') == 'advmod'
                                and x.get('head_index') == ccomp_tok['orig_index']
                                and str(x.get('surface', '')).lower().rstrip("'")
                                    in _neg_surfs_c), None)
            # Le 'que' comparatif vient APRÈS le ccomp_tok (pas le complémenteur avant)
            _c_comp_mark = next((x for x in T
                                 if x.get('dep') == 'mark'
                                 and x.get('pos') == 'SCONJ'
                                 and x.get('role') != 'temporal'
                                 and x['orig_index'] > ccomp_tok['orig_index']), None)
            # Référent comparatif = tête du 'que' mark, SAUF si spaCy l'attache
            # au ccomp_tok lui-même (parsing inconsistant) → fallback : premier
            # token avec bm après le mark (excl. particules et déterminants).
            _c_comp_ref = None
            if _c_comp_mark:
                _ref_by_head = next((x for x in T
                                     if x.get('orig_index') == _c_comp_mark.get('head_index')
                                     and x.get('bm')
                                     and x['orig_index'] != ccomp_tok['orig_index']), None)
                if _ref_by_head:
                    _c_comp_ref = _ref_by_head
                else:
                    _c_comp_ref = next((x for x in sorted(T, key=lambda t: t['orig_index'])
                                        if x['orig_index'] > _c_comp_mark['orig_index']
                                        and x.get('bm')
                                        and x.get('dep') not in ('mark', 'cc', 'det', 'case')
                                        and x.get('pos') not in ('PUNCT', 'SYM', 'DET', 'ADP')
                                        ), None)

            if _c_comp_adv and _c_comp_mark and _c_comp_ref:
                m['CCOMP'] = {
                    'type':     'comparative',
                    'S':        _subj_bm,
                    'adj_bm':   ccomp_head_bm,
                    'particle': G_kg.get('comparative_particle', ''),
                    'ref_bm':   _c_comp_ref.get('bm', ''),
                    'neg':      tree.get('neg', False),
                }
            elif ccomp_tok.get('pos') == 'ADV':
                # ccomp locatif/prédicatif : "il est là" → "ko a bɛ yèn"
                # UD : tête=ADV, cop=être (enfant), sujet=expl:subj ou nsubj.
                # Garde cop : sans copule on retombe sur équatif.
                _ccomp_cop_adv = next((x for x in T
                                       if x.get('dep') == 'cop'
                                       and x.get('head_index') == ccomp_tok['orig_index']), None)
                if _ccomp_cop_adv:
                    _ccomp_subj_adv = next((x for x in T
                                            if x.get('dep') in ('nsubj', 'expl:subj', 'nsubj:pass')
                                            and x.get('head_index') == ccomp_tok['orig_index']), None)
                    _s_bm_adv = (_build_subj_chain(_ccomp_subj_adv, T, G_kg)
                                  if _ccomp_subj_adv else _subj_bm)
                    _cop_tense = _ccomp_cop_adv.get('tense', 'pres')
                    _adv_neg = any(
                        x.get('dep') in ('advmod', 'fixed', 'mark')
                        and str(x.get('surface', '')).lower().rstrip("'") in _neg_surfs_c
                        and x.get('head_index') == ccomp_tok['orig_index']
                        for x in T)
                    _adv_tam = _resolve_tam(_cop_tense, _adv_neg, G_kg) or G_kg.get('tam_default', '')
                    m['CCOMP'] = {
                        'type': 'verbal',
                        'S':    _s_bm_adv,
                        'tam':  _adv_tam,
                        'O':    ccomp_head_bm,
                        'V':    '',
                    }
                else:
                    m['CCOMP'] = {
                        'S':   _subj_bm,
                        'tam': G_kg.get('equative_marker', '') or '',
                        'O':   ccomp_head_bm,
                        'V':   '',
                    }
            elif ccomp_tok.get('pos') == 'VERB':
                # ── Copule + locatif obl:arg : "nous sommes ici" → "ko anw bɛ yèn" ─
                # être/copula + ADV obl:arg → supprimer le verbe copule,
                # l'ADV locatif devient le prédicat (même logique que ADV ccomp).
                _ccomp_loc_adv = next((x for x in T
                                       if x.get('pos') == 'ADV'
                                       and x.get('dep') in ('obl:arg', 'advmod', 'obl:mod')
                                       and x.get('head_index') == ccomp_tok['orig_index']
                                       and x.get('bm')), None)
                _cop_handled = False
                if (_ccomp_loc_adv
                        and (ccomp_tok.get('semantic_class') == 'copula'
                             or _is_copula(ccomp_tok))):
                    _cop_tense_v = ccomp_tok.get('tense', 'pres')
                    _cop_neg_v   = any(
                        x.get('dep') in ('advmod', 'fixed', 'mark')
                        and str(x.get('surface', '')).lower().rstrip("'") in _neg_surfs_c
                        and x.get('head_index') == ccomp_tok['orig_index']
                        for x in T)
                    _cop_tam_v = _resolve_tam(_cop_tense_v, _cop_neg_v, G_kg) or G_kg.get('tam_default', '')
                    m['CCOMP'] = {
                        'type': 'verbal',
                        'S':    _subj_bm,
                        'tam':  _cop_tam_v,
                        'O':    _ccomp_loc_adv.get('bm', ''),
                        'V':    '',
                    }
                    _cop_handled = True
                    processed_indices.add(ccomp_tok['orig_index'])
                    processed_indices.add(_ccomp_loc_adv['orig_index'])
                # ccomp à tête verbale (… disant qu'elle partait) → clause
                # verbale ko S TAM (O) V, et non équative ko S yé O yé.
                _ccomp_neg = any(
                    x.get('head_index') == ccomp_tok['orig_index']
                    and x.get('dep') in ('advmod', 'fixed', 'mark')
                    and (str(x.get('surface', '')).lower().rstrip("'")
                         in _neg_surfs_c or x.get('role') == 'negation')
                    for x in T)
                _ccomp_tam = _resolve_tam(
                    ccomp_tok.get('tense', 'pres'), _ccomp_neg, G_kg) or G_kg.get('tam_default', '')
                _ccomp_obj = next((x for x in T
                                   if x.get('dep') in ('obj', 'iobj')
                                   and x.get('head_index') == ccomp_tok['orig_index']
                                   and x.get('bm')), None)
                _ccomp_v_bm    = ccomp_head_bm
                _ccomp_tam_out = _ccomp_tam
                _ccomp_sc      = ccomp_tok.get('semantic_class', '')
                _ccomp_intrans = ccomp_tok.get('intransitive_type', '')
                _ccomp_is_past = (ccomp_tok.get('tense') == 'past' and not _ccomp_neg)
                # Plus-que-parfait : l'auxiliaire (était/avait) porte l'imparfait
                # (tense='hab'), pas le participe lui-même ("parti" reste tense='past').
                # "qu'il était parti" ≠ "qu'il est parti" — sans ce signal sur
                # l'AUXILIAIRE, les deux s'aplatissaient sur le même résultatif V+ra
                # sans TAM, perdant la distinction plus-que-parfait/passé simple.
                _ccomp_aux_tense_tok = next((x for x in T
                                            if x.get('dep') in ('aux:tense', 'aux:pass')
                                            and x.get('head_index') == ccomp_tok['orig_index']), None)
                _ccomp_is_plup = bool(_ccomp_aux_tense_tok
                                     and _ccomp_aux_tense_tok.get('tense') == 'hab')
                # Intransitif au passé positif → résultatif V+ra/na, sans TAM
                # INTRANS_SC (motion, biological…) = toujours intransitif (signal KG,
                # fiable) — ignorer le it='ACTION' du LLM pour ces classes.
                if (_ccomp_is_past and ccomp_head_bm and not _ccomp_obj
                        and (_ccomp_intrans == 'ABSOLU'
                             or _ccomp_sc in INTRANS_SC)):
                    # Suffixes depuis KG (G_kg chargé par _load_grammar)
                    _sfx_n   = G_kg.get('resultative_suffix_n',   'na')
                    _sfx_v   = G_kg.get('resultative_suffix_vowel','la')
                    _sfx_def = G_kg.get('resultative_suffix',     'ra')
                    _trig_n  = G_kg.get('resultative_trigger_n',  'n')
                    _trig_v  = set(G_kg.get('resultative_trigger_vowel','o|u|ɔ').split('|'))
                    _sfx = (_sfx_n if ccomp_head_bm.endswith(_trig_n)
                            else _sfx_v if (ccomp_head_bm and ccomp_head_bm[-1] in _trig_v)
                            else _sfx_def)
                    _ccomp_v_bm    = ccomp_head_bm + _sfx
                    _ccomp_tam_out = (G_kg.get('statif_hab_prefix', '') or 'tùn') if _ccomp_is_plup else ''
                # NOM-ACTION (action/having/technique) : BM déjà nominal → V kɛ (jamais V+li)
                # ex: travailler=báara → ko a bɛ báara kɛ (pas báarali kɛ)
                elif (not _ccomp_obj and ccomp_head_bm
                        and _ccomp_sc in ('having', 'technique', 'action')):
                    _ccomp_v_bm = ccomp_head_bm + ' ' + G_kg.get('coord_action_suffix', '')
                # Transitif sans COD → nominalisé V+li kɛ
                elif (not _ccomp_obj and ccomp_head_bm
                        and _ccomp_intrans in ('ACTION', 'nominalized', 'support')
                        and _ccomp_sc not in INTRANS_SC
                        and _ccomp_sc not in ('having', 'technique', 'action')):
                    _nom_sfx_c = G_kg.get('nominalization_verb_suffix', '')
                    _nom_sfx2_c = G_kg.get('nominalization_verb_suffix_alt', '')
                    _vli = (ccomp_head_bm + _nom_sfx_c
                            if _nom_sfx_c and not ccomp_head_bm.endswith((_nom_sfx_c, _nom_sfx2_c))
                            else ccomp_head_bm)
                    _ccomp_v_bm = _vli + ' ' + G_kg.get('coord_action_suffix', '')
                # Est-ce que + être ROOT (copule) → ccomp = vrai verbe principal.
                # spaCy parse "Est-ce qu'elle travaille ?" comme être ROOT + travaille ccomp,
                # ce qui produirait "a bɛ ko a bɛ báara kɛ wà ?".
                # On promeut le ccomp au verbe principal et on redirige root_tok vers
                # ccomp_tok pour que l'aiguillage intransitif (line ~257) applique
                # correctement la classe action/INTRANS_SC (báara la, fáɲira, etc.).
                # Est-ce que + copule OU Est-ce que + ROOT=PRON(-ce)/AUX :
                # le ccomp est en réalité le verbe principal → promouvoir.
                _is_estceque_pron_root = (
                    tree.get('est_ce_que')
                    and root_tok
                    and root_tok.get('pos') in ('PRON', 'AUX')
                    and str(root_tok.get('surface', '')).lower().lstrip('-') in ('ce', 'est', '')
                )
                if not _cop_handled and (
                    tree.get('est_ce_que')
                    and root_tok
                    and (
                        # Cas 1 : copule ROOT + est-ce-que → promouvoir le vrai verbe
                        (root_tok.get('semantic_class') == 'copula'
                         and tree.get('clause_type') in ('interrogative', 'content_question',
                                                          'quest_ce_que', 'simple'))
                        or
                        # Cas 2 : ROOT=PRON(-ce) + ccomp = vraie verbe principal
                        _is_estceque_pron_root
                    )
                ):
                    m['V'] = ccomp_head_bm
                    if _subj_bm:
                        m['S'] = _subj_bm
                    m['TAM'] = _ccomp_tam
                    # Les advmods du verbe promu peuvent avoir été ajoutés à OBL_ALL
                    # par step5. Ils seront rendus via ADV en les retirant de OBL_ALL
                    # pour éviter la duplication.
                    _prom_adv_bms = {x.get('bm', '') for x in T
                                     if x.get('dep') == 'advmod'
                                     and x.get('head_index') == ccomp_tok['orig_index']
                                     and x.get('bm')}
                    if _prom_adv_bms and m.get('OBL_ALL'):
                        _adv_str = j(*sorted(_prom_adv_bms))
                        m['OBL_ALL'] = [obl for obl in m['OBL_ALL']
                                        if isinstance(obl, dict)
                                        and obl.get('HEAD', '') not in _prom_adv_bms]
                        if not m.get('ADV'):
                            m['ADV'] = _adv_str
                    root_tok = ccomp_tok   # l'aiguillage utilise maintenant la sémantique du ccomp
                    _cop_handled = True
                if not _cop_handled:
                    m['CCOMP'] = {
                        'type': 'verbal',
                        'S':    _subj_bm,
                        'tam':  _ccomp_tam_out,
                        'O':    (_ccomp_obj.get('bm', '') if _ccomp_obj else ''),
                        'V':    _ccomp_v_bm,
                    }
                    if _ccomp_obj:
                        processed_indices.add(_ccomp_obj['orig_index'])
            elif (ccomp_tok.get('pos') == 'ADJ'
                  and not ccomp_tok.get('is_statif')
                  and not ccomp_tok.get('is_participe_passe')
                  and not ccomp_tok.get('is_valeur')):
                # ccomp à tête ADJ de type QUALITE (la route est LONGUE) :
                # même structure que la copule autonome (S ka ADJ, cf
                # step6_copule/identificatoire.py), pas l'équative à tête
                # NOUN ci-dessous (S yé O yé) — sans cette distinction,
                # "ils disent que la route est longue" sortait
                # "...ko síraden yé búlubulu yé" au lieu de
                # "...ko síraden ka búlubulu".
                m['CCOMP'] = {
                    'type': 'qualite',
                    'S':    _subj_bm,
                    'adj_bm': ccomp_head_bm,
                }
            else:
                # ccomp équatif à tête NOUN : inclure son génitif éventuel
                # (les maîtres DU JEU → jeu [ka] mɛtiriw), sinon le complément
                # est largué. + pluriel sur le nom-tête.
                _has_ccomp_nmod = any(
                    x.get('dep') == 'nmod'
                    and x.get('head_index') == ccomp_tok['orig_index']
                    for x in T)
                _ccomp_o = (_build_genitive_chain(ccomp_tok, T, G_kg)
                            if _has_ccomp_nmod else ccomp_head_bm)
                if (ccomp_tok.get('is_plural')
                        and _ccomp_o and not _ccomp_o.endswith(G_kg.get('plural_noun_suffix', '') or 'w')
                        and ccomp_tok.get('pos') not in ('PRON', 'PROPN')):
                    _ccomp_o += 'w'
                # Possessif DET sur le prédicat nominal (mon ami → n téri)
                _ccomp_poss = next((x for x in T
                                    if x.get('dep') == 'det'
                                    and x.get('role') in ('pronoun', 'possessive')
                                    and x.get('head_index') == ccomp_tok['orig_index']
                                    and x.get('bm')), None)
                if _ccomp_poss:
                    _poss_bm  = _ccomp_poss.get('bm', '')
                    _p1sg_c   = G_kg.get('pron_1sg', '')
                    _gm_c     = G_kg.get('genitive_marker', '')
                    _rel_bms  = G_kg.get('relational_bms', set())
                    _is_rel_c = ccomp_tok.get('is_relational', False) or _ccomp_o in _rel_bms
                    if _is_rel_c or _poss_bm == _p1sg_c:
                        _ccomp_o = j(_poss_bm, _ccomp_o)   # inalienable : n téri
                    else:
                        _ccomp_o = j(_poss_bm, _gm_c, _ccomp_o)  # alienable : n ka X
                m['CCOMP'] = {
                    'S':   _subj_bm,
                    'tam': G_kg.get('equative_marker', '') or '',
                    'O':   _ccomp_o,
                    'V':   '',
                }

            ccomp_chunk = _gbc(ccomp_tok['orig_index'])
            processed_indices.update([t['orig_index'] for t in ccomp_chunk])
            if ccomp_subj:
                processed_indices.add(ccomp_subj['orig_index'])

    # ── AIGUILLAGE INTRANSITIF (matrice 4 cas B) ──────────────────────────────
    # Un ccomp tient lieu de complément du verbe (montrer QUE…) → le verbe
    # n'est PAS intransitif sans COD, on ne le nominalise pas en Vli kɛ.
    if (root_tok and root_tok.get('pos') == 'VERB'
            and not m.get('O')
            and not m.get('V_ACTION')
            and not m.get('CCOMP')
            # Verbe transitif sans COD → V+li kɛ, dans toutes les clauses
            # déclaratives (principale + subordonnées quand/si/lorsque). Les
            # types réfléchi/réciproque/passif/relatif/infinitif ont leur propre
            # COD ou gestion du verbe → clause_type distinct, non concernés ici.
            and tree.get('clause_type') not in (
                # Clause types avec gestion propre du verbe → exclure
                'refl_absolute', 'reciprocal', 'passive', 'passive_statif',
                'passive_statif_question', 'locative', 'existential_absolute',
                'existential_localized', 'existential_nominal', 'noun_phrase_have',
                'relative_nominal', 'infinitive', 'impersonal',
                'quest_ce_que_modal',  # quest_ce_que_modal gère V+V_ACT lui-même
            )
            # Optatif + verbe intransitif (subject=argument, context='modified_noun') → verbe nu
            # (hɛ́rɛ ka wó — régner avec paix=argument : context_type='modified_noun')
            # Optatif + verbe transitif sans COD → laissé passer → VERB kɛ ci-dessous
            # (Dieu ka bólodɛ̀mɛ kɛ — aider sans COD : context_type absent=transitif)
            and not (tree.get('clause_type') == 'optative'
                     and m.get('S')
                     and root_tok.get('context_type') == 'modified_noun')
            and not root_tok.get('is_participe_passe')
            and not root_tok.get('is_passive')
            and not root_tok.get('is_refl_passive')
            # Réflexif pronominal bypassé : V nu au présent, résultatif au passé
            # (géré par le renderer F6 via _refl_pronominal_bypass) — ne pas nominaliser.
            and not tree.get('_refl_pronominal_bypass')):
        _intrans_type = root_tok.get('intransitive_type', '')
        # Progressif ('prog') traité comme présent : TAM='bɛ kà' + verbe nu (pas li kɛ)
        _is_pres = (tree.get('tense', 'pres') in ('pres', 'hab', 'prog'))
        _v_root = root_tok.get('bm', '')
        # Classes sémantiques vraiment autonomes (ne prennent pas de -li)
        _sc = root_tok.get('semantic_class', '')
        # Coordination tail built by step3 _conj_root_verbs e.g. "báara wa sùnna"
        _v_str_7 = m.get('V', '')
        _v_wa_tail = (
            _v_str_7[len(_v_root) + 4:]
            if _v_root and _v_str_7.startswith(_v_root + ' wa ')
            else '')
        # consumption (solide) → nominalisé V+li kɛ : manger → dúnili kɛ.
        # Le LLM peut osciller entre 'consumption' et 'consumption_liquid' pour
        # boire ; le prompt est optimisé pour fiabiliser ce retour.
        _is_liquid = (_sc == 'consumption_liquid')

        # Verbe d'ACTIVITÉ intransitif (travailler=báara) : nom d'action.
        # On se base sur semantic_class=='action' (signal FIABLE) et NON sur
        # intransitive_type (verdict LLM ACTION/ABSOLU qui oscille pour
        # travailler). « Intransitive verb of ACTION don't take li kɛ » :
        #   présent/imparfait (pos ET nég) : S TAM V la
        #     (n bɛ báara la ; n tùn bɛ báara la ; n tɛ báara la)
        #   passé/futur (pos ET nég)        : S TAM V kɛ
        #     (n ye báara kɛ ; n ma báara kɛ)
        # Règle purement temporelle (la polarité est portée par le TAM).
        # (≠ manger=consumption → li kɛ ; ≠ dormir/parler ∈ INTRANS_SC → V nu)
        _has_refl_pass = any(t.get('dep') == 'expl:pass' for t in T)
        # Passif grammatical (être + participe passé) : "cela fut fait", "le riz a été mangé"
        # → résultatif V+ra (is_transitive=False via step6, géré par F6), PAS nominalization.
        _has_aux_pass_gramm = any(t.get('dep') == 'aux:pass' for t in T)
        # content_question : un PRON interrogatif en position COD (qui/quoi→jɔn/mún)
        # NE PAS inclure le sujet interrogatif "qui a mangé?" (qui=nsubj)
        _has_interrog_obj = (
            tree.get('clause_type') == 'content_question'
            and any(t.get('role') in ('interrogative', 'relative')
                    and t.get('pos') == 'PRON'
                    and t.get('dep') not in ('nsubj', 'nsubj:pass')  # qui=sujet pas objet
                    for t in T))
        if (_sc == 'action'
                and _v_root and not root_tok.get('is_statif')
                and not _has_refl_pass
                and not _has_aux_pass_gramm
                and not _has_interrog_obj):
            # Distributif présent (chaque semaine, tous les jours) → kɛ, pas la
            _distrib_surfs = (set(G_kg.get('distributive_each', {}).keys())
                              | set(G_kg.get('distributive_one', {}).keys()))
            _has_distrib = any(str(t.get('surface', '')).lower() in _distrib_surfs for t in T)
            # Optatif + ACTION + sans COD → VERB kɛ (pas 'la' en optative)
            if tree.get('clause_type') == 'optative' and not m.get('O'):
                m['V'] = j(_v_root, G_kg.get('coord_action_suffix', ''))
            elif _is_pres and not _has_distrib:
                if tree.get('tense') == 'prog':
                    # Progressif kà V kɛ : ACTION après kà prend kɛ (pas la du présent)
                    _as = G_kg.get('coord_action_suffix', '')
                    m['V'] = j(_v_root, _as, _v_wa_tail) if _v_wa_tail else j(_v_root, _as)
                    tree['is_transitive'] = True
                else:
                    _act_pres_sfx = G_kg.get('action_pres_suffix', '')
                    _coord_mk = G_kg.get('coord_verb_marker', '')
                    m['V'] = j(_v_root, _act_pres_sfx, _coord_mk, _v_wa_tail) if _v_wa_tail else j(_v_root, _act_pres_sfx)
            else:
                # Passé/futur (positif ET négatif) : nom d'action + kɛ, on garde
                # le TAM (yé/ma), pas de résultatif V+ra (is_transitive=True
                # bloque le bloc F6).
                m['O'] = _v_root
                _as = G_kg.get('coord_action_suffix', '')
                m['V'] = j(_as, G_kg.get('coord_verb_marker', ''), _v_wa_tail) if _v_wa_tail else _as
                if tree.get('est_ce_que'):
                    m['V'] = j(m['O'], m['V']); m['O'] = ''
                tree['is_transitive'] = True
        # consumption_liquid, expl:pass (passif réflexif) et aux:pass (passif grammatical)
        # exclus de la nominalization. Progressif inclus : verbe transitif sans COD
        # se nominalise même au progressif ("je suis en train de manger" → dúnli kɛ).
        elif (_intrans_type in ('nominalized', 'ACTION', 'support')
                and _sc not in INTRANS_SC and _sc not in ('having', 'technique')
                and not _is_liquid
                and not _has_refl_pass
                and not _has_aux_pass_gramm
                and not _has_interrog_obj
                and _v_root
                # psych_emotion en prohibitif/impératif reste verbe nu (kàna sò, kàna kàsi)
                and not (tree.get('clause_type') in ('prohibitive', 'imperative')
                         and _sc == 'psych_emotion')):
            _nom_sfx  = G_kg.get('nominalization_verb_suffix', '')
            _nom_sfx2 = G_kg.get('nominalization_verb_suffix_alt', '')
            if _nom_sfx and not (_v_root.endswith(_nom_sfx) or (_nom_sfx2 and _v_root.endswith(_nom_sfx2))):
                _nominalized = _v_root + _nom_sfx
            else:
                _nominalized = _v_root

            # Transitif sans COD → TOUJOURS S TAM V+li kɛ (présent, passé, négatif).
            # Le résultatif V+ra/na est réservé au PASSIF (le riz a été mangé →
            # ìri dúnna), traité ailleurs (is_passive, exclu de ce bloc).
            # Avec COD (j'ai mangé le riz → n yé ìri dún) : m['O'] est posé →
            # ce bloc est sauté, le verbe reste nu (chemin objet standard).
            # is_transitive=True garde le TAM (yé/ma) et bloque le résultatif F6.
            m['O'] = _nominalized
            _as = G_kg.get('coord_action_suffix', '')
            m['V'] = j(_as, G_kg.get('coord_verb_marker', ''), _v_wa_tail) if _v_wa_tail else _as
            if tree.get('est_ce_que'):
                m['V'] = j(m['O'], m['V']); m['O'] = ''
            tree['is_transitive'] = True
        elif (not _is_pres and m.get('V') and not root_tok.get('is_statif')
              and _sc not in ('having', 'technique')
              and (_intrans_type == 'ABSOLU'
                   or _sc in INTRANS_SC)
              and not tree.get('neg', False)):
            # Passé positif ABSOLU ou INTRANS_SC (partir, dormir…) → résultatif
            # V+na/-ra, pas de yé. INTRANS_SC = signal KG fiable — on ignore le
            # it='ACTION' du LLM pour ces classes (toujours intransitives).
            # 'having'/'technique' exclus (báara/travailler) : garde TAM + kɛ.
            tree['is_transitive'] = False
        # Est-ce que + faire : step3 a mis V='' (copule être), le elif ci-dessus
        # était la seule source de V='kɛ'. Avec _has_interrog_obj, on le saute,
        # mais on doit restaurer V depuis le bm du root redirigé.
        if _has_interrog_obj and not m.get('V') and _v_root:
            m['V'] = _v_root
    # content_question/motion_content_question/verb_serial + INTRANS_SC + passé : F6 doit s'appliquer
    # Ex: "Quand est-il parti?" → a fáɲira, "elle alla trouver" → a táara ɲɛ́sɔ̀rɔ
    if (tree.get('clause_type') in ('content_question', 'motion_content_question',
                                    'verb_serial', 'serial_motion_pres')
            and not (tree.get('tense', 'pres') in ('pres', 'hab'))
            and not tree.get('neg', False)
            and root_tok and root_tok.get('pos') == 'VERB'
            and root_tok.get('semantic_class', '') in INTRANS_SC
            and not m.get('O')
            and m.get('V')):
        tree['is_transitive'] = False
        # NB: le suffixage du verbe ordinaire sans objet (V la / V kɛ) est
        # désormais centralisé dans step6_copule pour éviter le double-suffixage.
        # Sinon (statif, négatif INTRANS_SC, présent INTRANS_SC) : V nu → rien à faire

    # ── AIGUILLAGE INTRANSITIF (infinitif, matrice 4 cas) ────────────────────
    # Même règle que pour 'simple' mais en mode infinitif (ka … kɛ).
    # Le V peut être un groupe coordonné "dún ani ka sùnɔgɔ" ; on remplace
    # uniquement le verbe racine en tête par "kɛ" et on place verb+li dans O.
    # Progressif 'bɛ kà V' : le clause_type 'infinitive' est sélectionné par la
    # PatternRule KG (cop + VERB ROOT), mais pour le progressif le verbe doit
    # rester nu — exclure la nominalization du bloc infinitif.
    if (root_tok and root_tok.get('pos') == 'VERB'
            and not m.get('O')
            and not m.get('V_ACTION')
            and tree.get('clause_type') == 'infinitive'
            and tree.get('tense') != 'prog'
            and not root_tok.get('is_participe_passe')
            and not root_tok.get('is_passive')):
        _intrans_type = root_tok.get('intransitive_type', '')
        _v_root       = root_tok.get('bm', '')
        _sc           = root_tok.get('semantic_class', '')
        _is_liquid = (_sc == 'consumption_liquid')

        if (_intrans_type in ('nominalized', 'ACTION', 'support')
                and _sc not in INTRANS_SC and _sc not in ('having', 'technique')
                and not _is_liquid
                and _v_root):
            _nom_sfx  = G_kg.get('nominalization_verb_suffix', '')
            _nom_sfx2 = G_kg.get('nominalization_verb_suffix_alt', '')
            if _nom_sfx and not (_v_root.endswith(_nom_sfx) or (_nom_sfx2 and _v_root.endswith(_nom_sfx2))):
                _nominalized = _v_root + _nom_sfx
            else:
                _nominalized = _v_root
            m['O'] = _nominalized
            # Replace root verb at head of V with 'kɛ', preserving any coordination tail
            _v_str = m.get('V', '')
            if _v_str.startswith(_v_root):
                _act_sfx_7 = G_kg.get('coord_action_suffix', '')
                m['V'] = _act_sfx_7 + _v_str[len(_v_root):]
            else:
                m['V'] = G_kg.get('coord_action_suffix', '')

    # ── PROGRESSIF + INFINITIVE : V kɛ / Vli kɛ selon la classe ────────────────
    # clause_type='infinitive' + tense='prog' : le bloc infinitif l'exclut (tense!='prog')
    # et le bloc principal ignore 'infinitive'. Traiter ici selon la classe :
    #   action       → V kɛ  (báara kɛ : pas de nominalization -li)
    #   transitive sans COD → Vli kɛ (dúnli kɛ : nominalization requise)
    if (root_tok and root_tok.get('pos') == 'VERB'
            and tree.get('clause_type') == 'infinitive'
            and tree.get('tense') == 'prog'
            and root_tok.get('bm')
            and not m.get('O')):
        _as   = G_kg.get('coord_action_suffix', '')
        _v_cur = m.get('V', '') or root_tok.get('bm', '')
        _it2  = root_tok.get('intransitive_type', '')
        _sc2  = root_tok.get('semantic_class', '')
        if _sc2 == 'action':
            if _as and not _v_cur.endswith(_as):
                m['V'] = j(_v_cur, _as)
        elif (_it2 in ('ACTION', 'nominalized', 'support')
              and _sc2 not in INTRANS_SC
              and _sc2 not in ('having', 'technique', 'consumption_liquid')):
            # Transitive sans COD → Vli/Vni kɛ (ex: manger→dúnli kɛ, acheter→sanni kɛ)
            _nom_sfx2     = G_kg.get('nominalization_verb_suffix', '')
            _nom_sfx2_alt = G_kg.get('nominalization_verb_suffix_alt', '')
            if _nom_sfx2 and not _v_cur.endswith((_nom_sfx2, _nom_sfx2_alt or _nom_sfx2)):
                m['O'] = _v_cur + _nom_sfx2
            else:
                m['O'] = _v_cur
            m['V'] = _as
        tree['is_transitive'] = True

    # ── ADV ORPHELINS dep='dep' ───────────────────────────────────────────────
    # Adverbes sententiels (vraiment, réellement…) que spaCy parse dep='dep'
    # dans les inversions/copulatives : step5 les filtre car dep≠advmod.
    # Si bm est posé et non encore traité, on les accroche en slot ADV (fin).
    _skip_adv_roles = {'negation', 'copula', 'subject', 'reflexive',
                       'interrogative', 'relative', 'temporal', 'temporal_already',
                       'locative', 'comitative', 'privative'}
    for _adv_t in T:
        if (_adv_t.get('pos') == 'ADV'
                and _adv_t.get('dep') == 'dep'
                and _adv_t.get('bm')
                and _adv_t['orig_index'] not in processed_indices
                and _adv_t.get('role') not in _skip_adv_roles):
            _adv_bm = _adv_t['bm']
            m['ADV'] = (m['ADV'] + ' ' + _adv_bm).strip() if m.get('ADV') else _adv_bm
            processed_indices.add(_adv_t['orig_index'])

    # ── DÉDUPLICATION WAGONS ──────────────────────────────────────────────────
    seen_wagons = set()
    deduped     = []
    for w in m['OBL_ALL']:
        if isinstance(w, dict):
            key = (w.get('HEAD', ''), w.get('MARKER', ''), w.get('COMPOUND', ''))
            if key not in seen_wagons:
                seen_wagons.add(key)
                deduped.append(w)
        else:
            deduped.append(w)
    m['OBL_ALL'] = deduped

    # ── SLOTS FINAUX ──────────────────────────────────────────────────────────
    m['SLOTS'] = {}
    idx_s = 1
    if m['S']:
        m['SLOTS'][f'X{idx_s}'] = m['S']; idx_s += 1
    if tree['tam'] and tree['clause_type'] != 'noun_phrase':
        m['SLOTS'][f'X{idx_s}'] = tree['tam']; idx_s += 1
    if m['O']:
        m['SLOTS'][f'X{idx_s}'] = m['O']; idx_s += 1
    if m['V']:
        m['SLOTS'][f'X{idx_s}'] = m['V']; idx_s += 1

    ordered_keys  = sorted(m['SLOTS'].keys(), key=lambda x: int(x[1:]))
    tree['final_string']      = j(*[m['SLOTS'][k] for k in ordered_keys])
    tree['local_clause_type'] = tree['clause_type']
    tree['_tokens']  = T

    return tree
