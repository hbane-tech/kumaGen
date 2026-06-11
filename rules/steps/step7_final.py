"""
rules/steps/step7_final.py
Étape 7 : harmonisation nominale, ccomp (complétive), intransitif kɛ,
déduplication wagons, slots finaux.
"""
from rules.core import j, get_bounded_chunk_tokens, INTRANS_SC, _resolve_tam
from rules.steps.step2_sujet import _build_subj_chain
from rules.steps.step4_objet.objet_standard import _build_genitive_chain


def run(T, tree, m, processed_indices, G_kg, NX_G, root_tok,
        get_bounded_chunk_tokens_fn=None):
    """Finalise tree/m. Retourne tree."""

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
            elif m['O'] == 'w':
                m['O'] = f"{m['S']}w"
                m['S'] = ''
            elif m['S'] in m['O']:
                m['S'] = ''
            else:
                m['O'] = j(m['S'], m['O'])
                m['S'] = ''

    # ── SUBORDONNÉE COMPLÉTIVE (ccomp) ───────────────────────────────────────
    if tree.get('clause_type') == 'simple':
        ccomp_tok = next((x for x in T if x.get('dep') == 'ccomp'), None)
        if ccomp_tok:
            ccomp_subj = next((x for x in T
                               if x.get('dep') == 'nsubj'
                               and x.get('head_index') == ccomp_tok['orig_index']), None)
            ccomp_head_bm = ccomp_tok.get('bm') or f"[{ccomp_tok.get('lemma')}]"

            # Chaîne génitif pour le sujet du ccomp (ex: raison de la venue de qqu'un)
            _subj_bm = (_build_subj_chain(ccomp_subj, T, G_kg)
                        if ccomp_subj else '')
            if ccomp_subj and ccomp_subj.get('is_plural') and not _subj_bm.endswith('w'):
                if ccomp_subj.get('pos') not in ('PRON', 'PROPN'):
                    _subj_bm += 'w'

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
                    'particle': 'ka tɛmɛ',
                    'ref_bm':   _c_comp_ref.get('bm', ''),
                    'neg':      tree.get('neg', False),
                }
            elif ccomp_tok.get('pos') == 'VERB':
                # ccomp à tête verbale (… disant qu'elle partait) → clause
                # verbale ko S TAM (O) V, et non équative ko S yé O yé.
                _ccomp_neg = any(
                    x.get('head_index') == ccomp_tok['orig_index']
                    and x.get('dep') in ('advmod', 'fixed', 'mark')
                    and (str(x.get('surface', '')).lower().rstrip("'")
                         in _neg_surfs_c or x.get('role') == 'negation')
                    for x in T)
                _ccomp_tam = _resolve_tam(
                    ccomp_tok.get('tense', 'pres'), _ccomp_neg, G_kg) or 'bɛ'
                _ccomp_obj = next((x for x in T
                                   if x.get('dep') in ('obj', 'iobj')
                                   and x.get('head_index') == ccomp_tok['orig_index']
                                   and x.get('bm')), None)
                m['CCOMP'] = {
                    'type': 'verbal',
                    'S':    _subj_bm,
                    'tam':  _ccomp_tam,
                    'O':    (_ccomp_obj.get('bm', '') if _ccomp_obj else ''),
                    'V':    ccomp_head_bm,
                }
                if _ccomp_obj:
                    processed_indices.add(_ccomp_obj['orig_index'])
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
                        and _ccomp_o and not _ccomp_o.endswith('w')
                        and ccomp_tok.get('pos') not in ('PRON', 'PROPN')):
                    _ccomp_o += 'w'
                m['CCOMP'] = {
                    'S':   _subj_bm,
                    'tam': G_kg.get('equative_marker', 'yé') or 'yé',
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
            and tree.get('clause_type') == 'simple'
            and not root_tok.get('is_participe_passe')
            and not root_tok.get('is_passive')
            and not root_tok.get('is_refl_passive')):
        _intrans_type = root_tok.get('intransitive_type', '')
        _is_pres = (tree.get('tense', 'pres') in ('pres', 'hab'))
        _v_root = root_tok.get('bm', '')
        # Classes sémantiques vraiment autonomes (ne prennent pas de -li)
        _sc = root_tok.get('semantic_class', '')

        # Verbe d'ACTIVITÉ intransitif (travailler=báara) : nom d'action.
        # On se base sur semantic_class=='action' (signal FIABLE) et NON sur
        # intransitive_type (verdict LLM ACTION/ABSOLU qui oscille pour
        # travailler). « Intransitive verb of ACTION don't take li kɛ » :
        #   présent/imparfait : S TAM V la  (n bɛ báara la ; n tùn bɛ báara la)
        #   passé/futur        : S TAM V kɛ  (n ye báara kɛ ; n ma báara kɛ)
        # (≠ manger=consumption → li kɛ ; ≠ dormir/parler ∈ INTRANS_SC → V nu)
        # Négatif présent EXCLU ici (laissé au traitement existant) ; négatif
        # passé/futur AUTORISÉ → garde le 'kɛ' (je n'ai pas travaillé = n ma báara kɛ).
        if (_sc == 'action'
                and _v_root and not root_tok.get('is_statif')
                and (not tree.get('neg', False) or not _is_pres)):
            if _is_pres:
                m['V'] = j(_v_root, 'la')
            else:
                # Passé/futur (positif ET négatif) : nom d'action + kɛ, on garde
                # le TAM (yé/ma), pas de résultatif V+ra (is_transitive=True
                # bloque le bloc F6).
                m['O'] = _v_root
                m['V'] = 'kɛ'
                tree['is_transitive'] = True
        # 'other' = LLM a échoué à classifier → ne pas supposer nominalisable
        elif (_intrans_type in ('nominalized', 'ACTION', 'support')
                and _sc not in INTRANS_SC and _sc != 'other' and _v_root):
            _action_noun = root_tok.get('action_noun')
            if _action_noun:
                _nominalized = _action_noun
            elif not (_v_root.endswith('li') or _v_root.endswith('ni')):
                _nominalized = _v_root + 'li'
            else:
                _nominalized = _v_root

            if _is_pres or tree.get('neg', False):
                # Présent/hab et passé négatif : S TAM V+li kɛ
                # (V+li la = progressif — réservé à être-en-train-de)
                m['O'] = _nominalized
                m['V'] = 'kɛ'
            else:
                # Passé positif sans objet : forme résultative V+na/-ra
                tree['is_transitive'] = False
        elif (not _is_pres and m.get('V') and not root_tok.get('is_statif')
              and (_intrans_type == 'ABSOLU'
                   or (_sc in INTRANS_SC and _intrans_type != 'ACTION'))
              and not tree.get('neg', False)):
            # Passé positif ABSOLU ou INTRANS_SC sans ACTION (partir, parler…)
            # → forme résultative V+na/-ra, pas de yé
            # Les verbes ACTION dans INTRANS_SC (báara…) gardent TAM + V + kɛ
            tree['is_transitive'] = False
        # NB: le suffixage du verbe ordinaire sans objet (V la / V kɛ) est
        # désormais centralisé dans step6_copule pour éviter le double-suffixage.
        # Sinon (statif, négatif INTRANS_SC, présent INTRANS_SC) : V nu → rien à faire

    # ── AIGUILLAGE INTRANSITIF (infinitif, matrice 4 cas) ────────────────────
    # Même règle que pour 'simple' mais en mode infinitif (ka … kɛ).
    # Le V peut être un groupe coordonné "dún ani ka sùnɔgɔ" ; on remplace
    # uniquement le verbe racine en tête par "kɛ" et on place verb+li dans O.
    if (root_tok and root_tok.get('pos') == 'VERB'
            and not m.get('O')
            and not m.get('V_ACTION')
            and tree.get('clause_type') == 'infinitive'
            and not root_tok.get('is_participe_passe')
            and not root_tok.get('is_passive')):
        _intrans_type = root_tok.get('intransitive_type', '')
        _v_root       = root_tok.get('bm', '')
        _sc           = root_tok.get('semantic_class', '')

        if (_intrans_type in ('nominalized', 'ACTION', 'support')
                and _sc not in INTRANS_SC and _sc != 'other' and _v_root):
            _action_noun = root_tok.get('action_noun')
            if _action_noun:
                _nominalized = _action_noun
            elif not (_v_root.endswith('li') or _v_root.endswith('ni')):
                _nominalized = _v_root + 'li'
            else:
                _nominalized = _v_root
            m['O'] = _nominalized
            # Replace root verb at head of V with 'kɛ', preserving any coordination tail
            _v_str = m.get('V', '')
            if _v_str.startswith(_v_root):
                m['V'] = 'kɛ' + _v_str[len(_v_root):]
            else:
                m['V'] = 'kɛ'

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
