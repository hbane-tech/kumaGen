"""
rules/r1_r60_engine.py
Moteur de règles Pure Data-Driven basé sur NetworkX.
Découpage récursif maximal par sous-graphes descendants (Chunks structurés).

FIXES appliqués sur la base du Document 6 :
  F1. Tense transfer : AUX aux:tense past → ROOT tense=past
  F2. xcomp ADJ séparé des PROPN obl:mod dans le slot O
  F3. Branche résultative dans tree_to_bambara (xcomp+past → kɛra O ye)
  F4. participial_to → ADV = S_bm + V_bm + tɔ
  F5. Chunk sujet : pas de duplication de la tête ROOT dans le bloc O
  F6. Tense=past intransitif : V+ra, TAM=''
"""
import networkx as nx


def j(*p):
    return ' '.join(str(x) for x in p if x and str(x).strip() and str(x).lower() != 'null')


def _resolve_genitive_chain(head_tok, all_tokens):
    """
    Résout récursivement la chaîne de génitifs bambara.
    Règle : possesseur AVANT possédé à chaque niveau.
    'le frère de la fille de mon ami' → n terikɛ denmuso bálimakɛ
      ami(terikɛ) + possessif(n) → n terikɛ
      fille(denmuso) + [n terikɛ] → n terikɛ denmuso
      frère(bálimakɛ) + [n terikɛ denmuso] → n terikɛ denmuso bálimakɛ
    """
    nmod = next((t for t in all_tokens
                 if t.get('dep') == 'nmod'
                 and t.get('head_index') == head_tok['orig_index']), None)
    possessif = next((t for t in all_tokens
                      if t.get('dep') == 'det'
                      and t.get('role') in ('pronoun', 'possessive')
                      and t.get('head_index') == head_tok['orig_index']), None)
    head_bm = head_tok.get('bm', '')
    if nmod:
        nmod_resolved = _resolve_genitive_chain(nmod, all_tokens)
        return j(nmod_resolved, head_bm)
    elif possessif:
        return j(possessif.get('bm', ''), head_bm)
    else:
        return head_bm


_GENITIVE_FALLBACK = 'ka'

_TAM_HARDCODED = {
    ('pres',  False): 'bɛ',      ('pres',  True):  'tɛ',
    ('past',  False): 'yé',      ('past',  True):  'ma',
    ('fut',   False): 'bɛ na',   ('fut',   True):  'tɛ na',
    ('cond',  False): 'bɛ na',   ('cond',  True):  'tɛ na',
    ('imp',   False): '',        ('imp',   True):  '',
    ('prog',  False): 'bɛ kà',   ('prog',  True):  'tɛ kà',
    ('hab',   False): 'tùn bɛ',  ('hab',   True):  'tùn tɛ',
    ('plup',  False): 'tùn yé',  ('plup',  True):  'tùn ma',
}


def _resolve_tam(tense: str, neg: bool, grammar: dict) -> str:
    kg = grammar.get('tam_table', {})
    if kg:
        return kg.get((tense, neg), _TAM_HARDCODED.get((tense, neg), ''))
    return _TAM_HARDCODED.get((tense, neg), '')


_GRAMMAR_FALLBACK = {
    'locative_markers':  set(),
    'temporal_markers':  set(),
    'genitive_marker':   'ka',
    'demonstrative_suffix': 'in',
    'resultative_marker': 'ye',
    'tam_default':       'bɛ'
}


def build_tree(tokens, db=None, grammar=None):
    G_kg = grammar or {}
    T = tokens

    # ── GRAPH NetworkX ────────────────────────────────────────────────────────
    NX_G = nx.DiGraph()
    for t in T:
        NX_G.add_node(t['orig_index'], token=t)
        if t['orig_index'] != t['head_index'] and t['head_index'] >= 0:
            NX_G.add_edge(t['head_index'], t['orig_index'])

    root_tok = next((t for t in T if t.get('is_root')), None)
    xcomp_verb_tok = next((t for t in T
                           if t.get('dep') == 'xcomp' and t.get('pos') == 'VERB'), None)

    # ── F2: xcomp ADJ séparé — ne pas mélanger avec PROPN obl:mod ────────────
    xcomp_adj_tok = next((t for t in T
                          if t.get('dep') == 'xcomp'
                          and t.get('pos') in ('ADJ', 'NOUN', 'PROPN')
                          and t.get('pos') != 'VERB'), None)

    clause_type_init = 'simple'
    if xcomp_verb_tok:
        clause_type_init = 'verb_serial'

    tree = {
        'clause_type': clause_type_init,
        'tam': G_kg.get('tam_default', 'bɛ'),
        'tense': 'pres',
        'neg': False,
        'main': {
            'S': '', 'V': '', 'V_ACTION': '', 'V_SUFFIX': '', 'O': '', 'O_COORD': '',
            'QUAL': '', 'ADV': '', 'AGENT': '', 'IOBJ': None, 'LOC': None,
            'OBL_ALL': [], 'CCOMP': '', 'O_IS_XCOMP': False, 'SLOTS': {}
        }
    }
    m = tree['main']
    processed_indices = set()

    # ── CHUNK BORNÉ ───────────────────────────────────────────────────────────
    def get_bounded_chunk_tokens(head_idx):
        if head_idx not in NX_G.nodes:
            return []
        descendants = nx.descendants(NX_G, head_idx)
        all_indices = descendants | {head_idx}
        valid_indices = all_indices - processed_indices

        final_indices = set()
        for idx in valid_indices:
            if idx not in NX_G.nodes:
                continue
            tok_item = NX_G.nodes[idx]['token']
            surf_clean = str(tok_item.get('surface', '')).lower().strip()

            if surf_clean in ("d", "l", "'", "\u2019", "\u00ab", "\u00bb"):
                final_indices.add(idx)
                continue

            parent_idx = tok_item.get('head_index')
            if (parent_idx in NX_G.nodes and parent_idx != head_idx):
                parent_tok = NX_G.nodes[parent_idx]['token']
                if (parent_tok.get('dep') in ('obl', 'obl:mod', 'obl:arg')
                        and parent_tok.get('head_index') != head_idx):
                    continue

            if tok_item.get('dep') == 'case' and tok_item.get('head_index') != head_idx:
                continue

            final_indices.add(idx)

        return sorted([NX_G.nodes[idx]['token'] for idx in final_indices],
                      key=lambda x: x['orig_index'])

    # ── ÉTAPE 1 : CHUNK SUJET ─────────────────────────────────────────────────
    subj_tok = next((x for x in T if x.get('dep') == 'nsubj'), None)

    if not subj_tok and root_tok and root_tok.get('pos') in ('PRON', 'NOUN'):
        subj_tok = root_tok
    if not subj_tok and root_tok:
        subj_tok = next((x for x in T
                         if x.get('pos') in ('PRON', 'NOUN')
                         and x.get('head_index') == root_tok['orig_index']
                         and x != root_tok), None)

    # Précalcul pour noun_phrase detection (utilisé dans S et O)
    root_noun = next((x for x in T if x.get('pos') == 'NOUN' and x.get('dep') == 'ROOT'), None)
    has_acl   = any(x.get('dep') == 'acl' for x in T)

    # Si la phrase est un syntagme nominal pur (root_noun + acl),
    # chercheur EST le root_noun → pas de slot S séparé, tout va dans O via head_block
    _is_pure_noun_phrase = (
        root_noun is not None and has_acl
        and subj_tok is not None
        and subj_tok.get('orig_index') == root_noun.get('orig_index')
    )

    if subj_tok and not _is_pure_noun_phrase:
        if subj_tok.get('pos') == 'PRON':
            m['S'] = subj_tok.get('bm') or f"[{subj_tok.get('lemma')}]"
            processed_indices.add(subj_tok['orig_index'])
        else:
            s_chunk = get_bounded_chunk_tokens(subj_tok['orig_index'])
            # F5: exclure le ROOT lui-même du chunk sujet si différent du subj_tok
            s_chunk = [t for t in s_chunk
                       if t.get('dep') not in ('case', 'det')
                       and t.get('pos') not in ('PUNCT', 'SYM')
                       and not (t.get('is_root') and t != subj_tok)]

            # Résolution du nmod locatif dans S : pas de marqueur locatif dans le slot S
            # (la/kɔnɔ entre nmod et nsubj → juxtaposition)
            nmod_s = next((x for x in s_chunk
                           if x.get('dep') == 'nmod'
                           and x.get('head_index') == subj_tok['orig_index']), None)
            if nmod_s:
                nmod_case = next((x for x in T
                                  if x.get('dep') == 'case'
                                  and x.get('head_index') == nmod_s['orig_index']), None)
                case_marker = nmod_case.get('bm_marker', '') if nmod_case else ''
                _loc = G_kg.get('locative_markers', set())
                _tmp = G_kg.get('temporal_markers', set())
                if case_marker in _loc or case_marker in _tmp:
                    # locatif → juxtaposition (pas de marqueur dans S)
                    sep = ''
                elif case_marker:
                    sep = case_marker
                else:
                    sep = G_kg.get('genitive_marker', '')
                nmod_bm = nmod_s.get('bm') or nmod_s.get('surface', '')
                # bm_suffix sur le sujet
                subj_bm = subj_tok.get('bm') or subj_tok.get('surface', '')
                if subj_tok.get('bm_suffix'):
                    subj_bm = subj_bm + subj_tok['bm_suffix']
                # Pluriel sur le sujet
                if (subj_tok.get('is_plural')
                        and subj_tok.get('pos') not in ('PRON', 'PROPN')
                        and not subj_bm.endswith('w')):
                    subj_bm = subj_bm + 'w'
                m['S'] = j(nmod_bm, sep, subj_bm)
                processed_indices.add(nmod_s['orig_index'])
                if nmod_case:
                    processed_indices.add(nmod_case['orig_index'])
            else:
                subj_bm = subj_tok.get('bm') or subj_tok.get('surface', '')
                if subj_tok.get('bm_suffix'):
                    subj_bm = subj_bm + subj_tok['bm_suffix']
                # Pluriel sur le sujet
                if (subj_tok.get('is_plural')
                        and subj_tok.get('pos') not in ('PRON', 'PROPN')
                        and not subj_bm.endswith('w')):
                    subj_bm = subj_bm + 'w'
                flat = next((x for x in T
                             if x.get('dep') in ('flat', 'flat:name')
                             and x.get('head_index') == subj_tok['orig_index']), None)
                if flat:
                    subj_bm = j(subj_bm, flat.get('bm') or flat.get('surface', ''))
                    processed_indices.add(flat['orig_index'])

            processed_indices.update([t['orig_index'] for t in s_chunk])

    # ── ÉTAPE 2 : VERBE ROOT ──────────────────────────────────────────────────
    if root_tok and root_tok.get('pos') in ('VERB', 'AUX'):
        root_bm = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
        if root_tok.get('bm_suffix'):
            root_bm = root_bm + root_tok['bm_suffix']
        m['V'] = root_bm
        processed_indices.add(root_tok['orig_index'])

    if xcomp_verb_tok:
        m['V_ACTION'] = xcomp_verb_tok.get('bm') or f"[{xcomp_verb_tok.get('lemma')}]"
        processed_indices.add(xcomp_verb_tok['orig_index'])

    # ── DÉTECTION NÉGATION : ne/pas → tree['neg']=True, TAM='tɛ' ─────────────
    # 'ne' et 'pas' sont des ADV advmod — le moteur doit les détecter
    # et les marquer comme négation AVANT de résoudre le TAM.
    _neg_surfaces = {'ne', 'pas', 'jamais', 'plus', 'rien', 'personne'}
    _has_neg_adv = any(
        str(x.get('surface', '')).lower() in _neg_surfaces
        and x.get('dep') in ('advmod', 'fixed')
        for x in T
    )
    if _has_neg_adv:
        tree['neg'] = True
    # Marquer les tokens ne/pas comme processed (ne pas les mettre dans wagons)
    for _nt in T:
        if (str(_nt.get('surface', '')).lower() in _neg_surfaces
                and _nt.get('dep') in ('advmod', 'fixed')):
            processed_indices.add(_nt['orig_index'])

    # ── F1 : TENSE TRANSFER depuis AUX aux:tense ─────────────────────────────
    aux_tense_tok = next((x for x in T
                          if x.get('dep') in ('aux', 'aux:tense')
                          and x.get('role') == 'auxiliary'
                          and x.get('tense') in ('past', 'hab', 'plup', 'imp')), None)
    if aux_tense_tok and root_tok:
        tree['tense'] = aux_tense_tok.get('tense', 'pres')
        tree['neg']   = aux_tense_tok.get('is_neg', False)
        tree['tam']   = _resolve_tam(tree['tense'], tree['neg'], G_kg)
    elif root_tok and root_tok.get('tense') in ('past', 'fut', 'cond'):
        tree['tense'] = root_tok['tense']
        tree['neg']   = root_tok.get('is_neg', False)
        tree['tam']   = _resolve_tam(tree['tense'], tree['neg'], G_kg)
    else:
        # TAM default
        copula_tok = next((x for x in T if x.get('dep') == 'cop'), None)
        aux_tok    = next((x for x in T if x.get('dep') in ('aux', 'aux:tense')
                           and x.get('role') == 'auxiliary'), None)
        if aux_tok and not aux_tense_tok:
            tree['tense'] = aux_tok.get('tense', 'pres')
            tree['tam']   = _resolve_tam(tree['tense'], tree['neg'], G_kg)
        elif copula_tok:
            pass  # géré plus bas
        else:
            # Utiliser tree['neg'] déjà détecté (ne/pas)
            tree['tam'] = G_kg.get('tam_default', '') or _resolve_tam('pres', tree['neg'], G_kg)

    # ── F4 : participial_to → ADV = S + V + tɔ ───────────────────────────────
    part_to_tok = next((x for x in T
                        if x.get('role') == 'participial_to' and x.get('bm')), None)
    if part_to_tok:
        s_bm = (next((x.get('bm') for x in T if x.get('dep') == 'nsubj'), None)
                or next((x.get('bm') for x in T if x.get('role') == 'pronoun'), None)
                or '')
        m['ADV'] = j(s_bm, part_to_tok.get('bm', '') + 'tɔ')
        processed_indices.add(part_to_tok['orig_index'])

    # ── ÉTAPE 3 : OBJET ───────────────────────────────────────────────────────
    if root_noun and has_acl:
        # Syntagme nominal avec participe (Chercheur associé...)
        tree['clause_type'] = 'noun_phrase'
        r_idx     = root_noun['orig_index']
        child_adj = next((x for x in T if x.get('dep') == 'amod' and x.get('head_index') == r_idx), None)
        acl_tok   = next((x for x in T if x.get('dep') == 'acl'  and x.get('head_index') == r_idx), None)
        acl_val   = acl_tok.get('bm', '') if acl_tok else ''
        if acl_val and not acl_val.endswith('len'):
            acl_val += 'len'
        head_block = j(root_noun.get('bm', root_noun.get('surface', '')),
                       child_adj.get('bm', '') if child_adj else '',
                       acl_val)
        if acl_tok and acl_tok.get('bm_suffix'):
            m['V_SUFFIX'] = acl_tok.get('bm_suffix', '')
        m['O'] = head_block
        m['V'] = ''
        _noyau = {root_noun['orig_index']}
        if acl_tok:   _noyau.add(acl_tok['orig_index'])
        if child_adj: _noyau.add(child_adj['orig_index'])
        processed_indices.update(_noyau)

    elif not (root_noun and has_acl):
        # ── F2 : xcomp ADJ → O_IS_XCOMP ; PROPN obl:mod → wagon séparé ──────
        if xcomp_adj_tok:
            m['O'] = xcomp_adj_tok.get('bm') or f"[{xcomp_adj_tok.get('lemma')}]"
            m['O_IS_XCOMP'] = True
            processed_indices.add(xcomp_adj_tok['orig_index'])
        else:
            obj_tok = next((x for x in T
                            if x.get('dep') in ('obj', 'xcomp')
                            and x.get('pos') in ('NOUN', 'PROPN', 'ADJ')), None)
            if not obj_tok and root_tok and root_tok.get('pos') in ('NOUN', 'PROPN') and not has_acl:
                obj_tok = root_tok
                tree['clause_type'] = 'noun_phrase'

            if obj_tok and obj_tok['orig_index'] not in processed_indices:
                o_chunk = get_bounded_chunk_tokens(obj_tok['orig_index'])
                if tree.get('clause_type') == 'noun_phrase':
                    # Vérifier si la chaîne contient des nmod (génitifs récursifs)
                    # Si oui, garder TOUS les nmod pour la résolution récursive
                    _has_any_nmod = any(x.get('dep') == 'nmod' for x in o_chunk)
                    if not _has_any_nmod:
                        o_chunk = [x for x in o_chunk
                                   if x.get('dep') not in ('nmod', 'obl', 'obl:mod', 'case')]

                o_chunk = sorted([x for x in o_chunk
                                  if x.get('pos') not in ('PUNCT', 'SYM')],
                                 key=lambda x: x['orig_index'])

                if obj_tok.get('dep') == 'xcomp':
                    m['O_IS_XCOMP'] = True

                # Résolution récursive de la chaîne de génitifs
                # 'le frère de la fille de mon ami' → n terikɛ denmuso bálimakɛ
                # À chaque 'de', le possesseur précède le possédé (règle bambara universelle)
                tete_tok = next((x for x in o_chunk
                                 if x == obj_tok or x.get('dep') == 'ROOT'), obj_tok)
                _has_nmod_chain = any(x.get('dep') == 'nmod' for x in o_chunk)
                objet_elements = []
                if _has_nmod_chain and tete_tok:
                    # Résolution récursive : n terikɛ denmuso bálimakɛ
                    _resolved = _resolve_genitive_chain(tete_tok, o_chunk + T)
                    # Si le possessif résolu commence par le même bm que le sujet,
                    # on le garde quand même (n terikɛ denmuso = mon ami fille)
                    objet_elements = [_resolved] if _resolved else []
                elif tete_tok:
                    objet_elements.append(tete_tok.get('bm') or f"[{tete_tok.get('lemma')}]")
                # Marquer tous les tokens de la chaîne génitif comme processed
                # pour éviter qu'ils créent des wagons OBL séparés
                for _ct in o_chunk:
                    if _ct.get('dep') in ('nmod', 'det', 'case'):
                        processed_indices.add(_ct['orig_index'])
                if not objet_elements:
                    objet_elements = [t.get('bm') or f"[{t.get('lemma')}]"
                                      for t in o_chunk
                                      if t.get('dep') not in ('det','case','cop','aux','aux:tense')]
                m['O'] = j(*[x for x in objet_elements if str(x).strip() != 'ni'])
                if (obj_tok.get('is_plural') or str(obj_tok.get('surface', '')).endswith('s')) \
                        and not str(m['O']).endswith('w') \
                        and tree.get('clause_type') != 'presentative':
                    if obj_tok.get('pos') not in ('PRON', 'PROPN'):
                        m['O'] = f"{m['O']}w"
                processed_indices.update([t['orig_index'] for t in o_chunk])

    # ── ÉTAPE 4 : OBLIQUES ────────────────────────────────────────────────────
    _loc_markers = G_kg.get('locative_markers', set())
    _tmp_markers = G_kg.get('temporal_markers', set())

    sorted_tokens = sorted(T, key=lambda x: x['orig_index'])
    for tok_item in sorted_tokens:
        if tok_item['orig_index'] in processed_indices:
            continue
        if tok_item.get('pos') not in ('NOUN', 'PROPN', 'PRON', 'ADV', 'NUM'):
            continue
        dep_case = next((x for x in T
                         if x.get('dep') == 'case'
                         and x.get('head_index') == tok_item['orig_index']), None)
        if not (dep_case or tok_item.get('is_loc')
                or tok_item.get('dep') in ('obl', 'obl:mod', 'obl:arg', 'advmod', 'nmod')):
            continue

        obl_chunk = get_bounded_chunk_tokens(tok_item['orig_index'])
        obl_chunk = [x for x in obl_chunk
                     if x.get('pos') not in ('PUNCT', 'SYM')
                     and str(x.get('surface', '')).strip() not in ('-', '–', '—')]
        if not obl_chunk:
            continue

        amod_toks = sorted([x for x in obl_chunk if x.get('dep') == 'amod'],
                           key=lambda x: x['orig_index'])
        # Un nmod avec ADP locatif/temporel propre → wagon séparé, pas COMPOUND
        def _nmod_has_loc_adp(nmod_t):
            own_adp = next((p for p in T
                            if p.get('dep') == 'case'
                            and p.get('head_index') == nmod_t['orig_index']), None)
            if not own_adp: return False
            mk = own_adp.get('bm_marker', '')
            return mk in _loc_markers or mk in _tmp_markers

        compound_toks = sorted([x for x in obl_chunk
                                 if x.get('dep') in ('nmod', 'nummod')
                                 and x != tok_item
                                 and not _nmod_has_loc_adp(x)],
                                key=lambda x: x['orig_index'])
        demo_tok = next((x for x in obl_chunk
                         if x.get('role') == 'demonstrative'
                         or str(x.get('surface', '')).lower() in ('ce', 'cet', 'cette', 'ces')), None)

        head_base = tok_item.get('bm') or f"[{tok_item.get('lemma')}]"
        if tok_item.get('bm_suffix'):
            head_base = head_base + tok_item['bm_suffix']
        if (tok_item.get('pos') not in ('PROPN', 'PRON')
                and not head_base.startswith('[')
                and (tok_item.get('is_plural') or str(tok_item.get('surface', '')).endswith('s'))
                and not head_base.endswith('w')):
            head_base = f"{head_base}w"

        marker_val = dep_case.get('bm_marker', '') if dep_case else ''

        if (marker_val in _loc_markers or tok_item.get('is_loc')
                or (dep_case and dep_case.get('role') == 'locative')):
            clause_type_val = 'locative'
            if not marker_val:
                marker_val = 'la'
        elif (dep_case and str(dep_case.get('surface', '')).lower() in ('avec', 'with', 'et')
              or dep_case and dep_case.get('role') == 'comitative'):
            clause_type_val = 'comitative'
            marker_val = G_kg.get('comitative_marker', 'ni') or 'ni'
        elif (dep_case and str(dep_case.get('surface', '')).lower() in ('avec', 'with', 'et')
              or dep_case and dep_case.get('role') == 'comitative'):
            clause_type_val = 'comitative'
            marker_val = G_kg.get('comitative_marker', 'ni') or 'ni'
        elif (marker_val in _tmp_markers
              or tok_item.get('dep') in ('obl:mod', 'advmod')
              or tok_item.get('role') == 'temporal'):
            clause_type_val = 'temporal'
        else:
            clause_type_val = 'simple'

        compound_elements = []
        absorbed_amods = set()
        for c_tok in compound_toks:
            c_val = c_tok.get('bm') or f"[{c_tok.get('lemma')}]"
            c_amod = next((x for x in obl_chunk
                           if x.get('dep') == 'amod'
                           and x.get('head_index') == c_tok['orig_index']), None)
            if c_amod:
                c_val = j(c_val, c_amod.get('bm') or f"[{c_amod.get('lemma')}]")
                absorbed_amods.add(c_amod['orig_index'])
            compound_elements.append(c_val)
        compound_base = j(*compound_elements) if compound_elements else ''

        clean_amod_toks = [a for a in amod_toks if a['orig_index'] not in absorbed_amods]
        amod_list = []
        for a in clean_amod_toks:
            a_val = a.get('bm') or f"[{a.get('lemma')}]"
            amod_list.append(a_val)
        mod_compiled = j(*amod_list)

        if demo_tok and compound_base:
            suff_val = G_kg.get('demonstrative_suffix', 'in') or 'in'
            if not compound_base.endswith(suff_val):
                compound_base = f"{compound_base} {suff_val}"
            pref_val = demo_tok.get('bm') or 'nin'
            suff_dict_val = ''
        else:
            pref_val = demo_tok.get('bm') if demo_tok else ''
            suff_dict_val = G_kg.get('demonstrative_suffix', 'in') if demo_tok else ''

        # Avant d'ajouter ce wagon, créer les sous-wagons locatifs des nmod exclus
        # (nmod avec ADP locatif/temporel → wagon séparé)
        loc_nmod_toks = sorted(
            [x for x in obl_chunk
             if x.get('dep') == 'nmod' and x != tok_item and _nmod_has_loc_adp(x)],
            key=lambda x: x['orig_index']
        )
        for loc_nmod in loc_nmod_toks:
            loc_adp = next((p for p in T
                            if p.get('dep') == 'case'
                            and p.get('head_index') == loc_nmod['orig_index']), None)
            loc_marker = loc_adp.get('bm_marker', '') if loc_adp else ''
            loc_lct = ('locative' if loc_marker in _loc_markers
                       else 'temporal' if loc_marker in _tmp_markers else 'simple')
            loc_head = loc_nmod.get('bm') or f"[{loc_nmod.get('lemma')}]"
            m['OBL_ALL'].append({
                'HEAD':              loc_head,
                'MARKER':            loc_marker,
                'local_clause_type': loc_lct,
                'COMPOUND':          '',
                'MOD':               '',
                'DEM_PREF':          '',
                'DEM_SUFF':          '',
                'DEP_TYPE':          'case',
            })
            processed_indices.add(loc_nmod['orig_index'])
            if loc_adp:
                processed_indices.add(loc_adp['orig_index'])

        wagon_dict = {
            'HEAD':                head_base,
            'MARKER':              marker_val if marker_val else '',
            'local_clause_type':   clause_type_val,
            'COMPOUND':            compound_base,
            'MOD':                 mod_compiled,
            'DEM_PREF':            pref_val,
            'DEM_SUFF':            suff_dict_val,
            'DEP_TYPE':            'case' if dep_case else '',
        }
        m['OBL_ALL'].append(wagon_dict)
        processed_indices.update([t['orig_index'] for t in obl_chunk])

    # ── ÉTAPE 5 : COPULE ET TAM ───────────────────────────────────────────────
    copula_tok = next((x for x in T if x.get('dep') == 'cop'), None)
    has_with   = any(str(x.get('surface', '')).lower().strip() in ('avec', 'et') for x in T)

    # Copule valide seulement si elle est rattachée au ROOT (pas à un ccomp)
    _root_idx = root_tok['orig_index'] if root_tok else -1
    _cop_is_on_root = copula_tok and copula_tok.get('head_index') == _root_idx
    if _cop_is_on_root:
        if has_with:
            tree['clause_type'] = 'presentative'
            tree['tam'] = G_kg.get('comitative_marker', 'ni')
        else:
            tree['clause_type'] = 'equative'
            tree['tam'] = G_kg.get('equative_marker', 'yé')
    elif not aux_tense_tok:
        if tree.get('neg'):
            tree['tam'] = 'tɛ na' if tree.get('tense') == 'fut' else 'tɛ'
        elif tree.get('clause_type') != 'noun_phrase':
            if not tree.get('tam'):
                tree['tam'] = G_kg.get('tam_default', '') or _resolve_tam('pres', False, G_kg)

    # ── ÉTAPE 6 : HARMONISATION NOMINALE ─────────────────────────────────────
    if tree.get('clause_type') == 'noun_phrase':
        if m['S'] and not m['O']:
            m['O'] = m['S']
            m['S'] = ''
        elif m['S'] and m['O'] and m['S'] != m['O']:
            if m['O'] == 'w':
                m['O'] = f"{m['S']}w"
            elif m['S'] in m['O']:
                # S déjà inclus dans O (ex: chercheur déjà dans head_block)
                pass
            else:
                m['O'] = j(m['S'], m['O'])
            m['S'] = ''


    # ── SUBORDONNÉE COMPLÉTIVE (ccomp) pour clauses simples ─────────────
    # 'montrent qu'ils sont maîtres du jeu' → ko olu ye jɛkajɛbagaw ye
    if tree.get('clause_type') == 'simple':
        ccomp_tok = next((x for x in T
                          if x.get('dep') == 'ccomp'
                          and x['orig_index'] not in processed_indices), None)
        if ccomp_tok:
            # Construire la subordonnée : sujet_ccomp + cop + tête_ccomp
            ccomp_subj = next((x for x in T
                               if x.get('dep') == 'nsubj'
                               and x.get('head_index') == ccomp_tok['orig_index']), None)
            ccomp_nmod = next((x for x in T
                               if x.get('dep') == 'nmod'
                               and x.get('head_index') == ccomp_tok['orig_index']), None)
            ccomp_head_bm = ccomp_tok.get('bm') or f"[{ccomp_tok.get('lemma')}]"
            if ccomp_nmod:
                ccomp_head_bm = j(ccomp_nmod.get('bm', ''), G_kg.get('genitive_marker',''), ccomp_head_bm)
            ccomp_subj_bm = ccomp_subj.get('bm', '') if ccomp_subj else ''
            # Stocker dans m['CCOMP'] pour tree_to_bambara
            m['CCOMP'] = {
                'S': ccomp_subj_bm,
                'tam': G_kg.get('equative_marker', 'yé') or 'yé',
                'O': ccomp_head_bm,
                'V': '',
            }
            processed_indices.add(ccomp_tok['orig_index'])
            if ccomp_subj: processed_indices.add(ccomp_subj['orig_index'])
            if ccomp_nmod:
                processed_indices.add(ccomp_nmod['orig_index'])
                _cn_adp = next((x for x in T if x.get('dep') == 'case'
                                and x.get('head_index') == ccomp_nmod['orig_index']), None)
                if _cn_adp: processed_indices.add(_cn_adp['orig_index'])

    # ── DÉDUPLICATION DES WAGONS IDENTIQUES ─────────────────────────────
    # Supprimer les wagons avec HEAD+MARKER identiques (phrase dupliquée dans input)
    seen_wagons = set()
    deduped = []
    for w in m['OBL_ALL']:
        if isinstance(w, dict):
            key = (w.get('HEAD', ''), w.get('MARKER', ''), w.get('COMPOUND', ''))
            if key not in seen_wagons:
                seen_wagons.add(key)
                deduped.append(w)
        else:
            deduped.append(w)
    m['OBL_ALL'] = deduped

    # ── ÉTAPE 6 : SLOTS FINAUX ─────────────────────────────────────────────
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
    ordered_keys = sorted(m['SLOTS'].keys(), key=lambda x: int(x[1:]))
    tree['final_string'] = j(*[m['SLOTS'][k] for k in ordered_keys])
    tree['local_clause_type'] = tree['clause_type']
    tree['_tokens'] = T

    return tree


def tree_to_bambara(tree, G=None, grammar=None):
    G = grammar or G or {}

    m   = tree['main']
    ct  = tree['clause_type']
    S   = m.get('S', '') or ''
    O   = m.get('O', '') or ''
    V   = m.get('V', '') or ''
    V_ACT = m.get('V_ACTION', '') or ''
    V_SUF = m.get('V_SUFFIX', '') or ''
    ADV   = m.get('ADV', '') or ''
    neg   = tree.get('neg', False)
    tn    = tree.get('tense', 'pres')

    tam_val = tree.get('tam', '')
    if not tam_val or tam_val.strip() == '':
        tam_val = 'bɛ'
    TAM = tam_val

    # ── COMPILATION DES WAGONS OBLIQUES ──────────────────────────────────────
    # Compile ccomp subclause (montrer que...) → ko S yé O yé
    _ccomp_data = m.get('CCOMP')
    _ccomp_str = ''
    if isinstance(_ccomp_data, dict) and _ccomp_data.get('O'):
        _ko = G.get('reported_intro', 'ko') or 'ko'
        _cs = _ccomp_data.get('S', '')
        _co = _ccomp_data.get('O', '')
        _ct = _ccomp_data.get('tam', 'yé')
        _ccomp_str = j(_ko, _cs, _ct, _co, _ct)  # ko olu yé maître ka jɛkajɛbagaw yé

    obl_strings = []
    for c in (m.get('OBL_ALL') or []):
        comp   = c.get('COMPOUND', '')
        head   = c.get('HEAD', '')
        mod    = c.get('MOD', '')
        pref   = c.get('DEM_PREF', '')
        suff   = c.get('DEM_SUFF', '')
        marker = c.get('MARKER', '')
        lct    = c.get('local_clause_type', 'simple')

        gen_marker = G.get('genitive_marker', '') or _GENITIVE_FALLBACK
        # COMPOUND + HEAD : toujours séparés par le génitif ka
        # sauf si le wagon est locatif/temporel → juxtaposition directe
        if comp:
            if lct in ('locative', 'temporal'):
                # locatif : COMPOUND juste avant HEAD (pas de ka)
                noun_base = j(comp, head)
            else:
                noun_base = j(comp, gen_marker, head)
        else:
            noun_base = head
        if mod:
            noun_base = j(noun_base, mod)
        full_chunk = j(pref, noun_base, suff) if (pref or suff) else noun_base
        if marker:
            full_chunk = j(full_chunk, marker)
        obl_strings.append(full_chunk)

    # ── F6 : PAST INTRANSITIF (pas d'objet direct) → V+ra, TAM='' ────────────
    _o_is_xcomp = m.get('O_IS_XCOMP', False)
    if tn == 'past' and not O and not _o_is_xcomp:
        if V and not V.endswith('ra') and not V.endswith('na'):
            V += 'ra'
        TAM = ''

    # ── AFFICHAGE SLOTS ───────────────────────────────────────────────────────
    print('\n  📦 SLOTS STRUCTURELS FINAUX :')
    if S:     print(f'     [ S     ] → {S}')
    if TAM and ct != 'noun_phrase':
              print(f'     [ TAM   ] → {TAM}')
    if O:     print(f'     [ O     ] → {O}')
    if V:     print(f'     [ V     ] → {V}')
    if V_ACT: print(f'     [ V_ACT ] → {V_ACT}')
    if ADV:   print(f'     [ ADV   ] → {ADV}')
    if V_SUF : print(f'    [V_SUF  ] -> {V_SUF}' )
    if _ccomp_str : print(f'    [ccomp_str  ] -> {_ccomp_str}' )

    
    for _i, _xv in enumerate(obl_strings):
        lct = (m['OBL_ALL'][_i].get('local_clause_type', '')
               if _i < len(m.get('OBL_ALL', [])) else '')
        print(f'     [ X{_i+1:<3d}   ] → {_xv}  ({lct})')
    print(f"  🏷️  clause_type = {ct}")

    # ── MATRICE DE RÈGLES ─────────────────────────────────────────────────────

    # ── F3 : RÉSULTATIF PASSÉ (xcomp + past) ─────────────────────────────────
    # Pattern : S [locatifs], o V-ra O ye
    # Ex: kùnnafoni dàbiɲama Mali kɔnɔ, o kɛra gɛ̀lɛn ye
    if (ct in ('simple', 'complex', 'conditional', 'temporal', 'relative_post')
            and _o_is_xcomp and O and tn == 'past'
            and V and not V.endswith('ra') and not V.endswith('na')):
        V_past = V + 'ra'
        x_pred = 'o ' + j(V_past, O, 'ye')
        if obl_strings:
            topic  = j(S, *obl_strings)
            result = topic + ', ' + x_pred
        else:
            result = j(S, x_pred)

    elif ct == 'verb_serial':
        # Comitatif (avec/ni) : O ni O2 avant V_ACT
        # Comitatif : O ni O2 avant V_ACT
        _com_marker = G.get('comitative_marker', 'ni') or 'ni'
        _com_heads, _other_obls = [], []
        _raw_obls = m.get('OBL_ALL', [])
        for _ci, _obl in enumerate(_raw_obls):
            if isinstance(_obl, dict) and _obl.get('local_clause_type') == 'comitative':
                # Prendre juste le HEAD du wagon (sans le marker 'ni' déjà compilé)
                _head_only = _obl.get('HEAD', '')
                if _obl.get('COMPOUND'):
                    _head_only = _obl['COMPOUND'] + ' ' + _head_only
                _com_heads.append(_head_only)
            else:
                _other_obls.append(obl_strings[_ci] if _ci < len(obl_strings) else '')
        # O ni O2 ni O3... (marker entre chaque objet)
        if _com_heads and O:
            _o_block = O + ' ' + _com_marker + ' ' + (' ' + _com_marker + ' ').join(_com_heads)
        elif _com_heads:
            _o_block = (' ' + _com_marker + ' ').join(_com_heads)
        else:
            _o_block = O
        result = j(S, TAM, V, 'ka', _o_block, V_ACT, V_SUF, *_other_obls)

    elif ct == 'interrogative':
        result = j(S, TAM, O, V, *obl_strings, 'wa ?')

    elif ct == 'equative':
        if neg:
            result = j(S, 'tɛ', O or V, 'yé')
        elif tn in ('past', 'fut', 'plup'):
            result = j(S, 'kɛra', O or V, 'yé')
        else:
            result = j(S, 'yé', O or V, 'yé')

    elif ct == 'presentative':
        if neg:
            result = j(S, 'tɛ', *obl_strings)
        else:
            # ni + O + dòn (comitative présentatif bambara)
            # 'Je suis avec la fille de mon ami' → n ni terikɛ denmuso dòn
            o_clean = O
            # Ne retirer le préfixe S que si O = 'S seul' (répétition du sujet)
            # Pas si O est une chaîne génitif commençant par le même pronom
            # Ex: O='n terikɛ denmuso' → garder le 'n' (c'est mon ami, pas le sujet)
            # Ex: O='n mùsoma' → le 'n' est possessif, garder
            # On retire seulement si O == S (strictement identique)
            if o_clean == S:
                o_clean = ''
            if o_clean:
                result = j(S, 'ni', o_clean, 'dòn', *obl_strings)
            else:
                result = j(S, 'dòn', *obl_strings)

    elif ct == 'noun_phrase':
        result = j(O, *obl_strings)

    elif ct in ('simple', 'complex', 'conditional', 'temporal',
                'relative_post', 'reported_comp', 'comitative'):
        # Adverbes temporels autonomes en tête (sini=demain, keɲɛ=hier...)
        # Seuls les ADV advmod (pos=ADV) vont en tête — pas les wagons équatifs.
        _raw_obls = m.get('OBL_ALL', [])
        _tmp_strs, _other_strs = [], []
        _tokens_ref = tree.get('_tokens', [])
        for _ci, _obl in enumerate(_raw_obls):
            _xv = obl_strings[_ci] if _ci < len(obl_strings) else ''
            if isinstance(_obl, dict) and _obl.get('local_clause_type') == 'temporal':
                # Vérifier si le HEAD vient d'un ADV advmod autonome
                _obl_head = _obl.get('HEAD', '')
                _src_tok = next((t for t in _tokens_ref
                                 if (t.get('bm') == _obl_head or t.get('surface') == _obl_head)
                                 and t.get('dep') == 'advmod'
                                 and t.get('pos') == 'ADV'), None)
                if _src_tok:
                    _tmp_strs.append(_xv)
                else:
                    _other_strs.append(_xv)
            else:
                _other_strs.append(_xv)
        if TAM in ('bɛ kà', 'tɛ kà'):
            result = j(*_tmp_strs, S, TAM, O, V, ADV, *_other_strs)
        else:
            result = j(*_tmp_strs, S, TAM, O, V, V_ACT, V_SUF, *_other_strs, _ccomp_str, ADV)

    elif ct == 'qualitative':
        result = j(S, 'man' if neg else 'ka', m.get('QUAL', ''))

    elif ct == 'locative':
        result = j(S, 'tɛ' if neg else 'bɛ', *obl_strings)

    elif ct == 'existential':
        result = j(S, 'tɛ' if neg else 'bɛ', *obl_strings)

    elif ct == 'passive':
        result = j(S, 'bɛ ka', V, *obl_strings)

    elif ct == 'imperative':
        result = j(V, O, *obl_strings)

    elif ct == 'prohibitive':
        result = j('kàna', V, O, *obl_strings)

    elif ct == 'participial_to':
        result = j(S, V, ADV, *obl_strings)

    elif ct == 'focus':
        result = j(S, 'de', TAM, O, V, *obl_strings)

    elif ct == 'exclamative':
        result = j(S, TAM, O, V, *obl_strings, 'dɛ !')

    elif ct == 'reciprocal':
        result = j(S, TAM, 'ɲɔgɔn', V, *obl_strings)

    elif ct == 'concessive':
        result = j(S, TAM, O, V, ADV, *obl_strings)

    elif ct == 'causal':
        result = j(S, TAM, O, V, ADV, *obl_strings)

    elif ct == 'relative_min':
        result = j(S, 'mìn', TAM, O, V, ',', 'ò', TAM, V, *obl_strings)

    elif ct == 'topicalised':
        result = j(O, ',', S, TAM, V, *obl_strings)

    elif ct in ('participial_len', 'participial_ta', 'participial_bali'):
        result = j(S, V, *obl_strings)

    elif ct == 'relative_post':
        result = j(S, 'mìn', TAM, O, V, *obl_strings)

    elif ct == 'refl_past':
        result = j(S, 'tun ye', V, O, *obl_strings, ADV)

    elif ct in ('reported_verb', 'reported_comp', 'reported'):
        if neg: result = j(S, 'ma', O, 'fɔ', ADV, *obl_strings)
        else:   result = j(S, TAM, O, V, ADV, *obl_strings)

    elif ct == 'content_question':
        result = j(S, TAM, O, V, *obl_strings)

    elif ct == 'noun_phrase_have':
        result = j(O, 'bɛ', S, 'fɛ', *obl_strings)

    elif ct == 'noun_phrase_inh':
        result = j(O, *obl_strings) if obl_strings else j(S, O)

    elif ct == 'comitative':
        _tokens = tree.get('_tokens', [])
        _avec = next((t for t in _tokens
                      if isinstance(t, dict) and t.get('role') == 'preposition'
                      and str(t.get('surface', '')).lower() in ('avec', 'with')), None)
        _comp = None
        if _avec:
            av_i = _avec.get('orig_index', -1)
            _cn  = next((t for t in _tokens
                         if isinstance(t, dict)
                         and t.get('pos') in ('NOUN', 'PROPN', 'PRON')
                         and (t.get('head_index', -1) == av_i
                              or _avec.get('head_index', -1) == t.get('orig_index', -1))), None)
            if _cn:
                _cp = next((t for t in _tokens
                             if isinstance(t, dict) and t.get('role') == 'possessive'
                             and t.get('head_index', -1) == _cn.get('orig_index', -1)), None)
                _bm = _cn.get('bm', '')
                if _cp:
                    _p = _cp.get('bm', '')
                    _comp = j('n', _bm) if _p == 'n' else j(_p, 'ka', _bm)
                else:
                    _comp = _bm
        companion = _comp or V
        if companion and V and V != S:
            result = j(S, 'ni', companion, TAM, V, *obl_strings)
        elif companion:
            result = j(S, 'ni', companion, 'dòn', *obl_strings)
        else:
            result = j(S, 'dòn', *obl_strings)


    else:
        result = j(S, TAM, O, V, V_ACT, V_SUF, *obl_strings, ADV)

    print(f"  ✂️  Clause 1 -> '{result}'")
    return result.strip().replace(' ,', ',').replace('« ', '«').replace(' »', '»')


# ── RULE ENGINE ───────────────────────────────────────────────────────────────
class RuleEngine:
    def __init__(self, db=None):
        self.db      = db
        self.grammar = dict(_GRAMMAR_FALLBACK)
        if db:
            try:
                from pipeline.proposition_parser import load_connector_bm
                load_connector_bm(db)
            except Exception:
                pass
            self._load_grammar()

    def _q(self, cypher, params=None):
        try:
            return self.db.query(cypher, params or {}) or []
        except Exception:
            return []

    def _marker_set(self, role: str) -> set:
        rows = self._q(
            "MATCH (p:Preposition) WHERE p.role = $role RETURN p.bm_marker AS m",
            {'role': role}
        )
        return {r['m'] for r in rows if r.get('m')}

    def _single_marker(self, cypher: str, params=None) -> str:
        rows = self._q(cypher, params)
        return rows[0].get('m', '') if rows else ''

    def _load_grammar(self):
        g = self.grammar
        g['locative_markers']  = self._marker_set('locative')
        g['temporal_markers']  = self._marker_set('temporal')

        rows = self._q(
            "MATCH (n) WHERE n.role IS NOT NULL AND n.semantic_class = 'temporal' "
            "RETURN DISTINCT n.role AS r"
        )
        g['temporal_roles'] = {r['r'] for r in rows if r.get('r')}
        g['temporal_roles'].add('temporal')

        g['genitive_marker'] = self._single_marker(
            "MATCH (p:Preposition) WHERE p.role = 'genitive' RETURN p.bm_marker AS m LIMIT 1"
        )
        g['comitative_marker'] = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'comitative' RETURN f.bm AS m LIMIT 1"
        )
        g['agent_postposition'] = self._single_marker(
            "MATCH (p:Preposition) WHERE p.role = 'agent' RETURN p.bm_marker AS m LIMIT 1"
        )
        g['purposive_marker'] = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'purposive' RETURN f.bm AS m LIMIT 1"
        )
        g['reported_intro'] = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'reported_intro' RETURN f.bm AS m LIMIT 1"
        )

        tam_rows = self._q(
            "MATCH (t:TamConfig) RETURN t.tense AS tense, t.neg AS neg, t.bm AS bm, "
            "t.aspect AS aspect, t.ctx AS ctx"
        )
        if tam_rows:
            g['tam_table'] = {
                (r['tense'], r['neg']): r['bm']
                for r in tam_rows
                if r.get('tense') is not None and r.get('neg') is not None and r.get('bm')
            }
            g['tam_default']      = g['tam_table'].get(('pres', False), '')
            g['tam_future']       = g['tam_table'].get(('fut',  False), '')
            g['progressive_tams'] = {r['bm'] for r in tam_rows
                                     if r.get('aspect') == 'progressive' and r.get('bm')}
            print(f"     TAM: {len(g['tam_table'])} entrées KG chargées.")
        else:
            g['tam_table'] = {}
            g['tam_default'] = ''
            g['tam_future']  = ''
            g['progressive_tams'] = set()
            print("     TAM: KG vide — fallback _TAM_HARDCODED actif.")

        print(f"  ✅ Grammar loaded from KG — "
              f"loc={len(g['locative_markers'])} "
              f"tmp={len(g['temporal_markers'])} "
              f"gen='{g['genitive_marker']}' "
              f"com='{g['comitative_marker']}' "
              f"tam='{g['tam_default']}'")

    def apply(self, tokens_or_tree, frame):
        if not tokens_or_tree:
            return ''
        if isinstance(tokens_or_tree, dict):
            return tree_to_bambara(tokens_or_tree, grammar=self.grammar)
        tree = build_tree(tokens_or_tree, grammar=self.grammar)
        return tree_to_bambara(tree, grammar=self.grammar)