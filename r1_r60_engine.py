"""
rules/r1_r60_engine.py
Moteur de règles Pure Data-Driven basé sur NetworkX.
Découpage récursif maximal par sous-graphes descendants (Chunks structurés).

Principes :
  - Zéro mot en dur (sauf TAM hardcoded comme fallback)
  - Tout passe par les roles/semantic_class du KG
  - être → semantic_class='copula'
  - avoir → semantic_class='having'
  - négation → role='negation'
  - expletif → role='expletive'
  - clitique → role='clitic'
"""
import networkx as nx


def j(*p):
    return ' '.join(str(x) for x in p if x and str(x).strip() and str(x).lower() != 'null')


def _resolve_genitive_chain(head_tok, all_tokens):
    """
    Résout récursivement la chaîne de génitifs bambara.
    Règle : possesseur AVANT possédé à chaque niveau.
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
        nmod_case = next((t for t in all_tokens
                          if t.get('dep') == 'case'
                          and t.get('head_index') == nmod['orig_index']), None)
        if nmod_case:
            return j(nmod_resolved, 'ka', head_bm)
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


def _is_copula(tok):
    """Détecte si un token est une copule via semantic_class ou dep."""
    if not tok:
        return False
    return (tok.get('semantic_class') == 'copula'
            or tok.get('dep') == 'cop')


def _is_avoir(tok):
    """Détecte si un token est 'avoir' via semantic_class."""
    if not tok:
        return False
    return tok.get('semantic_class') == 'having'


_GRAMMAR_FALLBACK = {
    'locative_markers':     set(),
    'temporal_markers':     set(),
    'temporal_roles':       set(),
    'genitive_marker':      'ka',
    'demonstrative_suffix': 'in',
    'resultative_marker':   'ye',
    'tam_default':          'bɛ',
    'quantifier_words':     {},
    'temporal_suffix_markers': set(),
    'privative_markers':    set(),
    'neg_surfaces':         set(),
    'expletive_roles':      {'expletive'},
    'clitic_roles':         {'clitic'},
    'relative_roles':       {'relative'},
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

    # Si root_tok est un expletif PRON et qu'un ADJ/NOUN ROOT existe → utiliser l'ADJ
    if (root_tok and root_tok.get('role') == 'expletive'
            and root_tok.get('pos') == 'PRON'):
        _real_root = next((t for t in T
                           if t.get('pos') in ('ADJ', 'NOUN')
                           and t.get('dep') == 'ROOT'), None)
        if _real_root:
            root_tok = _real_root
    # Si root_tok est un PRON élisionné (j) → chercher le vrai ROOT VERB
    elif (root_tok and root_tok.get('pos') == 'PRON'
            and len(str(root_tok.get('surface', ''))) == 1
            and any(t.get('pos') == 'VERB' and t.get('dep') == 'ROOT' for t in T)):
        _verb_root = next((t for t in T
                           if t.get('pos') == 'VERB'
                           and t.get('dep') == 'ROOT'), None)
        if _verb_root:
            root_tok = _verb_root
        _verb_root = next((t for t in T
                           if t.get('pos') == 'VERB'
                           and t.get('dep') == 'ROOT'), None)
        if _verb_root:
            root_tok = _verb_root
    
    print(f"DEBUG root_tok flags APRES: is_participe_passe={root_tok.get('is_participe_passe') if root_tok else None}")
    xcomp_verb_tok = next((t for t in T
                           if t.get('dep') == 'xcomp'
                           and t.get('pos') == 'VERB'), None)

    # ── SURFACES NÉGATIVES depuis KG ─────────────────────────────────────────
    _neg_surfaces = G_kg.get('neg_surfaces', set())

    # ── ROLES depuis KG ───────────────────────────────────────────────────────
    _expletive_roles = G_kg.get('expletive_roles', {'expletive'})
    _clitic_roles    = G_kg.get('clitic_roles', {'clitic'})
    _relative_roles  = G_kg.get('relative_roles', {'relative'})

    # ── DÉTECTION INTERROGATIVE ───────────────────────────────────────────────
    clause_type_init = 'simple'
    if xcomp_verb_tok:
        clause_type_init = 'verb_serial'

    _has_question_mark      = False
    _has_interrogative_word = False

    for x in T:
        _surf  = str(x.get('surface', '')).strip()
        _morph = str(x.get('morph', ''))
        if (x.get('role') == 'interrogative'
                or 'Int' in _morph
                or 'PronType=Int' in _morph
                or (x.get('pos') in ('PRON', 'ADJ')
                    and x.get('dep') == 'ROOT'
                    and not any(s.get('dep') == 'obj' for s in T))):
            _has_interrogative_word = True
        if '?' in _surf or (x.get('dep') == 'punct' and _surf == '?'):
            _has_question_mark = True

    if _has_question_mark:
        if _has_interrogative_word or (root_tok and root_tok.get('role') == 'interrogative'):
            clause_type_init = 'content_question'
        else:
            clause_type_init = 'interrogative'

    # nsubj avec det interrogatif → devient obj
    if clause_type_init == 'content_question':
        _interrog_subj = next((x for x in T
                               if x.get('dep') == 'nsubj'
                               and any(d.get('role') == 'interrogative'
                                       and d.get('head_index') == x['orig_index']
                                       for d in T)), None)
        if _interrog_subj:
            _interrog_subj['dep'] = 'obj'

    # ── IDENTIFICATOIRE ───────────────────────────────────────────────────────
    _has_expletive  = any(x.get('role') in _expletive_roles for x in T)
    _has_propn      = any(x.get('pos') == 'PROPN' for x in T)
    _has_pron_root  = any(x.get('pos') == 'PRON'
                          and (x.get('dep') == 'ROOT' or x.get('is_root'))
                          for x in T)
    _has_propn_root = any(x.get('pos') == 'PROPN'
                          and (x.get('dep') == 'ROOT' or x.get('is_root'))
                          for x in T)
    if _has_expletive and (_has_pron_root or _has_propn_root):
        _has_adj_any = any(x.get('pos') == 'ADJ' for x in T)
        if not _has_adj_any:
            clause_type_init = 'identificatory'
        else:
            # ADJ présent → capturer comme attribut équatif
            _adj_attr = next((x for x in T if x.get('pos') == 'ADJ'), None)
            if _adj_attr and _adj_attr.get('is_valeur') is True:
                clause_type_init = 'equative'
                # Forcer root_tok vers l'ADJ pour que l'étape 5 le traite
                root_tok = _adj_attr

    print(f"DEBUG: root_tok={root_tok}, _has_expletive={_has_expletive}, "
          f"_has_pron_root={_has_pron_root}, _has_propn={_has_propn}")

    xcomp_adj_tok = next((t for t in T
                          if t.get('dep') == 'xcomp'
                          and t.get('pos') in ('ADJ', 'NOUN', 'PROPN')
                          and t.get('pos') != 'VERB'), None)

    tree = {
        'clause_type': clause_type_init,
        'tam':   G_kg.get('tam_default', 'bɛ'),
        'tense': 'pres',
        'neg':   False,
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
        descendants  = nx.descendants(NX_G, head_idx)
        all_indices  = descendants | {head_idx}
        valid_indices = all_indices - processed_indices
        final_indices = set()
        for idx in valid_indices:
            if idx not in NX_G.nodes:
                continue
            tok_item   = NX_G.nodes[idx]['token']
            surf_clean = str(tok_item.get('surface', '')).lower().strip()
            if surf_clean in ("d", "l", "'", "\u2019", "\u00ab", "\u00bb"):
                final_indices.add(idx)
                continue
            parent_idx = tok_item.get('head_index')
            if parent_idx in NX_G.nodes and parent_idx != head_idx:
                parent_tok = NX_G.nodes[parent_idx]['token']
                if (parent_tok.get('dep') in ('obl', 'obl:mod', 'obl:arg')
                        and parent_tok.get('head_index') != head_idx):
                    continue
            if tok_item.get('dep') == 'case' and tok_item.get('head_index') != head_idx:
                continue
            final_indices.add(idx)
        return sorted([NX_G.nodes[idx]['token'] for idx in final_indices],
                      key=lambda x: x['orig_index'])

    # ── AVOIR ROOT → noun_phrase_have (possession) ──────────────────────────────
    # Cas 11/12 matrice : J'ai de l'argent → wári bɛ n bóló
    #                     J'ai un frère → bálimakɛ bɛ n fɛ
    # Détection : ROOT VERB avec semantic_class='having' + obj + pas expl:comp
    # Détecter 'il y a' : expl:subj + avoir(ROOT) → existential
    # même si 'y' est absent (spaCy tokenise parfois 'y'a' sans expl:comp)
    _is_il_ya = (
        root_tok
        and root_tok.get('semantic_class') == 'having'
        and root_tok.get('dep') == 'ROOT'
        and any(x.get('dep') in ('expl:subj', 'expl:comp') for x in T)
    )
    if _is_il_ya:
        _vrai_subj_ya2 = next((x for x in T
                               if x.get('dep') == 'obj'
                               and x.get('pos') in ('NOUN', 'PROPN')), None)
        if _vrai_subj_ya2:
            _has_loc_obl2 = any(
                x.get('dep') in ('obl', 'obl:mod', 'obl:arg')
                and any(p.get('dep') == 'case' and p.get('role') == 'locative'
                        for p in T if p.get('head_index') == x['orig_index'])
                for x in T)
            tree['clause_type'] = ('existential_localized'
                                   if _has_loc_obl2 else 'existential_absolute')
            _es2_bm = _vrai_subj_ya2.get('bm') or f"[{_vrai_subj_ya2.get('lemma')}]"
            if (_vrai_subj_ya2.get('is_plural')
                    or str(_vrai_subj_ya2.get('surface', '')).endswith('s')):
                if not _es2_bm.endswith('w'):
                    _es2_bm += 'w'
            m['S'] = ''
            m['O'] = _es2_bm
            processed_indices.add(root_tok['orig_index'])
            processed_indices.add(_vrai_subj_ya2['orig_index'])
            for _expl in T:
                if _expl.get('dep') in ('expl:subj', 'expl:comp'):
                    processed_indices.add(_expl['orig_index'])
            for _dt in T:
                if _dt.get('dep') in ('det', 'fixed') and                    _dt.get('head_index') == _vrai_subj_ya2['orig_index']:
                    processed_indices.add(_dt['orig_index'])

    _avoir_possession = (
        root_tok
        and root_tok.get('semantic_class') == 'having'
        and root_tok.get('pos') in ('VERB', 'AUX')
        and root_tok.get('dep') == 'ROOT'
        and not any(x.get('dep') in ('expl:comp', 'expl:subj') for x in T)
        and any(x.get('dep') == 'obj' for x in T)
        # Pas de xcomp → sinon c'est un verbe d'action (vouloir, devoir...)
        and not any(x.get('dep') == 'xcomp' for x in T)
    )
    if _avoir_possession:
        _obj_poss = next((x for x in T
                          if x.get('dep') == 'obj'
                          and x.get('pos') in ('NOUN', 'PROPN')
                          and x.get('bm')), None)
        _subj_poss = next((x for x in T
                           if x.get('dep') == 'nsubj'
                           and x.get('pos') == 'PRON'
                           and x.get('bm')), None)
        if _obj_poss and _subj_poss:
            # Déterminer matérielle vs abstraite via semantic_class de l'objet
            _obj_sc = _obj_poss.get('semantic_class', '')
            _poss_type = ('material'
                          if _obj_sc in ('object', 'money', 'vehicle', 'tool', 'food')
                          else 'abstract')
            tree['clause_type'] = 'noun_phrase_have'
            tree['possession_type'] = _poss_type
            # Inversion bambara : OBJ devient S, SUBJ devient OBL
            _obj_bm = _obj_poss.get('bm')
            # Pluriel de l'objet
            if (_obj_poss.get('is_plural')
                    or str(_obj_poss.get('surface', '')).endswith('s')):
                if not _obj_bm.endswith('w'):
                    _obj_bm += 'w'
            # Conj de l'objet (frère/sœur)
            _obj_conjs = [x for x in T
                          if x.get('dep') == 'conj'
                          and x.get('head_index') == _obj_poss['orig_index']
                          and x.get('bm')]
            for _oc in _obj_conjs:
                _cc_oc = next((x for x in T
                               if x.get('dep') == 'cc'
                               and x.get('head_index') == _obj_poss['orig_index']), None)
                _cc_bm_oc = _cc_oc.get('bm', '') if _cc_oc and _cc_oc.get('bm') else ''
                _obj_bm = j(_obj_bm, _cc_bm_oc, _oc.get('bm'))
                processed_indices.add(_oc['orig_index'])
                if _cc_oc: processed_indices.add(_cc_oc['orig_index'])
            m['S'] = _subj_poss.get('bm')
            m['O'] = _obj_bm
            m['V'] = ''
            processed_indices.add(_obj_poss['orig_index'])
            processed_indices.add(_subj_poss['orig_index'])
            processed_indices.add(root_tok['orig_index'])
            # Verrouiller det de l'objet
            for _dt in T:
                if _dt.get('dep') in ('det', 'fixed') and                    _dt.get('head_index') == _obj_poss['orig_index']:
                    processed_indices.add(_dt['orig_index'])

    # ── AVOIR ROOT → noun_phrase_have (possession) ──────────────────────────────
    # Cas 11/12 matrice : J'ai de l'argent → wári bɛ n bóló
    #                     J'ai un frère → bálimakɛ bɛ n fɛ
    # Détection : ROOT VERB avec semantic_class='having' + obj + pas expl:comp
    # Détecter 'il y a' : expl:subj + avoir(ROOT) → existential
    # même si 'y' est absent (spaCy tokenise parfois 'y'a' sans expl:comp)
    _is_il_ya = (
        root_tok
        and root_tok.get('semantic_class') == 'having'
        and root_tok.get('dep') == 'ROOT'
        and any(x.get('dep') in ('expl:subj', 'expl:comp') for x in T)
    )
    if _is_il_ya:
        _vrai_subj_ya2 = next((x for x in T
                               if x.get('dep') == 'obj'
                               and x.get('pos') in ('NOUN', 'PROPN')), None)
        if _vrai_subj_ya2:
            _has_loc_obl2 = any(
                x.get('dep') in ('obl', 'obl:mod', 'obl:arg')
                and any(p.get('dep') == 'case' and p.get('role') == 'locative'
                        for p in T if p.get('head_index') == x['orig_index'])
                for x in T)
            tree['clause_type'] = ('existential_localized'
                                   if _has_loc_obl2 else 'existential_absolute')
            _es2_bm = _vrai_subj_ya2.get('bm') or f"[{_vrai_subj_ya2.get('lemma')}]"
            if (_vrai_subj_ya2.get('is_plural')
                    or str(_vrai_subj_ya2.get('surface', '')).endswith('s')):
                if not _es2_bm.endswith('w'):
                    _es2_bm += 'w'
            m['S'] = ''
            m['O'] = _es2_bm
            processed_indices.add(root_tok['orig_index'])
            processed_indices.add(_vrai_subj_ya2['orig_index'])
            for _expl in T:
                if _expl.get('dep') in ('expl:subj', 'expl:comp'):
                    processed_indices.add(_expl['orig_index'])
            for _dt in T:
                if _dt.get('dep') in ('det', 'fixed') and                    _dt.get('head_index') == _vrai_subj_ya2['orig_index']:
                    processed_indices.add(_dt['orig_index'])

    _avoir_possession = (
        root_tok
        and root_tok.get('semantic_class') == 'having'
        and root_tok.get('pos') in ('VERB', 'AUX')
        and root_tok.get('dep') == 'ROOT'
        and not any(x.get('dep') in ('expl:comp', 'expl:subj') for x in T)
        and any(x.get('dep') == 'obj' for x in T)
        # Pas de xcomp → sinon c'est un verbe d'action (vouloir, devoir...)
        and not any(x.get('dep') == 'xcomp' for x in T)
    )
    if _avoir_possession:
        _obj_poss = next((x for x in T
                          if x.get('dep') == 'obj'
                          and x.get('pos') in ('NOUN', 'PROPN')
                          and x.get('bm')), None)
        _subj_poss = next((x for x in T
                           if x.get('dep') in ('nsubj', 'nsubj:pass')
                           and x.get('bm')), None)
        if _obj_poss and _subj_poss:
            _obj_sc = _obj_poss.get('semantic_class', '')
            _poss_type = ('material'
                          if _obj_sc in ('object', 'money', 'vehicle', 'tool', 'food')
                          else 'abstract')
            tree['clause_type'] = 'noun_phrase_have'
            tree['possession_type'] = _poss_type
            _obj_bm = _obj_poss.get('bm')
            if (_obj_poss.get('is_plural')
                    or str(_obj_poss.get('surface', '')).endswith('s')):
                if not _obj_bm.endswith('w'):
                    _obj_bm += 'w'
            _obj_conjs = [x for x in T
                          if x.get('dep') == 'conj'
                          and x.get('head_index') == _obj_poss['orig_index']
                          and x.get('bm')]
            for _oc in _obj_conjs:
                _cc_oc = next((x for x in T
                               if x.get('dep') == 'cc'
                               and x.get('head_index') == _obj_poss['orig_index']), None)
                _cc_bm_oc = _cc_oc.get('bm', '') if _cc_oc and _cc_oc.get('bm') else ''
                _obj_bm = j(_obj_bm, _cc_bm_oc, _oc.get('bm'))
                processed_indices.add(_oc['orig_index'])
                if _cc_oc: processed_indices.add(_cc_oc['orig_index'])
            m['S'] = _subj_poss.get('bm')
            m['O'] = _obj_bm
            m['V'] = ''
            processed_indices.add(_obj_poss['orig_index'])
            processed_indices.add(_subj_poss['orig_index'])
            processed_indices.add(root_tok['orig_index'])
            for _dt in T:
                if _dt.get('dep') in ('det', 'fixed') and                    _dt.get('head_index') == _obj_poss['orig_index']:
                    processed_indices.add(_dt['orig_index'])

    # ── ATTRIBUT NÉGATIF ROOT ─────────────────────────────────────────────────
    # Token ROOT avec role='negation' → c'est l'attribut, pas le sujet
    # 'il n'est rien' → rien(ROOT, role=negation) = attribut
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
        tree['clause_type'] = 'equative'
        tree['tam'] = 'tɛ' if tree.get('neg') else G_kg.get('equative_marker', 'yé')
    
    # Expletif pronominal avec copule → récupérer comme sujet
    # 'il n'est pas professeur' → il(expl:subj, pronoun) → S='a'
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

    # ── ÉTAPE 1 : SUJET ───────────────────────────────────────────────────────
    subj_tok = next((x for x in T if x.get('dep') in ('nsubj', 'nsubj:pass')), None)

    _expl_subj_tok = next((x for x in T if x.get('dep') == 'expl:subj'), None)
    _expl_comp_tok = next((x for x in T if x.get('dep') == 'expl:comp'), None)

    # avoir ROOT avec expl:comp → existential (il y a)
    _avoir_root = (root_tok
                   and _is_avoir(root_tok)
                   and root_tok.get('pos') in ('VERB', 'AUX'))

    if not _expl_subj_tok and _expl_comp_tok and _avoir_root:
        _vrai_subj_ya = next((x for x in T
                              if x.get('dep') in ('obj', 'nsubj')
                              and x.get('pos') in ('NOUN', 'PROPN', 'PRON')), None)
        if _vrai_subj_ya:
            subj_tok = None
            _has_loc_obl = any(
                x.get('dep') in ('obl', 'obl:mod', 'obl:arg')
                and any(p.get('dep') == 'case' and p.get('role') == 'locative'
                        for p in T if p.get('head_index') == x['orig_index'])
                for x in T)
            tree['clause_type'] = ('existential_localized'
                                   if _has_loc_obl else 'existential_absolute')
            processed_indices.add(_expl_comp_tok['orig_index'])
            processed_indices.add(root_tok['orig_index'])
            _es_bm = _vrai_subj_ya.get('bm') or f"[{_vrai_subj_ya.get('lemma')}]"
            _es_amod = next((x for x in T if x.get('dep') == 'amod'
                             and x.get('head_index') == _vrai_subj_ya['orig_index']), None)
            if _es_amod:
                _es_bm = j(_es_bm, _es_amod.get('bm') or f"[{_es_amod.get('lemma')}]")
                processed_indices.add(_es_amod['orig_index'])
            m['S'] = _es_bm
            processed_indices.add(_vrai_subj_ya['orig_index'])

    if _expl_subj_tok and not _expl_comp_tok:
        _vrai_subj = next((x for x in T
                           if x.get('dep') == 'obj'
                           and x.get('pos') in ('NOUN', 'PROPN')), None)
        if _vrai_subj:
            subj_tok = None
            _has_loc_obl_exist = any(
                x.get('dep') in ('obl', 'obl:mod', 'obl:arg')
                and any(p.get('dep') == 'case' and p.get('role') == 'locative'
                        for p in T if p.get('head_index') == x['orig_index'])
                for x in T)
            _has_acl_exist = any(x.get('dep') == 'acl'
                                 and x.get('head_index') == _vrai_subj['orig_index']
                                 for x in T)
            if _has_acl_exist:
                tree['clause_type'] = 'existential_nominal'
                for _t in T:
                    if _t['orig_index'] == _vrai_subj['orig_index']:
                        _t['dep']     = 'ROOT'
                        _t['is_root'] = True
                root_noun = _vrai_subj
                has_acl   = True
            else:
                tree['clause_type'] = ('existential_localized'
                                       if _has_loc_obl_exist else 'existential_absolute')
                _es_bm = _vrai_subj.get('bm') or f"[{_vrai_subj.get('lemma')}]"
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
            (x for x in T
             if x.get('pos') == 'PRON'
             and x.get('role') not in _expletive_roles | _clitic_roles
             and str(x.get('surface', '')).strip() not in ('?', '.', '-')),
            None)
        if vrai_pron_sujet:
            subj_tok = vrai_pron_sujet

    if clause_type_init == 'content_question' and subj_tok and subj_tok.get('pos') == 'NOUN':
        vrai_pron = next(
            (x for x in T
             if x.get('pos') == 'PRON'
             and x.get('role') not in _expletive_roles | _clitic_roles
             and str(x.get('surface', '')).strip() not in ('?', '.', '-')),
            None)
        if vrai_pron:
            subj_tok = vrai_pron

    # Nettoyage tirets d'inversion
    if subj_tok and str(subj_tok.get('surface', '')).strip() == '-':
        subj_tok = next((x for x in T
                         if x.get('pos') == 'PRON'
                         and x.get('role') not in _expletive_roles | _clitic_roles
                         and x != subj_tok), None)

    # Expletif → chercher PRON ROOT
    if subj_tok and subj_tok.get('role') in _expletive_roles:
        _root_pron = next((x for x in T
                           if x.get('pos') == 'PRON'
                           and x.get('dep') == 'ROOT'), None)
        if _root_pron:
            subj_tok = _root_pron
        # sinon garder subj_tok comme vrai sujet sémantique

    # Récupérer PRON expl:subj si subj_tok perdu
    if not subj_tok and not m.get('S'):
        _expl_subj_pron = next((x for x in T
                                if x.get('dep') == 'expl:subj'
                                and x.get('pos') == 'PRON'
                                and x.get('bm')), None)
        if _expl_subj_pron:
            m['S'] = _expl_subj_pron.get('bm')
            processed_indices.add(_expl_subj_pron['orig_index'])

    if not subj_tok and root_tok and root_tok.get('pos') in ('PRON', 'NOUN'):
        _root_has_nmod = any(x.get('dep') == 'nmod'
                             and x.get('head_index') == root_tok['orig_index']
                             for x in T)
        _root_has_amod = any(x.get('dep') in ('amod', 'conj')
                             and x.get('head_index') == root_tok['orig_index']
                             for x in T)
        if not _root_has_nmod and (not _root_has_amod
                                   or root_tok.get('pos') not in ('NOUN', 'PROPN')):
            subj_tok = root_tok

    if not subj_tok and root_tok and root_tok.get('pos') not in ('NOUN', 'PROPN'):
        subj_tok = next((x for x in T
                         if x.get('pos') in ('PRON', 'NOUN')
                         and x.get('head_index') == root_tok['orig_index']
                         and x != root_tok), None)

    root_noun = next((x for x in T
                      if x.get('pos') == 'NOUN' and x.get('dep') == 'ROOT'), None)
    has_acl = any(x.get('dep') == 'acl' for x in T)

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
            s_chunk = get_bounded_chunk_tokens(subj_tok['orig_index'])
            s_chunk = [t for t in s_chunk
                       if t.get('dep') not in ('case', 'det')
                       and t.get('pos') not in ('PUNCT', 'SYM')
                       and not (t.get('is_root') and t != subj_tok)]

            nmod_s = next((x for x in s_chunk
                           if x.get('dep') == 'nmod'
                           and x.get('head_index') == subj_tok['orig_index']), None)
            if nmod_s:
                def _build_subj_chain(tok, all_toks):
                    nmod = next((t for t in all_toks
                                 if t.get('dep') == 'nmod'
                                 and t.get('head_index') == tok['orig_index']), None)
                    amod = next((t for t in all_toks
                                 if t.get('dep') == 'amod'
                                 and t.get('head_index') == tok['orig_index']
                                 and t.get('bm')), None)
                    tok_bm = tok.get('bm') or tok.get('surface') or f"[{tok.get('lemma')}]"
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
                            nmod_bm = _build_subj_chain(nmod, all_toks)
                            return j(nmod_bm, 'ka', tok_bm)
                    return tok_bm
                m['S'] = _build_subj_chain(subj_tok, T)
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
                    subj_bm = j(subj_bm, _sa.get('bm') or f"[{_sa.get('lemma')}]")
                    processed_indices.add(_sa['orig_index'])

                _s_amod_indices = {a['orig_index'] for a in _s_amods} | {subj_tok['orig_index']}
                _s_conjs = [x for x in T
                            if x.get('dep') == 'conj'
                            and x.get('head_index') in _s_amod_indices]
                for _sc in _s_conjs:
                    _sc_cc = next((x for x in T
                                   if x.get('dep') == 'cc'
                                   and x.get('head_index') == subj_tok['orig_index']), None)
                    _cc_bm = _sc_cc.get('bm', '') if _sc_cc and _sc_cc.get('bm') else ''
                    if _sc_cc:
                        processed_indices.add(_sc_cc['orig_index'])
                    subj_bm = j(subj_bm, _cc_bm, _sc.get('bm') or f"[{_sc.get('lemma')}]")
                    processed_indices.add(_sc['orig_index'])

                _subj_demo = next((x for x in T
                                   if x.get('dep') == 'det'
                                   and x.get('role') == 'demonstrative'
                                   and x.get('head_index') == subj_tok['orig_index']), None)
                if _subj_demo:
                    _demo_suf = G_kg.get('demonstrative_suffix', 'in') or 'in'
                    subj_bm   = j('nin', subj_bm, _demo_suf)
                    processed_indices.add(_subj_demo['orig_index'])

                # Pluriel sujet
                if (subj_tok.get('is_plural')
                        and not subj_bm.endswith('w')
                        and subj_tok.get('pos') not in ('PRON', 'PROPN')):
                    subj_bm += 'w'

                m['S'] = subj_bm

            processed_indices.update([t['orig_index'] for t in s_chunk])

    # ── VERROU INTERROGATIF ───────────────────────────────────────────────────
    _phrase_contient_interrogation = any(
        '?' in str(x.get('surface', ''))
        or (x.get('dep') == 'punct' and str(x.get('surface', '')).strip() == '?')
        for x in T)
    print(f"DEBUG _phrase_contient_interrogation={_phrase_contient_interrogation}")

    if _phrase_contient_interrogation:
        # Exclure : expletifs, clitiques, pronoms relatifs/interrogatifs, ponctuation
        # Priorité au PRON nsubj non-interrogatif
        # Priorité au PRON nsubj non-interrogatif comme vrai sujet
        _vrai_pron_sujet = next(
            (x for x in T
             if x.get('pos') == 'PRON'
             and x.get('role') not in (_expletive_roles | _clitic_roles | _relative_roles
                                       | {'interrogative'})
             and 'Int' not in str(x.get('morph', ''))
             and x.get('dep') in ('nsubj', 'nsubj:pass', 'dep')
             and str(x.get('surface', '')).strip() not in ('?', '.', '-')
             and str(x.get('surface', '')).isalnum()),
            None)
        if not _vrai_pron_sujet:
            _vrai_pron_sujet = next(
                (x for x in T
                 if x.get('pos') == 'PRON'
                 and x.get('role') not in (_expletive_roles | _clitic_roles | _relative_roles
                                           | {'interrogative'})
                 and 'Int' not in str(x.get('morph', ''))
                 and x.get('dep') not in ('obj',)
                 and str(x.get('surface', '')).strip() not in ('?', '.', '-')
                 and str(x.get('surface', '')).isalnum()),
                None)
        # Fallback : tout PRON non-interrogatif non-objet
        if not _vrai_pron_sujet:
            _vrai_pron_sujet = next(
                (x for x in T
                 if x.get('pos') == 'PRON'
                 and x.get('role') not in (_expletive_roles | _clitic_roles | _relative_roles
                                           | {'interrogative'})
                 and 'Int' not in str(x.get('morph', ''))
                 and x.get('dep') not in ('obj',)
                 and str(x.get('surface', '')).strip() not in ('?', '.', '-')
                 and str(x.get('surface', '')).isalnum()),
                None)

        if _vrai_pron_sujet:
            print(f"DEBUG _vrai_pron_sujet={_vrai_pron_sujet}")
            print(f"DEBUG m_O_avant={m.get('O')}, clause_type_init={clause_type_init}")
            m['S'] = _vrai_pron_sujet.get('bm') or f"[{_vrai_pron_sujet.get('lemma')}]"
            if subj_tok and subj_tok['orig_index'] in processed_indices:
                processed_indices.remove(subj_tok['orig_index'])
            processed_indices.add(_vrai_pron_sujet['orig_index'])
            # Capturer le mot interrogatif comme O si pas déjà fait
            if not m.get('O') and clause_type_init == 'content_question':
                _interrog_pron = next((x for x in T
                                       if x.get('role') == 'interrogative'
                                       and x['orig_index'] not in processed_indices), None)
                if _interrog_pron:
                    _ip_bm = (_interrog_pron.get('bm')
                              or G_kg.get('interrogative_who', 'jɔn'))
                    m['O'] = _ip_bm
                    processed_indices.add(_interrog_pron['orig_index'])
        
        else:
            if subj_tok and subj_tok.get('pos') == 'NOUN':
                _poss_det = next((x for x in T
                                  if x.get('dep') == 'det'
                                  and x.get('role') in ('pronoun', 'possessive')
                                  and x.get('head_index') == subj_tok['orig_index']), None)
                subj_bm = subj_tok.get('bm') or subj_tok.get('surface', '')
                if _poss_det:
                    poss_val = _poss_det.get('bm') or f"[{_poss_det.get('lemma')}]"
                    _gen     = G_kg.get('genitive_marker', 'ka') or 'ka'
                    m['S']   = j(poss_val, subj_bm) if poss_val == 'n' else j(poss_val, _gen, subj_bm)
                    processed_indices.add(_poss_det['orig_index'])
                else:
                    if subj_tok.get('bm_suffix'):
                        subj_bm += subj_tok['bm_suffix']
                    if (subj_tok.get('is_plural')
                            and not subj_bm.endswith('w')
                            and subj_tok.get('pos') not in ('PRON', 'PROPN')):
                        subj_bm += 'w'
                    m['S'] = subj_bm
                processed_indices.add(subj_tok['orig_index'])

    # ── CLAUSE RELATIVE SUR LE SUJET ─────────────────────────────────────────
    _relcl_on_subj = next((x for x in T
                           if x.get('dep') == 'acl:relcl'
                           and subj_tok
                           and x.get('head_index') == subj_tok['orig_index']), None)
    if _relcl_on_subj:
        _rel_subj = next((x for x in T
                          if x.get('dep') == 'nsubj'
                          and x.get('head_index') == _relcl_on_subj['orig_index']), None)
        _rel_adv  = next((x for x in T
                          if x.get('dep') == 'advmod'
                          and x.get('head_index') == _relcl_on_subj['orig_index']), None)
        _rel_aux  = next((x for x in T
                          if x.get('dep') in ('aux', 'aux:tense')
                          and x.get('head_index') == _relcl_on_subj['orig_index']), None)

        _subj_bm    = subj_tok.get('bm') or subj_tok.get('surface', '')
        _rel_s_bm   = _rel_subj.get('bm', '') if _rel_subj else ''
        _rel_v_bm   = _relcl_on_subj.get('bm', '')
        _rel_obj_bm = next((x.get('bm', '') for x in T
                            if x.get('dep') == 'obj'
                            and x.get('head_index') == _relcl_on_subj['orig_index']
                            and x.get('bm')), '')
        _rel_adv_bm = _rel_adv.get('bm', '') if _rel_adv else ''
        _rel_marker = G_kg.get('relative_marker', 'mìn') or 'mìn'
        _rel_tam    = 'yé'  # passé composé de la relative

        # Éviter doublon TAM == V
        _rel_v_display = '' if _rel_v_bm == _rel_tam else _rel_v_bm

        # Structure bambara : tête mìn TAM O V ADV
        # Le sujet de la relative est implicite en bambara (pas de jɔn/qui)
        # 'Une femme qui a eu un enfant' → númanfɛlaka mìn yé dén [V]
        # Exclure _rel_s_bm si c'est un pronom relatif (qui/que/which)
        # Exclure le pronom relatif (qui/que) mais garder le vrai sujet (tu/il/elle)
        _rel_s_is_relative = (_rel_subj and
                              _rel_subj.get('role') == 'relative'
                              and _rel_subj.get('dep') in ('nsubj', 'obj')
                              and _rel_subj.get('head_index') == _relcl_on_subj['orig_index']
                              and str(_rel_subj.get('surface', '')).lower()
                              in {'qui', 'que', 'which', 'whom'})
        _rel_s_display = '' if _rel_s_is_relative else _rel_s_bm
        _topic_str = j(_rel_s_display, _rel_tam, _subj_bm, _rel_marker,
                       _rel_v_display, _rel_adv_bm)
        m['S'] = _topic_str
        tree['clause_type'] = 'relative_topic'

        processed_indices.add(_relcl_on_subj['orig_index'])
        if _rel_subj: processed_indices.add(_rel_subj['orig_index'])
        if _rel_adv:  processed_indices.add(_rel_adv['orig_index'])
        if _rel_aux:  processed_indices.add(_rel_aux['orig_index'])
        _rel_obj = next((x for x in T
                         if x.get('dep') == 'obj'
                         and x.get('head_index') == _relcl_on_subj['orig_index']), None)
        if _rel_obj:
            processed_indices.add(_rel_obj['orig_index'])

    # ── ÉTAPE 2 : VERBE ROOT ─────────────────────────────────────────────────
    if root_tok and root_tok.get('pos') in ('VERB', 'AUX'):
        # Copule ou avoir → ne pas mettre dans V
        _root_is_copula = (
            _is_copula(root_tok)
            or (
                _is_avoir(root_tok)
                and root_tok.get('dep') in ('aux', 'aux:tense', 'aux:pass', 'cop')
            )
        )
        if not _root_is_copula:
            root_bm = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
            if root_tok.get('bm_suffix'):
                root_bm += root_tok['bm_suffix']
            m['V'] = root_bm
            processed_indices.add(root_tok['orig_index'])
            # Capturer les advmod du verbe ROOT (bien, vraiment...)
            _v_advmods = [x for x in T
                          if x.get('dep') == 'advmod'
                          and x.get('head_index') == root_tok['orig_index']
                          and x.get('bm')
                          and x['orig_index'] not in processed_indices
                          and str(x.get('surface', '')).lower().rstrip("'").rstrip('\u2019')
                          not in _neg_surfaces
                          and x.get('role') not in ('temporal', 'temporal_already')
                          and x.get('pos') != 'ADV']
            for _va in _v_advmods:
                _va_sub = [x for x in T
                           if x.get('dep') == 'advmod'
                           and x.get('head_index') == _va['orig_index']
                           and x.get('bm')
                           and x['orig_index'] not in processed_indices
                           and str(x.get('surface', '')).lower().rstrip("'").rstrip('\u2019')
                           not in _neg_surfaces]
                _va_bm = _va.get('bm')
                for _vs in _va_sub:
                    _va_bm = j(_va_bm, _vs.get('bm'))
                    processed_indices.add(_vs['orig_index'])
                m['V'] = j(m['V'], _va_bm)
                processed_indices.add(_va['orig_index'])

    elif root_tok and root_tok.get('pos') in ('NOUN', 'PROPN'):
        pass  # traité à l'étape 3

    if xcomp_verb_tok:
        m['V_ACTION'] = xcomp_verb_tok.get('bm') or f"[{xcomp_verb_tok.get('lemma')}]"
        processed_indices.add(xcomp_verb_tok['orig_index'])
        # Objet du xcomp (son enfant → a den)
        _xcomp_obj = next((x for x in T
                           if x.get('dep') == 'obj'
                           and x.get('head_index') == xcomp_verb_tok['orig_index']
                           and x.get('pos') in ('NOUN', 'PROPN')), None)
        if _xcomp_obj and _xcomp_obj['orig_index'] not in processed_indices:
            _xcomp_obj_bm = _xcomp_obj.get('bm') or f"[{_xcomp_obj.get('lemma')}]"
            _xcomp_poss   = next((x for x in T
                                  if x.get('dep') == 'det'
                                  and x.get('role') in ('pronoun', 'possessive')
                                  and x.get('head_index') == _xcomp_obj['orig_index']), None)
            if _xcomp_poss:
                _poss_bm      = _xcomp_poss.get('bm', '')
                _xcomp_obj_bm = (j('n', _xcomp_obj_bm) if _poss_bm == 'n'
                                 else j(_poss_bm, _xcomp_obj_bm))
                processed_indices.add(_xcomp_poss['orig_index'])
            m['O'] = _xcomp_obj_bm
            processed_indices.add(_xcomp_obj['orig_index'])

    # Négation
    _has_neg_adv = any(
        str(x.get('surface', '')).lower().rstrip("'").rstrip('\u2019') in _neg_surfaces
        and x.get('dep') in ('advmod', 'fixed', 'mark')
        for x in T)
    if _has_neg_adv:
        tree['neg'] = True
    for _nt in T:
        surf = str(_nt.get('surface', '')).lower().rstrip("'").rstrip('\u2019')
        if surf in _neg_surfaces and _nt.get('dep') in ('advmod', 'fixed', 'mark'):
            processed_indices.add(_nt['orig_index'])


    # Prohibitif : verbe négatif sans sujet → kàna V
    _is_prohibitive = (root_tok
                       and root_tok.get('pos') == 'VERB'
                       and not m.get('S')
                       and tree.get('neg')
                       and not any(x.get('dep') in ('nsubj', 'nsubj:pass') for x in T))
    if _is_prohibitive:
        tree['clause_type'] = 'prohibitive'
        tree['neg'] = False

    

    # Impératif affirmatif : verbe sans sujet, sans négation
    _has_excl = any(str(x.get('surface', '')).strip() == '!' for x in T)
    _is_imperative_affirm = (root_tok
                             and root_tok.get('pos') == 'VERB'
                             and not m.get('S')
                             and not tree.get('neg')
                             and _has_excl
                             and not any(x.get('dep') in ('nsubj', 'nsubj:pass') for x in T))
    if _is_imperative_affirm:
        tree['clause_type'] = 'imperative'
        tree['tam'] = ''

    # Infinitif sans sujet
    _is_infinitive = (root_tok
                      and root_tok.get('pos') == 'VERB'
                      and 'Inf' in str(root_tok.get('morph', ''))
                      and not m.get('S'))
    if _is_infinitive:
        tree['clause_type'] = 'infinitive'

    # ── F1 : TENSE TRANSFER ───────────────────────────────────────────────────
    aux_tense_tok = next((x for x in T
                          if x.get('dep') in ('aux', 'aux:tense', 'aux:pass', 'cop')
                          and x.get('tense') in ('past', 'hab', 'plup', 'imp')), None)
    print(f"DEBUG aux_tense_tok={aux_tense_tok}")
    print(f"DEBUG T_tenses={[(x.get('surface'), x.get('dep'), x.get('tense')) for x in T]}")
    if not aux_tense_tok and root_tok and root_tok.get('tense') == 'past':
        _aux_pass = next((x for x in T if x.get('dep') == 'aux:pass'), None)
        if _aux_pass and not root_tok.get('is_statif'):
            aux_tense_tok = root_tok

    if aux_tense_tok and root_tok:
        tree['tense'] = aux_tense_tok.get('tense', 'pres')
        tree['neg']   = tree['neg'] or aux_tense_tok.get('is_neg', False)
        tree['tam']   = _resolve_tam(tree['tense'], tree['neg'], G_kg)
    elif (root_tok and root_tok.get('tense') in ('past', 'fut', 'cond')
          and not root_tok.get('is_statif')):
        tree['tense'] = root_tok['tense']
        tree['neg']   = tree['neg'] or root_tok.get('is_neg', False)
        tree['tam']   = _resolve_tam(tree['tense'], tree['neg'], G_kg)
    else:
        copula_tok_f1 = next((x for x in T if x.get('dep') == 'cop'), None)
        aux_tok_f1    = next((x for x in T
                              if x.get('dep') in ('aux', 'aux:tense')
                              and x.get('role') == 'auxiliary'), None)
        if aux_tok_f1 and not aux_tense_tok:
            tree['tense'] = aux_tok_f1.get('tense', 'pres')
            tree['tam']   = _resolve_tam(tree['tense'], tree['neg'], G_kg)
        elif copula_tok_f1:
            if tree.get('clause_type') == 'locative':
                tree['tam'] = _resolve_tam('pres', tree.get('neg', False), G_kg) or 'bɛ'
        else:
            tree['tam'] = (G_kg.get('tam_default', '')
                           or _resolve_tam('pres', tree['neg'], G_kg))

    # Déjà
    _deja_tok = next((x for x in T
                      if x.get('role') == 'temporal_already'), None)
    if _deja_tok:
        _deja_modifies_state = (root_tok and root_tok.get('pos') in ('NOUN', 'ADJ'))
        if tree.get('neg'):
            tree['already_marker'] = _deja_tok.get('bm_neg', 'fɔ́lɔ') or 'fɔ́lɔ'
        elif tree.get('tense') == 'past':
            tree['already_marker'] = _deja_tok.get('bm', 'kàban') or 'kàban'
        elif _deja_modifies_state:
            tree['already_marker'] = 'kàban'
        else:
            tree['already_marker'] = 'fɔ́lɔ'
        tree['already_position'] = ('HEAD'
                                    if _deja_tok.get('orig_index', 99) == 0
                                    and not _deja_modifies_state
                                    else 'END')
        processed_indices.add(_deja_tok['orig_index'])

    # Participial_to
    part_to_tok = next((x for x in T
                        if x.get('role') == 'participial_to' and x.get('bm')), None)
    if part_to_tok:
        s_bm = (next((x.get('bm') for x in T if x.get('dep') == 'nsubj'), None)
                or next((x.get('bm') for x in T if x.get('role') == 'pronoun'), None)
                or '')
        m['ADV'] = j(s_bm, part_to_tok.get('bm', '') + 'tɔ')
        processed_indices.add(part_to_tok['orig_index'])

    # Prédicat privatif NOUN ROOT
    _priv_case = next((x for x in T
                       if x.get('dep') == 'case'
                       and x.get('role') == 'privative'
                       and root_tok
                       and x.get('head_index') == root_tok['orig_index']
                       and root_tok.get('pos') == 'NOUN'), None)
    if _priv_case:
        tree['clause_type'] = 'privative_pred'
        _priv_marker = _priv_case.get('bm_marker', 'tan')
        _priv_type   = _priv_case.get('privative_type', 'suffix')
        _root_bm     = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
        m['O'] = _root_bm + _priv_marker if _priv_type == 'suffix' else j(_root_bm, _priv_marker)
        processed_indices.add(root_tok['orig_index'])
        processed_indices.add(_priv_case['orig_index'])

    # Prédicat privatif PRON ROOT
    _priv_on_root_pron = next((x for x in T
                               if x.get('dep') == 'case'
                               and x.get('role') == 'privative'
                               and root_tok
                               and x.get('head_index') == root_tok['orig_index']
                               and root_tok.get('pos') in ('PRON', 'NOUN')), None)
    if _priv_on_root_pron and root_tok and not _priv_case:
        tree['clause_type'] = 'privative_pred'
        _priv_marker = (G_kg.get('privative_pron_marker', 'kɔ')
                        if root_tok.get('pos') == 'PRON'
                        else _priv_on_root_pron.get('bm_marker', 'tan'))
        _root_bm = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
        m['O'] = j(_root_bm, _priv_marker)
        m['S'] = ''
        processed_indices.add(root_tok['orig_index'])
        processed_indices.add(_priv_on_root_pron['orig_index'])

    # ADJ/NOUN ROOT avec copule → O
    # Sauf si un case locatif est présent → cas 8 (localisation)
    _root_has_nmod_chain = (any(x.get('dep') == 'nmod'
                                and x.get('head_index') == root_tok['orig_index']
                                for x in T) if root_tok else False)
    _root_has_loc_case = (root_tok and any(
        x.get('dep') == 'case' and x.get('role') == 'locative'
        and x.get('head_index') == root_tok['orig_index']
        for x in T))
    
    print(f"DEBUG _root_has_loc_case={_root_has_loc_case}")
    print(f"DEBUG case_tokens={[(x.get('surface'), x.get('role')) for x in T if x.get('dep') == 'case']}")
    if (root_tok
            and root_tok.get('pos') in ('ADJ', 'NOUN')
            and any(x.get('dep') == 'cop' for x in T)
            and root_tok['orig_index'] not in processed_indices
            and not _root_has_nmod_chain
            and not _root_has_loc_case
            and not root_tok.get('is_participe_passe')
            and not root_tok.get('is_statif')):
        _root_bm = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
        _poss_on_root = next((x for x in T
                              if x.get('dep') == 'det'
                              and x.get('role') in ('pronoun', 'possessive')
                              and x.get('head_index') == root_tok['orig_index']
                              and x.get('bm')), None)
        if _poss_on_root:
            _poss_bm = _poss_on_root.get('bm', '')
            _root_bm = j('n', _root_bm) if _poss_bm == 'n' else j(_poss_bm, _root_bm)
            processed_indices.add(_poss_on_root['orig_index'])
        m['O'] = _root_bm
        processed_indices.add(root_tok['orig_index'])
    elif _root_has_loc_case and root_tok and root_tok['orig_index'] not in processed_indices:
        # Cas 8 : localisation → ex: "Je suis en route" → n bɛ síraden la
        tree['clause_type'] = 'locative'
        m['S'] = next((x.get('bm','') for x in T
                       if x.get('dep') in ('nsubj', 'nsubj:pass')
                       and x.get('bm')), m.get('S',''))
        _loc_case_root = next((x for x in T
                               if x.get('dep') == 'case'
                               and x.get('role') == 'locative'
                               and x.get('head_index') == root_tok['orig_index']), None)
        _loc_marker = _loc_case_root.get('bm_marker', 'la') if _loc_case_root else 'la'
        if not _loc_marker:
            _loc_marker = 'la'
        m['OBL_ALL'].append({
            'HEAD': root_tok.get('bm') or f"[{root_tok.get('lemma')}]",
            'MARKER': _loc_marker,
            'local_clause_type': 'locative',
            'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
            'DEP_TYPE': 'case', 'COMPOUND_IS_QUANTIFIER': False,
            'MARKER_IS_PREFIX': False,
        })
        processed_indices.add(root_tok['orig_index'])
        if _loc_case_root:
            processed_indices.add(_loc_case_root['orig_index'])

    # ── ÉTAPE 3 : OBJET ───────────────────────────────────────────────────────
    if root_noun and has_acl:
        if tree.get('clause_type') != 'existential_nominal':
            tree['clause_type'] = 'noun_phrase'
        r_idx     = root_noun['orig_index']
        child_adj = next((x for x in T
                          if x.get('dep') == 'amod'
                          and x.get('head_index') == r_idx), None)
        acl_tok   = next((x for x in T
                          if x.get('dep') == 'acl'
                          and x.get('head_index') == r_idx), None)
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
        if xcomp_adj_tok:
            m['O'] = xcomp_adj_tok.get('bm') or f"[{xcomp_adj_tok.get('lemma')}]"
            _xcomp_advs = [x for x in T
                           if x.get('dep') == 'advmod'
                           and x.get('head_index') == xcomp_adj_tok['orig_index']
                           and x.get('bm')]
            if _xcomp_advs:
                m['O'] = j(m['O'], j(*[x.get('bm') for x in _xcomp_advs]))
                for _xa in _xcomp_advs:
                    processed_indices.add(_xa['orig_index'])
            m['O_IS_XCOMP'] = True
            processed_indices.add(xcomp_adj_tok['orig_index'])
        else:
            obj_tok = next((x for x in T
                            if x.get('dep') in ('obj', 'xcomp')
                            and x.get('pos') in ('NOUN', 'PROPN', 'ADJ')), None)
            if not obj_tok:
                obj_tok = next((x for x in T
                                if x.get('dep') in ('obj', 'advmod', 'dep')
                                and x.get('role') == 'interrogative'
                                and x.get('bm')), None)
            _phrase_interrogative_brute = any(
                '?' in str(x.get('surface', ''))
                or (x.get('dep') == 'punct'
                    and str(x.get('surface', '')).strip() == '?')
                for x in T)
            if not obj_tok and _phrase_interrogative_brute:
                obj_tok = next((x for x in T
                                if x.get('dep') == 'dep'
                                and x.get('pos') == 'PROPN'
                                and x.get('bm')), None)
            if obj_tok and obj_tok.get('role') == 'interrogative':
                _interrog_noun = next((x for x in T
                                       if x.get('dep') in ('obl:arg', 'nmod')
                                       and x.get('head_index') == obj_tok['orig_index']
                                       and x.get('pos') in ('NOUN', 'PROPN')), None)
                if not _interrog_noun:
                    _case_d = next((x for x in T
                                    if x.get('dep') == 'case'
                                    and x.get('head_index') == obj_tok['orig_index']), None)
                    if _case_d:
                        _interrog_noun = next((x for x in T
                                               if x.get('dep') == 'obl:arg'
                                               and x.get('pos') in ('NOUN', 'PROPN')), None)
                if _interrog_noun:
                    tree['interrog_noun'] = _interrog_noun
                    processed_indices.add(_interrog_noun['orig_index'])

            if not obj_tok and clause_type_init == 'content_question':
                obj_tok = next((x for x in T
                                if x.get('pos') == 'NOUN'
                                and any(d.get('role') == 'interrogative'
                                        for d in T
                                        if d.get('head_index') == x['orig_index'])), None)

            if not obj_tok and _phrase_interrogative_brute:
                obj_tok = next((x for x in T if x.get('pos') == 'NOUN'), None)

            if not obj_tok and root_tok and root_tok.get('pos') in ('NOUN', 'PROPN') and not has_acl:
                obj_tok = root_tok
                _has_interrog_det = any(
                    x.get('role') == 'interrogative'
                    and x.get('head_index') == root_tok['orig_index']
                    for x in T)
                if _has_interrog_det and _has_question_mark:
                    tree['clause_type'] = 'content_question'
                    clause_type_init    = 'content_question'
                elif tree.get('clause_type') not in ('existential_nominal', 'locative'):
                    tree['clause_type'] = 'noun_phrase'

            if obj_tok and obj_tok['orig_index'] not in processed_indices:
                o_chunk = get_bounded_chunk_tokens(obj_tok['orig_index'])
                print(f"DEBUG o_chunk={[(t.get('surface'), t.get('dep'), t.get('orig_index')) for t in o_chunk]}")

                if tree.get('clause_type') == 'noun_phrase':
                    _has_any_nmod = any(x.get('dep') == 'nmod' for x in o_chunk)
                    o_chunk = [x for x in o_chunk if x.get('dep') != 'obl:mod']
                    if not _has_any_nmod:
                        o_chunk = [x for x in o_chunk
                                   if x.get('dep') not in ('nmod', 'obl', 'case')]

                o_chunk = sorted([x for x in o_chunk
                                  if x.get('pos') not in ('PUNCT', 'SYM')],
                                 key=lambda x: x['orig_index'])

                if obj_tok.get('dep') == 'xcomp':
                    m['O_IS_XCOMP'] = True

                tete_tok = next((x for x in o_chunk
                                 if x == obj_tok or x.get('dep') == 'ROOT'), obj_tok)

                _has_nmod_chain = any(x.get('dep') == 'nmod' for x in o_chunk)
                objet_elements  = []

                if _has_nmod_chain and tete_tok:
                    def _build_genitive_chain(tok, all_toks):
                        nmod  = next((t for t in all_toks
                                      if t.get('dep') == 'nmod'
                                      and t.get('head_index') == tok['orig_index']), None)
                        amods = [t for t in all_toks
                                 if t.get('dep') == 'amod'
                                 and t.get('head_index') == tok['orig_index']
                                 and t.get('bm')]
                        tok_bm = tok.get('bm') or tok.get('surface') or f"[{tok.get('lemma')}]"
                        _classifiants = [a for a in amods if a['orig_index'] > tok['orig_index']]
                        _qualifiants  = [a for a in amods if a['orig_index'] < tok['orig_index']]
                        _class_bms    = [a.get('bm', '') for a in _classifiants]
                        _qual_bms     = [a.get('bm', '') for a in _qualifiants]
                        if nmod:
                            nmod_bm = _build_genitive_chain(nmod, all_toks)
                            return j(*_class_bms, nmod_bm, 'ka', tok_bm, *_qual_bms)
                        return j(*_class_bms, tok_bm, *_qual_bms)


                    _resolved = _build_genitive_chain(tete_tok, T)
                    # Capturer les conj de la tête (et l'education)
                    _top_conjs = [x for x in o_chunk
                                  if x.get('dep') == 'conj'
                                  and x.get('head_index') == tete_tok['orig_index']]
                    print(f"DEBUG _top_conjs={[(c['surface'], c.get('bm'), c['orig_index']) for c in _top_conjs]}")

                    for _tc in _top_conjs:
                        _tc_bm = _tc.get('bm') or f"[{_tc.get('lemma')}]"
                        _tc_cc = next((x for x in T
                                       if x.get('dep') == 'cc'
                                       and x.get('head_index') == tete_tok['orig_index']), None)
                        _cc_bm_tc = _tc_cc.get('bm', '') if _tc_cc and _tc_cc.get('bm') else ''
                        _resolved = j(_resolved, _cc_bm_tc, _tc_bm)
                        processed_indices.add(_tc['orig_index'])
                        if _tc_cc:
                            processed_indices.add(_tc_cc['orig_index'])
                    objet_elements = [_resolved] if _resolved else []

                elif tete_tok:
                    _det_int = next((x for x in o_chunk
                                     if x.get('role') == 'interrogative'), None)
                    if ((clause_type_init == 'content_question'
                            or _phrase_interrogative_brute)
                            and _det_int
                            and tete_tok.get('pos') == 'NOUN'):
                        nom_val        = tete_tok.get('bm') or f"[{tete_tok.get('lemma')}]"
                        int_val        = (_det_int.get('bm')
                                          if _det_int.get('bm')
                                          else G_kg.get('interrogative_default', 'jùmɛn'))
                        objet_elements = [nom_val, int_val]
                    else:
                        tete_bm     = tete_tok.get('bm') or f"[{tete_tok.get('lemma')}]"
                        _global_postpos = []
                        _amods      = [x for x in o_chunk
                                       if x.get('dep') == 'amod'
                                       and x.get('head_index') == tete_tok['orig_index']]
                        _postpos_amods = [x for x in _amods if x.get('role') == 'quantifier']
                        _other_amods   = [x for x in _amods if x not in _postpos_amods]
                        _pre_amods     = [a for a in _other_amods
                                          if a['orig_index'] < tete_tok['orig_index']]
                        _post_amods    = [a for a in _other_amods
                                          if a['orig_index'] > tete_tok['orig_index']]
                        for _a in _post_amods:
                            tete_bm = j(_a.get('bm') or f"[{_a.get('lemma')}]", tete_bm)
                            processed_indices.add(_a['orig_index'])
                        for _a in _pre_amods:
                            tete_bm = j(tete_bm, _a.get('bm') or f"[{_a.get('lemma')}]")
                            processed_indices.add(_a['orig_index'])
                        _global_postpos = _postpos_amods

                        print(f"DEBUG tete_tok={tete_tok['surface']} idx={tete_tok['orig_index']}")
                        print(f"DEBUG _amods={[(a['surface'], a['head_index']) for a in _amods]}")
                        print(f"DEBUG tete_bm_avant_conjs={tete_bm!r}")

                        _amod_indices = {x['orig_index'] for x in _amods} | {tete_tok['orig_index']}
                        _conjs = [x for x in T
                                  if x.get('dep') == 'conj'
                                  and x.get('head_index') in _amod_indices]
                        print(f"DEBUG _conjs={[c['surface'] for c in _conjs]}")

                        for _c in _conjs:
                            _c_bm = _c.get('bm') or f"[{_c.get('lemma')}]"
                            print(f"DEBUG conj={_c['surface']} idx={_c['orig_index']}")
                            _c_relcl_tok = next((x for x in T
                                if x.get('dep') == 'acl:relcl'
                                and x.get('head_index') == _c['orig_index']), None)
                            _c_has_relcl = _c_relcl_tok is not None
                            # Verrouiller l'acl:relcl MAINTENANT avant le post-traitement
                            if _c_has_relcl and _c_relcl_tok:
                                _relcl_pre_desc = (nx.descendants(NX_G, _c_relcl_tok['orig_index'])
                                                   | {_c_relcl_tok['orig_index']})
                                processed_indices.update(_relcl_pre_desc)
                            if not _c_has_relcl:
                                _c_amods   = [x for x in o_chunk
                                              if x.get('dep') == 'amod'
                                              and x.get('head_index') == _c['orig_index']]
                                _c_postpos = [x for x in _c_amods
                                              if x['orig_index'] < _c['orig_index']
                                              and x.get('role') == 'quantifier']
                                _c_others  = [x for x in _c_amods if x not in _c_postpos]
                                _c_pre     = [a for a in _c_others
                                              if a['orig_index'] < _c['orig_index']]
                                _c_post    = [a for a in _c_others
                                              if a['orig_index'] > _c['orig_index']]
                                for _ca in _c_post:
                                    _c_bm = j(_ca.get('bm') or f"[{_ca.get('lemma')}]", _c_bm)
                                    processed_indices.add(_ca['orig_index'])
                                for _ca in _c_pre:
                                    _c_bm = j(_c_bm, _ca.get('bm') or f"[{_ca.get('lemma')}]")
                                    processed_indices.add(_ca['orig_index'])
                                for _ca in _c_postpos:
                                    _c_bm = j(_c_bm, _ca.get('bm') or f"[{_ca.get('lemma')}]")
                                    processed_indices.add(_ca['orig_index'])
                                # Propager le possessif de la tête sur les conj
                                _poss_on_head = next((x for x in T
                                                      if x.get('dep') == 'det'
                                                      and x.get('role') in ('pronoun', 'possessive')
                                                      and x.get('head_index') == tete_tok['orig_index']
                                                      and x.get('bm')), None)
                                if _poss_on_head:
                                    _poss_bm = _poss_on_head.get('bm', '')
                                    _c_bm    = (j('n', _c_bm) if _poss_bm == 'n'
                                                else j(_poss_bm, 'ka', _c_bm))

                            _cc_tok = next((x for x in T
                                            if x.get('dep') == 'cc'
                                            and x.get('head_index') == _c['orig_index']), None)
                            if not _cc_tok:
                                _cc_tok = next((x for x in T
                                                if x.get('dep') == 'cc'
                                                and x.get('head_index') == tete_tok['orig_index']),
                                               None)
                            _conj_marker = (_cc_tok.get('bm', '')
                                            if _cc_tok and _cc_tok.get('bm') else '')
                            if _cc_tok:
                                processed_indices.add(_cc_tok['orig_index'])
                            tete_bm = j(tete_bm, _conj_marker, _c_bm)
                            processed_indices.add(_c['orig_index'])

                        for _a in _global_postpos:
                            tete_bm = j(tete_bm, _a.get('bm') or f"[{_a.get('lemma')}]")
                            processed_indices.add(_a['orig_index'])
                        objet_elements = [tete_bm]

                for _ct in o_chunk:
                    if _ct.get('dep') in ('nmod', 'det', 'case'):
                        processed_indices.add(_ct['orig_index'])

                if not objet_elements:
                    objet_elements = [t.get('bm') or f"[{t.get('lemma')}]"
                                      for t in o_chunk
                                      if t.get('dep') not in ('det', 'case', 'cop',
                                                               'aux', 'aux:tense', 'fixed')
                                      and t.get('pos') not in ('PUNCT', 'SYM')
                                      and str(t.get('surface', '')).strip()
                                      not in ("'", "\u2019", "l", "L")]

                _interrog_det = next((x for x in o_chunk
                                      if x.get('role') == 'interrogative'
                                      and x.get('dep') == 'det'), None)
                if _interrog_det and _interrog_det.get('bm'):
                    if _interrog_det.get('bm') not in objet_elements:
                        objet_elements.append(_interrog_det.get('bm'))
                    processed_indices.add(_interrog_det['orig_index'])

                m['O'] = j(*[x for x in objet_elements if str(x).strip() != 'ni'])

                if ((obj_tok.get('is_plural')
                        or str(obj_tok.get('surface', '')).endswith('s'))
                        and not str(m['O']).endswith('w')
                        and tree.get('clause_type') != 'presentative'):
                    if obj_tok.get('pos') not in ('PRON', 'PROPN'):
                        m['O'] = f"{m['O']}w"
                processed_indices.update([t['orig_index'] for t in o_chunk])

            if tree.get('clause_type') == 'existential_nominal' and obj_tok is not None:
                _all_desc = nx.descendants(NX_G, obj_tok['orig_index']) | {obj_tok['orig_index']}
                _all_desc -= processed_indices
                o_chunk = sorted([NX_G.nodes[idx]['token'] for idx in _all_desc
                                  if idx in NX_G.nodes
                                  and NX_G.nodes[idx]['token'].get('pos') not in ('PUNCT', 'SYM')
                                  and NX_G.nodes[idx]['token'].get('dep') not in ('det',)
                                  and NX_G.nodes[idx]['token'].get('role')
                                  not in ('article', 'pronoun')],
                                 key=lambda x: x['orig_index'])

    # ── POST-TRAITEMENT acl:relcl ─────────────────────────────────────────────
    for _rt in sorted(T, key=lambda x: x['orig_index']):
        if _rt.get('dep') != 'acl:relcl':
            continue
        if _rt['orig_index'] in processed_indices:
            continue
        # Skip si la tête est un conj d'un obl comitative (traité par _com_conjs étape 4)
        _rt_head = next((x for x in T if x['orig_index'] == _rt.get('head_index')), None)
        if (_rt_head and _rt_head.get('dep') == 'conj'
                and any(x.get('dep') == 'case' and x.get('role') == 'comitative'
                        and x.get('head_index') == _rt_head.get('head_index')
                        for x in T)):
            continue
        _rel_subj  = next((x for x in T if x.get('dep') == 'nsubj'
                           and x.get('head_index') == _rt['orig_index']), None)
        _rel_xcomp = next((x for x in T if x.get('dep') == 'xcomp'
                           and x.get('head_index') == _rt['orig_index']), None)
        _rel_obj   = next((x for x in T if x.get('dep') == 'obj'
                           and x.get('head_index') == _rt['orig_index']), None)
        _rel_obj2  = next((x for x in T if x.get('dep') == 'obj'
                           and _rel_xcomp
                           and x.get('head_index') == _rel_xcomp['orig_index']), None)
        _rel_o2_amod = next((x for x in T if x.get('dep') == 'amod'
                             and _rel_obj2
                             and x.get('head_index') == _rel_obj2['orig_index']), None)
        _rel_s_bm  = _rel_subj.get('bm', '') if _rel_subj else ''
        _rel_v_bm  = _rt.get('bm', '')
        _rel_xc_bm = _rel_xcomp.get('bm', '') if _rel_xcomp else ''
        _rel_o_bm  = _rel_obj.get('bm', '') if _rel_obj else ''
        _rel_o2_bm = _rel_obj2.get('bm', '') if _rel_obj2 else ''
        if _rel_o2_amod:
            _rel_o2_bm = j(_rel_o2_bm, _rel_o2_amod.get('bm', ''))
        _rel_tam  = _resolve_tam(_rt.get('tense', 'pres'), False, G_kg)
        _rel_obls = []
        if _rel_xcomp:
            for _robl in sorted(T, key=lambda x: x['orig_index']):
                if _robl['orig_index'] in processed_indices:
                    continue
                if _robl.get('dep') not in ('obl:mod', 'obl:arg', 'obl'):
                    continue
                if not nx.has_path(NX_G, _rel_xcomp['orig_index'], _robl['orig_index']):
                    continue
                _robl_bm   = _robl.get('bm') or f"[{_robl.get('lemma')}]"
                _robl_case = next((x for x in T if x.get('dep') == 'case'
                                   and x.get('head_index') == _robl['orig_index']), None)
                _robl_nmod = next((x for x in T if x.get('dep') == 'nmod'
                                   and x.get('head_index') == _robl['orig_index']), None)
                _robl_marker = _robl_case.get('bm_marker', 'la') if _robl_case else 'la'
                if _robl_nmod:
                    _robl_nmod_bm = _robl_nmod.get('bm') or f"[{_robl_nmod.get('lemma')}]"
                    _rel_obls.append(j(_robl_nmod_bm, _robl_bm, _robl_marker))
                else:
                    _rel_obls.append(j(_robl_bm, _robl_marker))
        _rel_o_final = _rel_o2_bm or _rel_o_bm
        if _rel_xc_bm:
            _rel_str = j('min', _rel_tam, _rel_v_bm, 'ka', _rel_o_final, _rel_xc_bm, *_rel_obls)
        else:
            _rel_str = j('min', _rel_tam, _rel_o_final, _rel_v_bm, *_rel_obls)
        m['OBL_ALL'].append({
            'HEAD': _rel_str, 'MARKER': '', 'local_clause_type': 'simple',
            'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
            'DEP_TYPE': 'acl:relcl', 'COMPOUND_IS_QUANTIFIER': False,
            'MARKER_IS_PREFIX': False,
        })
        _rel_desc = nx.descendants(NX_G, _rt['orig_index']) | {_rt['orig_index']}
        processed_indices.update(_rel_desc)

    # Pré-marquage ccomp
    for _cc in T:
        if _cc.get('dep') == 'ccomp' and _cc['orig_index'] not in processed_indices:
            _cc_desc = nx.descendants(NX_G, _cc['orig_index']) | {_cc['orig_index']}
            processed_indices.update(_cc_desc)

    # ── ÉTAPE 4 : OBLIQUES ────────────────────────────────────────────────────
    _loc_markers = G_kg.get('locative_markers', set())
    _tmp_markers = G_kg.get('temporal_markers', set())

    for tok_item in sorted(T, key=lambda x: x['orig_index']):
        # Fusion week-end
        if tok_item.get('dep') == 'obl:mod' and tok_item.get('pos') == 'NOUN':
            _next_toks = [x for x in T
                          if x['orig_index'] > tok_item['orig_index']
                          and x.get('dep') == 'obl:mod'
                          and x['orig_index'] not in processed_indices]
            _dash = next((x for x in _next_toks
                          if str(x.get('surface', '')).strip() == '-'), None)
            if _dash:
                _after_dash = next((x for x in _next_toks
                                    if x['orig_index'] > _dash['orig_index']), None)
                if _after_dash:
                    _fused_bm = j(tok_item.get('bm') or f"[{tok_item.get('lemma')}]",
                                  _after_dash.get('bm') or f"[{_after_dash.get('lemma')}]")
                    tok_item = dict(tok_item)
                    tok_item['bm'] = _fused_bm
                    processed_indices.add(_dash['orig_index'])
                    processed_indices.add(_after_dash['orig_index'])

        if tok_item['orig_index'] in processed_indices:
            continue

        if tok_item.get('dep') == 'advcl':
            _mark = next((x for x in T
                          if x.get('dep') == 'mark'
                          and x.get('head_index') == tok_item['orig_index']), None)
            if _mark and _mark.get('role') == 'purposive':
                _purp_marker = G_kg.get('purposive_marker', 'walasa ka') or 'walasa ka'
                _verb_bm     = tok_item.get('bm') or f"[{tok_item.get('lemma')}]"
                _conj_verbs  = [x for x in T
                                if x.get('dep') == 'conj'
                                and x.get('head_index') == tok_item['orig_index']
                                and x.get('pos') == 'VERB']
                _cc_tok = next((x for x in T
                                if x.get('dep') == 'cc'
                                and x.get('head_index') == tok_item['orig_index']), None)
                _cc_bm  = _cc_tok.get('bm', '') if _cc_tok and _cc_tok.get('bm') else ''
                for _cv in _conj_verbs:
                    _cv_bm   = _cv.get('bm') or f"[{_cv.get('lemma')}]"
                    _verb_bm = j(_verb_bm, _cc_bm, 'ka', _cv_bm)
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
                if _mark:
                    processed_indices.add(_mark['orig_index'])
            elif _mark and _mark.get('role') == 'privative':
                _verb_bm = tok_item.get('bm') or f"[{tok_item.get('lemma')}]"
                m['OBL_ALL'].append({
                    'HEAD': _verb_bm + 'bali', 'MARKER': '',
                    'local_clause_type': 'simple', 'COMPOUND': '', 'MOD': '',
                    'DEM_PREF': '', 'DEM_SUFF': '', 'DEP_TYPE': 'advcl',
                    'COMPOUND_IS_QUANTIFIER': False, 'MARKER_IS_PREFIX': False,
                })
                processed_indices.add(tok_item['orig_index'])
                if _mark:
                    processed_indices.add(_mark['orig_index'])
            continue

        if (tok_item.get('dep') == 'acl:relcl'
                and tok_item['orig_index'] not in processed_indices):
            _rel_subj  = next((x for x in T if x.get('dep') == 'nsubj'
                               and x.get('head_index') == tok_item['orig_index']), None)
            _rel_xcomp = next((x for x in T if x.get('dep') == 'xcomp'
                               and x.get('head_index') == tok_item['orig_index']), None)
            _rel_obj   = next((x for x in T if x.get('dep') == 'obj'
                               and x.get('head_index') == tok_item['orig_index']), None)
            _rel_obj2  = next((x for x in T if x.get('dep') == 'obj'
                               and _rel_xcomp
                               and x.get('head_index') == _rel_xcomp['orig_index']), None)
            _rel_s_bm  = _rel_subj.get('bm', '') if _rel_subj else ''
            _rel_v_bm  = tok_item.get('bm', '')
            _rel_xc_bm = _rel_xcomp.get('bm', '') if _rel_xcomp else ''
            _rel_o_bm  = _rel_obj.get('bm', '') if _rel_obj else ''
            _rel_o2_bm = _rel_obj2.get('bm', '') if _rel_obj2 else ''
            _rel_o2_amod = next((x for x in T if x.get('dep') == 'amod'
                                 and _rel_obj2
                                 and x.get('head_index') == _rel_obj2['orig_index']), None)
            if _rel_o2_amod:
                _rel_o2_bm = j(_rel_o2_bm, _rel_o2_amod.get('bm', ''))
            _rel_tam = _resolve_tam(tok_item.get('tense', 'pres'), False, G_kg)
            _rel_str = j('minw', _rel_tam, _rel_v_bm,
                         'ka' if _rel_xc_bm else '',
                         _rel_xc_bm, _rel_o2_bm or _rel_o_bm)
            m['OBL_ALL'].append({
                'HEAD': _rel_str, 'MARKER': '', 'local_clause_type': 'simple',
                'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
                'DEP_TYPE': 'acl:relcl', 'COMPOUND_IS_QUANTIFIER': False,
                'MARKER_IS_PREFIX': False,
            })
            _rel_desc = nx.descendants(NX_G, tok_item['orig_index']) | {tok_item['orig_index']}
            processed_indices.update(_rel_desc)
            continue

        if tok_item.get('pos') not in ('NOUN', 'PROPN', 'PRON', 'ADV', 'NUM'):
            continue
        if str(tok_item.get('surface', '')).strip() in (
                "'", "\u2019", "\u2018", '"', ',', '.', '-', '–', '—', '«', '»'):
            continue
        if tok_item.get('role') in _clitic_roles:
            continue
        if (tok_item.get('pos') == 'PRON'
                and not tok_item.get('bm')
                and tok_item.get('dep') in ('obl:arg', 'obj', 'obl', 'expl:comp', 'expl')):
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

        def _nmod_has_loc_adp(nmod_t):
            own_adp = next((p for p in T
                            if p.get('dep') == 'case'
                            and p.get('head_index') == nmod_t['orig_index']), None)
            if not own_adp:
                return False
            mk = own_adp.get('bm_marker', '')
            return mk in _loc_markers or mk in _tmp_markers

        compound_toks = sorted(
            [x for x in obl_chunk
             if x.get('dep') in ('nmod', 'nummod', 'det')
             and x != tok_item
             and x.get('role') not in ('article', 'pronoun', 'demonstrative')
             and not (x.get('pos') == 'DET' and not x.get('bm'))
             and not _nmod_has_loc_adp(x)],
            key=lambda x: x['orig_index'])

        demo_tok = next((x for x in obl_chunk
                         if x.get('role') == 'demonstrative'), None)

        head_base = tok_item.get('bm') or f"[{tok_item.get('lemma')}]"
        if tok_item.get('bm_suffix'):
            head_base += tok_item['bm_suffix']
        if (tok_item.get('pos') not in ('PROPN', 'PRON')
                and not head_base.startswith('[')
                and (tok_item.get('is_plural')
                     or str(tok_item.get('surface', '')).endswith('s'))
                and not head_base.endswith('w')):
            head_base = f"{head_base}w"

        _demo_surfaces_set = G_kg.get('demonstrative_surfaces', set())
        _demo_det = next((x for x in T
                          if x.get('dep') == 'det'
                          and (x.get('role') == 'demonstrative'
                               or str(x.get('surface', '')).lower() in _demo_surfaces_set)
                          and x.get('head_index') == tok_item['orig_index']), None)
        # Cas week-end fusionné : chercher aussi dans obl_chunk
        if not _demo_det:
            _demo_det = next((x for x in obl_chunk
                              if x.get('dep') == 'det'
                              and (x.get('role') == 'demonstrative'
                                   or str(x.get('surface', '')).lower() in _demo_surfaces_set)),
                             None)
        pref_val      = ''
        suff_dict_val = ''
        if _demo_det and not demo_tok:
            pref_val      = 'nin'
            suff_dict_val = _demo_det.get('bm_suffix') or G_kg.get('demonstrative_suffix', 'in') or 'in'
            processed_indices.add(_demo_det['orig_index'])

        marker_val = dep_case.get('bm_marker', '') if dep_case else ''

        if (marker_val in _loc_markers
                or tok_item.get('is_loc')
                or (dep_case and dep_case.get('role') == 'locative')):
            clause_type_val = 'locative'
            if not marker_val:
                marker_val = 'la'
        elif dep_case and dep_case.get('role') == 'comitative':
            clause_type_val = 'comitative'
            marker_val      = G_kg.get('comitative_marker', 'ni') or 'ni'
            # Capturer les conj coordonnés (et ma fille)
            _com_conjs = [x for x in T
                          if x.get('dep') == 'conj'
                          and x.get('head_index') == tok_item['orig_index']
                          and x['orig_index'] not in processed_indices]
            for _cc in _com_conjs:
                _cc_bm = _cc.get('bm') or f"[{_cc.get('lemma')}]"
                # Possessif du conj (ma fille → n mùsona)
                _cc_poss = next((x for x in T
                                 if x.get('dep') == 'det'
                                 and x.get('role') in ('pronoun', 'possessive')
                                 and x.get('head_index') == _cc['orig_index']), None)
                if _cc_poss:
                    _poss_bm = _cc_poss.get('bm', '')
                    _cc_bm = j('n', _cc_bm) if _poss_bm == 'n' else j(_poss_bm, 'ka', _cc_bm)
                    processed_indices.add(_cc_poss['orig_index'])
                # Relative du conj (qui est malade → mìn ka jànkarotɔ)
                _cc_relcl = next((x for x in T
                                  if x.get('dep') == 'acl:relcl'
                                  and x.get('head_index') == _cc['orig_index']), None)
                if _cc_relcl:
                    _rel_marker = G_kg.get('relative_marker', 'mìn') or 'mìn'
                    _rel_adj    = next((x for x in T
                                        if x.get('head_index') == _cc_relcl['orig_index']
                                        and x.get('pos') == 'ADJ'), None)
                    _rel_v_bm    = _cc_relcl.get('bm', '')
                    _rel_adj_bm  = _rel_adj.get('bm', '') if _rel_adj else ''
                    # Qualificatif → ka :
                    # 1. acl:relcl lui-même est ADJ (malade)
                    # 2. acl:relcl a un enfant ADJ sans verbe propre
                    _relcl_self_adj = _cc_relcl.get('pos') == 'ADJ'
                    if _relcl_self_adj or (_rel_adj and not _rel_v_bm):
                        _rel_tam = 'ka'
                    else:
                        _rel_tam = _resolve_tam(_cc_relcl.get('tense', 'pres'), False, G_kg)
                    _cc_bm = j(_cc_bm, _rel_marker, _rel_tam, _rel_v_bm or _rel_adj_bm)
                    # Verrouiller tous les descendants de acl:relcl
                    _relcl_desc = nx.descendants(NX_G, _cc_relcl['orig_index']) | {_cc_relcl['orig_index']}
                    processed_indices.update(_relcl_desc)
                # Ajouter au head_base avec marqueur ni
                head_base = j(head_base, marker_val, _cc_bm)
                processed_indices.add(_cc['orig_index'])
                _cc_tok_com = next((x for x in T
                                if x.get('dep') == 'cc'
                                and x.get('head_index') == tok_item['orig_index']), None)
                if _cc_tok_com:
                    processed_indices.add(_cc_tok_com['orig_index'])
        elif dep_case and dep_case.get('role') == 'privative':
            clause_type_val = 'privative'
            if tok_item.get('pos') == 'VERB':
                marker_val = 'bali'
            elif any(x.get('dep') == 'det'
                     and x.get('role') in ('pronoun', 'possessive')
                     and x.get('head_index') == tok_item['orig_index']
                     for x in T):
                marker_val = 'kɔ'
            elif tok_item.get('pos') == 'PRON':
                marker_val = 'kɔ'
            else:
                marker_val = 'tan'
        elif (marker_val in _tmp_markers
              or tok_item.get('dep') in ('obl:mod', 'advmod')
              or tok_item.get('role') == 'temporal'):
            clause_type_val = 'temporal'
        else:
            clause_type_val = 'simple'

        compound_elements = []
        absorbed_amods    = set()
        for c_tok in compound_toks:
            c_val  = c_tok.get('bm') or f"[{c_tok.get('lemma')}]"
            # Ne PAS absorber les amod du compound dans compound_elements
            # Les amod du compound iront dans mod_compiled après le noun_base
            # afin de qualifier le bon nom (ex: bìnniw [djihadiste] pas [djihadiste] bìnniw)
            compound_elements.append(c_val)
        compound_base = j(*compound_elements) if compound_elements else ''

        # mod_compiled = tous les amods du head + amods des compounds
        _compound_amods = []
        for c_tok in compound_toks:
            c_amod = next((x for x in obl_chunk
                           if x.get('dep') == 'amod'
                           and x.get('head_index') == c_tok['orig_index']), None)
            if c_amod and c_amod['orig_index'] not in absorbed_amods:
                _compound_amods.append(c_amod)
                absorbed_amods.add(c_amod['orig_index'])
        clean_amod_toks = [a for a in amod_toks if a['orig_index'] not in absorbed_amods]
        mod_compiled    = j(*[a.get('bm') or f"[{a.get('lemma')}]"
                               for a in clean_amod_toks + _compound_amods])

        if demo_tok and compound_base:
            suff_val = G_kg.get('demonstrative_suffix', 'in') or 'in'
            if not compound_base.endswith(suff_val):
                compound_base = f"{compound_base} {suff_val}"
            pref_val      = demo_tok.get('bm') or 'nin'
            suff_dict_val = ''
        elif demo_tok:
            pref_val      = demo_tok.get('bm') or ''
            suff_dict_val = G_kg.get('demonstrative_suffix', 'in') or ''

        loc_nmod_toks = sorted(
            [x for x in obl_chunk
             if x.get('dep') == 'nmod' and x != tok_item and _nmod_has_loc_adp(x)],
            key=lambda x: x['orig_index'])
        for loc_nmod in loc_nmod_toks:
            loc_adp    = next((p for p in T
                               if p.get('dep') == 'case'
                               and p.get('head_index') == loc_nmod['orig_index']), None)
            loc_marker = loc_adp.get('bm_marker', '') if loc_adp else ''
            loc_lct    = ('locative' if loc_marker in _loc_markers
                          else 'temporal' if loc_marker in _tmp_markers else 'simple')
            loc_head   = loc_nmod.get('bm') or f"[{loc_nmod.get('lemma')}]"
            m['OBL_ALL'].append({
                'HEAD': loc_head, 'MARKER': loc_marker, 'local_clause_type': loc_lct,
                'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
                'DEP_TYPE': 'case',
            })
            processed_indices.add(loc_nmod['orig_index'])
            if loc_adp:
                processed_indices.add(loc_adp['orig_index'])

        _root_verb_suffix = root_tok.get('bm_suffix', '') if root_tok else ''
        if _root_verb_suffix and dep_case:
            marker_val      = _root_verb_suffix
            clause_type_val = 'simple'

        _has_quantifier_compound = any(x.get('role') == 'quantifier'
                                       for x in compound_toks)

        if (dep_case and dep_case.get('role') == 'purposive'
                and tok_item.get('pos') in ('NOUN', 'PROPN', 'NUM')):
            marker_val      = G_kg.get('equative_marker', 'yé') or 'yé'
            clause_type_val = 'simple'

        # Si le wagon comitative a des conj intégrés dans head_base,
        # le marker 'ni' est déjà dans head_base → ne pas le redoubler dans MARKER
        _effective_marker = marker_val if marker_val else ''
        if clause_type_val == 'comitative' and _com_conjs:
            _effective_marker = ''  # ni déjà intégré dans head_base
        m['OBL_ALL'].append({
            'HEAD':               head_base,
            'MARKER':             _effective_marker,
            'local_clause_type':  clause_type_val,
            'COMPOUND':           compound_base,
            'MOD':                mod_compiled,
            'DEM_PREF':           pref_val,
            'DEM_SUFF':           suff_dict_val,
            'DEP_TYPE':           'case' if dep_case else '',
            'COMPOUND_IS_QUANTIFIER': _has_quantifier_compound,
            'MARKER_IS_PREFIX': (
                clause_type_val == 'temporal'
                and dep_case
                and dep_case.get('role') == 'temporal'
                and dep_case.get('bm_marker', '') in {'Kabini', "k'an bɔ"}
            ),
        })
        processed_indices.update([t['orig_index'] for t in obl_chunk])

    # Appartenance
    _purp_case = next((x for x in T
                       if x.get('dep') == 'case'
                       and x.get('role') == 'purposive'
                       and root_tok
                       and x.get('head_index') == root_tok['orig_index']), None)
    _has_cop = any(x.get('dep') == 'cop' for x in T)
    # if _purp_case and _has_cop and root_tok and root_tok.get('pos') == 'NOUN':
    #     tree['clause_type'] = 'ownership'
    #     _poss_det = next((x for x in T
    #                       if x.get('dep') == 'det'
    #                       and x.get('role') in ('pronoun', 'possessive')
    #                       and x.get('head_index') == root_tok['orig_index']), None)
    #     _root_bm  = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
    #     if _poss_det:
    #         _poss_bm = _poss_det.get('bm', '')
    #         m['O']   = (j('n', _root_bm) if _poss_bm == 'n'
    #                     else j(_poss_bm, 'ka', _root_bm))
    #         processed_indices.add(_poss_det['orig_index'])
    #     else:
    #         m['O'] = _root_bm
    #     processed_indices.add(root_tok['orig_index'])
    #     processed_indices.add(_purp_case['orig_index'])

    if _purp_case and _has_cop and root_tok and root_tok.get('pos') in ('NOUN', 'PRON'):
        tree['clause_type'] = 'ownership'
        if root_tok.get('pos') == 'PRON':
            m['O'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
        else:
            _poss_det = next((x for x in T
                              if x.get('dep') == 'det'
                              and x.get('role') in ('pronoun', 'possessive')
                              and x.get('head_index') == root_tok['orig_index']), None)
            _root_bm = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
            if _poss_det:
                _poss_bm = _poss_det.get('bm', '')
                m['O'] = (j('n', _root_bm) if _poss_bm == 'n'
                          else j(_poss_bm, 'ka', _root_bm))
                processed_indices.add(_poss_det['orig_index'])
            else:
                m['O'] = _root_bm
        processed_indices.add(root_tok['orig_index'])
        processed_indices.add(_purp_case['orig_index'])
        # ── NOUVEAU : capturer le démonstratif sur le sujet ──
        _subj_for_ownership = next((x for x in T
                                    if x.get('dep') in ('nsubj', 'nsubj:pass')
                                    and x.get('pos') == 'NOUN'), None)
        if _subj_for_ownership:
            _demo_surfaces_ow = G_kg.get('demonstrative_surfaces', set())
            _demo_on_subj = next((x for x in T
                                  if x.get('dep') == 'det'
                                  and (x.get('role') == 'demonstrative'
                                       or str(x.get('surface', '')).lower()
                                       in _demo_surfaces_ow)
                                  and x.get('head_index') == _subj_for_ownership['orig_index']),
                                 None)
            _subj_bm_ow = _subj_for_ownership.get('bm') or _subj_for_ownership.get('surface', '')
            if _demo_on_subj:
                _demo_suf_ow = G_kg.get('demonstrative_suffix', 'in') or 'in'
                _demo_bm_ow  = _demo_on_subj.get('bm') or 'nin'
                m['S'] = j(_demo_bm_ow, _subj_bm_ow, _demo_suf_ow)
                processed_indices.add(_demo_on_subj['orig_index'])
            else:
                m['S'] = _subj_bm_ow
            processed_indices.add(_subj_for_ownership['orig_index'])

    # ── ÉTAPE 5 : TRANSITIVITÉ ET COPULE ─────────────────────────────────────
    _has_obj      = any(x.get('dep') == 'obj' for x in T)
    _has_aux_pass = any(x.get('dep') == 'aux:pass' for x in T)
    _has_aux_cop  = any(x.get('dep') in ('aux', 'aux:tense')
                        and _is_copula(x) for x in T)
    if _has_obj:
        tree['is_transitive'] = True
    elif _has_aux_pass or _has_aux_cop:
        tree['is_transitive'] = False
    else:
        tree['is_transitive'] = True

    copula_tok = next((x for x in T
                       if x.get('dep') in ('cop', 'aux:pass')
                       and _is_copula(x)), None)
    has_with   = any(x.get('role') == 'comitative' for x in T)
    _root_idx  = root_tok['orig_index'] if root_tok else -1

    # être ROOT sans copule séparée → traiter comme copule
    # Conditions : semantic_class='copula' + pas de cop + pas d'interrogatif + pas de locatif
    _etre_as_root_cop = (
        root_tok
        and _is_copula(root_tok)
        and root_tok.get('pos') in ('VERB', 'AUX')
        and not copula_tok
        and not any(x.get('role') == 'interrogative' for x in T)
        and not any(x.get('dep') == 'case' and x.get('role') == 'locative' for x in T)
        and not any(x.get('is_loc') for x in T)
    )
    if _etre_as_root_cop:
        copula_tok = root_tok
        # Capturer l'attribut
        _cop_attr = next((x for x in T
                          if x.get('dep') in ('attr', 'xcomp', 'nsubj', 'ROOT')
                          and x.get('pos') in ('NOUN', 'PROPN', 'ADJ', 'PRON')
                          and x.get('orig_index') != root_tok['orig_index']
                          and x.get('orig_index') not in processed_indices), None)
        if _cop_attr:
            m['O'] = _cop_attr.get('bm') or f"[{_cop_attr.get('lemma')}]"
            processed_indices.add(_cop_attr['orig_index'])
        # Si attribut capturé en S par erreur → déplacer en O
        if not m.get('O') and m.get('S'):
            _s_tok = next((x for x in T
                           if x.get('bm') == m['S']
                           and x.get('dep') in ('nsubj', 'attr')), None)
            _real_subj = next((x for x in T
                               if x.get('dep') == 'nsubj'
                               and x.get('pos') == 'PRON'), None)
            if _s_tok and _real_subj and _s_tok != _real_subj:
                m['O'] = m['S']
                m['S'] = _real_subj.get('bm') or f"[{_real_subj.get('lemma')}]"

    _cop_is_on_root = copula_tok and copula_tok.get('head_index') == _root_idx
    # Statif via participe passé : aux:pass joue le rôle de copule
    print(f"DEBUG _statif_is_past check: aux_tense_tok={aux_tense_tok}, tree_tense={tree.get('tense')}")
    if not _cop_is_on_root and root_tok and root_tok.get('is_statif'):
        _aux_pass_cop = next((x for x in T
                              if x.get('dep') == 'aux:pass'
                              and x.get('head_index') == _root_idx), None)
        if _aux_pass_cop:
            copula_tok = _aux_pass_cop
            _cop_is_on_root = True
            
    if _cop_is_on_root:
        if tree.get('clause_type') in ('identificatory', 'ownership', 'locative'):
            pass
        elif has_with:
            tree['clause_type'] = 'presentative'
            tree['tam']         = G_kg.get('comitative_marker', 'ni')
        elif (_has_expletive
              and root_tok and root_tok.get('pos') == 'NOUN'
              and root_tok['orig_index'] not in processed_indices):
            # Cas 14 : Ce sont mes frères → n bálimakɛw dòn (présentatif possessif)
            tree['clause_type'] = 'presentative'
            _root_bm = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
            # Possessif rattaché au ROOT
            _poss_root = next((x for x in T
                               if x.get('dep') == 'det'
                               and x.get('role') in ('pronoun', 'possessive')
                               and x.get('head_index') == root_tok['orig_index']), None)
            if _poss_root:
                _poss_bm = _poss_root.get('bm', '')
                _root_bm = j('n', _root_bm) if _poss_bm == 'n' else j(_poss_bm, 'ka', _root_bm)
                processed_indices.add(_poss_root['orig_index'])
            # Pluriel
            if (root_tok.get('is_plural') or str(root_tok.get('surface','')).endswith('s')):
                if not _root_bm.endswith('w') and not _root_bm.endswith('w)'):
                    # Ajouter w sur le premier mot (le nom)
                    _parts = _root_bm.split(' ')
                    _parts[-1] = _parts[-1] + 'w' if not _parts[-1].endswith('w') else _parts[-1]
                    _root_bm = ' '.join(_parts)

            # Conj coordonnés (frères/sœurs)
            _root_conjs = [x for x in T
                           if x.get('dep') == 'conj'
                           and x.get('head_index') == root_tok['orig_index']
                           and x.get('bm')]
            for _rc in _root_conjs:
                _rc_bm = _rc.get('bm')
                if _poss_root:
                    _poss_bm = _poss_root.get('bm', '')
                    _rc_bm = j('n', _rc_bm) if _poss_bm == 'n' else j(_poss_bm, 'ka', _rc_bm)
                _cc_rc = next((x for x in T if x.get('dep') == 'cc'
                               and x.get('head_index') == root_tok['orig_index']), None)
                _cc_bm_rc = _cc_rc.get('bm', '') if _cc_rc and _cc_rc.get('bm') else ''
                _root_bm = j(_root_bm, _cc_bm_rc, _rc_bm)
                processed_indices.add(_rc['orig_index'])
                if _cc_rc: processed_indices.add(_cc_rc['orig_index'])
            m['O'] = _root_bm
            m['S'] = ''  # pas de sujet pour le présentatif
            processed_indices.add(root_tok['orig_index'])
        else:
            # Valeur abstraite (vrai, faux, possible...) → équatif
            if root_tok and root_tok.get('is_valeur') is True:
                tree['clause_type'] = 'equative'
                m['O'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
                tree['tam'] = _resolve_tam('pres', tree.get('neg', False), G_kg) if tree.get('neg') else G_kg.get('equative_marker', 'yé')
                processed_indices.add(root_tok['orig_index'])
                # Sujet = expletif 'c'/ce → o
                if not m.get('S'):
                    _expl_tok = next((x for x in T if x.get('role') == 'expletive' and x.get('bm')), None)
                    if _expl_tok:
                        m['S'] = _expl_tok.get('bm')
                        processed_indices.add(_expl_tok['orig_index'])

            _statif_obl = next((x for x in T
                                if x.get('dep') in ('obl', 'obl:arg', 'nmod')
                                and x.get('head_index') == _root_idx
                                and any(p.get('dep') == 'case'
                                        and p.get('role') == 'locative'
                                        and p.get('head_index') == x['orig_index']
                                        for p in T)), None)
            _morph_str = str(root_tok.get('morph', '')) if root_tok else ''
            _is_statif_adj = (root_tok
                              and root_tok.get('pos') == 'ADJ'
                              and (root_tok.get('semantic_class') == 'statif'
                                   or root_tok.get('statif_root')
                                   or root_tok.get('is_statif') is True
                                   or 'VerbForm=Part' in _morph_str
                                   or 'Tense=Past' in _morph_str))

            if root_tok and root_tok.get('pos') == 'ADJ' and (_statif_obl or _is_statif_adj):


                tree['clause_type'] = 'statif'
                _statif_base = (root_tok.get('statif_root')
                                or root_tok.get('bm')
                                or f"[{root_tok.get('lemma')}]")
                # Strip suffixe adjectival -a avant -len
                if (_statif_base.endswith('a')
                        and not _statif_base.endswith('ba')
                        and not _statif_base.endswith('ma')
                        and not _statif_base.endswith('ka')):
                    _statif_base = _statif_base[:-1]
                if not _statif_base.endswith('len'):
                    _statif_base += 'len'
                m['QUAL'] = _statif_base
                # Utiliser seulement aux_tense_tok pour le passé, pas le tense du participe
                _statif_is_past = (aux_tense_tok is not None
                                   and aux_tense_tok.get('tense') in ('past', 'hab'))
                if _statif_is_past:
                    tree['tam'] = 'tùn tɛ' if tree.get('neg') else 'tùn bɛ'
                    tree['clause_type'] = 'statif_past'
                else:
                    tree['tam'] = 'tɛ' if tree.get('neg') else 'dòn'
                processed_indices.add(root_tok['orig_index'])
            
            elif root_tok and root_tok.get('pos') in ('ADJ', 'ADV'):
                if root_tok.get('is_statif') is True:
                    tree['clause_type'] = 'statif'
                    _statif_base = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
                    if (_statif_base.endswith('a')
                            and not _statif_base.endswith('ba')
                            and not _statif_base.endswith('ma')
                            and not _statif_base.endswith('ka')):
                        _statif_base = _statif_base[:-1]
                    if not _statif_base.endswith('len'):
                        _statif_base += 'len'
                    m['QUAL'] = _statif_base
                    tree['tam'] = 'tɛ' if tree.get('neg') else 'dòn'
                    processed_indices.add(root_tok['orig_index'])
                elif root_tok.get('is_participe_passe') or (
                        not root_tok.get('is_valeur')
                        and not root_tok.get('is_statif')
                        and root_tok.get('bm', '').startswith('[')):
                    tree['clause_type'] = 'simple'
                    tree['is_transitive'] = False
                    tree['tense'] = 'past'
                    tree['tam'] = ''
                    m['V'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
                    if not m.get('S'):
                        _expl_tok = next((x for x in T
                                        if x.get('role') == 'expletive'
                                        and x.get('bm')), None)
                        if _expl_tok:
                            m['S'] = _expl_tok.get('bm')
                            processed_indices.add(_expl_tok['orig_index'])
                    processed_indices.add(root_tok['orig_index'])
                else:
                    tree['clause_type'] = 'qualitative'
                    m['QUAL'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
                    _qual_advs = [x for x in T
                                  if x.get('dep') == 'advmod'
                                  and x.get('head_index') == root_tok['orig_index']
                                  and x.get('bm')]
                    for _qa in _qual_advs:
                        m['QUAL'] = j(m['QUAL'], _qa.get('bm'))
                        processed_indices.add(_qa['orig_index'])
                    if tree.get('neg'):
                        tree['tam'] = 'man'
                    else:
                        tree['tam'] = 'ka'
                    if root_tok['orig_index'] in processed_indices:
                        m['V'] = ''
                        m['O'] = ''
            else:
                if tree.get('clause_type') != 'locative':
                    tree['clause_type'] = 'equative'
                    # Préserver le TAM temporel si déjà calculé par F1
                    if tree.get('tense') in ('past', 'hab', 'plup'):
                        _eq_neg = tree.get('neg', False)
                        _eq_tun = _resolve_tam(tree['tense'], False, G_kg) or 'tùn bɛ'
                        _eq_tun_base = _eq_tun.split()[0]  # 'tùn'
                        tree['tam'] = j(_eq_tun_base, 'tɛ' if _eq_neg else 'yé')
                    elif tree.get('neg'):
                        tree['tam'] = _resolve_tam('pres', True, G_kg)
                    else:
                        tree['tam'] = G_kg.get('equative_marker', 'yé')

        if root_tok and root_tok['orig_index'] not in processed_indices:
            root_bm = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
            if root_tok.get('bm_suffix'):
                root_bm += root_tok['bm_suffix']
            m['V'] = root_bm
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

    # Valeur abstraite (vrai, faux...) → équatif, indépendamment de _cop_is_on_root
    # c'est + ADJ → toujours équatif, indépendamment de _cop_is_on_root
    _adj_root_tok = next((x for x in T
                          if x.get('pos') == 'ADJ'
                          and x.get('dep') == 'ROOT'), None)
    if (_adj_root_tok and _has_expletive
            and tree.get('clause_type') not in ('equative', 'statif', 'locative')):
        if _adj_root_tok.get('is_statif'):
            pass
        elif _adj_root_tok.get('is_participe_passe') or (
                not _adj_root_tok.get('is_valeur')
                and not _adj_root_tok.get('is_statif')
                and _adj_root_tok.get('bm', '').startswith('[')):
            tree['clause_type'] = 'simple'
            tree['is_transitive'] = False
            tree['tense'] = 'past'
            tree['tam'] = ''
            m['V'] = _adj_root_tok.get('bm') or f"[{_adj_root_tok.get('lemma')}]"
            m['S'] = next((x.get('bm', '') for x in T
                          if x.get('role') == 'expletive' and x.get('bm')), 'o')
            processed_indices.add(_adj_root_tok['orig_index'])
        else:
            tree['clause_type'] = 'equative'
            m['O'] = _adj_root_tok.get('bm')
            tree['tam'] = (_resolve_tam('pres', True, G_kg) if tree.get('neg')
                          else G_kg.get('equative_marker', 'yé'))
            if not m.get('S'):
                _expl_tok = next((x for x in T if x.get('role') == 'expletive'
                                  and x.get('bm')), None)
                if _expl_tok:
                    m['S'] = _expl_tok.get('bm')
                    processed_indices.add(_expl_tok['orig_index'])
            m['V'] = ''
            processed_indices.add(_adj_root_tok['orig_index'])

    # Alignement TAM équatif interrogatif
    if clause_type_init == 'content_question':
        _has_be_copula   = any(_is_copula(x) or x.get('dep') == 'cop' for x in T)
        _has_loc_interrog = any(x.get('role') == 'interrogative'
                                and x.get('dep') in ('advmod', 'dep', 'obj')
                                for x in T)
        if (_has_be_copula
                or (root_tok and root_tok.get('role') == 'interrogative')):
            if not _has_loc_interrog:
                tree['tam'] = G_kg.get('equative_marker', 'yé') or 'yé'

    # ── ÉTAPE 6 : HARMONISATION NOMINALE ─────────────────────────────────────
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

    # Subordonnée complétive
    if tree.get('clause_type') == 'simple':
        ccomp_tok = next((x for x in T if x.get('dep') == 'ccomp'), None)
        if ccomp_tok:
            ccomp_subj = next((x for x in T
                               if x.get('dep') == 'nsubj'
                               and x.get('head_index') == ccomp_tok['orig_index']), None)
            ccomp_nmod = next((x for x in T
                               if x.get('dep') == 'nmod'
                               and x.get('head_index') == ccomp_tok['orig_index']), None)
            ccomp_head_bm = ccomp_tok.get('bm') or f"[{ccomp_tok.get('lemma')}]"
            if ccomp_nmod:
                ccomp_head_bm = j(ccomp_nmod.get('bm', ''),
                                  G_kg.get('genitive_marker', ''),
                                  ccomp_head_bm)
            m['CCOMP'] = {
                'S':   ccomp_subj.get('bm', '') if ccomp_subj else '',
                'tam': G_kg.get('equative_marker', 'yé') or 'yé',
                'O':   ccomp_head_bm,
                'V':   '',
            }
            ccomp_chunk = get_bounded_chunk_tokens(ccomp_tok['orig_index'])
            processed_indices.update([t['orig_index'] for t in ccomp_chunk])
            if ccomp_subj:
                processed_indices.add(ccomp_subj['orig_index'])
            if ccomp_nmod:
                processed_indices.add(ccomp_nmod['orig_index'])
                _cn_adp = next((x for x in T
                                if x.get('dep') == 'case'
                                and x.get('head_index') == ccomp_nmod['orig_index']), None)
                if _cn_adp:
                    processed_indices.add(_cn_adp['orig_index'])

    # Aiguillage présent intransitif (matrice 4 : cas B)
    if (root_tok and root_tok.get('pos') == 'VERB'
            and not m.get('O')
            and not m.get('V_ACTION')
            and tree.get('clause_type') == 'simple'):
        _intrans_type = root_tok.get('intransitive_type', '')
        if _intrans_type == 'support' and root_tok.get('action_noun'):
            # Catégorie 2 : nom support dédié → dúmúní kɛ
            m['O'] = root_tok.get('action_noun')
            m['V'] = 'kɛ'
        elif _intrans_type == 'nominalized':
            # Catégorie 3 : suffixe -li/-ni automatique → tobíli kɛ
            _v_root = root_tok.get('bm', '')
            if _v_root and not _v_root.endswith('li') and not _v_root.endswith('ni'):
                _suffix = 'ni' if _v_root.endswith('n') else 'li'
                m['O'] = _v_root + _suffix
            m['V'] = 'kɛ'
        # Catégorie 1 : rien à faire, V reste nu
    # Déduplication wagons
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

    # Slots finaux
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

    ordered_keys        = sorted(m['SLOTS'].keys(), key=lambda x: int(x[1:]))
    tree['final_string']       = j(*[m['SLOTS'][k] for k in ordered_keys])
    tree['local_clause_type']  = tree['clause_type']
    tree['_tokens']            = T

    return tree


def tree_to_bambara(tree, G=None, grammar=None):
    G   = grammar or G or {}
    m   = tree['main']
    ct  = tree['clause_type']
    S   = m.get('S', '') or ''
    O   = m.get('O', '') or ''
    V   = m.get('V', '') or ''
    V_ACT = m.get('V_ACTION', '') or ''
    V_SUF = m.get('V_SUFFIX', '') or ''
    ADV   = m.get('ADV', '')   or ''
    neg   = tree.get('neg', False)
    tn    = tree.get('tense', 'pres')

    tam_val = tree.get('tam', '')
    if not tam_val or tam_val.strip() == '':
        tam_val = 'bɛ'
    TAM = tam_val

    # Ccomp
    _ccomp_data = m.get('CCOMP')
    _ccomp_str  = ''
    if isinstance(_ccomp_data, dict) and _ccomp_data.get('O'):
        _ko        = G.get('reported_intro', 'ko') or 'ko'
        _ccomp_str = j(_ko, _ccomp_data.get('S', ''),
                       _ccomp_data.get('tam', 'yé'),
                       _ccomp_data.get('O', ''),
                       _ccomp_data.get('tam', 'yé'))

    # Wagons obliques
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
        if comp:
            if c.get('COMPOUND_IS_QUANTIFIER'):
                noun_base = j(head, comp)
            # elif lct in ('locative', 'temporal'):
            #     noun_base = j(comp, head)
            elif lct in ('locative', 'temporal'):
                if pref or suff:
                    # Le démonstratif entoure uniquement la tête, pas le compound
                    head_with_demo = j(pref, head, suff)
                    noun_base = j(comp, head_with_demo)
                    pref = ''
                    suff = ''
                else:
                    noun_base = j(comp, head)
            else:
                noun_base = j(comp, gen_marker, head)
        else:
            noun_base = head
        if mod:
            noun_base = j(noun_base, mod)
        full_chunk = j(pref, noun_base, suff) if (pref or suff) else noun_base
        if marker:
            if (lct == 'temporal'
                    and (c.get('MARKER_IS_PREFIX')
                         or str(marker).strip().lower() in ('kabini', "k'an bɔ"))):
                full_chunk = j(marker, full_chunk)
            elif lct == 'privative' and marker in ('tan', 'bali'):
                full_chunk = full_chunk + marker
            else:
                full_chunk = j(full_chunk, marker)
        obl_strings.append(full_chunk)

    # F6 : past intransitif
    _o_is_xcomp = m.get('O_IS_XCOMP', False)
    if tn == 'past' and not tree.get('is_transitive', True) and not _o_is_xcomp:
            if V:
                _v_parts = V.split(' ')
                _root_v  = _v_parts[0]
                _rest_v  = _v_parts[1:]
                if not _root_v.endswith('ra') and not _root_v.endswith('na'):
                    _root_v += 'na' if _root_v.endswith('n') else 'ra'
                V = j(_root_v, *_rest_v)
            TAM = ''
    else:
        TAM = tam_val
    print(f"DEBUG F6: V={V!r}, TAM={TAM!r}, S={S!r}")
    # Affichage slots
    print('\n  📦 SLOTS STRUCTURELS FINAUX :')
    if S:        print(f'     [ S     ] → {S}')
    if TAM and ct != 'noun_phrase':
                 print(f'     [ TAM   ] → {TAM}')
    if O:        print(f'     [ O     ] → {O}')
    if V:        print(f'     [ V     ] → {V}')
    if V_ACT:    print(f'     [ V_ACT ] → {V_ACT}')
    if ADV:      print(f'     [ ADV   ] → {ADV}')
    if V_SUF:    print(f'    [V_SUF  ] -> {V_SUF}')
    if _ccomp_str: print(f'    [ccomp  ] -> {_ccomp_str}')
    for _i, _xv in enumerate(obl_strings):
        lct = (m['OBL_ALL'][_i].get('local_clause_type', '')
               if _i < len(m.get('OBL_ALL', [])) else '')
        print(f'     [ X{_i+1:<3d}   ] → {_xv}  ({lct})')
    print(f"  🏷️  clause_type = {ct}")

    result = ''

    # F3 : résultatif passé
    if (ct in ('simple', 'complex', 'conditional', 'temporal', 'relative_post')
            and _o_is_xcomp and O and tn == 'past'
            and V and not V.endswith('ra') and not V.endswith('na')):
        V_past = V + 'ra'
        if obl_strings or ' ' in S:
            result = j(S, *obl_strings) + ', o ' + j(V_past, O, 'ye')
        else:
            result = j(S, V_past, O, 'ye')

    elif ct in ('statif', 'statif_past'):
        if ct == 'statif_past':
            # ST_03/ST_04 : S tùn V-len dòn/tɛ (pas de bɛ)
            _tun = 'tùn'
            _assert = 'tɛ' if neg else 'dòn'
            result = j(S, _tun, m.get('QUAL', ''), _assert, *obl_strings)
        else:
            # ST_01/ST_02 : S V-len dòn/tɛ
            result = j(S, m.get('QUAL', ''), TAM, *obl_strings)

    elif ct == 'existential_nominal':
        _exist_op    = 'tɛ' if neg else 'bɛ'
        _normal_obls = [obl_strings[i] for i, c in enumerate(m.get('OBL_ALL', []))
                        if i < len(obl_strings) and c.get('DEP_TYPE') != 'acl:relcl']
        _relcl_obls  = [obl_strings[i] for i, c in enumerate(m.get('OBL_ALL', []))
                        if i < len(obl_strings) and c.get('DEP_TYPE') == 'acl:relcl']
        result = j(O, *_normal_obls, _exist_op, ',', *_relcl_obls)

    elif ct == 'existential_absolute':
        # Matrice cas 13 : [Nom] + bɛ (existence pure)
        # Le sujet bambara est le nom lui-même, pas le pronom français 'il'
        # 'Il y a du pain' → búuru bɛ (pas a bɛ búuru)
        # 'Il y a du pain ici' → búuru bɛ yàn
        _exist_op   = 'tɛ' if neg else 'bɛ'
        _exist_loc  = G.get('existence_loc_marker', 'yàn') or 'yàn'
        # Utiliser O comme sujet bambara (inversion)
        _exist_subj = O or S
        if obl_strings:
            result = j(_exist_subj, _exist_op, *obl_strings)
        else:
            result = j(_exist_subj, _exist_op)

    elif ct == 'existential_localized':
        # Matrice cas 13 : [Nom] + bɛ + [Lieu]
        # 'Il y a de l'eau dans la bouteille' → jí bɛ bútèli kɔnɔ
        _exist_op   = 'tɛ' if neg else 'bɛ'
        _exist_subj = O or S
        result      = j(_exist_subj, _exist_op, *obl_strings)

    elif ct == 'infinitive':
        result = j('ka', O, V, *obl_strings)

    elif ct == 'ownership':
        result = j(S, 'yé', O, 'de', 'ta', 'ye')

    elif ct == 'verb_serial':
        _com_marker = G.get('comitative_marker', 'ni') or 'ni'
        _com_heads, _other_obls = [], []
        _raw_obls = m.get('OBL_ALL', [])
        for _ci, _obl in enumerate(_raw_obls):
            if isinstance(_obl, dict) and _obl.get('local_clause_type') == 'comitative':
                _head_only = _obl.get('HEAD', '')
                if _obl.get('COMPOUND'):
                    _head_only = _obl['COMPOUND'] + ' ' + _head_only
                if _obl.get('MOD'):
                    _head_only = j(_head_only, _obl['MOD'])
                _com_heads.append(_head_only)
                _tokens_ref = tree.get('_tokens', [])
                _head_orig  = next((t.get('orig_index') for t in _tokens_ref
                                    if t.get('bm') == _obl.get('HEAD')
                                    or (t.get('bm') and t.get('bm') in _head_only)), None)
                if _head_orig is not None:
                    for _conj_tok in _tokens_ref:
                        if (_conj_tok.get('dep') == 'conj'
                                and _conj_tok.get('head_index') == _head_orig):
                            _conj_bm = _conj_tok.get('bm') or ''
                            _conj_has_relcl = any(
                                t.get('dep') == 'acl:relcl'
                                and t.get('head_index') == _conj_tok['orig_index']
                                for t in _tokens_ref)
                            if not _conj_has_relcl and _conj_bm:
                                _com_heads.append(_conj_bm)
            else:
                _other_obls.append(obl_strings[_ci] if _ci < len(obl_strings) else '')
        # _com_str : si les conj sont déjà dans head, utiliser directement
        if _com_heads:
            _com_str = j(_com_marker, (' ' + _com_marker + ' ').join(_com_heads))
        else:
            _com_str = ''
        if O:
            result = j(S, TAM, V, 'ka', O, _com_str, V_ACT, *_other_obls)
        else:
            result = j(S, TAM, V, 'ka', V_ACT, _com_str, *_other_obls)

    elif ct == 'interrogative':
        _question_marker = next((t.get('bm', '') for t in tree.get('_tokens', [])
                                 if t.get('role') == 'question_marker'
                                 and t.get('bm')), '')
        _alt_tok = next((t for t in tree.get('_tokens', [])
                         if t.get('role') == 'alternative' and t.get('bm')), None)
        _interrog_end = next((t.get('bm', '') for t in tree.get('_tokens', [])
                              if t.get('role') == 'interrogative_end'
                              and t.get('bm')), 'wà ?')
        _clean_obls = [_xv for _ci, _xv in enumerate(obl_strings)
                       if _ci < len(m.get('OBL_ALL', []))
                       and m['OBL_ALL'][_ci].get('DEP_TYPE') != 'fixed'
                       and not any(t.get('dep') == 'fixed'
                                   and t.get('bm') == m['OBL_ALL'][_ci].get('HEAD')
                                   for t in tree.get('_tokens', []))]
        # Éviter doublon TAM == V
        _v_display = V if (V and V.rstrip('́') != TAM.rstrip('́')) else ''

        if _alt_tok:
            _conj_v = next((t for t in tree.get('_tokens', [])
                            if t.get('dep') == 'conj'
                            and t.get('pos') == 'VERB'
                            and t.get('bm')), None)
            _conj_o = next((t for t in tree.get('_tokens', [])
                            if t.get('dep') == 'obj'
                            and _conj_v
                            and t.get('head_index') == _conj_v.get('orig_index')
                            and t.get('bm')), None)
            _conj_o_amod = next((t for t in tree.get('_tokens', [])
                                 if t.get('dep') == 'amod'
                                 and _conj_o
                                 and t.get('head_index') == _conj_o.get('orig_index')
                                 and t.get('bm')), None)
            _conj_o_bm = (j(_conj_o.get('bm', ''),
                            _conj_o_amod.get('bm', '') if _conj_o_amod else '')
                          if _conj_o else '')
            _conj_v_bm = _conj_v.get('bm', '') if _conj_v else ''
            result = j(S, TAM, O, _v_display, *_clean_obls,
                       _alt_tok.get('bm', ''), S, TAM, _conj_o_bm, _conj_v_bm, '?')
        else:
            _v_is_motion = next((t for t in tree.get('_tokens', [])
                                 if t.get('is_root')
                                 and t.get('semantic_class') == 'motion'), None)
            if _v_is_motion and _v_display:
                result = j(_question_marker, S, TAM, _v_display, O,
                           *_clean_obls, _interrog_end)
            else:
                result = j(_question_marker, S, TAM, O, _v_display,
                           *_clean_obls, _interrog_end)

    elif ct == 'equative':
        if neg and tree.get('tense') in ('past', 'hab', 'plup'):
            _tun_base = TAM.split()[0] if TAM and ' ' in TAM else 'tùn'
            result = j(S, _tun_base, 'tɛ', O or V, 'yé', *obl_strings)
        elif neg:
            # result = j(S, 'tɛ', O or V, 'yé', *obl_strings)
            result = j(S, 'tɛ', O or V, 'yé', *obl_strings)
        elif O and str(O).endswith('yé'):
            result = j(S, tree.get('tam', 'yé'), O, *obl_strings)
        elif V and not O and tn == 'past':
            _already = tree.get('already_marker', '')
            _v_past = V
            if not _v_past.endswith('ra') and not _v_past.endswith('na'):
                _v_past = _v_past + 'na' if _v_past.endswith('n') else _v_past + 'ra'
            result = j(S, _v_past, _already, *obl_strings)
        else:
            _why_tok = next((t for t in tree.get('_tokens', [])
                             if t.get('role') == 'interrogative'
                             and t.get('dep') == 'advmod'
                             and t.get('bm')), None)
            if _why_tok:
                _adj_bm = V or next((t.get('bm', '') for t in tree.get('_tokens', [])
                                     if t.get('pos') == 'ADJ'
                                     and t.get('is_root')), '')
                result = j(S, _adj_bm, 'lendòn', _why_tok.get('bm', ''), '?')
            else:
                tam_equatif = tree.get('tam', 'yé')
                result = j(S, tam_equatif, O or V, 'yé', *obl_strings, _ccomp_str)
    elif ct == 'identificatory':
        _focus_marker = G.get('focus_marker', 'de')
        _appos = next((t for t in tree.get('_tokens', [])
                       if t.get('pos') == 'PROPN'
                       and t.get('dep') in ('ROOT', 'appos', 'flat', 'flat:name')), None)
        _appos_bm = (_appos.get('bm') or _appos.get('surface', '')) if _appos else ''
        if not S and _appos_bm:
            result = j(_appos_bm, 'tɛ' if neg else 'dòn')
        elif _appos_bm and _appos_bm != S:
            result = j(S, _focus_marker, 'dòn') + ', ' + _appos_bm
        else:
            result = j(S, 'tɛ') if neg else j(S, 'dòn')

    elif ct == 'presentative':
        if neg:
            result = j(S, 'tɛ', *obl_strings)
        else:
            # Déictique (voilà) → [Nom] félé — role='deictique' depuis KG
            _is_deictique = any(t.get('role') == 'deictique' for t in tree.get('_tokens', []))
            if _is_deictique:
                result = j(O or S, G.get('deictique_marker', 'félé'))
            else:
                o_clean = O if O != S else ''
                if o_clean:
                    result = j(S, 'ni', o_clean, 'dòn', *obl_strings)
                else:
                    result = j(S, 'dòn', *obl_strings)

    elif ct == 'noun_phrase':
        _conn_suf = (tree['main'].get('SLOTS', {}).get('conn_suffix', '')
                     or next((c.get('DEM_SUFF', '')
                               for c in m.get('OBL_ALL', [])
                               if c.get('DEM_SUFF')), ''))
        if not _conn_suf and tree.get('_tokens'):
            _conn_suf = next((t.get('bm_suffix', '')
                              for t in tree['_tokens']
                              if t.get('bm_suffix')), '')
        _already     = tree.get('already_marker', '')
        _already_pos = tree.get('already_position', 'END')
        if _conn_suf and _conn_suf.strip():
            if not str(O).endswith(_conn_suf) and not any(
                    str(xv).endswith(_conn_suf) for xv in obl_strings):
                result = (j(_already, O, *obl_strings, _conn_suf)
                          if _already_pos == 'HEAD'
                          else j(O, _already, *obl_strings, _conn_suf))
            else:
                result = (j(_already, O, *obl_strings)
                          if _already_pos == 'HEAD'
                          else j(O, _already, *obl_strings))
        else:
            if _already_pos == 'HEAD':
                result = j(_already, O, *obl_strings)
            else:
                _tmp_obls   = [obl_strings[i]
                               for i, c in enumerate(m.get('OBL_ALL', []))
                               if i < len(obl_strings)
                               and isinstance(c, dict)
                               and c.get('local_clause_type') == 'temporal']
                _other_obls = [obl_strings[i]
                               for i, c in enumerate(m.get('OBL_ALL', []))
                               if i < len(obl_strings)
                               and isinstance(c, dict)
                               and c.get('local_clause_type') != 'temporal']
                result = j(O, *_other_obls, _already, *_tmp_obls)

    elif ct == 'privative_pred':
        if not S:
            result = j(O, *obl_strings)
        else:
            result = j(S, O, 'dòn', *obl_strings)

    elif ct == 'relative_topic':
        _rel_neg = tree.get('neg', False)
        _rel_tam = 'tɛ' if _rel_neg else 'bɛ'
        if V_ACT:
            # SOV bambara : S TAM V ka O V_ACT
            result = j(S, ',', 'o', _rel_tam, V, 'ka', O, V_ACT, *obl_strings)
        else:
            V_past = V if V.endswith('ra') or V.endswith('na') else (
                V + 'na' if V.endswith('n') else V + 'ra')
            result = j(S, ',', 'o', V_past, *obl_strings)

    elif ct in ('simple', 'complex', 'conditional', 'temporal',
                'relative_post', 'reported_comp', 'comitative'):
        _contrast_tok = next((t for t in tree.get('_tokens', [])
                              if t.get('role') == 'contrast' and t.get('bm')), None)
        _contrast_str = _contrast_tok.get('bm', '') if _contrast_tok else ''
        _raw_obls     = m.get('OBL_ALL', [])
        _tmp_strs, _other_strs = [], []
        _tokens_ref = tree.get('_tokens', [])
        for _ci, _obl in enumerate(_raw_obls):
            _xv = obl_strings[_ci] if _ci < len(obl_strings) else ''
            if isinstance(_obl, dict) and _obl.get('local_clause_type') == 'temporal':
                _obl_marker  = _obl.get('MARKER', '')
                _suffix_mkrs = G.get('temporal_suffix_markers', set())
                if _obl_marker and _obl_marker in _suffix_mkrs:
                    _other_strs.append(_xv)
                    continue
                _obl_head = _obl.get('HEAD', '')
                _src_tok  = next((t for t in _tokens_ref
                                  if (t.get('bm') == _obl_head
                                      or t.get('surface') == _obl_head)
                                  and t.get('dep') == 'advmod'
                                  and t.get('pos') == 'ADV'), None)
                if _src_tok:
                    _tmp_strs.append(_xv)
                else:
                    _other_strs.append(_xv)
            else:
                _other_strs.append(_xv)

        _already     = tree.get('already_marker', '')
        _already_pos = tree.get('already_position', 'END')

        if TAM in ('bɛ kà', 'tɛ kà'):
            result = j(*_tmp_strs, S, TAM, O, V, ADV, *_other_strs)
        elif _o_is_xcomp and O:
            if _already_pos == 'HEAD':
                result = j(_already, _contrast_str, S, TAM, V, O, V_ACT, V_SUF,
                           *_tmp_strs, ADV, *_other_strs, _ccomp_str)
            else:
                result = j(_contrast_str, S, TAM, V, O, V_ACT, V_SUF,
                           *_tmp_strs, ADV, *_other_strs, _ccomp_str, _already)
        else:
            if _already_pos == 'HEAD':
                result = j(_already, _contrast_str, S, TAM, O, V, V_ACT, V_SUF,
                           *_tmp_strs, ADV, *_other_strs, _ccomp_str)
            
            else:
                result = j(_contrast_str, S, TAM, O, V, V_ACT, V_SUF,
                           *_tmp_strs, ADV, *_other_strs, _ccomp_str, _already)

    elif ct == 'qualitative':
        _qual_content = m.get('QUAL', '') or O or V
        # Participe passé résultatif → V+ra/la/na (pas de ka)
        _tokens_ref = tree.get('_tokens', [])
        _root_is_participe = any(
            t.get('is_participe_passe') is True
            for t in _tokens_ref if t.get('is_root'))
        if _root_is_participe:
            _v = _qual_content
            if _v and not _v.startswith('['):
                if _v.endswith('n'):
                    _v += 'na'
                elif _v[-1] in ('o', 'u', 'ɔ'):
                    _v += 'la'
                else:
                    _v += 'ra'
            result = j(S, _v, *obl_strings)
        else:
            _qual_tam = TAM if TAM in ('man', 'ma') else ('man' if neg else 'ka')
            if tn == 'past' or tn == 'hab':
                _tun = _resolve_tam(tn, neg, G) or ('tùn tɛ' if neg else 'tùn bɛ')
                _tun_base = _tun.split()[0] if _tun else 'tùn'
                _qual_tam = j(_tun_base, 'ma' if neg else 'ka')
            result = j(S, _qual_tam, _qual_content, *obl_strings, _ccomp_str)

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
        if neg:
            result = j(S, 'ma', O, 'fɔ', ADV, *obl_strings)
        else:
            result = j(S, TAM, O, V, ADV, *obl_strings)

    elif ct == 'content_question':
        # Nom seul + det interrogatif (Quelle femme ?)
        _interrog_det_on_s = next((t for t in tree.get('_tokens', [])
                                   if t.get('role') == 'interrogative'
                                   and t.get('dep') == 'det'
                                   and t.get('bm')), None)
        if _interrog_det_on_s and S and not O and not V:
            result = j(S, _interrog_det_on_s.get('bm', 'jùmɛn'), '?')
            print(f"  ✂️  Clause 1 -> '{result}'")
            return result.strip()

        # S contient l'interrogatif → déplacer en O, récupérer vrai sujet
        _s_is_interrog = any(t.get('role') == 'interrogative' and t.get('bm') == S
                             for t in tree.get('_tokens', []))
        if _s_is_interrog:
            if not O:
                O = S
            _real_subj = next((t for t in tree.get('_tokens', [])
                               if t.get('pos') == 'PRON'
                               and t.get('dep') in ('nsubj', 'nsubj:pass', 'dep')
                               and t.get('bm')
                               and t.get('bm') != S
                               and t.get('role') not in {'interrogative', 'expletive', 'clitic'}
                               and 'Int' not in str(t.get('morph', ''))), None)
            S = _real_subj.get('bm', '') if _real_subj else ''
        elif not S:
            _subj_tok = next((t for t in tree.get('_tokens', [])
                              if t.get('pos') == 'PRON'
                              and t.get('dep') in ('nsubj', 'nsubj:pass')
                              and t.get('bm')), None)
            if _subj_tok:
                S = _subj_tok.get('bm', '')

            # Récupérer le mot interrogatif en O s'il est absent
            if not O:
                _interrog_tok = next((t for t in tree.get('_tokens', [])
                                    if t.get('role') == 'interrogative'
                                    and t.get('dep') in ('obj', 'nsubj', 'dep')), None)
                if _interrog_tok:
                    O = _interrog_tok.get('bm') or G.get('interrogative_who', 'jɔn')

        _interrog_adv = next((t for t in tree.get('_tokens', [])
                              if t.get('role') == 'interrogative'
                              and t.get('dep') in ('advmod', 'dep')
                              and t.get('bm')), None)
        _interrog_adv_bm  = _interrog_adv.get('bm', '') if _interrog_adv else ''
        _interrog_noun    = tree.get('interrog_noun')
        _interrog_noun_bm = _interrog_noun.get('bm', '') if _interrog_noun else ''
        if _interrog_noun_bm and O:
            O = j(_interrog_noun_bm, O)
        elif _interrog_noun_bm:
            O = _interrog_noun_bm

        if TAM == 'yé' and not _interrog_noun_bm:
            result = j(S, TAM, V or O, 'yé', _interrog_adv_bm, *obl_strings)
        elif TAM == 'yé' and _interrog_noun_bm:
            result = j(S, TAM, O, V, '?')
        else:
            _adv_already_in_O = _interrog_adv_bm and _interrog_adv_bm in O
            if _adv_already_in_O:
                result = j(S, TAM, V, O, *obl_strings, '?')
            else:
                _o_final = O or G.get('interrogative_who', 'jɔn')
                result = j(S, TAM, _o_final, V, _interrog_adv_bm, *obl_strings, '?')

    elif ct == 'noun_phrase_have':
        # Matrice document :
        # Possession matérielle (objet physique, argent, voiture) → bóló
        # Possession abstraite (famille, sentiment, âge, lien) → fɛ
        # Détection depuis KG via possession_type du token objet
        _poss_type   = tree.get('possession_type', '')
        if not _poss_type:
            # Fallback : détecter via semantic_class de l'objet
            _obj_tok_have = next((t for t in tree.get('_tokens', [])
                                  if t.get('bm') == O and t.get('pos') in ('NOUN', 'PROPN')), None)
            if _obj_tok_have:
                _sc = _obj_tok_have.get('semantic_class', '')
                _poss_type = 'material' if _sc in ('object', 'money', 'vehicle', 'tool') else 'abstract'
        _poss_marker = G.get('possession_material_marker', 'bóló') if _poss_type == 'material'                        else G.get('possession_abstract_marker', 'fɛ')
        _exist_op    = 'tɛ' if neg else 'bɛ'
        _qty_interrog = next((t for t in tree.get('_tokens', [])
                              if t.get('role') == 'interrogative'
                              and t.get('bm')
                              and t.get('dep') in ('det', 'advmod', 'dep', 'obj')), None)
        if _qty_interrog and _qty_interrog.get('bm') not in (O or ''):
            result = j(O, _exist_op, S, _poss_marker, _qty_interrog.get('bm'), '?')
        else:
            result = j(O, _exist_op, S, _poss_marker, *obl_strings)

    elif ct == 'noun_phrase_inh':
        result = j(O, *obl_strings) if obl_strings else j(S, O)

    elif ct == 'comitative':
        _tokens = tree.get('_tokens', [])
        _avec   = next((t for t in _tokens
                        if isinstance(t, dict)
                        and t.get('role') == 'preposition'
                        and t.get('dep') == 'case'), None)
        _comp   = None
        if _avec:
            av_i = _avec.get('orig_index', -1)
            _cn  = next((t for t in _tokens
                         if isinstance(t, dict)
                         and t.get('pos') in ('NOUN', 'PROPN', 'PRON')
                         and (t.get('head_index', -1) == av_i
                              or _avec.get('head_index', -1) == t.get('orig_index', -1))),
                        None)
            if _cn:
                _cp = next((t for t in _tokens
                             if isinstance(t, dict)
                             and t.get('role') == 'possessive'
                             and t.get('head_index', -1) == _cn.get('orig_index', -1)), None)
                _bm  = _cn.get('bm', '')
                if _cp:
                    _p    = _cp.get('bm', '')
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
            {'role': role})
        return {r['m'] for r in rows if r.get('m')}

    def _single_marker(self, cypher: str, params=None) -> str:
        rows = self._q(cypher, params)
        return rows[0].get('m', '') if rows else ''

    def _load_grammar(self):
        g = self.grammar

        g['action_noun_map'] = {}
        action_rows = self._q(
            "MATCH (v:Verb) WHERE v.action_noun IS NOT NULL "
            "RETURN v.bm AS bm, v.action_noun AS noun")
        g['action_noun_map'] = {r['bm']: r['noun']
                                for r in action_rows if r.get('bm')}
        
        # Négation depuis KG
        neg_rows = self._q(
            "MATCH (f:FunctionWord) WHERE f.role = 'negation' AND f.lang = 'fr' "
            "RETURN f.surface AS s")
        g['neg_surfaces'] = {r['s'].lower() for r in neg_rows if r.get('s')}
        g['neg_surfaces'] |= {r['s'].lower().rstrip("'").rstrip('\u2019')
                               for r in neg_rows if r.get('s')}

        # Marqueur privatif pour pronoms
        g['privative_pron_marker'] = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'privative_pron_marker' "
            "RETURN f.bm AS m LIMIT 1") or 'kɔ'

        # Rôles depuis KG
        exp_rows = self._q(
            "MATCH (f:FunctionWord) WHERE f.role = 'expletive' "
            "RETURN DISTINCT f.role AS r")
        g['expletive_roles'] = {r['r'] for r in exp_rows if r.get('r')} or {'expletive'}

        clit_rows = self._q(
            "MATCH (f:FunctionWord) WHERE f.role = 'clitic' "
            "RETURN DISTINCT f.role AS r")
        g['clitic_roles'] = {r['r'] for r in clit_rows if r.get('r')} or {'clitic'}

        rel_rows = self._q(
            "MATCH (f:FunctionWord) WHERE f.role = 'relative' "
            "RETURN DISTINCT f.role AS r")
        g['relative_roles'] = {r['r'] for r in rel_rows if r.get('r')} or {'relative'}

        g['privative_markers']     = self._marker_set('privative')
        g['locative_markers']      = self._marker_set('locative')
        g['temporal_markers']      = self._marker_set('temporal')
        g['temporal_suffix_markers'] = {
            r['m'] for r in self._q(
                "MATCH (p:Preposition) WHERE p.is_suffix = true "
                "RETURN p.bm_marker AS m") if r.get('m')}
        g['focus_marker'] = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'focus_marker' AND f.lang = 'bm' "
            "RETURN f.bm AS m LIMIT 1") or 'de'

        temporal_rows = self._q(
            "MATCH (n) WHERE n.role IS NOT NULL AND n.semantic_class = 'temporal' "
            "RETURN DISTINCT n.role AS r")
        g['temporal_roles'] = {r['r'] for r in temporal_rows if r.get('r')}
        g['temporal_roles'].add('temporal')

        quantifier_rows = self._q(
            "MATCH (f:FunctionWord) WHERE f.role = 'quantifier' AND f.lang = 'fr' "
            "RETURN f.surface AS surface, f.bm AS bm")
        g['quantifier_words'] = {r['surface']: r['bm']
                                  for r in quantifier_rows if r.get('surface')}

        g['genitive_marker']   = self._single_marker(
            "MATCH (p:Preposition) WHERE p.role = 'genitive' "
            "RETURN p.bm_marker AS m LIMIT 1")
        g['comitative_marker'] = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'comitative' "
            "RETURN f.bm AS m LIMIT 1")
        g['agent_postposition'] = self._single_marker(
            "MATCH (p:Preposition) WHERE p.role = 'agent' "
            "RETURN p.bm_marker AS m LIMIT 1")
        g['purposive_marker']  = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'purposive' "
            "RETURN f.bm AS m LIMIT 1")
        g['reported_intro']    = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'reported_intro' "
            "RETURN f.bm AS m LIMIT 1")
        g['relative_marker']   = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'relative_marker' "
            "RETURN f.bm AS m LIMIT 1") or 'mìn'
        g['equative_marker']   = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'equative_marker' "
            "RETURN f.bm AS m LIMIT 1") or 'yé'
        g['possession_material_marker'] = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'possession_material' "
            "RETURN f.bm AS m LIMIT 1") or 'bóló'
        g['possession_abstract_marker'] = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'possession_abstract' "
            "RETURN f.bm AS m LIMIT 1") or 'fɛ'
        g['deictique_marker']  = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'deictique' "
            "RETURN f.bm AS m LIMIT 1") or 'félé'
        
        g['existence_loc_marker'] = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'existence_loc' "
            "RETURN f.bm AS m LIMIT 1") or 'yàn'
        
        g['demonstrative_surfaces'] = {
            r['s'].lower() for r in self._q(
                "MATCH (f:FunctionWord) WHERE f.role = 'demonstrative' "
                "RETURN f.surface AS s") if r.get('s')}
        tam_rows = self._q(
            "MATCH (t:TamConfig) RETURN t.tense AS tense, t.neg AS neg, "
            "t.bm AS bm, t.aspect AS aspect, t.ctx AS ctx")
        if tam_rows:
            g['tam_table'] = {
                (r['tense'], r['neg']): r['bm']
                for r in tam_rows
                if r.get('tense') is not None
                and r.get('neg') is not None
                and r.get('bm')}
            g['tam_default']      = g['tam_table'].get(('pres', False), '')
            g['tam_future']       = g['tam_table'].get(('fut',  False), '')
            g['progressive_tams'] = {r['bm'] for r in tam_rows
                                     if r.get('aspect') == 'progressive' and r.get('bm')}
            print(f"     TAM: {len(g['tam_table'])} entrées KG chargées.")
        else:
            g['tam_table']        = {}
            g['tam_default']      = ''
            g['tam_future']       = ''
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