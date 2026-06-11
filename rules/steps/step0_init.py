"""
rules/steps/step0_init.py
Étape 0 : initialisation du graphe NetworkX, détection du root_tok,
clause_type_init (simple/verb_serial/interrogative/content_question/identificatoire).
"""
import networkx as nx
from rules.core import _is_copula, _is_avoir


def run(T, G_kg):
    """
    Retourne (NX_G, root_tok, xcomp_verb_tok, xcomp_adj_tok,
              clause_type_init, _has_expletive, _has_propn,
              _has_pron_root, _has_propn_root,
              _neg_surfaces, _expletive_roles, _clitic_roles, _relative_roles)
    """
    # ── GRAPH NetworkX ────────────────────────────────────────────────────────
    # On ajoute d'abord TOUS les nœuds (un par token réel), puis seulement les
    # arêtes dont la tête est un VRAI token. Sinon (head supprimé/fusionné, ex:
    # 'en train de' retiré au progressif → courir garde un head_index pendant)
    # add_edge créerait un nœud fantôme sans attribut 'token' → KeyError.
    # Pour tout graphe valide, le résultat est identique à l'ancien.
    NX_G = nx.DiGraph()
    _valid_orig = {t['orig_index'] for t in T}
    for t in T:
        NX_G.add_node(t['orig_index'], token=t)
    for t in T:
        _h = t['head_index']
        if t['orig_index'] != _h and _h in _valid_orig:
            NX_G.add_edge(_h, t['orig_index'])

    # ── ROOT TOK ──────────────────────────────────────────────────────────────
    root_tok = next((t for t in T if t.get('is_root')), None)

    if (root_tok and root_tok.get('role') == 'expletive'
            and root_tok.get('pos') == 'PRON'):
        _real_root = next((t for t in T
                           if t.get('pos') in ('ADJ', 'NOUN')
                           and t.get('dep') == 'ROOT'), None)
        if _real_root:
            root_tok = _real_root
    elif (root_tok and root_tok.get('pos') == 'PRON'
            and len(str(root_tok.get('surface', ''))) == 1
            and any(t.get('pos') == 'VERB' and t.get('dep') == 'ROOT' for t in T)):
        _verb_root = next((t for t in T
                           if t.get('pos') == 'VERB' and t.get('dep') == 'ROOT'), None)
        if _verb_root:
            root_tok = _verb_root

    print(f"DEBUG root_tok flags APRES: "
          f"is_participe_passe={root_tok.get('is_participe_passe') if root_tok else None}")

    # ── PROGRESSIF : en train de ─────────────────────────────────────────────
    # Idiotisme français: "être + en train de + VERB"
    # spaCy tokenise: train=ROOT, manger=acl (enfant de train)
    # Détection syntaxique: NOUN(train) + ADP + VERB(acl) → VERB devient root
    if (root_tok and root_tok.get('pos') == 'NOUN'
            and root_tok.get('lemma') == 'train'):
        # Chercher ADP enfants ('en', 'de', etc.)
        _has_en_train_pattern = any(
            x.get('pos') == 'ADP'
            and x.get('head_index') == root_tok['orig_index']
            for x in T)
        if _has_en_train_pattern:
            # Chercher VERB acl enfant du NOUN 'train'
            _prog_verb = next((x for x in T
                              if x.get('pos') == 'VERB'
                              and x.get('dep') == 'acl'
                              and x.get('head_index') == root_tok['orig_index']), None)
            if _prog_verb and _prog_verb.get('tense') == 'prog':
                # Reassigner le vrai verbe comme root_tok
                root_tok = _prog_verb
                # Marquer 'train' comme expletif
                _expletive_train = next((x for x in T if x.get('lemma') == 'train'
                                        and x.get('dep') == 'ROOT'), None)
                if _expletive_train:
                    _expletive_train['role'] = 'expletive'

    # ── COPULE AVEC ADVCL : fallback pour "être + en train de + V" non capturé ──
    # Si root_tok est une copule (est/être) avec advcl/xcomp enfant VERB,
    # utiliser ce verbe comme root_tok (ex: "il est en train de se laver")
    if root_tok and _is_copula(root_tok, T):
        _advcl_verb = next((t for t in T
                           if t.get('pos') == 'VERB'
                           and t.get('dep') in ('advcl', 'xcomp')
                           and t.get('head_index') == root_tok['orig_index']), None)
        if _advcl_verb:
            root_tok = _advcl_verb
    # Fallback : si root_tok est None, chercher copule avec enfant VERB acl/xcomp
    elif not root_tok:
        _copula_tok = next((t for t in T
                           if _is_copula(t, T) and t.get('dep') in ('ROOT', 'cop')), None)
        if _copula_tok:
            _advcl_verb = next((t for t in T
                               if t.get('pos') == 'VERB'
                               and t.get('dep') in ('advcl', 'xcomp')
                               and t.get('head_index') == _copula_tok['orig_index']), None)
            if _advcl_verb:
                root_tok = _advcl_verb

    # ── XCOMP ─────────────────────────────────────────────────────────────────
    # Exclure les tokens internes à une relative (acl:relcl) : un xcomp DANS une
    # relative (ex: 'maintenir' dans 'qui veulent maintenir…') n'est pas le xcomp
    # de la clause principale — il appartient à la relative.
    from rules.core import relcl_subtree_indices
    _relcl_idx = relcl_subtree_indices(T, NX_G)
    xcomp_verb_tok = next((t for t in T
                           if t.get('dep') == 'xcomp' and t.get('pos') == 'VERB'
                           and t.get('orig_index') not in _relcl_idx), None)
    # Fallback : V + à/de/pour + V-infinitif (dep='fixed' ou 'advcl')
    # ex: "pousse à partir", "aide à construire", "hésite à partir"
    # → même traitement que xcomp (verb_serial : V ka V)
    if not xcomp_verb_tok:
        _root_idx_x = root_tok['orig_index'] if root_tok else -1
        _has_linking_adp = any(
            t.get('pos') == 'ADP'
            and t.get('dep') in ('advmod', 'mark', 'case', 'fixed')
            for t in T)
        if _has_linking_adp:
            xcomp_verb_tok = next((
                t for t in T
                if t.get('pos') == 'VERB'
                and t.get('dep') in ('fixed', 'advcl')
                and t.get('orig_index') != _root_idx_x
                and t.get('orig_index') not in _relcl_idx
                and 'VerbForm=Inf' in str(t.get('morph', ''))
                # Exclure les advcl PURPOSIFS (pour cuisiner) et PRIVATIFS
                # (sans avertir) : ce ne sont pas des sérialisations (V ka V)
                # mais des subordonnées (walasa ka V / k'a sɔrɔ S ma V),
                # traitées par advcl.handle. 'aide à', 'commence à' (mark non
                # purposif/privatif) restent des verbes sériels.
                and not any(m.get('role') in ('purposive', 'privative')
                            and m.get('head_index') == t.get('orig_index')
                            for m in T)
            ), None)
    xcomp_adj_tok  = next((t for t in T
                           if t.get('dep') == 'xcomp'
                           and t.get('pos') in ('ADJ', 'NOUN', 'PROPN')
                           and t.get('pos') != 'VERB'), None)

    # ── ROLES depuis KG ───────────────────────────────────────────────────────
    _neg_surfaces    = G_kg.get('neg_surfaces', set())
    _expletive_roles = G_kg.get('expletive_roles', {'expletive'})
    _clitic_roles    = G_kg.get('clitic_roles', {'clitic'})
    _relative_roles  = G_kg.get('relative_roles', {'relative'})

    # ── CLAUSE TYPE INIT ──────────────────────────────────────────────────────
    clause_type_init = 'simple'
    if xcomp_verb_tok:
        clause_type_init = 'verb_serial'

    _has_question_mark      = False
    _has_interrogative_word = False
    for x in T:
        _surf  = str(x.get('surface', '')).strip()
        _morph = str(x.get('morph', ''))
        if ((x.get('role') == 'interrogative' and x.get('dep') != 'mark')
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
                          and (x.get('dep') == 'ROOT' or x.get('is_root')) for x in T)
    _has_propn_root = any(x.get('pos') == 'PROPN'
                          and x.get('dep') == 'ROOT' for x in T)

    # PROPN dep==ROOT + cop + PAS de sujet explicite → identificatoire
    # C'est Moussa → Moussa dòn  (expletif ce → pas de vrai sujet)
    # Je suis Hawa → n yé Hawa yé (sujet explicite je → equatif)
    _has_cop = any(x.get('dep') == 'cop' for x in T)
    _propn_root_tok = next((x for x in T
                            if x.get('pos') == 'PROPN'
                            and x.get('dep') == 'ROOT'), None)
    _has_real_subj = any(
        x.get('dep') in ('nsubj', 'nsubj:pass')
        and x.get('role') not in ('expletive', 'clitic')
        and x.get('pos') == 'PRON'
        and str(x.get('surface', '')).lower() not in ('ce', "c'", 'c')
        for x in T)
    if _propn_root_tok and _has_cop and not _has_real_subj:
        clause_type_init = 'identificatory'

    elif _has_expletive and (_has_pron_root or _has_propn_root):
        _has_adj_any = any(x.get('pos') == 'ADJ' for x in T)
        if not _has_adj_any:
            clause_type_init = 'identificatory'
        else:
            _adj_attr = next((x for x in T if x.get('pos') == 'ADJ'), None)
            if _adj_attr and _adj_attr.get('is_valeur') is True:
                clause_type_init = 'equative'
                root_tok = _adj_attr

    print(f"DEBUG: root_tok={root_tok}, _has_expletive={_has_expletive}, "
          f"_has_pron_root={_has_pron_root}, _has_propn={_has_propn}")

    return {
        'NX_G':             NX_G,
        'root_tok':         root_tok,
        'xcomp_verb_tok':   xcomp_verb_tok,
        'xcomp_adj_tok':    xcomp_adj_tok,
        'clause_type_init': clause_type_init,
        '_has_expletive':   _has_expletive,
        '_has_propn':       _has_propn,
        '_has_pron_root':   _has_pron_root,
        '_has_propn_root':  _has_propn_root,
        '_has_question_mark': _has_question_mark,
        '_neg_surfaces':    _neg_surfaces,
        '_expletive_roles': _expletive_roles,
        '_clitic_roles':    _clitic_roles,
        '_relative_roles':  _relative_roles,
    }