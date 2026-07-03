"""
rules/build_tree.py
Orchestrateur principal : appelle les steps 0→7 en séquence.

Chaque step reçoit des arguments explicites et modifie
tree / m / processed_indices en place (sauf step2 qui peut retourner
un résultat anticipé pour les relatives nominales).
"""
import networkx as nx
from rules.core import (j, _GRAMMAR_FALLBACK, _resolve_tam,
                        _is_copula, _is_avoir, get_bounded_chunk_tokens)
from rules.steps import step0_init
from rules.steps import step1_avoir
from rules.steps import step2_sujet
from rules.steps import step3_verbe
from rules.steps import step4_objet       # package
from rules.steps import step5_obliques    # package
from rules.steps import step6_copule      # package
from rules.steps import step7_final
from rules.steps import step_impersonal
from rules.kg_gateway import apply_kg_rules, apply_kg_semantic_behaviors, apply_kg_patterns_early


def build_tree(tokens, db=None, grammar=None):
    G_kg = grammar or {}
    T    = tokens

    # ══════════════════════════════════════════════════════════════
    # STEP 0 : Initialisation — graph NX, root_tok, clause_type
    # ══════════════════════════════════════════════════════════════
    ctx = step0_init.run(T, G_kg)
    NX_G             = ctx['NX_G']
    root_tok         = ctx['root_tok']
    xcomp_verb_tok   = ctx['xcomp_verb_tok']
    xcomp_adj_tok    = ctx['xcomp_adj_tok']
    clause_type_init = ctx['clause_type_init']
    _has_expletive   = ctx['_has_expletive']
    _has_question_mark = ctx['_has_question_mark']
    _neg_surfaces    = ctx['_neg_surfaces']
    _expletive_roles = ctx['_expletive_roles']
    _clitic_roles    = ctx['_clitic_roles']
    _relative_roles  = ctx['_relative_roles']

    # ══════════════════════════════════════════════════════════════
    # TREE INIT
    # ══════════════════════════════════════════════════════════════
    tree = {
        'clause_type': clause_type_init,
        'tam':         G_kg.get('tam_default', 'bɛ'),
        'tense':       'pres',
        'neg':         False,
        'main': {
            'S': '', 'V': '', 'V_ACTION': '', 'V_SUFFIX': '',
            'O': '', 'O_COORD': '', 'QUAL': '', 'ADV': '',
            'AGENT': '', 'IOBJ': None, 'LOC': None,
            'OBL_ALL': [], 'CCOMP': '', 'O_IS_XCOMP': False, 'SLOTS': {}
        }
    }
    m = tree['main']
    processed_indices = set()
    tree['_has_question_mark'] = _has_question_mark

    # ══════════════════════════════════════════════════════════════
    # KG EARLY : PatternRule → clause_type avant step1
    # Les steps ne font que remplir les slots (S/O/V/TAM/OBL_ALL).
    # ══════════════════════════════════════════════════════════════
    apply_kg_patterns_early(tree, T, G_kg)

    # Guard : ROOT NOUN/ADJ + clause_type temporelle/causale/conditionnelle
    # → le marqueur subordonnant est un modificateur du NOUN, pas la clause principale.
    # "Malheur quand il cligne" → root=malheur(NOUN), clause_type=temporal (LLM).
    # Résoudre en 'simple' : le NOUN devient S, l'advcl devient OBL temporel.
    _SUBORD_TYPES = {'temporal', 'causal', 'conditional', 'concessive'}
    if (root_tok and root_tok.get('pos') in ('NOUN', 'ADJ')
            and not _is_copula(root_tok)
            and not _is_avoir(root_tok)
            and tree.get('clause_type') in _SUBORD_TYPES):
        tree['clause_type'] = 'simple'
        clause_type_init = 'simple'

    # Closure locale pour get_bounded_chunk_tokens (capture NX_G, processed_indices)
    def _gbc(head_idx):
        return get_bounded_chunk_tokens(head_idx, NX_G, processed_indices)

    # ══════════════════════════════════════════════════════════════
    # STEP 1 : Avoir — possession, il y a, existential
    # ══════════════════════════════════════════════════════════════
    root_tok = step1_avoir.run(T, tree, m, processed_indices, G_kg, root_tok)

    # ══════════════════════════════════════════════════════════════
    # STEP 2 : Sujet — nsubj, expletif, verrou interrogatif,
    #                   relative nominale (retour anticipé)
    # ══════════════════════════════════════════════════════════════
    result2 = step2_sujet.run(
        T, tree, m, processed_indices, G_kg, NX_G, root_tok,
        clause_type_init, _has_expletive,
        _expletive_roles, _clitic_roles, _relative_roles)

    subj_tok, root_noun, has_acl, _has_relcl, _rel_str_nominal = result2

    # Retour anticipé : relative nominale (ex: "l'homme qui est parti")
    if _rel_str_nominal is not None:
        tree['final_string']      = _rel_str_nominal
        tree['local_clause_type'] = 'relative_nominal'
        tree['_tokens']           = T
        return tree

    # ══════════════════════════════════════════════════════════════
    # STEP IMPERSONNEL : il faut/doit, il s'agit de, il arrive,
    #                    il semble, il manque, il reste
    # ══════════════════════════════════════════════════════════════
    _imp_str = step_impersonal.run(T, tree, G_kg, root_tok)
    if _imp_str is not None:
        tree['final_string']      = _imp_str
        tree['clause_type']       = 'impersonal'
        tree['local_clause_type'] = 'impersonal'
        tree['_tokens']           = T
        return tree

    # ══════════════════════════════════════════════════════════════
    # KG PRE-ANNOTATION : comportements sémantiques KG → annotés sur root_tok
    # Avant step3 pour que les overrides KG soient disponibles dans les steps.
    # Remplace progressivement les conditions hardcodées (semantic_class=='biological'…)
    # ══════════════════════════════════════════════════════════════
    apply_kg_semantic_behaviors(tree, T, G_kg)

    # ══════════════════════════════════════════════════════════════
    # STEP 3 : Verbe ROOT — m['V'], xcomp, négation, prohibitif,
    #           impératif, infinitif, F1 tense, déjà, participial_to,
    #           privatif, ADJ/NOUN ROOT + copule
    # ══════════════════════════════════════════════════════════════
    aux_tense_tok = step3_verbe.run(
        T, tree, m, processed_indices, G_kg, NX_G,
        root_tok, xcomp_verb_tok, _neg_surfaces)

    # ══════════════════════════════════════════════════════════════
    # STEP 4 : Objet — noun_phrase, objet_standard, relcl_post,
    #           ownership
    # Signature : step4_objet.__init__.run(...)
    # ══════════════════════════════════════════════════════════════
    step4_objet.run(
        T, tree, m, processed_indices, G_kg, NX_G,
        root_tok, root_noun, has_acl, _has_relcl,
        xcomp_adj_tok, clause_type_init, _has_question_mark,
        _is_copula)   # fonction détectrice passée en argument

    # ══════════════════════════════════════════════════════════════
    # STEP 5 : Obliques — advcl, relcl_boucle, locatif, comitative,
    #           privatif, temporel, wagon (compound/démo/loc_nmod)
    # Signature : step5_obliques.__init__.run(...)
    # ══════════════════════════════════════════════════════════════
    step5_obliques.run(T, tree, m, processed_indices, G_kg, NX_G, root_tok)

    # ══════════════════════════════════════════════════════════════
    # STEP 6 : Copule / être — participial_to, etre_root,
    #           presentative, statif, participes (résultatif + potential),
    #           équatif, qualitative, identificatoire
    # Signature : step6_copule.__init__.run(...)
    # ══════════════════════════════════════════════════════════════
    step6_copule.run(
        T, tree, m, processed_indices, G_kg, root_tok,
        _has_expletive, aux_tense_tok, clause_type_init)

    # ══════════════════════════════════════════════════════════════
    # STEP 7 : Finalisation — harmonisation nominale, ccomp,
    #           intransitif kɛ, déduplication wagons, slots finaux
    # ══════════════════════════════════════════════════════════════
    tree = step7_final.run(
        T, tree, m, processed_indices, G_kg, NX_G, root_tok,
        get_bounded_chunk_tokens_fn=_gbc)

    # ══════════════════════════════════════════════════════════════
    # KG GATEWAY : après step7, les règles KG prennent la décision finale.
    # Les steps ont extrait les features ; le KG choisit la construction.
    # Règles KG priorité > 60 → override décisions Python.
    # ══════════════════════════════════════════════════════════════
    tree['_processed_indices'] = processed_indices
    tree = apply_kg_rules(tree, T, G_kg)

    # Post-KG guard : ROOT NOUN/ADJ + clause_type subordonnant
    # → reset à 'simple' en préservant S et l'OBL temporel construit par advcl.handle.
    # Ce guard court APRÈS apply_kg_rules pour neutraliser les PatternRules temporelles
    # qui s'appliquent sur le marqueur subordonnant (quand/si/parce que) même quand
    # le ROOT est un nom (ex: "Malheur quand il cligne les yeux").
    _SUBORD_CT = {'temporal', 'causal', 'conditional', 'concessive'}
    if (root_tok and root_tok.get('pos') in ('NOUN', 'ADJ')
            and not _is_copula(root_tok)
            and not _is_avoir(root_tok)
            and tree.get('clause_type') in _SUBORD_CT):
        tree['clause_type'] = 'simple'
        _m = tree.get('main', {})
        # Supprimer les slots (X1/X2/X3) injectés par le template subordonnant.
        _m.pop('SLOTS', None)
        # Supprimer le temporal_marker (évite que le renderer le préfixe au résultat).
        tree.pop('temporal_marker', None)
        # L'objet du verbe advcl a été affecté à m['O'] par apply_kg_rules.
        # Ce n'est pas l'objet de la clause principale — vider O.
        # advcl verbes avec marqueur subordonnant (SCONJ ou role temporal/causal/…)
        _subord_advcl_idxs = {
            a['orig_index'] for a in T
            if a.get('dep') == 'advcl'
            and any(mk.get('dep') == 'mark'
                    and (mk.get('role') in ('temporal','causal','conditional','concessive')
                         or mk.get('pos') == 'SCONJ')
                    and mk.get('head_index') == a['orig_index']
                    for mk in T)
        }
        _advcl_bms = {
            t.get('bm') for t in T
            if t.get('dep') == 'obj'
            and t.get('head_index') in _subord_advcl_idxs
        }
        _plur = G_kg.get('plural_noun_suffix', '') or 'w'
        _advcl_bms |= {b + _plur for b in _advcl_bms if b}
        if _m.get('O') in _advcl_bms:
            _m['O'] = ''

    return tree
