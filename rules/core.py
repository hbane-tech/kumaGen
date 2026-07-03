"""
rules/core.py
Fonctions utilitaires, constantes TAM et détecteurs partagés.
Importé par tous les sous-modules.

Toutes les valeurs linguistiques bambara viennent du KG via rule_engine.
Les constantes ci-dessous sont des FALLBACKS d'urgence (KG non chargé).
"""
import networkx as nx

# ── HELPER ────────────────────────────────────────────────────────────────────
def j(*p):
    return ' '.join(str(x) for x in p if x and str(x).strip() and str(x).lower() != 'null')


def apply_affixes(bm_prefix, bm, bm_suffix):
    """Assemble un token selon le schéma slot bm_prefix - bm - bm_suffix.
    Chaque slot est optionnel (None/'' accepté) — absent → omis, jamais de valeur par défaut."""
    return j(bm_prefix, bm, bm_suffix)


def tok_bm(tok):
    """Lit un token selon le schéma bm_prefix - bm - bm_suffix (tous optionnels)."""
    return apply_affixes(tok.get('bm_prefix'), tok.get('bm'), tok.get('bm_suffix'))


# ── CONSTANTES CHARGÉES DEPUIS LE KG PAR rule_engine ─────────────────────────
# Initialisées vides — le KG est l'unique source de vérité.
# rule_engine les surcharge au démarrage via _load_grammar().

_GENITIVE_FALLBACK    = ''    # chargé depuis G_kg.get('genitive_marker')
_ADJ_EPITH_SUFFIX     = ''    # chargé depuis G_kg.get('adj_epithet_suffix')
_ADJ_EXCLUDE_SUFFIXES = ()    # chargé depuis MorphoRule KG 'adjective_epithet'
_AVOIR_LEMMAS         = set() # chargé depuis FunctionWord KG 'avoir_lemma'

# Classes sémantiques structurellement intransitives.
# Surchargé par rule_engine depuis TransformRule KG 'intrans_sc'.
INTRANS_SC = frozenset({
    'motion', 'biological', 'posture', 'spontaneous', 'meteorological',
    'saying', 'sound', 'communication', 'emission',
    'copula', 'having', 'stative_cognitive',
})

# Valeurs par défaut grammaire quand KG non chargé.
_GRAMMAR_FALLBACK = {
    'locative_markers':        set(),
    'temporal_markers':        set(),
    'temporal_roles':          set(),
    'genitive_marker':         _GENITIVE_FALLBACK,
    'resultative_marker':      '',
    'tam_default':             '',
    'quantifier_words':        {},
    'distributive_each':       {},
    'distributive_one':        {},
    'temporal_suffix_markers': set(),
    'privative_markers':       set(),
    'neg_surfaces':            set(),
    'expletive_roles':         {'expletive'},
    'clitic_roles':            {'clitic'},
    'relative_roles':          {'relative'},
}


def adj_man(bm: str, is_classifying: bool = False) -> str:
    """Forme épithète bambara : ADJ + suffix depuis KG (_ADJ_EPITH_SUFFIX).
    Si le bm est vide/fallback ou se termine déjà par un suffixe nominal, retourne tel quel.
    """
    if not bm or bm.startswith('['):
        return bm
    for _suffix in _ADJ_EXCLUDE_SUFFIXES:
        if bm.endswith(_suffix):
            return bm
    if is_classifying:
        return bm
    return bm + _ADJ_EPITH_SUFFIX


# ── DÉTECTEURS ────────────────────────────────────────────────────────────────
def _resolve_tam(tense: str, neg: bool, grammar: dict) -> str:
    kg = grammar.get('tam_table', {})
    if kg:
        return kg.get((tense, neg), '')
    # Fallback vide — forcer la correction KG plutôt que des bambara hardcodés
    return ''


def _is_copula(tok) -> bool:
    if not tok:
        return False
    _struct_cop = (tok.get('dep') == 'cop' or tok.get('role') == 'copula')
    if tok.get('semantic_class') == 'copula' and not _struct_cop:
        return False
    return (tok.get('semantic_class') == 'copula' or _struct_cop)


def _is_avoir(tok) -> bool:
    if not tok:
        return False
    _lemma = str(tok.get('lemma', '')).lower()
    _sc    = tok.get('semantic_class', '')
    # Trust LLM 'having' classification directly (posséder, détenir, etc.)
    if _sc == 'having':
        return True
    # 'copula' still requires the lemma to be in the KG-loaded set
    return _sc == 'copula' and _lemma in _AVOIR_LEMMAS


# ── CHUNK BORNÉ ───────────────────────────────────────────────────────────────
def get_bounded_chunk_tokens(head_idx, NX_G, processed_indices):
    if head_idx not in NX_G.nodes:
        return []
    descendants   = nx.descendants(NX_G, head_idx)
    all_indices   = descendants | {head_idx}
    valid_indices = all_indices - processed_indices
    final_indices = set()
    for idx in valid_indices:
        if idx not in NX_G.nodes:
            continue
        tok_item   = NX_G.nodes[idx]['token']
        surf_clean = str(tok_item.get('surface', '')).lower().strip()
        if surf_clean in ("d", "l", "'", "’", "«", "»"):
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
        if (idx != head_idx
                and tok_item.get('dep') == 'advmod'
                and (tok_item.get('role') in ('locative', 'temporal')
                     or tok_item.get('is_loc'))):
            continue
        final_indices.add(idx)
    _relcl_indices = set()
    for idx in final_indices:
        if idx not in NX_G.nodes:
            continue
        if NX_G.nodes[idx]['token'].get('dep') == 'acl:relcl':
            _relcl_indices.update(nx.descendants(NX_G, idx) | {idx})
    final_indices -= _relcl_indices
    return sorted([NX_G.nodes[idx]['token'] for idx in final_indices],
                  key=lambda x: x['orig_index'])


def relcl_subtree_indices(T, NX_G):
    rel = set()
    for t in T:
        idx = t.get('orig_index')
        if t.get('dep') == 'acl:relcl' and idx in NX_G.nodes:
            rel.update(nx.descendants(NX_G, idx) | {idx})
    return rel


# ── GÉNITIF RÉCURSIF ─────────────────────────────────────────────────────────
def _resolve_genitive_chain(head_tok, all_tokens, genitive_marker=None):
    _gen = genitive_marker or _GENITIVE_FALLBACK
    nmod = next((t for t in all_tokens
                 if t.get('dep') == 'nmod'
                 and t.get('head_index') == head_tok['orig_index']), None)
    possessif = next((t for t in all_tokens
                      if t.get('dep') == 'det'
                      and t.get('role') in ('pronoun', 'possessive')
                      and t.get('head_index') == head_tok['orig_index']), None)
    head_bm = head_tok.get('bm', '')
    if nmod:
        nmod_resolved = _resolve_genitive_chain(nmod, all_tokens, _gen)
        nmod_case = next((t for t in all_tokens
                          if t.get('dep') == 'case'
                          and t.get('head_index') == nmod['orig_index']), None)
        if nmod_case:
            return j(nmod_resolved, _gen, head_bm)
        return j(nmod_resolved, head_bm)
    elif possessif:
        return j(possessif.get('bm', ''), head_bm)
    else:
        return head_bm
