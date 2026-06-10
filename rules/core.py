"""
rules/core.py
Fonctions utilitaires, constantes TAM et détecteurs partagés.
Importé par tous les sous-modules.
"""
import networkx as nx

# ── HELPER ────────────────────────────────────────────────────────────────────
def j(*p):
    return ' '.join(str(x) for x in p if x and str(x).strip() and str(x).lower() != 'null')


def adj_man(bm: str) -> str:
    """Forme épithète bambara : ADJ + man. Si le bm est vide/fallback ou se termine
    déjà par un suffixe nominal (-man, -len, -nen, -nin), on le retourne tel quel."""
    if not bm or bm.startswith('['):
        return bm
    for _suffix in ('man', 'len', 'nen', 'nin', 'ya', 'la'):
        if bm.endswith(_suffix):
            return bm
    return bm + 'man'


# ── CONSTANTES ────────────────────────────────────────────────────────────────
_GENITIVE_FALLBACK = 'ka'

# Classes sémantiques structurellement intransitives en bambara.
# Un verbe dans l'une de ces classes n'utilise PAS la forme V+li kɛ sans COD.
INTRANS_SC = frozenset({
    'motion', 'biological', 'posture', 'spontaneous', 'meteorological',
    'perception', 'saying', 'sound', 'communication', 'emission',
    'copula', 'having',
})

_TAM_HARDCODED = {
    ('pres',  False): 'bɛ',     ('pres',  True):  'tɛ',
    ('past',  False): 'yé',     ('past',  True):  'ma',
    ('qualitative_pres', True): 'mán',
    ('fut',   False): 'bɛ na',  ('fut',   True):  'tɛ na',
    ('cond',  False): 'bɛ na',  ('cond',  True):  'tɛ na',
    ('imp',   False): '',       ('imp',   True):  '',
    ('prog',  False): 'bɛ kà',  ('prog',  True):  'tɛ kà',
    ('hab',   False): 'tùn bɛ', ('hab',   True):  'tùn tɛ',
    ('plup',  False): 'tùn yé', ('plup',  True):  'tùn ma',
}

_GRAMMAR_FALLBACK = {
    'locative_markers':        set(),
    'temporal_markers':        set(),
    'temporal_roles':          set(),
    'genitive_marker':         'ka',
    'demonstrative_suffix':    'in',
    'resultative_marker':      'ye',
    'tam_default':             'bɛ',
    'quantifier_words':        {},
    'temporal_suffix_markers': set(),
    'privative_markers':       set(),
    'neg_surfaces':            set(),
    'expletive_roles':         {'expletive'},
    'clitic_roles':            {'clitic'},
    'relative_roles':          {'relative'},
}


# ── DÉTECTEURS ────────────────────────────────────────────────────────────────
def _resolve_tam(tense: str, neg: bool, grammar: dict) -> str:
    kg = grammar.get('tam_table', {})
    if kg:
        return kg.get((tense, neg), _TAM_HARDCODED.get((tense, neg), ''))
    return _TAM_HARDCODED.get((tense, neg), '')


def _is_copula(tok) -> bool:
    if not tok:
        return False
    return (tok.get('semantic_class') == 'copula'
            or tok.get('dep') == 'cop'
            or tok.get('role') == 'copula')


# Lemmes francais du verbe avoir (garde double)
_AVOIR_LEMMAS = {'avoir'}


def _is_avoir(tok) -> bool:
    # Verifie semantic_class='having' ET lemme='avoir'
    # Evite que boire/recruter... mal classes declenchent la possession
    if not tok:
        return False
    _lemma = str(tok.get('lemma', '')).lower()
    _sc    = tok.get('semantic_class', '')
    return _sc == 'having' and _lemma in _AVOIR_LEMMAS


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
    # Exclure acl:relcl et leurs descendants
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
    """Indices de TOUS les tokens internes à une relative (acl:relcl) : le verbe
    acl:relcl + tous ses descendants. Permet d'empêcher la détection des slots de
    la clause PRINCIPALE (xcomp en step0, objet en step4) d'aspirer des tokens
    qui appartiennent en réalité à une proposition relative imbriquée.
    """
    rel = set()
    for t in T:
        idx = t.get('orig_index')
        if t.get('dep') == 'acl:relcl' and idx in NX_G.nodes:
            rel.update(nx.descendants(NX_G, idx) | {idx})
    return rel


# ── GÉNITIF RÉCURSIF ─────────────────────────────────────────────────────────
def _resolve_genitive_chain(head_tok, all_tokens):
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