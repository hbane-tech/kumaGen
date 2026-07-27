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


def render_obl_entry(c: dict, G: dict) -> str:
    """Rend une entrée OBL_ALL (dict produit par wagon.append/relcl_post/etc.)
    en chaîne bambara. Logique partagée entre le renderer (rules/renderers/
    __init__.py, wagons obliques normaux) et tout step qui doit incorporer
    le même type de complément directement dans un autre slot (ex: m['O']
    d'un nom existentiel modifié par 'lié à X' — cf. step4_objet/noun_phrase.py)."""
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
        elif lct in ('locative', 'temporal'):
            if pref or suff:
                noun_base = j(pref, comp, head, suff)
                pref = suff = ''
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
        if c.get('MARKER_IS_PREFIX'):
            full_chunk = j(marker, full_chunk)
        elif lct == 'privative':
            full_chunk = full_chunk + marker
        else:
            full_chunk = j(full_chunk, marker)
    return full_chunk


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


def nominalize_verb(v_root: str, action_noun: str, grammar: dict) -> str:
    """Nominalise un verbe transitif/support sans COD (ex: manger sans objet
    → dumuni kɛ, pas *dúnli kɛ). Priorité au nom d'action irrégulier stocké
    dans le KG (Sense.action_noun) ; sinon suffixe régulier -li/-ni
    (nominalization_verb_suffix). Fonction centralisée : appelée partout où
    un verbe sans COD doit être nominalisé (step7_final, step_impersonal…)
    pour que la même règle grammaticale s'applique quel que soit l'endroit
    du pipeline où le verbe apparaît (ROOT, xcomp d'un modal comme 'falloir'…).
    """
    if not v_root:
        return v_root
    if action_noun:
        return action_noun
    _nom_sfx  = grammar.get('nominalization_verb_suffix', '')
    _nom_sfx2 = grammar.get('nominalization_verb_suffix_alt', '')
    if _nom_sfx and not (v_root.endswith(_nom_sfx)
                         or (_nom_sfx2 and v_root.endswith(_nom_sfx2))):
        return v_root + _nom_sfx
    return v_root


def _is_copula(tok) -> bool:
    if not tok:
        return False
    _struct_cop = (tok.get('dep') == 'cop' or tok.get('role') == 'copula')
    if tok.get('semantic_class') == 'copula' and not _struct_cop:
        return False
    return (tok.get('semantic_class') == 'copula' or _struct_cop)


def _is_misparsed_relative_qui(tok, T) -> bool:
    """'qui' (KG Pronoun{surface:'qui'} role='relative' — un seul sens en KG)
    n'est un vrai relatif QUE quand le verbe dont il est nsubj est une
    relative (dep='acl:relcl', modifiant un antécédent). S'il est nsubj du
    ROOT de la phrase (pas d'antécédent), c'est forcément l'interrogatif
    "qui" ("qui veut partir ?") — le morph PronType=Rel/Int de spaCy n'est
    pas fiable ici (spaCy tague parfois 'qui' sujet du ROOT en PronType=Rel
    malgré l'absence de toute relative). Utilisé identiquement par
    step0_init._has_interrogative_word et kg_gateway._build_features pour
    qu'elles ne divergent jamais (décision 2026-07-13, bug : 'qui veut
    partir ?' routé en 'interrogative' générique avec 'wà' au lieu de
    'content_question', faute de ce garde dans l'un des deux calculs).

    Exception : un fragment issu du split d'une relative après virgule
    ("Mali, qui s'enfonce...") perd son antécédent lors de la retokenisation
    isolée du segment — 'qui' redevient nsubj du ROOT sans acl:relcl. Le
    marqueur '_relative_continuation' (posé par translation_engine sur ce
    cas précis) empêche cette phrase d'être reclassée à tort en interrogative."""
    if tok.get('_relative_continuation'):
        return False
    if not (tok.get('pos') == 'PRON' and tok.get('dep') == 'nsubj'
            and tok.get('role') == 'relative'):
        return False
    _head = next((h for h in T if h.get('orig_index') == tok.get('head_index')), None)
    return _head is not None and _head.get('dep') == 'ROOT'


def _is_avoir_lexeme(tok) -> bool:
    """Vérifie STRICTEMENT le lemme (KG-sourced _AVOIR_LEMMAS), sans le
    branchement semantic_class=='having' de _is_avoir(). Utilisé par les
    guards structurels qui doivent justement distinguer "ce token est
    lexicalement avoir" de "le LLM l'a classé having" (ex: détecter si un
    verbe promu root_tok est un AUTRE verbe que 'avoir', alors que 'avoir'
    n'apparaît que comme son auxiliaire) — passer par _is_avoir() ici
    réintroduirait le bruit de classification que le guard cherche à éviter."""
    if not tok:
        return False
    return str(tok.get('lemma', '')).lower() in _AVOIR_LEMMAS


def _is_avoir(tok) -> bool:
    if not tok:
        return False
    _lemma = str(tok.get('lemma', '')).lower()
    _sc    = tok.get('semantic_class', '')
    # Le verbe "avoir" lui-même est toujours de confiance, indépendamment du
    # bruit du classificateur LLM. _AVOIR_LEMMAS vient du KG (FunctionWord
    # role='avoir_lemma'), pas d'un littéral Python — même source que le
    # branchement 'copula' juste en dessous.
    if _lemma in _AVOIR_LEMMAS:
        return True
    # 'having' (posséder, détenir, etc.) : la classification LLM seule n'est
    # PAS une garantie suffisante — un verbe absent du KG (aucun candidat,
    # ex: "valider") peut être mal classé 'having' par pur bruit du LLM sans
    # aucun ancrage lexical, ce qui route alors toute la clause vers
    # noun_phrase_have et fait disparaître le vrai verbe du rendu (bug trouvé
    # 2026-07-10 : "je valide ce riz" → "n irilen dòn"). On exige donc une
    # corroboration : le verbe doit avoir un candidat KG confirmé (bm non
    # vide et non un placeholder '[lemme]' — cf. translation_engine.py:1862,
    # qui pose ce placeholder exactement quand aucun candidat n'existe).
    # Un vrai verbe de possession (posséder, détenir) a normalement une
    # entrée KG et passe donc cette vérification sans problème.
    if _sc == 'having':
        _bm = str(tok.get('bm', '') or '')
        return bool(_bm) and not _bm.startswith('[')
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
