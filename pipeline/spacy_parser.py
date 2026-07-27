"""
pipeline/spacy_parser.py

Hybrid parser: spaCy (POS/dep/morph) + LLM (lemma) + KG (config).
All linguistic lists loaded from KG — zero hardcoded word lists.

Token dict keys:
  surface, lemma, pos, dep, tense, lang, role, bm, bm_marker,
  head_index, orig_index, is_neg, is_root, is_loc, is_plural,
  is_genitive, is_passive, is_agent, is_refl_past, is_dative,
  is_verbal_noun, modal
"""

import re
from typing import Optional
from config.settings import OLLAMA_GENERATE_URL
from utils.normalize import normalize_token
from llm.morphological_parser import MorphologicalParser

_nlp_cache = {}

# ── Classificateur SCONJ (temporal vs complémenteur) ─────────────────────────
_sconj_circuit_open_until: float = 0.0
_SCONJ_RETRY_GAP = 60


def _classify_sconj(surface: str, sentence: str, model: str = 'qwen2.5:3b') -> Optional[str]:
    """Demande au LLM si ce SCONJ mark est temporel ou complémenteur.
    Retourne 'TEMPORAL', 'COMPLETEUR', ou None si indisponible.
    """
    global _sconj_circuit_open_until
    import time as _t
    if _t.time() < _sconj_circuit_open_until:
        return None
    import requests
    prompt = (
        f"Dans la phrase : « {sentence} »\n"
        f"Le mot « {surface} » (conjonction de subordination) introduit-il :\n"
        f"A) une proposition temporelle/conditionnelle (quand, lorsque, si…) → réponds TEMPORAL\n"
        f"B) un complément de verbe (que après dire, penser, savoir…) → réponds COMPLETEUR\n"
        f"Réponds UNIQUEMENT par : TEMPORAL ou COMPLETEUR"
    )
    try:
        r = requests.post(
            OLLAMA_GENERATE_URL,
            json={'model': model, 'prompt': prompt, 'stream': False,
                  'options': {'temperature': 0, 'num_predict': 5}},
            timeout=8)
        raw = r.json().get('response', '').strip().upper()
        if 'TEMPORAL' in raw:
            return 'TEMPORAL'
        if 'COMPLETEUR' in raw or 'COMPLÉTEUR' in raw:
            return 'COMPLETEUR'
        return None
    except Exception:
        _sconj_circuit_open_until = _t.time() + _SCONJ_RETRY_GAP
        return None


# ── Désambiguïsation ADJ vs VERBE (court = adj 'court' / verbe 'courir') ──────
_adjverb_circuit_open_until: float = 0.0


def _classify_adj_verb(surface: str, sentence: str,
                       model: str = 'qwen2.5:3b') -> Optional[str]:
    """Quand une clause n'a aucun verbe, un mot tagué ADJ amod est peut-être en
    réalité le prédicat (le chien court = courir, pas l'adjectif 'court').
    Retourne l'infinitif du verbe si c'en est un, sinon None (adjectif ou
    LLM indisponible → on garde le tag ADJ).
    """
    global _adjverb_circuit_open_until
    import time as _t
    if _t.time() < _adjverb_circuit_open_until:
        return None
    import requests
    prompt = (
        f"Dans la phrase : « {sentence} »\n"
        f"Le mot « {surface} » est-il employé comme VERBE (une action) "
        f"ou comme ADJECTIF (une qualité) ?\n"
        f"Si c'est un VERBE, donne son infinitif.\n"
        f"Réponds UNIQUEMENT par : VERBE=<infinitif>  ou  ADJECTIF"
    )
    try:
        r = requests.post(
            OLLAMA_GENERATE_URL,
            json={'model': model, 'prompt': prompt, 'stream': False,
                  'options': {'temperature': 0, 'num_predict': 10}},
            timeout=8)
        raw = r.json().get('response', '').strip()
        m = re.match(r'\s*VERBE\W+([a-zà-ÿ]+)', raw, re.IGNORECASE)
        return m.group(1).lower() if m else None
    except Exception:
        _adjverb_circuit_open_until = _t.time() + 60
        return None


# ── Désambiguïsation imparfait/conditionnel/présent via le lexique LEFFF ────
# (faire/savoir/plaire/taire/traire : "il fait/sait/plaît/taît/trait"
# partagent une terminaison de surface avec l'imparfait "-ais/-ait" mais sont
# au présent). Le lexique LEFFF (Sagot 2010, via spacy-lefff) donne le code
# morphologique exact de chaque forme fléchie : aucune collision I/P
# n'existe dans tout le lexique pour les formes -ais/-ait, donc pas besoin
# de liste de verbes ni d'appel LLM — une consultation de dictionnaire
# suffit à trancher.
_lefff_verb_codes_cache: Optional[dict] = None


def _lefff_verb_codes() -> dict:
    """Charge (une fois) form -> {codes morphologiques LEFFF} pour cat='v'."""
    global _lefff_verb_codes_cache
    if _lefff_verb_codes_cache is None:
        d: dict = {}
        try:
            import io
            from spacy_lefff.lefff import DATA_DIR, LEFFF_FILE_NAME
            path = f"{DATA_DIR}/{LEFFF_FILE_NAME}"
            with io.open(path, encoding='utf-8') as f:
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 4 or parts[1] != 'v':
                        continue
                    d.setdefault(parts[0].lower(), set()).add(parts[3])
        except Exception:
            d = {}
        _lefff_verb_codes_cache = d
    return _lefff_verb_codes_cache


def _lefff_tense(surface: str) -> Optional[str]:
    """Tranche le temps d'une forme verbale via le code morphologique LEFFF
    (I=imparfait, C=conditionnel, F=futur, J=passé simple). Retourne None si
    la forme est absente du lexique ou ne correspond à aucun de ces temps
    (présent, impératif, participe... laissés au reste de _tense).

    Garde anti-collision (bug 2026-07-07) : contrairement à I/P (aucune
    collision pour les formes -ais/-ait, cf. note plus haut), J (passé
    simple) ET P (présent) coexistent réellement pour certains verbes
    irréguliers — ex: "dit" (dire) a les codes {P3s, J3s, Kms} : présent
    ("il dit bonjour") ET passé simple ("il dit alors...") partagent la
    même forme de surface. Sans cette garde, "il dit" (présent, aucun
    auxiliaire) était systématiquement tranché 'past', écrasant le
    Tense=Pres pourtant correctement déterminé par spaCy depuis le contexte
    (repli ligne ~261). Si 'P' est aussi un code valide, l'ambiguïté est
    réelle → laisser le repli contextuel trancher plutôt que ce lexique
    hors-contexte."""
    codes = _lefff_verb_codes().get(surface.lower())
    if not codes:
        return None
    leadings = {c[0] for c in codes if c}  # Skip empty strings
    if 'P' in leadings:
        return None
    for prefix, tense in (('I', 'hab'), ('C', 'fut'), ('F', 'fut'), ('J', 'past')):
        if prefix in leadings:
            return tense
    return None


_lefff_verb_lemma_cache: Optional[dict] = None


def _lefff_verb_lemmas() -> dict:
    """Charge (une fois) form -> {lemmes LEFFF} pour cat='v'."""
    global _lefff_verb_lemma_cache
    if _lefff_verb_lemma_cache is None:
        d: dict = {}
        try:
            import io
            from spacy_lefff.lefff import DATA_DIR, LEFFF_FILE_NAME
            path = f"{DATA_DIR}/{LEFFF_FILE_NAME}"
            with io.open(path, encoding='utf-8') as f:
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 4 or parts[1] != 'v':
                        continue
                    d.setdefault(parts[0].lower(), set()).add(parts[2].lower())
        except Exception:
            d = {}
        _lefff_verb_lemma_cache = d
    return _lefff_verb_lemma_cache


def _lefff_lemma(surface: str) -> Optional[str]:
    """Lemme infinitif d'une forme verbale via LEFFF. Retourne None si la
    forme est absente du lexique ou ambiguë entre plusieurs lemmes (rare :
    ~0.15% des formes verbales) — dans ces cas, on laisse spaCy/LLM décider.
    Corrige des cas où spaCy mélabelle le lemme (ex: 'lave' → lemme 'lave'
    au lieu de 'laver', probablement par confusion avec le nom 'lave' (la
    roche volcanique), un homographe exact)."""
    lemmas = _lefff_verb_lemmas().get(surface.lower())
    if lemmas and len(lemmas) == 1:
        return next(iter(lemmas))
    return None


import spacy

_nlp_cache = {}

def _get_nlp():
    """Charge et met en cache le parseur spaCy français — seule langue
    supportée par le pipeline (grammaire des règles + lexique KG français)."""
    if 'fr' not in _nlp_cache:
        # Liste triée de la précision maximale vers la vitesse maximale
        for model_name in ("fr_dep_news_trf", "fr_core_news_lg", "fr_core_news_md", "fr_core_news_sm"):
            try:
                _nlp_cache['fr'] = spacy.load(model_name)
                break
            except Exception:
                continue
        else:
            _nlp_cache['fr'] = None
    return _nlp_cache['fr']


# def _tense(tok):
#     # 1. Mode first (Mood)
#     mood = str(tok.morph.get('Mood')).lower()
#     if 'imp' in mood:
#         return 'imp'

#     # 2. Tense
#     tense = str(tok.morph.get('Tense')).lower()
#     if 'imp' in tense:
#         return 'hab'
#     if 'past' in tense:
#         return 'past'
#     if 'fut' in tense:
#         return 'fut'

#     return 'pres'

# APRÈS
def _tense(tok):
    # 1. Mode first (Mood)
    mood = str(tok.morph.get('Mood')).lower()
    if 'imp' in mood:
        return 'imp'
    # Conditionnel → traité comme futur en bambara
    if 'cnd' in mood:
        return 'fut'
    # Subjonctif → optatif bambara (S ka V)
    if 'sub' in mood:
        return 'sub'

    # 2. Lexique LEFFF (forme fléchie exacte -> code morphologique) :
    # autorité prioritaire sur le Tense de spaCy, qui mélabelle régulièrement
    # aussi bien l'imparfait lui-même (Tense=Pres pour "lavais") que les
    # présents irréguliers homographes de l'imparfait (Tense=Imp pour
    # "tait"/"taît"/"sait"/"fait" — faire/savoir/plaire/taire/traire).
    if tok.pos_ in ('VERB', 'AUX'):
        _lefff_t = _lefff_tense(tok.text)
        if _lefff_t:
            return _lefff_t

    # 3. Tense (repli si la forme est absente du lexique LEFFF)
    tense = str(tok.morph.get('Tense')).lower()
    if 'imp' in tense:
        return 'hab'
    if 'past' in tense:
        return 'past'
    if 'fut' in tense:
        return 'fut'

    return 'pres'


def _merge_causative_faire(tokens, db):
    """
    Fusionne 'faire' + infinitif adjacent en un seul token quand le KG a une
    entrée composée dédiée (ex: 'faire grossir.' → 'lábònya', un verbe
    causatif préfixé lá-, distinct de 'grossir' seul → 'bònya').

    Le causatif français "faire + Vinf" produit régulièrement un arbre de
    dépendances incohérent chez spaCy (l'infinitif se retrouve attaché comme
    'conj' de la racine de la phrase plutôt que comme complément de 'faire',
    et 'faire' lui-même reçoit un dep vague comme 'dep' plutôt qu'une
    relation causative reconnue) — bug trouvé 2026-07-19 sur "ceci m'a fait
    grossir". Plutôt que de réparer l'arbre de dépendances brisé, on détecte
    la paire par adjacence linéaire (lemme='faire' suivi immédiatement d'un
    VERB à l'infinitif) et on interroge le KG pour un sens composé dédié
    AVANT que la construction de clause ne s'appuie sur les rattachements
    erronés du parseur.
    """
    _causative_rows = db.query(
        "MATCH (fw:FunctionWord {lang:'fr', role:'causative_aux'}) "
        "RETURN DISTINCT fw.surface AS s")
    _CAUSATIVE_LEMMAS = {r['s'].lower() for r in _causative_rows if r.get('s')}
    if not _CAUSATIVE_LEMMAS:
        return tokens
    if not any(t.get('lemma', '').lower() in _CAUSATIVE_LEMMAS
               and t.get('pos') in ('VERB', 'AUX') for t in tokens):
        return tokens

    survivors = []
    old_to_survivor_old = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        _next = tokens[i + 1] if i + 1 < len(tokens) else None
        if (t.get('lemma', '').lower() in _CAUSATIVE_LEMMAS
                and t.get('pos') in ('VERB', 'AUX')
                and _next is not None
                and _next.get('pos') == 'VERB'
                and 'VerbForm=Inf' in str(_next.get('morph', ''))):
            _inf_lemma = _next.get('lemma', '')
            _causative_lemma = t.get('lemma', '').lower()
            _res = db.query(
                "MATCH (s:Sense) WHERE toLower(s.fr) STARTS WITH toLower($prefix) "
                "RETURN s.bm AS bm, s.fr AS fr LIMIT 1",
                {'prefix': f"{_causative_lemma} {_inf_lemma}"})
            if _res and _res[0].get('bm'):
                t['bm'] = _res[0]['bm']
                t['sens_fr'] = _res[0].get('fr', '')
                t['pos'] = 'VERB'
                t['semantic_class'] = None  # laisser la classification normale se faire sur ce nouveau bm
                # Le causatif "faire + Vinf" laisse souvent le parseur avec un
                # arbre incohérent : l'infinitif (maintenant absorbé) portait
                # parfois le seul dep clair, tandis que 'faire' hérite d'un
                # dep vague ('dep') et qu'un autre token (souvent le sujet
                # démonstratif/pronominal, ex: "ceci") reste seul avec
                # dep='ROOT'. On ne réutilise le dep de l'infinitif absorbé
                # que s'il est structurellement exploitable (pas 'conj', pas
                # 'ROOT' lui-même tant qu'on n'a pas vérifié le vrai sujet) ;
                # sinon on re-racine explicitement le token fusionné.
                _next_dep = _next.get('dep')
                if _next_dep not in ('conj', 'ROOT', None):
                    t['dep'] = _next_dep
                else:
                    _root_tok = next((x for x in tokens
                                      if x.get('dep') == 'ROOT'
                                      and x is not t and x is not _next), None)
                    if (_root_tok is not None
                            and _root_tok.get('pos') in ('PRON', 'NOUN', 'PROPN')):
                        _root_tok['dep'] = 'nsubj'
                        _root_tok['head_index'] = t['orig_index']
                        _root_tok['is_root'] = False
                        t['dep'] = 'ROOT'
                        t['is_root'] = True
                    elif t.get('dep') in ('dep', 'conj', None):
                        t['dep'] = 'ROOT'
                        t['is_root'] = True
                old_to_survivor_old[_next['orig_index']] = t['orig_index']
                survivors.append(t)
                i += 2
                continue
        survivors.append(t)
        i += 1

    if not old_to_survivor_old:
        return tokens

    full_old_map = {t['orig_index']: t['orig_index'] for t in survivors}
    full_old_map.update(old_to_survivor_old)
    old_to_new = {t['orig_index']: idx for idx, t in enumerate(survivors)}

    for t in survivors:
        _mapped_head = full_old_map.get(t['head_index'], t['head_index'])
        t['head_index'] = old_to_new.get(_mapped_head, t['head_index'])
        t['orig_index'] = old_to_new[t['orig_index']]
    return survivors


def resolve_auxiliary_lemmas(tokens, db):
    """
    Validation universelle et redressement des lemmes d'auxiliaires.
    """
    # Surfaces de clitiques objet FR (me/te/le/la/les/lui/leur/nous/vous +
    # formes élidées/toniques) — dérivées du KG Pronoun{role IN object*},
    # pas d'une liste Python séparée (décision 2026-07-13, audit hardcode-KG).
    _clitic_rows = db.query(
        "MATCH (n:Pronoun {lang:'fr'}) WHERE n.role IN ['object_pronoun', 'object'] "
        "RETURN DISTINCT n.surface AS s")
    _CLITIC_SURFS = {r['s'].lower() for r in _clitic_rows if r.get('s')}

    # Homographe 'suis' (être 1sg vs suivre) : surface KG-sourcée, pas un
    # littéral Python (décision 2026-07-13, audit hardcode-KG).
    _etre_suivre_rows = db.query(
        "MATCH (fw:FunctionWord {lang:'fr'}) WHERE fw.role = 'etre_suivre_homograph_surface' "
        "RETURN fw.bm AS b")
    _ETRE_SUIVRE_SURFS = {r['b'].lower() for r in _etre_suivre_rows if r.get('b')} or {'suis'}

    # Ancres de négation véritables ('ne'/"n'"/'pas'/'jamais'...) — servent à
    # valider les mots discontinus (KG FunctionWord.requires_partner, ex:
    # 'plus' pour 'ne...plus') plus bas : sans ancre co-occurrente, ces mots
    # gardent leur sens normal (ex: 'plus' comparatif/intensifieur dans
    # "toujours plus") au lieu d'être figés sur leur bm négatif.
    _neg_anchor_rows = db.query(
        "MATCH (fw:FunctionWord {role:'negation'}) "
        "WHERE fw.requires_partner IS NULL OR fw.requires_partner = false "
        "RETURN DISTINCT fw.surface AS s")
    _NEG_ANCHOR_SURFS = {r['s'].lower() for r in _neg_anchor_rows if r.get('s')}

    for t in tokens:
        # Ne pas normaliser les pronoms inversés pluriels
        if str(t.get('surface', '')).startswith('-') and t.get('is_plural'):
            continue

        surf_lower = t.get('surface', '').lower()
        if not surf_lower or surf_lower == 'none':
            continue
        lang_curr = t.get('lang', 'fr')

        # Correction spaCy : lemme erroné pour une forme verbale fléchie
        # (ex: "lave" → lemme 'lave' au lieu de 'laver', homographe du nom
        # 'lave' = roche volcanique). Autorité : LEFFF (lemme non-ambigu
        # uniquement) — appliqué avant la lemmatisation LLM (étape 3) pour
        # que celle-ci ne l'écrase pas si elle échoue/se trompe.
        if t.get('pos') in ('VERB', 'AUX'):
            _lefff_lem = _lefff_lemma(surf_lower)
            if _lefff_lem and _lefff_lem != t.get('lemma', '').lower():
                t['lemma'] = _lefff_lem
                t['_lefff_lemma_fixed'] = True

        # Correction spaCy : impératifs 1ère conjugaison mal lemmatisés
        # ex: "Donne" → lemma 'donne' au lieu de 'donner'. Détection :
        # Mood=Imp + lemme finit en 'e' (non 're') → tenter lemme + 'r' via KG.
        _lemma_curr = t.get('lemma', '')
        if (t.get('pos') == 'VERB'
                and 'Mood=Imp' in str(t.get('morph', ''))
                and _lemma_curr.endswith('e')
                and not _lemma_curr.endswith('re')):
            _inf_cand = _lemma_curr + 'r'
            _kg_check = db.query(
                "MATCH (n:Sense) WHERE n.fr = $fr RETURN n.bm AS bm LIMIT 1",
                {'fr': _inf_cand + '.'})
            if _kg_check and _kg_check[0].get('bm'):
                t['lemma'] = _inf_cand

        # Correction spaCy : 'suis' est parfois lemmatisé 'suivre' au lieu de 'être'
        # Règle : si ROOT ou cop sans obj direct → c'est 'être'
        if (surf_lower in _ETRE_SUIVRE_SURFS
                and t.get('pos') == 'VERB'
                and t.get('dep') in ('ROOT', 'cop')):
            _has_obj = any(
                tok.get('dep') == 'obj'
                and tok.get('head_index') == t.get('orig_index')
                for tok in tokens
            )
            if not _has_obj:
                t['lemma'] = 'être'
                t['bm'] = ''
                t['role'] = 'copula'
                t['semantic_class'] = 'copula'
                continue

        # res = db.query(
        #     "MATCH (n) WHERE n.lang = $lang AND n.surface = $surface "
        #     "RETURN n.lemma AS lemma, n.bm AS bm, labels(n) AS labels",
        #     {'lang': lang_curr, 'surface': surf_lower}
        # )

        # Clitique objet le/l'/la/les rattaché à un verbe (il l'a dit → a yé a
        # fɔ́ ; il les voit → … u …) : c'est un PRONOM objet, pas un article.
        # Désambiguïsation SYNTAXIQUE (dep=obj/iobj) car le/les sont ambigus
        # (article 'le chat' = det vs pronom 'il le voit' = obj). On préfère le
        # nœud Pronoun{role:'object_pronoun'} (bm le/l'/la→a, les→u) au nœud
        # Article. Les déterminants (dep=det) tombent dans la requête générale.
        # 'dep' catches "Dis lui"/"Donne lui"/"prends le" where spaCy mislabels
        # the clitic as dep+ADV (lui) or dep+PUNCT (le) in short imperatives.
        # spaCy mislabels postverbal clitics without a hyphen (e.g. "parle lui")
        # as ADV+advmod instead of PRON+dep; extend the guard to catch that case.
        _dep_is_likely_dative = (
            t.get('pos') in ('PRON', 'ADV', 'PUNCT')
            and any(x.get('dep') == 'ROOT' and x.get('pos') in ('VERB', 'AUX')
                    and x.get('head_index') == t.get('head_index')
                    for x in tokens)
            and (
                t.get('dep') == 'dep'
                or (t.get('dep') == 'advmod'
                    and surf_lower.lstrip('-') in _CLITIC_SURFS)
            )
        )
        if ((t.get('pos') in ('PRON', 'DET') and t.get('dep') in ('obj', 'iobj', 'dep'))
                or _dep_is_likely_dative):
            # Strip leading '-' for imperative postverbal clitics (aide-moi → moi)
            _surf_lookup = surf_lower.lstrip('-')
            _objp = db.query(
                "MATCH (n:Pronoun {lang:$lang}) "
                "WHERE toLower(n.surface) = $surface "
                "AND n.role IN ['object_pronoun', 'object'] "
                "RETURN n.bm AS bm, n.role AS role LIMIT 1",
                {'lang': lang_curr, 'surface': _surf_lookup})
            if _objp and _objp[0].get('bm'):
                t['bm']   = _objp[0]['bm']
                # Preserve KG role: 'object_pronoun'=accusatif (le/la/les),
                # 'object'=datif (lui/leur/me/te). Permet à step3 de distinguer
                # objet direct (→ m['O']) vs indirect (→ OBL_ALL avec ma/yé).
                t['role'] = _objp[0].get('role') or 'object_pronoun'
                t['pos']  = 'PRON'
                if t.get('dep') == 'advmod':
                    t['dep'] = 'dep'
                continue

        res = db.query(
            "MATCH (n) WHERE n.lang = $lang "
            "AND (n.surface = $surface OR n.lemma = $lemma) "
            "RETURN n.lemma AS lemma, n.bm AS bm, n.role AS role, "
            "n.semantic_class AS sc, n.bm_suffix AS bm_suffix, labels(n) AS labels, "
            "n.requires_partner AS requires_partner "
            "ORDER BY CASE WHEN n.role = 'demonstrative' AND $is_det THEN 0 ELSE 1 END "
            "LIMIT 1",
            {'lang': lang_curr, 'surface': surf_lower,
             'lemma': t.get('lemma', surf_lower).lower(),
             'is_det': t.get('dep') == 'det'}
        )


        if res and isinstance(res, list) and len(res) > 0:
            node_data = res[0]
            # Mot discontinu (ex: 'plus' pour 'ne...plus') sans ancre de
            # négation co-occurrente dans la phrase → ce n'est pas ce sens-là ;
            # ignorer entièrement ce nœud pour laisser le mot suivre son
            # traitement normal (comparatif/intensifieur, retrieval KG normal).
            if node_data.get('requires_partner'):
                _has_anchor = any(
                    str(ot.get('surface', '')).lower().rstrip("'").rstrip('’') in _NEG_ANCHOR_SURFS
                    for ot in tokens if ot is not t)
                if not _has_anchor:
                    continue
            # 'le/la/les' sont ambigus : article (dep=det) OU pronom clitique (dep=obj).
            # La requête générique retourne souvent le nœud Pronoun en premier (LIMIT 1
            # sans ORDER BY). Si le token est un article (dep='det'), rejeter le rôle
            # et le bm du pronom objet pour ne pas contaminer le slot S/O en aval.
            _is_det_ctx = (t.get('dep') == 'det'
                           and node_data.get('role') == 'object_pronoun')
            if node_data.get('lemma'):
                t['lemma'] = node_data['lemma']
            if node_data.get('bm') and not _is_det_ctx:
                t['bm'] = node_data['bm']
            if node_data.get('role') and node_data['role'] != 'None' and not _is_det_ctx:
                t['role'] = node_data['role']
            if node_data.get('sc'):
                t['semantic_class'] = node_data['sc']
            if node_data.get('bm_suffix') and not _is_det_ctx:
                t['bm_suffix'] = node_data['bm_suffix']

        elif res and isinstance(res, dict):
            if res.get('lemma'): t['lemma'] = res['lemma']
            if res.get('bm'):    t['bm']    = res['bm']

    return tokens


def _split_contractions(text: str) -> str:
    """
    Split French contractions that confuse spaCy tokenizer.
    L'université → L' université
    d'Abidjan    → d' Abidjan
    """
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    text = re.sub(r"([lLdDjJmMtTsS])['']([A-Za-zÀ-ÖØ-öø-ÿ])", r"\1' \2", text)
    return text


_APOS_CHARS = {"'", "’", "ʼ"}

def _expand_elision(surf_lower: str, elision_map: dict) -> str:
    """
    je/ne/que/le/de/se/me/te/ce élidés (j'/n'/qu'/...) devant voyelle.

    Règle orthographique fermée du français (8 clitiques + ce) — pas un
    choix sémantique, donc pas besoin de LLM, mais KG-sourced (FunctionWord
    role='elision_expansion') plutôt qu'une table Python séparée (décision
    2026-07-13, audit hardcode-KG — consolidée avec la copie dupliquée dans
    _detect_progressive). Sans cette expansion, "j'ai" a pour surface "j'"
    qui ne matche pas l'entrée KG Pronoun "je" (bm='n'), et retombe sur le
    fallback "[lemma]" (ex: "[j]") au lieu de "n".
    """
    for _a in _APOS_CHARS:
        if surf_lower.endswith(_a):
            return elision_map.get(surf_lower, surf_lower)
    return surf_lower


def _merge_orphan_apostrophe(tokens):
    """
    Fusionne un token apostrophe isolé dans le token précédent.

    _split_contractions() force spaCy à séparer "J'ai" en "J' ai", mais le
    tokenizer spaCy peut alors produire l'apostrophe comme TOKEN À PART
    (ex: 'J' + "'" au lieu de "J'"), surtout en début de phrase (majuscule).
    Cette apostrophe orpheline hérite d'un dep parasite du parser (observé :
    dep='expl:comp', comme le "y" de "il y a") qui déclenche à tort les
    routes existentielles pour n'importe quelle phrase avec sujet élidé
    (j'/n'/qu'/l'/d'/s'/m'/t' + verbe). Sans ce merge, "j'ai eu une voiture"
    route vers existential_absolute au lieu de la possession.
    """
    if not any(t.get('surface') in _APOS_CHARS for t in tokens):
        return tokens

    survivors = []
    old_to_survivor_old = {}
    for t in tokens:
        if t.get('surface') in _APOS_CHARS and survivors:
            prev = survivors[-1]
            prev['surface'] = prev['surface'] + t['surface']
            prev['text'] = prev.get('text', prev['surface']) + t['surface']
            old_to_survivor_old[t['orig_index']] = prev['orig_index']
            continue
        survivors.append(t)

    full_old_map = {t['orig_index']: t['orig_index'] for t in survivors}
    full_old_map.update(old_to_survivor_old)
    old_to_new = {t['orig_index']: i for i, t in enumerate(survivors)}

    for t in survivors:
        _mapped_head = full_old_map.get(t['head_index'], t['head_index'])
        t['head_index'] = old_to_new.get(_mapped_head, t['head_index'])
        t['orig_index'] = old_to_new[t['orig_index']]
    return survivors


def _fix_pos_errors(tokens, grammar):
    adp       = grammar.get('adp_surfaces', set())
    cconj     = grammar.get('cconj_surfaces', set())
    clitic    = grammar.get('clitic_surfaces', set())
    expletives = grammar.get('expletive_surfaces', set())
    pronoun_person = grammar.get('pronoun_person', {})

    # ── Personne grammaticale des pronoms : correction Person= dans morph ──
    # spaCy mistague parfois le Person d'un clitique/pronom dans une inversion
    # interrogative avec trait d'union (ex: "t'appelles-tu ?" → 'tu' lui-même
    # tagué Person=3). Fait grammatical fermé (je=1, tu=2…) lu depuis le KG,
    # pas une décision lexicale — on corrige avant toute autre passe.
    if pronoun_person:
        for t in tokens:
            if t.get('pos') != 'PRON':
                continue
            _key = str(t.get('surface', '')).lower().strip("'’")
            _person = pronoun_person.get(_key)
            if not _person:
                continue
            _morph = str(t.get('morph', '') or '')
            _parts = [p for p in _morph.split('|') if p and not p.startswith('Person=')]
            _parts.append(f'Person={_person}')
            t['morph'] = '|'.join(_parts)

    # ── PASSE 1 : corrections POS ─────────────────────────────────
    for t in tokens:
        surf = str(t.get('surface', '')).lower().strip()

        if t.get('pos') == 'ADJ' and t.get('dep') in ('obl:arg', 'nsubj', 'obj', 'attr'):
            _has_nominal_deps = any(
                x.get('head_index') == t['orig_index']
                and x.get('dep') in ('amod', 'det', 'nmod', 'case')
                for x in tokens
            )
            if _has_nominal_deps:
                t['pos'] = 'NOUN'
        # DISABLED: Profession noun detection was too broad
        # Would reclassify ALL ADJ+ROOT+cop to NOUN, including generic adjectives like 'belle'
        # Need proper semantic analysis first — keep adjectives as ADJ for now
        # if (t.get('pos') == 'ADJ' and t.get('dep') == 'ROOT'
        #         and any(x.get('dep') == 'cop' for x in tokens)):
        #     _morph_str = str(t.get('morph', ''))
        #     _has_part_morph = any(x in _morph_str for x in (
        #         'VerbForm=Part', 'Tense=Past', 'Degree='))
        #     _has_nsubj = any(x.get('dep') in ('nsubj', 'nsubj:pass') for x in tokens)
        #     if not _has_part_morph and _has_nsubj:
        #         _is_profession = _is_profession_word(surf, t.get('lemma'))
        #         if _is_profession:
        #             t['pos'] = 'NOUN'

        if t.get('pos') == 'NOUN' and t.get('dep') == 'amod':
            _has_own_amod = any(
                x.get('dep') == 'amod' and x.get('head_index') == t['orig_index']
                for x in tokens
            )
            if not _has_own_amod:
                t['pos'] = 'ADJ'

        # Adjectif substantivé SUJET : "ce vieux voit" → spaCy attache 'vieux'
        # (ADJ) en amod sur le VERBE et n'assigne AUCUN nsubj. Un ADJ amod dont
        # la tête est un VERBE, sans nsubj dans la clause et placé AVANT le verbe,
        # est un adjectif substantivé sujet (le/ce vieux = le vieil homme) → NOM
        # nsubj. (amod régit normalement un NOM ; amod-sur-verbe = anomalie.)
        if t.get('pos') == 'ADJ' and t.get('dep') == 'amod':
            _adj_head = next((x for x in tokens
                              if x.get('orig_index') == t.get('head_index')), None)
            _clause_has_nsubj = any(x.get('dep') in ('nsubj', 'nsubj:pass')
                                    for x in tokens)
            if (_adj_head and _adj_head.get('pos') in ('VERB', 'AUX')
                    and not _clause_has_nsubj
                    and t['orig_index'] < _adj_head['orig_index']):
                t['pos'] = 'NOUN'
                t['dep'] = 'nsubj'
                # Réattacher le déterminant (ce/le/la) co-attaché au verbe → au
                # nom sujet, pour que 'ce vieux' forme un groupe nominal.
                for _d in tokens:
                    if (_d.get('dep') == 'det'
                            and _d.get('head_index') == _adj_head['orig_index']
                            and _d['orig_index'] < t['orig_index']):
                        _d['head_index'] = t['orig_index']

        # Adjectif substantivé COMPLÉMENT : "avec le vieux" — spaCy tague
        # 'vieux' PRON (pas ADJ) en obl:arg au lieu de NOM. Sans retaguer,
        # ce token reste sans bm et le filtre PRON+no-bm+obl:arg de
        # step5_obliques (comitatif, locatif...) l'exclut silencieusement,
        # perdant tout le complément (bug trouvé 2026-07-25 : "il va causer
        # avec le vieux tout le temps" → "avec le vieux" disparaissait
        # entièrement). Garde étroite : seul un PRON avec SON PROPRE
        # déterminant est retagué — un vrai pronom (lui/elle) n'en prend
        # jamais.
        if (t.get('pos') == 'PRON' and t.get('dep') == 'obl:arg'
                and any(d.get('dep') == 'det' and d.get('head_index') == t['orig_index']
                        for d in tokens)):
            t['pos'] = 'NOUN'

        # PUNCT ROOT/xcomp content → verbe infinitif mal étiqueté par spaCy
        if (t.get('pos') == 'PUNCT'
                and t.get('dep') in ('ROOT', 'xcomp')
                and t.get('role') == 'content'):
            t['pos'] = 'VERB'

        if t.get('pos') == 'AUX' and surf in adp:
            t['pos'] = 'ADP'
        if t.get('pos') == 'AUX' and surf in cconj:
            t['pos'] = 'CCONJ'
        if surf in clitic and t.get('pos') == 'PRON' and not t.get('bm'):
            t['role'] = 'clitic'
        if (t.get('pos') in ('NOUN', 'ADJ')
                and not t.get('is_plural')
                and 'Plur' in str(t.get('morph', ''))):
            t['is_plural'] = True

   # ── PASSE 2 : réattachement après corrections POS ─────────────
    for t in tokens:
        if t.get('pos') == 'NOUN' and t.get('dep') == 'nmod':
            print(f"DEBUG nmod: {t['surface']} idx={t['orig_index']} head={t['head_index']}")
            head_idx = t.get('head_index')
            t_idx = t['orig_index']
            for other in tokens:
                if (other.get('dep') == 'amod'
                        and other.get('head_index') == head_idx
                        and other.get('pos') == 'ADJ'):
                    # Réattacher uniquement si l'ADJ vient APRÈS le nmod :
                    # il qualifie alors le nmod, pas la tête.
                    # Si l'ADJ est entre la tête et le nmod (ou avant), il qualifie la tête → skip.
                    if other['orig_index'] <= t_idx:
                        print(f"DEBUG skip prenominal: {other['surface']} idx={other['orig_index']} ≤ nmod idx={t_idx}")
                        continue
                    print(f"DEBUG reattach: {other['surface']} head {head_idx}→{t_idx}")
                    other['head_index'] = t_idx
                    # if other['orig_index'] < t_idx:
                    #     print(f"DEBUG skip prenominal: {other['surface']} idx={other['orig_index']} < nmod idx={t_idx}")
                    #     continue
                    # print(f"DEBUG reattach: {other['surface']} head {head_idx}→{t_idx}")
                    # other['head_index'] = t_idx
                else:
                    if other.get('dep') == 'amod':
                        print(f"DEBUG skip amod: {other['surface']} dep={other['dep']} head={other['head_index']} pos={other['pos']} orig={other['orig_index']}")

    # ── PASSE 3 : expletifs et tirets ─────────────────────────────
    for st in tokens:
        surf_st = str(st.get('surface', '')).lower().rstrip("'").rstrip('\u2019')
        if st.get('dep') == 'nsubj' and surf_st in expletives:
            st['dep']  = 'expletive'
            st['role'] = 'expletive'
    for st in tokens:
        if str(st.get('surface', '')).strip() == '-' and st.get('dep') in ('nsubj', 'dep'):
            st['dep']  = 'punct'
            st['role'] = 'punct'
            st['pos']  = 'PUNCT'

    # ── PASSE 4 : arbre multi-ROOT cassé (fragment subordonné) ────────────────
    # spaCy peut produire PLUSIEURS dep='ROOT' sur un fragment sans proposition
    # principale (ex: "Quand il voit ce vieux" → voir/ce/vieux tous ROOT).
    # On garde le 1er VERBE comme ROOT et on réattache les autres :
    #   NOUN/ADJ/PROPN ROOT après le verbe → obj du verbe
    #   DET/PRON ROOT (démonstratif) → det du nom/adj objet suivant le plus proche
    # GARDE : ne se déclenche QUE s'il y a ≥2 ROOTs → les phrases normales
    # (un seul ROOT) ne sont JAMAIS touchées (pas de régression).
    _roots = [t for t in tokens if t.get('dep') == 'ROOT']
    if len(_roots) >= 2:
        _verb_root = next((t for t in _roots
                           if t.get('pos') in ('VERB', 'AUX')), None)
        if _verb_root:
            _vr_idx = _verb_root['orig_index']
            # 1) NOUN/ADJ/PROPN ROOT après le verbe → objet du verbe
            for t in tokens:
                if (t is not _verb_root and t.get('dep') == 'ROOT'
                        and t.get('pos') in ('NOUN', 'ADJ', 'PROPN')
                        and t['orig_index'] > _vr_idx):
                    t['dep'] = 'obj'
                    t['head_index'] = _vr_idx
                    t['is_root'] = False
            # 2) DET/PRON ROOT → det du nom/adj objet suivant le plus proche
            for d in tokens:
                if (d.get('dep') == 'ROOT'
                        and d.get('pos') in ('DET', 'PRON')
                        and d['orig_index'] > _vr_idx):
                    _head_noun = next(
                        (x for x in sorted(tokens, key=lambda z: z['orig_index'])
                         if x['orig_index'] > d['orig_index']
                         and x.get('dep') == 'obj'
                         and x.get('pos') in ('NOUN', 'ADJ', 'PROPN')), None)
                    if _head_noun:
                        d['dep'] = 'det'
                        d['head_index'] = _head_noun['orig_index']
                        d['is_root'] = False
            # 3) PRON ROOT après hyphen → nsubj inversé du verbe (ex: as-tu, est-il)
            # Note: _fix_pos_errors est appelé avant l'assignation des rôles,
            # donc on ne peut pas filtrer sur role= ; on utilise uniquement pos=PRON + position
            for t in tokens:
                if (t is not _verb_root and t.get('dep') == 'ROOT'
                        and t.get('pos') == 'PRON'
                        and t['orig_index'] > _vr_idx):
                    # Vérifier qu'il y a un hyphen entre verb_root et t (inversion as-tu)
                    _between = [x for x in tokens
                                if _vr_idx < x['orig_index'] < t['orig_index']
                                and x.get('surface', '').strip() in ('-', '–', '–', '‐')]
                    # Ou que PRON est juste +2 positions (hyphen déjà absorbé)
                    _is_adjacent = (t['orig_index'] - _vr_idx <= 2)
                    if _between or _is_adjacent:
                        t['dep'] = 'nsubj'
                        t['head_index'] = _vr_idx
                        t['is_root'] = False
                        for _hyph in _between:
                            _hyph['dep'] = 'punct'
                            _hyph['head_index'] = _vr_idx
            # 4) PUNCT/VERB ROOT orphelin (ex: "?") → punct du verbe
            for t in tokens:
                if (t is not _verb_root and t.get('dep') == 'ROOT'
                        and t.get('pos') in ('PUNCT', 'VERB')
                        and t['orig_index'] > _vr_idx):
                    t['dep'] = 'punct'
                    t['head_index'] = _vr_idx
                    t['is_root'] = False

        else:
            # Pas de VERB ROOT : chercher le NOUN/ADJ ROOT + ADV/PRON locatif ROOT.
            # Pattern : "Le cahier est ici" → spaCy retourne cahier ROOT + ici ROOT.
            # Pattern : "c'est vrai"/"c'est beau" → spaCy retourne c' ROOT + vrai ROOT.
            _noun_root = next((t for t in _roots
                               if t.get('pos') in ('NOUN', 'PROPN', 'ADJ')), None)
            if _noun_root:
                for t in _roots:
                    if (t is not _noun_root
                            and t.get('pos') == 'ADV'
                            and (t.get('role') in ('locative', 'temporal')
                                 or t.get('is_loc'))):
                        t['dep'] = 'advmod'
                        t['head_index'] = _noun_root['orig_index']
                        t['is_root'] = False
                    elif (t is not _noun_root
                            and t.get('pos') == 'PRON'
                            and t.get('role') in ('demonstrative', 'expletive', 'clitic', '')):
                        # PRON ROOT expletif/déictique (c', ce) → expl:subj du prédicat ADJ/NOUN
                        t['dep'] = 'expl:subj'
                        t['head_index'] = _noun_root['orig_index']
                        t['is_root'] = False

    return tokens

def _detect_participial_to(tokens, grammar=None):
    """
    Detect French gérondif (en V-ant) = simultaneous -tɔ in Bambara.
    Markers loaded from KG: FunctionWord{role:'participial_marker'}.
    """
    markers = (grammar or {}).get('participial_markers', set())
    result = []
    i = 0
    while i < len(tokens):
        if (i + 1 < len(tokens)
                and tokens[i].get('surface','').lower() in markers
                and tokens[i+1].get('pos') == 'VERB'
                and tokens[i+1].get('dep') in ('advcl','xcomp','ccomp','obj')):
            i += 1
            tokens[i]['role']  = 'participial_to'
            tokens[i]['tense'] = 'participial_to'
            result.append(tokens[i])
            i += 1
            continue
        result.append(tokens[i])
        i += 1
    return result


def _detect_progressive(tokens, grammar=None):
    """
    Detect progressive constructions (ex: 'être en train de V') → tense='prog'.
    Patterns loaded from KG: FunctionWord{role:'progressive_marker'}.
    Each pattern is a list of surfaces stored as space-separated string in f.surface.
    """
    patterns = (grammar or {}).get('progressive_markers', [])
    _elision_map = (grammar or {}).get('elision_map', {})
    result = []
    i = 0
    while i < len(tokens):
        matched = False
        for pattern in patterns:
            n = len(pattern)
            if i + n <= len(tokens):
                # Normaliser les formes élidées : d'→de, l'→le, j'→je, s'→se, m'→me, n'→ne
                # (KG-sourced, même source que _expand_elision — décision 2026-07-13)
                def _norm_surf(s):
                    sl = s.lower()
                    return _elision_map.get(sl, sl)
                window = [_norm_surf(tokens[i+k].get('surface','')) for k in range(n)]
                if window == pattern:
                    # Vérifier si le ROOT est parmi les tokens consommés
                    _consumed_is_root = any(tokens[i+k].get('is_root') for k in range(n))
                    _consumed_nsubj = next(
                        (x for x in tokens[:i] if x.get('dep') in ('nsubj', 'nsubj:pass')),
                        None)
                    i += n
                    # Sauter les pronoms clitiques (se, me, te…) pour trouver le VERB
                    # Les pronoms restent dans result et gardent leur association au verbe
                    _verb_i = i
                    while _verb_i < len(tokens) and tokens[_verb_i].get('pos') == 'PRON':
                        _verb_i += 1
                    if _verb_i < len(tokens) and tokens[_verb_i].get('pos') == 'VERB':
                        tokens[_verb_i]['tense'] = 'prog'
                        if _consumed_is_root:
                            tokens[_verb_i]['is_root'] = True
                            tokens[_verb_i]['dep'] = 'ROOT'
                    matched = True
                    break
        if not matched:
            result.append(tokens[i])
            i += 1
    return result


def _detect_compound_nouns(tokens, grammar=None):
    """
    Fusionne les noms composés en un seul token pour la traduction :
      1. Composés-trait-d'union : week-end, après-midi, porte-monnaie
         Pattern : NOUN/ADJ + PUNCT('-') + NOUN/ADJ
      2. Composés-génitif via PUNCT('-')+nmod déjà dans le flux
    Le token fusionné conserve le pos/dep de la tête et utilise
    la surface combinée pour le lookup KG + retriever.
    """
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        # Cas 1 : NOUN/ADJ/VERB + '-' + NOUN/ADJ (composé trait-d'union)
        if (i + 2 < len(tokens)
                and t.get('pos') in ('NOUN', 'ADJ', 'PROPN', 'VERB')
                and str(tokens[i+1].get('surface', '')).strip() == '-'
                and tokens[i+1].get('pos') in ('PUNCT', 'NOUN', 'SYM')
                and tokens[i+2].get('pos') in ('NOUN', 'ADJ', 'PROPN', 'VERB', 'ADV')):
            compound_surface = (t.get('surface', '') + '-'
                                 + tokens[i+2].get('surface', ''))
            compound_lemma   = (t.get('lemma', '') + '-'
                                 + tokens[i+2].get('lemma', ''))
            merged = {
                **t,
                'surface':    compound_surface,
                'lemma':      compound_lemma.lower(),
                'role':       t.get('role', 'content'),
                'bm':         '',   # sera rempli par le retriever
            }
            out.append(merged)
            i += 3
        else:
            out.append(t)
            i += 1
    return out


def _merge_multiword(tokens, funcs, lang):
    """Combine adjacent function_candidates into multiword tokens."""
    out, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            _raw_pair = tokens[i]['surface'] + ' ' + tokens[i+1]['surface']
            # Clé sans espace : réservée aux formes ÉLIDÉES (premier token finissant
            # par une apostrophe), ex: "jusqu'" + "au" → "jusqu'au".
            # Sans ce garde, "elle" + "s" (clitique réfléchi 's'' découpé) fusionnait
            # à tort en "elles" (pronom pluriel).
            _first_elided = str(tokens[i]['surface']).rstrip().endswith(("'", '\u2019'))
            _keys = [
                _raw_pair,
                _raw_pair.replace("qu'", "que").replace("qu’", "que"),
                re.sub(r'que\s+', "qu'", _raw_pair),   # jusque au → jusqu'au
            ]
            if _first_elided:
                _keys.append(tokens[i]['surface'] + tokens[i+1]['surface'])
            for key in _keys:
                fi = funcs.get((key.lower(), lang))
                if fi:
                    t = {**tokens[i], 'surface': key.lower(),
                         'lemma': key.lower(),
                         'role': fi['role'], 'bm': fi['bm']}
                    # Quand les deux tokens n'ont pas le même head :
                    # - Si le second pointe sur le premier (ex: un→quelque dans quelqu'un)
                    #   → garder le head du premier (quelque→venue) : c'est la tête externe.
                    # - Sinon (ex: jusque→dormir, au→matin) → prendre le head du second.
                    if tokens[i].get('head_index') != tokens[i+1].get('head_index'):
                        if tokens[i+1].get('head_index') == tokens[i].get('orig_index'):
                            pass  # head_index du premier déjà dans t
                        else:
                            t['head_index'] = tokens[i+1]['head_index']
                    if fi.get('bm_suffix'):
                        t['bm_suffix'] = fi['bm_suffix']
                    out.append(t); i += 2; break
            else:
                if tokens[i].get('role') != 'function_candidate':
                    out.append(tokens[i])
                i += 1
        else:
            if tokens[i].get('role') != 'function_candidate':
                out.append(tokens[i])
            i += 1
    return out


class SpacyParser:
    """
    Hybrid parser. All config from KG nodes:
      Pronoun, Article, NegMarker, Auxiliary, FunctionWord,
      Preposition, Word(NUM), PosMapping, SemanticClass
    """

    def __init__(self, db, backend='ollama', model=None, llm_model=None):
        self.db         = db
        self.llm        = MorphologicalParser(db, backend=backend, model=llm_model)
        self._llm_model = llm_model or 'qwen2.5:3b'
        self._g         = None

    def _grammar(self):
        """Load all KG config once. Returns cached dict."""
        if self._g: return self._g
        if not self.db:
            self._g = {k: (set() if isinstance(v, set) else v)
                       for k, v in {
                           'pron':set(),'sing':set(),'rel_pron':set(),
                           'art':set(),'art_suffix':{},'neg':set(),'demo':set(),
                           'refl':set(),'agent':set(),'dative':set(),
                           'gerund':set(),'loc':set(),'gen':set(),
                           'preps':{},'funcs':{},'aux':{},'nums':{},
                           # ── data-driven POS correction ──
                           'adp_surfaces':set(),'cconj_surfaces':set(),
                           'clitic_surfaces':set(),'quantifiers':{},
                           'distributive_each':{},'distributive_one':{},
                           'progressive_markers':[],'participial_markers':set(),
                           'expletive_surfaces':set(),'pronoun_person':{},
                           'locative_markers':set(),'temporal_markers':set(),
                           'elision_map':{},'loc_dat_ambiguous_surfaces':set(),
                           'expl_demo_surfaces':set(),'when_homograph_lemmas':set(),
                       }.items()}
            return self._g

        def q(cypher, p=None):
            try: return self.db.query(cypher, p or {})
            except: return []

        def ss(cypher, p=None):
            return {r['s'].lower() for r in q(cypher,p) if r.get('s')}

        pron     = ss("MATCH (n:Pronoun) RETURN n.surface AS s")
        sing     = ss("MATCH (n:Pronoun) WHERE n.singular=true RETURN n.surface AS s")
        rel_pron = ss("MATCH (n:Pronoun {role:'relative'}) RETURN n.surface AS s")

        # ── FunctionWord : chargement complet ────────────────────────────────
        fw = q("MATCH (f:FunctionWord) RETURN f.surface AS s, f.lang AS l, f.bm AS b, f.role AS r, f.bm_suffix AS suf, f.elided_form AS ef, f.requires_partner AS req")

        demo  = {r['s'].lower() for r in fw if r.get('r') == 'demonstrative'}
        refl  = {r['s'].lower() for r in fw if r.get('r') == 'reflexive'}

        # On embarque le bm et le bm_suffix dans le dictionnaire de cache funcs
        funcs = {}
        for r in fw:
            if r.get('s'):
                funcs[(r['s'].lower(), r.get('l','fr'))] = {
                    'bm': r.get('b', ''),
                    'role': r.get('r', 'content'),
                    'bm_suffix': r.get('suf', ''), # Transmis de force au parseur !
                    # 'plus' etc. : mot discontinu (ne...plus) — n'est une
                    # négation que si un autre token role='negation' co-occurre.
                    'requires_partner': bool(r.get('req')),
                }


        for (s,_), v in list(funcs.items()):
            funcs[(s,'en')] = v

        # ── Sets data-driven pour correction POS ─────────────────────────────
        cconj_surfaces = {r['s'].lower() for r in fw
                          if r.get('r') in ('conjunction', 'disjunction', 'contrast',
                                            'causal', 'consequence', 'conditional',
                                            'concessive', 'purpose')}

        clitic_surfaces = {r['s'].lower() for r in fw if r.get('r') == 'clitic'}
        _loc_dat_ambiguous_surfaces = {r['b'].lower() for r in fw
                                       if r.get('r') == 'locative_dative_ambiguous_surface' and r.get('b')}
        _when_homograph_lemmas = {r['b'].lower() for r in fw
                                  if r.get('r') == 'when_homograph_lemma' and r.get('b')}
        _expl_demo_surfaces = {r['b'].lower() for r in fw
                               if r.get('r') == 'expletive_demonstrative_surface' and r.get('b')}

        # Élision FR (j'/n'/qu'/l'/d'/s'/m'/t'/c' → je/ne/que/le/de/se/me/te/ce) :
        # règle orthographique fermée, KG-sourced (FunctionWord role='elision_expansion').
        # Stockée sur f.elided_form (PAS f.surface) : un vrai f.surface="j'" est
        # repris par les requêtes génériques "MATCH (n) WHERE n.surface=$x" (sans
        # filtre de rôle) utilisées ailleurs dans le pipeline, qui posaient alors
        # bm="je" littéralement sur le token FR au lieu de la vraie traduction bm="n"
        # (régression découverte 2026-07-13, corrigée en isolant le champ).
        # Toutes les variantes d'apostrophe (', ’, ʼ) mappées vers la même expansion.
        elision_map = {}
        for r in fw:
            if r.get('r') == 'elision_expansion' and r.get('ef') and r.get('b'):
                _base = r['ef'].lower().rstrip("'’ʼ")
                for _a in ("'", "’", "ʼ"):
                    elision_map[_base + _a] = r['b'].lower()

        quantifiers = {r['s'].lower(): r.get('b', '') for r in fw
                       if r.get('r') == 'quantifier'}

        distributive_each = {r['s'].lower(): r.get('b', '') for r in fw
                              if r.get('r') == 'distributive_each'}
        distributive_one  = {r['s'].lower(): r.get('b', '') for r in fw
                              if r.get('r') == 'distributive_one'}

        progressive_markers = [r['s'].lower().split() for r in fw
                                if r.get('r') == 'progressive_marker']

        participial_markers = {r['s'].lower() for r in fw
                               if r.get('r') == 'participial_marker'}

        expletive_surfaces = {r['s'].lower().rstrip("'").rstrip('\u2019')
                              for r in fw if r.get('r') == 'expletive'}

        # \u2500\u2500 Personne grammaticale des pronoms (fait fixe, pas une traduction) \u2500\u2500
        # spaCy mistague parfois Person= sur les clitiques dans les inversions
        # interrogatives avec trait d'union ("t'appelles-tu ?") \u2014 jusqu'\u00e0 taguer
        # 'tu' lui-m\u00eame en Person=3. Ces valeurs sont un fait grammatical ferm\u00e9
        # (je=1, tu=2, il=3\u2026), pas un choix lexical : on les corrige depuis le KG.
        pronoun_person = {r['s'].lower(): str(r['p']) for r in q(
            "MATCH (n:Pronoun) WHERE n.lang='fr' AND n.person IS NOT NULL "
            "RETURN n.surface AS s, n.person AS p")}

        # ── Prepositions ─────────────────────────────────────────────────────
        pr = q("MATCH (p:Preposition) RETURN p.surface AS s, p.role AS r, p.bm_marker AS m, p.is_prefix AS pfx")
        preps = {}
        preps_by_role = {}   # (surface, lang, role) -> entry — certaines prépositions
                              # (ex: 'à' = locatif OU datif) ont plusieurs rôles KG ;
                              # `preps` ne garde que le dernier lu, `preps_by_role`
                              # permet de retrouver une variante précise au besoin.
        for r in pr:
            if not r.get('s'):
                continue
            e = {'role': r.get('r'), 'bm_marker': r.get('m'), 'is_prefix': bool(r.get('pfx'))}
            preps[(r['s'].lower(), 'fr')] = e
            preps[(r['s'].lower(), 'en')] = e
            if r.get('r'):
                preps_by_role[(r['s'].lower(), 'fr', r['r'])] = e
                preps_by_role[(r['s'].lower(), 'en', r['r'])] = e

        adp_surfaces = {r['s'].lower() for r in fw
                        if r.get('s')
                        if r.get('r') in ('temporal', 'locative', 'genitive',
                                          'associative', 'benefactive')}
        adp_surfaces |= {r['s'].lower() for r in pr if r.get('s')}

        self._g = {
            'pron': pron, 'sing': sing, 'rel_pron': rel_pron,
            'demo': demo, 'refl': refl,
            'funcs': funcs, 'preps': preps, 'preps_by_role': preps_by_role,
            # ── correction POS data-driven ──
            'adp_surfaces':      adp_surfaces,
            'cconj_surfaces':    cconj_surfaces,
            'clitic_surfaces':   clitic_surfaces,
            'elision_map':       elision_map,
            'loc_dat_ambiguous_surfaces': _loc_dat_ambiguous_surfaces,
            'when_homograph_lemmas':      _when_homograph_lemmas,
            'expl_demo_surfaces':         _expl_demo_surfaces,
            'quantifiers':       quantifiers,
            'distributive_each': distributive_each,
            'distributive_one':  distributive_one,
            'progressive_markers':  progressive_markers,
            'participial_markers':  participial_markers,
            'expletive_surfaces':   expletive_surfaces,
            'pronoun_person':       pronoun_person,
            # ── markers ──
            'locative_markers':  {r['m'].lower() for r in pr
                                  if r.get('r') == 'locative' and r.get('m')},
            'temporal_markers':  {r['m'].lower() for r in pr
                                  if r.get('r') == 'temporal' and r.get('m')},
        }
        return self._g
    
    def parse(self, sentence: str) -> list:
        """
        Parses text into structured token dicts.
        """
        tokens = []
        lang = 'fr'
        nlp  = _get_nlp()

        if not nlp:
            return tokens

        clean_text = _split_contractions(sentence)
        doc = nlp(clean_text)

        # ── 1. Structure de base spaCy ────────────────────────────────────────
        for i, tok in enumerate(doc):
            surf_val = str(tok.text).strip() if tok.text else ""
            if not surf_val:
                continue

            t_dict = {
                'surface':        surf_val,
                'lemma':          tok.lemma_ if tok.lemma_ else surf_val,
                'pos':            tok.pos_,
                'dep':            tok.dep_,
                'text':           surf_val,
                'tense':          _tense(tok),
                'lang':           lang,
                'role':           'content',
                'bm':             '',
                'bm_marker':      '',
                'bm_suffix':      '',
                'head_index':     tok.head.i,
                'orig_index':     i,
                'is_neg':         False,
                'is_root':        (tok.dep_ == 'ROOT'),
                'is_loc':         False,
                'is_plural':      ('Plur' in str(tok.morph.get('Number'))),
                'is_genitive':    False,
                'is_passive':     ('Pass' in str(tok.morph.get('Voice'))),
                'is_agent':       False,
                'is_refl_past':   False,
                'is_dative':      False,
                'is_verbal_noun': False,
                'modal':          None,
                'morph':          str(tok.morph),
            }
            tokens.append(t_dict)

        tokens = _merge_orphan_apostrophe(tokens)
        tokens = _merge_causative_faire(tokens, self.db)

        # ── 2. Corrections et détections structurelles ────────────────────────
        tokens = resolve_auxiliary_lemmas(tokens, self.db)

        # ── 4. Enrichissement KG (needed before detect functions) ─────────────
        G = self._grammar()
        print("DEBUG avec in funcs:", ('avec', lang) in G.get('funcs', {}))
        tokens = _fix_pos_errors(tokens, G)
        tokens = _detect_participial_to(tokens, G)
        tokens = _detect_progressive(tokens, G)
        tokens = _detect_compound_nouns(tokens, G)

        # ── 3. LLM lemmatisation ──────────────────────────────────────────────
        try:
            llm_lemmas = self.llm.parse(clean_text)
            llm_lem = {}

            if isinstance(llm_lemmas, list):
                for item in llm_lemmas:
                    if isinstance(item, dict):
                        s_val = item.get('surface') or item.get('text')
                        l_val = item.get('lemma')
                        if s_val and l_val:
                            llm_lem[str(s_val).lower()] = str(l_val).lower()
            elif isinstance(llm_lemmas, dict):
                llm_lem = {str(k).lower(): str(v).lower()
                           for k, v in llm_lemmas.items() if v}

        except Exception as e:
            print(f"  LLM parse failed: {e}")
            llm_lem = {}

        for t in tokens:
            surf_lower = t['surface'].lower()
            if surf_lower in llm_lem and not t.get('_lefff_lemma_fixed'):
                t['lemma'] = llm_lem[surf_lower]

        for t in tokens:
            surf_lower  = t.get('surface', '').lower()
            if not surf_lower or surf_lower == 'none':
                continue
            surf_lower = _expand_elision(surf_lower, G.get('elision_map', {}))

            lemma_raw   = t.get('lemma')
            lemma_lower = lemma_raw.lower() if lemma_raw else surf_lower

            if surf_lower in G.get('pron', set()):
                if (surf_lower in G.get('rel_pron', set())
                        and t.get('dep') == 'nsubj'
                        and any(x.get('dep') == 'acl:relcl'
                                and x.get('orig_index') == t.get('head_index')
                                for x in tokens)):
                    t['role'] = 'relative'
                # Don't overwrite object pronoun/object roles already set by
                # resolve_auxiliary_lemmas (clitic detection takes priority)
                elif t.get('role') not in ('object_pronoun', 'object'):
                    t['role'] = 'pronoun'
                t['is_plural'] = surf_lower not in G.get('sing', set())
                # Appliquer bm et role précis depuis funcs si disponible
                _func = G.get('funcs', {}).get((surf_lower, lang), {})
                if _func.get('bm') and not t.get('bm'):
                    t['bm'] = _func['bm']
                # Don't overwrite object clitic roles with funcs article/pronoun.
                # Guard: dep='det' → article défini (le riz), pas un pronom clitique
                # objet → ne pas lui attribuer role='object_pronoun'.
                _func_role_is_obj = _func.get('role') in ('object_pronoun', 'object')
                _is_article_det = (t.get('dep') == 'det' and _func_role_is_obj)
                if (_func.get('role') and _func['role'] != 'content'
                        and t.get('role') not in ('object_pronoun', 'object')
                        and not _is_article_det):
                    t['role'] = _func['role']

            elif surf_lower in G.get('demo', set()):
                t['role'] = 'demonstrative'

            elif surf_lower in G.get('quantifiers', {}):
                t['role'] = 'quantifier'
                t['bm']   = G['quantifiers'][surf_lower]

            elif surf_lower in G.get('distributive_each', {}):
                t['role'] = 'distributive_each'
                t['bm']   = G['distributive_each'][surf_lower]

            elif surf_lower in G.get('distributive_one', {}):
                t['role'] = 'distributive_one'
                t['bm']   = G['distributive_one'][surf_lower]

            elif (surf_lower, lang) in G.get('preps', {}):
                prep_config  = G['preps'][(surf_lower, lang)]
                _prep_role = prep_config.get('role')
                # 'à' est structurellement ambigu (locatif "à Paris" vs datif
                # "donne le livre à l'enfant") — le KG a les deux entrées mais
                # une seule est retenue par surface. Signal structurel : si le
                # verbe régissant a DÉJÀ un objet direct (dep=obj) distinct de
                # ce syntagme, "à + NOM" ne peut pas être un second complément
                # de lieu du même verbe — c'est le destinataire (datif).
                if _prep_role == 'locative' and surf_lower in G.get('loc_dat_ambiguous_surfaces', {'à', 'au', 'a'}):
                    _noun_idx = t.get('head_index')
                    _noun_tok = next((x for x in tokens if x.get('orig_index') == _noun_idx), None)
                    _verb_idx = _noun_tok.get('head_index') if _noun_tok else None
                    _has_direct_obj = _verb_idx is not None and any(
                        x.get('dep') == 'obj' and x.get('head_index') == _verb_idx
                        and x.get('orig_index') != _noun_idx
                        for x in tokens)
                    if _has_direct_obj:
                        _dat_entry = G.get('preps_by_role', {}).get((surf_lower, lang, 'dative'))
                        if _dat_entry:
                            _prep_role  = 'dative'
                            prep_config = _dat_entry
                if _prep_role:
                    t['role'] = _prep_role
                t['bm_marker'] = prep_config.get('bm_marker', '')
                t['is_prefix_marker'] = bool(prep_config.get('is_prefix'))
                # 'is_loc' doit couvrir toute la famille des cas spatiaux
                # (dans→inside, sur→surface, sous→under, chez→associative),
                # pas seulement le rôle littéral 'locative' : sinon "dans"
                # (dont le KG retient le sous-rôle 'inside' pour choisir le
                # postposition 'kɔnɔ' plutôt que 'la'/'kan') ne déclenche
                # jamais la détection de clause locative en aval (kg_gateway
                # has_loc_case/has_loc_obl), et "X est dans Y" retombe à tort
                # sur une construction équative ("X yé yé Y").
                if _prep_role in ('locative', 'inside', 'surface', 'under', 'associative'):
                    t['is_loc'] = True
                # Si preps n'a pas de role, chercher dans funcs
                if not _prep_role and (surf_lower, lang) in G.get('funcs', {}):
                    _func = G['funcs'][(surf_lower, lang)]
                    if _func.get('role'):
                        t['role'] = _func['role']
                    if _func.get('bm') and not t.get('bm'):
                        t['bm'] = _func['bm']

            elif (surf_lower, lang) in G.get('funcs', {}):
                func_config = G['funcs'][(surf_lower, lang)]
                # Mots discontinus (KG FunctionWord.requires_partner, ex: 'plus'
                # pour 'ne...plus') : n'appliquer ce sens QUE si un autre token
                # négation-ancre ('ne'/'n\''/'pas'...) co-occurre dans la phrase.
                # Sinon "plus" reste un mot normal (comparatif/intensifieur :
                # "V toujours plus") au lieu d'être figé sur son bm négatif.
                if func_config.get('requires_partner'):
                    _neg_anchor_surfaces = {
                        s for (s, l), v in G.get('funcs', {}).items()
                        if l == lang and v.get('role') == 'negation' and not v.get('requires_partner')}
                    _has_neg_partner = any(
                        _expand_elision(str(ot.get('surface', '')).lower(), G.get('elision_map', {})) in _neg_anchor_surfaces
                        for ot in tokens if ot is not t)
                    if not _has_neg_partner:
                        continue
                # Toujours appliquer le role depuis funcs (pas seulement si 'content')
                if func_config.get('role'):
                    t['role'] = func_config['role']
                if func_config.get('bm') and not t.get('bm'):
                    t['bm'] = func_config['bm']
                if func_config.get('bm_suffix'):
                    t['bm_suffix'] = func_config['bm_suffix']
                    
            elif t.get('role') == 'negation_noun' and (surf_lower, lang) in G.get('funcs', {}):
                func_config = G['funcs'][(surf_lower, lang)]
                if func_config.get('bm'):
                    t['bm'] = func_config['bm']

        # ── Désambiguïsation des homographes 'quand'/'when' ──────────────────
        # Un mot interrogatif ('quand' → túma jùmɛn) employé comme conjonction
        # de subordination (dep='mark', SCONJ) est en réalité un subordonnant
        # TEMPOREL ('quand il vient' → tuma min). Le KG ne peut porter qu'un
        # seul sens par surface ; on tranche ici par la dépendance syntaxique.
        # Marqueur temporel data-driven depuis le KG (entrée 'lorsque'), repli
        # sur la valeur du CSV function_words.csv.
        _temporal_when_bm = (
            G.get('funcs', {}).get(('lorsque', lang), {}).get('bm')
            or 'túma mín'
        )
        # Désambiguïsation : LLM d'abord, fallback structurel si indisponible.
        # LLM : TEMPORAL → subordonnant temporel ; COMPLETEUR → complémenteur de verbe
        # Fallback : tête VERB + dep ≠ ccomp/xcomp/acl → temporel ; sinon complémenteur
        _ccomp_deps = ('ccomp', 'xcomp', 'acl', 'acl:relcl')
        _when_homographs = G.get('when_homograph_lemmas', {'quand', 'lorsque'})
        for t in tokens:
            # Restreint à l'homographe 'quand'/'lorsque' que cette désambiguïsation
            # cible explicitement (cf commentaire ci-dessus) : sans ce filtre de
            # surface, n'importe quel autre SCONJ marqué role='interrogative' en
            # amont (ex: 'que' dans l'optatif "Que Dieu t'aide") se faisait happer
            # par le même repli structurel et reclassé TEMPORAL à tort (tête=ROOT
            # non exclue), perdant son vrai sens au profit de 'tuma min'.
            if (t.get('dep') == 'mark'
                    and t.get('pos') == 'SCONJ'
                    and t.get('role') == 'interrogative'
                    and str(t.get('surface', '')).lower() in _when_homographs):
                _verdict = _classify_sconj(
                    t.get('surface', ''), sentence,
                    model=getattr(self, '_llm_model', 'qwen2.5:3b'))
                if _verdict is None:
                    _head = next((x for x in tokens
                                  if x.get('orig_index') == t.get('head_index')), None)
                    _verdict = ('TEMPORAL'
                                if (_head
                                    and _head.get('pos') == 'VERB'
                                    and _head.get('dep') not in _ccomp_deps)
                                else 'COMPLETEUR')
                if _verdict == 'TEMPORAL':
                    t['role'] = 'temporal'
                    t['bm']   = _temporal_when_bm

        # ── Désambiguïsation ADJ vs VERBE : clause SANS verbe ───────────────
        # 'le chien court' : spaCy tague 'court' ADJ(amod) → faux syntagme
        # nominal. Une clause déclarative a besoin d'un prédicat : s'il n'y a
        # AUCUN verbe, on privilégie le verbe → l'ADJ amod du nom racine est
        # retagué VERBE/ROOT (le nom devient nsubj). Le LLM fournit l'infinitif
        # (court → courir) ; s'il dit ADJECTIF ou est indisponible → on garde
        # le tag ADJ (dégradation sûre vers le syntagme nominal).
        _has_verb = any(x.get('pos') in ('VERB', 'AUX') for x in tokens)
        if not _has_verb:
            _adj_amod = next((x for x in tokens
                              if x.get('pos') == 'ADJ'
                              and x.get('dep') == 'amod'), None)
            _noun_head = (next((x for x in tokens
                                if x.get('orig_index') == _adj_amod.get('head_index')
                                and x.get('pos') == 'NOUN'), None)
                          if _adj_amod else None)
            if _adj_amod and _noun_head:
                _inf = _classify_adj_verb(
                    _adj_amod.get('surface', ''), sentence,
                    model=getattr(self, '_llm_model', 'qwen2.5:3b'))
                if _inf:
                    _adj_amod['pos']        = 'VERB'
                    _adj_amod['dep']        = 'ROOT'
                    _adj_amod['lemma']      = _inf
                    _adj_amod['role']       = 'content'
                    _adj_amod['is_root']    = True
                    _adj_amod['head_index'] = _adj_amod['orig_index']
                    _adj_amod['tense']      = _adj_amod.get('tense') or 'pres'
                    _noun_head['dep']        = 'nsubj'
                    _noun_head['is_root']    = False
                    _noun_head['head_index'] = _adj_amod['orig_index']

        # ── Désambiguïsation 'pour' : benefactive vs purposive ──────────────
        # 'pour toi' / 'pour ma fille' (dep='case' sur un nom) → benefactive (ye).
        # 'pour cuisiner' (dep='mark' sur un VERB infinitif)  → purposive (walasa ka).
        # Strictement ciblé : seul un prép. benefactive en dep='mark' attaché à un
        # VERB est reclassé. Les benefactifs sur noms (dep='case') ne sont pas touchés.
        _purp_bm = G.get('funcs', {}).get(('pour', lang), {}).get('bm') or 'walasa ka'
        for t in tokens:
            if (t.get('role') == 'benefactive'
                    and t.get('dep') == 'mark'):
                _head = next((x for x in tokens
                              if x.get('orig_index') == t.get('head_index')), None)
                if _head and _head.get('pos') == 'VERB':
                    t['role'] = 'purposive'
                    t['bm_marker'] = _purp_bm

        # ── Normaliser les pronoms inversés (-ils, -vous...) via G.funcs ─────
        for st in tokens:
            surf = str(st.get('surface', '')).strip()
            if (surf.startswith('-')
                    and st.get('pos') == 'PRON'
                    and st.get('dep') in ('nsubj', 'nsubj:pass')):
                _clean = surf.lstrip('-').lower()
                _func = G.get('funcs', {}).get((_clean, lang), {})
                if _func.get('bm'):
                    st['bm'] = _func['bm']
                    st['role'] = _func.get('role', 'pronoun')
        # Après la boucle for t in tokens:
        for t in tokens:
            if t.get('surface', '').lower() == 'avec':
                print(f"DEBUG avec final: role={t.get('role')} bm={t.get('bm')} dep={t.get('dep')}")

        # ── Optatif : "que" mark sur ROOT même si Mood=Ind (spaCy le rate) ──
        # "que Moussa mange" → spaCy donne Mood=Ind pour 'mange' mais c'est
        # un optatif. On force tense='sub' si le ROOT porte un mark SCONJ 'que'.
        # EXCLUSION : passé composé (aux:tense présent) ou participe passé
        # → subordination factuelle (que j'ai dit…), pas optatif.
        # EXCLUSION : "Est-ce que" → interrogatif polaire (Yala …  wà ?), pas optatif.
        # Signal : '-ce' dep='nsubj' présent.
        _has_estce_que = any(
            str(_t.get('surface', '')).lower().strip('-') in G.get('expl_demo_surfaces', {'ce'})
            and _t.get('dep') in ('nsubj', 'expl:subj')
            for _t in tokens)
        for t in tokens:
            if (t.get('is_root')
                    and t.get('pos') == 'VERB'
                    and t.get('tense') != 'sub'
                    and not _has_estce_que
                    and 'VerbForm=Part' not in str(t.get('morph', ''))
                    and not any(x.get('dep') == 'aux:tense'
                                and x.get('head_index') == t['orig_index']
                                for x in tokens)):
                _que_mark = any(
                    x.get('dep') == 'mark'
                    and x.get('pos') == 'SCONJ'
                    and str(x.get('surface', '')).lower().rstrip("'").rstrip('\u2019')
                    in ('que', 'qu')
                    and x.get('head_index') == t['orig_index']
                    for x in tokens)
                if _que_mark:
                    t['tense'] = 'sub'

        # ── Fusion des mots grammaticaux multi-tokens ─────────────────────
        tokens = _merge_multiword(tokens, G.get('funcs', {}), lang)

        return tokens