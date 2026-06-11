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
from utils.normalize import normalize_token
from utils.language import detect_language
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
            'http://localhost:11434/api/generate',
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
            'http://localhost:11434/api/generate',
            json={'model': model, 'prompt': prompt, 'stream': False,
                  'options': {'temperature': 0, 'num_predict': 10}},
            timeout=8)
        raw = r.json().get('response', '').strip()
        m = re.match(r'\s*VERBE\W+([a-zà-ÿ]+)', raw, re.IGNORECASE)
        return m.group(1).lower() if m else None
    except Exception:
        _adjverb_circuit_open_until = _t.time() + 60
        return None


def _get_nlp(lang):
    """Load and cache spaCy model for lang."""
    if lang not in _nlp_cache:
        import spacy
        for m in (['fr_dep_news_trf','fr_core_news_lg','fr_core_news_md','fr_core_news_sm'] if lang=='fr'
                  else ['en_core_web_trf','en_core_web_lg','en_core_web_md','en_core_web_sm']):
            try: _nlp_cache[lang] = spacy.load(m); break
            except: continue
        else: _nlp_cache[lang] = None
    return _nlp_cache[lang]


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

    # 2. Tense
    tense = str(tok.morph.get('Tense')).lower()
    if 'imp' in tense:
        return 'hab'
    if 'past' in tense:
        return 'past'
    if 'fut' in tense:
        return 'fut'

    return 'pres'


def resolve_auxiliary_lemmas(tokens, db):
    """
    Validation universelle et redressement des lemmes d'auxiliaires.
    """
    for t in tokens:
        # Ne pas normaliser les pronoms inversés pluriels
        if str(t.get('surface', '')).startswith('-') and t.get('is_plural'):
            continue

        surf_lower = t.get('surface', '').lower()
        if not surf_lower or surf_lower == 'none':
            continue
        lang_curr = t.get('lang', 'fr')

        # Correction spaCy : 'suis' est parfois lemmatisé 'suivre' au lieu de 'être'
        # Règle : si ROOT ou cop sans obj direct → c'est 'être'
        if (surf_lower == 'suis'
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

        res = db.query(
            "MATCH (n) WHERE n.lang = $lang "
            "AND (n.surface = $surface OR n.lemma = $lemma) "
            "RETURN n.lemma AS lemma, n.bm AS bm, n.role AS role, "
            "n.semantic_class AS sc, labels(n) AS labels LIMIT 1",
            {'lang': lang_curr, 'surface': surf_lower,
             'lemma': t.get('lemma', surf_lower).lower()}
        )


        if res and isinstance(res, list) and len(res) > 0:
            node_data = res[0]
            if node_data.get('lemma'):
                t['lemma'] = node_data['lemma']
            if node_data.get('bm'):
                t['bm'] = node_data['bm']
            if node_data.get('role') and node_data['role'] != 'None':
                t['role'] = node_data['role']
            if node_data.get('sc'):
                t['semantic_class'] = node_data['sc']

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

def _fix_pos_errors(tokens, grammar):
    adp       = grammar.get('adp_surfaces', set())
    cconj     = grammar.get('cconj_surfaces', set())
    clitic    = grammar.get('clitic_surfaces', set())
    expletives = grammar.get('expletive_surfaces', set())

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
        if st.get('dep') == 'nsubj' and str(st.get('surface', '')).strip() == '-':
            st['dep']  = 'punct'
            st['role'] = 'punct'

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
    result = []
    i = 0
    while i < len(tokens):
        matched = False
        for pattern in patterns:
            n = len(pattern)
            if i + n <= len(tokens):
                window = [tokens[i+k].get('surface','').lower() for k in range(n)]
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
            _first_elided = str(tokens[i]['surface']).rstrip().endswith(("'", '’'))
            _keys = [
                _raw_pair,
                _raw_pair.replace("qu'", "que").replace("qu'", "que"),
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
                           'progressive_markers':[],'participial_markers':set(),
                           'expletive_surfaces':set(),
                           'locative_markers':set(),'temporal_markers':set(),
                           'demonstrative_suffix':'in','resultative_marker':'ye',
                           'tam_default':'bɛ',
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
        fw = q("MATCH (f:FunctionWord) RETURN f.surface AS s, f.lang AS l, f.bm AS b, f.role AS r, f.bm_suffix AS suf")

        demo  = {r['s'].lower() for r in fw if r.get('r') == 'demonstrative'}
        refl  = {r['s'].lower() for r in fw if r.get('r') == 'reflexive'}
        
        # On embarque le bm et le bm_suffix dans le dictionnaire de cache funcs
        funcs = {}
        for r in fw:
            if r.get('s'):
                funcs[(r['s'].lower(), r.get('l','fr'))] = {
                    'bm': r.get('b', ''),
                    'role': r.get('r', 'content'),
                    'bm_suffix': r.get('suf', '') # Transmis de force au parseur !
                }


        for (s,_), v in list(funcs.items()):
            funcs[(s,'en')] = v

        # ── Sets data-driven pour correction POS ─────────────────────────────
        cconj_surfaces = {r['s'].lower() for r in fw
                          if r.get('r') in ('conjunction', 'disjunction', 'contrast',
                                            'causal', 'consequence', 'conditional',
                                            'concessive', 'purpose')}

        clitic_surfaces = {r['s'].lower() for r in fw if r.get('r') == 'clitic'}

        quantifiers = {r['s'].lower(): r.get('b', '') for r in fw
                       if r.get('r') == 'quantifier'}

        progressive_markers = [r['s'].lower().split() for r in fw
                                if r.get('r') == 'progressive_marker']

        participial_markers = {r['s'].lower() for r in fw
                               if r.get('r') == 'participial_marker'}

        expletive_surfaces = {r['s'].lower().rstrip("'").rstrip('\u2019')
                              for r in fw if r.get('r') == 'expletive'}

        # ── Prepositions ─────────────────────────────────────────────────────
        pr = q("MATCH (p:Preposition) RETURN p.surface AS s, p.role AS r, p.bm_marker AS m")
        preps = {}
        for r in pr:
            if not r.get('s'):
                continue
            e = {'role': r.get('r'), 'bm_marker': r.get('m')}
            preps[(r['s'].lower(), 'fr')] = e
            preps[(r['s'].lower(), 'en')] = e

        # Override : 'au'/'à' locatif → marqueur 'la' (pas 'kɔnɔ' qui signifie 'dans')
        for _au_surf in ('au', 'à', 'a'):
            for _l in ('fr', 'en'):
                _entry = preps.get((_au_surf, _l))
                if _entry and _entry.get('role') == 'locative':
                    _entry['bm_marker'] = 'la'

        adp_surfaces = {r['s'].lower() for r in fw
                        if r.get('s')
                        if r.get('r') in ('temporal', 'locative', 'genitive',
                                          'associative', 'benefactive')}
        adp_surfaces |= {r['s'].lower() for r in pr if r.get('s')}

        self._g = {
            'pron': pron, 'sing': sing, 'rel_pron': rel_pron,
            'demo': demo, 'refl': refl,
            'funcs': funcs, 'preps': preps,
            # ── correction POS data-driven ──
            'adp_surfaces':      adp_surfaces,
            'cconj_surfaces':    cconj_surfaces,
            'clitic_surfaces':   clitic_surfaces,
            'quantifiers':       quantifiers,
            'progressive_markers':  progressive_markers,
            'participial_markers':  participial_markers,
            'expletive_surfaces':   expletive_surfaces,
            # ── markers ──
            'locative_markers':  {r['m'].lower() for r in pr
                                  if r.get('r') == 'locative' and r.get('m')},
            'temporal_markers':  {r['m'].lower() for r in pr
                                  if r.get('r') == 'temporal' and r.get('m')},
            'demonstrative_suffix': 'in',
            'resultative_marker':   'ye',
            'tam_default':          'bɛ',
        }
        return self._g
    
    def parse(self, sentence: str) -> list:
        """
        Parses text into structured token dicts.
        """
        tokens = []
        lang = detect_language(sentence)
        nlp  = _get_nlp(lang)

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

        # ── 2. Corrections et détections structurelles ────────────────────────
        tokens = resolve_auxiliary_lemmas(tokens, self.db)

        # ── 4. Enrichissement KG (needed before detect functions) ─────────────
        G = self._grammar()
        print("DEBUG avec in funcs:", ('avec', lang) in G.get('funcs', {}))
        tokens = _fix_pos_errors(tokens, G)
        tokens = _detect_participial_to(tokens, G)
        tokens = _detect_progressive(tokens, G)

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
            print(f"⚠️  LLM parse failed: {e}")
            llm_lem = {}

        for t in tokens:
            surf_lower = t['surface'].lower()
            if surf_lower in llm_lem:
                t['lemma'] = llm_lem[surf_lower]
            # Garde-fou lemme verbal : spaCy renvoie parfois un lemme aberrant
            # (chante -> cher). Si, pour un VERBE, le lemme retenu est PLUS
            # COURT que la surface ET ne partage pas un prefixe d'au moins
            # 3 lettres, il est invraisemblable -> on retombe sur la surface
            # (meilleure pour le retrieval embedding que le mauvais lemme).
            # Les infinitifs irreguliers (est->etre, vais->aller) sont aussi
            # longs ou plus longs -> jamais touches.
            if t.get('pos') == 'VERB':
                _lem = (t.get('lemma') or '').lower()
                if (_lem and _lem != surf_lower and _lem.isalpha()
                        and len(_lem) < len(surf_lower)):
                    _pref = 0
                    for _a, _b in zip(_lem, surf_lower):
                        if _a != _b:
                            break
                        _pref += 1
                    if _pref < 3:
                        t['lemma'] = t['surface']

        for t in tokens:
            surf_lower  = t.get('surface', '').lower()
            if not surf_lower or surf_lower == 'none':
                continue

            lemma_raw   = t.get('lemma')
            lemma_lower = lemma_raw.lower() if lemma_raw else surf_lower

            if surf_lower in G.get('pron', set()):
                if (surf_lower in G.get('rel_pron', set())
                        and t.get('dep') == 'nsubj'
                        and any(x.get('dep') == 'acl:relcl'
                                and x.get('orig_index') == t.get('head_index')
                                for x in tokens)):
                    t['role'] = 'relative'
                else:
                    t['role'] = 'pronoun'
                t['is_plural'] = surf_lower not in G.get('sing', set())
                # Appliquer bm et role précis depuis funcs si disponible
                _func = G.get('funcs', {}).get((surf_lower, lang), {})
                if _func.get('bm') and not t.get('bm'):
                    t['bm'] = _func['bm']
                if _func.get('role') and _func['role'] != 'content':
                    t['role'] = _func['role']

            elif surf_lower in G.get('demo', set()):
                t['role'] = 'demonstrative'

            elif surf_lower in G.get('quantifiers', {}):
                t['role'] = 'quantifier'
                t['bm']   = G['quantifiers'][surf_lower]

            elif (surf_lower, lang) in G.get('preps', {}):
                prep_config  = G['preps'][(surf_lower, lang)]
                _prep_role = prep_config.get('role')
                if _prep_role:
                    t['role'] = _prep_role
                t['bm_marker'] = prep_config.get('bm_marker', '')
                if _prep_role == 'locative':
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
            or 'tuma min'
        )
        # Désambiguïsation : LLM d'abord, fallback structurel si indisponible.
        # LLM : TEMPORAL → subordonnant temporel ; COMPLETEUR → complémenteur de verbe
        # Fallback : tête VERB + dep ≠ ccomp/xcomp/acl → temporel ; sinon complémenteur
        _ccomp_deps = ('ccomp', 'xcomp', 'acl', 'acl:relcl')
        for t in tokens:
            if (t.get('dep') == 'mark'
                    and t.get('pos') == 'SCONJ'
                    and t.get('role') == 'interrogative'):
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

        # ── Fusion des mots grammaticaux multi-tokens ─────────────────────
        tokens = _merge_multiword(tokens, G.get('funcs', {}), lang)

        return tokens