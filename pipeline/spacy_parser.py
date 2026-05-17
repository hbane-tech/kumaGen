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
from utils.normalize import normalize_token
from utils.language import detect_language
from llm.morphological_parser import MorphologicalParser

_nlp_cache = {}


def _get_nlp(lang):
    """Load and cache spaCy model for lang."""
    if lang not in _nlp_cache:
        import spacy
        for m in (['fr_dep_news_trf','fr_core_news_sm'] if lang=='fr'
                  else ['en_core_web_trf','en_core_web_sm']):
            try: _nlp_cache[lang] = spacy.load(m); break
            except: continue
        else: _nlp_cache[lang] = None
    return _nlp_cache[lang]


def _tense(tok):
    # 1. Mode first (Mood)
    mood = str(tok.morph.get('Mood')).lower()
    if 'imp' in mood:
        return 'imp'   # Imperative

    # 2. Tense
    tense = str(tok.morph.get('Tense')).lower()
    if 'imp' in tense:
        return 'hab'   # Imperfect → tùn bɛ
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
        surf_lower = t.get('surface', '').lower()
        if not surf_lower or surf_lower == 'none':
            continue
        lang_curr = t.get('lang', 'fr')
        
        res = db.query(
            "MATCH (n) WHERE n.lang = $lang AND n.surface = $surface "
            "RETURN n.lemma AS lemma, n.bm AS bm, labels(n) AS labels",
            {'lang': lang_curr, 'surface': surf_lower}
        )
        
        # CORRECTION : Extraire le dictionnaire res[0] pour ne pas manipuler la liste brute
            
        if res and isinstance(res, list) and len(res) > 0:
            node_data = res[0]  # <-- APPLIQUER LE INDICE [0] DE MANIÈRE STRICTE
            if node_data.get('lemma'):
                t['lemma'] = node_data['lemma']
            if node_data.get('bm'):
                t['bm'] = node_data['bm']


            if 'Auxiliary' in node_data.get('labels', []) or 'FunctionWord' in node_data.get('labels', []):
                t['pos'] = 'AUX'
                
        elif res and isinstance(res, dict):
            if res.get('lemma'): t['lemma'] = res['lemma']
            if res.get('bm'): t['bm'] = res['bm']
            
    return tokens

# def _split_contractions(text: str) -> str:
#     """
#     Split French contractions that confuse spaCy tokenizer.
#     l'Institut → l' Institut
#     d'Abidjan  → d' Abidjan
#     """
#     text = re.sub(r"([ldldld])['’]([A-Za-zÀ-ÿ])", r"\1' \2", text)
#     return text

def _split_contractions(text: str) -> str:
    """
    Split French contractions that confuse spaCy tokenizer.
    L'université → L' université
    d'Abidjan    → d' Abidjan
    """
    import re
    # PROTECTION GLOBALE : On inclut les majuscules, les minuscules et les apostrophes typographiques (’)
    # On ajoute également un nettoyage immédiat des codes d'échappement ANSI (\x1b...) du terminal
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    
    # Séparation étanche de l'élision française (l, L, d, D, j, J, m, M, t, T, s, S)
    text = re.sub(r"([lLdDjJmMtTsS])['’]([A-Za-zÀ-ÖØ-öø-ÿ])", r"\1' \2", text)
    return text


def _detect_participial_to(tokens):
    """
    Detect French 'en V-ant' (gérondif) = simultaneous -tɔ in Bambara.
    'en passing' → passer.role='participial_to'
    """
    result = []
    i = 0
    while i < len(tokens):
        if (i + 1 < len(tokens)
            and tokens[i].get('surface','').lower() == 'en'
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


def _detect_progressive(tokens):
    """
    Detect French 'être en train de V' → marks main verb tense='prog'.
    Drops the en/train/de tokens.
    """
    result = []
    i = 0
    while i < len(tokens):
        if (i + 2 < len(tokens)
            and tokens[i].get('surface','').lower() == 'en'
            and tokens[i+1].get('surface','').lower() == 'train'
            and tokens[i+2].get('surface','').lower() in ('de', "d'", "d’")):
            i += 3
            if i < len(tokens) and tokens[i].get('pos') == 'VERB':
                tokens[i]['tense'] = 'prog'
            continue
        result.append(tokens[i])
        i += 1
    return result


def _merge_multiword(tokens, funcs, lang):
    """Combine adjacent function_candidates into multiword tokens."""
    out, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            for key in [tokens[i]['surface']+' '+tokens[i+1]['surface'],
                        (tokens[i]['surface']+' '+tokens[i+1]['surface'])
                        .replace("qu'","que").replace("qu’","que")]:
                fi = funcs.get((key.lower(), lang))
                if fi:
                    t = {**tokens[i], 'surface': key.lower(),
                         'lemma': key.lower(),
                         'role': fi['role'], 'bm': fi['bm']}
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
        self.db  = db
        self.llm = MorphologicalParser(db, backend=backend, model=llm_model)
        self._g  = None   # grammar cache

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

        fw = q("MATCH (f:FunctionWord) RETURN f.surface AS s, f.lang AS l, f.bm AS b, f.role AS r")
        demo  = {r['s'].lower() for r in fw if r.get('r')=='demonstrative'}
        refl  = {r['s'].lower() for r in fw if r.get('r')=='reflexive'}
        funcs = {(r['s'].lower(), r.get('l','fr')): {'bm':r.get('b'),'role':r.get('r')}
                 for r in fw if r.get('s')}
        for (s,_), v in list(funcs.items()):
            funcs[(s,'en')] = v

        pr = q("MATCH (p:Preposition) RETURN p.surface AS s, p.role AS r, p.bm_marker AS m")
        preps = {}
        for r in pr:
            if not r.get('s'):
                continue
            e = {'role':r.get('r'),'bm_marker':r.get('m')}
            preps[(r['s'].lower(),'fr')] = e
            preps[(r['s'].lower(),'en')] = e

        self._g = {
            'pron': pron, 'sing': sing, 'rel_pron': rel_pron, 'demo': demo, 'refl': refl,
            'funcs': funcs, 'preps': preps,
            'locative_markers': {r['m'].lower() for r in pr if r.get('r') == 'locative' and r.get('m')},
            'temporal_markers': {r['m'].lower() for r in pr if r.get('r') == 'temporal' and r.get('m')},
            'demonstrative_suffix': 'in',
            'resultative_marker': 'ye',
            'tam_default': 'bɛ'
        }
        return self._g

    def parse(self, sentence: str) -> list:
        """
        Parses text into structured token dicts.
        Garantit la sanctuarisation de 'surface' pour empêcher l'injection de None.
        """
        tokens = []
        lang = detect_language(sentence)
        nlp = _get_nlp(lang)
        
        if not nlp:
            return tokens

        clean_text = _split_contractions(sentence)
        doc = nlp(clean_text)

        # 2. CONSTRUIRE LA STRUCTURE DE BASE DE SPACY
        for i, tok in enumerate(doc):
            # SANCTUARISATION DE LA SURFACE : Forcer une valeur de chaîne non-nulle
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
                'modal':          None
            }
            tokens.append(t_dict)

        # Corrections morphologiques et structurelles (Data-Driven)
        tokens = resolve_auxiliary_lemmas(tokens, self.db)
        tokens = _detect_participial_to(tokens)
        tokens = _detect_progressive(tokens)

        # # 3. INTERROGATION SÉCURISÉE DU COUPLAGE LLM
        # try:
        #     # ALIGNEMENT INTERFACE LLM : Utilisation de la méthode générique .parse()
        #     llm_lemmas = self.llm.parse(clean_text)
        #     llm_lem = {k.lower(): v.lower() for k, v in llm_lemmas.items() if v}
        # except Exception as e:
        #     print(f"⚠️ Layer 1 LLM parse failed: {e}. Falling back to native spaCy lemmas.")
        #     llm_lem = {}

        # 3. INTERROGATION SÉCURISÉE DU COUPLAGE LLM
        try:
            llm_lemmas = self.llm.parse(clean_text)
            llm_lem = {}
            
            # SÉCURISATION DU TYPE (DATA-DRIVEN) : On s'adapte dynamiquement au format renvoyé par le parser
            if isinstance(llm_lemmas, list):
                # Si c'est une liste de dictionnaires du type [{'surface': '...', 'lemma': '...'}]
                for item in llm_lemmas:
                    if isinstance(item, dict):
                        # On extrait selon les clés standards de votre projet
                        s_val = item.get('surface') or item.get('text')
                        l_val = item.get('lemma')
                        if s_val and l_val:
                            llm_lem[str(s_val).lower()] = str(l_val).lower()
            elif isinstance(llm_lemmas, dict):
                # Si c'est déjà un dictionnaire natif
                llm_lem = {str(k).lower(): str(v).lower() for k, v in llm_lemmas.items() if v}
                
        except Exception as e:
            print(f"⚠️ Layer 1 LLM parse failed: {e}. Falling back to native spaCy lemmas.")
            llm_lem = {}


        for t in tokens:
            surf_lower = t['surface'].lower()
            if surf_lower in llm_lem:
                t['lemma'] = llm_lem[surf_lower]

        # 4. ENRICHISSEMENT CONFIGURATIONNEL VIA LE GRAPHE (KG)
        G = self._grammar()
        for t in tokens:
            surf_lower = t.get('surface', '').lower()
            if not surf_lower or surf_lower == 'none':
                continue
                
            lemma_raw = t.get('lemma')
            lemma_lower = lemma_raw.lower() if lemma_raw is not None else surf_lower

            # Assignation des rôles en protégeant la surface originale des tokens
            if surf_lower in G.get('pron', set()):
                t['role'] = 'pronoun'
                t['is_plural'] = surf_lower not in G.get('sing', set())
            elif surf_lower in G.get('demo', set()):
                t['role'] = 'demonstrative'
            elif (surf_lower, lang) in G.get('preps', {}):
                prep_config = G['preps'][(surf_lower, lang)]
                t['role'] = prep_config.get('role', 'content')
                t['bm_marker'] = prep_config.get('bm_marker', '')
                if prep_config.get('role') == 'locative':
                    t['is_loc'] = True

        return tokens
