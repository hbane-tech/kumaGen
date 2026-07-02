"""
llm/morphological_parser.py

Single responsibility: given a sentence, return lemma for each word
AND the clause construction type.

The LLM does two things in one call:
  1. Lemmatize each token
  2. Classify the clause construction

Result is cached in Neo4j ParseCache node.
"""

import re
import json
import hashlib
from config.settings import OLLAMA_GENERATE_URL

# ── System prompt ─────────────────────────────────────────────────

_SYSTEM_PROMPT = """Analyze this French or English sentence. Return JSON with two fields.

1. "clause_type": choose the MOST SPECIFIC type. Rules:
   être+ADJ → "qualitative" (NOT presentative). être+NOUN → "equative". c'est/voici → "presentative".

SENTENCE STRUCTURE:
  "simple"          : S V O  (je mange du riz)
  "complex"         : modal+infinitive  (je veux manger, il peut partir)
  "verb_serial"     : chained actions  (il but de l'eau et parla ensuite)
  "purposive"       : motion verb + bare V  (il est allé appeler le chef)

NON-VERBAL (être+...):
  "qualitative"     : être+ADJ  (je suis belle, le cheval est rapide, il est grand)
  "equative"        : être+un/une NOUN  (Musa est un chasseur, c'est un médecin)
  "locative"        : être+location  (Musa est au village, il est ici, elle est là)
  "presentative"    : c'est X / voici X  (c'est Musa, voici le livre)
  "existential"     : il y a  (il y a un problème, il n'y a rien)

VERBAL:
  "passive"         : passive voice  (il est mis en déroute, il a été battu)
  "refl_past"       : reflexive past  (il s'était engagé, elle s'est levée)
  "imperative"      : command  (mange ! partez ! va-t-en !)
  "prohibitive"     : negative command  (ne mange pas ! n'y va pas !)

ADNOMINAL:
  "relative_min"    : qui/que clause  (l'homme qui mange, la femme que j'aime)
  "relative_post"   : postnominal relative  (un repas qui augmente le lait)
  "participial_len" : past participle modifier  (l'homme blessé, la porte fermée)
  "participial_bali": without/lacking  (sans éducation, illettré)
  "participial_to"  : while doing  (en passant, tout en mangeant)

QUOTATIVE:
  "reported_verb"   : direct speech verb  (il a dit: je pars)
  "reported_comp"   : that-clause  (il a dit que tu viendras)

NOUN PHRASE ONLY:
  "noun_phrase"     : alienable possession  (la maison de mon mari, sa voiture)
  "noun_phrase_inh" : inalienable possession  (mon père, sa tête, son bras)
 "noun_phrase_have": avoir/posséder un objet, une mesure ou une quantité physique
                      (j'ai de l'argent, il a une voiture, quel âge as-tu ?,
                       combien d'enfants as-tu ?, tu as combien de frères ?)
                      NOTE: 'quel âge as-tu ?' = possession matérielle + quantité → noun_phrase_have

PRAGMATICS:
  "topicalised"     : topic fronted  (Bamako, ses habitants sont nombreux)
  "focus"           : emphatic focus  (c'est MOI qui l'ai fait)
  "exclamative"     : exclamation  (comme c'est beau ! que c'est bon !)

INTERROGATIVES:
  "interrogative"   : yes/no question  (tu veux manger ? est-ce qu'il vient ?)
  "content_question": wh-question  (qui mange ? que fais-tu ? où vas-tu ?)

SUBORDINATION:
  "conditional"     : si-clause  (si tu manges, si vous venez)
  "concessive"      : même si / bien que  (même s'il pleut)
  "causal"          : parce que / car  (parce qu'il est malade)
  "temporal"        : quand / lorsque  (quand il arrive)

COMPARISON:
  "comparison"      : plus ADJ que  (le cheval est plus rapide que l'âne)
  "superlative"     : le plus ADJ  (c'est le plus rapide de tous)

COMITATIVE:
  "comitative"      : avec  (je suis avec mon mari, je mange avec lui)
  "reciprocal"      : each other / l'un l'autre  (nous nous aimons, ils se battent, vous vous parlez)

2. "tokens": {surface, lemma} for EVERY word.
   Verbs → always infinitive (salués→saluer NOT saler, blessé→blesser, dit→dire).
   Past participles → infinitive NOT noun form.
   Gerunds (en V-ant) → infinitive (en passant→passer).
   Nouns/adj/adv → unchanged. Pronouns → unchanged.

EXAMPLES:
"je suis belle" -> {"clause_type":"qualitative","tokens":[{"surface":"je","lemma":"je"},{"surface":"suis","lemma":"être"},{"surface":"belle","lemma":"beau"}]}
"Musa est un chasseur" -> {"clause_type":"equative","tokens":[{"surface":"Musa","lemma":"Musa"},{"surface":"est","lemma":"être"},{"surface":"un","lemma":"un"},{"surface":"chasseur","lemma":"chasseur"}]}
"Musa est au village" -> {"clause_type":"locative","tokens":[{"surface":"Musa","lemma":"Musa"},{"surface":"est","lemma":"être"},{"surface":"au","lemma":"au"},{"surface":"village","lemma":"village"}]}
"c'est Musa" -> {"clause_type":"presentative","tokens":[{"surface":"c'","lemma":"ce"},{"surface":"est","lemma":"être"},{"surface":"Musa","lemma":"Musa"}]}
"je mange du riz" -> {"clause_type":"simple","tokens":[{"surface":"je","lemma":"je"},{"surface":"mange","lemma":"manger"},{"surface":"du","lemma":"du"},{"surface":"riz","lemma":"riz"}]}
"je veux manger" -> {"clause_type":"complex","tokens":[{"surface":"je","lemma":"je"},{"surface":"veux","lemma":"vouloir"},{"surface":"manger","lemma":"manger"}]}
"il a dit que tu viendras" -> {"clause_type":"reported_comp","tokens":[{"surface":"il","lemma":"il"},{"surface":"a","lemma":"avoir"},{"surface":"dit","lemma":"dire"},{"surface":"que","lemma":"que"},{"surface":"tu","lemma":"tu"},{"surface":"viendras","lemma":"venir"}]}
"si tu manges" -> {"clause_type":"conditional","tokens":[{"surface":"si","lemma":"si"},{"surface":"tu","lemma":"tu"},{"surface":"manges","lemma":"manger"}]}
"mange !" -> {"clause_type":"imperative","tokens":[{"surface":"mange","lemma":"manger"}]}
"je suis avec mon mari" -> {"clause_type":"comitative","tokens":[{"surface":"je","lemma":"je"},{"surface":"suis","lemma":"être"},{"surface":"avec","lemma":"avec"},{"surface":"mon","lemma":"mon"},{"surface":"mari","lemma":"mari"}]}
"il y a un problème" -> {"clause_type":"existential","tokens":[{"surface":"il","lemma":"il"},{"surface":"y","lemma":"y"},{"surface":"a","lemma":"avoir"},{"surface":"un","lemma":"un"},{"surface":"problème","lemma":"problème"}]}
"quel âge as-tu ?" -> {"clause_type":"noun_phrase_have","tokens":[{"surface":"quel","lemma":"quel"},{"surface":"âge","lemma":"âge"},{"surface":"as","lemma":"avoir"},{"surface":"tu","lemma":"tu"}]}
"nous nous aimons" -> {"clause_type":"reciprocal","tokens":[{"surface":"nous","lemma":"nous"},{"surface":"nous","lemma":"nous"},{"surface":"aimons","lemma":"aimer"}]}

Return ONLY valid JSON. No explanation. No markdown."""

_USER_TEMPLATE = 'Sentence: "{sentence}"'

# ── Valid clause types ────────────────────────────────────────────

VALID_CLAUSE_TYPES = {
    'simple', 'complex', 'verb_serial', 'purposive',
    'qualitative', 'equative', 'locative', 'presentative', 'existential',
    'passive', 'refl_past', 'imperative', 'prohibitive',
    'relative_min', 'relative_post',
    'participial_len', 'participial_ta', 'participial_bali', 'participial_to',
    'reported_verb', 'reported_comp',
    'noun_phrase', 'noun_phrase_inh', 'noun_phrase_have',
    'topicalised', 'focus', 'exclamative',
    'interrogative', 'content_question',
    'conditional', 'concessive', 'causal', 'temporal',
    'comparison', 'superlative',
    'comitative', 'refl_past', 'reciprocal',
}


# ── JSON extraction helpers ───────────────────────────────────────

def _extract_tokens(data) -> list:
    """Extract token list from parsed JSON."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ('tokens', 'words', 'result', 'lemmas'):
            if key in data and isinstance(data[key], list):
                return data[key]
        # Flat dict with surface keys
        for v in data.values():
            if isinstance(v, list):
                return v
    return []


def _extract_clause_type(data) -> str:
    """Extract clause_type from parsed JSON."""
    if isinstance(data, dict):
        ct = data.get('clause_type', '')
        if ct in VALID_CLAUSE_TYPES:
            return ct
    return 'simple'  # safe default


def _parse_response(text: str) -> dict:
    """
    Parse LLM response into {clause_type, tokens}.
    Handles messy JSON, markdown code blocks, etc.
    """
    # Strip markdown fences
    text = re.sub(r'```(?:json)?', '', text).strip()

    # Try direct parse
    try:
        data = json.loads(text)
        return {
            'clause_type': _extract_clause_type(data),
            'tokens':      _extract_tokens(data),
        }
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return {
                'clause_type': _extract_clause_type(data),
                'tokens':      _extract_tokens(data),
            }
        except json.JSONDecodeError:
            pass

    # Try extracting JSON array (legacy format)
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            tokens = json.loads(match.group())
            if isinstance(tokens, list):
                return {'clause_type': 'simple', 'tokens': tokens}
        except json.JSONDecodeError:
            pass

    return {'clause_type': 'simple', 'tokens': []}


# ── LLM backends ─────────────────────────────────────────────────

def _prompt(sentence: str) -> str:
    return _SYSTEM_PROMPT + '\n\n' + _USER_TEMPLATE.format(sentence=sentence)


_parser_circuit_open_until: float = 0.0   # epoch time; 0 = circuit closed
_PARSER_RETRY_GAP = 60                    # seconds before retrying


def _call_ollama(sentence: str, model: str = 'qwen2.5:3b') -> dict:
    global _parser_circuit_open_until
    import time as _time
    if _time.time() < _parser_circuit_open_until:
        return {}
    import requests
    payload = {
        'model':  model,
        'prompt': _prompt(sentence),
        'stream': False,
        'format': 'json',
        'options': {'temperature': 0, 'num_predict': 512},
    }
    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json=payload, timeout=60)
        return _parse_response(response.json().get('response', ''))
    except (requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError):
        _parser_circuit_open_until = _time.time() + _PARSER_RETRY_GAP
        print(f"⚠️  Ollama morphologique indisponible (réessai dans {_PARSER_RETRY_GAP}s)")
        return {}

# ── Main parser class ─────────────────────────────────────────────

class MorphologicalParser:
    """
    LLM-based morphological parser.
    Returns {clause_type, tokens} for each sentence.
    Results cached in Neo4j ParseCache nodes.
    """

    def __init__(self, db, backend: str = 'ollama', model: str = None):
        self.db      = db
        self.backend = backend
        self.model   = model

    def parse(self, sentence: str) -> dict:
        """
        Parse sentence. Returns:
        {
          'clause_type': 'simple' | 'qualitative' | 'equative' | ...
          'tokens':      [{'surface': 'je', 'lemma': 'je'}, ...]
        }
        """
        # Cache disabled — always call LLM for fresh results
        return self._call(sentence)

    def _call(self, sentence: str) -> dict:
        """Ollama uniquement — pas de fallback Gemini."""
        if self.backend == 'ollama':
            result = _call_ollama(sentence, self.model or 'qwen2.5:3b')
            if result:
                return result
        return {'clause_type': 'simple', 'tokens': []}