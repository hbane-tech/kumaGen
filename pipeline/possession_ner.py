"""
pipeline/possession_ner.py
Détecte la catégorie NER (PER/ORG/LOC/DATE/VALUE/OTHER) du possesseur et du
possédé dans une relation génitive nominale ("X de Y" / "X du Y"), et tranche
l'aliénabilité ('ka' ou non) via la table possesseur×possédé définie avec
l'utilisateur (2026-07-07) — remplace l'ancienne heuristique lexicale
(parenté/corps vs objets physiques) qui ne regardait que le nom possédé et
ignorait entièrement le possesseur.

Le parseur de dépendances principal (fr_dep_news_trf, voir spacy_parser.py)
n'a pas de composant NER. On charge donc ici un second pipeline spaCy
(fr_core_news_lg) uniquement pour cette classification par catégorie —
jamais pour reparser la phrase entière.

PROD (produits/marques) n'a pas de détecteur fiable sans liste ni LLM :
aucun mot n'est actuellement classé PROD, les cases PROD de la table
restent donc pour l'instant inatteignables sauf extension future.

ZÉRO HARDCODE (2026-07-07) : aucune liste de mots (mois, jours, monnaies,
parenté...) n'est écrite en dur ici. spaCy fournit seulement 4 catégories
NER pour le français (LOC/MISC/ORG/PER, pas de DATE/VALUE — vérifié via
`nlp.get_pipe('ner').labels`), et la similarité vectorielle de mots
(nlp(x).similarity(nlp(y))) s'est révélée trop bruitée pour trancher de
façon fiable (ex: "voiture" plus proche de "personne" que "frère" ne l'est).
Faute d'un signal spaCy fiable, DATE/VALUE ne sont détectés que par des
signaux structurels génériques (regex année, token.is_currency).

PER pour un nom COMMUN (pas une entité nommée) — ex: "frère" désignant
une personne sans être un nom propre — n'a pas non plus de signal fiable
sans LLM : essayés et rejetés dans l'ordre : spaCy NER (ne tague que les
noms propres), Stanza NER (même limite, vérifié empiriquement — modèle
"wikinergold", entraîné sur le même type de corpus), similarité vectorielle
(trop bruitée), LEFFF (parenté suppléfive comme frère/sœur : deux lemmes
sans lien dérivationnel, indétectable par appariement de genre). Un appel
LLM optionnel (`person_classify_fn`) est donc utilisé en dernier recours,
uniquement si spaCy NER + regex + is_currency n'ont rien trouvé.

Cas des adjectifs substantivés ("le vieux" = le vieil homme) : le
signal structurel (ADJ employé sans nom tête, comme possesseur nominal
d'un génitif) est transmis au LLM comme fait grammatical par l'appelant
— jamais une liste de mots pré-remplie — pour l'aider à trancher sans
biaiser le prompt vers un lexique fixe.
"""
import re

_ner_nlp_cache = {}


def _get_ner_nlp():
    if 'fr' not in _ner_nlp_cache:
        import spacy
        for _name in ('fr_core_news_lg', 'fr_core_news_md', 'fr_core_news_sm'):
            try:
                _ner_nlp_cache['fr'] = spacy.load(_name)
                break
            except Exception:
                continue
        else:
            _ner_nlp_cache['fr'] = None
    return _ner_nlp_cache['fr']


# spaCy French NER ne labellise nativement que LOC/MISC/ORG/PER.
_SPACY_NER_MAP = {'PER': 'PER', 'ORG': 'ORG', 'LOC': 'LOC'}
_YEAR_RE = re.compile(r'^\d{3,4}$')


def detect_category(surface: str, person_classify_fn=None) -> str:
    """
    Retourne PER / ORG / LOC / DATE / VALUE / OTHER pour un nom donné.

    person_classify_fn : callable(str) -> bool optionnel, appelé en dernier
    recours (dernier recours = spaCy NER + signaux structurels n'ont rien
    donné) pour trancher si un nom COMMUN désigne une personne (ex: "frère")
    — voir docstring module pour pourquoi aucun signal non-LLM ne marche ici.
    """
    surf = (surface or '').strip()

    # DATE : signal structurel générique (année à 3-4 chiffres), pas de liste
    # de mois/jours — un token purement numérique de cette forme est une
    # année dans la quasi-totalité des cas.
    if _YEAR_RE.match(surf):
        return 'DATE'

    nlp = _get_ner_nlp()
    if nlp is not None and surf:
        doc = nlp(surf)
        # VALUE : signal structurel spaCy (symbole monétaire reconnu nativement
        # par is_currency), pas de liste de noms de devises.
        if any(t.is_currency for t in doc):
            return 'VALUE'
        for ent in doc.ents:
            if ent.label_ in _SPACY_NER_MAP:
                return _SPACY_NER_MAP[ent.label_]

    if person_classify_fn is not None and surf and person_classify_fn(surf):
        return 'PER'

    return 'OTHER'


# possesseur -> possédé -> aliénable (True = 'ka' requis)
# Table exacte fournie par l'utilisateur (2026-07-07) ; DATE→VALUE et
# VALUE→VALUE n'étaient pas spécifiés (marqués incertains) — mis à
# NON-ALIÉNABLE par défaut, cohérent avec le reste de leurs deux lignes.
_TABLE = {
    'PER':   {'PER': False, 'ORG': True,  'LOC': True,  'DATE': True,
              'PROD': True,  'VALUE': True,  'OTHER': True},
    'ORG':   {'PER': False, 'ORG': False, 'LOC': False, 'DATE': False,
              'PROD': True,  'VALUE': False, 'OTHER': False},
    'LOC':   {'PER': False, 'ORG': False, 'LOC': False, 'DATE': False,
              'PROD': False, 'VALUE': False, 'OTHER': False},
    'DATE':  {'PER': False, 'ORG': False, 'LOC': False, 'DATE': False,
              'PROD': False, 'VALUE': False, 'OTHER': False},
    'PROD':  {'PER': False, 'ORG': False, 'LOC': False, 'DATE': False,
              'PROD': False, 'VALUE': False, 'OTHER': False},
    'VALUE': {'PER': False, 'ORG': False, 'LOC': False, 'DATE': False,
              'PROD': False, 'VALUE': False, 'OTHER': False},
    'OTHER': {'PER': False, 'ORG': False, 'LOC': False, 'DATE': False,
              'PROD': False, 'VALUE': False, 'OTHER': False},
}


def is_alienable(possessor_cat: str, possessed_cat: str) -> bool:
    return _TABLE.get(possessor_cat, {}).get(possessed_cat, False)
