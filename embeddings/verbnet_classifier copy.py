"""
embeddings/verbnet_classifier.py

Classe sémantique (action/consumption/motion/...) d'un verbe français sans
LLM et sans cache KG : verbe français -> WOLF (Wordnet Libre du Français,
distribué dans nltk omw-1.4) -> synsets WordNet -> lemmes anglais -> classes
VerbNet (nltk). Le meilleur candidat (sens + classe) est choisi par
similarité d'embedding cross-lingue (le même sentence-transformer
multilingue déjà utilisé ailleurs dans le pipeline), pas par le premier
synset retourné par WOLF (ordre non fiable, voir _verbnet_classifier tests).

Pas de cache : les lookups WordNet/VerbNet sont des lectures locales
déterministes (pas d'appel réseau, pas de LLM) — aucun risque de figer une
valeur erronée/périmée comme c'était le cas avec le cache KG sur
semantic_class.
"""
from typing import Optional

from embeddings.word2vec_encoder import encode
from kg.retriever import cosine

_BUCKET_DESCRIPTIONS = {
    'motion':         "déplacement dans l'espace (aller, venir, courir, marcher)",
    'biological':     'processus vital du corps (vivre, mourir, naître, respirer)',
    'posture':        'position ou changement de position du corps (rester, dormir, se lever)',
    'spontaneous':    'réaction involontaire (rire, crier, pleurer)',
    'perception':     'voir, entendre, sentir, percevoir',
    'meteorological': 'phénomène atmosphérique (pleuvoir, neiger)',
    'copula':         'lien attributif (être, sembler, paraître)',
    'action':         'action intentionnelle (faire, donner, prendre, manger)',
    'consumption':    'ingestion ',
    'preparation':    'transformation, préparation (cuisiner, préparer)',
    'technique':      'travail spécialisé (construire, réparer)',
    'craft':          'création artistique (peindre, écrire)',
    'communication':  'parole (dire, raconter, demander)',
    'having':         'possession (avoir, posséder)',
}

_bucket_vecs_cache: Optional[dict] = None
_nltk_ready: Optional[bool] = None


def _ensure_nltk_data() -> bool:
    """Vérifie (et télécharge si besoin) verbnet/wordnet/omw-1.4."""
    global _nltk_ready
    if _nltk_ready is not None:
        return _nltk_ready
    try:
        import nltk
        for pkg in ('verbnet', 'wordnet', 'omw-1.4'):
            try:
                nltk.data.find(f'corpora/{pkg}')
            except LookupError:
                nltk.download(pkg, quiet=True)
        _nltk_ready = True
    except Exception:
        _nltk_ready = False
    return _nltk_ready


def _bucket_vecs() -> dict:
    global _bucket_vecs_cache
    if _bucket_vecs_cache is None:
        _bucket_vecs_cache = {b: encode(d) for b, d in _BUCKET_DESCRIPTIONS.items()}
    return _bucket_vecs_cache


def _bare_lemma(lemma_fr: str) -> str:
    bare = lemma_fr.rstrip('.').lower().strip()
    for prefix in ("se ", "s'"):
        if bare.startswith(prefix):
            return bare[len(prefix):]
    return bare


def _verb_synsets(lemma_fr: str):
    from nltk.corpus import wordnet as wn
    bare = _bare_lemma(lemma_fr)
    try:
        return [s for s in wn.synsets(bare, lang='fra') if s.pos() == 'v']
    except Exception:
        return []


def verbnet_semantic_class(lemma_fr: str) -> Optional[str]:
    """Bucket sémantique (une des clés de _BUCKET_DESCRIPTIONS) ou None si
    VerbNet/WOLF n'a aucune couverture pour ce verbe (le caller retombe
    alors sur 'other')."""
    if not _ensure_nltk_data():
        return None
    from nltk.corpus import verbnet as vn

    # Signal direct (appartenance à eat-39.1-1/-2, sans ambiguïté) :
    # prioritaire sur le classement par similarité ci-dessous, qui compare
    # des textes courts et peut bruiter même quand VerbNet est sans appel
    # (cf. 'manger' → eat-39.1-1 correctement identifié, mais le score
    # cosinus du texte de la classe contre les 14 descriptions de bucket
    # plaçait 'having'/'communication' devant 'consumption').
    if consumption_subtype(lemma_fr):
        return 'consumption'

    syns = _verb_synsets(lemma_fr)
    candidates = []
    for s in syns:
        for lem in s.lemma_names('eng'):
            for cid in vn.classids(lemma=lem.lower()):
                members = vn.lemmas(cid) or [cid.split('-')[0]]
                candidates.append((cid, s.definition() + ' ' + ' '.join(members)))
    if not candidates:
        return None

    lemma_vec = encode(lemma_fr)
    best_text = max(candidates, key=lambda c: cosine(lemma_vec, encode(c[1])))[1]

    bucket_vecs = _bucket_vecs()
    best_text_vec = encode(best_text)
    best_bucket = max(bucket_vecs.items(), key=lambda kv: cosine(best_text_vec, kv[1]))
    return best_bucket[0]


def consumption_subtype(lemma_fr: str) -> Optional[str]:
    """Distingue ingestion solide vs liquide via les classes VerbNet
    'eat-39.1-1' (membre unique : 'eat') et 'eat-39.1-2' (membre unique :
    'drink') — une distinction structurelle propre, pas une heuristique de
    notre cru. Retourne 'solid' | 'liquid' | None (verbe absent de WOLF)."""
    if not _ensure_nltk_data():
        return None
    syns = _verb_synsets(lemma_fr)
    en_lemmas = {lem.lower() for s in syns for lem in s.lemma_names('eng')}
    if 'drink' in en_lemmas:
        return 'liquid'
    if 'eat' in en_lemmas:
        return 'solid'
    return None
