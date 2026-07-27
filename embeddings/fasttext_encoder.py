"""
FastText subword model trained on le corpus des gloses KG (fr), utilisé
UNIQUEMENT pour détecter une parenté MORPHOLOGIQUE (même racine, suffixe
grammatical différent — ex. "coopératif"/"coopérative") — pas une similarité
sémantique générale (ça reste le rôle de CamemBERT, cf. labse_encoder.py).

Pourquoi un modèle entraîné maison plutôt qu'un modèle FR pré-entraîné
téléchargé : la composition par n-grammes de caractères capture la parenté
de racine même pour des mots hors-vocabulaire, sans les Go de téléchargement
d'un modèle FastText FR officiel — le corpus des ~13k gloses multi-mots du
KG suffit à entraîner cette distinction en quelques secondes.
"""
import os
import re

_MODEL = None
_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fasttext_fr.model')


def _train_and_save():
    from kg.neo4j_client import Neo4jClient
    from gensim.models import FastText

    db = Neo4jClient()
    res = db.query("MATCH (s:Sense) WHERE s.fr IS NOT NULL RETURN s.fr AS fr")
    sentences = []
    for r in res:
        txt = (r['fr'] or '').lower()
        words = re.findall(r"[a-zàâäéèêëïîôöùûüçœ]+", txt)
        if len(words) > 1:
            sentences.append(words)

    model = FastText(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=0,
        min_n=3,
        max_n=6,
        epochs=40,
        negative=10,
    )
    model.save(_MODEL_PATH)
    return model


def _get_model():
    global _MODEL
    if _MODEL is None:
        from gensim.models import FastText
        if os.path.exists(_MODEL_PATH):
            _MODEL = FastText.load(_MODEL_PATH)
        else:
            print("      [FASTTEXT] Entraînement du modèle de racines FR (une fois)...")
            _MODEL = _train_and_save()
    return _MODEL


def root_similarity(word_a: str, word_b: str) -> float:
    """Cosine de similarité subword entre deux mots (racine partagée)."""
    if not word_a or not word_b:
        return 0.0
    import numpy as np
    model = _get_model()
    va = model.wv[word_a.lower().strip()]
    vb = model.wv[word_b.lower().strip()]
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def same_root(word_a: str, word_b: str, threshold: float = 0.85) -> bool:
    """
    Seuil calibré sur des paires connues (cf. commentaire dans
    translation_engine.py) : coopératif/coopérative=0.99, actif/active=0.99,
    final/finale=0.87 (même racine, au-dessus du seuil) vs
    participant/participation=0.78, recruter/recrutement=0.69 (racine
    partagée mais PAS interchangeables telles quelles — traités par d'autres
    mécanismes dédiés, cf. règle 'tigi'), cuisson/cuisant=0.41, sur/sûr=0.31
    (mots différents malgré la ressemblance de forme).
    """
    return root_similarity(word_a, word_b) >= threshold
