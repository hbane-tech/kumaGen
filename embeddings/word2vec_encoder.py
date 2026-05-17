"""
embeddings/word2vec_encoder.py

Uses a pretrained multilingual sentence-transformer model instead of
a custom Word2Vec. This gives real semantic similarity between French
words without requiring a large training corpus.

Model: paraphrase-multilingual-MiniLM-L12-v2
- Free, runs locally, no API key needed
- 384 dimensions
- Trained on 50+ languages including French
- Semantically meaningful: chien ~ canidé, manger ~ nourriture
"""

import numpy as np
from utils.normalize import clean_gloss

_model = None
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading sentence-transformer model '{MODEL_NAME}'...")
        _model = SentenceTransformer(MODEL_NAME)
        print("Model loaded.")
    return _model


def encode(text: str) -> np.ndarray:
    """
    Encode a text string using the multilingual sentence-transformer.
    Returns a 384-dim float32 vector.
    Returns a zero vector for empty input.
    """
    if not isinstance(text, str) or not text.strip():
        return np.zeros(384, dtype=np.float32)

    cleaned = clean_gloss(text.strip())
    if not cleaned:
        return np.zeros(384, dtype=np.float32)

    model = _get_model()
    vec = model.encode(cleaned, convert_to_numpy=True)
    return vec.astype(np.float32)


def encode_batch(texts: list) -> np.ndarray:
    """
    Encode a list of strings in one efficient batch call.
    Returns a (N, 384) float32 array.
    """
    cleaned = [clean_gloss(t.strip()) if isinstance(t, str) else "" for t in texts]
    model = _get_model()
    vecs = model.encode(cleaned, convert_to_numpy=True, batch_size=64,
                        show_progress_bar=True)
    return vecs.astype(np.float32)


def reload_model():
    global _model
    _model = None
    return _get_model()
