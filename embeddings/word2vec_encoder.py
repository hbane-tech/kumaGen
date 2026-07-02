"""
embeddings/word2vec_encoder.py

Multilingual BERT sentence encoder for KG semantic retrieval.

Model: LaBSE (Language-agnostic BERT Sentence Embedding)
- BERT-based architecture, fine-tuned for cross-lingual sentence similarity
- 768 dimensions (vs 384 for the previous MiniLM model)
- Supports French and English (needed for French queries + English KG glosses)
- Trained on 109 languages with translation pairs — strong cross-lingual alignment
- Runs fully local, no API key
"""

import numpy as np
from utils.normalize import clean_gloss

_model = None
MODEL_NAME = "sentence-transformers/LaBSE"
DIM = 768


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading BERT sentence encoder '{MODEL_NAME}'...")
        _model = SentenceTransformer(MODEL_NAME)
        print("Model loaded.")
    return _model


def encode(text: str) -> np.ndarray:
    """
    Encode text with LaBSE BERT encoder.
    Returns a 768-dim float32 vector, or a zero vector for empty input.
    """
    if not isinstance(text, str) or not text.strip():
        return np.zeros(DIM, dtype=np.float32)

    cleaned = clean_gloss(text.strip())
    if not cleaned:
        return np.zeros(DIM, dtype=np.float32)

    model = _get_model()
    vec = model.encode(cleaned, convert_to_numpy=True)
    return vec.astype(np.float32)


def encode_batch(texts: list) -> np.ndarray:
    """
    Encode a list of strings in one efficient batch call.
    Returns a (N, 768) float32 array.
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
