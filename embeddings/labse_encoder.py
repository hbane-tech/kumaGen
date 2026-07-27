"""
embeddings/labse_encoder.py

French sentence encoder for KG semantic retrieval.
(Module kept as 'labse_encoder' for import stability — actual model below.)

Model: sentence-camembert-large (Lajavaness)
- CamemBERT fine-tuned via SBERT specifically for French sentence similarity
- 1024 dimensions
- Switched from LaBSE (multilingual, 768 dim) 2026-07-20 : LaBSE's cross-
  lingual alignment (109 languages) does not cleanly separate true French
  synonyms from superficially similar but unrelated compounds — e.g. for
  "gens" (people), LaBSE scores the unrelated compound "gens de caste"
  nearly as high as the true synonym "personne" (0.69 vs 0.69) and only
  modestly below "peuple" (0.83). sentence-camembert-large separates both
  synonyms from the wrong compound by a wide margin instead (0.72 and 0.64
  vs 0.33). Sense.en (English gloss) was found to be empty in practice, so
  LaBSE's cross-lingual capability wasn't even being used.
- Runs fully local, no API key
"""

import numpy as np
from utils.normalize import clean_gloss

_model = None
MODEL_NAME = "Lajavaness/sentence-camembert-large"
DIM = 1024


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
