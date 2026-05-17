"""
train_embeddings.py

With sentence-transformers there is nothing to train.
Just ensure the model is downloaded, then re-ingest.

Steps:
    pip install sentence-transformers
    python train_embeddings.py   (downloads model if needed)
    python main.py               (re-ingests KG with new embeddings)
"""

from embeddings.word2vec_encoder import _get_model
import numpy as np

model = _get_model()

# Quick sanity check
from embeddings.word2vec_encoder import encode
from kg.retriever import cosine

pairs = [
    ("chien",   "canidé",   "should be HIGH (synonyms)"),
    ("manger",  "nourriture","should be HIGH (related)"),
    ("chien",   "manger",   "should be LOW  (unrelated)"),
    ("eau",     "boire",    "should be HIGH (related)"),
    ("femme",   "uranium",  "should be VERY LOW"),
]

print("\n🧪 Semantic similarity sanity check\n")
for a, b, note in pairs:
    va, vb = encode(a), encode(b)
    c = cosine(va, vb)
    flag = "✅" if (
        ("HIGH" in note and c > 0.5) or
        ("LOW"  in note and c < 0.4) or
        ("VERY LOW" in note and c < 0.2)
    ) else "❌"
    print(f"  {flag} {a:12s} ~ {b:12s}: {c:.3f}  ({note})")

print("\n✅ Model ready. Now run: python main.py\n")
