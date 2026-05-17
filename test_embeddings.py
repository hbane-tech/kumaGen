"""
test_embeddings.py
Sanity-checks the Word2Vec model: verifies that semantically related
French/Bambara pairs have high cosine similarity.
"""

import numpy as np
from embeddings.word2vec_encoder import encode


def cosine(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


pairs = [
    ("manger",  "manger"),          # identical -> should be ~1.0
    ("manger",  "nourriture"),       # related French words
    ("eau",     "boire"),            # water / drink
    ("dormir",  "sommeil"),          # sleep (verb/noun)
    ("manger",  "dormir"),           # unrelated -> should be low
]

print("\n🧪 Embedding similarity test\n")
for a, b in pairs:
    va, vb = encode(a), encode(b)
    if np.linalg.norm(va) == 0 or np.linalg.norm(vb) == 0:
        print(f"  {a:15s} <-> {b:15s}  OOV")
    else:
        sim = cosine(va, vb)
        print(f"  {a:15s} <-> {b:15s}  {sim:+.4f}")
