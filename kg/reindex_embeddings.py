"""
kg/reindex_embeddings.py

Re-encode all Sense.embedding values in Neo4j using the new LaBSE model (768 dim).
Run once after switching from paraphrase-multilingual-MiniLM-L12-v2 (384 dim)
to sentence-transformers/LaBSE (768 dim).

Usage:
    python3 -m kg.reindex_embeddings
"""

import sys
import numpy as np
from kg.neo4j_client import Neo4jClient
from embeddings.word2vec_encoder import encode_batch, DIM

BATCH = 256   # Sense nodes per Neo4j write batch


def _fetch_all_senses(db) -> list:
    """Return [{id, fr, en}] for every Sense node that has a French or English gloss."""
    rows = db.query("""
    MATCH (s:Sense)
    RETURN id(s) AS nid, s.fr AS fr, s.en AS en
    """)
    return rows


def _text_for_sense(row: dict) -> str:
    """
    Pick the best text to encode:
    prefer s.fr (French gloss), fall back to s.en (English gloss).
    """
    fr = (row.get('fr') or '').strip()
    en = (row.get('en') or '').strip()
    return fr or en


def _write_batch(db, batch: list) -> None:
    """
    Write a batch of {nid, embedding} to Neo4j.
    Uses UNWIND for efficient bulk update.
    """
    db.query("""
    UNWIND $rows AS row
    MATCH (s:Sense) WHERE id(s) = row.nid
    SET s.embedding = row.embedding
    """, {"rows": batch})


def run(db=None):
    if db is None:
        db = Neo4jClient()

    print("Fetching all Sense nodes…")
    senses = _fetch_all_senses(db)
    total = len(senses)
    print(f"  {total} Sense nodes found.")

    # Filter out senses with no text
    valid = [(row, _text_for_sense(row)) for row in senses]
    valid = [(row, txt) for row, txt in valid if txt]
    print(f"  {len(valid)} have text to encode.")

    processed = 0
    write_batch = []

    # Process in encode_batch chunks
    for start in range(0, len(valid), BATCH):
        chunk = valid[start:start + BATCH]
        texts = [txt for _, txt in chunk]

        vecs = encode_batch(texts)  # (BATCH, 768)

        for (row, _), vec in zip(chunk, vecs):
            # Sanity: skip zero vectors (empty text after clean_gloss)
            if np.linalg.norm(vec) < 1e-6:
                continue
            write_batch.append({
                "nid":       row['nid'],
                "embedding": vec.tolist(),
            })

        # Flush write batch every BATCH rows
        if write_batch:
            _write_batch(db, write_batch)
            processed += len(write_batch)
            write_batch = []

        pct = min(100, int((start + len(chunk)) / len(valid) * 100))
        print(f"  {pct:3d}% — encoded {processed}/{len(valid)}", end='\r', flush=True)

    # Final flush
    if write_batch:
        _write_batch(db, write_batch)
        processed += len(write_batch)

    print(f"\nDone. {processed} Sense nodes updated with {DIM}-dim LaBSE embeddings.")


if __name__ == '__main__':
    run()
