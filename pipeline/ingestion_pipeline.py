"""
pipeline/ingestion_pipeline.py
Scrapes Bamadaba, builds Word/Sense/Concept/Frame nodes in Neo4j,
and stores clean Word2Vec embeddings on each Sense.
"""

import hashlib
import numpy as np

from scraper.bamadaba_scraper import scrape_all
from embeddings.word2vec_encoder import encode
from utils.normalize import clean_gloss, normalize_source


class IngestionPipeline:

    def __init__(self, db):
        self.db = db

    # -----------------------------
    # IDS
    # -----------------------------
    def get_sense_id(self, lemma, pos):
        return "S_" + hashlib.md5((lemma + pos).encode()).hexdigest()[:12]

    def get_concept_id(self, entry):
        base = entry.get("lemma") or ""
        return "C_" + hashlib.md5(base.encode()).hexdigest()[:12]

    # -----------------------------
    # FRAME DETECTION FROM POS
    # -----------------------------
    def infer_frame(self, entry):
        t = (entry.get("type") or "").lower()
        if "verb" in t:
            return "ACTION"
        if "noun" in t:
            return "ENTITY"
        if "adj" in t or "adv" in t:
            return "QUALITY"
        return "GENERIC"

    # -----------------------------
    # EMBEDDING SOURCE
    # Uses fr_emb (richer gloss) when available, falls back to fr.
    # Strips trailing punctuation so tokens match the Word2Vec vocab.
    # -----------------------------
    def build_embedding_source(self, entry):
        fr     = clean_gloss(entry.get("fr")     or "")
        fr_emb = clean_gloss(entry.get("fr_emb") or "")
        lemma  = (entry.get("lemma") or "").strip()

        # prefer the encyclopedic gloss for richer context
        gloss = fr_emb if fr_emb else fr

        parts = [p for p in [gloss, lemma] if p]
        raw   = " ".join(parts)
        return normalize_source(raw)   # lowercase + NFC

    # -----------------------------
    # ENSURE FRAME NODE EXISTS
    # -----------------------------
    def ensure_frame(self, frame):
        self.db.query("""
        MERGE (f:Frame {name: $frame})
        """, {"frame": frame})

    # -----------------------------
    # MAIN PIPELINE
    # -----------------------------
    def run(self):

        entries = scrape_all()
        print(f"\n📥 Ingesting {len(entries)} entries...\n")

        # Pre-create all Frame nodes
        for frame in ("ACTION", "ENTITY", "QUALITY", "GENERIC"):
            self.ensure_frame(frame)

        for i, e in enumerate(entries, 1):

            lemma = e.get("lemma") or ""
            fr    = e.get("fr")    or ""
            en    = e.get("en")    or ""
            pos   = e.get("type")  or "Other"

            sense_id   = self.get_sense_id(lemma, pos)
            concept_id = self.get_concept_id(e)
            frame      = self.infer_frame(e)

            embedding_source = self.build_embedding_source(e)
            emb = np.array(encode(embedding_source), dtype=np.float32)

            # -- WORD --
            self.db.query("""
            MERGE (w:Word {text: $w})
            """, {"w": lemma})

            # -- SENSE --
            self.db.query("""
            MERGE (s:Sense {id: $id})
            SET s.bm        = $bm,
                s.fr        = $fr,
                s.en        = $en,
                s.pos       = $pos,
                s.frame     = $frame,
                s.embedding = $emb
            """, {
                "id":    sense_id,
                "bm":    lemma,
                "fr":    fr,
                "en":    en,
                "pos":   pos,
                "frame": frame,
                "emb":   emb.tolist(),
            })

            # -- CONCEPT --
            self.db.query("""
            MERGE (c:Concept {id: $cid})
            """, {"cid": concept_id})

            # -- LINKS: Word->Sense->Concept --
            self.db.query("""
            MATCH (w:Word   {text: $w})
            MATCH (s:Sense  {id:   $sid})
            MATCH (c:Concept{id:   $cid})
            MERGE (w)-[:HAS_SENSE]->(s)
            MERGE (s)-[:MAPS_TO]  ->(c)
            """, {"w": lemma, "sid": sense_id, "cid": concept_id})

            # -- LINK: Sense->Frame (fixes FrameParser) --
            self.db.query("""
            MATCH (s:Sense {id:    $sid})
            MATCH (f:Frame {name:  $frame})
            MERGE (s)-[:IN_FRAME]->(f)
            """, {"sid": sense_id, "frame": frame})

            if i % 500 == 0:
                print(f"  ✔ {i}/{len(entries)} ingested...")

        print("\n✅ KG built successfully\n")
