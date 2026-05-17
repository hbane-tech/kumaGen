"""
main.py
Step 2: Ingest the Bamadaba lexicon into Neo4j.
Run AFTER train_embeddings.py.
"""

from pipeline.ingestion_pipeline import IngestionPipeline
from kg.neo4j_client import Neo4jClient


def build_kg():
    db = Neo4jClient()
    pipeline = IngestionPipeline(db)
    pipeline.run()


if __name__ == "__main__":
    build_kg()
