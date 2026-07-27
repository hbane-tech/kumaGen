"""
kg/ingest_grammar.py
Ingests all grammatical nodes into the KG from CSV files.
Zero hardcoded lexicons — all data lives in data/ folder.

Run once after main.py:
    python -m kg.ingest_grammar
"""

import csv
import os
from kg.neo4j_client import Neo4jClient

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def _load_csv(filename):
    path = os.path.join(_DATA, filename)
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def run():
    db = Neo4jClient()

    # Pronouns
    pronouns = _load_csv('pronouns.csv')
    print(f"Ingesting {len(pronouns)} pronouns...")
    for row in pronouns:
        db.query("""
        MERGE (p:Pronoun {surface: $surface, lang: $lang, role: $role})
        SET p.bm = $bm
        """, row)
    print("   Done")

    # Negation markers
    markers = _load_csv('negation_markers.csv')
    print(f"Ingesting {len(markers)} negation markers...")
    for row in markers:
        db.query("""
        MERGE (n:NegMarker {surface: $surface, lang: $lang})
        """, row)
    print("   Done")

    # Articles
    articles = _load_csv('articles.csv')
    print(f"Ingesting {len(articles)} articles...")
    for row in articles:
        db.query("""
        MERGE (a:Article {surface: $surface, lang: $lang})
        """, row)
    print("   Done")

    # Prepositions
    prepositions = _load_csv('prepositions.csv')
    print(f"Ingesting {len(prepositions)} prepositions...")
    for row in prepositions:
        db.query("""
        MERGE (p:Preposition {surface: $surface, lang: $lang})
        SET p.bm_marker = $bm_marker, p.role = $role
        """, row)
    print("   Done")

    # Function words
    function_words = _load_csv('function_words.csv')
    print(f"Ingesting {len(function_words)} function words...")
    for row in function_words:
        db.query("""
        MERGE (f:FunctionWord {surface: $surface, lang: $lang})
        SET f.bm = $bm, f.role = $role
        """, row)
    print("   Done")

    # Auxiliaries
    auxiliaries = _load_csv('auxiliaries.csv')
    print(f"Ingesting {len(auxiliaries)} auxiliaries...")
    for row in auxiliaries:
        db.query("""
        MERGE (a:Auxiliary {surface: $surface, lang: $lang})
        SET a.tense = $tense
        """, row)
    print("   Done")

    # Language markers
    # lang_markers = _load_csv('language_markers.csv')
    # print(f"Ingesting {len(lang_markers)} language markers...")
    # for row in lang_markers:
    #     db.query("""
    #     MERGE (m:LangMarker {surface: $surface, lang: $lang})
    #     """, row)
    # print("   Done")

    # Indexes
    db.query("CREATE INDEX pronoun_surface IF NOT EXISTS FOR (p:Pronoun) ON (p.surface)")
    db.query("CREATE INDEX negmarker_surface IF NOT EXISTS FOR (n:NegMarker) ON (n.surface)")
    db.query("CREATE INDEX article_surface IF NOT EXISTS FOR (a:Article) ON (a.surface)")
    db.query("CREATE INDEX preposition_surface IF NOT EXISTS FOR (p:Preposition) ON (p.surface)")
    db.query("CREATE INDEX functionword_surface IF NOT EXISTS FOR (f:FunctionWord) ON (f.surface)")
    db.query("CREATE INDEX auxiliary_surface IF NOT EXISTS FOR (a:Auxiliary) ON (a.surface)")
    print("   Indexes created")
    print("\n Grammar nodes ready.\n")


if __name__ == "__main__":
    run()