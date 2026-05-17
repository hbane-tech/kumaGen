"""
interactive_translate.py
Interactive French -> Bambara translation shell.
Shows top 5 candidates per token so you can validate or override.
"""

from pipeline.translation_engine import TranslationEngine
from kg.neo4j_client import Neo4jClient


def main():
    db     = Neo4jClient()
    engine = TranslationEngine(db)

    print("\n Sense-Based Bambara Translator")
    print("   Shows top 5 candidates per token.")
    print("   Type a French sentence. 'exit' to quit.\n")

    while True:
        try:
            text = input("FR > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not text:
            continue
        if text.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        result = engine.translate(text)
        print(f"\n Result : {result['bambara']}")
        # print(f"   Frame  : {result['frame']}")
        print("-" * 65 + "\n")


if __name__ == "__main__":
    main()
