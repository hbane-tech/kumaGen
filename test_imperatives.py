"""Quick test for imperative/object-pronoun fixes."""
import sys
sys.path.insert(0, '.')
from pipeline.translation_engine import TranslationEngine
from kg.neo4j_client import Neo4jClient

db = Neo4jClient()
engine = TranslationEngine(db)

tests = [
    ('Dis lui',             'láfɔ a yé'),
    ('Donne lui',           'di a ma'),
    ('prends le',           'a mɔ́n'),
    ('il les avait',        'u tùn bɛ a bóló'),
    ('donne lui son argent','a ka wári di a ma'),
    ('complètes en venant', 'kɔ́nbilen nàtɔ'),
]

for fr, expected in tests:
    got = engine.translate(fr)['bambara']
    ok = '✓' if got == expected else '✗'
    print(f'  {ok} {fr!r}')
    if got != expected:
        print(f'       got:      {got!r}')
        print(f'       expected: {expected!r}')
