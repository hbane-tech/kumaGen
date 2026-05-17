"""
pipeline/tokenizer.py
Routes to SpacyParser which combines LLM lemma + spaCy structure.
"""

_parser = None


def _get_parser(db, backend: str, llm_model: str):
    global _parser
    if _parser is None:
        from pipeline.spacy_parser import SpacyParser
        _parser = SpacyParser(db, backend=backend, llm_model=llm_model)
    return _parser


def tokenize(sentence: str, db=None,
             backend: str = None, model: str = None) -> list:
    from config.settings import LLM_BACKEND, LLM_MODEL
    backend   = backend   or LLM_BACKEND
    llm_model = model     or LLM_MODEL

    parser = _get_parser(db, backend, llm_model)
    return parser.parse(sentence)


def spacy_available(lang: str = 'fr') -> bool:
    return True