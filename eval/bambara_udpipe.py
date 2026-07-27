"""
eval/bambara_udpipe.py
Wrapper around the pretrained UDPipe model trained on the UD_Bambara treebank
(Aplonova & Tyers, https://github.com/KatyaAplonova/UD_Bambara) — gives real
UPOS tags and dependency relations for generated Bambara sentences, replacing
ad hoc keyword-based POS heuristics.

Model file: eval/ud/bambara.model (download once, see download_model()).
"""
import os

_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ud', 'bambara.model')
_MODEL_URL  = "https://raw.githubusercontent.com/KatyaAplonova/UD_Bambara/master/bambara.model"

_pipeline = None
_model    = None  # kept alive deliberately: Pipeline only holds a raw SWIG
                   # pointer into it, so if `model` were a local var it could
                   # be garbage-collected once _get_pipeline() returns,
                   # leaving Pipeline with a dangling reference (surfaces as
                   # a spurious "the model does not have a tokenizer!" error).


def download_model(dest: str = _MODEL_PATH) -> bool:
    """Download the pretrained UDPipe Bambara model if not already present."""
    if os.path.exists(dest):
        return True
    try:
        import urllib.request
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, dest)
        return os.path.exists(dest)
    except Exception as e:
        print(f"  Échec du téléchargement du modèle UDPipe bambara : {e}")
        return False


def _get_pipeline():
    global _pipeline, _model
    if _pipeline is not None:
        return _pipeline
    if not os.path.exists(_MODEL_PATH):
        if not download_model():
            return None
    try:
        from ufal.udpipe import Model, Pipeline
        _model = Model.load(_MODEL_PATH)
        if not _model:
            return None
        _pipeline = Pipeline(_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        return _pipeline
    except Exception as e:
        print(f"  Échec du chargement du modèle UDPipe bambara : {e}")
        return None


def parse_bambara(text: str) -> list[dict]:
    """
    Parse a Bambara sentence with the UD_Bambara-trained UDPipe model.
    Returns a list of token dicts: {id, form, lemma, upos, head, deprel}
    (id/head are 1-based CoNLL-U indices; head=0 means root).
    Returns [] if the model is unavailable or parsing fails.
    """
    if not text or not text.strip():
        return []
    pipeline = _get_pipeline()
    if pipeline is None:
        return []
    try:
        from ufal.udpipe import ProcessingError
        error = ProcessingError()
        conllu = pipeline.process(text, error)
        if error.occurred():
            return []
    except Exception:
        return []

    # UDPipe may split `text` into several CoNLL-U sentence blocks (each
    # restarting ids at 1) even when the caller intends a single sentence —
    # renumber globally and offset head references so the result is one flat,
    # internally-consistent tree instead of colliding/duplicate ids.
    tokens = []
    offset = 0
    block_ids = set()
    for line in conllu.splitlines():
        if not line:
            continue
        if line.startswith('#'):
            continue
        cols = line.split('\t')
        if len(cols) < 8:
            continue
        tok_id = cols[0]
        if '-' in tok_id or '.' in tok_id:
            continue  # multiword/empty CoNLL-U nodes
        local_id = int(tok_id)
        if local_id in block_ids:
            # new sentence block started (ids restarted at 1)
            offset += max(block_ids)
            block_ids = set()
        block_ids.add(local_id)
        head_local = int(cols[6]) if cols[6].isdigit() else 0
        tokens.append({
            'id':     local_id + offset,
            'form':   cols[1],
            'lemma':  cols[2],
            'upos':   cols[3],
            'head':   (head_local + offset) if head_local != 0 else 0,
            'deprel': cols[7],
        })
    return tokens


def pos_tree_string(tokens: list[dict]) -> str:
    """Space-separated UPOS sequence (same shape as the old heuristic output)."""
    return ' '.join(t['upos'] for t in tokens)


def dep_tree_string(tokens: list[dict]) -> str:
    """Compact serialization for CSV storage: 'form/UPOS/deprel->head form/UPOS/deprel->head ...'"""
    return ' '.join(f"{t['form']}/{t['upos']}/{t['deprel']}->{t['head']}" for t in tokens)


def parse_dep_tree_string(s: str) -> list[dict]:
    """Inverse of dep_tree_string(), for reading the CSV column back."""
    if not s or not s.strip():
        return []
    tokens = []
    for i, chunk in enumerate(s.split(), start=1):
        try:
            rest, head = chunk.rsplit('->', 1)
            form, upos, deprel = rest.split('/', 2)
            tokens.append({'id': i, 'form': form, 'upos': upos,
                            'deprel': deprel, 'head': int(head)})
        except ValueError:
            continue
    return tokens
