"""
Identificatoire / focus :
  run_adj_expletif → c'est + ADJ + expletif → équatif ou participe résultatif
  (focus_marker dè géré dans tree_to_bambara via slot SLOTS)
"""
from rules.core import j, _resolve_tam
from rules.steps.step6_copule import participes


def run_adj_expletif(T, tree, m, processed_indices, G_kg, root_tok, _has_expletive):
    _adj_root_tok = next((x for x in T if x.get('pos') == 'ADJ'
                          and x.get('dep') == 'ROOT'), None)
    if not (_adj_root_tok and _has_expletive
            and tree.get('clause_type') not in ('equative', 'statif', 'locative',
                                                'presentative', 'identificatory')):
        return
    if _adj_root_tok.get('is_statif') or _adj_root_tok.get('is_valeur'):
        return
    if _adj_root_tok.get('is_participe_passe') or (
            not _adj_root_tok.get('is_valeur')
            and not _adj_root_tok.get('is_statif')
            and _adj_root_tok.get('bm', '').startswith('[')):
        tree['clause_type']  = 'simple'
        tree['is_transitive'] = False
        tree['tense']        = 'past'
        tree['tam']          = ''
        m['V'] = _adj_root_tok.get('bm') or f"[{_adj_root_tok.get('lemma')}]"
        m['S'] = next((x.get('bm', '') for x in T
                       if x.get('role') == 'expletive' and x.get('bm')), 'o')
        processed_indices.add(_adj_root_tok['orig_index'])
    elif _adj_root_tok.get('is_nominal_adj'):
        # ADJ avec article (c'est le vrai) → rôle nominal → identificatoire : bɛ́rɛ dòn
        tree['clause_type'] = 'identificatory'
        m['S'] = _adj_root_tok.get('bm') or f"[{_adj_root_tok.get('lemma')}]"
        m['O'] = ''
        m['V'] = ''
        tree['tam'] = 'tɛ' if tree.get('neg') else 'dòn'
        processed_indices.add(_adj_root_tok['orig_index'])
        for _ex in T:
            if _ex.get('role') == 'expletive':
                processed_indices.add(_ex['orig_index'])
    else:
        # ADJ + expletif (c'est bon) → qualitatif (o ka ADJ), pas équatif
        tree['clause_type'] = 'qualitative'
        m['QUAL'] = _adj_root_tok.get('bm') or f"[{_adj_root_tok.get('lemma')}]"
        tree['tam'] = _resolve_tam('pres', True, G_kg) if tree.get('neg') else 'ka'
        if not m.get('S'):
            _expl_tok = next((x for x in T if x.get('role') == 'expletive'
                              and x.get('bm')), None)
            if _expl_tok:
                m['S'] = _expl_tok.get('bm')
                processed_indices.add(_expl_tok['orig_index'])
        m['V'] = ''
        m['O'] = ''
        processed_indices.add(_adj_root_tok['orig_index'])
