"""être ROOT sans copule séparée → devient copule."""
from rules.core import _is_copula
from rules.core import j


def detect(T, root_tok, copula_tok, m, processed_indices, G_kg):
    _etre_as_root_cop = (
        root_tok and _is_copula(root_tok)
        and root_tok.get('pos') in ('VERB', 'AUX')
        and not copula_tok
        and not any(x.get('role') == 'interrogative' for x in T)
        and not any(x.get('dep') == 'case' and x.get('role') == 'locative' for x in T)
        and not any(x.get('is_loc') for x in T))
    if not _etre_as_root_cop:
        return copula_tok
    copula_tok = root_tok
    _cop_attr = next((x for x in T
                      if x.get('dep') in ('attr', 'xcomp', 'nsubj', 'ROOT')
                      and x.get('pos') in ('NOUN', 'PROPN', 'ADJ', 'PRON')
                      and x.get('orig_index') != root_tok['orig_index']
                      and x.get('orig_index') not in processed_indices), None)
    if _cop_attr:
        m['O'] = _cop_attr.get('bm') or f"[{_cop_attr.get('lemma')}]"
        processed_indices.add(_cop_attr['orig_index'])
    if not m.get('O') and m.get('S'):
        _s_tok = next((x for x in T if x.get('bm') == m['S']
                       and x.get('dep') in ('nsubj', 'attr')), None)
        _real_subj = next((x for x in T if x.get('dep') == 'nsubj'
                           and x.get('pos') == 'PRON'), None)
        if _s_tok and _real_subj and _s_tok != _real_subj:
            m['O'] = m['S']
            m['S'] = _real_subj.get('bm') or f"[{_real_subj.get('lemma')}]"
    return copula_tok
