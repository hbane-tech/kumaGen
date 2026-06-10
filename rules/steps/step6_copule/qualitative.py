"""
Qualitative bambara : S ka/man ADJ
Markers : ka (pos) / man (neg) + advmod du ROOT.
"""
from rules.core import j


def run(T, tree, m, processed_indices, G_kg, root_tok):
    tree['clause_type'] = 'qualitative'
    m['QUAL'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
    _qual_advs = [x for x in T if x.get('dep') == 'advmod'
                  and x.get('head_index') == root_tok['orig_index'] and x.get('bm')]
    for _qa in _qual_advs:
        m['QUAL'] = j(m['QUAL'], _qa.get('bm'))
        processed_indices.add(_qa['orig_index'])
    tree['tam'] = 'man' if tree.get('neg') else 'ka'
    if root_tok['orig_index'] in processed_indices:
        m['V'] = ''
        m['O'] = ''
