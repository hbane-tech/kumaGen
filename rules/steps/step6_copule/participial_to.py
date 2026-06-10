"""
Participe simultané : -tɔ
Détection : role='participial_to' sur un token (simultaneous participle).
Ex : 'en mangeant' → S dúmúní kɛtɔ ...
Structure : S bɛ O V + '-tɔ' suffix → géré ici dans m['ADV'] ou V_SUFFIX.
"""
from rules.core import j


def run(T, tree, m, processed_indices, G_kg, root_tok):
    part_to_tok = next((x for x in T if x.get('role') == 'participial_to'
                        and x.get('bm')), None)
    if not part_to_tok:
        return
    s_bm = (next((x.get('bm') for x in T if x.get('dep') == 'nsubj'), None)
             or next((x.get('bm') for x in T if x.get('role') == 'pronoun'), None)
             or '')
    m['ADV'] = j(s_bm, part_to_tok.get('bm', '') + 'tɔ')
    processed_indices.add(part_to_tok['orig_index'])
