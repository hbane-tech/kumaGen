"""
Équatif bambara : yé...yé / tùn yé / tɛ yé / tùn tɛ
  run_valeur    → is_valeur (vrai, faux, possible)
  run_default   → else branch (NOUN/PROPN attribut, équatif passé)
"""
from rules.core import j, _resolve_tam


def run_valeur(T, tree, m, processed_indices, G_kg, root_tok):
    tree['clause_type'] = 'equative'
    m['O'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
    tree['tam'] = (_resolve_tam('pres', tree.get('neg', False), G_kg)
                   if tree.get('neg') else G_kg.get('equative_marker', 'yé'))
    processed_indices.add(root_tok['orig_index'])
    if not m.get('S'):
        _expl_tok = next((x for x in T if x.get('role') == 'expletive'
                          and x.get('bm')), None)
        if _expl_tok:
            m['S'] = _expl_tok.get('bm')
            processed_indices.add(_expl_tok['orig_index'])


def run_default(T, tree, m, processed_indices, G_kg, root_tok, aux_tense_tok):
    if tree.get('clause_type') == 'locative':
        return
    tree['clause_type'] = 'equative'
    # Équatif passé : tùn yé / tùn tɛ
    if tree.get('tense') in ('past', 'hab', 'plup'):
        _eq_neg      = tree.get('neg', False)
        _eq_tun      = _resolve_tam(tree['tense'], False, G_kg) or 'tùn bɛ'
        _eq_tun_base = _eq_tun.split()[0]  # 'tùn'
        tree['tam']  = j(_eq_tun_base, 'tɛ' if _eq_neg else 'yé')
    elif tree.get('neg'):
        tree['tam'] = _resolve_tam('pres', True, G_kg)
    else:
        tree['tam'] = G_kg.get('equative_marker', 'yé')
    # Expletif (c'est X) : S = 'o' si aucun sujet défini
    if not m.get('S') and any(x.get('role') == 'expletive' for x in T):
        _expl_tok = next((x for x in T if x.get('role') == 'expletive'
                          and x.get('bm')), None)
        m['S'] = _expl_tok.get('bm', 'o') if _expl_tok else 'o'
