"""
Statif : ADJ + -len dòn / tùn bɛ
  Résultatif adnominal : root_tok.is_statif ou VerbForm=Part ou Tense=Past
  Si bm est un fallback [lemma], garder les crochets et ajouter len directement.
"""
def run(T, tree, m, processed_indices, G_kg, root_tok, aux_tense_tok):
    tree['clause_type'] = 'statif'
    _statif_base = (root_tok.get('statif_root')
                    or root_tok.get('bm')
                    or f"[{root_tok.get('lemma')}]")
    # Si bm est un fallback entre crochets → ne pas toucher, juste ajouter len
    if _statif_base.startswith('[') and _statif_base.endswith(']'):
        _statif_base = _statif_base + 'len'
    else:
        # Strip -a final avant -len (sauf ba/ma/ka)
        if (_statif_base.endswith('a')
                and not _statif_base.endswith('ba')
                and not _statif_base.endswith('ma')
                and not _statif_base.endswith('ka')):
            _statif_base = _statif_base[:-1]
        if not _statif_base.endswith('len'):
            _statif_base += 'len'
    m['QUAL'] = _statif_base
    _statif_is_past = (aux_tense_tok is not None
                       and aux_tense_tok.get('tense') in ('past', 'hab'))
    if _statif_is_past:
        tree['tam']         = 'tùn tɛ' if tree.get('neg') else 'tùn bɛ'
        tree['clause_type'] = 'statif_past'
    else:
        tree['tam'] = 'tɛ' if tree.get('neg') else 'dòn'
    processed_indices.add(root_tok['orig_index'])