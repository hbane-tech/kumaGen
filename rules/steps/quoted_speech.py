"""
rules/steps/quoted_speech.py
Discours rapporté cité entre guillemets : "il dit « bonjour »" →
S TAM V IOBJ yé ko "bonjour" (contenu cité verbatim, non traduit — cohérent
avec la protection par guillemets posée en amont dans translation_engine.py).

Doit tourner AVANT step4_objet : selon la présence d'un auxiliaire (temps
composé), spaCy tague le span cité tantôt dep='obl:arg' (présent, "il dit
bonjour"), tantôt dep='obj' (passé composé, "il m'a dit bonjour") — et
step4_objet consomme les tokens 'obj' avant que step5_obliques ne les voie.
Appelé tôt, ce module réserve les indices du span cité dans processed_indices
avant que quoi que ce soit d'autre ne puisse le happer.
"""


def handle(T, tree, m, processed_indices, G_kg, root_tok):
    _quote_marks = {'«', '»', '"', '“', '”'}
    _is_saying_root = (root_tok is not None
                        and (root_tok.get('semantic_class') == 'saying'
                             or root_tok.get('lemma') == 'dire'))
    if not _is_saying_root:
        return

    _quoted_toks = sorted(
        [x for x in T
         if x.get('is_protected')
         and str(x.get('surface', '')).strip() not in _quote_marks
         and x['orig_index'] not in processed_indices],
        key=lambda x: x['orig_index'])
    if not _quoted_toks:
        return

    _quote_text = ' '.join(x.get('surface', '') for x in _quoted_toks)
    _ko = G_kg.get('reported_intro', '') or 'ko'
    m.setdefault('OBL_ALL', []).append({
        'HEAD': _quote_text, 'MARKER': _ko, 'local_clause_type': 'simple',
        'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
        'DEP_TYPE': '', 'COMPOUND_IS_QUANTIFIER': False,
        'MARKER_IS_PREFIX': True,
    })
    for x in T:
        if (x.get('is_protected')
                and (x in _quoted_toks
                     or str(x.get('surface', '')).strip() in _quote_marks)):
            processed_indices.add(x['orig_index'])

    # Le contenu cité est déplacé hors de m['O'] vers OBL_ALL (ci-dessus).
    # 'saying' est dans INTRANS_SC (rules/core.py) pour les emplois VRAIMENT
    # intransitifs ("il a parlé", sans complément) : step7_final.py force
    # alors is_transitive=False au passé (résultatif V+suffixe, "fɔla").
    # Mais ici "dire" a un complément (la citation), juste déplacé hors de
    # m['O'] — sans ce flag, step7_final ne peut pas faire la différence et
    # écrase quand même en résultatif. _has_quoted_speech le signale
    # explicitement à step7_final pour qu'il saute cette règle.
    tree['is_transitive']     = True
    tree['_has_quoted_speech'] = True
