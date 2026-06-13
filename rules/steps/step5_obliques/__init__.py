"""rules/steps/step5_obliques/__init__.py — orchestrateur obliques."""
from rules.steps.step5_obliques import advcl
from rules.steps.step5_obliques import relcl_boucle
from rules.steps.step5_obliques import comitative
from rules.steps.step5_obliques import privatif
from rules.steps.step5_obliques import wagon


def run(T, tree, m, processed_indices, G_kg, NX_G, root_tok):
    _loc_markers  = G_kg.get('locative_markers', set())
    _tmp_markers  = G_kg.get('temporal_markers', set())
    _clitic_roles = G_kg.get('clitic_roles', {'clitic'})

    for tok_item in sorted(T, key=lambda x: x['orig_index']):
        # Fusion week-end
        tok_item = wagon.fuse_weekend(tok_item, T, processed_indices, G_kg)

        if tok_item['orig_index'] in processed_indices:
            continue

        # advcl : purposive walasa ka / privative bali
        if tok_item.get('dep') == 'advcl':
            advcl.handle(tok_item, T, m, processed_indices, G_kg)
            continue

        # acl:relcl dans la boucle obliques
        if (tok_item.get('dep') == 'acl:relcl'
                and tok_item['orig_index'] not in processed_indices):
            relcl_boucle.handle(tok_item, T, m, processed_indices, G_kg, NX_G)
            continue

        # Filtres généraux
        if tok_item.get('pos') not in ('NOUN', 'PROPN', 'PRON', 'ADV', 'NUM'):
            continue
        if str(tok_item.get('surface', '')).strip() in (
                "'", "\u2019", "\u2018", '"', ',', '.', '-', '–', '—', '«', '»'):
            continue
        if tok_item.get('role') in _clitic_roles:
            continue
        if (tok_item.get('pos') == 'PRON'
                and not tok_item.get('bm')
                and tok_item.get('dep') in ('obl:arg', 'obj', 'obl', 'expl:comp', 'expl')):
            continue
        # Exclure tokens de négation (ne/n'/pas...) même si bm assigné par erreur
        _neg_surfaces_local = G_kg.get('neg_surfaces', set())
        # Strip toutes les variantes d'apostrophe + élisions FR (n', l', j'...)
        _surf_raw   = str(tok_item.get('surface', '')).lower().strip()
        _surf_clean = _surf_raw.rstrip("'").rstrip('’').rstrip('‘').rstrip('ʼ')
        if (_surf_clean in _neg_surfaces_local or tok_item.get('role') == 'negation'):
            processed_indices.add(tok_item['orig_index'])
            continue

        dep_case = next((x for x in T if x.get('dep') == 'case'
                         and x.get('head_index') == tok_item['orig_index']), None)
        if not (dep_case or tok_item.get('is_loc')
                or tok_item.get('dep') in ('obl', 'obl:mod', 'obl:arg', 'advmod', 'nmod')):
            continue

        _role       = dep_case.get('role', '') if dep_case else ''
        # bm_marker (Preposition nodes) a la priorité ; fallback sur bm (FunctionWord)
        _marker_val = (dep_case.get('bm_marker') or dep_case.get('bm') or '') if dep_case else ''

        # Préposition composée sans traduction (role='content', bm vide) :
        # construire un placeholder visible à partir de tous les tokens case du même head.
        # Ex: jusqu' (content, bm='') + au (locative) → [jusqu'au]
        if not _marker_val and dep_case and _role == 'content':
            _sibling_cases = sorted(
                [x for x in T
                 if x.get('dep') == 'case'
                 and x.get('head_index') == tok_item['orig_index']
                 and x['orig_index'] != dep_case['orig_index']],
                key=lambda x: x['orig_index']
            )
            if _sibling_cases:
                _all_case_toks = sorted([dep_case] + _sibling_cases,
                                        key=lambda x: x['orig_index'])
                _composite = ''.join(c.get('surface', '') for c in _all_case_toks)
                _marker_val = f'[{_composite}]'

        # Complément d'un adjectif statif épistémique/émotionnel :
        # "sûr de toi", "amoureux de toi" → le complément (obl:arg porté par
        # l'ADJ statif ROOT) prend la postposition locative 'la' → "i la".
        _is_statif_adj_complement = (
            root_tok is not None
            and root_tok.get('pos') == 'ADJ'
            and root_tok.get('is_statif')
            and tok_item.get('dep') == 'obl:arg'
            and tok_item.get('head_index') == root_tok.get('orig_index'))

        # Dispatcher selon rôle/marqueur
        if _is_statif_adj_complement:
            wagon.append(tok_item, T, m, processed_indices, G_kg, NX_G,
                         'locative', _marker_val or 'la', dep_case)

        elif (_marker_val in _loc_markers
                or tok_item.get('is_loc')
                or _role == 'locative'):
            wagon.append(tok_item, T, m, processed_indices, G_kg, NX_G,
                         'locative', _marker_val or 'la', dep_case)

        elif _role == 'comitative':
            comitative.handle(tok_item, T, m, processed_indices, G_kg, NX_G)

        elif _role == 'privative':
            privatif.handle(tok_item, T, m, processed_indices, G_kg, dep_case)

        elif (_marker_val in _tmp_markers
              or tok_item.get('role') == 'temporal'
              or (tok_item.get('dep') == 'obl:mod')
              or (tok_item.get('dep') == 'advmod'
                  and (tok_item.get('role') in ('temporal', 'temporal_already')
                       or tok_item.get('pos') == 'ADV'))):
            # advmod : role temporal explicite OU ADV (hier, demain, maintenant...)
            wagon.append(tok_item, T, m, processed_indices, G_kg, NX_G,
                         'temporal', _marker_val, dep_case)

        elif (dep_case and dep_case.get('dep') == 'case'
              and tok_item.get('dep') in ('obl', 'obl:arg', 'nmod')):
            # Oblique avec case explicite (preposition)
            wagon.append(tok_item, T, m, processed_indices, G_kg, NX_G,
                         'simple', _marker_val, dep_case)

        elif tok_item.get('dep') in ('obl', 'obl:arg'):
            # Oblique sans case mais dep obl explicite
            wagon.append(tok_item, T, m, processed_indices, G_kg, NX_G,
                         'simple', _marker_val, dep_case)

        # advmod sans role temporal ni case → attribut/qualificatif → ignorer ici