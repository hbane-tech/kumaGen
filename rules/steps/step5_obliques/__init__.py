"""rules/steps/step5_obliques/__init__.py — orchestrateur obliques."""
from rules.core import j
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

        # acl with temporal SCONJ mark (quand/lorsque on ROOT noun) → temporal clause
        if (tok_item.get('dep') == 'acl'
                and any(x.get('dep') == 'mark' and x.get('pos') == 'SCONJ'
                        and x.get('head_index') == tok_item['orig_index']
                        for x in T)):
            advcl.handle(tok_item, T, m, processed_indices, G_kg)
            continue

        # acl:relcl dans la boucle obliques
        if (tok_item.get('dep') == 'acl:relcl'
                and tok_item['orig_index'] not in processed_indices):
            relcl_boucle.handle(tok_item, T, m, processed_indices, G_kg, NX_G)
            continue

        # Filtres généraux
        # ADJ role='quantifier' (tous/chacun...) fonctionne comme un pronom
        # nominal dans un oblique casé ("pour TOUS" → complément bénéfactif),
        # pas comme un adjectif épithète — sinon silencieusement perdu ici
        # (décision 2026-07-13, bug : "... est meilleur pour tous" perdait
        # tout le complément bénéfactif).
        if (tok_item.get('pos') not in ('NOUN', 'PROPN', 'PRON', 'ADV', 'NUM')
                and not (tok_item.get('pos') == 'ADJ' and tok_item.get('role') == 'quantifier')):
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

        # Adjoint temporel NU attaché en dep='conj' sur le ROOT ("tout le
        # temps", "chaque jour" en fin de phrase, sans virgule) : spaCy
        # rattache parfois ces adjoints temporels sans préposition comme un
        # faux coordonné du ROOT plutôt qu'en obl:mod/advmod. Repéré via son
        # propre amod quantifieur ("tout"/"tous"→role=quantifier, ou lemme
        # 'tout' même si le KG ne connaît que la forme plurielle 'tous' comme
        # FunctionWord quantifier — le singulier 'tout' n'est PAS ajouté au
        # dict quantifiers global pour ne pas percuter "tout petit"/"tout à
        # fait" qui restent des intensifieurs ADJ, donc vérifié ici par lemme
        # sur ce motif précis uniquement) ou déterminant distributif
        # ("chaque") — motif volontairement étroit pour ne jamais confondre
        # avec une VRAIE coordination nominale ("Paul et Marie") (bug trouvé
        # 2026-07-25 : "il va causer avec le vieux TOUT LE TEMPS" perdait cet
        # adjoint entièrement, ni consommé par S4 ni par aucune branche
        # ci-dessous).
        _distrib_surfaces_bare = (set(G_kg.get('distributive_each', {}).keys())
                                   | set(G_kg.get('distributive_one', {}).keys()))
        _is_bare_temporal_conj = (
            tok_item.get('dep') == 'conj'
            and tok_item.get('pos') in ('NOUN', 'PROPN')
            and root_tok is not None
            and tok_item.get('head_index') == root_tok.get('orig_index')
            and any(x.get('head_index') == tok_item['orig_index']
                    and ((x.get('dep') == 'amod'
                          and (x.get('role') == 'quantifier'
                               or str(x.get('lemma', '')).lower() == 'tout'))
                         or (x.get('dep') == 'det'
                             and str(x.get('surface', '')).lower() in _distrib_surfaces_bare))
                    for x in T)
        )

        if not (dep_case or tok_item.get('is_loc') or _is_bare_temporal_conj
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
                         'locative', _marker_val or G_kg.get('locative_default_marker', 'la'), dep_case)

        elif (_marker_val in _loc_markers
                or tok_item.get('is_loc')
                or _role == 'locative'):
            # Toponyme propre (Bamako, Ségou...) : en bambara, le nom de lieu
            # propre est un adjoint locatif nu, sans postposition — contrairement
            # à un nom commun de lieu ("mosquée", "maison") qui prend 'la'.
            _loc_marker_final = ('' if tok_item.get('pos') == 'PROPN'
                                  else (_marker_val or G_kg.get('locative_default_marker', 'la')))
            wagon.append(tok_item, T, m, processed_indices, G_kg, NX_G,
                         'locative', _loc_marker_final, dep_case)

        elif _role == 'comitative':
            comitative.handle(tok_item, T, m, processed_indices, G_kg, NX_G)

        elif _role == 'privative' and tok_item.get('dep') != 'ROOT':
            # Privative oblique (ex: "travailler sans argent" → obl + case)
            # Exclure ROOT : "ce légume est sans cuisson" → copula + privative pred
            privatif.handle(tok_item, T, m, processed_indices, G_kg, dep_case)

        elif _role == 'agent' or tok_item.get('dep') == 'obl:agent':
            # Agent du passif ("fondée PAR Steve Jobs") : dep='obl:agent',
            # jusqu'ici absent de tous les filtres dep ci-dessous → silencieusement
            # perdu. Postposition agentive depuis le KG (Preposition role='agent').
            # Nom composé (prénom+nom, "Steve Jobs") : le second nom pend en
            # dep='flat:name' — get_bounded_chunk_tokens ne le restitue pas
            # correctement dans le rendu wagon, donc fusion manuelle du bm ici
            # (même schéma que le sujet, cf. step2_sujet.py:_build_subj_chain).
            _flat_name = next((x for x in T
                               if x.get('dep') in ('flat', 'flat:name')
                               and x.get('head_index') == tok_item['orig_index']), None)
            if _flat_name:
                tok_item['bm'] = j(tok_item.get('bm') or tok_item.get('surface', ''),
                                    _flat_name.get('bm') or _flat_name.get('surface', ''))
                processed_indices.add(_flat_name['orig_index'])
            wagon.append(tok_item, T, m, processed_indices, G_kg, NX_G,
                         'simple', _marker_val or G_kg.get('agent_postposition', 'fè'), dep_case)

        elif _is_bare_temporal_conj:
            # 'tout' (lemme) sans role='quantifier' pré-posé (KG ne connaît
            # que 'tous' pluriel) : forcer le role ici pour que wagon.append
            # le postpose nu ("temps bɛɛ") au lieu de le traiter comme un
            # épithète -man ordinaire (adj_man), qui produirait un accord
            # grammaticalement faux pour ce mot fonctionnel.
            for _amod_t in T:
                if (_amod_t.get('dep') == 'amod'
                        and _amod_t.get('head_index') == tok_item['orig_index']
                        and str(_amod_t.get('lemma', '')).lower() == 'tout'
                        and _amod_t.get('role') != 'quantifier'):
                    _amod_t['role'] = 'quantifier'
            wagon.append(tok_item, T, m, processed_indices, G_kg, NX_G,
                         'temporal', '', dep_case)

        elif (_marker_val in _tmp_markers
              or tok_item.get('role') == 'temporal'
              or (tok_item.get('dep') == 'obl:mod')
              or (tok_item.get('dep') == 'advmod'
                  and (tok_item.get('role') in ('temporal', 'temporal_already')
                       or tok_item.get('pos') == 'ADV'))):
            # advmod : role temporal explicite OU ADV (hier, demain, maintenant...)
            # "pour 20 minutes" : obl:mod + nummod + marker vide → marqueur durée 'yé'
            if (not _marker_val
                    and tok_item.get('dep') == 'obl:mod'
                    and any(x.get('dep') == 'nummod'
                            and x.get('head_index') == tok_item['orig_index']
                            for x in T)):
                _marker_val = G_kg.get('comitative_end_marker', 'yé')
            wagon.append(tok_item, T, m, processed_indices, G_kg, NX_G,
                         'temporal', _marker_val, dep_case)

        elif (dep_case and dep_case.get('role') in ('gerund', 'genitive')
              and tree.get('venir_source_locatif')
              and tok_item.get('dep') == 'obl:arg'
              and tok_item.get('pos') == 'NOUN'):
            # "venir de [lieu commun]" → locatif 'la' (ex: Je viens de l'école)
            wagon.append(tok_item, T, m, processed_indices, G_kg, NX_G,
                         'locative', G_kg.get('locative_default_marker', 'la'), dep_case)

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