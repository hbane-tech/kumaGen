"""
rules/kg_gateway.py
Moteur de règles KG pur — zéro règle linguistique hardcodée.

Architecture :
  PatternRule   → clause_type (variant inclus)
  SlotFillRule  → slots simples depuis les tokens
  GraphWalkRule → slots complexes via traversée de dépendances
  TransformRule → TAM + morphologie

Python = moteur d'exécution générique.
Toutes les décisions linguistiques sont dans le KG.

Flux :
  spaCy → apply_kg_semantic_behaviors (annotation tokens)
        → steps 0-7 (extraction slots S/O/V/TAM/OBL_ALL)
        → apply_kg_rules (PatternRule + SlotFill + GraphWalk + Transform)
        → renderer (fill_template)
"""

from rules.core import j, _resolve_tam
from rules.kg_rule_engine import apply_morpho_suffix, apply_statif_morpho


# ──────────────────────────────────────────────────────────────────────────────
# Extraction de features (pure — aucune décision linguistique)
# ──────────────────────────────────────────────────────────────────────────────

def _build_features(tree: dict, T: list, G_kg: dict = None) -> dict:
    """Extrait les features de la phrase pour les PatternRule KG."""
    G_kg = G_kg or {}
    root_tok = next((t for t in T if t.get('is_root')), None)
    neg      = tree.get('neg', False)
    _neg_surf = G_kg.get('neg_surfaces', set())
    _has_neg  = (neg or any(
        t.get('role') == 'negation'
        or str(t.get('surface', '')).lower().rstrip("'") in _neg_surf
        for t in T))
    _has_excl = any(str(t.get('surface', '')).strip() == '!'
                    or (t.get('dep') == 'punct' and str(t.get('surface', '')).strip() == '!')
                    for t in T)
    m = tree.get('main', {})
    return {
        'root_pos':              root_tok.get('pos', '')                if root_tok else '',
        'root_lemma':            (root_tok.get('lemma') or '').lower()  if root_tok else '',
        'root_tense':            tree.get('tense', 'pres'),
        'tense':                 tree.get('tense', 'pres'),   # alias pour cond_tense dans les PatternRules
        'root_semantic_class':   root_tok.get('semantic_class', '')     if root_tok else '',
        'root_role':             root_tok.get('role', '')               if root_tok else '',
        'root_is_statif':        'true' if (root_tok and root_tok.get('is_statif'))           else 'false',
        'root_is_participe_passe':'true' if (root_tok and root_tok.get('is_participe_passe')) else 'false',
        'root_is_nominal_adj':   'true' if (root_tok and root_tok.get('is_nominal_adj'))      else 'false',
        'is_passive':            'true' if tree.get('_is_passive')      else 'false',
        'has_cop':               'true' if (tree.get('_has_cop')
                                 or any(t.get('dep') in ('cop', 'aux:pass')
                                        and t.get('pos') in ('AUX', 'VERB') for t in T))     else 'false',
        # expl:subj = sujet réel quand être est ROOT ou cop
        # (spaCy false positive : "il est professeur", "il n'est pas à la maison")
        'has_nsubj':             'true' if (
                                    any(t.get('dep') in ('nsubj', 'nsubj:pass') for t in T)
                                    # expl:subj + role='subject' = pronom personnel réel (spaCy
                                    # mal-étiqueté pour passé composé avec être : "il est tombé")
                                    or any(t.get('dep') == 'expl:subj'
                                           and t.get('role') == 'subject' for t in T)
                                    or (any(t.get('dep') == 'expl:subj' for t in T)
                                        and any(t.get('dep') in ('cop', 'aux:pass')
                                                or (t.get('role') == 'copula' and t.get('is_root'))
                                                for t in T)))
                                 else 'false',
        'has_obj':               'true' if any(t.get('dep') == 'obj' for t in T)             else 'false',
        'has_expletive':         'true' if any(
                                    t.get('role') == 'expletive'
                                    or t.get('dep') in ('expl:subj', 'expl:comp', 'expl', 'expletive')
                                    or str(t.get('surface', '')).lower().rstrip("'") == 'y'
                                    for t in T)                                               else 'false',
        'has_question':          'true' if tree.get('_has_question_mark') else 'false',
        'has_adv_interrogative': 'true' if any(t.get('role') == 'interrogative'
                                 and t.get('dep') in ('advmod', 'dep') for t in T)           else 'false',
        'has_interrogative_word':'true' if any(t.get('role') == 'interrogative' for t in T)  else 'false',
        'has_loc_case':          'true' if (tree.get('_has_cop') and any(
                                    t.get('role') == 'locative' or t.get('is_loc') for t in T)) else 'false',
        'has_loc_obl':           'true' if any(
                                    (t.get('role') == 'locative' or t.get('is_loc'))
                                    and t.get('dep') in ('advmod', 'obl', 'obl:mod', 'obl:arg', 'case')
                                    for t in T)                                               else 'false',
        'has_acl':               'true' if any(t.get('dep') == 'acl' for t in T)             else 'false',
        'has_comitative_obl':    'true' if any(t.get('role') == 'comitative' for t in T)     else 'false',
        'has_xcomp_verb':        'true' if any(t.get('dep') == 'xcomp' and t.get('pos') == 'VERB' for t in T) else 'false',
        'has_purposive_case':    'true' if any(t.get('dep') == 'case' and t.get('role') == 'purposive' for t in T) else 'false',
        'has_mark_locative':     'true' if any(t.get('dep') == 'mark' and t.get('role') == 'locative' for t in T) else 'false',
        'root_pos_noun_or_pron': 'true' if (root_tok and root_tok.get('pos') in ('NOUN', 'PRON')) else 'false',
        'has_acl_only':          'true' if (any(t.get('dep') == 'acl' for t in T)
                                 and not any(t.get('dep') == 'acl:relcl' for t in T))        else 'false',
        'has_negation':          'true' if _has_neg                                           else 'false',
        'has_exclamation':       'true' if _has_excl                                          else 'false',
        # Interrogatif / content_question
        'has_est_ce_que':        'true' if tree.get('est_ce_que')                            else 'false',
        'has_motion_root':       'true' if (root_tok and root_tok.get('semantic_class') == 'motion') else 'false',
        'has_modal_xcomp':       'true' if (m.get('V_ACTION')
                                 or any(t.get('dep') == 'xcomp' and t.get('pos') == 'VERB' for t in T)) else 'false',
        'has_loc_case_interrog': 'true' if any(t.get('role') == 'locative' and t.get('dep') == 'case' for t in T) else 'false',
        'has_action_no_obj':     'true' if (root_tok
                                 and root_tok.get('intransitive_type') in ('ACTION', 'nominalized', 'support')
                                 and not any(t.get('dep') == 'obj' for t in T))              else 'false',
        # Possession / copule
        'possession_type':       tree.get('possession_type', ''),
        'clause_type':           tree.get('clause_type') or 'simple',  # défaut 'simple' pour PatternRules
        # Classe sémantique du ROOT (pour PatternRules comme reciprocal_comm_r)
        'root_semantic_class':   root_tok.get('semantic_class', '') if root_tok else '',
        # Verbe dep='dep' de classe modal (pouvoir dans quest_ce_que structure)
        'has_dep_modal':         'true' if any(
                                    t.get('dep') == 'dep' and t.get('pos') == 'VERB'
                                    and t.get('semantic_class') == 'modal'
                                    for t in T) else 'false',
        # Verbe dep='dep' de classe saying (ex: "disons" dans cleft-naming)
        'has_dep_saying_verb':   'true' if any(
                                    t.get('dep') == 'dep' and t.get('pos') == 'VERB'
                                    and t.get('semantic_class') == 'saying'
                                    for t in T) else 'false',
        # Expletif démonstratif (c'/ça, pas il/elle) → distingue "c'est X" de "il n'est pas X"
        'has_demonstrative_expletive': 'true' if any(
                                    t.get('dep') in ('expl:subj', 'expl:comp', 'expl', 'expletive')
                                    and t.get('role') in ('demonstrative', 'expletive')
                                    for t in T) else 'false',
        # cop directement sur ROOT (pas dans clause subordonnée)
        'has_root_cop':          'true' if (
                                    root_tok and any(
                                        t.get('dep') == 'cop'
                                        and t.get('head_index') == root_tok.get('orig_index')
                                        for t in T)) else 'false',
        # Cas comitatif (avec) → PatternRule etre_comitative_r
        'has_comitative_case':   'true' if any(
                                    t.get('dep') == 'case' and t.get('role') == 'comitative'
                                    for t in T) else 'false',
        # nsubj démonstratif (Ce/ce/ça) → distingue "Ce n'est pas moi" de "il n'est rien"
        'has_demonstrative_nsubj': 'true' if any(
                                    t.get('dep') in ('nsubj', 'nsubj:pass')
                                    and t.get('role') in ('demonstrative', 'expletive')
                                    for t in T) else 'false',
        # Est-ce que : token 'ce' dep='dep' + token 'que'/'qu'' dep='mark'
        # (détecté AVANT apply_kg_patterns_early, avant step3)
        'has_est_ce_que_struct': 'true' if (
                                    any(str(t.get('surface', '')).lower().strip('-') == 'ce'
                                        and t.get('dep') in ('dep', 'nsubj', 'expl:subj') for t in T)
                                    and any(str(t.get('surface', '')).lower().rstrip("'") in ('que', 'qu')
                                            and t.get('dep') == 'mark' for t in T))  else 'false',
        # Interrogatif âge : DET interrogatif + nom bm='saan' (an/âge/année)
        'has_age_interrog':      'true' if (
                                    any(t.get('role') == 'interrogative'
                                        and t.get('dep') == 'det' for t in T)
                                    and any(t.get('bm') == 'saan'
                                            or (t.get('lemma') or '').lower() in ('an', 'âge', 'année')
                                            for t in T))              else 'false',
        # Conditional
        'conditional_marker_mana': 'true' if tree.get('conditional_marker') == 'mána' else 'false',
        # Verb serial sub-cases (read V_ACTION from m = tree['main'])
        'has_venir_de_verb':     'true' if (
                                    root_tok and (root_tok.get('lemma') or '').lower() == 'venir'
                                    and any(t.get('dep') == 'mark'
                                            and str(t.get('surface', '')).lower() == 'de' for t in T)
                                    and tree.get('main', {}).get('V_ACTION'))           else 'false',
        'has_aller_pres':        'true' if (
                                    root_tok and (root_tok.get('lemma') or '').lower() == 'aller'
                                    and tree.get('tense', 'pres') != 'past'
                                    and tree.get('main', {}).get('V_ACTION'))           else 'false',
        # Simple sub-cases
        'is_support_nom':        'true' if (
                                    root_tok
                                    and root_tok.get('semantic_class') in ('having', 'technique')
                                    and not any(t.get('dep') == 'obj' for t in T))     else 'false',
        'has_venir_de_source':   'true' if (
                                    root_tok and (root_tok.get('lemma') or '').lower() == 'venir'
                                    and any(t.get('dep') == 'case'
                                            and (str(t.get('surface', '')).lower() in ('de', 'du', 'des')
                                                 or str(t.get('surface', '')).lower().rstrip("''") in ('d', 'de'))
                                            and t.get('role') in ('gerund', 'genitive') for t in T)
                                    and not tree.get('main', {}).get('V_ACTION'))       else 'false',
        # Imperative person (surface de la racine verbale)
        'imperative_person':     ('1pl' if root_tok and str(root_tok.get('surface', '')).lower().endswith('ons')
                                  else '2pl' if root_tok and str(root_tok.get('surface', '')).lower().endswith('ez')
                                  else '2sg'),
        # Relative topic / comitative
        'has_verb':              'true' if (tree.get('main', {}).get('V') or
                                            tree.get('main', {}).get('V_ACTION')) else 'false',
        'has_propn':             'true' if any(t.get('pos') == 'PROPN' and t.get('bm')
                                               for t in T) else 'false',
        # Features step3 (depuis tokens bruts)
        'has_restrictive_mark':  'true' if any(t.get('role') == 'restrictive' for t in T)    else 'false',
        'root_is_infinitive':    'true' if (root_tok and 'Inf' in str(root_tok.get('morph', ''))) else 'false',
        'has_participial_to':    'true' if any(t.get('role') == 'participial_to' for t in T) else 'false',
        'has_conditional_mark':  'true' if any(t.get('dep') == 'mark' and t.get('role') == 'conditional' for t in T) else 'false',
        'has_temporal_mark':     'true' if any(t.get('dep') == 'mark' and t.get('role') == 'temporal' for t in T) else 'false',
        # Exclure les verbes de préparation/arrangement de la détection réciproque
        '_root_sc_for_recip':    root_tok.get('semantic_class', '') if root_tok else '',
        'has_reciprocal':        'true' if (
                                    # Ne pas déclencher ɲɔgɔn pour les verbes qui ne sont pas mutuels par nature
                                    (root_tok.get('semantic_class', '') not in ('preparation', 'grooming', 'cognitive'))
                                    # expl:comp/expl:pass (se/nous) ou obj PRON pluriel réflexif (nous nous).
                                    # expl:pass inclus : spaCy tague "se parler" en expl:pass même au
                                    # pluriel ("ils se parlent"), contrairement à "nous nous parlons"
                                    # (expl:comp) — incohérence de tagging UD, pas une distinction
                                    # linguistique réelle (même construction réfléchie communicative).
                                    and (any(t.get('dep') in ('expl:comp', 'expl:pass')
                                         and t.get('role') in ('reflexive', 'clitic') for t in T)
                                     or any(t.get('dep') == 'obj' and t.get('pos') == 'PRON'
                                            and (t.get('is_plural') or 'Number=Plur' in str(t.get('morph', '')))
                                            and any(s.get('dep') in ('nsubj', 'nsubj:pass')
                                                    and s.get('lemma') == t.get('lemma')
                                                    for s in T)  # lemme identique = reflexif (pas relatif)
                                            for t in T)
                                     # "nous nous": expl:comp/expl:pass même lemme que nsubj pluriel
                                     or any(t.get('dep') in ('expl:comp', 'expl:pass') and t.get('pos') == 'PRON'
                                            and any(s.get('dep') in ('nsubj', 'nsubj:pass')
                                                    and s.get('lemma') == t.get('lemma')
                                                    for s in T)
                                            for t in T))
                                    and any(t.get('is_plural') or ('Number=Plur' in str(t.get('morph', '')))
                                            for t in T if t.get('dep') in ('nsubj', 'nsubj:pass'))) else 'false',
        'has_privative_pred':    'true' if any(
                                    t.get('dep') == 'case' and t.get('role') == 'privative'
                                    and any(x.get('is_root') and x.get('pos') in ('NOUN', 'PRON')
                                            and x.get('orig_index') == t.get('head_index') for x in T)
                                    for t in T) else 'false',
        'has_relative_topic':    'true' if tree.get('_is_relative_topic') else 'false',
        'has_refl_absolute':     'true' if tree.get('_is_refl_absolute')  else 'false',
        # V_ACT (xcomp) est un verbe d'action sans objet → nominalisation
        'is_vact_action_no_obj': 'true' if (
                                    tree.get('main', {}).get('V_ACTION')
                                    and not any(t.get('dep') == 'obj' for t in T)
                                    and any(t.get('dep') == 'xcomp' and t.get('pos') == 'VERB'
                                            and t.get('intransitive_type') in ('ACTION', 'nominalized', 'support')
                                            and t.get('semantic_class', '') not in (G_kg.get('intrans_sc') or __import__('rules.core', fromlist=['INTRANS_SC']).INTRANS_SC)
                                            for t in T)) else 'false',
    }


# ──────────────────────────────────────────────────────────────────────────────
# Moteurs de règles génériques (lisent le KG, aucune logique hardcodée)
# ──────────────────────────────────────────────────────────────────────────────

def _match_pattern_rules(rules: list, features: dict, current_ct: str) -> dict:
    """Cherche la PatternRule de plus haute priorité matchant toutes les cond_*."""
    _STRONG_CT = {
        'prohibitive', 'imperative', 'content_question', 'interrogative',
        'verb_serial', 'noun_phrase_have', 'existential_absolute',
        'existential_localized', 'existential_nominal', 'refl_absolute', 'reciprocal',
        # Clause types spécialisés qui ne doivent pas être écrasés par simple_default (prio=10)
        'simple_venir_de_pres', 'simple_venir_de_past', 'content_question_venir_de',
        'optative', 'possession_abstract_pres', 'possession_abstract_past',
        'possession_abstract_fut', 'possession_material_pres', 'possession_material_past',
        'possession_material_fut', 'possession_age_pres', 'possession_age_interrog',
        'possession_experiencer_pres', 'possession_experiencer_past', 'possession_pain_pres',
        'possession_statif_pres_pos', 'possession_statif_past_pos',
        'locative', 'qualitative', 'equative', 'identificatory', 'presentative',
        'locative_interrog', 'qualitative_interrog', 'equative_interrog',
        'possession_abstract_pres_interrog', 'possession_material_pres_interrog',
        'focus_cleft_naming', 'venir_de_verb', 'copula_comitative', 'interrogative_action',
        'quest_ce_que_modal', 'quest_ce_que', 'modal_interrog_cod', 'motion_content_question',
        'motion_interrog',
    }
    best, best_prio = {}, -1
    for rule in rules:
        conds = {k[5:]: v for k, v in rule.items() if k.startswith('cond_')}
        if not conds:
            continue
        if all(str(features.get(k, '')) == str(v) for k, v in conds.items()):
            prio = rule.get('priority', 0) or 0
            if prio > best_prio:
                best_prio = prio
                best = rule
    if not best:
        return {}
    if current_ct in _STRONG_CT and (best.get('priority', 0) or 0) < 80:
        return {}
    return best


def _apply_transform_rules(tree: dict, m: dict, G_kg: dict) -> None:
    """Applique les TransformRule KG : TAM + morphologie. Aucune valeur hardcodée."""
    ct    = tree.get('clause_type', '')
    neg   = 'true' if tree.get('neg', False) else 'false'
    tense = tree.get('tense', 'pres')

    rules = sorted(G_kg.get('kg_transform_rules', []),
                   key=lambda r: r.get('priority', 0) or 0, reverse=True)

    applied_targets = set()
    for rule in rules:
        cond_ct    = rule.get('cond_clause_type', '')
        cond_neg   = rule.get('cond_neg', '')
        cond_tense = rule.get('cond_tense', '')
        cond_it    = rule.get('cond_intransitive', '')
        cond_trans = rule.get('cond_transitive', '')
        target     = rule.get('target_slot', '')
        transform  = rule.get('transform', '')

        if not target or not transform:
            continue
        if cond_ct    and cond_ct    != ct:    continue
        if cond_neg   and cond_neg   != neg:   continue
        if cond_tense and cond_tense != tense: continue
        if cond_it == 'true' and tree.get('is_transitive', True):  continue
        if cond_it == 'false' and not tree.get('is_transitive', True): continue
        if cond_trans == 'true' and not tree.get('is_transitive', True): continue
        # cond_is_action_no_obj (verbe principal) et cond_is_vact_action_no_obj (xcomp)
        _cond_iano = rule.get('cond_is_action_no_obj', '')
        if _cond_iano == 'true' and not tree.get('_is_action_no_obj', False): continue
        if _cond_iano == 'false' and tree.get('_is_action_no_obj', False):    continue
        _cond_vact = rule.get('cond_is_vact_action_no_obj', '')
        if _cond_vact == 'true' and not tree.get('_is_vact_action_no_obj', False): continue
        if _cond_vact == 'false' and tree.get('_is_vact_action_no_obj', False):    continue

        if target in applied_targets:
            continue  # highest-priority rule per slot already applied

        # Impératif/prohibitif : TAM='' posé par step3 (_is_imperative_affirm) —
        # aucune TransformRule ne doit l'écraser (ni générique ni spécifique imperative).
        if ct in ('imperative', 'prohibitive') and target == 'TAM':
            if transform in ('set_from_kg_tam',) or transform.startswith('set_value:'):
                applied_targets.add(target)
                continue

        # (Guard prohibitive nominalization via _is_action_no_obj computation above)

        # Progressif : TAM 'bɛ kà' posé par step3 et verbe nu — pas de réécriture
        # par les TransformRules (qui nominaliseraient ou changeraient le TAM/V/O).
        if tense == 'prog' and not cond_tense:
            if target in ('TAM', 'V', 'V_ACTION', 'O', 'QUAL'):
                if (transform.startswith('set_value:')
                        or transform == 'set_from_kg_tam'
                        or 'nominali' in transform
                        or 'statif' in transform):
                    applied_targets.add(target)
                    continue

        # Exécution du transform
        if transform.startswith('set_value:'):
            val = transform[len('set_value:'):]
            if target == 'TAM':
                tree['tam'] = val
            else:
                m[target] = val
            applied_targets.add(target)

        elif transform.startswith('set_value_if_equal:'):
            # format : set_value_if_equal:OLD→NEW
            _cond_str = transform[len('set_value_if_equal:'):]
            if '→' in _cond_str:
                _old, _new = _cond_str.split('→', 1)
                _cur = tree['tam'] if target == 'TAM' else m.get(target, '')
                if _cur == _old:
                    if target == 'TAM':
                        tree['tam'] = _new
                    else:
                        m[target] = _new
                    applied_targets.add(target)

        elif transform == 'set_from_kg_tam':
            _tam = _resolve_tam(tense, neg == 'true', G_kg)
            if _tam and target == 'TAM':
                tree['tam'] = _tam
                applied_targets.add(target)

        elif transform == 'set_from_kg_equative':
            if target == 'TAM':
                tree['tam'] = G_kg.get('equative_marker', 'yé') or 'yé'
                applied_targets.add(target)

        elif transform in ('set_tam_past_ka', 'set_tam_past_ma',
                           'set_tam_past_ye', 'set_tam_past_te'):
            _base = (_resolve_tam(tense, False, G_kg) or 'tùn bɛ').split()[0]
            if transform == 'set_tam_past_ka':
                tree['tam'] = j(_base, 'ka')
            elif transform == 'set_tam_past_ma':
                tree['tam'] = j(_base, 'ma')
            elif transform in ('set_tam_past_ye', 'set_tam_past_te'):
                m['TAM_BASE'] = _base
                tree['tam'] = ''
            applied_targets.add(target)

        elif transform == 'add_resultative_suffix':
            _morpho = G_kg.get('morpho_rules', {}).get('resultative', {})
            v = m.get(target, '')
            if v and _morpho:
                m[target] = apply_morpho_suffix(v, _morpho)
            tree['tam'] = ''
            applied_targets.add(target)

        elif transform == 'add_statif_suffix' or transform == 'add_statif_suffix_pos':
            _morpho_stat = G_kg.get('morpho_rules', {}).get('statif', {})
            qual = m.get('QUAL', '') or m.get('O', '')
            # QUAL vient toujours de step6 sans suffixe déjà appliqué : pas de garde
            # endswith ici (un adjectif comme 'bìlen' finit coïncidentellement par
            # 'len' sans porter le suffixe statif → un garde naïf bloquerait 'bìlenlen').
            if qual and _morpho_stat and not m.get('_qual_suffixed'):
                _sfx = _morpho_stat.get('suffix', 'len')
                qual += _sfx
                m['QUAL'] = qual
                m['_qual_suffixed'] = True
                if not m.get('QUAL_WAS_O') and m.get('O') == m.get('QUAL', ''):
                    m['O'] = ''
            applied_targets.add(target)

        elif transform == 'add_statif_suffix_neg':
            _morpho_stat = G_kg.get('morpho_rules', {}).get('statif', {})
            qual = m.get('QUAL', '') or m.get('O', '')
            if qual and _morpho_stat and not m.get('_qual_suffixed'):
                _sfx = _morpho_stat.get('suffix', 'len')
                qual += _sfx
                m['QUAL'] = qual
                m['_qual_suffixed'] = True
            applied_targets.add(target)

        elif transform == 'add_statif_suffix_hab':
            applied_targets.add(target)

        elif transform == 'add_nominalization_vact':
            vact = m.get(target, '')
            _ns  = G_kg.get('nominalization_verb_suffix', '')
            _as  = G_kg.get('coord_action_suffix', '')
            # Action class xcomp → NOM-ACTION (báara kɛ) pas nominalization complète (báarali kɛ)
            _xcomp_sc = tree.get('_xcomp_semantic_class', '')
            if _xcomp_sc == 'action':
                # Classe action : V + kɛ (pas li kɛ)
                if vact and _as and not vact.endswith(_as):
                    m[target] = j(vact, _as)
            else:
                _sfx = (_ns + (' ' + _as if _as else '')) if _ns else ''
                if vact and _sfx and not vact.endswith((_ns, _as, _sfx.strip())):
                    m[target] = vact + _sfx
            applied_targets.add(target)

        elif transform == 'add_nominalization':
            v   = m.get(target, '')
            _ns = G_kg.get('nominalization_verb_suffix', '')
            _as = G_kg.get('coord_action_suffix', '')
            _sfx = (_ns + (' ' + _as if _as else '')) if _ns else ''
            if v and _sfx and not v.endswith((_ns, _as, _sfx.strip())):
                m[target] = v + _sfx
            applied_targets.add(target)

        elif transform == 'add_passive_len':
            # Guard : aux:pass au passé = action passive (V+ra) → PAS statif
            if not tree.get('_has_aux_pass_past'):
                v         = m.get(target, '')
                _morpho   = G_kg.get('morpho_rules', {})
                _res      = _morpho.get('resultative', {})
                _stat     = _morpho.get('statif', {})
                _sfx_ra   = _res.get('suffix_default', '')
                _sfx_na   = _res.get('suffix_after_n', '')
                _sfx_la   = _res.get('suffix_after_vowel', '')
                _sfx_len  = _stat.get('suffix', '')
                _exclude  = tuple(s for s in (_sfx_ra, _sfx_na, _sfx_la, _sfx_len) if s)
                if v and _sfx_len and not v.endswith(_exclude):
                    m[target] = v + _sfx_len
                tree['_passive_statif'] = True
            applied_targets.add(target)

        elif transform == 'move_o_to_qual':
            # Qualitative : si QUAL vide, déplace O → QUAL
            if not m.get('QUAL') and m.get('O'):
                m['QUAL'] = m.pop('O')
                m['O'] = ''
            applied_targets.add(target)


def _apply_slot_fill_rules(tree: dict, m: dict, T: list, G_kg: dict) -> None:
    """Remplit les slots spéciaux en lisant SlotFillRule KG — aucun hardcode."""
    ct = tree.get('clause_type', '')
    _processed = tree.get('_processed_indices') or set()
    rules = sorted(G_kg.get('kg_slot_fill_rules', []),
                   key=lambda r: r.get('priority', 0) or 0, reverse=True)
    for rule in rules:
        target = rule.get('target_slot', '')
        dep    = rule.get('match_dep', '')
        role   = rule.get('match_role', '')
        pos    = rule.get('match_pos', '')
        vsrc   = rule.get('value_source', 'bm')
        if not target or m.get(target):
            continue
        matched = next((t for t in T
                        if (not dep  or t.get('dep') == dep)
                        and (not role or t.get('role') == role)
                        and (not pos  or t.get('pos') == pos)
                        and t.get(vsrc)
                        and t.get('orig_index') not in _processed
                        # Exclure les clitiques datifs/accusatifs mislabélés dep='nsubj'
                        # du slot S (role='object' = COI/COD, jamais sujet grammatical).
                        and not (target == 'S' and t.get('role') == 'object')), None)
        if matched:
            val = matched.get(vsrc, '')
            if val:  # ne pas écraser avec une valeur vide
                m[target] = val


def _apply_graph_walk_rules(tree: dict, m: dict, T: list, G_kg: dict) -> None:
    """Remplit les slots complexes en lisant GraphWalkRule KG (traversée deps)."""
    ct = tree.get('clause_type', '')
    rules = G_kg.get('kg_graph_walk_rules', [])
    nom_rules = G_kg.get('kg_nominal_rules', [])
    gen_marker = G_kg.get('genitive_marker', '') or 'ka'

    for rule in rules:
        rule_ct = rule.get('for_clause_type', '')
        if rule_ct and rule_ct != ct:
            continue
        target = rule.get('target_slot', '')
        dep    = rule.get('match_dep', '')
        pos    = rule.get('match_pos', '')
        vsrc   = rule.get('value_source', 'bm')
        if not target or m.get(target):
            continue

        start = next((t for t in T
                      if (not dep or t.get('dep') == dep)
                      and (not pos or t.get('pos') == pos)), None)
        if not start:
            continue

        if vsrc == 'genitive_chain':
            # Traversée : token + possessif DET → {poss ka head} or {n head}
            # Guidée par NominalChainRule KG (np_poss_n / np_poss_other)
            bm = start.get('bm', '')
            _poss = next((t for t in T if t.get('dep') == 'det'
                          and t.get('role') in ('pronoun', 'possessive')
                          and t.get('head_index') == start.get('orig_index')), None)
            if _poss:
                pb = _poss.get('bm', '')
                _np_n     = next((r for r in nom_rules if r.get('name') == 'np_poss_n'), {})
                _np_other = next((r for r in nom_rules if r.get('name') == 'np_poss_other'), {})
                if pb == 'n':
                    tpl = _np_n.get('template', '{poss} {head}')
                    bm = tpl.replace('{poss}', pb).replace('{head}', bm)
                else:
                    tpl = _np_other.get('template', '{poss} ka {head}')
                    bm = tpl.replace('{poss}', pb).replace('{head}', bm)
            m[target] = bm
        else:
            m[target] = start.get(vsrc, '')


# ──────────────────────────────────────────────────────────────────────────────
# FunctionWord : remplit les slots à partir de mots fonctionnels KG
# ──────────────────────────────────────────────────────────────────────────────

def _apply_function_words(tree: dict, m: dict, G_kg: dict) -> None:
    """Remplit les slots fixes depuis FunctionWord KG (Yala, félé, etc.)."""
    ct = tree.get('clause_type', '')
    for fw in G_kg.get('function_words', []) or []:
        target = fw.get('target_slot', '')
        applies = fw.get('applies_to_ct', '')
        if applies and applies != ct:
            continue
        if target and not m.get(target):
            m[target] = fw.get('bm', '')


# ──────────────────────────────────────────────────────────────────────────────
# API publique
# ──────────────────────────────────────────────────────────────────────────────

def apply_kg_patterns_early(tree: dict, T: list, G_kg: dict) -> None:
    """Appel PRÉCOCE : PatternRule → clause_type avant step1."""
    rules    = G_kg.get('kg_pattern_rules', [])
    if not rules:
        return
    features = _build_features(tree, T, G_kg)
    current  = tree.get('clause_type', 'simple')
    best     = _match_pattern_rules(rules, features, current)
    if best:
        tree['clause_type'] = best.get('clause_type', current)


def apply_kg_rules(tree: dict, T: list, G_kg: dict) -> dict:
    """
    Moteur KG post-step : lit les règles KG dans l'ordre suivant —
      1. PatternRule  → clause_type variant
      2. TransformRule → TAM + morphologie
      3. SlotFillRule → slots simples depuis tokens
      4. GraphWalkRule → slots complexes via traversée
      5. FunctionWord → marqueurs fixes (Yala, félé…)
      6. Règles legacy (météo, réflexif…) via ancien système FeaturePattern
    """
    m = tree.get('main', {})

    # ── 0. Pré-annoter tree avec signaux strucutrels utiles aux TransformRules ─
    # Note: _is_vact_action_no_obj est recalculé ici après step3 (V_ACTION posé)
    _root_tok_kg = next((t for t in T if t.get('is_root')), None)
    if _root_tok_kg:
        _sc_kg = _root_tok_kg.get('semantic_class', '')
        _it_kg = _root_tok_kg.get('intransitive_type', '')
        from rules.core import INTRANS_SC
        tree['_root_semantic_class'] = _sc_kg
        # is_action_no_obj : ACTION/nominalized/support + pas dans INTRANS_SC + pas d'objet
        # iobj compte aussi : verbes communication_transitive ont leur objet via iobj
        # m.get('O') : objet déjà posé en step2 (ex: 'jɔn' pour "Qui aiment-ils?")
        _has_obj_or_iobj_kg = any(t.get('dep') in ('obj', 'iobj') for t in T)
        tree['_is_action_no_obj'] = (
            _it_kg in ('ACTION', 'nominalized', 'support')
            and _sc_kg not in INTRANS_SC
            and _sc_kg not in ('action', 'communication_transitive')  # ces classes gèrent leur propre objet
            and not _has_obj_or_iobj_kg
            and not m.get('O')
            # psych_emotion en prohibitif reste verbe nu (kàna kàsi, pas kàna kàsili kɛ).
            # Hors prohibitif, psych_emotion sans COD peut être nominalisé (elle aime → kànuli kɛ).
            and not (tree.get('clause_type') == 'prohibitive' and _sc_kg == 'psych_emotion')
        )

    # ── 0b. _is_vact_action_no_obj : xcomp ACTION sans objet ni INTRANS_SC ──────
    # Stocker la semantic_class du xcomp pour add_nominalization_vact
    _xcomp_tok_0b = next((t for t in T if t.get('dep') == 'xcomp' and t.get('pos') == 'VERB'), None)
    tree['_xcomp_semantic_class'] = _xcomp_tok_0b.get('semantic_class', '') if _xcomp_tok_0b else ''
    _intrans_sc_kg = G_kg.get('intrans_sc') or __import__('rules.core', fromlist=['INTRANS_SC']).INTRANS_SC
    _xcomp_action = next((t for t in T
                          if t.get('dep') == 'xcomp' and t.get('pos') == 'VERB'
                          and t.get('intransitive_type') in ('ACTION', 'nominalized', 'support')
                          and t.get('semantic_class', '') not in _intrans_sc_kg), None)
    tree['_is_vact_action_no_obj'] = bool(
        m.get('V_ACTION')
        and _xcomp_action
        and not any(t.get('dep') == 'obj' for t in T)
    )

    # ── 1. PatternRule → clause_type ─────────────────────────────────────────
    pattern_rules = G_kg.get('kg_pattern_rules', [])
    _pre_pattern_ct = tree.get('clause_type', '')  # clause_type avant PatternRule
    if pattern_rules:
        features   = _build_features(tree, T, G_kg)
        current_ct = tree.get('clause_type', '')
        best       = _match_pattern_rules(pattern_rules, features, current_ct)
        if best:
            _new_ct = best.get('clause_type', current_ct)
            # Préserver les types posés explicitement par les steps Python :
            # - impératif/prohibitif (posé par step3 _is_imperative_affirm)
            # - quest_ce_que_modal impersonnel (posé par step3 construction expl:subj+modal)
            _protected_cts = {'imperative', 'prohibitive', 'quest_ce_que_modal'}
            if current_ct in _protected_cts and _new_ct not in _protected_cts:
                pass  # garder le type explicitement posé
            else:
                tree['clause_type'] = _new_ct

    # ── 2. TransformRule → TAM + morphologie ─────────────────────────────────
    # Pré-calculer le flag passif-action (aux:pass au passé = action, pas statif)
    # pour que add_passive_len puisse le lire sans avoir accès à T.
    tree['_has_aux_pass_past'] = any(
        t.get('dep') == 'aux:pass' and t.get('tense') in ('past', 'plup')
        for t in T)
    # Si PatternRule a sélectionné passive_statif mais c'est une action passive
    # (aux:pass au passé) → revenir à la clause antérieure pour que F6
    # puisse construire le résultatif V+ra (pas V+len dòn).
    if (tree.get('_has_aux_pass_past')
            and tree.get('clause_type', '') in ('passive_statif', 'passive_statif_question')):
        tree['clause_type'] = _pre_pattern_ct or 'simple'
        tree['is_transitive'] = False  # F6 : passé intransitif → résultatif
    _apply_transform_rules(tree, m, G_kg)

    # ── 2-hab. Imparfait (tense='hab') pour clauses possession/copule/nom ────
    # Aucune PatternRule KG n'existe pour hab → la règle _pres s'applique par
    # défaut et pose TAM='bɛ'. On préfixe 'tùn' pour former 'tùn bɛ'/'tùn tɛ'.
    # Seulement pour les types possessifs (pas simple/refl/reciprocal qui gèrent
    # leur propre TAM via F6 / _resolve_tam).
    _hab_poss_cts = {
        'noun_phrase_have', 'possession_abstract_pres', 'possession_material_pres',
        'possession_abstract_pres_interrog', 'possession_material_pres_interrog',
        'possession_experiencer_pres', 'possession_statif_pres_pos',
        'possession_age_pres', 'possession_pain_pres',
    }
    if (tree.get('tense') == 'hab'
            and tree.get('clause_type', '') in _hab_poss_cts):
        _cur_tam = tree.get('tam') or 'bɛ'
        _tun = 'tùn'
        if not _cur_tam.startswith(_tun):
            tree['tam'] = j(_tun, _cur_tam)

    # ── 2-own. Possession pronominale : COMPANION = focus + possessif ('de ta') ──
    # Le TransformRule identificatory pose COMPANION='de' (focus seul), écrasant
    # le 'de ta' posé par ownership.py. On repose 'ta' si manquant.
    if tree.get('ownership_o_is_pron') and m.get('COMPANION'):
        _ta_own = G_kg.get('ownership_marker') or 'ta'
        if _ta_own and not m['COMPANION'].endswith(_ta_own):
            m['COMPANION'] = j(m['COMPANION'], _ta_own)

    # ── 2-np. Syntagme nominal sans verbe : supprimer TAM parasite ──────────────
    # Clause 'simple' avec ROOT=NOUN et pas de verbe réel (V='', V_ACTION='', O='') :
    # les TransformRules ou _resolve_tam posent 'bɛ' par défaut → s'intercale entre
    # S ('ɲíninikɛla jɛ̀len') et l'oblique ('cɛ́sira … la'), résultat : '… bɛ …'.
    # Un syntagme nominal ne porte pas de TAM → forcer ''.
    if (tree.get('clause_type', '') == 'simple'
            and not m.get('V') and not m.get('V_ACTION') and not m.get('O')
            and m.get('S') and m.get('OBL_ALL')):
        tree['tam'] = ''

    # ── 2b. "Dire" présent + ccomp → S ko CCOMP (V='', TAM='') ─────────────────
    # APRÈS les TransformRules (qui pourraient mettre TAM='yé') pour forcer TAM=''
    # ── 2a. "s'appeler" (communication_transitive + réflexif) → impersonnel "a" ──
    # "Je m'appelle" → a bɛ yɛrɛ kánbìla / "Je m'appelle Hawa" → a bɛ kánbìla Hawa
    if (_root_tok_kg and _root_tok_kg.get('semantic_class') == 'communication_transitive'
            and any(t.get('dep') == 'expl:comp'    # réflexif vrai seulement (pas iobj d'une autre personne)
                    and t.get('pos') == 'PRON'
                    and str(t.get('surface', '')).lower().lstrip('-') in
                        ('me', "m'", 'm', 'te', "t'", 't', 'se', "s'", 's')
                    for t in T)):
        _refl_self = G_kg.get('reflexive_self_marker', '')
        _impersonal = G_kg.get('impersonal_subj', 'a') or 'a'
        _has_propn = next((t for t in T if t.get('pos') == 'PROPN' and t.get('bm')), None)
        _root_bm_2a = _root_tok_kg.get('bm', '') if _root_tok_kg else ''
        if _has_propn:
            # "Je m'appelle Hawa" → a bɛ kánbìla Hawa (V + complement nominal)
            # Le verbe precede le complement (pas SOV standard) → V inclut le complement
            m['S'] = _impersonal
            m['O'] = ''   # pas d'objet séparé
            m['V'] = j(_root_bm_2a, _has_propn.get('bm', ''))  # kánbìla Hawa
            tree['clause_type'] = 'simple'
        elif _refl_self:
            # Sujet pluriel ("ils s'appellent") : transitif + pluriel = réciproque
            # (ils se nomment l'un l'autre), pas réflexif singulier → ɲɔgɔn au
            # lieu de yɛrɛ. Le sujet impersonnel 'a' reste (construction figée
            # de dénomination, indépendante de la personne/nombre du français),
            # seul le marqueur O change.
            _subj_2a = next((t for t in T if t.get('dep') in ('nsubj', 'nsubj:pass')), None)
            _subj_is_plural_2a = bool(
                _subj_2a and (_subj_2a.get('is_plural')
                              or 'Number=Plur' in str(_subj_2a.get('morph', ''))))
            _recip_marker = G_kg.get('reciprocal_marker', '') if _subj_is_plural_2a else ''
            # "Je m'appelle" → a bɛ yɛrɛ kánbìla (réflexif-naming, sujet impersonnel)
            m['S'] = _impersonal
            m['O'] = _recip_marker or _refl_self   # ɲɔgɔn (pluriel) / yɛrɛ (singulier)
            if _root_bm_2a:
                m['V'] = _root_bm_2a  # kánbìla (écrase la nominalization)
            tree['clause_type'] = 'simple'

    # ── 2b. "Dire" présent + ccomp → clause_type='saying_present' (template={S})
    # APRÈS les TransformRules pour avoir V et tam corrects pour détection
    _root_sc_2b = _root_tok_kg.get('semantic_class', '') if _root_tok_kg else ''
    _has_ccomp_2b = bool(m.get('CCOMP'))
    # Vérifier aux:tense UNIQUEMENT sur le verbe ROOT (pas sur les ccomp comme "il est venu")
    _root_idx_2b  = _root_tok_kg.get('orig_index') if _root_tok_kg else None
    _has_aux_tense_2b = any(
        t.get('dep') == 'aux:tense' and t.get('head_index') == _root_idx_2b
        for t in T
    )
    if _root_sc_2b == 'saying' and _has_ccomp_2b and not _has_aux_tense_2b:
        # Présent "dire" = S ko CCOMP ; le template '{S}' laisse le ccomp s'attacher
        tree['clause_type'] = 'saying_present'

    # ── copula_comitative COMPANION : depuis m['O'] (chaîne génitive complète
    # déjà construite par step4_objet, ex: "avec la fille du frère de mon ami"
    # → tête + chaîne entière), AVANT le GraphWalkRule KG générique ci-dessous.
    # Nécessaire ici et pas plus bas : GraphWalkRule 'comitative_companion_gw'
    # capture le premier NOUN dep='nmod' de TOUTE la phrase sans lien avec la
    # construction comitative réelle (ex: 'frère', un nmod intermédiaire de la
    # chaîne, plutôt que 'fille' la vraie tête) — sa propre garde
    # `if m.get(target): continue` le neutralise si COMPANION est déjà posé ici.
    if tree.get('clause_type') == 'copula_comitative':
        _comit_ni_fb_early = G_kg.get('comitative_marker', 'ni') or 'ni'
        if (not m.get('COMPANION') or m.get('COMPANION') == _comit_ni_fb_early) \
                and m.get('O') and m['O'] != _comit_ni_fb_early:
            m['COMPANION'] = m['O']
            m['O'] = ''

    # ── 3. SlotFillRule → slots simples ──────────────────────────────────────
    _apply_slot_fill_rules(tree, m, T, G_kg)

    # ── 4. GraphWalkRule → slots complexes ───────────────────────────────────
    _apply_graph_walk_rules(tree, m, T, G_kg)

    # ── 5. FunctionWord → marqueurs fixes ────────────────────────────────────
    _apply_function_words(tree, m, G_kg)

    # ── 5b. Slots spéciaux post-PatternRule ──────────────────────────────────
    _ct = tree.get('clause_type', '')

    # ── restrictive ne...que : S {TAM} RESTRICT_MK yé ni ATTR tɛ ───────────────
    # TAM calculé depuis tense + neg forcé (tu ne serais qu'un pleutre → tɛ na)
    # Détection backup si step3 a raté (clause_type_init=content_question absorbe "que")
    if not tree.get('restrictive_attr'):
        _neg_t  = any(t.get('role') == 'negation' for t in T)
        _que_r  = next((t for t in T
                        if t.get('dep') in ('mark', 'advmod')
                        and str(t.get('surface', '')).lower()
                              .replace('\u2018', "'").replace('\u2019', "'")
                              .rstrip("'") in ('que', 'qu')), None)
        if _neg_t and _que_r:
            # Gardes contre les faux positifs :
            # 1. ccomp présent ("disent que") → le 'que' est complémenteur, pas restrictif
            # 2. 'plus' est le seul négatif → comparatif "plus ADJ que", pas "ne... que"
            _has_ccomp_t = any(t.get('dep') == 'ccomp' for t in T)
            _only_plus_neg = (any(str(t.get('surface', '')).lower() == 'plus'
                                  and t.get('role') == 'negation' for t in T)
                              and not any(str(t.get('surface', '')).lower().rstrip("'") in ('ne', 'pas', 'plus' )
                                          and t.get('role') == 'negation'
                                          and str(t.get('surface', '')).lower() != 'plus'
                                          for t in T))
            if not _has_ccomp_t and not _only_plus_neg:
                _attr_r = next((t for t in T
                                if t.get('orig_index', 0) > _que_r.get('orig_index', 0)
                                and t.get('pos') in ('NOUN', 'ADJ', 'PROPN')
                                and t.get('bm')), None)
                if _attr_r:
                    tree['restrictive_attr'] = _attr_r.get('bm') or f"[{_attr_r.get('lemma', '?')}]"
                    # Sauvegarder le bm original avant de l'effacer (pour nettoyer m['ADV'])
                    tree['_restrictive_que_bm'] = _que_r.get('bm', '')
                    _que_r['bm'] = ''
    if tree.get('restrictive_attr') and _ct != 'restrictive':
        tree['clause_type'] = 'restrictive'
        _ct = 'restrictive'
    if _ct == 'restrictive' and not m.get('ATTR'):
        m['ATTR'] = tree.get('restrictive_attr', '')
        # Effacer m['ADV'] si = bm du marqueur 'que/qu' (posé par step5 avant kg_gateway)
        _que_bm_saved = tree.get('_restrictive_que_bm', '')
        if _que_bm_saved and m.get('ADV') == _que_bm_saved:
            m['ADV'] = ''
        _rk = G_kg.get('restrictive_exclusive_marker', '') or 'dɔwɛrɛ'
        # TAM forcé neg=True selon tense (pres→tɛ, fut/cond→tɛ na, past→ma)
        # Chercher le tense du cop (serais=fut, était=hab) non détecté par step3 F1
        _cop_tense_tok = next((t for t in T
                               if t.get('dep') == 'cop' and t.get('tense')), None)
        _restr_tense = (_cop_tense_tok.get('tense') if _cop_tense_tok
                        else tree.get('tense', 'pres'))
        _restr_tam = _resolve_tam(_restr_tense, True, G_kg) or 'tɛ'
        tree['tam'] = _restr_tam
        # Non-présent : le copule (être) doit apparaître avant le marqueur restrictif
        # Ex: "tu ne serais qu'un pleutre" → i tɛ na [être] dɔwɛrɛ yé ni sègɛ tɛ
        if _restr_tense not in ('pres', 'hab') and _cop_tense_tok:
            _cop_bm = (_cop_tense_tok.get('bm')
                       or f"[{_cop_tense_tok.get('lemma', 'être')}]")
            m['RESTRICT_MK'] = j(_cop_bm, _rk)
        else:
            m['RESTRICT_MK'] = _rk

    # ── météo impersonnel : il fait chaud/froid → [phénomène] bɛ ────────────────
    if _ct == 'meteorological_impersonal' and not tree.get('meteo_bm'):
        _adj_tok = next((t for t in T
                         if t.get('pos') == 'ADJ'
                         and t.get('dep') in ('advmod', 'xcomp', 'amod', 'attr')), None)
        if _adj_tok:
            _phenom_bm = _adj_tok.get('bm') or f"[{_adj_tok.get('lemma', '?')}]"
            tree['meteo_bm'] = _phenom_bm

    # ── quest_ce_que_modal : Qu'est-ce qu'il peut faire? → S TAM V ka O V_ACT ? ──
    if _ct == 'quest_ce_que_modal':
        # V ← verbe modal dep='dep' (pouvoir=se)
        if not m.get('V'):
            _dep_modal_tok = next((t for t in T
                                   if t.get('dep') == 'dep' and t.get('pos') == 'VERB'
                                   and t.get('semantic_class') == 'modal' and t.get('bm')), None)
            if _dep_modal_tok:
                m['V'] = _dep_modal_tok.get('bm', '')
        # V_ACT inclure l'objet de l'xcomp si présent (fusil mɔ́n)
        if m.get('V_ACTION'):
            _xcomp_tok = next((t for t in T if t.get('dep') == 'xcomp' and t.get('pos') == 'VERB'), None)
            if _xcomp_tok:
                _xcomp_obj = next((t for t in T
                                   if t.get('dep') in ('obj', 'iobj')
                                   and t.get('head_index') == _xcomp_tok.get('orig_index')
                                   and t.get('bm')), None)
                if _xcomp_obj and _xcomp_obj.get('bm') not in m.get('V_ACTION', ''):
                    m['V_ACTION'] = j(_xcomp_obj.get('bm', ''), m['V_ACTION'])

    # ── verb_serial : obj de l'xcomp avant V_ACT (cela vaut la peine de prendre un fusil)
    if _ct == 'verb_serial' and m.get('V_ACTION'):
        _xcomp_t2 = next((t for t in T if t.get('dep') == 'xcomp' and t.get('pos') == 'VERB'), None)
        if _xcomp_t2:
            _xcomp_obj2 = next((t for t in T
                                 if t.get('dep') in ('obj', 'iobj')
                                 and t.get('head_index') == _xcomp_t2.get('orig_index')
                                 and t.get('bm')
                                 and t.get('bm') not in m.get('V_ACTION', '')), None)
            if _xcomp_obj2:
                m['V_ACTION'] = j(_xcomp_obj2.get('bm', ''), m['V_ACTION'])
        # Objet du verbe PRINCIPAL (ex: peine dans "vaut la peine de V") :
        # le template verb_serial n'a pas de slot {O} → le fusionner dans m['V']
        # pour qu'il apparaisse : {S} {TAM} {V=peine+valoir} ka {V_ACT}.
        # Garde : si m['O'] est déjà dans V_ACTION (xcomp obj), ne pas le dupliquer.
        if (m.get('O')
                and m['O'] not in m.get('V', '')
                and m['O'] not in m.get('V_ACTION', '')):
            m['V'] = j(m['O'], m.get('V', ''))
            m['O'] = ''

    # ── "Tu as combien d'enfants?" : quantificateur interrogatif ajouté à O ───────
    # Cas 1 : step1 a posé m['O']=dénw (noun) → on appende jóli.
    # Cas 2 : step4 a posé m['O']='jóli' (combien = dep=obj direct) ET stocké le
    #   nom dans tree['interrog_noun'] → reconstruire m['O'] = dénw + jóli.
    if _ct in ('possession_abstract_pres', 'possession_abstract_pres_interrog',
                'noun_phrase_have') and m.get('O'):
        _interrog_qty = next((t for t in T
                              if t.get('dep') == 'obj'
                              and t.get('role') == 'interrogative'
                              and t.get('bm')), None)
        _interrog_noun_tok = tree.get('interrog_noun')
        if _interrog_noun_tok and _interrog_qty:
            # Reconstruire dénw + jóli depuis le nom stocké par step4
            _noun_bm = (_interrog_noun_tok.get('bm', '')
                        or f"[{_interrog_noun_tok.get('lemma', '')}]")
            if (_interrog_noun_tok.get('is_plural')
                    or str(_interrog_noun_tok.get('surface', '')).endswith('s')):
                if not _noun_bm.endswith('w'):
                    _noun_bm += 'w'
            m['O'] = j(_noun_bm, _interrog_qty.get('bm', ''))
            # COMPANION = marqueur de possession abstraite (fɛ) — le FunctionWord KG
            # l'applique pour possession_abstract_pres mais pas pour la variante _interrog.
            if not m.get('COMPANION'):
                m['COMPANION'] = G_kg.get('possession_abstract_marker', 'fɛ') or 'fɛ'
        elif _interrog_qty and _interrog_qty.get('bm', '') not in m.get('O', ''):
            m['O'] = j(m['O'], _interrog_qty.get('bm', ''))

    # ── #3 "Quelle femme?" : DET interrogatif après NOUN ROOT en content_question ──
    if (_ct == 'content_question' and not m.get('V') and not m.get('O') and m.get('S')):
        _interrog_det = next((t for t in T
                              if t.get('dep') == 'det'
                              and t.get('role') == 'interrogative'
                              and t.get('bm')), None)
        if _interrog_det:
            m['S'] = j(m['S'], _interrog_det.get('bm', ''))
            tree['clause_type'] = 'content_question_nominal'  # template {S} ? sans TAM
            _ct = 'content_question_nominal'

    # ── focus_cleft_naming : C'est X que nous disons Y ──────────────────────────
    # Slots : S=nsubj du dep, V=dep verb bm, PROPN=ROOT PROPN+flat, O=obj du dep
    if _ct == 'focus_cleft_naming':
        _dep_saying = next((t for t in T
                            if t.get('dep') == 'dep' and t.get('pos') == 'VERB'
                            and t.get('semantic_class') == 'saying'), None)
        if _dep_saying:
            _dep_nsubj = next((t for t in T if t.get('dep') == 'nsubj' and t.get('bm')), None)
            _dep_obj   = next((t for t in T if t.get('dep') == 'obj' and t.get('bm')), None)
            # flat:name pour les prénoms composés (Fanta Maa, Kolima Maa)
            _dep_obj_flat = next((t for t in T
                                  if t.get('dep') == 'flat:name'
                                  and _dep_obj and t.get('head_index') == _dep_obj['orig_index']
                                  and t.get('bm')), None)
            _root_flat = next((t for t in T
                               if t.get('dep') == 'flat:name' and _root_tok_kg
                               and t.get('head_index') == _root_tok_kg['orig_index']
                               and t.get('bm')), None)
            if _dep_nsubj:
                m['S'] = _dep_nsubj.get('bm', '')
            _v_bm = _dep_saying.get('bm') or f"[{_dep_saying.get('lemma','dire')}]"
            m['V'] = _v_bm
            _propn_bm = (_root_tok_kg.get('bm') or
                         _root_tok_kg.get('surface', '')) if _root_tok_kg else ''
            if _root_flat:
                _propn_bm = _propn_bm + ' ' + (_root_flat.get('bm') or _root_flat.get('surface', ''))
            m['PROPN'] = _propn_bm.strip()
            if _dep_obj:
                _obj_bm = _dep_obj.get('bm') or _dep_obj.get('surface', '')
                if _dep_obj_flat:
                    _obj_bm += ' ' + (_dep_obj_flat.get('bm') or _dep_obj_flat.get('surface', ''))
                m['O'] = _obj_bm.strip()
            m['OBL_ALL'] = []  # consommé

    # ── psych_emotion + xcomp → transitif (J'aime manger → n bɛ dúnli kànu) ──────
    # Contrairement aux modaux (vouloir → verb_serial), les verbes psych_emotion
    # prennent leur xcomp comme COD nominalisé (O avant V, SOV standard)
    if (_ct == 'verb_serial' and _root_tok_kg
            and _root_tok_kg.get('semantic_class') == 'psych_emotion'
            and m.get('V_ACTION')):
        _vact = m.get('V_ACTION', '')
        _ns = G_kg.get('nominalization_verb_suffix', '')
        # Extraire la base du V_ACTION : 'dúnli kɛ' → 'dún', nominalisé → 'dúnli'
        _vact_base = _vact.split(' ')[0] if _vact else ''
        if _vact_base.endswith(_ns if _ns else ''):
            _nom = _vact_base  # déjà nominalisé
        elif _ns:
            _nom = _vact_base + _ns
        else:
            _nom = _vact_base
        m['O'] = _nom          # dúnli = O (COD nominalisé)
        m['V_ACTION'] = ''     # effacer V_ACTION
        tree['clause_type'] = 'simple'
        _ct = 'simple'

    # ── motion_interrog : déplacer O (locatif de step4) vers OBL avec marqueur la ──
    if _ct == 'motion_interrog' and m.get('O') and not m.get('OBL_ALL'):
        _loc_mk = G_kg.get('locative_postposition', '') or G_kg.get('locative_marker', 'la') or 'la'
        m['OBL_ALL'] = [{'HEAD': m['O'], 'MARKER': _loc_mk, 'local_clause_type': 'locative',
                         'COMPOUND': '', 'MOD': '', 'DEM_PREF': '', 'DEM_SUFF': '',
                         'DEP_TYPE': 'obl:arg', 'COMPOUND_IS_QUANTIFIER': False, 'MARKER_IS_PREFIX': False}]
        m['O'] = ''

    # ── serial_motion_pres + passé + INTRANS_SC → V_RES (résultatif) pour template ──
    # "elle alla trouver" → a táara ɲɛ́sɔ̀rɔ (táa+ra=táara, sans TAM ni ka)
    if (_ct == 'serial_motion_pres'
            and tree.get('tense') not in ('pres', 'hab')
            and not tree.get('neg', False)
            and _root_tok_kg
            and m.get('V') and not m.get('V_RES')):
        from rules.core import INTRANS_SC as _INTRANS_SC_SMP
        if _root_tok_kg.get('semantic_class', '') in _INTRANS_SC_SMP:
            _morpho_res_smp = G_kg.get('morpho_rules', {}).get('resultative', {})
            if _morpho_res_smp:
                m['V_RES'] = apply_morpho_suffix(m['V'], _morpho_res_smp)
            tree['is_transitive'] = False

    # ── content_question : mot interrogatif → O (Qui est-il?, Qui aiment-ils?) ──
    if _ct == 'content_question' and not m.get('O'):
        # ROOT PRON interrogatif (Qui est-il?) ou dep=obj PRON interrogatif (Qui aiment-ils?)
        # Inclure dep=nsubj SEULEMENT si un autre nsubj existe (le vrai sujet)
        # "Qui aiment-ils?" → qui=nsubj mais il y a aussi ils=nsubj → qui est l'objet
        # "Qui a mangé?" → qui=nsubj SEUL → qui est le sujet, pas O
        _other_real_subj = any(s for s in T
                               if s.get('dep') in ('nsubj', 'nsubj:pass')
                               and s.get('role') not in ('relative', 'interrogative'))
        _interrog_pron = next((t for t in T
                               if t.get('pos') == 'PRON'
                               and t.get('role') in ('relative', 'interrogative')
                               and t.get('bm')
                               and (t.get('is_root') or t.get('dep') == 'obj'
                                    or (t.get('dep') in ('nsubj', 'nsubj:pass') and _other_real_subj))), None)
        if _interrog_pron:
            m['O'] = _interrog_pron.get('bm', '')
            # Si V a été nominalisé (kànuli kɛ) à cause d'O vide → restaurer bm de base
            if _root_tok_kg and m.get('V'):
                _base_bm = _root_tok_kg.get('bm', '')
                _ns = G_kg.get('nominalization_verb_suffix', '')
                _as = G_kg.get('coord_action_suffix', '')
                if _base_bm and m['V'] == _base_bm + _ns + (' ' + _as if _as else ''):
                    m['V'] = _base_bm  # restaurer forme non nominalisée
            if _interrog_pron.get('is_root'):
                tree['clause_type'] = 'equative_interrog'
                _ct = 'equative_interrog'

    # ── content_question + O déjà posé + interrogatif ROOT → equative_interrog ──
    # "Qui est-il?" : step2 a posé m['O']='jɔn' (qui=ROOT), la ligne 955 saute
    # (not m.get('O') = False). Détecter ici que le ROOT est l'interrogatif en O.
    if (_ct == 'content_question' and m.get('O') and not m.get('V')):
        _interrog_root_eq = next((t for t in T
                                  if t.get('pos') == 'PRON'
                                  and t.get('role') in ('relative', 'interrogative')
                                  and t.get('is_root')
                                  and t.get('bm') == m.get('O')), None)
        if _interrog_root_eq:
            tree['clause_type'] = 'equative_interrog'
            _ct = 'equative_interrog'

    # ── content_question : interrogatif adverbial (pourquoi, quand) → ADV seul ──
    # Évite la duplication dans template {S} {TAM} {O} {V} {ADV} ?
    if _ct == 'content_question' and m.get('O'):
        _interrog_adv = next((t for t in T
                              if t.get('dep') in ('advmod', 'obl:mod')
                              and t.get('role') == 'interrogative'
                              and t.get('bm') == m['O']), None)
        if _interrog_adv:
            if not m.get('ADV'):
                m['ADV'] = m['O']  # déplacer vers ADV si pas encore là
            m['O'] = ''  # supprimer de O pour éviter duplication


    # ── copula_comitative : COMPANION ← oblique comitatif (HEAD réel, pas juste marqueur)
    # Ne s'applique QUE si un bloc précédent (avant le GraphWalkRule générique,
    # voir plus haut) n'a pas déjà posé un COMPANION valide depuis m['O'].
    if (_ct == 'copula_comitative'
            and not (m.get('COMPANION') and m['COMPANION'] != G_kg.get('comitative_marker', 'ni'))):
        # OBL_ALL[0].HEAD = '{companion} yé' — on veut juste le nom sans le marqueur final
        _comit_end = G_kg.get('comitative_end_marker', 'yé') or 'yé'
        _obl_heads = [o.get('HEAD', '') for o in m.get('OBL_ALL', []) if o.get('HEAD')]
        if _obl_heads:
            _companion_raw = _obl_heads[0]
            # Retirer le marqueur de fin si présent (n fúrucɛ yé → n fúrucɛ)
            if _companion_raw.endswith(' ' + _comit_end):
                _companion_raw = _companion_raw[: -(len(_comit_end) + 1)]
            m['COMPANION'] = _companion_raw
            m['OBL_ALL'] = []  # consommé par le template

    # Pour simple_venir_de_pres/past : SRC ← token obl:arg (Bamako, école…)
    if _ct in ('simple_venir_de_pres', 'simple_venir_de_past') and not m.get('SRC'):
        _src_tok = next((t for t in T
                         if t.get('dep') in ('obl:arg', 'obl', 'obl:mod')
                         and t.get('bm')
                         and t.get('role') not in ('interrogative',)), None)
        if _src_tok:
            m['SRC'] = _src_tok.get('bm', '')
            # Vider OBL_ALL pour éviter le doublon
            m['OBL_ALL'] = [x for x in m.get('OBL_ALL', [])
                            if x.get('HEAD') != _src_tok.get('bm', '')]

    # Pour content_question_venir_de : O ← token obl:arg (où interrogatif)
    if _ct == 'content_question_venir_de' and not m.get('O'):
        _obl_where = next((t for t in T
                           if t.get('dep') in ('obl:arg', 'obl', 'advmod')
                           and t.get('bm')
                           and t.get('role') in ('interrogative', 'locative')), None)
        if _obl_where:
            m['O'] = _obl_where.get('bm', '')

    # Pour passive_statif avec expletive + cop + participe passé ADJ :
    # step6 section 4 ne s'active que pour VERB passif → il faut remplir m['V'] ici
    # On applique aussi le suffixe statif (-len) que add_passive_len aurait mis
    if _ct == 'passive_statif' and not m.get('V'):
        _root_pp = next((t for t in T
                         if t.get('is_root') and t.get('pos') == 'ADJ'
                         and t.get('is_participe_passe') and t.get('bm')), None)
        if _root_pp:
            _v_bm = _root_pp.get('bm', '')
            _sfx_len = G_kg.get('morpho_rules', {}).get('statif', {}).get('suffix', '')
            m['V'] = (_v_bm + _sfx_len
                      if _sfx_len and not _v_bm.endswith(_sfx_len)
                      else _v_bm)

    # Pour equative avec ROOT PRON : O ← bm du ROOT PRON (il n'est rien, il est lui)
    if _ct == 'equative' and not m.get('O'):
        _root_pron_eq = next((t for t in T
                              if t.get('is_root') and t.get('pos') == 'PRON' and t.get('bm')), None)
        if _root_pron_eq:
            m['O'] = _root_pron_eq.get('bm', '')

    # Correction : "il n'est rien" = expl:subj (il→a) + copula ROOT + nsubj indéfini négatif (rien→foyi)
    # SpaCy met "rien" en nsubj (faux) → step2 pose S='foyi'. Il faut S='a' et O='foyi'.
    if _ct == 'equative' and m.get('S') and not m.get('O'):
        _indef_nsubj = next((t for t in T
                             if t.get('dep') in ('nsubj', 'nsubj:pass')
                             and t.get('role') == 'indefinite_neg'
                             and t.get('bm')), None)
        if _indef_nsubj:
            _expl_subj = next((t for t in T
                               if t.get('dep') in ('expl:subj', 'expl:comp', 'expletive')
                               and t.get('bm')), None)
            if _expl_subj:
                m['O'] = _indef_nsubj.get('bm', '')   # rien → foyi = O
                m['S'] = _expl_subj.get('bm', '')      # il   → a   = S

    # Pour identificatory : S ← ROOT PRON / PROPN / NOUN (c'est moi, c'est Musa, c'est le médecin)
    if _ct == 'identificatory':
        # Appartenance pronominale "c'est pour toi" → {O} dòn  (ex: "i ta dòn")
        # m['O'] = 'i ta' déjà posé par ownership.py ; m['V'] = copule dòn.
        # Ne pas écraser m['S'] depuis root_pron sinon le PRON sert de sujet et non de bénéficiaire.
        if tree.get('ownership_o_is_pron') and m.get('O'):
            # Template {S} de {TAM} → TAM doit porter 'ta dòn' pour 'i de ta dòn'
            _ta_v = G_kg.get('ownership_marker') or 'ta'
            _statif_sup = G_kg.get('statif_pos_support', '')
            tree['tam'] = j(_ta_v, _statif_sup)
        else:
            _root_pron = next((t for t in T
                               if t.get('is_root') and t.get('pos') == 'PRON' and t.get('bm')), None)
            if _root_pron:
                m['S'] = _root_pron.get('bm', '')  # Override 'o' démonstratif par le vrai prédicat
        if not tree.get('ownership_o_is_pron') and not m.get('PROPN'):
            _root_propn = next((t for t in T
                                if t.get('is_root') and t.get('pos') == 'PROPN' and t.get('bm')), None)
            if _root_propn:
                if m.get('S'):
                    # "C'est moi Hawa" : PRON(moi→n) + PROPN(Hawa) tous deux ROOT
                    # → n dòn Hawa (PROPN après le copule via ADV)
                    m['ADV'] = _root_propn.get('bm', '')
                else:
                    m['PROPN'] = _root_propn.get('bm', '')
                    m['O'] = ''  # Effacer m['O'] posé par step6
            elif not m.get('S'):
                # ROOT NOUN (c'est le médecin → medesɛn de dòn)
                _root_noun = next((t for t in T
                                   if t.get('is_root') and t.get('pos') == 'NOUN' and t.get('bm')), None)
                if _root_noun:
                    m['S'] = _root_noun.get('bm', '')
                    m['O'] = ''

    # Pour possession_age_interrog : INTERROG ← DET interrogatif
    if _ct == 'possession_age_interrog' and not m.get('INTERROG'):
        _interrog_det = next((t for t in T
                              if t.get('role') == 'interrogative'
                              and t.get('dep') in ('det', 'advmod')
                              and t.get('bm')), None)
        if _interrog_det:
            m['INTERROG'] = _interrog_det.get('bm', '')

    # ── 6. Règles legacy (ancien système FeaturePattern/ConstructionRule) ─────
    if not G_kg.get('kg_rules'):
        return tree
    _features = _build_features(tree, T, G_kg)
    best_rule, best_prio = {}, -1
    for key, rule in G_kg.get('kg_rules', {}).items():
        if len(key) == 3:
            feat, val, ctx = key
            if ctx: continue
        else:
            feat, val = key
        if feat in _features and str(_features[feat]) == str(val):
            prio = rule.get('priority', 0) or 0
            if prio > best_prio:
                best_prio = prio
                best_rule = rule
    if best_rule and best_prio >= 60:
        rule_name = best_rule.get('rule_name', '')
        if 'passive_statif' in rule_name and best_prio >= 85:
            # Guard : aux:pass au passé = action passive ("cela fut fait") → résultatif,
            # PAS statif. Statif seulement pour présent ("est fermée" = état résultant).
            _aux_pass_past = any(t.get('dep') == 'aux:pass'
                                 and t.get('tense') in ('past', 'plup')
                                 for t in T)
            if not _aux_pass_past:
                tree['_passive_statif'] = True
        if 'refl_' in rule_name and best_prio >= 65:
            rt = best_rule.get('refl_treatment', '')
            if rt:
                tree['_kg_refl_treatment'] = rt
        if rule_name == 'meteorological' and best_prio >= 90:
            mv = best_rule.get('motion_verb', '')
            if mv and not tree.get('meteo_motion'):
                tree['meteo_motion'] = mv
        if 'imperative_' in rule_name and best_prio >= 80:
            tree['_kg_imp_prefix']   = best_rule.get('prefix', '')
            tree['_kg_imp_template'] = best_rule.get('template', '')
        if best_rule.get('template') and best_prio >= 75:
            tree['_kg_template']   = best_rule.get('template', '')
            tree['_kg_rule_name']  = rule_name
            tree['_kg_word_order'] = best_rule.get('word_order', '')

    # ── content_question_adv : {S} {TAM} {V_NOM} {O} ? — V_NOM jamais posé ────
    # ailleurs (slot mort) : le verbe racine disparaissait entièrement du rendu
    # ("Comment cela se fait-il ?" → "a bɛ cógo dì ?", sans trace de 'faire').
    # Lu en tout dernier via tree['clause_type'] (pas la variable locale _ct,
    # un instantané pris plus haut qui ne reflète pas la transition posée par
    # un second passage de PatternRule KG : content_question → content_question_adv).
    if tree.get('clause_type') == 'content_question_adv' and not m.get('V_NOM') and _root_tok_kg:
        _vnom_bm = _root_tok_kg.get('bm') or f"[{_root_tok_kg.get('lemma', '')}]"
        if not m.get('O') or _vnom_bm != m.get('O'):
            m['V_NOM'] = _vnom_bm

    # ── Filet de sécurité : verbe réflexif racine disparu du rendu ────────────
    # Un verbe réflexif RÉEL (root_tok avec marqueur 'se' expl:comp/expl:pass)
    # peut se retrouver sans trace dans aucun slot (V/V_ACTION/V_NOM/O) quand
    # la classification LLM de sa catégorie réflexive (pronominal/posture/
    # actif/accidentel — non-déterministe) l'aiguille à tort vers une
    # construction qui ne le consomme pas (ex: "Comment cela se fait-il ?"
    # classé POSTURE à tort → verbe absorbé nulle part). Scopé strictement
    # aux verbes réflexifs pour ne pas affecter d'autres constructions
    # (ex: le verbe d'une relative acl:relcl, qui n'est pas root_tok ici).
    _root_has_refl_marker = bool(_root_tok_kg) and any(
        x.get('dep') in ('expl:comp', 'expl:pass')
        and x.get('head_index') == _root_tok_kg.get('orig_index')
        for x in T)
    if (_root_tok_kg and _root_tok_kg.get('pos') == 'VERB' and _root_has_refl_marker
            and not m.get('V') and not m.get('V_ACTION')
            and not m.get('V_NOM') and not m.get('QUAL')
            and not tree.get('_is_passive') and not tree.get('_is_statif')):
        _root_consumed_elsewhere = any(
            _root_tok_kg.get('bm') and _root_tok_kg.get('bm') in (m.get(k) or '')
            for k in ('S', 'O', 'ADV', 'QUAL'))
        if not _root_consumed_elsewhere:
            m['V'] = _root_tok_kg.get('bm') or f"[{_root_tok_kg.get('lemma', '')}]"

    return tree


def apply_kg_semantic_behaviors(tree: dict, T: list, G_kg: dict) -> None:
    """Annote les tokens avec leurs comportements sémantiques KG."""
    behaviors = G_kg.get('semantic_behaviors', {})
    if not behaviors:
        return
    for tok in T:
        sc = tok.get('semantic_class', '')
        if sc and sc in behaviors:
            for btype, bdata in behaviors[sc].items():
                key = f'_kg_{btype}'
                if key not in tok:
                    tok[key] = bdata.get('value', '')
