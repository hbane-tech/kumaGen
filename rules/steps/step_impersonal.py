"""
rules/steps/step_impersonal.py
Constructions impersonnelles : il faut/doit, il s'agit de, il arrive,
il semble, il manque, il reste.

Extraction pure : lit ImpersonalRule KG pour le template,
extrait les slots depuis les tokens, remplit le template.
Aucun bambara hardcodé.
"""
from rules.core import j, _resolve_tam, nominalize_verb, INTRANS_SC
from rules.kg_rule_engine import fill_template


# Lemmes lus depuis ImpersonalRule KG (chargés dans G_kg['impersonal_trigger_lemmas'])


def _find_il(T):
    return next((x for x in T if str(x.get('surface', '')).lower() == 'il'), None)


def _lemma(tok):
    return (tok.get('raw_lemma') or tok.get('lemma', '')).lower()


def _has_negation(T, G_kg):
    _NEG = G_kg.get('neg_surfaces', frozenset())
    return (any(x.get('is_neg', False) for x in T)
            or any(str(x.get('surface', '')).lower() in _NEG
                   and x.get('dep') in ('advmod', 'det') for x in T))


def _is_triggered(T, root_tok, G_kg):
    """Le dispatch se fait sur mark_pattern (propriété KG ImpersonalRule),
    pas sur le lemme littéral — un même mark_pattern encode la même
    exigence structurelle quel que soit le lemme qui le porte (décision
    2026-07-12, audit hardcode-KG)."""
    lemma = _lemma(root_tok)
    _imp_lemmas = G_kg.get('impersonal_trigger_lemmas', set())
    if not _imp_lemmas or lemma not in _imp_lemmas:
        return False
    _by_pattern = G_kg.get('impersonal_lemmas_by_pattern', {})
    _ccomp_lemmas = _by_pattern.get('ccomp', set())
    # mark_pattern='' recouvre deux sous-groupes distincts : falloir/devoir
    # (aussi présents dans le groupe 'ccomp' via leur variante imp_falloir_que
    # — le choix inf/que se fait au rendu du template, pas ici : déclenchent
    # sans condition) vs manquer/rester (jamais dans le groupe ccomp,
    # nécessitent 'il').
    if lemma in _by_pattern.get('', set()):
        if lemma in _ccomp_lemmas:
            return True  # falloir/devoir
        return _find_il(T) is not None  # manquer/rester
    if lemma in _by_pattern.get('expl:comp_se', set()):
        if not _find_il(T):
            return False
        return any(x.get('dep') == 'expl:comp'
                   and str(x.get('surface', '')).lower().rstrip("'").rstrip('’') in ('se', 's')
                   for x in T)
    if lemma in _by_pattern.get('ccomp', set()):
        if not _find_il(T):
            return False
        return (any(x.get('dep') in ('ccomp', 'advcl') and x.get('pos') == 'VERB' for x in T)
                or any(x.get('dep') in ('nsubj', 'nsubj:pass', 'obj')
                       and x.get('pos') in ('NOUN', 'PROPN')
                       and str(x.get('surface', '')).lower() not in ('il', 'elle', 'on', 'lui')
                       for x in T))
    return False


def _find_ccomp(T):
    return next((x for x in T if x.get('dep') in ('ccomp', 'advcl', 'xcomp')
                 and x.get('pos') == 'VERB'), None)


def _find_real_noun(T):
    return next((x for x in T
                 if x.get('dep') in ('nsubj', 'nsubj:pass', 'obj')
                 and x.get('pos') in ('NOUN', 'PROPN')
                 and str(x.get('surface', '')).lower() not in ('il', 'elle', 'on', 'lui')
                 and x.get('bm')), None)


def _extract_slots(T, rule, G_kg, _tam, _neg):
    """Extrait les slots nécessaires au template ImpersonalRule."""
    slots = {'TAM': _tam}
    _AUX_LEMMAS = G_kg.get('impersonal_trigger_lemmas', set()) | {'être', 'avoir'}

    # O (objet principal)
    _obj = next((x for x in T
                 if x.get('dep') in ('obj', 'obl:arg')
                 and x.get('pos') in ('NOUN', 'PROPN', 'PRON')
                 and x.get('bm')), None)
    if _obj:
        slots['O'] = _obj.get('bm', '')

    # V2 (verbe complémentaire xcomp/ccomp)
    _xcomp = next((x for x in T
                   if x.get('dep') in ('xcomp', 'ccomp', 'dep')
                   and x.get('pos') == 'VERB'
                   and x.get('bm')
                   and (x.get('lemma', '') or '').lower() not in _AUX_LEMMAS), None)
    if not _xcomp:
        _xcomp = next((x for x in T
                       if x.get('dep') == 'ROOT' and x.get('pos') == 'VERB'
                       and x.get('bm')), None)
    if _xcomp:
        # Verbe transitif/support sans COD propre (ex: "il ne faut pas manger")
        # → même nominalisation que partout ailleurs dans le pipeline (O=nom
        # d'action, V2=marqueur support 'kɛ'), pas le bm nu. 'O' n'est posé ici
        # que si le xcomp a son propre objet (dep='obj' rattaché à lui) — sinon
        # le _obj générique ci-dessus a pu capturer un objet d'un AUTRE verbe
        # (ex: 'falloir' n'a pas d'objet propre) et ne doit pas bloquer la
        # nominalisation.
        _xcomp_has_own_obj = any(
            x.get('dep') == 'obj' and x.get('head_index') == _xcomp.get('orig_index')
            for x in T)
        _it = _xcomp.get('intransitive_type', '')
        _sc = _xcomp.get('semantic_class', '')
        _needs_nom = (
            _it in ('nominalized', 'ACTION', 'support')
            and _sc not in INTRANS_SC
            and _sc not in ('having', 'technique', 'consumption_liquid')
            and not _xcomp_has_own_obj
            # Ne pas écraser un slot['O'] déjà posé par le vrai complément
            # (ex: "il s'agit de toi" → 'toi' dep='obl:arg', pas 'obj', donc
            # _xcomp_has_own_obj=False à tort — mais l'objet réel a déjà été
            # trouvé par la recherche _obj ci-dessus et ne doit pas être
            # remplacé par la nominalisation du verbe lui-même. Décision
            # 2026-07-15, bug : "il s'agit de toi/d'argent" perdait le
            # complément réel au profit du mauvais candidat lexical de 'agir'.
            and 'O' not in slots)
        if _needs_nom:
            slots['O']  = nominalize_verb(_xcomp.get('bm', ''), _xcomp.get('action_noun', ''), G_kg)
            slots['V2'] = G_kg.get('coord_action_suffix', '')
        else:
            slots['V2'] = _xcomp.get('bm', '')

        # Adverbe modifiant l'infinitif xcomp (ex: "il faut parler CLAIREMENT")
        # — la branche _ccomp plus bas a la même extraction (_adv2), mais elle
        # est explicitement désactivée quand _ccomp et _xcomp sont le MÊME
        # token (cas normal pour "falloir"+xcomp), donc son ADV n'était jamais
        # posé. Bug trouvé 2026-07-19 : "il faut parler clairement" perdait
        # l'adverbe silencieusement.
        _adv_xcomp = next((x for x in T if x.get('dep') == 'advmod'
                           and x.get('head_index') == _xcomp.get('orig_index')
                           and x.get('bm')
                           and x.get('role') not in ('negation', 'temporal')), None)
        if _adv_xcomp:
            slots['ADV'] = _adv_xcomp.get('bm', '')

    # Clause ccomp slots (S2, TAM2, O2, V2, ADV)
    # Garde : si _ccomp est le MÊME token que _xcomp (ex: 'manger' avec dep=xcomp
    # sous 'falloir'), le bloc xcomp ci-dessus l'a déjà traité (nominalisation
    # incluse) — le retraiter ici écraserait slots['V2'] avec le bm nu, perdant
    # la nominalisation ("dumuni kɛ" → "dumuni dún").
    _ccomp = _find_ccomp(T)
    if _ccomp and _xcomp and _ccomp.get('orig_index') == _xcomp.get('orig_index'):
        _ccomp = None
    if _ccomp:
        _s2 = next((x for x in T if x.get('dep') in ('nsubj', 'nsubj:pass')
                    and x.get('head_index') == _ccomp['orig_index'] and x.get('bm')), None)
        _o2 = next((x for x in T if x.get('dep') == 'obj'
                    and x.get('head_index') == _ccomp['orig_index'] and x.get('bm')), None)
        _adv2 = next((x for x in T if x.get('dep') == 'advmod'
                      and x.get('head_index') == _ccomp['orig_index'] and x.get('bm')
                      and x.get('role') not in ('negation', 'temporal')), None)
        _sub_t = _ccomp.get('tense') or 'pres'
        if _sub_t in ('sub', 'cond'):
            _sub_t = 'pres'
        slots['S2']   = _s2.get('bm', '') if _s2 else ''
        slots['TAM2'] = _resolve_tam(_sub_t, False, G_kg) or G_kg.get('tam_default', '')
        slots['O2']   = _o2.get('bm', '') if _o2 else ''
        slots['V2']   = _ccomp.get('bm') or G_kg.get('coord_action_suffix', '')
        slots['ADV']  = _adv2.get('bm', '') if _adv2 else ''

        # LOC (locatif dans le ccomp)
        _obl = next((x for x in T if x.get('dep') in ('obl', 'obl:mod', 'obl:arg')
                     and x.get('head_index') == _ccomp['orig_index'] and x.get('bm')), None)
        if _obl:
            _loc_case = next((x for x in T if x.get('dep') == 'case'
                              and x.get('role') == 'locative'
                              and x.get('head_index') == _obl['orig_index']), None)
            _loc_mk = G_kg.get('locative_suffix', '') or (list(G_kg.get('locative_markers', ['la']))[0] if G_kg.get('locative_markers') else '')
            slots['LOC'] = j(_obl.get('bm', ''), _loc_mk) if _loc_case else ''

    return slots


def run(T, tree, G_kg, root_tok):
    """
    Retourne la string bambara depuis le template ImpersonalRule KG,
    ou None si pas de construction impersonnelle détectée.
    """
    if not root_tok:
        return None
    lemma = _lemma(root_tok)
    if not _is_triggered(T, root_tok, G_kg):
        return None

    # Lire ImpersonalRule depuis KG
    imp_rules = G_kg.get('kg_impersonal_rules', [])
    rule = next((r for r in imp_rules
                 if str(r.get('trigger_lemma', '')).lower() == lemma), None)
    if not rule:
        return None

    _neg   = _has_negation(T, G_kg)
    _tense = root_tok.get('tense') or 'pres'
    _tam   = _resolve_tam(_tense, _neg, G_kg) or G_kg.get('tam_default', '')

    # Sélectionner le template (négatif ou positif)
    template = (rule.get('template_neg') if _neg else None) or rule.get('template', '')
    if not template:
        return None

    # Extraire les slots depuis les tokens
    slots = _extract_slots(T, rule, G_kg, _tam, _neg)

    # Cas spécial : rester/manquer → chercher le nom réel (pas 'il')
    _real = _find_real_noun(T)
    if _real and 'O' not in slots:
        slots['O'] = _real.get('bm', '')

    # Remplir le template depuis KG
    result = fill_template(template, slots)
    return result.strip() if result and result.strip() else None
