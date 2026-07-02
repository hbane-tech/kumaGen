"""
rules/steps/step_impersonal.py
Constructions impersonnelles : il faut/doit, il s'agit de, il arrive,
il semble, il manque, il reste.

Extraction pure : lit ImpersonalRule KG pour le template,
extrait les slots depuis les tokens, remplit le template.
Aucun bambara hardcodé.
"""
from rules.core import j, _resolve_tam
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
    lemma = _lemma(root_tok)
    _imp_lemmas = G_kg.get('impersonal_trigger_lemmas', set())
    if not _imp_lemmas or lemma not in _imp_lemmas:
        return False
    if lemma == 'falloir':
        return True
    if lemma == 'devoir':
        return _find_il(T) is not None
    if lemma == 'agir':
        if not _find_il(T):
            return False
        return any(x.get('dep') == 'expl:comp'
                   and str(x.get('surface', '')).lower().rstrip("'").rstrip('’') in ('se', 's')
                   for x in T)
    if lemma in ('arriver', 'sembler'):
        if not _find_il(T):
            return False
        return (any(x.get('dep') in ('ccomp', 'advcl') and x.get('pos') == 'VERB' for x in T)
                or any(x.get('dep') in ('nsubj', 'nsubj:pass', 'obj')
                       and x.get('pos') in ('NOUN', 'PROPN')
                       and str(x.get('surface', '')).lower() not in ('il', 'elle', 'on', 'lui')
                       for x in T))
    if lemma in ('manquer', 'rester'):
        if not _find_il(T):
            return False
        return any(x.get('dep') in ('nsubj', 'nsubj:pass', 'obj')
                   and x.get('pos') in ('NOUN', 'PROPN')
                   and str(x.get('surface', '')).lower() not in ('il', 'elle', 'on', 'lui')
                   for x in T)
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
        slots['V2'] = _xcomp.get('bm', '')

    # Clause ccomp slots (S2, TAM2, O2, V2, ADV)
    _ccomp = _find_ccomp(T)
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
