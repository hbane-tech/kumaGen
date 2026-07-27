"""
rules/steps/step4_objet/coord_relative_on_amod.py
Relative coordonnée réduite sur un amod ("maison CHARMANTE et QUI ÉTAIT
digne d'elle") : spaCy attache le second prédicat ('digne') en dep='conj'
de l'amod du ROOT plutôt qu'en acl:relcl sur le ROOT lui-même — aucun autre
chemin du pipeline ne regarde les enfants (nsubj/cop/obl:arg) d'un conj
d'amod, donc ce prédicat entier disparaissait silencieusement, tandis que
root_tok (le nom antécédent) se faisait à tort fusionner dans m['O'] par le
garde cop générique de step3_verbe.py (corrigé séparément — voir garde
_root_direct_cop) faute de savoir que le cop qu'il détectait ne le
concernait pas (bug trouvé 2026-07-26).
"""
from rules.core import j, adj_man, _resolve_tam


def run(T, tree, m, processed_indices, G_kg, NX_G, root_tok):
    if not (root_tok and root_tok.get('pos') == 'NOUN'
            and root_tok['orig_index'] not in processed_indices):
        return
    _amod = next((t for t in T if t.get('dep') == 'amod'
                  and t.get('head_index') == root_tok['orig_index']), None)
    if not _amod:
        return
    _pred = next((t for t in T if t.get('dep') == 'conj' and t.get('pos') == 'ADJ'
                  and t.get('head_index') == _amod['orig_index']
                  and any(x.get('dep') == 'cop' and x.get('head_index') == t['orig_index']
                          for x in T)), None)
    if not _pred:
        return
    _pred_nsubj = next((t for t in T if t.get('dep') == 'nsubj'
                        and t.get('head_index') == _pred['orig_index']), None)
    _pred_cop = next((t for t in T if t.get('dep') == 'cop'
                      and t.get('head_index') == _pred['orig_index']), None)

    _consumed = {root_tok['orig_index'], _amod['orig_index'], _pred['orig_index']}
    if _pred_nsubj:
        _consumed.add(_pred_nsubj['orig_index'])
    if _pred_cop:
        _consumed.add(_pred_cop['orig_index'])

    # Antécédent : root_tok + amod ("maison charmante" → "só sárama")
    _root_bm = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
    _amod_bm = _amod.get('bm') or f"[{_amod.get('lemma')}]"
    if _amod.get('pos') == 'ADJ':
        _amod_bm = adj_man(_amod_bm, is_classifying=_amod.get('is_classifying_adj', False))
    _antecedent_bm = j(_root_bm, _amod_bm)

    # Complément obl:arg du prédicat coordonné ("digne D'ELLE" → "ni a yé")
    _pred_obl = next((t for t in T if t.get('dep') == 'obl:arg'
                      and t.get('head_index') == _pred['orig_index']), None)
    _comp_str = ''
    if _pred_obl and _pred_obl.get('bm'):
        _comitative_mk = G_kg.get('comitative_marker', '') or 'ni'
        _comitative_end = G_kg.get('comitative_end_marker', '') or 'yé'
        _comp_str = j(_comitative_mk, _pred_obl.get('bm', ''), _comitative_end)
        _consumed.add(_pred_obl['orig_index'])
        # tolère le head_index auto-référent que spaCy pose parfois sur la
        # préposition élidée ("d'") : repli positionnel (case juste avant le
        # complément) si le head_index déclaré ne pointe pas vers lui.
        _pred_obl_case = (
            next((t for t in T if t.get('dep') == 'case'
                 and t.get('head_index') == _pred_obl['orig_index']), None)
            or next((t for t in T if t.get('dep') == 'case'
                     and t.get('orig_index') == _pred_obl['orig_index'] - 1), None))
        if _pred_obl_case:
            _consumed.add(_pred_obl_case['orig_index'])

    _pred_tense = _pred_cop.get('tense', 'past') if _pred_cop else 'past'
    _pred_tam = _resolve_tam(_pred_tense, False, G_kg) or G_kg.get('past_tam_default', '') or 'tùn'
    _pred_bm = _pred.get('bm') or f"[{_pred.get('lemma')}]"
    _rel_marker = G_kg.get('relative_marker', '') or 'min'
    _rel_str = j(_rel_marker, _pred_tam, _pred_bm, _comp_str)

    # Court-circuit du rendu générique (même schéma que _is_free_relative /
    # _is_privative) : sans ça, un cop/comitative SANS RAPPORT ailleurs dans
    # la phrase ('était' sur 'digne', 'entre' mal étiqueté comitatif sur
    # "entre toutes") fait matcher à tort le PatternRule KG 'copula_comitative'
    # (cond_has_cop/cond_has_comitative_case ne vérifient aucun gouverneur),
    # qui écrase tree['clause_type'] et regénère 'maison' comme un faux
    # prédicat comitatif — dupliquant S et insérant un TAM parasite (bug
    # trouvé 2026-07-26 : "só sárama tùn bɛ ni só yé ni [tout] yé mìn...").
    # m['OBL_ALL'] reste peuplé normalement (par step5_obliques, y compris
    # le wagon "entre toutes" laissé volontairement intact) ; step7 assemble
    # le final_string une fois ces obliques connus.
    tree['_is_coord_relative_on_amod'] = True
    tree['_coord_rel_prefix'] = _antecedent_bm
    tree['_coord_rel_suffix'] = _rel_str
    processed_indices.update(_consumed)
