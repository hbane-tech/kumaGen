"""
Participes bambara :
  run_resultatif  → -ra/-la/-na (accompli intransitif) ou équatif
  run_potential   → -ta (potential : capacité/aptitude) ← NOUVEAU
"""
from rules.core import j, _resolve_tam


def run_resultatif(T, tree, m, processed_indices, G_kg, root_tok, _has_expletive):
    # Stocker le flag dans tree pour tree_to_bambara
    tree['is_participe_passe'] = True
    tree['participe_bm'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
    """
    ADJ is_participe_passe ou bm starts '[' → résultatif ou équatif selon contexte.
    """
    _morph_str = str(root_tok.get('morph', ''))
    _is_true_participe = (
        'VerbForm=Part' in _morph_str
        or _has_expletive
        or not any(x.get('dep') == 'cop' for x in T))

    if not _is_true_participe:
        # ADJ + cop + pas expletif → équatif (profession, état)
        if root_tok.get('is_participe_passe'):
            root_tok['is_participe_passe'] = False
            root_tok['is_statif'] = True
        tree['clause_type'] = 'equative'
        m['O'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
        tree['tam'] = (_resolve_tam('pres', True, G_kg) if tree.get('neg')
                       else G_kg.get('equative_marker', 'yé'))
        processed_indices.add(root_tok['orig_index'])
        return

    if ((root_tok.get('is_participe_passe') and 'VerbForm=Part' in _morph_str)
            or (not root_tok.get('is_valeur')
                and not root_tok.get('is_statif')
                and not root_tok.get('is_participe_passe')
                and root_tok.get('bm', '').startswith('['))):
        _has_cop_or_expletive = (
            _has_expletive
            or any(x.get('dep') == 'cop' for x in T)
            or any(x.get('role') == 'expletive' for x in T))
        _subj_is_pron = any(x.get('dep') in ('nsubj', 'nsubj:pass')
                            and x.get('pos') == 'PRON' for x in T)
        if _has_cop_or_expletive or _subj_is_pron:
            # Résultatif : S TAM(vide) V+ra/la/na
            tree['clause_type']  = 'simple'
            tree['is_transitive'] = False
            tree['tense']        = 'past'
            tree['tam']          = ''
            _v = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
            # Suffixe ra/la/na appliqué dans tree_to_bambara via V_SUFFIX ou dans V
            m['V'] = _v
            if not m.get('S'):
                _expl_tok = next((x for x in T if x.get('role') == 'expletive'
                                  and x.get('bm')), None)
                if _expl_tok:
                    m['S'] = _expl_tok.get('bm')
                else:
                    _subj = next((x for x in T if x.get('dep') in ('nsubj', 'nsubj:pass')
                                  and x.get('bm')), None)
                    if _subj:
                        m['S'] = _subj.get('bm')
        else:
            tree['clause_type'] = 'equative'
            m['O'] = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
            tree['tam'] = (_resolve_tam('pres', True, G_kg) if tree.get('neg')
                           else G_kg.get('equative_marker', 'yé'))
        processed_indices.add(root_tok['orig_index'])


def run_potential(T, tree, m, processed_indices, G_kg, root_tok):
    """
    Participe potential : V + -ta (aptitude, état potentiel).
    Détection : root_tok.get('is_potential') is True
                ou role='potential' sur un token fils
                ou semantic_class='potential'.
    Ex : 'faisable' → kɛta dòn
    """
    tree['clause_type'] = 'statif'   # structure : S dòn/tɛ V-ta
    _base = root_tok.get('bm') or f"[{root_tok.get('lemma')}]"
    # Strip -a final avant -ta si présent
    if _base.endswith('a') and not any(_base.endswith(s) for s in ('ba', 'ma', 'ka')):
        _base = _base[:-1]
    if not _base.endswith('ta'):
        _base += 'ta'
    m['QUAL'] = _base
    tree['tam'] = 'tɛ' if tree.get('neg') else 'dòn'
    processed_indices.add(root_tok['orig_index'])