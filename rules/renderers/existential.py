"""
rules/renderers/existential.py
Clause types : existential_nominal, existential_absolute, existential_localized
"""
from rules.core import j


def render_existential_nominal(tree, m, O, neg, obl_strings):
    _exist_op = 'tɛ' if neg else 'bɛ'
    _normal = [obl_strings[i] for i, c in enumerate(m.get('OBL_ALL', []))
               if i < len(obl_strings) and c.get('DEP_TYPE') != 'acl:relcl']
    _relcl  = [obl_strings[i] for i, c in enumerate(m.get('OBL_ALL', []))
               if i < len(obl_strings) and c.get('DEP_TYPE') == 'acl:relcl']
    return j(O, *_normal, _exist_op, ',', *_relcl)


def render_existential_absolute(tree, S, O, neg, obl_strings, G):
    _exist_op   = 'tɛ' if neg else 'bɛ'
    _exist_subj = O or S
    if obl_strings:
        return j(_exist_subj, _exist_op, *obl_strings)
    return j(_exist_subj, _exist_op)


def render_existential_localized(S, O, neg, obl_strings):
    _exist_op   = 'tɛ' if neg else 'bɛ'
    _exist_subj = O or S
    return j(_exist_subj, _exist_op, *obl_strings)
