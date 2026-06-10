"""
rules/renderers/nominal.py
Clause types : noun_phrase, noun_phrase_inh, relative_nominal
"""
from rules.core import j


def render_noun_phrase(tree, m, O, obl_strings):
    _conn_suf = (tree['main'].get('SLOTS', {}).get('conn_suffix', '')
                 or next((c.get('DEM_SUFF', '') for c in m.get('OBL_ALL', [])
                          if c.get('DEM_SUFF')), ''))
    if not _conn_suf and tree.get('_tokens'):
        _conn_suf = next((t.get('bm_suffix', '')
                          for t in tree['_tokens'] if t.get('bm_suffix')), '')
    _already     = tree.get('already_marker', '')
    _already_pos = tree.get('already_position', 'END')

    if _conn_suf and _conn_suf.strip():
        if not str(O).endswith(_conn_suf) and not any(
                str(xv).endswith(_conn_suf) for xv in obl_strings):
            return (j(_already, O, *obl_strings, _conn_suf)
                    if _already_pos == 'HEAD'
                    else j(O, _already, *obl_strings, _conn_suf))
        return (j(_already, O, *obl_strings)
                if _already_pos == 'HEAD'
                else j(O, _already, *obl_strings))

    if _already_pos == 'HEAD':
        return j(_already, O, *obl_strings)

    _tmp_obls = [obl_strings[i] for i, c in enumerate(m.get('OBL_ALL', []))
                 if i < len(obl_strings)
                 and isinstance(c, dict)
                 and c.get('local_clause_type') == 'temporal']
    _other_obls = [obl_strings[i] for i, c in enumerate(m.get('OBL_ALL', []))
                   if i < len(obl_strings)
                   and isinstance(c, dict)
                   and c.get('local_clause_type') != 'temporal']
    return j(O, *_other_obls, _already, *_tmp_obls)
