"""
rules/rule_engine.py
RuleEngine : charge la grammaire KG et orchestre build_tree + tree_to_bambara.
"""
from rules.core import _GRAMMAR_FALLBACK
from rules.build_tree import build_tree
# from rules.tree_to_bambara import tree_to_bambara
from rules.renderers import tree_to_bambara

class RuleEngine:
    def __init__(self, db=None):
        self.db      = db
        self.grammar = dict(_GRAMMAR_FALLBACK)
        if db:
            try:
                from pipeline.proposition_parser import load_connector_bm
                load_connector_bm(db)
            except Exception:
                pass
            self._load_grammar()

    def _q(self, cypher, params=None):
        try:
            return self.db.query(cypher, params or {}) or []
        except Exception:
            return []

    def _marker_set(self, role: str) -> set:
        rows = self._q(
            "MATCH (p:Preposition) WHERE p.role = $role RETURN p.bm_marker AS m",
            {'role': role})
        return {r['m'] for r in rows if r.get('m')}

    def _single_marker(self, cypher: str, params=None) -> str:
        rows = self._q(cypher, params)
        return rows[0].get('m', '') if rows else ''

    def _load_grammar(self):
        g = self.grammar

        g['action_noun_map'] = {}
        action_rows = self._q(
            "MATCH (v:Verb) WHERE v.action_noun IS NOT NULL "
            "RETURN v.bm AS bm, v.action_noun AS noun")
        g['action_noun_map'] = {r['bm']: r['noun']
                                for r in action_rows if r.get('bm')}
        
        # Négation depuis KG
        neg_rows = self._q(
            "MATCH (f:FunctionWord) WHERE f.role = 'negation' AND f.lang = 'fr' "
            "RETURN f.surface AS s")
        g['neg_surfaces'] = {r['s'].lower() for r in neg_rows if r.get('s')}
        g['neg_surfaces'] |= {r['s'].lower().rstrip("'").rstrip('\u2019')
                               for r in neg_rows if r.get('s')}

        # Marqueur privatif pour pronoms
        g['privative_pron_marker'] = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'privative_pron_marker' "
            "RETURN f.bm AS m LIMIT 1") or 'kɔ'

        # Rôles depuis KG
        exp_rows = self._q(
            "MATCH (f:FunctionWord) WHERE f.role = 'expletive' "
            "RETURN DISTINCT f.role AS r")
        g['expletive_roles'] = {r['r'] for r in exp_rows if r.get('r')} or {'expletive'}

        clit_rows = self._q(
            "MATCH (f:FunctionWord) WHERE f.role = 'clitic' "
            "RETURN DISTINCT f.role AS r")
        g['clitic_roles'] = {r['r'] for r in clit_rows if r.get('r')} or {'clitic'}

        rel_rows = self._q(
            "MATCH (f:FunctionWord) WHERE f.role = 'relative' "
            "RETURN DISTINCT f.role AS r")
        g['relative_roles'] = {r['r'] for r in rel_rows if r.get('r')} or {'relative'}

        g['privative_markers']     = self._marker_set('privative')
        g['locative_markers']      = self._marker_set('locative')
        g['temporal_markers']      = self._marker_set('temporal')
        g['temporal_suffix_markers'] = {
            r['m'] for r in self._q(
                "MATCH (p:Preposition) WHERE p.is_suffix = true "
                "RETURN p.bm_marker AS m") if r.get('m')}
        g['focus_marker'] = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'focus_marker' AND f.lang = 'bm' "
            "RETURN f.bm AS m LIMIT 1") or 'de'

        temporal_rows = self._q(
            "MATCH (n) WHERE n.role IS NOT NULL AND n.semantic_class = 'temporal' "
            "RETURN DISTINCT n.role AS r")
        g['temporal_roles'] = {r['r'] for r in temporal_rows if r.get('r')}
        g['temporal_roles'].add('temporal')

        quantifier_rows = self._q(
            "MATCH (f:FunctionWord) WHERE f.role = 'quantifier' AND f.lang = 'fr' "
            "RETURN f.surface AS surface, f.bm AS bm")
        g['quantifier_words'] = {r['surface']: r['bm']
                                  for r in quantifier_rows if r.get('surface')}

        distrib_rows = self._q(
            "MATCH (f:FunctionWord) WHERE f.role IN ['distributive_each','distributive_one'] "
            "AND f.lang = 'fr' RETURN f.surface AS surface, f.bm AS bm, f.role AS role")
        g['distributive_each'] = {r['surface']: r['bm']
                                   for r in distrib_rows if r.get('role') == 'distributive_each'}
        g['distributive_one']  = {r['surface']: r['bm']
                                   for r in distrib_rows if r.get('role') == 'distributive_one'}

        # Noms relationnels/inalienables → juxtaposition directe sans 'ka'
        # (parenté, rôles, parties du corps)
        _rel_rows = self._q(
            "MATCH (s:Sense) WHERE s.relational = true OR s.inalienable = true "
            "RETURN s.bm AS bm")
        # L'inaliénabilité est détectée par le LLM par token (is_relational via
        # _detect_relational_noun) ; le KG (Sense.relational) sert de cache.
        # Zéro mot en dur ici.
        g['relational_bms'] = {r['bm'] for r in _rel_rows if r.get('bm')}

        g['genitive_marker']   = self._single_marker(
            "MATCH (p:Preposition) WHERE p.role = 'genitive' "
            "RETURN p.bm_marker AS m LIMIT 1")
        g['comitative_marker'] = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'comitative' "
            "RETURN f.bm AS m LIMIT 1")
        g['agent_postposition'] = self._single_marker(
            "MATCH (p:Preposition) WHERE p.role = 'agent' "
            "RETURN p.bm_marker AS m LIMIT 1")
        g['dative_marker']     = self._single_marker(
            "MATCH (p:Preposition) WHERE p.role = 'dative' "
            "RETURN p.bm_marker AS m LIMIT 1") or 'ma'
        g['purposive_marker']  = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'purposive' "
            "RETURN f.bm AS m LIMIT 1")
        g['reported_intro']    = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'reported_intro' "
            "RETURN f.bm AS m LIMIT 1")
        g['relative_marker']   = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'relative_marker' "
            "RETURN f.bm AS m LIMIT 1") or 'mìn'
        g['equative_marker']   = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'equative_marker' "
            "RETURN f.bm AS m LIMIT 1") or 'yé'
        g['possession_material_marker'] = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'possession_material' "
            "RETURN f.bm AS m LIMIT 1") or 'bóló'
        g['possession_abstract_marker'] = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'possession_abstract' "
            "RETURN f.bm AS m LIMIT 1") or 'fɛ'
        g['deictique_marker']  = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'deictique' "
            "RETURN f.bm AS m LIMIT 1") or 'félé'
        
        g['existence_loc_marker'] = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'existence_loc' "
            "RETURN f.bm AS m LIMIT 1") or 'yàn'
        
        g['demonstrative_surfaces'] = {
            r['s'].lower() for r in self._q(
                "MATCH (f:FunctionWord) WHERE f.role = 'demonstrative' "
                "RETURN f.surface AS s") if r.get('s')}
        tam_rows = self._q(
            "MATCH (t:TamConfig) RETURN t.tense AS tense, t.neg AS neg, "
            "t.bm AS bm, t.aspect AS aspect, t.ctx AS ctx")
        if tam_rows:
            g['tam_table'] = {
                (r['tense'], r['neg']): r['bm']
                for r in tam_rows
                if r.get('tense') is not None
                and r.get('neg') is not None
                and r.get('bm')}
            g['tam_default']      = g['tam_table'].get(('pres', False), '')
            g['tam_future']       = g['tam_table'].get(('fut',  False), '')
            g['progressive_tams'] = {r['bm'] for r in tam_rows
                                     if r.get('aspect') == 'progressive' and r.get('bm')}
            print(f"     TAM: {len(g['tam_table'])} entrées KG chargées.")
        else:
            g['tam_table']        = {}
            g['tam_default']      = ''
            g['tam_future']       = ''
            g['progressive_tams'] = set()
            print("     TAM: KG vide — fallback _TAM_HARDCODED actif.")

        # Pronoms et FunctionWords pour content_question (résolution pronoms inversés)
        pron_rows = self._q(
            "MATCH (n:Pronoun) RETURN n.surface AS s, n.bm AS bm, "
            "n.singular AS sing, n.lang AS lang")
        g['pron'] = {r['s'].lower() for r in pron_rows if r.get('s')}
        g['sing'] = {r['s'].lower() for r in pron_rows
                     if r.get('s') and r.get('sing')}
        fw_rows = self._q(
            "MATCH (f:FunctionWord) RETURN f.surface AS s, f.lang AS l, "
            "f.bm AS bm, f.role AS r")
        g['funcs'] = {(r['s'].lower(), r.get('l', 'fr')): {
                          'bm': r.get('bm', ''),
                          'role': r.get('r', 'content')}
                      for r in fw_rows if r.get('s')}
        
        print(f"  ✅ Grammar loaded from KG — "
              f"loc={len(g['locative_markers'])} "
              f"tmp={len(g['temporal_markers'])} "
              f"gen='{g['genitive_marker']}' "
              f"com='{g['comitative_marker']}' "
              f"tam='{g['tam_default']}'")

    def apply(self, tokens_or_tree, frame):
        if not tokens_or_tree:
            self._last_tree = {}
            return ''
        if isinstance(tokens_or_tree, dict):
            self._last_tree = tokens_or_tree
            return tree_to_bambara(tokens_or_tree, grammar=self.grammar)
        tree = build_tree(tokens_or_tree, grammar=self.grammar)
        self._last_tree = tree   # exposé pour l'évaluation
        if tree.get('local_clause_type') == 'relative_nominal':
            return tree.get('final_string', '')
        return tree_to_bambara(tree, grammar=self.grammar)