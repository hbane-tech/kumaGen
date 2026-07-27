"""
rules/rule_engine.py
RuleEngine : charge la grammaire KG et orchestre la traduction.
Les règles sont dans le KG ; build_tree applique la logique de remplissage de slots.
"""
from rules.core import _GRAMMAR_FALLBACK
from rules.build_tree import build_tree
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
            "MATCH (f:FunctionWord) WHERE f.role = 'comitative' AND f.bm IS NOT NULL "
            "RETURN f.bm AS m LIMIT 1")
        g['agent_postposition'] = self._single_marker(
            "MATCH (p:Preposition) WHERE p.role = 'agent' "
            "RETURN p.bm_marker AS m LIMIT 1")
        # Anciennement littéraux Python purs (jamais lus depuis le KG) —
        # migrés en FunctionWord 2026-07-09 : comitative_end_marker n'avait
        # AUCUNE ligne de chargement (toujours le défaut Python 'yé' dans
        # step5_obliques/__init__.py) ; locative_default_marker était un
        # littéral 'la' non conditionnel (step5_obliques/__init__.py:158).
        g['comitative_end_marker'] = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'comitative_end_marker' "
            "RETURN f.bm AS m LIMIT 1") or 'yé'
        g['locative_default_marker'] = self._single_marker(
            "MATCH (f:FunctionWord) WHERE f.role = 'locative_default_marker' "
            "RETURN f.bm AS m LIMIT 1") or 'la'
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
        g['template_placeholder_prefix'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'template_placeholder_prefix' "
            "RETURN fw.bm AS m LIMIT 1") or 'voir'
        g['template_slot_open'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'template_slot_open' "
            "RETURN fw.bm AS m LIMIT 1") or '{'

        g['question_suffix']    = self._single_marker(
            "MATCH (f:FunctionWord {lang:'bm'}) WHERE f.role = 'question_suffix' "
            "RETURN f.bm AS m LIMIT 1")
        g['question_suffix_bare'] = self._single_marker(
            "MATCH (f:FunctionWord {lang:'bm'}) WHERE f.role = 'question_suffix_bare' "
            "RETURN f.bm AS m LIMIT 1")
        g['reflexive_self_marker'] = self._single_marker(
            "MATCH (f:FunctionWord {lang:'bm'}) WHERE f.role = 'reflexive_self' "
            "RETURN f.bm AS m LIMIT 1")
        g['reciprocal_marker'] = self._single_marker(
            "MATCH (f:FunctionWord {lang:'bm'}) WHERE f.role = 'reciprocal_marker' "
            "RETURN f.bm AS m LIMIT 1")
        # Marqueur datif/oblique par classe sémantique (ex: saying→yé, communication→fɛ)
        # — lu depuis SemanticClass.dative_marker, jamais un nom de classe en dur en Python.
        g['dative_marker_by_sc'] = {
            r['name']: r['dm'] for r in self._q(
                "MATCH (s:SemanticClass) WHERE s.dative_marker IS NOT NULL "
                "RETURN s.name AS name, s.dative_marker AS dm")
        }
        g['reflexive_pron_default'] = self._single_marker(
            "MATCH (f:FunctionWord {lang:'bm'}) WHERE f.role = 'reflexive_pron_default' "
            "RETURN f.bm AS m LIMIT 1")
        g['f3_separator'] = self._single_marker(
            "MATCH (f:FunctionWord {lang:'bm'}) WHERE f.role = 'f3_separator' "
            "RETURN f.bm AS m LIMIT 1")
        g['question_marker_yala'] = self._single_marker(
            "MATCH (f:FunctionWord {lang:'bm'}) WHERE f.role = 'question_marker_yala' "
            "RETURN f.bm AS m LIMIT 1")

        # Listes sémantiques pour la coordination (migré depuis ConfigRule,
        # label retiré 2026-07-09 : ne portait plus que 2 entrées, chacune
        # rattachée thématiquement à sa propre table — CoordRule ici)
        _coord_rule = self._q("MATCH (r:CoordRule {name:'coord_nominalize_intrans_types', lang:'bm'}) RETURN r LIMIT 1")
        if _coord_rule:
            _cr = _coord_rule[0]['r']
            g['coord_intrans_types'] = set((_cr.get('intrans_types') or '').split('|'))
            g['coord_exclude_sc']    = set((_cr.get('exclude_sc') or '').split('|'))
            g['coord_intrans_sc']    = set((_cr.get('intrans_sc') or '').split('|'))
            g['coord_excl_verb_sfx'] = tuple(s for s in (_cr.get('exclude_verb_suffixes') or '').split('|') if s)
        else:
            g['coord_intrans_types'] = set()
            g['coord_exclude_sc']    = set()
            g['coord_intrans_sc']    = set()
            g['coord_excl_verb_sfx'] = ()

        g['biological_sc_name'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'semantic_class_name' AND fw.name = 'biological_sc_name' "
            "RETURN fw.bm AS m LIMIT 1") or 'biological'
        # Classes sémantiques réflexives-absolues rendues au résultatif (V+ra,
        # pas de pronom répété) au passé — ex: 'il s'est assis' → 'a sìgira'
        # (pas 'a yé a sìgi'). KG-driven, extensible sans toucher le code.
        _refl_res_rows = self._q(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'semantic_class_name' "
            "RETURN fw.bm AS m")
        g['refl_resultative_sc_names'] = ({r['m'] for r in _refl_res_rows if r.get('m')}
                                           or {'biological'})
        # Règles typographiques depuis KG FunctionWord
        _typo_from = self._q("MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'typo_fix_from' RETURN fw.bm AS bm, fw.name AS name")
        _typo_to   = self._q("MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'typo_fix_to'   RETURN fw.bm AS bm, fw.name AS name")
        g['typo_rules'] = [(f['bm'], t['bm']) for f, t in
                           zip(sorted(_typo_from, key=lambda x: x['name']),
                               sorted(_typo_to,   key=lambda x: x['name']))
                           if f.get('bm') and t.get('bm')]

        g['pronoun_3pl_coord']  = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'pronoun_3pl_coord' "
            "RETURN fw.bm AS m LIMIT 1")
        g['pronoun_3sg_default'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'pronoun_3sg_default' "
            "RETURN fw.bm AS m LIMIT 1")
        g['action_pres_suffix'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'action_pres_suffix' "
            "RETURN fw.bm AS m LIMIT 1")
        g['verbal_coord_inf']   = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'verbal_coord_inf' "
            "RETURN fw.bm AS m LIMIT 1")
        g['infinitive_marker']  = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'infinitive_marker' "
            "RETURN fw.bm AS m LIMIT 1")
        g['verbal_coordinator'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'verbal_coordinator' "
            "RETURN fw.bm AS m LIMIT 1")
        # Ensemble autonomous_sc depuis KG (advcl). Bug trouvé 2026-07-09 : le
        # label interrogé était TransformRule (qui ne contient PAS ce name) →
        # _auto_row toujours vide, g['autonomous_sc'] toujours set() → aucune
        # classe sémantique ne rendait jamais en verbe nu dans les clauses
        # purposive/privative (cf. rules/steps/step5_obliques/advcl.py:43) —
        # ex: "il travaille pour aller au marché" → "wáli kɛ" (nominalisé,
        # faux) au lieu de "wáli" nu (aller = motion = classe autonome).
        # Migré depuis ConfigRule (label retiré 2026-07-09) vers Advcl_Rule,
        # qui porte déjà les autres règles de clause adverbiale.
        _auto_row = self._q("MATCH (r:Advcl_Rule {name:'advcl_autonomous_sc', lang:'bm'}) RETURN r.autonomous_sc AS sc LIMIT 1")
        g['autonomous_sc'] = set((_auto_row[0]['sc'] or '').split('|')) if _auto_row and _auto_row[0].get('sc') else set()

        g['locative_suffix'] = self._single_marker(
            "MATCH (p:Preposition) WHERE p.role = 'locative' "
            "RETURN p.bm_marker AS m LIMIT 1")

        # Surfaces relatives et expletif FR depuis KG
        _rel_rows = self._q("MATCH (fw:FunctionWord) WHERE fw.role = 'relative_pron_surface' RETURN fw.bm AS bm")
        g['relative_pron_surfaces'] = {r['bm'].lower() for r in _rel_rows if r.get('bm')}
        _expl_rows = self._q("MATCH (fw:FunctionWord) WHERE fw.role = 'expletive_fr_surface' RETURN fw.bm AS bm")
        g['expletive_fr_surfaces'] = {r['bm'].lower() for r in _expl_rows if r.get('bm')}
        # Lemmes impersonnels depuis ImpersonalRule KG
        _imp_rows = self._q(
            "MATCH (r:ImpersonalRule {lang:'bm'}) "
            "RETURN r.trigger_lemma AS l, r.also_triggers AS alt, r.mark_pattern AS mp")
        g['impersonal_trigger_lemmas'] = {r['l'].lower() for r in _imp_rows if r.get('l')} | \
            {r['alt'].lower() for r in _imp_rows if r.get('alt')}
        # Regroupement par mark_pattern (ex: 'expl:comp_se', 'ccomp', '') — l'exigence
        # structurelle (il+réfléchi / il+ccomp-ou-sujet-réel / aucune) est déjà une
        # propriété du nœud KG ; le dispatch Python se fait sur ce champ, pas sur le
        # lemme littéral (décision 2026-07-12, audit hardcode-KG).
        g['impersonal_lemmas_by_pattern'] = {}
        for r in _imp_rows:
            _mp = r.get('mp') or ''
            for _l in (r.get('l'), r.get('alt')):
                if _l:
                    g['impersonal_lemmas_by_pattern'].setdefault(_mp, set()).add(_l.lower())

        g['plural_noun_suffix']    = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'plural_noun_suffix' "
            "RETURN fw.bm AS m LIMIT 1")
        g['adj_epith_suffix']      = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'adj_epith_suffix' "
            "RETURN fw.bm AS m LIMIT 1")
        g['already_pres_marker']   = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'already_pres_marker' "
            "RETURN fw.bm AS m LIMIT 1")
        g['already_past_marker']   = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'already_past_marker' "
            "RETURN fw.bm AS m LIMIT 1")
        g['already_neg_marker']    = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'already_neg_marker' "
            "RETURN fw.bm AS m LIMIT 1")
        g['pron_1sg']              = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'pron_1sg' "
            "RETURN fw.bm AS m LIMIT 1")
        g['avoir_mal_fr']          = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'avoir_mal_fr' "
            "RETURN fw.bm AS m LIMIT 1")
        g['plural_fr_ending']      = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'plural_fr_ending' "
            "RETURN fw.bm AS m LIMIT 1")
        g['interrogative_who']     = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'interrogative_who' "
            "RETURN fw.bm AS m LIMIT 1")
        g['comparative_particle'] = self._single_marker(
            "MATCH (f:FunctionWord {lang:'bm'}) WHERE f.role = 'comparative_particle' "
            "RETURN f.bm AS m LIMIT 1")
        g['comparative_ref_marker'] = self._single_marker(
            "MATCH (f:FunctionWord {lang:'bm'}) WHERE f.role = 'comparative_ref' "
            "RETURN f.bm AS m LIMIT 1")
        g['coord_verb_marker'] = self._single_marker(
            "MATCH (f:FunctionWord {lang:'bm'}) WHERE f.role = 'coord_verb' "
            "RETURN f.bm AS m LIMIT 1")
        g['coord_action_suffix'] = self._single_marker(
            "MATCH (f:FunctionWord {lang:'bm'}) WHERE f.role = 'coord_action_suffix' "
            "RETURN f.bm AS m LIMIT 1")
        g['privative_verb_marker'] = self._single_marker(
            "MATCH (f:FunctionWord {lang:'bm'}) WHERE f.role = 'privative_verb_marker' "
            "RETURN f.bm AS m LIMIT 1")

        # Charger TOUS les FunctionWord bambara comme liste accessible par rôle
        _fw_bm_rows = self._q(
            "MATCH (fw:FunctionWord {lang:'bm'}) RETURN fw.role AS role, fw.bm AS bm, fw.name AS name")
        g['function_words'] = [{'role': r.get('role', ''), 'bm': r.get('bm', ''), 'name': r.get('name', '')}
                                for r in _fw_bm_rows if r.get('role') and r.get('bm')]

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
            print("     TAM: KG vide — valeurs TAM non disponibles.")

        # Pronoms et FunctionWords pour content_question (résolution pronoms inversés)
        pron_rows = self._q(
            "MATCH (n:Pronoun) RETURN n.surface AS s, n.bm AS bm, "
            "n.singular AS sing, n.lang AS lang")
        g['pron'] = {r['s'].lower() for r in pron_rows if r.get('s')}
        g['sing'] = {r['s'].lower() for r in pron_rows
                     if r.get('s') and r.get('sing')}
        fw_rows = self._q(
            "MATCH (f:FunctionWord) RETURN f.surface AS s, f.lang AS l, "
            "f.bm AS bm, f.role AS r, f.requires_partner AS req")
        g['funcs'] = {(r['s'].lower(), r.get('l', 'fr')): {
                          'bm': r.get('bm', ''),
                          'role': r.get('r', 'content'),
                          'requires_partner': bool(r.get('req'))}
                      for r in fw_rows if r.get('s')}
        
        # ── Phase 2 : Règles morphologiques ─────────────────────────────────────
        morpho_rows = self._q(
            "MATCH (m:MorphoRule) WHERE m.lang = 'bm' "
            "RETURN m.name AS name, m.suffix AS suffix, "
            "m.suffix_default AS suffix_default, "
            "m.suffix_after_n AS suffix_after_n, "
            "m.suffix_after_vowel AS suffix_after_vowel, "
            "m.suffix_after_o_u_o AS suffix_after_o_u_o, "
            "m.trigger_n AS trigger_n, m.trigger_vowel AS trigger_vowel, "
            "m.pos_support AS pos_support, m.neg_support AS neg_support, "
            "m.hab_prefix AS hab_prefix, "
            "m.support AS support, m.condition AS condition")
        g['morpho_rules'] = {r['name']: r for r in morpho_rows if r.get('name')}

        # Suffixe adj_man depuis MorphoRule
        _adj_rule = g['morpho_rules'].get('adjective_epithet', {})
        g['adj_epithet_suffix'] = _adj_rule.get('suffix', '')
        if g['adj_epithet_suffix']:
            import rules.core as _core_mod
            _core_mod._ADJ_EPITH_SUFFIX = g['adj_epithet_suffix']

        # Raccourcis — valeurs lues depuis KG, exposées pour les steps
        _res = g['morpho_rules'].get('resultative', {})
        g['resultative_trigger_n']     = _res.get('trigger_n', '')
        g['resultative_trigger_vowel'] = _res.get('trigger_vowel', '')
        g['resultative_suffix_n']      = _res.get('suffix_after_n', '')
        g['resultative_suffix_vowel']  = _res.get('suffix_after_vowel', '')
        g['resultative_suffix']        = _res.get('suffix_default', '')
        _stat = g['morpho_rules'].get('statif', {})
        g['statif_pos_marker'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'statif_pos' "
            "RETURN fw.bm AS m LIMIT 1")
        g['statif_neg_marker'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'statif_neg' "
            "RETURN fw.bm AS m LIMIT 1")
        g['statif_suffix']             = _stat.get('suffix', '')
        g['statif_pos_support']        = _stat.get('pos_support', '')
        g['statif_neg_support']        = _stat.get('neg_support', '')
        g['statif_hab_prefix']         = _stat.get('hab_prefix', '')
        _nom = g['morpho_rules'].get('nominalization_action', {})
        g['nominalization_suffix']     = _nom.get('suffix', '')
        g['nominalization_support']    = _nom.get('support', '')
        g['nominalization_verb_suffix'] = _nom.get('suffix', '')  # alias court
        g['nominalization_verb_suffix_alt'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'nominalization_verb_suffix_alt' "
            "RETURN fw.bm AS m LIMIT 1")
        g['possession_past_verb'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'possession_past_verb' "
            "RETURN fw.bm AS m LIMIT 1")
        g['refl_serial_postpos'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'refl_serial_postpos' "
            "RETURN fw.bm AS m LIMIT 1")
        _refl_emph = self._q("MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'refl_emphasis_surface' RETURN fw.bm AS bm")
        g['refl_emphasis_surfaces'] = {r['bm'].lower() for r in _refl_emph if r.get('bm')}
        g['participial_to_suffix'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'participial_to_suffix' "
            "RETURN fw.bm AS m LIMIT 1")
        g['meteo_motion_default'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'meteo_motion_default' "
            "RETURN fw.bm AS m LIMIT 1")
        g['privative_pron_pred'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'privative_pron_pred' "
            "RETURN fw.bm AS m LIMIT 1")
        g['privative_noun_pred'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'privative_noun_pred' "
            "RETURN fw.bm AS m LIMIT 1")
        _tmp_pfx = self._q("MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'temporal_prefix_bm' RETURN fw.bm AS bm")
        g['temporal_prefix_markers'] = {r['bm'] for r in _tmp_pfx if r.get('bm')}
        _res_vowel = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'resultative_vowel_trigger' "
            "RETURN fw.bm AS m LIMIT 1")
        g['resultative_vowel_trigger'] = set((_res_vowel or '').split('|')) if _res_vowel else set()
        # INTRANS_SC depuis KG
        _intrans_row = self._q("MATCH (r:TransformRule {name:'intrans_sc', lang:'bm'}) RETURN r.intrans_sc AS sc LIMIT 1")
        if _intrans_row and _intrans_row[0].get('sc'):
            import rules.core as _core_mod2
            _core_mod2.INTRANS_SC = frozenset(_intrans_row[0]['sc'].split('|'))
            g['intrans_sc'] = _core_mod2.INTRANS_SC
        g['possessive_pron_marker'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'possessive_pron_marker' "
            "RETURN fw.bm AS m LIMIT 1")
        g['avoir_acquisition_verb'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'avoir_acquisition_verb' "
            "RETURN fw.bm AS m LIMIT 1")
        g['obligation_pres_marker'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'obligation_pres_marker' "
            "RETURN fw.bm AS m LIMIT 1")
        g['obligation_neg_marker'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'obligation_neg_marker' "
            "RETURN fw.bm AS m LIMIT 1")
        g['imperative_neg_marker'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'imperative_neg_marker' "
            "RETURN fw.bm AS m LIMIT 1")
        g['eventuality_marker'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'eventuality_marker' "
            "RETURN fw.bm AS m LIMIT 1")
        g['passive_tam_marker'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'passive_tam_marker' "
            "RETURN fw.bm AS m LIMIT 1")
        g['refl_past_tam_marker'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'refl_past_tam_marker' "
            "RETURN fw.bm AS m LIMIT 1")
        g['relative_marker_plural'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'relative_marker_plural' "
            "RETURN fw.bm AS m LIMIT 1")
        g['demonstrative_subject_np_marker'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'demonstrative_subject_np_marker' "
            "RETURN fw.bm AS m LIMIT 1")
        g['venir_lemmas'] = {r['m'] for r in self._q(
            "MATCH (fw:FunctionWord {lang:'fr'}) WHERE fw.role = 'venir_lemma' "
            "RETURN fw.bm AS m") if r.get('m')}
        g['aller_lemmas'] = {r['m'] for r in self._q(
            "MATCH (fw:FunctionWord {lang:'fr'}) WHERE fw.role = 'aller_lemma' "
            "RETURN fw.bm AS m") if r.get('m')}
        g['progressive_periphrasis_lemmas'] = {r['m'] for r in self._q(
            "MATCH (fw:FunctionWord {lang:'fr'}) WHERE fw.role = 'progressive_periphrasis_lemma' "
            "RETURN fw.bm AS m") if r.get('m')}
        g['expletive_demonstrative_pronoun'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'expletive_demonstrative_pronoun' "
            "RETURN fw.bm AS m LIMIT 1")
        g['eventuality_trigger_lemmas'] = {r['m'] for r in self._q(
            "MATCH (fw:FunctionWord {lang:'fr'}) WHERE fw.role = 'eventuality_trigger_lemma' "
            "RETURN fw.bm AS m") if r.get('m')}
        g['expletive_demonstrative_surfaces'] = {r['m'] for r in self._q(
            "MATCH (fw:FunctionWord {lang:'fr'}) WHERE fw.role = 'expletive_demonstrative_surface' "
            "RETURN fw.bm AS m") if r.get('m')}
        g['complementizer_que_surfaces'] = {r['m'] for r in self._q(
            "MATCH (fw:FunctionWord {lang:'fr'}) WHERE fw.role = 'complementizer_que_surface' "
            "RETURN fw.bm AS m") if r.get('m')}
        g['reflexive_clitic_surfaces'] = {r['m'] for r in self._q(
            "MATCH (fw:FunctionWord {lang:'fr'}) WHERE fw.role = 'reflexive_clitic_surface' "
            "RETURN fw.bm AS m") if r.get('m')}
        g['age_noun_lemmas'] = {r['m'] for r in self._q(
            "MATCH (fw:FunctionWord {lang:'fr'}) WHERE fw.role = 'age_noun_lemma' "
            "RETURN fw.bm AS m") if r.get('m')}
        g['source_de_surfaces'] = {r['m'] for r in self._q(
            "MATCH (fw:FunctionWord {lang:'fr'}) WHERE fw.role = 'source_de_surface' "
            "RETURN fw.bm AS m") if r.get('m')}
        g['imperative_1pl_fr_suffix'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'fr'}) WHERE fw.role = 'imperative_1pl_fr_suffix' "
            "RETURN fw.bm AS m LIMIT 1")
        g['imperative_2pl_fr_suffix'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'fr'}) WHERE fw.role = 'imperative_2pl_fr_suffix' "
            "RETURN fw.bm AS m LIMIT 1")
        g['copula_default_lemma'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'fr'}) WHERE fw.role = 'copula_default_lemma' "
            "RETURN fw.bm AS m LIMIT 1")
        g['saying_default_lemma'] = self._single_marker(
            "MATCH (fw:FunctionWord {lang:'fr'}) WHERE fw.role = 'saying_default_lemma' "
            "RETURN fw.bm AS m LIMIT 1")
        g['intensifier_bm_markers'] = {
            r['m'] for r in self._q(
                "MATCH (fw:FunctionWord {lang:'bm'}) WHERE fw.role = 'intensifier_bm_marker' "
                "RETURN fw.bm AS m") if r.get('m')}
        g['intensifier_trigger_lemmas'] = {
            r['s'].lower(): r['m'] for r in self._q(
                "MATCH (fw:FunctionWord {lang:'fr'}) WHERE fw.role = 'intensifier_trigger_lemma' "
                "RETURN fw.surface AS s, fw.bm AS m") if r.get('s') and r.get('m')}
        g['support_verb_adj_lemmas'] = {
            r['m'] for r in self._q(
                "MATCH (fw:FunctionWord {lang:'fr'}) WHERE fw.role = 'support_verb_adj_lemma' "
                "RETURN fw.bm AS m") if r.get('m')}
        # _AVOIR_LEMMAS depuis KG
        _avoir_rows = self._q("MATCH (fw:FunctionWord) WHERE fw.role = 'avoir_lemma' RETURN fw.bm AS bm")
        if _avoir_rows:
            import rules.core as _core_mod3
            _core_mod3._AVOIR_LEMMAS = {r['bm'] for r in _avoir_rows if r.get('bm')}
            g['avoir_lemmas'] = _core_mod3._AVOIR_LEMMAS
        # Suffixes exclusion adj_man depuis MorphoRule
        _adj_excl_rule = g.get('morpho_rules', {}).get('adjective_epithet', {})
        g['adj_exclude_suffixes'] = tuple(s for s in (_adj_excl_rule.get('exclude_suffixes', '') or '').split('|') if s)
        if g['adj_exclude_suffixes']:
            import rules.core as _core_excl
            _core_excl._ADJ_EXCLUDE_SUFFIXES = g['adj_exclude_suffixes']


        # ── Phase 3 : Templates de clause ────────────────────────────────────
        tpl_rows = self._q(
            "MATCH (t:ClauseTemplate) WHERE t.lang = 'bm' "
            "RETURN t.clause_type AS ct, t.template AS tpl")
        g['clause_templates'] = {r['ct']: r['tpl'] for r in tpl_rows if r.get('ct')}

        # ── Phase 4 : Comportements sémantiques ──────────────────────────────
        # Chargés via traversée SemanticClass -[HAS_BEHAVIOR]-> SemanticBehavior
        # Garantit la cohérence : un seul endroit pour ajouter un comportement (KG)
        beh_rows = self._q("""
            MATCH (sc:SemanticClass)-[:HAS_BEHAVIOR]->(sb:SemanticBehavior {lang:'bm'})
            RETURN sc.name AS sc,
                   sb.behavior_type AS bt,
                   sb.behavior_value AS bv,
                   sb.extra AS extra
            UNION
            MATCH (sb:SemanticBehavior {lang:'bm'})
            WHERE NOT EXISTS { (:SemanticClass)-[:HAS_BEHAVIOR]->(sb) }
            RETURN sb.sem_class AS sc, sb.behavior_type AS bt,
                   sb.behavior_value AS bv, sb.extra AS extra
        """)
        g['semantic_behaviors'] = {}
        for r in beh_rows:
            if r.get('sc') and r.get('bt'):
                g['semantic_behaviors'].setdefault(r['sc'], {})[r['bt']] = {
                    'value': r.get('bv'), 'extra': r.get('extra')}

        # ── Toutes les règles KG : zéro hardcode Python ──────────────────────────
        def _load_rule_nodes(label):
            rows = self._q(f"MATCH (r:{label} {{lang:'bm'}}) RETURN r")
            return [r['r'] for r in rows] if rows else []

        g['kg_pattern_rules']    = _load_rule_nodes('PatternRule')
        g['kg_slot_fill_rules']  = _load_rule_nodes('SlotFillRule')
        g['kg_transform_rules']  = _load_rule_nodes('TransformRule')
        g['kg_nominal_rules']    = _load_rule_nodes('NominalChainRule')
        g['kg_graph_walk_rules'] = _load_rule_nodes('GraphWalkRule')
        g['kg_impersonal_rules'] = _load_rule_nodes('ImpersonalRule')

        # ── Graphe de règles complet : FeaturePattern→ConstructionRule→ClauseTemplate
        rule_rows = self._q("""
            MATCH (fp:FeaturePattern)-[:TRIGGERS]->(cr:ConstructionRule)
            OPTIONAL MATCH (cr)-[:USES_TEMPLATE]->(ct:ClauseTemplate)
            OPTIONAL MATCH (cr)-[:APPLIES_MORPHO]->(mr:MorphoRule)
            RETURN fp.feature AS feature, fp.value AS value,
                   fp.context AS context, fp.neg AS neg,
                   cr.name AS rule_name, cr.priority AS priority,
                   cr.refl_treatment AS refl_treatment,
                   cr.motion_verb AS motion_verb,
                   cr.prefix AS prefix, cr.tam_value AS tam_value,
                   cr.preserves_clause_type AS preserves_ct,
                   ct.template AS template, ct.word_order AS word_order,
                   ct.pos_form AS pos_form, ct.neg_form AS neg_form,
                   ct.hab_form AS hab_form, ct.past_form AS past_form,
                   mr.name AS morpho_name,
                   mr.suffix_default AS morpho_sfx,
                   mr.suffix_after_n AS morpho_sfx_n,
                   mr.suffix_after_o_u_o AS morpho_sfx_ou,
                   mr.pos_support AS morpho_pos_sup,
                   mr.neg_support AS morpho_neg_sup
            ORDER BY cr.priority DESC
        """)
        # Index par (feature, value) → règle prioritaire
        g['kg_rules'] = {}
        for r in rule_rows:
            key = (r.get('feature', ''), str(r.get('value', '')))
            ctx = r.get('context') or ''
            full_key = (key[0], key[1], ctx)
            if full_key not in g['kg_rules']:
                g['kg_rules'][full_key] = r
            # Aussi indexer sans context pour lookup générique
            if key not in g['kg_rules']:
                g['kg_rules'][key] = r

        # Index rapide clause_type → template (sans requête Neo4j au moment du rendu)
        g['clause_type_templates'] = {
            r.get('value'): r.get('template', '')
            for r in rule_rows
            if r.get('feature') == 'clause_type' and r.get('template')
        }

        print(f"   Grammar loaded from KG — "
              f"loc={len(g['locative_markers'])} "
              f"tmp={len(g['temporal_markers'])} "
              f"gen='{g['genitive_marker']}' "
              f"com='{g['comitative_marker']}' "
              f"tam='{g['tam_default']}' "
              f"morpho={len(g['morpho_rules'])} "
              f"tpl={len(g['clause_templates'])} "
              f"behaviors={len(g['semantic_behaviors'])} "
              f"rules={len(g['kg_rules'])}")

    def apply(self, tokens_or_tree):
        if not tokens_or_tree:
            self._last_tree = {}
            return ''
        if isinstance(tokens_or_tree, dict):
            self._last_tree = tokens_or_tree
            return tree_to_bambara(tokens_or_tree, grammar=self.grammar)
        tree = build_tree(tokens_or_tree, grammar=self.grammar)
        self._last_tree = tree
        if tree.get('local_clause_type') in ('relative_nominal', 'impersonal'):
            return tree.get('final_string', '')
        return tree_to_bambara(tree, grammar=self.grammar)