"""
kg/migrate_grammar_to_kg.py
Migration progressive des règles hardcodées vers le KG Neo4j.

Phase 1 : Système TAM  → nœuds TamConfig
Phase 2 : Morphologie  → nœuds MorphoRule
Phase 3 : Templates    → nœuds ClauseTemplate
Phase 4 : Comportements sémantiques → nœuds SemanticBehavior

Usage :
    python -m kg.migrate_grammar_to_kg [--phase 1] [--lang bm]
"""

import argparse
from kg.neo4j_client import Neo4jClient


def migrate_tam(db, lang='bm'):
    """Phase 1 : Système TAM → TamConfig nodes.
    Remplace _TAM_HARDCODED dans rules/core.py.
    Le code _resolve_tam() lit déjà ces nœuds si présents.
    """
    print(f"\n[Phase 1] TAM system pour lang={lang!r}")

    tam_rules = [
        # (tense, neg, bm, description)
        ('pres',  False, 'bɛ',     'présent positif'),
        ('pres',  True,  'tɛ',     'présent négatif'),
        ('past',  False, 'yé',     'passé simple positif'),
        ('past',  True,  'ma',     'passé simple négatif'),
        ('fut',   False, 'bɛ́nà',  'futur positif'),
        ('fut',   True,  'tɛ́nà',  'futur négatif'),
        ('cond',  False, 'mána',  'conditionnel positif'),
        ('cond',  True,  'tɛ́na',  'conditionnel négatif'),
        ('hab',   False, 'tùn bɛ', 'imparfait/habituel positif'),
        ('hab',   True,  'tùn tɛ', 'imparfait/habituel négatif'),
        ('plup',  False, 'tùn yé', 'plus-que-parfait positif'),
        ('plup',  True,  'tùn ma', 'plus-que-parfait négatif'),
        ('prog',  False, 'bɛ́kà',  'progressif positif'),
        ('prog',  True,  'tɛ́kà',  'progressif négatif'),
        ('imp',   False, '',        'impératif (pas de TAM)'),
        ('sub',   False, '',        'subjonctif (pas de TAM séparé)'),
        ('qualitative_pres', True, 'mán', 'qualitatif présent négatif'),
    ]

    count = 0
    for tense, neg, bm, desc in tam_rules:
        db.query("""
        MERGE (t:TamConfig {tense: $tense, neg: $neg, lang: $lang})
        SET t.bm = $bm, t.description = $desc
        """, {'tense': tense, 'neg': neg, 'bm': bm, 'desc': desc, 'lang': lang})
        count += 1

    print(f"   {count} règles TAM créées dans le KG")
    print(f"     Le fallback _TAM_HARDCODED reste actif si KG vide.")


def migrate_morpho_rules(db, lang='bm'):
    """Phase 2 : Règles morphologiques → MorphoRule nodes."""
    print(f"\n[Phase 2] Règles morphologiques pour lang={lang!r}")

    # Suffixe résultatif (passé intransitif)
    db.query("""
    MERGE (m:MorphoRule {lang: $lang, name: 'resultative'})
    SET m.condition = 'past+intransitive',
        m.suffix_after_n = 'na',
        m.suffix_after_o_u_o = 'la',
        m.suffix_default = 'ra',
        m.description = 'Forme résultative intransitif : V+ra/na/la'
    """, {'lang': lang})

    # Construction statif (-len dòn)
    db.query("""
    MERGE (m:MorphoRule {lang: $lang, name: 'statif'})
    SET m.suffix = 'len',
        m.pos_support = 'dòn',
        m.neg_support = 'tɛ',
        m.hab_prefix = 'tùn',
        m.description = 'État statif : V+len dòn (pos) / V+len tɛ (neg)'
    """, {'lang': lang})

    # Nominalisation d'action (V+li kɛ)
    db.query("""
    MERGE (m:MorphoRule {lang: $lang, name: 'nominalization_action'})
    SET m.suffix = 'li',
        m.support = 'kɛ',
        m.condition = 'action_verb+no_object+present',
        m.description = 'Nominalisation action : V+li kɛ (présent sans COD)'
    """, {'lang': lang})

    # Pluriel nominal
    db.query("""
    MERGE (m:MorphoRule {lang: $lang, name: 'plural_noun'})
    SET m.suffix = 'w',
        m.condition = 'noun+plural',
        m.description = 'Pluriel nominal : N+w'
    """, {'lang': lang})

    print(f"   Règles morphologiques créées dans le KG")


def migrate_clause_templates(db, lang='bm'):
    """Phase 3 : Templates de construction → ClauseTemplate nodes."""
    print(f"\n[Phase 3] Templates de clause pour lang={lang!r}")

    templates = [
        ('simple',         '{S} {TAM} {O} {V} {ADV}',      'Clause déclarative standard (SOV)'),
        ('equative_pos',   '{S} yé {O} yé',                 'Équatif positif'),
        ('equative_neg',   '{S} tɛ {O} yé',                 'Équatif négatif'),
        ('optative',       '{S} ka {O} {V}',                 'Optatif/jussif'),
        ('prohibitive',    'kàna {O} {V}',                   'Prohibitif'),
        ('passive_statif', '{S} {V}len {dòn|tɛ} {ADV}',    'Passif = statif (-len dòn)'),
        ('relative_obj',   'mìn {TAM} {O} {V}',             'Proposition relative objet'),
        ('relative_subj',  '{S} mìn {TAM} {O} {V}',        'Proposition relative sujet'),
        ('serial_motion',  '{S} {TAM} {V1} {O} {V2}',      'Sérielle mouvement+but'),
        ('content_question', '{S} {TAM} {O} {V} {ADV} ?',  'Question de contenu'),
        ('polar_question', '{S} {TAM} {O} {V} wà ?',        'Question polaire'),
        ('meteorological', '{PHENOMENON} {TAM} {MOTION}',   'Météorologique (il pleut)'),
    ]

    count = 0
    for ct, template, desc in templates:
        db.query("""
        MERGE (t:ClauseTemplate {lang: $lang, clause_type: $ct})
        SET t.template = $template, t.description = $desc
        """, {'lang': lang, 'ct': ct, 'template': template, 'desc': desc})
        count += 1

    print(f"   {count} templates de clause créés dans le KG")


def migrate_semantic_behaviors(db, lang='bm'):
    """Phase 4 : Comportements par classe sémantique → SemanticBehavior nodes."""
    print(f"\n[Phase 4] Comportements sémantiques pour lang={lang!r}")

    behaviors = [
        ('biological',       'reflexive',      'pronominal',    None,
         'Verbes biologiques avec se → verbe nu (se réveiller)'),
        ('posture',          'reflexive',      'posture',       None,
         'Posture avec se → refl_absolute sans yɛrɛ (s\'asseoir)'),
        ('motion',           'past_form',      'resultative',   None,
         'Mouvement passé → forme résultative (partir → fáɲira)'),
        ('motion',           'relcl_form',     'resultative',   None,
         'Mouvement dans relative → résultatif (l\'homme qui est parti)'),
        ('meteorological',   'construction',   'phenomenon_na', 'na',
         'Météo → [phénomène] bɛ na (il pleut → san bɛ na)'),
        ('statif_emotionnel','construction',   'statif_len_don', None,
         'État émotionnel → V+len dòn (avoir peur → jàpapalen dòn)'),
        ('experiencer_phys', 'construction',   'experiencer_la', None,
         'Sensation physique → STATE bɛ SUBJ la (avoir faim → kɔngɔ bɛ n la)'),
        ('passive',          'construction',   'statif_len_don', None,
         'Passif FR → statif bambara (est bloqué → dábɛrɛbɛrɛmalen dòn)'),
    ]

    count = 0
    for sem_class, behavior_type, behavior_value, extra, desc in behaviors:
        db.query("""
        MERGE (b:SemanticBehavior {lang: $lang, sem_class: $sc, behavior_type: $bt})
        SET b.behavior_value = $bv,
            b.extra = $extra,
            b.description = $desc
        """, {'lang': lang, 'sc': sem_class, 'bt': behavior_type,
              'bv': behavior_value, 'extra': extra, 'desc': desc})
        count += 1

    print(f"   {count} comportements sémantiques créés dans le KG")


def migrate_grammar_markers(db, lang='bm'):
    """Phase 5 : Marqueurs grammaticaux spéciaux (compléments des FunctionWord existants)."""
    print(f"\n[Phase 5] Marqueurs grammaticaux pour lang={lang!r}")

    # Ces marqueurs sont déjà partiellement dans FunctionWord KG
    # On ajoute ceux qui manquent
    markers = [
        # (role, bm, description)
        ('serial_purpose',   'ka',   'Marqueur de but en construction sérielle (V ka V)'),
        ('optative',         'ka',   'Marqueur optatif/jussif'),
        ('prohibitive',      'kàna', 'Marqueur prohibitif'),
        ('interrogative_end','wà ?', 'Marqueur interrogatif polaire'),
        ('statif_pos',       'dòn',  'Support statif positif (-len dòn)'),
        ('statif_neg',       'tɛ',   'Support statif négatif (-len tɛ)'),
        ('resultative_ya',   'ye',   'Marqueur résultatif pour topic'),
        ('age_marker',       'saan', 'Marqueur d\'âge (il a X ans)'),
    ]

    count = 0
    for role, bm, desc in markers:
        # Use MERGE to avoid duplicates, check if FunctionWord doesn't already have it
        existing = db.query(
            "MATCH (f:FunctionWord) WHERE f.role = $role AND f.lang = $lang RETURN f LIMIT 1",
            {'role': role, 'lang': lang})
        if not existing:
            db.query("""
            MERGE (f:FunctionWord {role: $role, lang: $lang, bm: $bm})
            SET f.description = $desc, f.surface = $bm
            """, {'role': role, 'lang': lang, 'bm': bm, 'desc': desc})
            count += 1

    print(f"   {count} marqueurs nouveaux créés (existants préservés)")


def verify_migration(db, lang='bm'):
    """Vérifie que les nœuds sont bien créés."""
    print(f"\n[Vérification] pour lang={lang!r}")
    checks = [
        ("TamConfig", f"MATCH (n:TamConfig {{lang:'{lang}'}}) RETURN count(n) AS c"),
        ("MorphoRule", f"MATCH (n:MorphoRule {{lang:'{lang}'}}) RETURN count(n) AS c"),
        ("ClauseTemplate", f"MATCH (n:ClauseTemplate {{lang:'{lang}'}}) RETURN count(n) AS c"),
        ("SemanticBehavior", f"MATCH (n:SemanticBehavior {{lang:'{lang}'}}) RETURN count(n) AS c"),
    ]
    for label, q in checks:
        res = db.query(q)
        count = res[0]['c'] if res else 0
        status = '' if count > 0 else ''
        print(f"  {status} {label}: {count} nœuds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4, 5], default=None,
                        help='Phase spécifique (1=TAM, 2=Morpho, 3=Templates, 4=Semantic, 5=Markers)')
    parser.add_argument('--lang', default='bm', help='Code langue cible (default: bm)')
    parser.add_argument('--all', action='store_true', help='Toutes les phases')
    args = parser.parse_args()

    db = Neo4jClient()
    print(f"Migration KG pour lang={args.lang!r}")

    if args.all or args.phase == 1 or args.phase is None:
        migrate_tam(db, args.lang)
    if args.all or args.phase == 2:
        migrate_morpho_rules(db, args.lang)
    if args.all or args.phase == 3:
        migrate_clause_templates(db, args.lang)
    if args.all or args.phase == 4:
        migrate_semantic_behaviors(db, args.lang)
    if args.all or args.phase == 5:
        migrate_grammar_markers(db, args.lang)

    verify_migration(db, args.lang)
    print("\n Migration terminée.")


if __name__ == '__main__':
    main()
