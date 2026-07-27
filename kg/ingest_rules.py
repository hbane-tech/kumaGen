"""
kg/ingest_rules.py
Encode toutes les règles de traduction FR→BM dans le KG comme chemins de traversée.

Structure :
  FeaturePattern --[TRIGGERS]--> ConstructionRule --[USES_TEMPLATE]--> ClauseTemplate
                                                   --[APPLIES_MORPHO]--> MorphoRule
                                                   --[REQUIRES]--> FeaturePattern (AND)
                                                   --[OVERRIDES]--> ConstructionRule

Le moteur Python traverse ces chemins en fonction des features extraites de la phrase.
Un seul changement de langue = remplacer les nœuds cibles (bm) dans le KG.

Usage :
    python -m kg.ingest_rules
"""

from kg.neo4j_client import Neo4jClient


def create_indexes(db):
    db.query("CREATE INDEX feature_pattern IF NOT EXISTS FOR (n:FeaturePattern) ON (n.feature, n.value)")
    db.query("CREATE INDEX construction_rule IF NOT EXISTS FOR (n:ConstructionRule) ON (n.name, n.lang)")
    db.query("CREATE INDEX clause_template IF NOT EXISTS FOR (n:ClauseTemplate) ON (n.clause_type, n.lang)")


def ingest_rules(db, lang='bm'):
    print(f"\n[KG Rules] Ingestion des règles pour lang={lang!r}")

    # ── 1. SYSTÈME TAM ────────────────────────────────────────────────────────
    # Feature: tense+neg → TAM marker
    tam_rules = [
        ('pres', False, 'bɛ'),
        ('pres', True,  'tɛ'),
        ('past', False, 'yé'),
        ('past', True,  'ma'),
        ('fut',  False, 'bɛ́nà'),
        ('fut',  True,  'tɛ́nà'),
        ('hab',  False, 'tùn bɛ'),
        ('hab',  True,  'tùn tɛ'),
        ('plup', False, 'tùn yé'),
        ('plup', True,  'tùn ma'),
        ('prog', False, 'bɛ́kà'),
        ('prog', True,  'tɛ́kà'),
    ]
    for tense, neg, bm in tam_rules:
        db.query("""
        MERGE (fp:FeaturePattern {feature:'tense', value:$tense, neg:$neg, lang:$lang})
        MERGE (cr:ConstructionRule {name:'tam_'+$tense+'_'+toString($neg), lang:$lang})
          SET cr.tam_value = $bm, cr.priority = 10
        MERGE (fp)-[:TRIGGERS]->(cr)
        """, {'tense': tense, 'neg': neg, 'bm': bm, 'lang': lang})

    print(f"   TAM: {len(tam_rules)} règles")

    # ── 2. CONSTRUCTION PASSIVE → STATIF ─────────────────────────────────────
    # is_passive=True → V+len dòn (pas V+ra comme le résultatif ordinaire)
    db.query("""
    MERGE (fp:FeaturePattern {feature:'is_passive', value:'true', lang:$lang})
    MERGE (cr:ConstructionRule {name:'passive_statif', lang:$lang})
      SET cr.description = 'Passif FR = statif bambara : V+len dòn',
          cr.priority = 90
    MERGE (mr:MorphoRule {name:'statif', lang:$lang})
    MERGE (ct:ClauseTemplate {clause_type:'passive_statif', lang:$lang})
      SET ct.template = '{S} {V}len {dòn|tɛ} {ADV}',
          ct.word_order = 'S V+len TAM ADV'
    MERGE (fp)-[:TRIGGERS]->(cr)
    MERGE (cr)-[:APPLIES_MORPHO]->(mr)
    MERGE (cr)-[:USES_TEMPLATE]->(ct)
    """, {'lang': lang})
    print(f"   Passive → statif")

    # ── 3. RÉFLEXIFS PAR CLASSE SÉMANTIQUE ──────────────────────────────────
    refl_rules = [
        # (semantic_class, refl_treatment, description)
        ('biological', 'pronominal',
         'Verbe biologique avec se → verbe nu (se réveiller, s\'endormir)'),
        ('posture',    'posture',
         'Posture avec se → refl_absolute sans yɛrɛ (s\'asseoir → a bɛ a sìgi)'),
        ('perception', 'actif',
         'Perception avec se → actif + yɛrɛ (se regarder → a bɛ a yɛrɛ fílɛ)'),
        ('spontaneous','accidentel',
         'Spontané avec se → accidentel + yɛrɛ (se blesser → a bɛ a yɛrɛ màjógin)'),
    ]
    for sc, treatment, desc in refl_rules:
        db.query("""
        MERGE (fp:FeaturePattern {feature:'semantic_class', value:$sc,
                                  context:'reflexive', lang:$lang})
        MERGE (cr:ConstructionRule {name:'refl_'+$sc, lang:$lang})
          SET cr.refl_treatment = $treatment,
              cr.description = $desc,
              cr.priority = 70
        MERGE (fp)-[:TRIGGERS]->(cr)
        """, {'sc': sc, 'treatment': treatment, 'desc': desc, 'lang': lang})
    print(f"   Réflexifs: {len(refl_rules)} règles")

    # ── 4. CONSTRUCTION MÉTÉOROLOGIQUE ────────────────────────────────────────
    db.query("""
    MERGE (fp:FeaturePattern {feature:'semantic_class', value:'meteorological', lang:$lang})
    MERGE (cr:ConstructionRule {name:'meteorological', lang:$lang})
      SET cr.motion_verb = 'na',
          cr.description = 'Impersonnel météo : phénomène bɛ na (il pleut → san bɛ na)',
          cr.priority = 95
    MERGE (ct:ClauseTemplate {clause_type:'meteorological', lang:$lang})
      SET ct.template = '{PHENOMENON} {TAM} {MOTION_VERB}',
          ct.word_order = 'PHENOMENON TAM MOTION'
    MERGE (fp)-[:TRIGGERS]->(cr)
    MERGE (cr)-[:USES_TEMPLATE]->(ct)
    """, {'lang': lang})
    print(f"   Météorologique")

    # ── 5. MORPHOLOGIE RÉSULTATIVE ────────────────────────────────────────────
    # past + intransitive → V+ra (pas de TAM)
    db.query("""
    MERGE (fp1:FeaturePattern {feature:'tense', value:'past', lang:$lang})
    MERGE (fp2:FeaturePattern {feature:'intransitive_type', value:'ABSOLU', lang:$lang})
    MERGE (cr:ConstructionRule {name:'resultative_past', lang:$lang})
      SET cr.description = 'Passé intransitif → résultatif V+ra (sans TAM)',
          cr.priority = 80
    MERGE (mr:MorphoRule {name:'resultative', lang:$lang})
    MERGE (ct:ClauseTemplate {clause_type:'resultative', lang:$lang})
      SET ct.template = '{S} {V+sfx}',
          ct.word_order = 'S V+resultative_suffix'
    MERGE (fp1)-[:TRIGGERS]->(cr)
    MERGE (fp2)-[:REQUIRED_BY]->(cr)
    MERGE (cr)-[:APPLIES_MORPHO]->(mr)
    MERGE (cr)-[:USES_TEMPLATE]->(ct)
    """, {'lang': lang})
    print(f"   Résultatif passé intransitif")

    # ── 6. ÉTAT EMOTIONNEL (avoir peur) ──────────────────────────────────────
    db.query("""
    MERGE (fp:FeaturePattern {feature:'possession_type', value:'STATIF', lang:$lang})
    MERGE (cr:ConstructionRule {name:'avoir_statif_emotionnel', lang:$lang})
      SET cr.description = 'avoir peur/honte → S V+len dòn (état émotionnel)',
          cr.priority = 85
    MERGE (mr:MorphoRule {name:'statif', lang:$lang})
    MERGE (ct:ClauseTemplate {clause_type:'avoir_statif', lang:$lang})
      SET ct.template = '{S} {V}len {dòn|tɛ}',
          ct.pos_form = '{S} {V}len dòn',
          ct.neg_form = '{S} {V}len tɛ',
          ct.hab_form = '{S} tùn {V}len {dòn|tɛ}',
          ct.past_form = '{S} {V+sfx}'
    MERGE (fp)-[:TRIGGERS]->(cr)
    MERGE (cr)-[:APPLIES_MORPHO]->(mr)
    MERGE (cr)-[:USES_TEMPLATE]->(ct)
    """, {'lang': lang})
    print(f"   État émotionnel (STATIF)")

    # ── 7. SÉRIELLE MOUVEMENT+BUT ────────────────────────────────────────────
    db.query("""
    MERGE (fp1:FeaturePattern {feature:'clause_type', value:'verb_serial', lang:$lang})
    MERGE (fp2:FeaturePattern {feature:'root_semantic_class', value:'motion', lang:$lang})
    MERGE (cr:ConstructionRule {name:'serial_motion_purpose', lang:$lang})
      SET cr.purpose_marker = 'ka',
          cr.description = 'Sérielle mouvement+but : V1 ka O V2 (aller chercher)',
          cr.priority = 75
    MERGE (ct:ClauseTemplate {clause_type:'serial_motion_purpose', lang:$lang})
      SET ct.template = '{S} {TAM} {V1} ka {O} {V2}',
          ct.word_order = 'S TAM MOTION ka O PURPOSE'
    MERGE (fp1)-[:TRIGGERS]->(cr)
    MERGE (fp2)-[:REQUIRED_BY]->(cr)
    MERGE (cr)-[:USES_TEMPLATE]->(ct)
    """, {'lang': lang})
    print(f"   Sérielle mouvement+but")

    # ── 8. IMPÉRATIF PAR PERSONNE ────────────────────────────────────────────
    imperative_rules = [
        ('2sg', '',      '{O} {V}',         'Impératif 2e sing → verbe nu'),
        ('2pl', 'aw yé', 'aw yé {O} {V}',  'Impératif 2e plur → aw yé V'),
        ('1pl', 'an ka', 'an ka {O} {V}',  'Impératif 1e plur → an ka V'),
    ]
    for person, marker, template, desc in imperative_rules:
        db.query("""
        MERGE (fp:FeaturePattern {feature:'imperative_person', value:$person, lang:$lang})
        MERGE (cr:ConstructionRule {name:'imperative_'+$person, lang:$lang})
          SET cr.prefix = $marker,
              cr.description = $desc,
              cr.priority = 85
        MERGE (ct:ClauseTemplate {clause_type:'imperative_'+$person, lang:$lang})
          SET ct.template = $template
        MERGE (fp)-[:TRIGGERS]->(cr)
        MERGE (cr)-[:USES_TEMPLATE]->(ct)
        """, {'person': person, 'marker': marker, 'template': template,
               'desc': desc, 'lang': lang})
    print(f"   Impératif: {len(imperative_rules)} règles personne")

    # ── 9. RELATIVE CLAUSE → RÉSULTATIF ─────────────────────────────────────
    db.query("""
    MERGE (fp1:FeaturePattern {feature:'dep', value:'acl:relcl', lang:$lang})
    MERGE (fp2:FeaturePattern {feature:'semantic_class', value:'motion',
                               context:'relcl', lang:$lang})
    MERGE (cr:ConstructionRule {name:'relcl_motion_resultative', lang:$lang})
      SET cr.description = 'Relative motion passé → résultatif (qui est parti → mìn fáɲira)',
          cr.priority = 78
    MERGE (mr:MorphoRule {name:'resultative', lang:$lang})
    MERGE (ct:ClauseTemplate {clause_type:'relcl_motion', lang:$lang})
      SET ct.template = 'mìn {V+sfx}',
          ct.relative_marker = 'mìn'
    MERGE (fp1)-[:TRIGGERS]->(cr)
    MERGE (fp2)-[:REQUIRED_BY]->(cr)
    MERGE (cr)-[:APPLIES_MORPHO]->(mr)
    MERGE (cr)-[:USES_TEMPLATE]->(ct)
    """, {'lang': lang})
    print(f"   Relative motion résultatif")

    # ── 10. QUESTION POURQUOI + PASSIF ───────────────────────────────────────
    db.query("""
    MERGE (fp1:FeaturePattern {feature:'is_passive', value:'true', lang:$lang})
    MERGE (fp2:FeaturePattern {feature:'has_adv_interrogative', value:'true', lang:$lang})
    MERGE (cr:ConstructionRule {name:'passive_statif_question', lang:$lang})
      SET cr.description = 'Passif + pourquoi → S V+len dòn ADV ?',
          cr.priority = 92,
          cr.preserves_clause_type = 'content_question'
    MERGE (ct:ClauseTemplate {clause_type:'passive_statif_question', lang:$lang})
      SET ct.template = '{S} {V}len {dòn|tɛ} {ADV} ?'
    MERGE (fp1)-[:TRIGGERS]->(cr)
    MERGE (fp2)-[:REQUIRED_BY]->(cr)
    MERGE (cr)-[:OVERRIDES]->(:ConstructionRule {name:'passive_statif', lang:$lang})
    MERGE (cr)-[:USES_TEMPLATE]->(ct)
    """, {'lang': lang})
    print(f"   Passif + question")

    # ── 11. TOUTES LES CONSTRUCTIONS PAR CLAUSE_TYPE ─────────────────────────
    clause_constructions = [
        # (clause_type, template, word_order, description)
        ('simple',          '{S} {TAM} {O} {V} {ADV}',        'SOV',
         'Clause déclarative standard'),
        ('equative',        '{S} yé {O} yé',                   'S=O',
         'Équatif positif (il est médecin)'),
        ('equative_neg',    '{S} tɛ {O} yé',                   'S≠O',
         'Équatif négatif'),
        ('identificatory',  '{S} yé {O} yé',                   'S=O',
         'Identificatoire (c\'est X)'),
        ('noun_phrase',     '{O} bɛ {S} {MARKER}',             'NP',
         'Possession/être avoir (O bɛ S fɛ/bóló)'),
        ('noun_phrase_have','voir ClauseTemplate possession_type','NP_have',
         'Avoir+sensation (kɔngɔ bɛ n la / n jàpapalen dòn)'),
        ('optative',        '{S} ka {O} {V}',                  'S ka O V',
         'Optatif/jussif (Que la paix règne)'),
        ('prohibitive',     'kàna {O} {V}',                    'kàna O V',
         'Prohibitif (ne mange pas)'),
        ('interrogative',   '{S} {TAM} {O} {V} wà ?',          'SOV wà ?',
         'Question polaire'),
        ('content_question','{S} {TAM} {O} {V} {ADV} ?',       'SOV ADV ?',
         'Question de contenu (qui, quoi, pourquoi)'),
        ('locative',        '{S} bɛ {OBL}',                    'S bɛ LOC',
         'Locatif (il est à Bamako)'),
        ('existential',     '{O} bɛ yan',                      'O bɛ',
         'Existentiel (il y a)'),
        ('reciprocal',      '{S} {TAM} ɲɔgɔn {V}',            'S TAM ɲɔgɔn V',
         'Réciproque (ils se voient)'),
        ('refl_absolute',   '{S} {TAM} {S} {yɛrɛ} {V}',       'S TAM S (yɛrɛ) V',
         'Réflexif absolu (il se blesse → a bɛ a yɛrɛ V)'),
        ('verb_serial',     '{S} {TAM} {V1} ka {O} {V2}',      'S TAM V1 ka O V2',
         'Construction sérielle (V ka V)'),
        ('passive_statif',  '{S} {V}len {dòn|tɛ} {ADV}',       'S V+len dòn',
         'Passif = statif (est bloqué → V+len dòn)'),
        ('relative_topic',  '{S} mìn {TAM} {O} {V}, o {TAM2} {O2} {V2}',
         'rel, o ...',
         'Relative topicalisée (l\'homme que tu vois est...)'),
        ('meteorological',  '{PHENOMENON} {TAM} {MOTION}',     'PHEN TAM na',
         'Météorologique (il pleut → san bɛ na)'),
        ('statif',          '{S} {V+len} {dòn|tɛ}',            'S V+len dòn',
         'Statif adjectival (il est assis → a dòn/sìgilen dòn)'),
        ('imperative_2sg',  '{O} {V}',                          'O V',
         'Impératif 2e sing (aide-moi → n bólodɛ̀mɛ)'),
        ('imperative_2pl',  'aw yé {O} {V}',                   'aw yé O V',
         'Impératif 2e plur (aidez-moi → aw yé n bólodɛ̀mɛ)'),
        ('imperative_1pl',  'an ka {O} {V}',                   'an ka O V',
         'Impératif 1e plur (parlons → an ka kuma)'),
    ]

    for ct, template, word_order, desc in clause_constructions:
        # FeaturePattern sur clause_type
        db.query("""
        MERGE (fp:FeaturePattern {feature:'clause_type', value:$ct, lang:$lang})
        MERGE (cr:ConstructionRule {name:'render_'+$ct, lang:$lang})
          SET cr.description = $desc, cr.priority = 50
        MERGE (ct_node:ClauseTemplate {clause_type:$ct, lang:$lang})
          SET ct_node.template = $template,
              ct_node.word_order = $word_order,
              ct_node.description = $desc
        MERGE (fp)-[:TRIGGERS]->(cr)
        MERGE (cr)-[:USES_TEMPLATE]->(ct_node)
        """, {'ct': ct, 'template': template, 'word_order': word_order,
               'desc': desc, 'lang': lang})

    print(f"   {len(clause_constructions)} constructions clause_type")

    # Vérification finale
    counts = {}
    for label in ['FeaturePattern', 'ConstructionRule', 'ClauseTemplate', 'MorphoRule']:
        res = db.query(f"MATCH (n:{label} {{lang:$lang}}) RETURN count(n) AS c",
                       {'lang': lang})
        counts[label] = res[0]['c'] if res else 0

    print(f"\n  [KG Rules graph pour lang={lang!r}]")
    for label, c in counts.items():
        print(f"    {label}: {c} nœuds")

    # Vérifier les chemins
    paths = db.query("""
    MATCH (fp:FeaturePattern {lang:$lang})-[:TRIGGERS]->(cr:ConstructionRule {lang:$lang})
    -[:USES_TEMPLATE]->(ct:ClauseTemplate {lang:$lang})
    RETURN count(*) AS paths
    """, {'lang': lang})
    print(f"    Chemins complets (FP→CR→CT): {paths[0]['paths'] if paths else 0}")


def create_traversal_query(lang='bm') -> str:
    """Retourne la requête Cypher pour traverser les règles.
    Usage dans le moteur : match features → find rules → get template.
    """
    return """
    MATCH (fp:FeaturePattern {lang: $lang})-[:TRIGGERS]->(cr:ConstructionRule {lang: $lang})
    WHERE fp.feature IN $feature_keys
      AND fp.value IN $feature_values
    WITH cr ORDER BY cr.priority DESC
    OPTIONAL MATCH (cr)-[:USES_TEMPLATE]->(ct:ClauseTemplate {lang: $lang})
    OPTIONAL MATCH (cr)-[:APPLIES_MORPHO]->(mr:MorphoRule {lang: $lang})
    RETURN cr.name AS rule_name,
           cr.priority AS priority,
           cr.description AS description,
           ct.template AS template,
           ct.word_order AS word_order,
           mr.name AS morpho_rule,
           cr.refl_treatment AS refl_treatment,
           cr.motion_verb AS motion_verb,
           cr.prefix AS prefix
    LIMIT 5
    """


if __name__ == '__main__':
    db = Neo4jClient()
    create_indexes(db)
    ingest_rules(db)
    print(f"\n Règles de traduction ingérées dans le KG.")
    print(f"\nExemple de requête de traversée :")
    print(create_traversal_query())
