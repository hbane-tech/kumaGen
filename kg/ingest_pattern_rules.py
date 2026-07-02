"""
kg/ingest_pattern_rules.py
Encode TOUTES les règles de détection/remplissage dans le KG.

PatternRule  : "si sentence a features X → clause_type = Y"
SlotFillRule : "si token a dep=X pos=Y → mettre bm dans slot Z"
TransformRule: "si tense=past et intransitive → ajouter suffixe résultatif à V"

Le moteur Python devient un interpréteur pur qui lit ces règles.
Zéro hardcode dans le code Python.
"""

from kg.neo4j_client import Neo4jClient


def create_indexes(db):
    db.query("CREATE INDEX pattern_rule IF NOT EXISTS FOR (n:PatternRule) ON (n.name, n.lang)")
    db.query("CREATE INDEX slot_fill_rule IF NOT EXISTS FOR (n:SlotFillRule) ON (n.match_dep, n.lang)")
    db.query("CREATE INDEX transform_rule IF NOT EXISTS FOR (n:TransformRule) ON (n.name, n.lang)")


def ingest_pattern_rules(db, lang='bm'):
    """
    PatternRule : détection du clause_type à partir des features de la phrase.
    Chaque règle contient :
      feature_key   : nom de la feature testée
      feature_value : valeur attendue
      clause_type   : résultat si la règle matche
      priority      : règle de plus haute priorité gagne
    """
    print(f"\n[PatternRules] pour lang={lang!r}")

    rules = [
        # (name, feature_conditions, clause_type, priority, description)
        # feature_conditions = liste de (feature_key, feature_value)

        # ── Détection IMPÉRATIF ─────────────────────────────────────────
        ('imperative_affirm',
         [('root_pos', 'VERB'), ('has_nsubj', 'false'), ('has_exclamation', 'true'),
          ('has_negation', 'false')],
         'imperative', 85,
         'Verbe ROOT sans sujet + ! → impératif'),

        ('prohibitive',
         [('root_pos', 'VERB'), ('has_nsubj', 'false'), ('has_negation', 'true')],
         'prohibitive', 85,
         'Verbe ROOT sans sujet + négation → prohibitif'),

        # ── Détection PASSIF ────────────────────────────────────────────
        ('passive_with_adv_q',
         [('is_passive', 'true'), ('has_adv_interrogative', 'true')],
         'passive_statif_question', 92,
         'Passif + adverbe interrogatif → statif + question'),

        ('passive_simple',
         [('is_passive', 'true')],
         'passive_statif', 88,
         'Passif → construction statif (V+len dòn)'),

        # ── Détection MÉTÉOROLOGIQUE ────────────────────────────────────
        ('meteorological',
         [('root_semantic_class', 'meteorological')],
         'meteorological', 95,
         'Verbe météo impersonnel → san bɛ na'),

        # ── Détection ÉQUATIF ───────────────────────────────────────────
        ('equative',
         [('has_cop', 'true'), ('root_pos', 'NOUN'), ('has_nsubj', 'true')],
         'equative', 80,
         'NOUN ROOT + copule → équatif (il est médecin)'),

        # ── Détection SÉRIELLE ──────────────────────────────────────────
        ('verb_serial',
         [('has_xcomp_verb', 'true'), ('root_pos', 'VERB')],
         'verb_serial', 60,
         'VERB ROOT + xcomp VERB → sérielle'),

        # ── Détection QUESTION CONTENU ──────────────────────────────────
        ('content_question_pron_root',
         [('root_pos', 'PRON'), ('root_role', 'interrogative'), ('has_question', 'true')],
         'content_question', 75,
         'PRON ROOT interrogatif → question de contenu'),

        ('content_question_adv',
         [('has_adv_interrogative', 'true'), ('has_question', 'true')],
         'content_question', 70,
         'Adverbe interrogatif + ? → question de contenu'),

        # ── Détection INTERROGATIF POLAIRE ──────────────────────────────
        ('interrogative_polar',
         [('has_question', 'true'), ('has_interrogative_word', 'false')],
         'interrogative', 65,
         'Question sans mot interrogatif → polaire'),

        # ── Détection OPTATIVE ──────────────────────────────────────────
        ('optative',
         [('root_tense', 'sub'), ('has_nsubj', 'false')],
         'optative', 72,
         'Subjonctif sans sujet explicite → optatif'),

        ('optative_with_nsubj',
         [('root_tense', 'sub'), ('has_nsubj', 'true')],
         'optative', 71,
         'Subjonctif avec sujet → optatif (Que la paix règne)'),

        # ── Détection EXISTENTIEL ───────────────────────────────────────
        ('existential_il_y_a',
         [('root_lemma', 'avoir'), ('has_expletive', 'true')],
         'existential_absolute', 90,
         'il y a → existentiel'),

        # ── Détection LOCATIF ───────────────────────────────────────────
        ('locative',
         [('has_loc_case', 'true'), ('root_pos', 'VERB')],
         'locative', 68,
         'Verbe avec cas locatif → locatif'),

        # ── SIMPLE (défaut) ─────────────────────────────────────────────
        ('simple_default',
         [('root_pos', 'VERB'), ('has_nsubj', 'true')],
         'simple', 10,
         'VERB ROOT + sujet → clause simple (SOV par défaut)'),

        ('simple_noun_root',
         [('root_pos', 'NOUN')],
         'noun_phrase', 15,
         'NOUN ROOT → syntagme nominal'),
    ]

    count = 0
    for name, conditions, clause_type, priority, desc in rules:
        # Créer le nœud PatternRule
        db.query("""
        MERGE (r:PatternRule {name: $name, lang: $lang})
        SET r.clause_type = $ct,
            r.priority    = $priority,
            r.description = $desc
        """, {'name': name, 'lang': lang, 'ct': clause_type,
              'priority': priority, 'desc': desc})

        # Créer les conditions comme propriétés JSON
        for i, (fkey, fval) in enumerate(conditions):
            db.query("""
            MATCH (r:PatternRule {name: $name, lang: $lang})
            SET r[$fk] = $fv
            """, {'name': name, 'lang': lang,
                  'fk': f'cond_{fkey}', 'fv': fval})

        count += 1

    print(f"  ✅ {count} PatternRules créées")


def ingest_slot_fill_rules(db, lang='bm'):
    """
    SlotFillRule : quel slot remplir pour chaque token selon dep/pos/role.
    Chaque règle dit : "si ce token matche → mettre son bm dans ce slot"
    """
    print(f"\n[SlotFillRules] pour lang={lang!r}")

    rules = [
        # (name, match_dep, match_pos, match_role, target_slot, value_source, priority, desc)
        # value_source : 'bm' | 'tam_from_tense' | 'morpho_resultative' | 'chain'

        # ── SUJET (slot S) ──────────────────────────────────────────────
        ('subj_pron',       'nsubj',      'PRON', None,      'S',     'bm',    80, 'Pronom sujet → S'),
        ('subj_noun',       'nsubj',      'NOUN', None,      'S',     'chain', 75, 'Nom sujet (+ det/poss) → S'),
        ('subj_propn',      'nsubj',      'PROPN',None,      'S',     'bm',    80, 'Nom propre sujet → S'),
        ('subj_pass',       'nsubj:pass', 'NOUN', None,      'S',     'chain', 80, 'Sujet passif → S'),
        ('subj_pass_pron',  'nsubj:pass', 'PRON', None,      'S',     'bm',    80, 'Pronom sujet passif → S'),

        # ── VERBE (slot V) ──────────────────────────────────────────────
        ('verb_root',       'ROOT',       'VERB', 'content', 'V',     'bm',    90, 'Verbe ROOT → V'),
        ('verb_root_aux',   'ROOT',       'AUX',  'content', 'V',     'bm',    88, 'AUX ROOT → V (rare)'),

        # ── VERBE SÉRIEL (slot V_ACT) ───────────────────────────────────
        ('xcomp_verb',      'xcomp',      'VERB', 'content', 'V_ACT', 'bm',    85, 'xcomp VERB → V_ACT'),

        # ── OBJET (slot O) ──────────────────────────────────────────────
        ('obj_noun',        'obj',        'NOUN', 'content', 'O',     'chain', 80, 'Nom objet → O'),
        ('obj_pron',        'obj',        'PRON', 'object',  'O',     'bm',    82, 'Pronom objet → O'),
        ('obj_propn',       'obj',        'PROPN','content', 'O',     'bm',    80, 'Nom propre objet → O'),
        ('obj_interrog',    'obj',        'PRON', 'interrogative', 'O', 'bm',  85, 'Pronom interrog objet → O'),

        # ── ADVERBE (slot ADV) ──────────────────────────────────────────
        ('adv_interrog',    'advmod',     'ADV',  'interrogative', 'ADV', 'bm', 80, 'Adverbe interrog → ADV'),
        ('adv_manner',      'advmod',     'ADV',  'content', 'ADV',   'bm',    50, 'Adverbe manière → ADV'),
    ]

    count = 0
    for (name, dep, pos, role, slot, vsrc, prio, desc) in rules:
        db.query("""
        MERGE (r:SlotFillRule {name: $name, lang: $lang})
        SET r.match_dep    = $dep,
            r.match_pos    = $pos,
            r.match_role   = $role,
            r.target_slot  = $slot,
            r.value_source = $vsrc,
            r.priority     = $prio,
            r.description  = $desc
        """, {'name': name, 'lang': lang, 'dep': dep, 'pos': pos,
              'role': role, 'slot': slot, 'vsrc': vsrc,
              'prio': prio, 'desc': desc})
        count += 1

    print(f"  ✅ {count} SlotFillRules créées")


def ingest_transform_rules(db, lang='bm'):
    """
    TransformRule : transformations morphologiques sur les slots.
    Conditions sur le tree (tense, neg, intransitive, etc.) → action sur slot.
    """
    print(f"\n[TransformRules] pour lang={lang!r}")

    rules = [
        # (name, conditions, target_slot, transform_type, priority, desc)
        ('resultative_past_intrans',
         {'tense': 'past', 'intransitive': 'true', 'neg': 'false'},
         'V', 'add_resultative_suffix', 80,
         'Passé intransitif positif → V+ra/na/la (depuis MorphoRule.resultative)'),

        ('resultative_neg',
         {'tense': 'past', 'intransitive': 'true', 'neg': 'true'},
         'TAM', 'set_value:ma', 80,
         'Passé intransitif négatif → TAM=ma'),

        ('statif_pres_pos',
         {'construction': 'statif', 'tense': 'pres', 'neg': 'false'},
         'V', 'add_statif_suffix_pos', 85,
         'Statif présent positif → V+len dòn'),

        ('statif_pres_neg',
         {'construction': 'statif', 'tense': 'pres', 'neg': 'true'},
         'V', 'add_statif_suffix_neg', 85,
         'Statif présent négatif → V+len tɛ'),

        ('statif_hab',
         {'construction': 'statif', 'tense': 'hab'},
         'V', 'add_statif_suffix_hab', 85,
         'Statif habituel → tùn V+len dòn/tɛ'),

        ('nominalization_action_pres',
         {'action_no_obj': 'true', 'tense': 'pres', 'neg': 'false'},
         'V', 'add_nominalization', 70,
         'Action sans objet présent → V+li kɛ'),

        ('tam_pres_pos',
         {'tense': 'pres', 'neg': 'false'},
         'TAM', 'set_from_kg_tam', 50,
         'Présent positif → TAM=bɛ depuis KG TamConfig'),

        ('tam_past_pos',
         {'tense': 'past', 'transitive': 'true', 'neg': 'false'},
         'TAM', 'set_from_kg_tam', 50,
         'Passé positif transitif → TAM=yé depuis KG'),

        ('tam_optative',
         {'clause_type': 'optative'},
         'TAM', 'set_value:ka', 90,
         'Optatif → TAM=ka'),
    ]

    count = 0
    for (name, conds, slot, transform, prio, desc) in rules:
        params = {'name': name, 'lang': lang, 'slot': slot,
                  'transform': transform, 'prio': prio, 'desc': desc}
        params.update({f'cond_{k}': v for k, v in conds.items()})

        set_clauses = ', '.join([f'r.cond_{k} = ${k_}' for k, k_ in
                                 [(k, f'cond_{k}') for k in conds]])
        db.query(f"""
        MERGE (r:TransformRule {{name: $name, lang: $lang}})
        SET r.target_slot = $slot,
            r.transform   = $transform,
            r.priority    = $prio,
            r.description = $desc,
            {set_clauses}
        """, params)
        count += 1

    print(f"  ✅ {count} TransformRules créées")


def verify(db, lang='bm'):
    print(f"\n[Vérification] pour lang={lang!r}")
    for label in ['PatternRule', 'SlotFillRule', 'TransformRule']:
        res = db.query(f"MATCH (n:{label} {{lang:$lang}}) RETURN count(n) AS c", {'lang': lang})
        c = res[0]['c'] if res else 0
        print(f"  {label}: {c} nœuds")


if __name__ == '__main__':
    db = Neo4jClient()
    create_indexes(db)
    ingest_pattern_rules(db)
    ingest_slot_fill_rules(db)
    ingest_transform_rules(db)
    verify(db)
    print("\n✅ Règles KG complètes.")
