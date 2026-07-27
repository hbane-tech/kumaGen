"""
kg/ingest_full_migration.py
Migration complète de TOUTE la logique Python vers le KG.

Nouveaux types de règles :
  CoordRule      → coordination verbale (wa, ani ka)
  ModalRule      → constructions modales (se ka, ŋàniya ka)
  ObliqueFillRule → remplissage des obliques avec marqueurs
  NominalChainRule → construction des syntagmes nominaux
  AuxTemporalRule → temps composés (passé composé, futur proche)
  ReflexiveRule   → réflexif/réciproque
  PossessionRule  → avoir + type de possession

Usage :
    python -m kg.ingest_full_migration
"""

from kg.neo4j_client import Neo4jClient


def create_indexes(db):
    for label in ['CoordRule','ModalRule','ObliqueFillRule',
                  'NominalChainRule','AuxTemporalRule','PossessionRule']:
        db.query(f"CREATE INDEX {label.lower()} IF NOT EXISTS FOR (n:{label}) ON (n.name, n.lang)")


def ingest_coord_rules(db, lang='bm'):
    """Coordination verbale : V wa V2 / V ani ka V2 / V ka V2."""
    rules = [
        ('coord_et_action',
         {'cconj_surface':'et', 'coord_v_semantic':'action'},
         'wa', '{S} {TAM} {V} wa {S_PRON} {TAM} {O2} {V2}',
         60, 'et + verbe action → wa (il mange et dort)'),

        ('coord_et_infinitif',
         {'cconj_surface':'et', 'coord_v_infinitif':'true'},
         'ani ka', '{S} {TAM} {V} ani ka {V2}',
         60, 'et + infinitif coordonné → ani ka V2'),

        ('coord_ou',
         {'cconj_role':'alternative'},
         'wàlima', '{S} {TAM} {V} wàlima {S} {TAM} {V2}',
         60, 'ou → wàlima'),
    ]
    for name, conds, marker, template, prio, desc in rules:
        params = {'name':name,'lang':lang,'marker':marker,'template':template,'prio':prio,'desc':desc}
        set_parts = ['r.coord_marker=$marker','r.template=$template','r.priority=$prio','r.description=$desc']
        for k,v in conds.items():
            params[f'c_{k}'] = v
            set_parts.append(f'r.cond_{k}=$c_{k}')
        db.query(f'MERGE (r:CoordRule {{name:$name,lang:$lang}}) SET '+', '.join(set_parts), params)
    print(f'   {len(rules)} CoordRules')


def ingest_modal_rules(db, lang='bm'):
    """Constructions modales : pouvoir=se, vouloir=ŋàniya, devoir=ka kan, falloir=ka kan."""
    rules = [
        ('modal_pouvoir',  'pouvoir', 'se',       'se ka', '{S} {TAM} se ka {O} {V2}', 85, 'pouvoir → se ka V'),
        ('modal_vouloir',  'vouloir', 'ŋàniya',   'ŋàniya ka', '{S} {TAM} ŋàniya ka {O} {V2}', 85, 'vouloir → ŋàniya ka V'),
        ('modal_devoir',   'devoir',  'ka kan',   'ka kan', '{S} {V2} ka kan', 85, 'devoir → V ka kan'),
        ('modal_savoir',   'savoir',  'dɔ́n',      'se', '{S} {TAM} dɔ́n {O} {V2} ka', 85, 'savoir + inf → dɔ́n … ka'),
        ('modal_commencer','commencer','daminɛ',   'ka', '{S} {TAM} daminɛ ka {V2}', 80, 'commencer à → daminɛ ka V'),
        ('modal_continuer','continuer','tɔgɔ',     'ka', '{S} {TAM} tɔgɔ ka {V2}', 80, 'continuer à → tɔgɔ ka V'),
        ('modal_cesser',   'cesser',  'banna',     '', '{S} yé {V2}li kɛ banna', 80, 'cesser de → V+li kɛ banna'),
        ('modal_essayer',  'essayer', 'kɔ́nikɔni',  'ka', '{S} {TAM} kɔ́nikɔni ka {V2}', 80, 'essayer de → kɔ́nikɔni ka V'),
    ]
    for name, lemma, v_bm, marker, template, prio, desc in rules:
        db.query('''
        MERGE (r:ModalRule {name:$name, lang:$lang})
        SET r.trigger_lemma=$lemma, r.verb_bm=$vbm, r.purpose_marker=$marker,
            r.template=$tpl, r.priority=$prio, r.description=$desc
        ''', {'name':name,'lang':lang,'lemma':lemma,'vbm':v_bm,
              'marker':marker,'tpl':template,'prio':prio,'desc':desc})
    print(f'   {len(rules)} ModalRules')


def ingest_oblique_fill_rules(db, lang='bm'):
    """Règles de remplissage des obliques avec marqueurs postpositionnels."""
    rules = [
        # (name, dep, case_role, marker_source, default_marker, frame, priority, desc)
        ('obl_locative',   'obl:arg','locative',   'bm_marker', 'la',  'locative',  70, 'Oblique locatif → X la'),
        ('obl_locative2',  'obl',    'locative',   'bm_marker', 'la',  'locative',  70, 'obl locatif → X la'),
        ('obl_temporal',   'obl:mod','temporal',   'bm_marker', '',    'temporal',  70, 'Oblique temporel → marqueur temporel'),
        ('obl_dative',     'obl:arg','dative',     'bm_marker', 'ma',  'dative',    75, 'Oblique datif → X ma'),
        ('obl_comitative', 'obl:mod','comitative', 'bm',        'ni',  'comitative',70, 'Oblique comitatif → ni X'),
        ('obl_genitive',   'nmod',   'genitive',   'bm_marker', 'ka',  'genitive',  65, 'nmod génitif → X ka Y'),
        ('obl_agent',      'obl:arg','agent',      'bm_marker', 'fɛ',  'agent',     75, 'Agent du passif → X fɛ'),
        ('obl_source',     'obl:arg','source',     'bm_marker', 'la',  'locative',  72, 'Source/origine → bɔ X la'),
        ('obl_purposive',  'obl:arg','purposive',  'bm',        'ka',  'purposive', 72, 'Objet purposif → ka X'),
        ('obl_concessive', 'obl:mod','concessive', 'bm_marker', 'nìn', 'concessive',68, 'Concessif → nìn fɔ...'),
        ('obl_associative','obl:arg','associative','bm_marker', 'fɛ',  'associative',70,'Associatif → X fɛ'),
        ('obl_surface',    'obl:arg','surface',    'bm_marker', 'kan', 'surface',   70, 'Surface → X kan'),
        ('obl_under',      'obl:arg','under',      'bm_marker', 'kɔrɔ','under',    70, 'Sous → X kɔrɔ'),
    ]
    for (name,dep,role,mksrc,defmk,frame,prio,desc) in rules:
        db.query('''
        MERGE (r:ObliqueFillRule {name:$name, lang:$lang})
        SET r.match_dep=$dep, r.match_case_role=$role,
            r.marker_source=$mksrc, r.default_marker=$defmk,
            r.frame=$frame, r.priority=$prio, r.description=$desc
        ''', {'name':name,'lang':lang,'dep':dep,'role':role,'mksrc':mksrc,
              'defmk':defmk,'frame':frame,'prio':prio,'desc':desc})
    print(f'   {len(rules)} ObliqueFillRules')


def ingest_nominal_chain_rules(db, lang='bm'):
    """Construction des syntagmes nominaux : det/poss/nummod + tête."""
    rules = [
        ('np_poss_n',   'n',  'ka', '{poss} {head}',      80, 'Possessif n (mon) : n head'),
        ('np_poss_other','',  'ka', '{poss} ka {head}',   80, 'Possessif autre : poss ka head'),
        ('np_demo_prefix','nin','in','{demo} {head} {sfx}',78,'Démonstratif : nin head in'),
        ('np_nummod',   '',   '',   '{head} {num}',        72, 'Nombre après le nom : head num'),
        ('np_amod',     '',   'man','{head}{man}',         70, 'Adjectif épithète : head+man'),
        ('np_plural',   '',   'w',  '{head}w',             75, 'Pluriel : head+w'),
    ]
    for (name,poss_bm,marker,template,prio,desc) in rules:
        db.query('''
        MERGE (r:NominalChainRule {name:$name, lang:$lang})
        SET r.poss_trigger=$poss, r.gen_marker=$marker,
            r.template=$tpl, r.priority=$prio, r.description=$desc
        ''', {'name':name,'lang':lang,'poss':poss_bm,'marker':marker,
              'tpl':template,'prio':prio,'desc':desc})
    print(f'   {len(rules)} NominalChainRules')


def ingest_possession_rules(db, lang='bm'):
    """Règles de possession (avoir) : MATERIAL, ABSTRACT, EXPERIENCER, STATIF, AGE."""
    rules = [
        # (name, possession_type, template_pos, template_neg, tam_pos, tam_neg, priority, desc)
        ('poss_material', 'MATERIAL',
         '{O} {TAM} {S} bóló',  '{O} {TAM} {S} bóló',
         'bɛ', 'tɛ', 85,
         'Possession matérielle → O bɛ S bóló (il a une voiture)'),

        ('poss_abstract', 'ABSTRACT',
         '{O} {TAM} {S} fɛ',    '{O} {TAM} {S} fɛ',
         'bɛ', 'tɛ', 85,
         'Possession abstraite → O bɛ S fɛ (il a un ami)'),

        ('poss_experiencer', 'EXPERIENCER',
         '{O} {TAM} {S} la',    '{O} {TAM} {S} la',
         'bɛ', 'tɛ', 86,
         'Expérienceur physique → O bɛ S la (il a faim → kɔngɔ bɛ a la)'),

        ('poss_experiencer_past', 'EXPERIENCER_PAST',
         '{O} yé {S} minɛ',     '{O} ma {S} minɛ',
         'yé', 'ma', 87,
         'Expérienceur passé → O yé S minɛ (il a eu faim)'),

        ('poss_statif', 'STATIF',
         '{S} {V}len dòn',      '{S} {V}len tɛ',
         'dòn', 'tɛ', 88,
         'Statif émotionnel → S V+len dòn (il a peur → a jàpapalen dòn)'),

        ('poss_statif_hab', 'STATIF_HAB',
         'tùn {S} {V}len dòn',  'tùn {S} {V}len tɛ',
         'tùn dòn', 'tùn tɛ', 88,
         'Statif émotionnel habituel → tùn S V+len dòn'),

        ('poss_age', 'AGE',
         '{S} {TAM} saan {O} la', '{S} {TAM} saan {O} la',
         'bɛ', 'tɛ', 85,
         'Âge → S bɛ saan X la (il a 20 ans)'),

        ('poss_pain', 'EXPERIENCER_PAIN',
         '{O} {TAM} {S} {pain}', '{O} {TAM} {S} {pain}',
         'bɛ', 'tɛ', 89,
         'Douleur → SUBJ_POSS+BODY bɛ SUBJ PAIN (il a mal à la tête)'),
    ]
    for (name,ptype,tpl_pos,tpl_neg,tam_pos,tam_neg,prio,desc) in rules:
        db.query('''
        MERGE (r:PossessionRule {possession_type:$ptype, lang:$lang})
        SET r.name=$name, r.template_pos=$tpos, r.template_neg=$tneg,
            r.tam_positive=$tamp, r.tam_negative=$tamn,
            r.priority=$prio, r.description=$desc
        ''', {'name':name,'lang':lang,'ptype':ptype,'tpos':tpl_pos,'tneg':tpl_neg,
              'tamp':tam_pos,'tamn':tam_neg,'prio':prio,'desc':desc})
    print(f'   {len(rules)} PossessionRules')


def ingest_aux_temporal_rules(db, lang='bm'):
    """Temps composés : passé composé (avoir/être + V), futur proche (aller + V)."""
    rules = [
        ('passe_compose_avoir_trans',
         {'aux_lemma':'avoir','participe_past':'true','is_transitive':'true'},
         'yé', '', 'S yé O V', 82,
         'Passé composé avoir transitif → S yé O V (il a mangé le riz)'),

        ('passe_compose_avoir_intrans',
         {'aux_lemma':'avoir','participe_past':'true','is_transitive':'false'},
         'yé', '', 'S yé O sɔrɔ', 80,
         'Passé composé avoir intransitif → S yé O sɔrɔ'),

        ('passe_compose_etre',
         {'aux_lemma':'être','participe_past':'true','verb_is_motion':'true'},
         '', 'ra', 'S V+ra', 83,
         'Passé composé être + mouvement → S V+ra (il est parti → a fáɲira)'),

        ('futur_proche',
         {'root_lemma':'aller','xcomp_pos':'VERB','root_tense':'pres'},
         'bɛ́nà', '', 'S bɛ́nà O V2', 85,
         'Futur proche → S bɛ́nà V2 (il va manger → a bɛ́nà dúnli kɛ)'),

        ('venir_de_recent_past',
         {'root_lemma':'venir','mark_surface':'de','xcomp_pos':'VERB'},
         '', 'ra', 'S bɔra ka V2', 85,
         'Passé récent venir de → S bɔra ka V2 (il vient de manger → a bɔra ka dúnli kɛ)'),

        ('venir_de_source',
         {'root_lemma':'venir','mark_role':'genitive','obl_type':'NOUN'},
         '', '', 'S bɛ bɔ OBL la', 83,
         'Venir de + lieu → S bɛ bɔ X la (je viens de l école → n bɛ bɔ kàlankɛyɔrɔ la)'),

        ('plus_que_parfait',
         {'aux_tense':'hab','participe_past':'true'},
         'tùn yé', 'tùn ma', 'S tùn yé O V', 84,
         'Plus-que-parfait → S tùn yé V (il avait mangé)'),
    ]
    for (name,conds,tam_pos,sfx,template,prio,desc) in rules:
        params = {'name':name,'lang':lang,'tamp':tam_pos,'sfx':sfx,
                  'tpl':template,'prio':prio,'desc':desc}
        set_parts = ['r.tam_positive=$tamp','r.verb_suffix=$sfx',
                     'r.template=$tpl','r.priority=$prio','r.description=$desc']
        for k,v in conds.items():
            params[f'c_{k}'] = v
            set_parts.append(f'r.cond_{k}=$c_{k}')
        db.query(f'MERGE (r:AuxTemporalRule {{name:$name,lang:$lang}}) SET '+', '.join(set_parts), params)
    print(f'   {len(rules)} AuxTemporalRules')


def ingest_advcl_rules(db, lang='bm'):
    """Subordonnées adverbiales : purposive (pour), privative (sans), conditional (si)."""
    rules = [
        ('advcl_purposive',  'purposive',  'walasa ka', '{V}',          72, 'Purposif → walasa ka V'),
        ('advcl_privative',  'privative',  'bali',      '{V}li kɛ bali',70, 'Privatif sans → Vli kɛ bali'),
        ('advcl_conditional','conditional','ni',         'ni {S} {TAM} {V}', 74,'Conditionnel si → ni S TAM V'),
        ('advcl_concessive', 'concessive', 'nìn fɔ',    'nìn fɔ {S} {TAM} {V}', 70,'Concessif → nìn fɔ...'),
        ('advcl_temporal',   'temporal',   '',          '{S} {TAM} {V}', 68,'Temporel quand → S TAM V'),
    ]
    for (name,role,marker,template,prio,desc) in rules:
        db.query('''
        MERGE (r:Advcl_Rule {name:$name, lang:$lang})
        SET r.advcl_role=$role, r.marker=$marker,
            r.template=$tpl, r.priority=$prio, r.description=$desc
        ''', {'name':name,'lang':lang,'role':role,'marker':marker,
              'tpl':template,'prio':prio,'desc':desc})
    print(f'   {len(rules)} AdvcaRules')


def load_into_grammar(db, lang='bm'):
    """Charge les nouvelles règles dans _load_grammar via index G_kg."""
    # Test de lecture
    results = {}
    for label in ['CoordRule','ModalRule','ObliqueFillRule','NominalChainRule',
                  'AuxTemporalRule','PossessionRule','Advcl_Rule']:
        res = db.query(f"MATCH (n:{label} {{lang:$lang}}) RETURN count(n) AS c", {'lang':lang})
        results[label] = res[0]['c'] if res else 0
    return results


def run():
    db = Neo4jClient()
    create_indexes(db)
    print(f"[Migration complète vers KG]")
    ingest_coord_rules(db)
    ingest_modal_rules(db)
    ingest_oblique_fill_rules(db)
    ingest_nominal_chain_rules(db)
    ingest_possession_rules(db)
    ingest_aux_temporal_rules(db)
    ingest_advcl_rules(db)

    counts = load_into_grammar(db)
    print(f"\n[Résumé KG complet]")
    for label, c in counts.items():
        print(f"  {label}: {c} nœuds")

    total = db.query("MATCH (n) WHERE n.lang='bm' RETURN count(n) AS c")
    print(f"\n  Total nœuds lang=bm: {total[0]['c']}")
    print("\n Migration complète terminée.")


if __name__ == '__main__':
    run()
