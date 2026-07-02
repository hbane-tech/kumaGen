"""
kg/ingest_remaining_rules.py
Migration des règles restantes vers le KG :
  participes.py, participial_to.py, etre_root.py
  advcl.py, privatif.py, relcl_boucle.py, step_impersonal.py
"""
from kg.neo4j_client import Neo4jClient


def run():
    db = Neo4jClient()
    lang = 'bm'
    count = 0

    # ── participes.py ──────────────────────────────────────────────────────
    pattern_rules = [
        ('adj_participe_passe',
         {'root_pos':'ADJ', 'root_is_participe_passe':'true', 'has_cop':'true'},
         'simple', 76,
         'ADJ participe passé + cop → résultatif simple'),
        ('etre_root_equative',
         {'root_lemma':'être', 'has_nsubj':'true'},
         'equative', 82,
         'être ROOT + sujet → équatif'),
    ]
    for name, conds, ct, prio, desc in pattern_rules:
        params = {'name': name, 'lang': lang, 'ct': ct, 'prio': prio, 'desc': desc}
        sets = ['r.clause_type=$ct', 'r.priority=$prio', 'r.description=$desc']
        for k, v in conds.items():
            params[f'c{k}'] = v
            sets.append(f'r.cond_{k}=$c{k}')
        db.query(f'MERGE (r:PatternRule {{name:$name, lang:$lang}}) SET {", ".join(sets)}', params)
        count += 1

    # ── participial_to.py ──────────────────────────────────────────────────
    db.query("""
    MERGE (r:MorphoRule {name:'participial_to', lang:$lang})
    SET r.suffix='tɔ',
        r.condition='advcl_gerund',
        r.description='Gérondif participial : en mangeant → dúnùntɔ'
    """, {'lang': lang})
    count += 1

    # ── privatif.py ────────────────────────────────────────────────────────
    privative = [
        ('priv_noun', 'NOUN', 'tan',  'S TAM O tan',  78, 'Privatif NOUN → tan'),
        ('priv_verb', 'VERB', 'bali', 'S TAM V bali', 78, 'Privatif VERB → bali'),
        ('priv_pron', 'PRON', 'kɔ',  'S TAM O kɔ',   78, 'Privatif PRON → kɔ'),
    ]
    for name, pos, marker, tpl, prio, desc in privative:
        db.query("""
        MERGE (r:PrivativeRule {name:$name, lang:$lang})
        SET r.match_pos=$pos, r.marker=$marker,
            r.template=$tpl, r.priority=$prio, r.description=$desc
        """, {'name': name, 'lang': lang, 'pos': pos, 'marker': marker,
              'tpl': tpl, 'prio': prio, 'desc': desc})
        count += 1

    # ── advcl.py ───────────────────────────────────────────────────────────
    advcl = [
        ('advcl_absolu',      'ABSOLU',      '',                   'V',       68),
        ('advcl_intrans_sc',  '',            'motion_biological',  'V',       68),
        ('advcl_liquid',      '',            'consumption_liquid', 'V',       68),
        ('advcl_action_pres', 'ACTION',      'action',             'V la',    68),
        ('advcl_nominalized', 'nominalized', '',                   'V li kɛ', 68),
    ]
    for name, intr, sc, tpl, prio in advcl:
        db.query("""
        MERGE (r:AdvcaRule {name:$name, lang:$lang})
        SET r.intransitive_type=$intr,
            r.semantic_class=$sc,
            r.template=$tpl,
            r.priority=$prio
        """, {'name': name, 'lang': lang, 'intr': intr, 'sc': sc,
              'tpl': tpl, 'prio': prio})
        count += 1

    # ── step_impersonal.py ─────────────────────────────────────────────────
    impersonal = [
        ('imp_falloir_inf',   'falloir', 'VERB', '', 'ka {V2}',          85, 'il faut + inf → ka V'),
        ('imp_falloir_que',   'falloir', '',  'ccomp', 'ko {S2} {TAM} {V2}', 85, 'il faut que → ko S V'),
        ('imp_sagit_de',      'agir',    '', 'de_mark', '{O} ko',         80, 'il s agit de → X ko'),
        ('imp_arriver_que',   'arriver', '', 'ccomp',   'tuma dow la {S2} {TAM} {V2}', 75, 'il arrive que'),
        ('imp_sembler_que',   'sembler', '', 'ccomp',   'a kàn ko {S2} {TAM} {V2}', 75, 'il semble que'),
        ('imp_manquer',       'manquer', '', '',        '{O} ma sɔrɔ',   78, 'il manque X → O ma sɔrɔ'),
        ('imp_rester',        'rester',  '', '',        '{O} tùn bɛ',    78, 'il reste X → O tùn bɛ'),
        ('imp_exister',       'exister', '', '',        '{O} bɛ yan',    78, 'il existe X → O bɛ yan'),
    ]
    for name, lemma, xcomp_pos, mark, tpl, prio, desc in impersonal:
        db.query("""
        MERGE (r:ImpersonalRule {name:$name, lang:$lang})
        SET r.trigger_lemma=$lemma,
            r.xcomp_pos=$xpos,
            r.mark_pattern=$mark,
            r.template=$tpl,
            r.priority=$prio,
            r.description=$desc
        """, {'name': name, 'lang': lang, 'lemma': lemma,
              'xpos': xcomp_pos, 'mark': mark,
              'tpl': tpl, 'prio': prio, 'desc': desc})
        count += 1

    # ── relcl_boucle.py ────────────────────────────────────────────────────
    db.query("""
    MERGE (r:GraphWalkRule {name:'relcl_oblique', lang:$lang})
    SET r.match_dep='acl:relcl',
        r.context='obl',
        r.template_transitive='mìn {TAM} {O} {V}',
        r.template_intransitive='mìn {V_resultative}',
        r.statif_form='len_don',
        r.description='Relative en position oblique'
    """, {'lang': lang})
    count += 1

    print(f'\n[Résumé] {count} règles ingérées')
    for label in ['PatternRule', 'MorphoRule', 'PrivativeRule',
                  'AdvcaRule', 'ImpersonalRule', 'GraphWalkRule']:
        res = db.query(f"MATCH (n:{label} {{lang:$lang}}) RETURN count(n) AS c",
                       {'lang': lang})
        c = res[0]['c'] if res else 0
        print(f'  {label}: {c}')


if __name__ == '__main__':
    run()
