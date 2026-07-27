"""
kg/ingest_semantic_classes.py
Ingère toutes les classes sémantiques dans le KG SemanticClass.
"""
from kg.neo4j_client import Neo4jClient


SEMANTIC_CLASSES = [
    # (name, description, transitivity, is_intransitive, bambara_behavior)
    # ── CLASSES INTRANSITIVES (INTRANS_SC) ───────────────────────────────────
    ('motion',
     'Deplacement dans espace (aller, venir, partir, courir, monter)',
     'intransitive', True,
     'past→resultative V+sfx; relcl→resultative; serial→motion+but'),

    ('biological',
     'Processus vital ou transition etat corporel (naitre, mourir, se reveiller, lever)',
     'intransitive', True,
     'reflexive→pronominal (verbe nu); past→resultative'),

    ('posture',
     'Configuration spatiale stable du corps (asseoir, coucher, pencher)',
     'intransitive', True,
     'reflexive→posture (a V); se lever→pronominal si causatif'),

    ('spontaneous',
     'Reaction involontaire ou evenement subi (rire, pleurer, tousser, se blesser)',
     'intransitive', True,
     'reflexive→accidentel (a yere V)'),

    ('meteorological',
     'Phenomene atmospherique impersonnel (pleuvoir, neiger, tonner, venter)',
     'intransitive', True,
     'construction: PHENOMENON TAM motion_verb (san be na)'),

    ('copula',
     'Lien attributif ou equatif (etre, sembler, paraitre, devenir)',
     'intransitive', True,
     'equative: S ye O ye; locatif: S be LOC'),

    ('stative_cognitive',
     'Etat mental statif (savoir, croire, penser, connaitre, ignorer)',
     'intransitive', True,
     'bare verb; avec ccomp→ko clause; jamais Vli ke'),

    ('sound',
     'Emission sonore (chanter, crier, siffler, murmurer)',
     'intransitive', True,
     'bare verb; avec objet sonore possible'),

    ('emission',
     'Emission de lumiere ou de rayonnement (briller, rayonner, luire)',
     'intransitive', True,
     'bare verb; jamais Vli ke'),

    ('saying',
     'Parole verbale (dire, raconter, annoncer, expliquer, repondre)',
     'transitive', True,
     'dative yé (parler à); ccomp avec ko; bare verb possible'),

    ('communication',
     'Echange communicatif (parler, discuter, telephoner, correspondre)',
     'transitive', True,
     'dative yé; no-obj→V la ou V ke; avec objet→V nu'),

    # ── CLASSES TRANSITIVES ──────────────────────────────────────────────────
    ('perception',
     'Reception sensorielle directe (voir, entendre, sentir, regarder)',
     'transitive', False,
     'reflexive→actif+yere; no-obj→Vli ke; with-obj→V nu'),

    ('psych_emotion',
     'Etat psychologique ou emotionnel avec objet (aimer, craindre, admirer)',
     'transitive', False,
     'statif emotionnel; objet possible; statif_len_don sans objet'),

    ('action',
     'Action intentionnelle generique (faire, donner, prendre, aider, montrer)',
     'transitive', False,
     'present→V la; past→V ke; no-obj→Vli ke; with-obj→V nu'),

    ('consumption',
     'Ingestion solide uniquement (manger, croquer, devorer, avaler solide)',
     'transitive', False,
     'no-obj→action_noun ke (dunli ke); with-obj→V nu (dun)'),

    ('consumption_liquid',
     'Ingestion de liquide (boire, siroter; jamais consumption pour boire)',
     'transitive', False,
     'no-obj→V direct sans nominalisation; with-obj→OBJ V'),

    ('preparation',
     'Transformation ou creation physique (cuisiner, preparer, fabriquer, confectionner)',
     'transitive', False,
     'no-obj→Vli ke; with-obj→V nu'),

    ('technique',
     'Travail specialise ou professionnel (construire, reparer, programmer, operer)',
     'transitive', False,
     'no-obj→V ke (pattern nom-action); with-obj→V nu'),

    ('craft',
     'Creation artistique ou artisanale (peindre, ecrire, sculpter, composer)',
     'transitive', False,
     'no-obj→Vli ke; with-obj→V nu'),

    ('having',
     'Possession ou etat de possession (avoir, posseder, detenir, contenir)',
     'transitive', True,
     'construction possession: O be S fe/bolo; sensation: STATE be SUBJ la'),

    ('modal',
     'Verbe modal de capacite ou volonte (pouvoir, vouloir, savoir+inf)',
     'transitive', False,
     'serial: S TAM V ka V2; pouvoir→se ka V'),

    ('obligation',
     'Obligation ou necessite (devoir, falloir, etre oblige)',
     'transitive', False,
     'serial ou impersonnel; falloir→impersonnel'),

    # ── CLASSES SUPPLEMENTAIRES ───────────────────────────────────────────────
    ('being',
     'Etat d existence ou d identite (exister, etre present, se trouver)',
     'intransitive', True,
     'locatif ou existentiel; be yan'),

    ('giving',
     'Transfert a quelqu un (donner, offrir, envoyer, remettre)',
     'transitive', False,
     'S TAM O V yé (dative); double objet possible'),

    ('other',
     'Classe par defaut - aucune categorie ne convient mieux',
     'transitive', False,
     'fallback general; V la au present'),
]


def run():
    db = Neo4jClient()
    db.query("CREATE INDEX semantic_class_name IF NOT EXISTS FOR (n:SemanticClass) ON (n.name)")

    count = 0
    for name, desc, trans, intrans, bambara in SEMANTIC_CLASSES:
        db.query(
            "MERGE (s:SemanticClass {name: $name}) "
            "SET s.description = $desc, "
            "    s.transitivity = $trans, "
            "    s.is_intransitive = $intrans, "
            "    s.bambara_behavior = $bambara, "
            "    s.lang = 'fr'",
            {'name': name, 'desc': desc, 'trans': trans,
             'intrans': intrans, 'bambara': bambara}
        )
        count += 1

    res = db.query("MATCH (n:SemanticClass) RETURN count(n) AS c")
    total = res[0]['c'] if res else 0
    print(f" {count} classes ingérées → {total} nœuds SemanticClass dans le KG")
    print()

    res2 = db.query(
        "MATCH (n:SemanticClass) "
        "RETURN n.name AS name, n.is_intransitive AS intrans, "
        "n.transitivity AS t "
        "ORDER BY n.is_intransitive DESC, n.name"
    )
    print("Classes sémantiques complètes :")
    print(f"  {'Nom':<22} {'Transitivité':<14} {'Dans INTRANS_SC'}")
    print(f"  {'-'*22} {'-'*14} {'-'*15}")
    for r in res2:
        marker = "OUI" if r['intrans'] else "non"
        print(f"  {r['name']:<22} {r['t']:<14} {marker}")


if __name__ == '__main__':
    run()
