# CANVAS — ADAPTATION KUMA_MT À UNE NOUVELLE LANGUE
# lang='bm' (bambara) → lang='XX' (nouvelle langue)
# ═══════════════════════════════════════════════════════════════════════
# 15 types de nœuds à créer/adapter + 2 réutilisables = 17 actifs total
# ═══════════════════════════════════════════════════════════════════════

## ──────────────────────────────────────────────
## RÉUTILISABLES (ne pas recréer, déjà en KG)
## ──────────────────────────────────────────────

### PatternRule (74 entrées)
  # Détection des constructions (features spaCy) — RÉUTILISABLE
  # → basé sur spaCy dep/pos/role : identique pour toute langue

### SlotFillRule (29 entrées)
  # Remplissage des slots S/O/V (features spaCy) — RÉUTILISABLE
  # → basé sur spaCy dep/pos/role : identique pour toute langue

## ──────────────────────────────────────────────
## 15 TYPES À CRÉER POUR UNE NOUVELLE LANGUE
## ──────────────────────────────────────────────

### 01. TamConfig  [TABLE]  (17 entrées bambara)
# Tableau TAM complet : (tense, neg) → marqueur
# Schéma : {lang, tense, neg, bm}
# Exemple bambara :
# MERGE (t:TamConfig {lang:'XX', tense:'cond', neg:false}) SET t.bm='???'
# MERGE (t:TamConfig {lang:'XX', tense:'cond', neg:true}) SET t.bm='???'
# MERGE (t:TamConfig {lang:'XX', tense:'fut', neg:false}) SET t.bm='???'
# MERGE (t:TamConfig {lang:'XX', tense:'fut', neg:true}) SET t.bm='???'
# MERGE (t:TamConfig {lang:'XX', tense:'hab', neg:false}) SET t.bm='???'

### 02. FunctionWord  [DATA]  (80 entrées bambara)
# Marqueurs fonctionnels : genitive, equative, coord, suffixes...
# Schéma : {name, lang, bm, role}
# Marqueurs critiques (rôles) :
# MERGE (fw:FunctionWord {name:'genitive_marker', lang:'XX'}) SET fw.bm='???'  # bm='ka'
# MERGE (fw:FunctionWord {name:'equative_marker', lang:'XX'}) SET fw.bm='???'  # bm='yé'
# MERGE (fw:FunctionWord {name:'comitative_marker', lang:'XX'}) SET fw.bm='???'  # bm='?'
# MERGE (fw:FunctionWord {name:'coord_verb', lang:'XX'}) SET fw.bm='???'  # bm='wa'
# MERGE (fw:FunctionWord {name:'coord_action_suffix', lang:'XX'}) SET fw.bm='???'  # bm='kɛ'
# MERGE (fw:FunctionWord {name:'nominalization_verb_suffix', lang:'XX'}) SET fw.bm='???'  # bm='?'
# MERGE (fw:FunctionWord {name:'plural_noun_suffix', lang:'XX'}) SET fw.bm='???'  # bm='w'
# MERGE (fw:FunctionWord {name:'relative_marker', lang:'XX'}) SET fw.bm='???'  # bm='mìn'
# MERGE (fw:FunctionWord {name:'reported_intro', lang:'XX'}) SET fw.bm='???'  # bm='ko'
# ... (80 entrées total)

### 03. MorphoRule  [DATA]  (8 entrées bambara)
# Règles morpho : resultative, statif, nominalization, plural, adj
# Schéma : {name, lang, suffix, condition, description, + props spécifiques}
# MERGE (m:MorphoRule {name:'a_strip_exclusions', lang:'XX'}) SET m.suffix='', ...  # bambara=''
# MERGE (m:MorphoRule {name:'adjective_epithet', lang:'XX'}) SET m.suffix='man', ...  # bambara='man'
# MERGE (m:MorphoRule {name:'nominalization_action', lang:'XX'}) SET m.suffix='li', ...  # bambara='li'
# MERGE (m:MorphoRule {name:'participial_ta', lang:'XX'}) SET m.suffix='ta', ...  # bambara='ta'
# MERGE (m:MorphoRule {name:'participial_to', lang:'XX'}) SET m.suffix='tɔ', ...  # bambara='tɔ'
# MERGE (m:MorphoRule {name:'plural_noun', lang:'XX'}) SET m.suffix='w', ...  # bambara='w'
# MERGE (m:MorphoRule {name:'resultative', lang:'XX'}) SET m.suffix='ra', ...  # bambara='ra'
# MERGE (m:MorphoRule {name:'statif', lang:'XX'}) SET m.suffix='len', ...  # bambara='len'

### 04. ClauseTemplate  [TEMPLATE]  (161 entrées bambara)
# Ordre des mots de chaque construction
# Schéma : {lang, clause_type, template}
# !! CŒUR DE L'ADAPTATION : encoder l'ordre des mots de la langue cible !!
# MERGE (t:ClauseTemplate {clause_type:'content_question_default', lang:'XX'}) SET t.template='???'  # bm: '{S} {TAM} {O} {V} {ADV} ?'
# MERGE (t:ClauseTemplate {clause_type:'equative_neg', lang:'XX'}) SET t.template='???'  # bm: '{S} tɛ {O} yé'
# MERGE (t:ClauseTemplate {clause_type:'equative_pos', lang:'XX'}) SET t.template='???'  # bm: '{S} yé {O} yé'
# MERGE (t:ClauseTemplate {clause_type:'existential_absolute', lang:'XX'}) SET t.template='???'  # bm: '{O} {TAM}'
# MERGE (t:ClauseTemplate {clause_type:'imperative_2sg', lang:'XX'}) SET t.template='???'  # bm: '{O} {V}'
# MERGE (t:ClauseTemplate {clause_type:'locative', lang:'XX'}) SET t.template='???'  # bm: '{S} {TAM} {OBL}'
# MERGE (t:ClauseTemplate {clause_type:'prohibitive', lang:'XX'}) SET t.template='???'  # bm: 'kàna {O} {V}'
# MERGE (t:ClauseTemplate {clause_type:'qualitative', lang:'XX'}) SET t.template='???'  # bm: '{S} {TAM} {QUAL}'
# MERGE (t:ClauseTemplate {clause_type:'serial_default', lang:'XX'}) SET t.template='???'  # bm: '{S} {TAM} {V} ka {V_ACT}'
# MERGE (t:ClauseTemplate {clause_type:'simple', lang:'XX'}) SET t.template='???'  # bm: '{S} {TAM} {O} {V} {ADV}'
# MERGE (t:ClauseTemplate {clause_type:'statif', lang:'XX'}) SET t.template='???'  # bm: '{S} {QUAL} {TAM}'
# MERGE (t:ClauseTemplate {clause_type:'verb_serial', lang:'XX'}) SET t.template='???'  # bm: '{S} {TAM} {V} ka {V_ACT}'
# ... (161 entrées total)

### 05. TransformRule  [RULE]  (40 entrées bambara)
# Morphologie sur les slots : TAM, -len, -ra, li kɛ...
# Schéma propre à TransformRule
# Entrées bambara : resultative_past_intrans, resultative_neg, statif_pres_pos, statif_pres_neg, statif_hab
# MERGE (n:TransformRule {name:'...', lang:'XX'}) SET n.prop1='???', ...

### 06. CoordRule  [RULE]  (3 entrées bambara)
# Coordination verbale et nominale
# Schéma propre à CoordRule
# Entrées bambara : coord_et_action, coord_et_infinitif, coord_ou
# MERGE (n:CoordRule {name:'...', lang:'XX'}) SET n.prop1='???', ...

### 07. ModalRule  [RULE]  (4 entrées bambara)
# Verbes modaux : pouvoir, vouloir, savoir, devoir
# Schéma propre à ModalRule
# Entrées bambara : modal_pouvoir, modal_vouloir, modal_devoir, modal_savoir
# MERGE (n:ModalRule {name:'...', lang:'XX'}) SET n.prop1='???', ...

### 08. ObliqueFillRule  [RULE]  (13 entrées bambara)
# Marqueurs obliques : locatif, datif, génitif, agent...
# Schéma propre à ObliqueFillRule
# Entrées bambara : obl_locative, obl_locative2, obl_temporal, obl_dative, obl_comitative
# MERGE (n:ObliqueFillRule {name:'...', lang:'XX'}) SET n.prop1='???', ...

### 09. NominalChainRule  [RULE]  (6 entrées bambara)
# Syntagme nominal : possessif, démo, pluriel, numéral
# Schéma propre à NominalChainRule
# Entrées bambara : np_poss_n, np_poss_other, np_demo_prefix, np_nummod, np_amod
# MERGE (n:NominalChainRule {name:'...', lang:'XX'}) SET n.prop1='???', ...

### 10. PossessionRule  [RULE]  (8 entrées bambara)
# Possession : matérielle, abstraite, âge, expérienceur
# Schéma propre à PossessionRule
# Entrées bambara : poss_material, poss_abstract, poss_experiencer, poss_experiencer_past, poss_statif
# MERGE (n:PossessionRule {name:'...', lang:'XX'}) SET n.prop1='???', ...

### 11. AuxTemporalRule  [RULE]  (6 entrées bambara)
# Auxiliaires : passé composé, futur proche, venir de...
# Schéma propre à AuxTemporalRule
# Entrées bambara : passe_compose_avoir_trans, passe_compose_etre, futur_proche, venir_de_recent_past, venir_de_source
# MERGE (n:AuxTemporalRule {name:'...', lang:'XX'}) SET n.prop1='???', ...

### 12. Advcl_Rule  [RULE]  (5 entrées bambara)
# Clauses adverbiales : purposive, privative, conditionnel...
# Schéma propre à Advcl_Rule
# Entrées bambara : advcl_purposive, advcl_privative, advcl_conditional, advcl_concessive, advcl_temporal
# MERGE (n:Advcl_Rule {name:'...', lang:'XX'}) SET n.prop1='???', ...

### 13. GraphWalkRule  [RULE]  (9 entrées bambara)
# Traversée du graphe de dépendances pour NP/obliques
# Schéma propre à GraphWalkRule
# Entrées bambara : relcl_oblique, genitive_chain_root_noun, genitive_chain_conj_noun, propn_conj_focus, propn_root_ident
# MERGE (n:GraphWalkRule {name:'...', lang:'XX'}) SET n.prop1='???', ...

### 14. PrivativeRule  [RULE]  (3 entrées bambara)
# Construction privative : NOUN/VERB/PRON sans X
# Schéma propre à PrivativeRule
# Entrées bambara : priv_noun, priv_verb, priv_pron
# MERGE (n:PrivativeRule {name:'...', lang:'XX'}) SET n.prop1='???', ...

### 15. ImpersonalRule  [RULE]  (7 entrées bambara)
# Constructions impersonnelles : falloir, manquer, rester...
# Schéma propre à ImpersonalRule
# Entrées bambara : imp_falloir_inf, imp_falloir_que, imp_sagit_de, imp_arriver_que, imp_sembler_que
# MERGE (n:ImpersonalRule {name:'...', lang:'XX'}) SET n.prop1='???', ...


## ──────────────────────────────────────────────
## ORDRE D'INGESTION RECOMMANDÉ
## ──────────────────────────────────────────────

# 1.  TamConfig         → Tableau TAM (obligatoire en premier)
# 2.  FunctionWord      → Marqueurs grammaticaux (toujours chargé)
# 3.  MorphoRule        → Règles morpho (avant TransformRule qui les référence)
# 4.  ClauseTemplate    → Ordre des mots (cœur de l'adaptation)
# 5.  TransformRule     → Transformations morphosyntaxiques
# 6.  CoordRule         → Coordination
# 7.  ModalRule         → Modaux
# 8.  ObliqueFillRule   → Obliques
# 9.  NominalChainRule  → NP
# 10. PossessionRule    → Possession
# 11. AuxTemporalRule   → Auxiliaires
# 12. Advcl_Rule        → Clauses adverbiales
# 13. GraphWalkRule     → Graphe de dépendances
# 14. PrivativeRule     → Privatif
# 15. ImpersonalRule    → Impersonnels

## ──────────────────────────────────────────────
## COMMANDES DE BASE
## ──────────────────────────────────────────────

# Créer une entrée :
# MERGE (n:NodeType {name:'rule_name', lang:'XX'}) SET n.prop='val'

# ClauseTemplate (identifié par clause_type, pas name) :
# MERGE (t:ClauseTemplate {clause_type:'simple', lang:'XX'}) SET t.template='{S} {TAM} {O} {V}'

# TamConfig (identifié par tense+neg) :
# MERGE (t:TamConfig {tense:'pres', neg:false, lang:'XX'}) SET t.bm='MARQUEUR_PRESENT'

# Copier toutes les PatternRule (réutilisables) :
# (pas besoin — PatternRule n'a pas de lang-specific content)

## ──────────────────────────────────────────────
## LEXIQUE (dictionnaire bilingue)
## 3 types de nœuds : Word, Sense, Concept
## ──────────────────────────────────────────────

### L0. Concept  [15245 nœuds]  ♻️ RÉUTILISABLE
# Concepts sémantiques universels, partagés par toutes les langues
# Propriétés : {id}
# → Ne pas recréer : les Concept bambara sont déjà dans le KG

### L1. Sense  [16630 nœuds]  ✏️ À CRÉER
# Un Sense = une traduction/acception d'un concept dans la langue cible
# Propriétés :
#   bm        → MOT dans la langue cible (ex: 'dén' pour bambara, 'child' pour anglais)
#   fr        → définition/glose en français (langue pivot)
#   en        → définition/glose en anglais (optionnel)
#   pos       → Part of speech: Adjective, Adverb, Conjunction, Copula, Interjection, Noun...
#   frame     → Cadre sémantique: ACTION, ENTITY, GENERIC, QUALITY
#   id        → identifiant unique (format S_xxxx)
#   embedding → vecteur sémantique (384 dimensions, généré par train_embeddings.py)
#
# Création d'un Sense :
# CREATE (s:Sense {
#   id: 'S_' + apoc.create.uuid(),
#   bm: 'mot_dans_langue_cible',
#   fr: 'glose en français',
#   pos: 'Noun',
#   frame: 'ENTITY'
# })
# MERGE (s)-[:MAPS_TO]->(c:Concept {id:'C_xxx'})
#
# → En pratique : utiliser le scraper (kg/retriever.py) pour importer le dictionnaire
#   ou remplir les CSV dans data/lang_template/

### L2. Word  [15273 nœuds]  ✏️ À CRÉER
# Un Word = une forme graphique dans la langue cible
# Propriétés : {text}
# Relation : (Word)-[:HAS_SENSE]->(Sense)
#
# Création :
# MERGE (w:Word {text: 'mot_dans_langue_cible'})
# MERGE (w)-[:HAS_SENSE]->(s:Sense {id: 'S_xxx'})
#
# → Généré automatiquement lors de l'import du dictionnaire

## RÉSUMÉ GLOBAL DES NŒUDS À CRÉER :
## ────────────────────────────────────────────────────────
##  Type           Nb (bm)   Description
##  TamConfig         17    Tableau TAM
##  FunctionWord      80    Marqueurs grammaticaux
##  MorphoRule         8    Règles morphologiques
##  ClauseTemplate   161    Ordre des mots
##  TransformRule     40    Transformations morphosyntaxiques
##  CoordRule          3    Coordination
##  ModalRule          4    Verbes modaux
##  ObliqueFillRule   13    Obliques
##  NominalChainRule   6    Syntagmes nominaux
##  PossessionRule     8    Possession
##  AuxTemporalRule    6    Auxiliaires temporels
##  Advcl_Rule         5    Clauses adverbiales
##  GraphWalkRule      9    Traversée graphe
##  PrivativeRule      3    Privatif
##  ImpersonalRule     7    Impersonnels
##  ─── TOTAL : 15 types de règles grammaticales ──────────
##  Sense          16630    Dictionnaire (bm → fr/en)
##  Word           15273    Formes graphiques
##  ─── TOTAL LEXIQUE : 2 types ───────────────────────────
##  GRAND TOTAL : 17 types à créer pour une nouvelle langue
##  (+ PatternRule + SlotFillRule = réutilisables)
