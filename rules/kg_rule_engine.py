"""
rules/kg_rule_engine.py
Moteur de traversée générique du graphe de règles KG.

Au lieu de règles hardcodées Python, on extrait les features de la phrase
et on traverses les chemins FeaturePattern → ConstructionRule → ClauseTemplate.

Pour ajouter une langue : ingérer les règles dans le KG avec --lang XX.
Ce fichier Python ne change pas.
"""

from rules.core import j


_TRAVERSAL_QUERY = """
MATCH (fp:FeaturePattern {lang: $lang})-[:TRIGGERS]->(cr:ConstructionRule {lang: $lang})
WHERE fp.feature IN $feature_keys
  AND fp.value IN $feature_values
  AND (fp.context IS NULL OR fp.context IN $contexts)
WITH cr ORDER BY cr.priority DESC
OPTIONAL MATCH (cr)-[:USES_TEMPLATE]->(ct:ClauseTemplate {lang: $lang})
OPTIONAL MATCH (cr)-[:APPLIES_MORPHO]->(mr:MorphoRule {lang: $lang})
RETURN cr.name          AS rule_name,
       cr.priority       AS priority,
       cr.description    AS description,
       cr.refl_treatment AS refl_treatment,
       cr.motion_verb    AS motion_verb,
       cr.prefix         AS prefix,
       cr.tam_value      AS tam_value,
       cr.preserves_clause_type AS preserves_ct,
       ct.template       AS template,
       ct.word_order     AS word_order,
       ct.pos_form       AS pos_form,
       ct.neg_form       AS neg_form,
       ct.hab_form       AS hab_form,
       ct.past_form      AS past_form,
       mr.name           AS morpho_name,
       mr.suffix_default AS morpho_sfx,
       mr.suffix_after_n AS morpho_sfx_n,
       mr.suffix_after_o_u_o AS morpho_sfx_ou,
       mr.pos_support    AS morpho_pos_sup,
       mr.neg_support    AS morpho_neg_sup
LIMIT 1
"""


def extract_features(tree: dict, root_tok: dict) -> dict:
    """Extrait les features linguistiques de la phrase pour la traversée KG."""
    features = {}

    if root_tok:
        if root_tok.get('semantic_class'):
            features['semantic_class'] = root_tok['semantic_class']
        if root_tok.get('tense'):
            features['tense'] = root_tok['tense']
        if root_tok.get('is_passive'):
            features['is_passive'] = 'true'
        if root_tok.get('intransitive_type'):
            features['intransitive_type'] = root_tok['intransitive_type']

    if tree:
        ct = tree.get('clause_type')
        if ct:
            features['clause_type'] = ct
        if tree.get('possession_type'):
            features['possession_type'] = tree['possession_type']

    return features


def match_rules(features: dict, contexts: list, db, lang: str = 'bm') -> list:
    """Traversée KG : features → règles applicables (triées par priorité)."""
    if not features or not db:
        return []

    feature_keys   = list(features.keys())
    feature_values = [str(v) for v in features.values()]

    results = db.query(_TRAVERSAL_QUERY, {
        'lang': lang,
        'feature_keys':   feature_keys,
        'feature_values': feature_values,
        'contexts':       contexts or [''],
    })
    return results or []


def get_best_rule(features: dict, contexts: list, db, lang: str = 'bm') -> dict:
    """Retourne la règle de plus haute priorité applicable."""
    rules = match_rules(features, contexts, db, lang)
    return rules[0] if rules else {}


def apply_morpho_suffix(bm: str, rule: dict) -> str:
    """Applique le suffixe résultatif depuis la règle KG.
    Toutes les valeurs viennent du KG — zéro hardcode Python.
    KG fournit : trigger_n, trigger_vowel, suffix_after_n, suffix_after_vowel, suffix_default.
    """
    trigger_n      = rule['trigger_n']           # KG: 'n'
    trigger_vowels = set(rule['trigger_vowel'].split('|'))  # KG: 'o|u|ɔ' → {'o','u','ɔ'}
    sfx_n          = rule['suffix_after_n']      # KG: 'na'
    sfx_vowel      = rule['suffix_after_vowel']  # KG: 'la'
    sfx_default    = rule['suffix_default']      # KG: 'ra'

    # Guard minimal : seulement 'ra' double (le plus commun)
    if bm.endswith(sfx_default) and not bm.endswith(trigger_n + sfx_default):
        return bm
    # Verbes en Cnu (màkúnu, kanu…) : le 'n' avant 'u' est le vrai trigger → 'na'
    if (len(bm) >= 2 and bm[-1] in trigger_vowels and bm[-2] == trigger_n):
        return bm + sfx_n
    if bm.endswith(trigger_n):
        return bm + sfx_n
    if bm and bm[-1] in trigger_vowels:
        return bm + sfx_vowel
    return bm + sfx_default


def apply_statif_morpho(bm: str, rule: dict, neg: bool = False,
                        tense: str = 'pres') -> str:
    """Construction statif : V+len dòn / V+len tɛ / tùn V+len dòn.
    Toutes les valeurs viennent du KG : suffix, pos_support, neg_support, hab_prefix.
    """
    sfx      = rule['suffix']      # KG: 'len'
    pos_sup  = rule['pos_support'] # KG: 'dòn'
    neg_sup  = rule['neg_support'] # KG: 'tɛ'
    hab_pfx  = rule['hab_prefix']  # KG: 'tùn'

    v_len   = bm + sfx
    support = neg_sup if neg else pos_sup

    if tense == 'hab':
        return j(hab_pfx, v_len, support)
    return j(v_len, support)


def fill_template(template: str, slots: dict) -> str:
    """Remplit un template ClauseTemplate avec les slots S/TAM/O/V/ADV."""
    result = template
    for key, val in slots.items():
        result = result.replace('{' + key + '}', val or '')
    # Nettoyer les slots non remplis
    import re
    result = re.sub(r'\{[^}]+\}', '', result)
    return j(*result.split())


# ── Interface publique utilisée par les steps ─────────────────────────────────

def lookup_rule(feature: str, value: str, G_kg: dict,
                context: str = '') -> dict:
    """Lookup rapide depuis le cache G_kg (pas de requête Neo4j au moment du rendu).
    Utilise g['kg_rules'] pré-chargé par _load_grammar().
    """
    rules = G_kg.get('kg_rules', {})
    # Essai avec context d'abord
    if context:
        r = rules.get((feature, str(value), context))
        if r:
            return r
    # Sans context
    return rules.get((feature, str(value)), {})


def lookup_clause_template(clause_type: str, G_kg: dict) -> str:
    """Retourne le template d'un clause_type depuis le KG (cache pré-chargé)."""
    return G_kg.get('clause_templates', {}).get(clause_type, '')


def resolve_refl_treatment(semantic_class: str, db, lang: str = 'bm') -> str:
    """Retourne le traitement réflexif depuis le KG (pronominal/posture/actif/accidentel)."""
    features = {'semantic_class': semantic_class}
    rule = get_best_rule(features, ['reflexive'], db, lang)
    return rule.get('refl_treatment', '') if rule else ''


def resolve_construction(features: dict, db, lang: str = 'bm') -> dict:
    """Retourne la construction complète (template + morpho) depuis le KG."""
    return get_best_rule(features, [''], db, lang)


def demo_traversal(db, lang='bm'):
    """Démo : traversée pour quelques cas."""
    print(f"\n[Démo traversée KG pour lang={lang!r}]")

    test_cases = [
        ({'is_passive': 'true'}, [], 'Passif (est bloqué)'),
        ({'semantic_class': 'biological'}, ['reflexive'], 'Biologique réflexif (se réveiller)'),
        ({'tense': 'past', 'intransitive_type': 'ABSOLU'}, [], 'Passé intransitif (partir)'),
        ({'semantic_class': 'meteorological'}, [], 'Météorologique (il pleut)'),
        ({'possession_type': 'STATIF'}, [], 'État émotionnel (avoir peur)'),
        ({'imperative_person': '2pl'}, [], 'Impératif 2e plur (Aidez-moi)'),
    ]

    for features, contexts, desc in test_cases:
        rule = get_best_rule(features, contexts, db, lang)
        if rule:
            print(f"  {desc}:")
            print(f"    → Règle: {rule.get('rule_name')} (prio={rule.get('priority')})")
            print(f"    → Template: {rule.get('template') or 'N/A'}")
            print(f"    → Morpho: {rule.get('morpho_name') or 'N/A'}")
        else:
            print(f"  {desc}: [aucune règle]")
