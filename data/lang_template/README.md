# CSV Templates — Adaptation à une nouvelle langue cible

Remplacer `XX` par le code ISO 639-3 de la langue cible (ex: `wo`=Wolof, `ff`=Fulfulde, `ha`=Hausa).
Remplacer tous les `REMPLIR` par les formes dans la langue cible.

## Ordre de remplissage recommandé

| # | Fichier | Priorité | Pourquoi |
|---|---------|----------|---------|
| 1 | `1_tam_config.csv` | CRITIQUE | Aucune phrase sans TAM |
| 2 | `5_pronouns.csv` | CRITIQUE | Sujets/objets de toutes les clauses |
| 3 | `4_grammar_markers.csv` | CRITIQUE | Marqueurs structurels |
| 4 | `7_lexicon.csv` | HAUTE | Vocabulaire de base |
| 5 | `2_morpho_rules.csv` | HAUTE | Formes fléchies |
| 6 | `3_clause_templates.csv` | HAUTE | Construction des phrases |
| 7 | `6_semantic_behaviors.csv` | MOYENNE | Comportements fins |
| 8 | `8_prepositions.csv` | MOYENNE | Marqueurs casuels |
| 9 | `9_negation_markers.csv` | BASSE | Fixe (source française) |

## Ingestion dans le KG

```bash
# 1. Copier les CSV remplis dans data/
cp data/lang_template/*.csv data/

# 2. Ingérer les données structurelles
python -m kg.ingest_grammar

# 3. Ingérer les règles de traduction
python -m kg.ingest_rules --lang XX

# 4. Ingérer le lexique (via ingest_sense.py ou script dédié)
python -m kg.ingest_lexicon --lang XX --file data/7_lexicon.csv
```

## Questions clés à répondre pour chaque langue

### Ordre des mots
- SOV (Sujet-Objet-Verbe) ? SVO ? VSO ? autre ?
- L'objet précède-t-il le verbe ? → `clause_templates.csv`

### Système TAM
- Le TAM est-il un préfixe, suffixe, ou mot séparé ?
- Combien de temps/aspects distincts ?
- → `1_tam_config.csv`

### Morphologie verbale
- Le passé intransitif a-t-il une forme spéciale (résultatif) ?
- Les états (passif, émotion) ont-ils une forme participiale (-len dòn en bambara) ?
- → `2_morpho_rules.csv`

### Possession
- Comment dit-on "le livre de Pierre" ? Marqueur génitif ?
- Possession matérielle ≠ abstraite ?
- → `4_grammar_markers.csv`

### Réflexif
- Comment "se laver", "se voir" sont-ils exprimés ?
- Verbe nu / répétition du pronom / marqueur spécial ?
- → `6_semantic_behaviors.csv`
