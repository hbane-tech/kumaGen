# Syntactic Tree Comparison Guide

## Overview

The system now exports POS (Part-of-Speech) trees for comparing syntactic structures between:
- **kuma_mt**: Our French-to-Bambara translation system
- **Google Translate**: Baseline 1
- **NLLB**: Baseline 2 (No Language Left Behind)

## Quick Start

### 1. Generate Test Results with POS Trees

```bash
python test_phrases.py --run
```

This creates a CSV file like `resultats_bambara_20260614_xxxxxx.csv` with columns:
- `phrase_fr`: French input
- `french_pos_tree`: POS sequence of French input (PRON VERB DET NOUN...)
- `bambara_obtenu`: Our system's translation
- `bambara_pos_tree`: POS sequence of our output
- `statut`: PASS/FAIL status

### 2. Analyze Your Results

```bash
python compare_translation_trees.py resultats_bambara_20260614_xxxxxx.csv
```

This shows:
- Individual tree comparisons
- Tree similarity scores (0-1 scale)
- Depth changes (token count differences)
- Complexity metrics

## Extended Comparison with Baselines

To compare all three systems, follow these steps:

### Step 1: Install Dependencies

```bash
# For NLLB model
pip install transformers torch

# For Google Translate
pip install google-cloud-translate

# For POS tagging
pip install spacy
python -m spacy download xx_sent_ud_sm  # Multilingual model
```

### Step 2: Create Extended CSV with Baselines

```python
# compare_baselines.py
import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from google.cloud import translate
import spacy

# Load models
nllb_model = ...
spacy_nlp = spacy.load('xx_sent_ud_sm')

# For each phrase in test_phrases.py:
# 1. Get kuma_mt translation (already in CSV)
# 2. Call Google Translate API
# 3. Call NLLB model
# 4. Extract POS from each output
# 5. Add columns to CSV
```

### Step 3: Compare Tree Structures

The comparison metrics include:

#### 1. **Tree Similarity** (Sequence Matcher)
- Measures how much POS sequence overlap exists
- 100% = identical tree structure
- 0% = completely different structure

```
Example:
French:  PRON VERB DET NOUN ADJ
Kuma:    PRON AUX VERB DET NOUN
Google:  PRON VERB DET NOUN
NLLB:    PRON VERB ADJ NOUN DET

Kuma vs Google similarity: 80%
NLLB vs Google similarity: 60%
```

#### 2. **Tree Depth** (Token Count)
- Indicates expansion/compression during translation
- Positive = more tokens than input
- Negative = fewer tokens than input

```
French:     5 tokens
Kuma:       6 tokens (+1 expansion)
Google:     5 tokens (same)
NLLB:       4 tokens (-1 compression)
```

#### 3. **POS Distribution**
- Which POS tags appear most frequently
- Indicates structural complexity

```
French:  NOUN:3, VERB:2, DET:1, ADJ:1
Kuma:    NOUN:3, VERB:2, DET:2, AUX:1
Google:  NOUN:3, VERB:2, DET:1, ADJ:1
NLLB:    NOUN:4, VERB:1, ADJ:1
```

## Interpreting Results

### High Similarity (>80%)
- System follows similar syntactic structure to input
- Good for morphologically similar languages
- May indicate word-by-word translation

### Low Similarity (<40%)
- System restructures significantly
- May indicate:
  - Reordering (SOV vs SVO)
  - Significant grammar adaptation
  - Potential quality issues

### Depth Changes
- **+2 to +5**: Normal expansion (adding grammatical markers)
- **+10+**: Potential over-expansion (adding unnecessary words)
- **-3 or more**: Potential under-compression (dropping important elements)

## Example Analysis

```
Rule 6: ne...que restrictive
========================================
Input:    tu ne serais qu'un pleutre
French:   PRON AUX VERB DET NOUN
Kuma:     PRON AUX VERB ADV VERB DET NOUN
Status:   PASS ✅

Tree Analysis:
- Similarity:     60% (restructured for 'foyi yé')
- Depth change:   +2 (added ADV for 'foyi')
- Complexity:     Kuma adds more verbs (verbal noun structure)
- Interpretation: Correct structure, expected expansion
```

## Integration with Evaluation

Use tree comparison for:

1. **Quality Analysis**: Correlate tree similarity with translation quality
2. **Rule Coverage**: Which rules cause which structural changes?
3. **Baseline Comparison**: How does kuma_mt compare to Google/NLLB?
4. **Error Analysis**: Do FAIL cases have unusual tree patterns?

## Next Steps

1. Implement baseline API calls (see `compare_baselines.py` template)
2. Generate extended CSV with all three systems
3. Run statistical analysis on tree similarities
4. Visualize tree comparisons (dendrograms, heatmaps)
5. Correlate tree patterns with PASS/FAIL status

## Files

- `test_phrases.py`: Main test runner (now with POS extraction)
- `compare_translation_trees.py`: Tree analysis script
- `compare_baselines.py`: Template for baseline integration
- `resultats_bambara_*.csv`: Test results with POS trees
