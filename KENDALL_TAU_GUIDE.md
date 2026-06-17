# Kendall Tau for Word Order Analysis

## What You Just Created

A word order comparison system using **Kendall tau** to check if your system outputs maintain correct Bambara word order against references.

## Current Results

```
✅ Average Kendall τ: +1.000
✅ Exact matches: 100% (6/6)
✅ Word order quality: EXCELLENT 🟢
```

**Meaning**: All outputs match the reference word order exactly. No reordering issues detected.

---

## How Kendall Tau Works for Word Order

**Kendall τ** is a rank correlation coefficient (-1 to +1):

### Interpretation

| τ Value | Word Order Quality | Meaning |
|---------|-------------------|---------|
| **+1.0** | 🟢 Perfect | Output matches reference order exactly |
| **+0.7 to +0.9** | 🟢 Excellent | Minor reordering acceptable for SOV |
| **+0.3 to +0.6** | 🟡 Good | Significant reordering, check if intentional |
| **0.0 to +0.3** | 🟡 Acceptable | Major reordering (e.g., SVO→SOV transformation) |
| **< 0.0** | 🔴 Problem | Reverse/scrambled word order |

---

## How It Compares Your System

**Process**:
1. Extract POS (Part-of-Speech) sequence from reference Bambara
2. Extract POS sequence from your system's output
3. Calculate Kendall tau between the two sequences
4. Values > +0.7 = Good word order consistency

**Example**:
```
Reference: n bɛ kà báara kɛ
POS:       PRON AUX AUX VERB NOUN
          (1)  (2)  (3)  (4)   (5)

Output:    n bɛ kà báara kɛ  
POS:       PRON AUX AUX VERB NOUN
          (1)  (2)  (3)  (4)   (5)

Kendall τ = +1.000 ✅ (identical order)
```

---

## Your Bambara Word Order Rules (Extracted)

### Basic Patterns

| Clause Type | Pattern | Example |
|-------------|---------|---------|
| **Simple** | S TAM [O] V [obliques] | n bɛ càma dún (eat rice) |
| **Conditional** | ní S TAM [O] V | ní o bɛ sìra la (if he's on way) |
| **Temporal** | tuma min S TAM [O] V | tuma min a bɛ dún (when he eats) |
| **Restrictive** | S TAM foyi yé ni ATTR tɛ | i bɛ foyi yé ni sègɛ tɛ (only coward) |
| **Question** | S TAM [O] V wà ? | o bɛ báara la wà ? (is working?) |
| **Reflexive** | S TAM S [yɛrɛ] V | a bɛ a yɛrɛ móri (hurt himself) |
| **Serial** | S TAM [bɔra ka] V_ACT | a bɛ bɔra ka táa (just went) |

### Key Markers

- **TAM**: bɛ, yé, ma, tɛ, tùn, kà, na, ra (auxiliary/tense markers)
- **Pronouns**: n, i, a, o, u (subject position)
- **Particles**: wà, dun, foyi, ni, la, le (phrase markers)
- **Verb forms**: -li, -ra, -na (verb suffixes)

---

## Using the Script

### Run the analysis:
```bash
python word_order_kendall.py
```

### Output shows:
- **Individual comparisons**: Each phrase's τ value and quality
- **Summary statistics**: Average τ, min/max range, exact match percentage
- **Interpretation**: What the numbers mean for word order quality

---

## What Kendall Tau Tells You

### For Pass/Fail Analysis
- **PASS + τ > +0.7**: ✅ Correct word order
- **PASS + τ < +0.3**: ⚠️ Acceptable reordering (check if intentional)
- **FAIL + τ > +0.7**: ⚠️ Same order as reference but different words
- **FAIL + τ < +0.3**: 🔴 Word order AND content problem

### For Rule Validation
- **Rule working correctly**: τ ≈ +1.0 (matches reference order)
- **Rule causes reordering**: τ 0.3-0.7 (acceptable for SOV rules)
- **Rule breaks word order**: τ < 0 (serious issue)

---

## Comparing with Baselines (Future)

When adding Google/NLLB comparisons:

```
Kendall τ (your output vs reference):  +0.95
Kendall τ (Google vs reference):       +0.78
Kendall τ (NLLB vs reference):         +0.82

→ Your system maintains Bambara word order BEST
```

---

## Combined Metrics for Full Analysis

### For comprehensive quality:

| Metric | What It Shows |
|--------|---------------|
| **Kendall τ** | Word order consistency (rank correlation) |
| **Sequence Similarity** | Overall structure match (0-1) |
| **Edit Distance** | Insertion/deletion count (0-1) |
| **Tree Depth** | Token expansion/compression |

**Example interpretation**:
```
τ = +0.95 (excellent order)
+ Similarity = 0.88 (good structure)
+ Edit distance = 0.10 (few changes)
= Overall PASS ✅
```

---

## Implementation Reference

Your extraction rules (from source code):

```python
TAM_MARKERS = {'bɛ', 'yé', 'ma', 'tɛ', 'tùn', 'kà', 'ka', 'na', 'ra'}
PRONOUNS = {'n', 'i', 'a', 'o', 'u', 'anw', 'aw', 'ùw', 'inw', 'iw'}
PARTICLES = {'wà', 'dun', 'foyi', 'ni', 'le', 'te', 'den', 'min', 'don', 'dòn', 'ko', 'ní', 'mána', 'la', 'kɔnɔ'}

EXPECTED_ORDERS = {
    'simple': ['S', 'TAM', 'O', 'V', 'OBL'],
    'conditional': ['ní', 'S', 'TAM', 'O', 'V'],
    'temporal': ['tuma_min', 'S', 'TAM', 'O', 'V'],
    'restrictive': ['S', 'TAM', 'foyi', 'yé', 'ni', 'ATTR', 'tɛ'],
    ...
}
```

---

## Your Current Score

### Word Order Quality: ✅ EXCELLENT

- **Kendall τ**: +1.000 (perfect alignment)
- **Exact matches**: 100%
- **Interpretation**: System maintains correct Bambara word order

### Next Steps:

1. **Test more categories** - Run on full test suite
2. **Add baseline comparisons** - Compare against Google/NLLB
3. **Rule-by-rule analysis** - Which rules have τ > 0.7?
4. **Track regressions** - Monitor τ over time as you add new rules
