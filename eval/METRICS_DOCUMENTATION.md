# Tree Comparison Metrics Documentation

## Current Metrics Used

### 1. **Sequence Similarity (Longest Common Subsequence)**

**Formula**: Using `difflib.SequenceMatcher.ratio()`
```
Similarity = 2 * LCS_length / (len(tree1) + len(tree2))
```

**Range**: 0.0 (completely different) to 1.0 (identical)

**Example**:
```
French:  PRON VERB DET NOUN ADJ
Kuma:    PRON AUX VERB DET NOUN
LCS:     PRON VERB DET NOUN = 4 matches
Ratio:   2*4 / (5+5) = 0.80 (80% similarity)
```

**Pros**:
- Handles insertions/deletions well
- Order-sensitive (important for syntax)
- Computationally efficient

**Cons**:
- Doesn't account for semantic POS relationships
- Treats all POS differences equally

---

### 2. **Tree Depth (Token Count)**

**Formula**: `depth = len(pos_sequence)`

**Change**: `depth_change = kuma_depth - french_depth`

**Range**: Any integer (typically -10 to +10)

**Example**:
```
French depth:  5 tokens
Kuma depth:    6 tokens
Depth change:  +1 (1 token expansion)
```

**Interpretation**:
- **+1 to +3**: Normal grammatical expansion (markers, auxiliaries)
- **+4 to +8**: Significant restructuring
- **+9+**: Potential over-generation (quality issue?)

---

### 3. **Complexity Metrics**

#### a) **Unique POS Count**
- Number of distinct POS tags used
- `unique_pos = len(set(pos_tree))`

**Example**:
```
French: {PRON, VERB, DET, NOUN, ADJ} = 5 unique POS
Kuma:   {PRON, AUX, VERB, DET, NOUN} = 5 unique POS
```

#### b) **POS Distribution**
- Frequency of each POS tag

**Example**:
```
French: NOUN:3, VERB:2, DET:1, ADJ:1
Kuma:   NOUN:3, VERB:2, DET:2, AUX:1
```

#### c) **Most Common POS**
- Which POS tag appears most frequently

**Example**:
```
French: NOUN (appears 3 times = 60%)
Kuma:   NOUN (appears 3 times = 50%)
```

---

## Advanced Metrics (To Implement)

### 1. **Edit Distance (Levenshtein Distance)**

**Formula**: Minimum edits needed to transform tree1 → tree2

```python
def levenshtein_distance(tree1, tree2):
    # Count: insertions, deletions, substitutions
    # Return normalized score (0-1)
```

**Range**: 0 (identical) to 1 (completely different)

**Advantages**:
- More granular than Sequence Matcher
- Separates insertions/deletions/substitutions
- Standard in linguistics

**Example**:
```
French:  PRON VERB DET NOUN
Kuma:    PRON AUX VERB DET NOUN
Edits:   1 insertion (AUX)
Distance: 1/4 = 0.25 (normalized)
```

---

### 2. **POS Overlap Coefficient (Jaccard Similarity)**

**Formula**: 
```
similarity = |set1 ∩ set2| / |set1 ∪ set2|
```

**Range**: 0 (no overlap) to 1 (identical sets)

**Example**:
```
French POS set: {PRON, VERB, DET, NOUN, ADJ}
Kuma POS set:   {PRON, AUX, VERB, DET, NOUN}
Intersection:   {PRON, VERB, DET, NOUN} = 4
Union:          {PRON, AUX, VERB, DET, NOUN, ADJ} = 6
Jaccard:        4/6 = 0.67 (67% overlap)
```

**Use case**: Assess vocabulary compatibility

---

### 3. **Distribution Distance (Wasserstein Distance)**

**Formula**: Compares POS frequency distributions

```
distance = sum(|freq1[pos] - freq2[pos]|) / total_tokens
```

**Range**: 0 (identical distribution) to 1 (completely different)

**Example**:
```
French:  NOUN:60%, VERB:30%, DET:10%
Kuma:    NOUN:50%, VERB:33%, DET:17%
Distance: (10% + 3% + 7%) / 100 = 0.20
```

**Use case**: Detect structural differences

---

### 4. **Reordering Index**

**Formula**: Measures how much words moved positions

```
reordering_index = sum(|pos_old[i] - pos_new[i]|) / (2 * n)
```

**Range**: 0 (no reordering) to 1 (completely reversed)

**Example**:
```
French order:  [1:PRON, 2:VERB, 3:DET, 4:NOUN]
Kuma order:    [1:PRON, 2:AUX, 3:VERB, 4:DET, 5:NOUN]
Total movement: 2 positions
Reordering:    2/(2*4) = 0.25
```

**Use case**: Detect SOV ↔ SVO word order changes

---

### 5. **Quality Score (Composite Metric)**

Combines multiple metrics for overall quality:

```
quality_score = (w1 * similarity) 
              + (w2 * (1 - |depth_change|/max_depth)) 
              + (w3 * jaccard_similarity)
              + (w4 * (1 - reordering_index))

Where: w1=0.4, w2=0.2, w3=0.2, w4=0.2
```

**Range**: 0 (poor) to 1 (excellent)

---

## Recommendations for Baseline Comparison

### Metric Priority by Use Case

**For Overall Quality**:
1. Sequence Similarity (primary)
2. Depth Change (secondary)
3. Distribution Distance (tertiary)

**For Syntactic Analysis**:
1. Reordering Index
2. Jaccard Similarity
3. Edit Distance

**For Rule Validation**:
1. Specific POS patterns (custom per rule)
2. Complexity metrics
3. Distribution changes

**For Error Detection**:
1. Outlier detection (depth > 2σ)
2. POS distribution anomalies
3. Quality score < 0.5

---

## Implementation Example

```python
from difflib import SequenceMatcher
from collections import Counter
import math

def calculate_all_metrics(french_pos: str, output_pos: str) -> Dict:
    """Calculate comprehensive metrics for tree comparison."""
    
    fr_tree = french_pos.split()
    out_tree = output_pos.split()
    
    metrics = {
        # Basic metrics
        'sequence_similarity': SequenceMatcher(None, fr_tree, out_tree).ratio(),
        'depth_french': len(fr_tree),
        'depth_output': len(out_tree),
        'depth_change': len(out_tree) - len(fr_tree),
        
        # Complexity
        'unique_pos_french': len(set(fr_tree)),
        'unique_pos_output': len(set(out_tree)),
        
        # Distribution
        'fr_distribution': dict(Counter(fr_tree)),
        'out_distribution': dict(Counter(out_tree)),
        'distribution_distance': wasserstein_distance(fr_tree, out_tree),
        
        # Advanced
        'jaccard_similarity': jaccard_similarity(set(fr_tree), set(out_tree)),
        'edit_distance': levenshtein_distance(fr_tree, out_tree),
        'reordering_index': calculate_reordering(fr_tree, out_tree),
        
        # Composite
        'quality_score': compute_quality_score(fr_tree, out_tree),
    }
    
    return metrics
```

---

## Quick Reference Table

| Metric | Range | Interpretation | Use For |
|--------|-------|-----------------|---------|
| Similarity | 0-1 | 0=different, 1=identical | Overall quality |
| Depth Δ | -∞ to +∞ | Negative=compression, Positive=expansion | Efficiency |
| Jaccard | 0-1 | 0=no overlap, 1=identical | Vocabulary |
| Distance | 0-1 | 0=same dist, 1=opposite dist | Structure |
| Reordering | 0-1 | 0=no change, 1=reversed | Word order |
| Quality | 0-1 | 0=poor, 1=excellent | Overall score |

---

## CSV Output Format Recommendation

```csv
#,category,phrase_fr,bambara_obtenu,statut,french_pos_tree,bambara_pos_tree,similarity,depth_change,jaccard,distribution_distance,quality_score
1,rule1,phrase,translation,PASS,pos_seq,pos_seq,0.85,+1,0.80,0.15,0.81
```

Add columns when comparing with baselines:
- `google_pos_tree`, `google_similarity`, `google_quality`
- `nllb_pos_tree`, `nllb_similarity`, `nllb_quality`

Then compare: `kuma_similarity` vs `google_similarity` vs `nllb_similarity`
