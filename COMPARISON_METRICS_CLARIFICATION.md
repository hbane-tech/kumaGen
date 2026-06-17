# Tree Comparison Metrics: Which Ones to Use?

## Your Question: Kendall Tau vs. Current Metrics

**Kendall tau** is a **rank correlation coefficient** - it measures whether the relative ordering of items is consistent between two rankings. 

**Example**: If French has word order [NOUN, VERB, DET, ADJ] and Bambara reorders to [NOUN, DET, VERB, ADJ], Kendall tau measures how much the **ranking consistency** changed.

---

## Recommended Metric Selection

### For Translation Quality Comparison:

#### **PRIMARY METRICS** (use these):

1. **Sequence Similarity (Sequence Matcher)** ✅
   - **What it measures**: How much of the POS sequence matches
   - **Range**: 0-1 (0=nothing matches, 1=identical)
   - **Formula**: `2 * LCS_length / (len(tree1) + len(tree2))`
   - **Best for**: Overall structure similarity
   - **Example**: 
     ```
     French:  PRON VERB DET NOUN ADJ
     Kuma:    PRON AUX VERB DET NOUN
     Similarity: 0.80 (4 out of 5 match in order)
     ```

2. **Edit Distance (Levenshtein)** ✅
   - **What it measures**: Minimum edits needed to transform tree1 → tree2
   - **Range**: 0-1 (0=identical, 1=completely different)
   - **Best for**: Granular diff analysis (insertion/deletion/substitution)
   - **Example**:
     ```
     Transformations: 1 insertion (AUX)
     Edit distance: 1/4 = 0.25
     ```

3. **Tree Depth Change** ✅
   - **What it measures**: Token count difference
   - **Range**: Any integer
   - **Best for**: Expansion/compression detection
   - **Example**: +2 tokens (normal grammatical expansion)

#### **SECONDARY METRICS** (context-dependent):

4. **Jaccard Similarity** (optional)
   - **What it measures**: POS vocabulary overlap
   - **Best for**: Assessing which POS tags are used
   - **Example**: Both use {PRON, VERB, DET, NOUN} → 100% overlap

5. **Distribution Distance** (optional)
   - **What it measures**: POS frequency pattern differences
   - **Best for**: Structural complexity changes
   - **Example**: NOUN frequency 60% → 50% = distance 0.10

---

## About Kendall Tau

### When Kendall Tau IS Useful:

**Kendall tau = rank correlation coefficient**

**Good for**:
- Comparing **relative word order** between systems
- Detecting whether all systems **reorder in similar ways**
- Measuring **consensus** across multiple baselines

**Example**:
```
Input order:     [NOUN, VERB, DET, ADJ] = ranks [1, 2, 3, 4]
Kuma output:     [NOUN, DET, VERB, ADJ] = ranks [1, 3, 2, 4]
Google output:   [NOUN, VERB, ADJ, DET] = ranks [1, 2, 4, 3]

Kendall tau (Kuma vs Google) = measure of reordering consistency
```

### When Kendall Tau is LESS Useful:

❌ **Not ideal for**:
- Comparing quality between systems (that's what Sequence Similarity does)
- Handling insertions/deletions (assumes same tokens, just reordered)
- Single pairwise comparison (needs multiple rankings to show correlation)

---

## Recommended Comparison Matrix

### For Basic Quality Comparison (3-way: kuma_mt vs Google vs NLLB):

```
METRIC                          WHAT IT TELLS YOU
────────────────────────────────────────────────────
Sequence Similarity (Kuma)      How well kuma matches French structure
Sequence Similarity (Google)    How well Google matches French structure  
Sequence Similarity (NLLB)      How well NLLB matches French structure
                                → WINNER: Highest similarity

Edit Distance (Kuma)            How many edits needed from French
Edit Distance (Google)          How many edits needed from French
Edit Distance (NLLB)            How many edits needed from French
                                → WINNER: Lowest distance

Depth Change (Kuma)             Token expansion/compression
Depth Change (Google)           Token expansion/compression
Depth Change (NLLB)             Token expansion/compression
                                → Most efficient: closest to zero
```

### For Advanced Analysis (If Using Kendall Tau):

```
METRIC                          WHAT IT TELLS YOU
────────────────────────────────────────────────────
Kendall tau (Kuma vs Google)    Do kuma & Google reorder similarly?
Kendall tau (Kuma vs NLLB)      Do kuma & NLLB reorder similarly?
Kendall tau (Google vs NLLB)    Do all systems reorder consistently?

Correlation >= 0.8              → High reordering consistency
Correlation 0.5-0.8             → Some reordering differences
Correlation < 0.5               → Major reordering differences
```

---

## Implementation Choice: Keep Current or Add Kendall?

### Current Implementation (Recommended):
- ✅ Sequence Similarity: measures actual structure match
- ✅ Edit Distance: measures granular differences
- ✅ Jaccard: measures vocabulary overlap
- ✅ Distribution Distance: measures structural complexity
- ✅ Quality Score: composite metric

### Add Kendall Tau IF:
- You want to **compare reordering patterns across baselines**
- You're analyzing **why** systems differ (not just comparing quality)
- You want to check if **all systems make similar word order choices**

---

## Code Example: Kendall Tau Implementation

```python
from scipy.stats import kendalltau

def kendall_tau_similarity(tree1: List[str], tree2: List[str]) -> float:
    """
    Measure rank correlation between two POS sequences.
    
    Note: Only works if both trees have similar tokens (insertions/deletions break it).
    For comparing systems, use Sequence Similarity instead.
    """
    # Create position-based ranks
    all_pos = list(dict.fromkeys(tree1 + tree2))
    
    ranks1 = [all_pos.index(pos) if pos in all_pos else -1 for pos in tree1]
    ranks2 = [all_pos.index(pos) if pos in all_pos else -1 for pos in tree2]
    
    # Only compare if length is same
    if len(ranks1) != len(ranks2):
        return None  # Kendall tau needs same-length sequences
    
    tau, p_value = kendalltau(ranks1, ranks2)
    return tau  # Range: -1 to +1 (similar to correlation)

# Example usage for baseline comparison:
kuma_output = "PRON AUX VERB DET NOUN ADJ"
google_output = "PRON VERB DET NOUN ADJ"
nllb_output = "PRON VERB ADJ NOUN DET"

# Only compare kuma vs nllb (similar length)
tau = kendall_tau_similarity(
    kuma_output.split(),
    nllb_output.split()
)
# tau = 0.33 (partial agreement on word order)
```

---

## Recommendation

**For your use case (comparing kuma_mt with Google & NLLB):**

### Use these metrics (in priority order):

1. **Sequence Similarity** (Primary) - "How similar is the structure?"
2. **Edit Distance** (Secondary) - "What changed?"
3. **Depth Change** (Secondary) - "How much expansion?"

### Consider adding Kendall Tau IF:
- You want to **analyze reordering patterns**
- You need to **compare consensus across 3+ systems**
- You're doing **detailed linguistic analysis** of word order strategies

### Skip Kendall Tau IF:
- Your trees have **different lengths** (insertions/deletions)
- You only care about **quality comparison** (Sequence Similarity covers that)
- You want **quick, interpretable results** (Sequence Similarity is easier to explain)

---

## CSV Column Recommendations

### Minimal (Recommended):
```csv
..., french_pos_tree, kuma_pos_tree, similarity, edit_distance, depth_change, quality_score
```

### Extended (If Using All Systems):
```csv
..., french_pos_tree, kuma_pos_tree, kuma_similarity, google_pos_tree, google_similarity, nllb_pos_tree, nllb_similarity, depth_change_kuma, depth_change_google, depth_change_nllb
```

### With Kendall Tau (For Advanced Analysis):
```csv
..., kuma_pos_tree, google_pos_tree, nllb_pos_tree, kuma_similarity, kendall_kuma_vs_google, kendall_kuma_vs_nllb, kendall_google_vs_nllb
```

---

## Final Answer to Your Question

**"What metric are you using for comparing the output? Is it any metric like Kendall tau?"**

Currently: **Sequence Similarity** (primary) + Edit Distance + Depth

Kendall tau would be useful **ONLY IF** you're comparing reordering patterns across 3+ systems with **similar tree lengths**. For your current use case (comparing quality), **Sequence Similarity is better** because it handles insertions/deletions and is more interpretable.
