# Phase 10: Summation Structure & Zero-Cord Analysis

**Generated:** 2026-03-02 (updated)  
**Database:** K-CAT SQLite database (built from KFG source data)  
**Script:** `scripts/run_phase10_summation.py`  
**Inputs:** `data/kfg/khipu_database.db` · Phase 7/8/9 assignments  
**Status:** ✅ Complete

---

## Research Question

What fraction of parent-child cord groups satisfy the additive summation constraint (parent ≈ sum of children)? How are zero-value cords distributed across khipu types? What observable features predict summation compliance?

**Data scope:** 709 khipus, 62,746 cords, 10,381 parent-child groups.

---

## Methods

### Summation compliance classification

For every parent cord with at least one subsidiary child, the summation ratio `r = sum(children) / parent_value` is computed. Groups are classified:

| Class | Ratio range | Count | Share |
|---|---|---|---|
| `sub` | r < 0.50 | 2,565 | 24.7% |
| `trivial_parent_zero` | parent = 0 | 2,126 | 20.5% |
| `children_all_zero` | all children = 0 | 1,967 | 18.9% |
| `supra` | r > 1.50 | 1,469 | 14.2% |
| `partial_sub` | 0.50 ≤ r < 0.95 | 1,119 | 10.8% |
| **`compliant`** | **0.95 ≤ r ≤ 1.05** | **680** | **6.6%** |
| `partial_supra` | 1.05 < r ≤ 1.50 | 455 | 4.4% |

### Ratio landmark test

The ratio distribution is tested for clustering near simple fractions (1/10, 1/5, 1/3, 1/2, 2/3, 1/1, 2/1, 10/1).

### Zero-cord profile

Per-khipu: fraction of pendant cords with value = 0, fraction of subsidiary cords with value = 0, cross-tabulated with behavioral cluster and tree depth.

### Predictive model

Logistic regression (L2, class-weight balanced) with 5-fold CV predicting whether a khipu has ≥ 10% summation-compliant groups. Features: tree depth, branching entropy, % round-5 values, % zero cords, balance score, behavioral cluster (OHE), structural type (OHE).

---

## Results

### 1. The Summation Constraint

The additive model (`parent = sum(children)`) applies to **6.6%** of parent-child groups.

The three most common classes:

- **`sub` (24.7%):** Children sum to less than half the parent value. Median summation ratio across the corpus is **0.35** — parent values are typically about three times the sum of their recorded subsidiaries.
- **`trivial_parent_zero` (20.5%) and `children_all_zero` (18.9%):** Together 39.4% of all groups — hierarchy is present but values are zero on one side of the relationship.
- **`supra` (14.2%):** Children sum exceeds the parent value.

### 2. Ratio Landmark Analysis

| Ratio landmark | Occurrences | Share |
|---|---|---|
| 1/1 (compliant) | 680 | 8.2% |
| 1/10 | 382 | 4.6% |
| 1/2 | 310 | 3.8% |
| 1/3 | 282 | 3.4% |
| 1/5 | 271 | 3.3% |
| 2/3 | 183 | 2.2% |
| 2/1 | 268 | 3.2% |
| 10/1 | 52 | 0.6% |

The 1/1 landmark is the strongest concentration. No single non-unity landmark dominates. The spread across fractional ratios (1/10 through 2/3) indicates a mixture of ratio patterns in the corpus.

### 3. Zero-Cord Distribution by Behavioral Cluster

| Cluster | Median % zeros | Summation compliance | Median ratio |
|---|---|---|---|
| B1 | **100.0%** | 0.0% | — |
| B2 | 31.6% | 6.9% | 0.50 |
| B3 | 15.1% | 0.0% | 0.29 |
| B4 | 20.7% | 4.2% | 0.36 |
| B5 | 14.3% | 0.0% | 0.28 |
| B6 | 17.6% | 0.0% | 0.22 |

B1 = 100% zeros, confirming Phase 8's finding that B1 khipus carry no numeric content. B2 has the highest compliance rate (6.9%) and median ratio (0.50). B3, B5, B6 show 0% median compliance with median ratios in the 0.22–0.29 range.

### 4. T1 vs T2 Summation Behavior

| Measure | T1 (n = 653) | T2 (n = 56) | p-value |
|---|---|---|---|
| Median % compliant groups | 0.0% | 4.1% | < 0.001 *** |
| Median % zero-value cords | 17.6% | 26.9% | < 0.001 *** |

T2 khipus have both more summation-compliant groups and more zero-value cords.

### 5. Predictive Model

Logistic regression predicting "high-compliance khipu" (≥ 10% of groups compliant):

**5-fold CV ROC-AUC = 0.736 ± 0.053**

Top predictors by absolute coefficient magnitude:

| Rank | Feature | |coefficient| | Direction |
|---|---|---|---|
| 1 | `depth` | 1.18 | deeper → more compliant |
| 2 | `beh_B2` | 0.42 | B2 → more compliant |
| 3 | `pct_zero` | 0.38 | more zeros → higher compliance |
| 4 | `beh_B1` | 0.35 | B1 → more compliant (trivially — zero-zero groups excluded from testing) |
| 5 | `beh_B4` | 0.32 | B4 → more compliant |

Depth is the strongest predictor: deeper khipus are more likely to contain compliant parent-child groups.

---

## Limitations

- **Parent-child matching** depends on the accuracy of the KFG `cord_name` / `parent_cord` transcription. Errors would inflate the `supra` and `sub` classes.
- **Cord color and knot type** are not incorporated into the compliance model.
- **B1 compliance** is trivially elevated by zero-zero group exclusions; this should not be treated as evidence of arithmetic structure.

---

## Outputs

| File | Description |
|---|---|
| `data/processed/phase10_summation_groups.csv` | Per-parent-group: children sum, parent value, ratio, compliance class |
| `data/processed/phase10_zero_analysis.csv` | Per-khipu: zero rates, compliance rates, behavioral label |
| `visualizations/phase10/ratio_distribution.png` | Log-scale ratio histogram + compliance class breakdown |
| `visualizations/phase10/compliance_by_cluster.png` | Compliance rate and zero rate by behavioral cluster |
| `visualizations/phase10/zero_cord_patterns.png` | Zero distribution patterns across khipus |
| `visualizations/phase10/compliance_predictors.png` | Feature importances and round-5 vs compliance scatter |

---

*Corpus sweep run against K-CAT SQLite database. Re-run with `scripts/run_phase10_summation.py` to refresh.*
