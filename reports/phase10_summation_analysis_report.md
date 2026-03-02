# Phase 10: Summation Structure & Zero-Cord Analysis

*Khipu Computational Toolkit — kfg-integration branch*

---

## Overview

Phase 10 was designed as a missing-value prediction exercise.  Preliminary
analysis found **zero missing values** in the KFG cord table — all 62,746 cords
carry a recorded entry.  However, two facts motivate a reorientation:

1. **30.3 % of cords have value = 0** (19,040 cords) — these may represent unused
   recording slots rather than genuine zero counts.
2. **Only 6.6 % of parent-child groups satisfy the additive summation constraint**
   (`sum of children ≈ parent value`) — the dominant model of khipu arithmetic in
   the literature applies to less than one in fifteen hierarchical relationships.

Phase 10 therefore investigates:
- What arithmetic pattern *does* hold between parent and child cords?
- Are zero-value cords randomly distributed, or do they cluster by khipu type?
- What predicts the rare but genuine summation-compliant groups?

**Data scope:** 709 khipus, 62,746 cords, 10,381 parent-child groups.

---

## Methods

### Summation compliance classification

For every parent cord that has at least one subsidiary child, the **summation
ratio** `r = sum(children) / parent_value` is computed.  Groups fall into one of
seven classes:

| Class | Ratio range | Count | Share |
|-------|-------------|------:|------:|
| `sub` | r < 0.50 | 2,565 | 24.7 % |
| `trivial_parent_zero` | parent = 0 | 2,126 | 20.5 % |
| `children_all_zero` | all children = 0 | 1,967 | 18.9 % |
| `supra` | r > 1.50 | 1,469 | 14.2 % |
| `partial_sub` | 0.50 ≤ r < 0.95 | 1,119 | 10.8 % |
| **`compliant`** | **0.95 ≤ r ≤ 1.05** | **680** | **6.6 %** |
| `partial_supra` | 1.05 < r ≤ 1.50 | 455 | 4.4 % |

### Ratio landmark test

The ratio distribution is tested for clustering near simple fractions (1/10, 1/5,
1/3, 1/2, 2/3, 1/1, 2/1, 10/1) to detect decimal or proportional accounting
conventions.

### Zero-cord profile

Per-khipu: fraction of pendant cords with value = 0, fraction of subsidiary cords
with value = 0, and cross-comparison with behavioral cluster, tree depth, and
round-number affinity.

### Predictive model

Logistic regression (L2, class-weight balanced) with 5-fold CV to predict whether
a khipu has ≥ 10 % summation-compliant groups.  Features: tree depth, branching
entropy, % round-5 values, % zero cords, balance score, behavioral cluster (OHE),
structural type (OHE).

---

## Results

### 1. The summation constraint rarely holds

The additive model (`parent = sum(children)`) applies to only **6.6 %** of
parent-child groups.  The majority of groups fall into three alternative classes:

- **`sub` (24.7 %):** The children summed together are less than half the parent
  value.  This is the most common pattern.  The median summation ratio across the
  corpus is **0.35** — parent values are typically three times larger than the sum
  of their recorded subsidiaries.

- **`trivial_parent_zero` (20.5 %) and `children_all_zero` (18.9 %):** Together
  these form 39.4 % of all groups, meaning structural hierarchy is present but no
  actual values are recorded on one side of the relationship.

- **`supra` (14.2 %):** The children sum exceeds the parent value.  This cannot
  represent simple aggregation and suggests either partial records on the parent
  side, or a different unit of measurement between levels.

**Interpretation:** The parent cord in a khipu hierarchy does not typically store
the arithmetic sum of its subsidiaries.  Instead, it likely records an
*independent total from a higher accounting tier* — a separate audit or census
figure that the subsidiary cords partially decompose.  This is consistent with
Andean decimal administration where sub-accounts (pachakas, ayllus) reported
independently to higher officers, who maintained their own running totals.

### 2. Ratio landmark analysis

| Ratio landmark | Occurrences | Share |
|----------------|------------:|------:|
| 1/1  (compliant) | 680 | 8.2 % |
| 1/10 | 382 | 4.6 % |
| 1/2  | 310 | 3.8 % |
| 1/3  | 282 | 3.4 % |
| 1/5  | 271 | 3.3 % |
| 2/3  | 183 | 2.2 % |
| 2/1  | 268 | 3.2 % |
| 10/1 | 52 | 0.6 % |

The 1/1 landmark is the strongest concentration, but no single non-unity landmark
dominates.  The spread across fractional ratios (1/10 through 2/3) suggests that
the corpus contains a mixture of recording conventions rather than a single
decimal rule.  The 4.6 % at 1/10 is consistent with **decimal sub-sampling** — a
pendant recording one tenth of a total that stays on the parent cord.

### 3. Zero-cord distribution by behavioral cluster

| Cluster | Median % zeros | Summation compliance | Median ratio |
|---------|---------------:|---------------------:|-------------:|
| **B1** (non-recording) | **100.0 %** | 0.0 % | — |
| **B2** (unit-count) | 31.6 % | 6.9 % | 0.50 |
| **B3** (L-knot workhorse) | 15.1 % | 0.0 % | 0.29 |
| **B4** (two-tier hierarchical) | 20.7 % | 4.2 % | 0.36 |
| **B5** (flat high-variety) | 14.3 % | 0.0 % | 0.28 |
| **B6** (quota / round-5) | 17.6 % | 0.0 % | 0.22 |

**B1 = 100 % zeros** independently confirms what Phase 8 found from knot-type
features: B1 khipus are structurally present but carry no numeric content.  They
are categorically distinct from all other clusters by this measure alone.

**B2 shows the highest compliance rate (6.9 %)** and the highest median ratio
(0.50).  B2 was identified in Phase 8 as the "unit-count, high E-knot" cluster.
A ratio of 0.50 could indicate a **two-for-one conversion factor** — subsidiary
cords recording half-units while the parent accumulates full-unit totals.

**B3, B5, B6 show 0 % median compliance**.  Their median ratios (0.22–0.29)
mean parent values are roughly four times their children sums.  These clusters
appear to use the parent-subsidiary hierarchy as an *organisational device* (for
grouping related counts) rather than an arithmetic one.

### 4. T1 vs T2 summation behaviour

| Measure | T1 (n = 653) | T2 (n = 56) | p-value |
|---------|------------:|------------:|--------:|
| Median % compliant groups | 0.0 % | 4.1 % | < 0.001 *** |
| Median % zero-value cords | 17.6 % | 26.9 % | < 0.001 *** |

T2 khipus (large, administratively apex) have both more summation-compliant
groups and more zero-value cords.  The higher zero rate supports the reading of
T2 as an **administrative template** — larger numbers of reserved slots preprinted
into the cord structure, many of which remain unfilled in any given census cycle.
The higher compliance rate suggests those T2 slots that *are* filled more often
satisfy the additive summation rule, consistent with T2 serving as a multi-tier
aggregation record.

### 5. Predictive model

A logistic regression predicting "high-compliance khipu" (≥ 10 % of groups
compliant) achieves **5-fold CV ROC-AUC = 0.736 ± 0.053** — significantly above
chance, indicating that observable khipu features do encode accounting structure.

Top predictors by absolute coefficient magnitude:

| Rank | Feature | |coefficient| | Direction |
|------|---------|--------------|-----------|
| 1 | `depth` | 1.18 | ↑ deeper → more compliant |
| 2 | `beh_B2` | 0.42 | B2 cluster → more compliant |
| 3 | `pct_zero` | 0.38 | more zeros → higher compliance |
| 4 | `beh_B1` | 0.35 | B1 cluster → more compliant (trivially — child zeros make ratios degenerate) |
| 5 | `beh_B4` | 0.32 | B4 → more compliant |

**Depth as top predictor** confirms the intuition from Phase 9: deeper khipus
are the ones recording genuine hierarchical aggregations.  The positive coefficient
for `pct_zero` is initially counter-intuitive but makes sense: zero-heavy khipus
include many `trivial_parent_zero` and `children_all_zero` groups which are
excluded from compliance testing, leaving behind a purer subset of numeric groups
that more often comply.

---

## Possible interpretations

### The parent cord records the administrative total, not the record total

The prevailing model, that `parent = sum(children)`, seems calibrated to the small
subset of khipus that serve as summation ledgers.  For the majority, the parent
cord appears to record an independently-arrived-at total (e.g., a census figure
from a higher administrative authority) while the subsidiary cords record the
partial breakdown known at the local level.  The two figures do not match because
they originate from different nodes in the bureaucratic network.

### Zero cords as structural placeholders

The concentration of zeros at high rates in B1 and B2, and the positive
relationship between zero rate and compliance, suggests that many zero-value cords
are **reserved entries** — cord slots built into the template awaiting population
in a future census cycle.  This would explain T2's higher zero rate (more
pre-allocated slots for a larger administrative unit) without requiring that T2
records were systematically incomplete.

### The 1/10 ratio peak as decimal sub-sampling

382 groups (4.6 % of ratios) cluster near r = 0.10.  If this is a genuine decimal
convention — subsidiaries recording one-tenth of the parent total — it would be
consistent with the Inca pachaka (100-unit) / chunka (10-unit) hierarchy, where a
parent cord records the full 100-unit total and each subsidiary records one chunka.

---

## Limitations

- The parent-child matching depends on the accuracy of the KFG `cord_name` /
  `parent_cord` transcription.  Transcription errors would inflate the `supra` and
  `sub` classes.
- Cord *color* and *knot type* are not incorporated into the compliance model —
  these may carry unit-conversion information that would explain supra-compliant
  groups.
- The B1 compliance figure is trivially high (zero-zero groups pass the trivial
  filter) and should not be interpreted as evidence of B1 arithmetic.

---

## Outputs

| File | Description |
|------|-------------|
| `data/processed/phase10_summation_groups.csv` | Per-parent-group: children sum, parent value, ratio, compliance class |
| `data/processed/phase10_zero_analysis.csv` | Per-khipu: zero rates, compliance rates, behavioral label |
| `visualizations/phase10/ratio_distribution.png` | Log-scale ratio histogram + compliance class breakdown |
| `visualizations/phase10/compliance_by_cluster.png` | Compliance rate and zero rate by behavioral cluster |
| `visualizations/phase10/zero_cord_patterns.png` | Zero distribution patterns across khipus |
| `visualizations/phase10/compliance_predictors.png` | Feature importances and round-5 vs compliance scatter |
