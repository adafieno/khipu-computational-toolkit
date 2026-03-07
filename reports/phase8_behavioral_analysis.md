# Phase 8: Behavioral Recording Analysis

**Generated:** 2026-03-02 (updated)  
**Database:** K-CAT SQLite database (built from KFG source data)  
**Script:** `scripts/run_phase8_behavior.py`  
**Inputs:** `data/kfg/khipu_database.db` · Phase 7 typology assignments  
**Status:** ✅ Complete

---

## Research Question

Can khipus be partitioned by their *recording behavior* — the statistical properties of the values they encode — independently of their structural form (T1/T2)? Do behavioral clusters cross-cut the structural typology?

---

## Behavioral Signals

Seven signals derived from knot values and hierarchy rather than cord counts or color:

| Signal | What it measures |
|---|---|
| `value_register` | Median non-zero cord value |
| `pct_nonzero` | Fraction of cords carrying any encoded value |
| `pct_round5` | Fraction of non-zero values divisible by 5 |
| `entropy_per_cord` | Shannon entropy of cord values ÷ recorded cords |
| `max_hier_level` | Deepest hierarchy level |
| `knot_L_ratio` | Fraction of long-knots (digits 2–9) among all knot clusters |
| `knot_E_ratio` | Fraction of figure-eight knots (unit digit) among all knot clusters |

---

## Corpus-Wide Baseline

| Metric | Value |
|---|---|
| Cord values range | 0–320,535 |
| Median cord value | 3 |
| % cords with non-zero values | ~70% |
| % values divisible by 5 | 48.1% |
| % values divisible by 10 | 42.4% |
| Khipus with ≥ 3 hierarchy levels | 48 (6.8%) |
| Knot type split | L: 43.7%  S: 41.7%  E: 13.7%  other: 0.9% |
| Mean Shannon entropy per cord | 3.4 bits (range 0–7.3) |

48.1% of non-zero values are divisible by 5 (random expectation: 20%).

---

## Clustering

K-means sweep (k = 2–6, log-transformed `value_register` and `entropy_per_cord`):

| k | Silhouette |
|---|---|
| 2 | 0.210 |
| 3 | 0.180 |
| 4 | 0.179 |
| 5 | 0.187 |
| **6** | **0.212** |

k = 6 optimal. The lower silhouette relative to the structural k = 2 (0.560) reflects behavioral diversity as a continuous spectrum rather than a sharp discontinuity. The six groups describe modal recording styles.

---

## Behavioral Cluster Profiles

### B1 (n = 15, 2.1%)

| Feature | Value |
|---|---|
| Median cord value | 0 |
| % round-5 | 6.7% |
| Entropy / cord | 0.000 bits |
| Max hierarchy level | 0 (median) |
| L-knot ratio | 0.00 |
| E-knot ratio | 0.03 |
| % T2 | 0% |

No encoded numeric content. Zero entropy and zero L-knot ratio.

### B2 (n = 80, 11.3%)

| Feature | Value |
|---|---|
| Median cord value | 2 |
| % round-5 | 7.4% |
| Entropy / cord | 0.076 bits |
| Max hierarchy level | 1 |
| L-knot ratio | 0.39 |
| E-knot ratio | **0.53** (highest) |
| % T2 | 12.5% |

Highest E-knot ratio. Small values (median = 2). Very low round-5 affinity (7.4%).

### B3 (n = 245, 34.6%)

| Feature | Value |
|---|---|
| Median cord value | 6.5 |
| % round-5 | 21.3% |
| Entropy / cord | 0.099 bits |
| Max hierarchy level | 1 |
| L-knot ratio | **0.577** (highest) |
| E-knot ratio | 0.119 |
| % T2 | 2.9% |

Largest single group. Highest L-knot ratio. Values concentrated in the 2–9 range.

### B4 (n = 126, 17.8%)

| Feature | Value |
|---|---|
| Median cord value | 10 |
| % round-5 | 26.2% |
| Entropy / cord | 0.062 bits |
| Max hierarchy level | **2** (median) |
| L-knot ratio | 0.458 |
| E-knot ratio | 0.136 |
| % T2 | **30.2%** (highest) |

Deepest hierarchy (median depth = 2). Highest T2 fraction.

### B5 (n = 104, 14.7%)

| Feature | Value |
|---|---|
| Median cord value | 13 |
| % round-5 | 20.2% |
| Entropy / cord | **0.366** (highest) |
| Max hierarchy level | 0 (median — flat) |
| L-knot ratio | 0.538 |
| E-knot ratio | 0.073 |
| % T2 | 0% |

Highest entropy per cord. Flat hierarchy. Exclusively T1.

### B6 (n = 139, 19.6%)

| Feature | Value |
|---|---|
| Median cord value | **80** |
| % round-5 | **58.4%** (highest) |
| Entropy / cord | 0.167 bits |
| Max hierarchy level | 1 |
| L-knot ratio | 0.198 (lowest among active groups) |
| E-knot ratio | 0.039 (lowest) |
| % T2 | 0.7% |

Largest median value. Strongest round-5 affinity. Minimal knot complexity relative to value magnitude.

---

## Statistical Hypothesis Tests

### H1: Round-number affinity by geographic zone

**Kruskal-Wallis: H = 9.595, p = 0.213 → Not significant**

Round-number affinity does not differ significantly across geographic zones.

### H2: Multi-tier hierarchy (depth ≥ 3) by geographic zone

**Chi-square: χ² = 25.896, p = 0.0005 → Significant**

| Zone | n depth ≥ 3 |
|---|---|
| Cañete–Pisco | 13 |
| Central Coast | 13 |
| Arica & N. Chile | 3 |
| Chachapoyas | 2 |
| Nazca & Far South | 1 |

Multi-level hierarchy is concentrated in coastal zones.

### H3: Behavioral clusters cross-cut T1/T2 structural typology

**Chi-square: χ² = 116.768, p < 0.0001 → Significant**

Behavioral clusters are not reducible to the structural binary. B6 is 99.3% T1 yet records the largest values. B4 is 30% T2 but 70% T1. B1 and B5 are both 100% T1 despite having opposite value-layer profiles (zero content vs. high entropy).

### H4: Round-number affinity by summation pattern type

**Kruskal-Wallis: H = 23.108, p = 0.0003 → Significant**

| Pattern type | Mean % round-5 |
|---|---|
| has_is | 37.6% |
| has_sp | 37.5% |
| has_gg | 36.7% |
| has_pp | 28.1% |
| has_psn | 25.3% |
| has_cp | 6.3% |
| has_gsb | 7.2% |

Patterns operating across groups or segments (IS, SP, GG) associate with higher round-5 rates. Intra-pendant patterns (CP, GSB) associate with lower round-5 rates.

---

## Structural vs. Behavioral Comparison

| Dimension | Phase 7 (structural) | Phase 8 (behavioral) |
|---|---|---|
| What is measured | Cord count, hierarchy size, color vocab | Knot values, rounding, entropy, depth |
| Best k | 2 (silhouette 0.560) | 6 (silhouette 0.212) |
| Primary variation axis | Scale / complexity | Value magnitude / rounding / entropy |
| T2 concentration | By definition | B4 = 30% T2; B1, B5, B6 < 1% T2 |
| Geographic signal | Leymebamba drives T2 | Coastal zones drive multi-tier depth |

---

## Outputs

| File | Description |
|---|---|
| `data/processed/phase8_behavioral_features.csv` | Per-khipu behavioral features (7 signals + metadata) |
| `data/processed/phase8_behavioral_clusters.csv` | Per-khipu cluster assignment B1–B6 |
| `data/processed/phase8_behavioral_profiles.csv` | Per-cluster feature means |
| `visualizations/phase8/silhouette_curve.png` | K-sweep silhouette and inertia |
| `visualizations/phase8/behavioral_heatmap.png` | Row-normalized feature heatmap for B1–B6 |
| `visualizations/phase8/value_register.png` | Value register distributions and round-number affinity boxplots |
| `visualizations/phase8/round_number_zone.png` | Round-number affinity by geographic zone |
| `visualizations/phase8/cross_structural.png` | T1/T2 composition, hierarchy depth, and entropy per behavioral cluster |

---

## Limitations

1. **`cords.value` is pre-computed.** Zero-valued cords (30.3%) are excluded from behavioral ratios, but distinguishing "zero recorded" from "not recorded" requires cord-level audit.
2. **Round-number affinity is a statistical proxy.** Natural counts that happen to fall on multiples of 5 will contribute to the signal.
3. **Silhouette 0.212.** Behavioral clusters overlap substantially. B1–B6 are modal signatures, not mutually exclusive categories.
4. **Provenance sparsity.** 35% missing provenance limits geographic analysis power, which may explain H1's non-significant result.

---

*Corpus sweep run against K-CAT SQLite database. Re-run with `scripts/run_phase8_behavior.py` to refresh.*
