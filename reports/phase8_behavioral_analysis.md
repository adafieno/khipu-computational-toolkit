# Phase 8: Behavioral Recording Analysis — Report

## Overview

Phase 7 confirmed that the corpus divides robustly into two structural types (T1/T2)
across any combination of morphological features. This phase takes a different
starting point: instead of asking *what does a khipu look like*, it asks *what
recording behavior does it exhibit*?

Seven behavioral signals are derived directly from the knot values and structural
hierarchy rather than from cord counts or color vocabulary:

| Signal | What it measures |
|--------|-----------------|
| `value_register` | Median non-zero cord value — the accounting "scale" |
| `pct_nonzero` | Fraction of cords carrying any encoded value |
| `pct_round5` | Fraction of non-zero values divisible by 5 — proxy for quota/tribute standardisation |
| `entropy_per_cord` | Shannon entropy of cord values ÷ recorded cords — information density |
| `max_hier_level` | Deepest hierarchy level — aggregation tier depth |
| `knot_L_ratio` | Fraction of long-knots (encode digits 2-9) among all knot clusters |
| `knot_E_ratio` | Fraction of figure-eight knots (encode the unit digit / terminal digit) |

The core claim: **two khipus can be structurally identical (same cord count, same T1/T2
classification) while exhibiting completely different recording behaviors**.
This phase tests that claim directly.

---

## Corpus-Wide Behavioral Baseline

Before clustering, the corpus-wide behavioral signals contextualize what we are
partitioning:

| Metric | Value |
|--------|-------|
| Cord values range | 0 – 320,535 |
| Median cord value | 3 |
| % cords with non-zero values | ~70% (including zeros as placeholders) |
| % values divisible by 5 | **48.1%** — strong corpus-wide round-number affinity |
| % values divisible by 10 | 42.4% |
| Khipus with ≥3 hierarchy levels | 48 (6.8%) |
| Knot type split | L: 43.7%  S: 41.7%  E: 13.7%  other: 0.9% |
| Mean Shannon entropy per cord | 3.4 bits (range 0 – 7.3 bits) |

The 48.1% round-five affinity is a major behavioral signal at corpus level: nearly
half of all encoded values are multiples of five. This is far above the random
expectation of 20% and is consistent with Andean base-ten positional counting where
standardized quota units (mit'a levy, tribute allotments, storage quantities) were
commonly expressed in round units.

---

## Clustering on Behavioral Axes

A k-means sweep (k = 2–6, log-transformed value_register and entropy_per_cord)
produced k = 6 as the optimal partition (silhouette = 0.212). While this is a modest
silhouette — behavioral signals are inherently noisier than structural dimensions —
the six groups are clearly differentiated along specific axes.

| k | Silhouette |
|---|-----------|
| 2 | 0.210 |
| 3 | 0.180 |
| 4 | 0.179 |
| 5 | 0.187 |
| **6** | **0.212** |

> Note: The lower silhouette relative to the structural k=2 (0.560) is expected.
> Behavioral diversity is a continuous spectrum, not a sharp discontinuity.
> The six groups describe *modal recording styles*, not mutually exclusive categories.

---

## Behavioral Recording Profiles

### B1 — Non-numeric / Unrecorded (n = 15, 2.1%)

| Feature | Value |
|---------|-------|
| Median cord value | 0 |
| % round-5 | 6.7% |
| Entropy / cord | 0.000 bits |
| Max hierarchy level | 0 (median) |
| L-knot ratio | 0.00 |
| E-knot ratio | 0.03 |
| % T2 structural | 0% |

These khipus carry no encoded numeric content. Zero entropy and zero L-knot ratio
indicate no positional-notation knots are present. Maximum hierarchy level of 0 means
they consist only of a primary cord with no pendant structure, or that the cord
structure is unregistered.

**Data signals suggest:** Narrative / non-quantitative khipus; severely incomplete
records; cord bundles that served as mnemonic or signaling objects without positional
encoding.

---

### B2 — Small-unit, E-knot dominant records (n = 80, 11.3%)

| Feature | Value |
|---------|-------|
| Median cord value | 2 |
| % round-5 | 7.4% |
| Entropy / cord | 0.076 bits |
| Max hierarchy level | 1 |
| L-knot ratio | 0.39 |
| E-knot ratio | **0.53** (highest) |
| % T2 structural | 12.5% |

In Andean positional knot notation, figure-eight knots (E) encode the terminal unit
digit and mark the end of a knot cluster. A high E-knot ratio in the context of very
small values (median = 2) means these khipus primarily record single-digit quantities
where almost every cord terminates with a unit marker. The very low round-5 affinity
(7.4%) indicates quantities are not being rounded — these are specific small counts.

**Data signals suggest:** Household-level or individual-level tallies where
discrete objects (people, animals, containers) are counted without rounding; fine-
grained census registers or inspection records.

---

### B3 — L-knot intensive, small-range census style (n = 245, 34.6%)

| Feature | Value |
|---------|-------|
| Median cord value | 6.5 |
| % round-5 | 21.3% |
| Entropy / cord | 0.099 bits |
| Max hierarchy level | 1 |
| L-knot ratio | **0.577** (highest) |
| E-knot ratio | 0.119 |
| % T2 structural | 2.9% |

The largest single behavioral group and the most L-knot-intensive. Long-knots encode
non-unit digits (2-9) in the positional system, so their prevalence here indicates
that most values fall in the 2–9 range. Round-number affinity is moderate (21.3%),
suggesting a mix of rounded and specific quantities.

**Data signals suggest:** The workhorse accounting cluster — general-purpose
records covering commodities, labour units, or population counts in the sub-ten range.
The breadth of this group (35% of corpus) is consistent with it being the default
administrative recording style for routine local or decimal-unit bookkeeping.

---

### B4 — Two-tier hierarchical, medium scale (n = 126, 17.8%)

| Feature | Value |
|---------|-------|
| Median cord value | 10 |
| % round-5 | 26.2% |
| Entropy / cord | 0.062 bits |
| Max hierarchy level | **2** (median) |
| L-knot ratio | 0.458 |
| E-knot ratio | 0.136 |
| % T2 structural | **30.2%** (highest) |

The most structurally complex behavioral group: median two-tier hierarchy, moderate
values near 10, and the highest fraction of T2 (extended-register) khipus (30%).
Two-tier hierarchy means pendant cords bear subsidiary cords — values encoded at the
subsidiary level are typically partial quantities that sum to pendant values, or
parallel attribute channels for the same item.

**Data signals suggest:** District or decimal-group-level records where subsidiary
cords break out component contributions. The co-occurrence with T2 suggests that the
extended-register structural type serves primarily this hierarchical aggregation
function.

---

### B5 — Flat high-variety records (n = 104, 14.7%)

| Feature | Value |
|---------|-------|
| Median cord value | 13 |
| % round-5 | 20.2% |
| Entropy / cord | **0.366** (highest) |
| Max hierarchy level | 0 (median — flat) |
| L-knot ratio | 0.538 |
| E-knot ratio | 0.073 |
| % T2 structural | 0% |

The most information-dense behavioral group: highest entropy per cord (0.366 bits),
flat hierarchy (median depth 0), and exclusively T1 structural type. Despite being
structurally compact, these khipus pack significantly more distinct values per cord
than any other group. Low round-number affinity (20.2%) reinforces specificity.

**Data signals suggest:** Multi-commodity inventories where each cord records a
distinct item type with a specific (non-standardized) quantity. The word “inventory”
is appropriate: flat structure = one level = one cord per commodity = no hierarchical
aggregation needed. High entropy = many different quantities recorded.

---

### B6 — Large-scale quota records (n = 139, 19.6%)

| Feature | Value |
|---------|-------|
| Median cord value | **80** |
| % round-5 | **58.4%** (highest) |
| Entropy / cord | 0.167 bits |
| Max hierarchy level | 1 |
| L-knot ratio | 0.198 (lowest among active groups) |
| E-knot ratio | 0.039 (lowest) |
| % T2 structural | 0.7% |

The most distinctive behavioral cluster: large values (median = 80, mean far higher),
the strongest round-number affinity (58.4% divisible by 5), and minimal knot
complexity. Low L-ratio and E-ratio despite large values may appear paradoxical but
is consistent with values that are systematically expressed in multiples of 5/10/100,
requiring fewer distinct digit knots. Almost exclusively T1 structural type.

**Data signals suggest:** Tribute/mit’a quota records at district or province
level. Round numbers of 50, 100, 200, 500 characterize standardized labor or
commodity assessments — the Inka tribute system assigned fixed quotas in round units.
The structural simplicity (T1) of these records is expected: a tribute tally does not
need hierarchical breakdown, just a list of quota amounts per category.

---

## Statistical Hypothesis Tests

### H1: Round-number affinity varies by geographic zone

**Kruskal-Wallis stat = 9.595, p = 0.213 → Cannot reject H0**

Round-number affinity is distributed uniformly across geographic zones. This is
counter-intuitive: if zones specialised in tribute vs. census accounting, we would
expect zonal differences. The absence of signal may indicate either that quota
recording was standardized empire-wide, or that geographic attribution is too sparse
to detect zonal patterns (35% of corpus lacks provenance).

### H2: Multi-tier hierarchy (depth ≥ 3) is geographically concentrated

**Chi-square = 25.896, p = 0.0005 → Reject H0**

Multi-level aggregation khipus are significantly concentrated on the coast:

| Zone | n depth≥3 |
|------|-----------|
| Cañete–Pisco | 13 |
| Central Coast | 13 |
| Arica & N. Chile | 3 |
| Chachapoyas | 2 |
| Nazca & Far South | 1 |

This is a strong result. Coastal zones, particularly Cañete–Pisco and the Central
Coast, appear to be sites where multi-level administrative hierarchies were maintained
in khipu form. These zones correspond to densely administered coastal valleys that
were major producers of tribute goods and mit'a labour.

### H3: Behavioral clusters significantly cross-cut T1/T2 structural typology

**Chi-square = 116.768, p < 0.0001 → Reject H0**

Behavioral clusters are not reducible to the structural binary. The B6 quota-records
group is 99.3% T1 (structurally compact) yet records the largest values. B4
hierarchical group has 30% T2, but the remaining 70% are T1. B1 and B5 are 100% T1
despite being behaviorally opposite (unrecorded vs. high-variety).

This confirms the central claim: **a khipu's size and complexity tell you its form,
not its function**. Different accounting behaviors are implemented in structurally
similar packages.

### Bonus: Round-number affinity varies by summation pattern type

**Kruskal-Wallis stat = 23.108, p = 0.0003 → Reject H0**

| Pattern type | Mean % round-5 | Interpretation |
|-------------|---------------|----------------|
| has_is | 37.6% | Inter-segment: highly standardized quotas |
| has_sp | 37.5% | Standard pendant summation: quota-linked |
| has_gg | 36.7% | Group-to-group: moderately standardized |
| has_pp | 28.1% | Pendant-to-pendant: mixed |
| has_psn | 25.3% | Pendant-subset: mixed |
| has_cp | 6.3% | Cross-pendant: low standardisation |
| has_gsb | 7.2% | Group-sub: low standardisation |

Summation patterns that operate *across groups or segments* (IS, SP, GG) are
associated with higher round-number affinity, suggesting these pattern types served
to aggregate standardized quota streams. Intra-pendant patterns (CP, GSB) are
associated with non-rounded, specific values — more census-like recording.

---

## The Structural vs. Behavioral Distinction — Summary

| Dimension | Phase 7 structural | Phase 8 behavioral |
|-----------|-------------------|-------------------|
| What is measured | Cord count, hierarchy size, color vocab | Knot values, rounding, entropy, depth |
| Best k | 2 (silhouette 0.560) | 6 (silhouette 0.212) |
| Primary axis of variation | Scale / complexity | Accounting style |
| T2 concentration | By definition T2 = extended-register | B4 is 30% T2; B1, B5, B6 are <1% T2 |
| Geographic signal | Leymebamba drives T2 | Coastal zones drive multi-tier depth |
| Actionable hypothesis | Leymebamba = archival depot | Coastal valleys = tribute quota centers |

The two analyses are complementary, not competitive. T1/T2 tells you what class of
physical object a khipu is. B1–B6 tells you what accounting task it was performing.

---

## Outputs

| File | Description |
|------|-------------|
| `data/processed/phase8_behavioral_features.csv` | Per-khipu behavioral features (7 signals + metadata) |
| `data/processed/phase8_behavioral_clusters.csv` | Per-khipu cluster assignment B1–B6 |
| `data/processed/phase8_behavioral_profiles.csv` | Per-cluster feature means |
| `visualizations/phase8/silhouette_curve.png` | K-sweep silhouette and inertia |
| `visualizations/phase8/behavioral_heatmap.png` | Row-normalized feature heatmap for B1–B6 |
| `visualizations/phase8/value_register.png` | Value register distributions and round-number affinity boxplots |
| `visualizations/phase8/round_number_zone.png` | Round-number affinity and accounting scale by geographic zone |
| `visualizations/phase8/cross_structural.png` | T1/T2 composition, hierarchy depth, and entropy per behavioral cluster |

---

## Limitations

1. **`cords.value` is pre-computed**: The analysis uses the `value` column in the
   KFG `cords` table, which aggregates knot cluster values. Zero-valued cords (30.3%)
   are excluded from behavioral ratios, but distinguishing "zero recorded" from "not
   recorded" requires cord-level audit (outside scope here).

2. **Round-number affinity is a feature proxy**: Divisibility by 5 is a useful proxy
   for quota-style recording, but khipus recording natural counts that happen to
   fall on multiples of 5 will be misclassified. The signal is statistical, not
   deterministic.

3. **Silhouette 0.212**: Behavioral clusters overlap more than structural ones.
   Treat B1–B6 as *modal behavioral signatures* rather than discrete categories.
   A given khipu may show characteristics of more than one group.

4. **Provenance sparsity**: H1 (round-number affinity × zone) returned non-significant
   results; 35% missing provenance limits geographic analysis power.


