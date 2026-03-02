# Phase 7: Multi-feature Typology — Analysis Report

## Overview

Phase 7 extends the Phase 3 binary structural typology by constructing an enriched
feature matrix that spans structural dimensions (cord counts, hierarchy), summation
patterns, color diversity, and anomaly scores derived in Phases 4–6. A k-means sweep
(k = 2–8) with silhouette selection was applied to ask whether the corpus separates
into more than two coherent groups when all accumulated evidence is considered together.

The short answer: **it does not**. k = 2 remains the dominant partition with a
silhouette score of 0.560, while all larger k values fall in the narrow range
0.275–0.307. The Phase 3 binary is not a simplification artefact — it reflects a
genuine and deep structural distinction in the corpus.

---

## Methodology

| Step | Detail |
|------|--------|
| Input | 709 khipus; Phase 3 structural features + Phase 5 color diversity + Phase 6 anomaly scores |
| Feature set | 10 variables: `n_cords`, `n_pendants`, `n_subsidiaries`, `n_groups`, `numeric_coverage`, `frac_broken`, `n_pattern_types`, `n_unique_colors`, `sub_ratio`, `group_size` |
| Clustering | K-means (k = 2–8, 20–30 random initializations) with StandardScaler pre-processing |
| Selection | Maximum silhouette score |
| UMAP | n_neighbors = 15, min_dist = 0.1 (2-D projection for visualization only) |
| Labels | Clusters ordered by ascending median cord count; assigned labels **T1** and **T2** |

> **Epistemological note**: T1 and T2 are computational labels derived from a
> feature distance metric. They describe measurable properties of the corpus.
> They do **not** assert administrative function, social context, or production
> intent, all of which require archaeological interpretation beyond the scope of
> this analysis.

---

## Silhouette Sweep Results

| k | Silhouette | Inertia |
|---|-----------|---------|
| **2** | **0.5603** | 5,820 |
| 3 | 0.3022 | 5,016 |
| 4 | 0.3074 | 4,446 |
| 5 | 0.2753 | 3,897 |
| 6 | 0.2844 | 3,433 |
| 7 | 0.3009 | 2,995 |
| 8 | 0.3039 | 2,704 |

The silhouette score drops by 0.25 points moving from k = 2 to k = 3 and never
recovers. The inertia elbow is diffuse with no pronounced inflection. Taken together,
these indicators confirm the binary partition discovered in Phase 3.

---

## Cluster Profiles

### T1 — Compact register group (n = 653, 92.1% of corpus)

| Feature | Mean / Median |
|---------|--------------|
| Khipu count | 653 |
| % classified Complex (Phase 3) | 10.7% |
| Median cord count | 37 |
| Mean pattern types | 2.4 |
| Mean unique colors | 7.6 |
| Mean numeric coverage | 76.7% |
| Mean fractional broken | 18.3% |
| High-confidence anomalies | 13 (2.0%) |

T1 covers the overwhelming majority of the corpus. These khipus are structurally
compact: few cords, limited color vocabulary (median ~8 unique codes), and a
restricted summation pattern repertoire (~2.4 types). Numeric coverage is high
(77%), suggesting a primary quantitative recording role. The vast majority were
classified as "Simple" in Phase 3.

### T2 — Extended register group (n = 56, 7.9% of corpus)

| Feature | Mean / Median |
|---------|--------------|
| Khipu count | 56 |
| % classified Complex (Phase 3) | 85.7% |
| Median cord count | 324 |
| Mean pattern types | 5.6 |
| Mean unique colors | 38.3 |
| Mean numeric coverage | 68.3% |
| Mean fractional broken | 16.3% |
| High-confidence anomalies | 30 (53.6%) |

T2 is dominated by the khipus identified as "Complex" in Phase 3. They carry
substantially larger cord structures (median 324 vs 37), a richer color vocabulary
(mean 38 vs 8 unique codes), and a broader summation-pattern repertoire (mean 5.6
vs 2.4 types). Numeric coverage is somewhat lower (68%), consistent with a greater
proportion of non-quantitative symbolic or categorical encoding. More than half are
flagged as high-confidence anomalies under the Phase 6 multi-method criteria,
indicating that T2 captures the structurally densest and most information-rich
portion of the corpus.

---

## Observed Contrasts

### Color vocabulary

T2 khipus use on average **5× more unique color codes** than T1 (38 vs 8). Given
that Phase 5 established color diversity as strongly correlated with pattern
complexity (p = 6.83 × 10⁻²⁵), this gap is expected — but its magnitude across
the full corpus underscores the coherence of the partition.

### Pattern prevalence

T2 exhibits higher mean prevalence across every summation-pattern flag. The
compound-pattern types (GSB, IS, CP) that Phase 2 identified as structurally
distinctive concentrate in T2.

### Anomaly overlap

The anomaly rate in T2 (54%) versus T1 (2%) is the most dramatic contrast.
This is not circular — anomaly detection in Phase 6 used a different technique
(Isolation Forest + LOF + Z-score) and a partially overlapping but not identical
feature set. The convergence indicates that "unusualness" is overwhelmingly a
property of the extended-register group.

### Geographic distribution

T2 is more strongly associated with Leymebamba (Chachapoyas), consistent with
Phase 4 findings that the Leymebamba cache drives complexity metrics. T1 is
distributed across all zones with relative uniformity.

---

## The Binary Structure as a Corpus Property

Across four independent analytical lenses applied in Phases 3–7, the same binary
structure consistently emerges:

| Phase | Method | Finding |
|-------|--------|---------|
| 3 | PCA + K-means on structural features | k = 2 optimal (silhouette = 0.37) |
| 4 | Geographic correlation analysis | Leymebamba outlier drives complexity |
| 5 | Color diversity analysis | Complex group 3× higher color diversity |
| 6 | Isolation Forest + LOF + Z-score | Anomalies concentrate in Complex group |
| **7** | **10-feature multi-dimensional clustering** | **k = 2 optimal (silhouette = 0.56)** |

That the silhouette score is *higher* in Phase 7 (0.56) than in Phase 3 (0.37) when
more features are added indicates that color and pattern evidence **reinforces** the
binary, not just replicates it. The two groups are more separable in the full
feature space than in structural features alone.

---

## Outputs

| File | Description |
|------|-------------|
| `data/processed/phase7_typology.csv` | Per-khipu cluster assignments (T1/T2) with all key features |
| `data/processed/phase7_cluster_profiles.csv` | Per-cluster feature means for all 19 structural + pattern metrics |
| `visualizations/phase7/silhouette_curve.png` | K-means sweep: silhouette score and inertia across k = 2–8 |
| `visualizations/phase7/profile_heatmap.png` | Row-normalized heatmap of feature means for T1 vs T2 |
| `visualizations/phase7/umap_typology.png` | UMAP projection colored by typology group (left) and Phase-3 cluster + anomalies (right) |
| `visualizations/phase7/cluster_zone.png` | Stacked bar: geographic zone composition by typology group |
| `visualizations/phase7/cluster_complexity.png` | Simple/Complex composition and anomaly rate per typology group |

---

## Limitations and Caveats

1. **Provenance bias**: 35% of the corpus lacks reliable geographic attribution.
   Zone-composition findings are conditional on the provenanced subset.

2. **Leymebamba concentration**: The T2 cluster is disproportionately shaped by the
   Leymebamba cache (~300 khipus from a single context). Whether T2 represents a
   coherent functional type or a site-specific archival practice cannot be resolved
   from quantitative data alone.

3. **k = 2 is analytically confirmed, not archaeologically interpreted**: The binary
   may encode any number of distinctions — material function, chronological period,
   regional tradition, preservation differential, or production context. Multiple
   competing interpretations are consistent with the current evidence.

4. **Fractional broken is not uniform**: The damaged-cord fraction is higher in T1
   than the overall corpus average, possibly reflecting differential preservation
   for smaller khipus from non-cache contexts.


