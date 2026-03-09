# Phase 7: Multi-feature Typology

**Generated:** 2026-03-08  
**Database:** K-CAT SQLite database (built from KFG source data)  
**Script:** `scripts/run_phase7_typology.py`  
**Inputs:** Phase 3 structural features + Phase 5 color diversity + Phase 6 anomaly scores  
**Status:** ✅ Complete

---

## Research Question

Does the corpus separate into more than two coherent groups when structural, summation, color, and anomaly features are combined? Or does the Phase 3 binary partition persist?

---

## Methodology

| Step | Detail |
|---|---|
| Input | 709 khipus; Phase 3 structural features + Phase 5 color diversity + Phase 6 anomaly scores |
| Feature set | 10 variables: `n_cords`, `n_pendants`, `n_subsidiaries`, `n_groups`, `numeric_coverage`, `frac_broken`, `n_pattern_types`, `n_unique_colors`, `sub_ratio`, `group_size` |
| Clustering | K-means (k = 2–8, 20–30 random initializations) with StandardScaler pre-processing |
| Selection | Maximum silhouette score |
| Projection | PCA or UMAP (2-D projection for visualization only) |
| Labels | Clusters ordered by ascending median cord count; assigned labels **T1** and **T2** |

> T1 and T2 are computational labels derived from a feature distance metric. They describe measurable properties of the corpus. They do not assert function, social context, or production intent.

---

## Silhouette Sweep

| k | Silhouette | Inertia |
|---|---|---|
| **2** | **0.5603** | 5,820 |
| 3 | 0.3022 | 5,016 |
| 4 | 0.3074 | 4,446 |
| 5 | 0.2753 | 3,897 |
| 6 | 0.2844 | 3,433 |
| 7 | 0.3009 | 2,995 |
| 8 | 0.3039 | 2,704 |

The silhouette score drops by 0.25 points from k = 2 to k = 3 and does not recover. The inertia elbow is diffuse. k = 2 is the optimal partition.

---

## Cluster Profiles

### T1 (n = 653, 92.1%)

| Feature | Value |
|---|---|
| % classified Complex (Phase 3) | 10.7% |
| Median cord count | 37 |
| Mean pattern types | 2.4 |
| Mean unique colors | 7.6 |
| Mean numeric coverage | 76.7% |
| Mean frac broken | 18.3% |
| High-confidence anomalies | 13 (2.0%) |

### T2 (n = 56, 7.9%)

| Feature | Value |
|---|---|
| % classified Complex (Phase 3) | 85.7% |
| Median cord count | 324 |
| Mean pattern types | 5.6 |
| Mean unique colors | 38.3 |
| Mean numeric coverage | 68.3% |
| Mean frac broken | 16.3% |
| High-confidence anomalies | 30 (53.6%) |

T2 khipus have larger cord structures (median 324 vs 37), more distinct color codes (mean 38 vs 8), broader pattern repertoire (mean 5.6 vs 2.4 types), and a lower numeric coverage rate (68% vs 77%). More than half of T2 are flagged as Phase 6 high-confidence anomalies.

---

## Observed Contrasts

### Color vocabulary
T2 khipus use on average **5× more unique color codes** than T1 (38 vs 8).

### Pattern prevalence
T2 has higher mean prevalence across every summation-pattern flag. The compound-pattern types (GSB, IS, CP) concentrate in T2.

### Anomaly overlap
Anomaly rate: T2 = 54%, T1 = 2%. Phase 6 anomaly detection used a different technique (IF + LOF + Z-score) and a partially overlapping feature set — the convergence is not circular.

### Geographic distribution
T2 is more strongly associated with Leymebamba (Chachapoyas), consistent with Phase 4 findings. T1 is distributed across all zones.

---

## Outputs

| File | Description |
|---|---|
| `data/processed/phase7_typology.csv` | Per-khipu T1/T2 assignment with all key features |
| `data/processed/phase7_cluster_profiles.csv` | Per-cluster feature means |
| `visualizations/phase7/silhouette_curve.png` | Silhouette and inertia across k = 2–8 |
| `visualizations/phase7/profile_heatmap.png` | Row-normalized feature heatmap T1 vs T2 |
| `visualizations/phase7/umap_typology.png` | Projection colored by typology group |
| `visualizations/phase7/cluster_zone.png` | Geographic zone composition by typology group |
| `visualizations/phase7/cluster_complexity.png` | Simple/Complex composition and anomaly rate per group |

---

## Limitations

1. **Provenance bias.** 35% of the corpus lacks reliable geographic attribution. Zone-composition findings are conditional on the provenanced subset.
2. **Leymebamba concentration.** T2 is disproportionately shaped by the Leymebamba cache (~300 khipus from a single context). Whether T2 reflects a coherent type or a site-specific collection effect cannot be resolved from these data alone.
3. **k = 2 is statistically confirmed, not interpreted.** The binary may encode material function, chronological period, regional tradition, preservation differential, production context, or other distinctions. Multiple explanations remain consistent with the data.

---

*Corpus sweep run against K-CAT SQLite database. Re-run with `scripts/run_phase7_typology.py` to refresh.*
