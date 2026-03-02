````markdown
# Phase 3: Structural Typology

**Generated:** 2026-03-02  
**Database:** K-CAT SQLite database (built from KFG source data)  
**Script:** `scripts/run_phase3_typology.py`  
**Feature matrix:** `src/analysis/feature_matrix.py`  
**Status:** Results complete — not yet for publication (awaiting KFG team responses, see Phase 2 open questions)

---

## Research Question

Do the 9 summation pattern types cluster 709 khipus into recognisable structural *types*? Do those types correlate with known external variables — provenance, institutional origin, cord complexity, or the KFG's own use-category labels? Or does the pattern space form a continuum without clear discrete boundaries?

Phase 2 established that 80.3% of K-CAT khipus carry at least one summation pattern, and that individual khipus often carry multiple types simultaneously (55.5% carry ≥ 4). Phase 3 asks whether those co-occurrence patterns are structured — i.e., whether certain combinations recur enough to constitute archetypal khipu types.

---

## Methodology

### Feature Matrix

`src/analysis/feature_matrix.py` builds a flat per-khipu DataFrame with:

**Binary pattern flags (0/1)** — one per pattern type, using the same significance thresholds as Phase 2:

| Column | Pattern | Threshold |
|--------|---------|-----------|
| `has_pp` | `pendant_pendant_sum` | ≥ 1 match |
| `has_ip` | `indexed_pendant_sum` | ≥ 1 match |
| `has_cp` | `colored_pendant_sum` | ≥ 1 match |
| `has_sp` | `subsidiary_pendant_sum` | ≥ 1 match |
| `has_gg` | `group_group_sum` | ≥ 1 match |
| `has_gsb` | `group_sum_bands` | ≥ 1 match |
| `has_is` | `indexed_subsidiary_sum` | > 1 match |
| `has_psn` | `pendant_sub_neighbor` | > 1 match |
| `has_adg` | `ascher_decreasing_group` | ≥ 1 match |

**Structural features** (scaled before clustering):

| Column | Description |
|--------|-------------|
| `n_cords` | Total cord count |
| `n_pendants` | Level-1 cords |
| `n_subsidiaries` | Level ≥ 2 cords |
| `n_groups` | Distinct pendant groups |
| `numeric_coverage` | Fraction of cords with decoded value > 0 |
| `frac_broken` | Fraction of cords with termination = `B` |
| `n_colors` | Distinct primary color codes |
| `n_pattern_types` | Number of has_* flags equal to 1 |

**Metadata for enrichment (not used as clustering inputs):**

`region`, `provenance_display`, `creation_date`

> `museum_country` / `museum_name` are intentionally excluded. They record the current exhibition location, not the object's place of origin — inappropriate as a geographic signal for a corpus where many khipus were displaced from Peru during the colonial period.

### Clustering Approach

The clustering input `X` combines the 9 binary pattern columns with 3 scaled structural columns (`n_cords`, `n_groups`, `numeric_coverage`). Structural columns are z-scored before concatenation so they do not dominate Euclidean distance over the categorical pattern flags.

1. **K-means (k=2..10)** — silhouette score computed for each k; best k selected automatically.
2. **2-D embedding** — UMAP (`umap-learn`) if installed, PCA fallback otherwise. Used for visualisation only — does not affect cluster assignments.

### Caveat on Pending KFG Questions

The pattern flags used here are identical to Phase 2 K-CAT detections. Two open questions (the PP "significant" vs "matching" secondary threshold, and the PSN "statistical fluke" interpretation) may affect the binary flags for ~150 khipus. Cluster boundaries may shift once those questions are resolved. Accordingly this report is **not for publication** until Phase 2 open questions are answered.

---

## Results

*All figures below are placeholders — run `scripts/run_phase3_typology.py` and fill in.*

### Feature Matrix Summary

| Metric | Count |
|--------|-------|
| Khipus in matrix | 709 |
| Columns | 25 |
| With any pattern (`n_pattern_types` > 0) | 561 (79.1%) |
| With 0 patterns | 148 (20.9%) |
| With exactly 1 pattern | 126 (17.8%) |
| With 4+ patterns | 238 (33.6%) |
| With all 9 patterns | 4 (0.6%) |

### Silhouette Analysis

| k | Silhouette score |
|---|-----------------|
| 2 | **0.3698** |
| 3 | 0.3139 |
| 4 | 0.2813 |
| 5 | 0.2463 |
| 6 | 0.2323 |
| 7 | 0.2400 |
| 8 | 0.2037 |
| 9 | 0.1982 |
| 10 | 0.2009 |

**Best k = 2** (silhouette = 0.3698). The score drops monotonically from k=2, indicating no strong evidence for more than two discrete structural types in this corpus. The binary nature of the pattern vector favours two-cluster solutions in Euclidean k-means.

Full curve: `visualizations/phase3/silhouette_curve.png`.

### Cluster Summary (k = 2)

| Cluster | Size | Mean pattern types | Mean n_cords | Character |
|---------|------|--------------------|--------------|-----------|
| **0** (Simple) | 591 (83.4%) | 1.99 | 45 | Low-complexity; 1–3 patterns; small cordage |
| **1** (Complex) | 118 (16.6%) | 5.87 | 304 | High-complexity; 5–7 patterns; large cordage |

The dominant axis of variation is **size / complexity**: Cluster 1 khipus average 6.75x more cords and 2.95x more pattern types than Cluster 0. This reflects a structural continuum compressed into two groups rather than two qualitatively distinct khipu genres.

**Pattern prevalence per cluster:**

| Pattern | Cluster 0 (Simple) | Cluster 1 (Complex) |
|---------|-------------------|---------------------|
| `has_pp` | 60.9% | **96.6%** |
| `has_ip` | 30.8% | **94.1%** |
| `has_cp` | 28.6% | **88.1%** |
| `has_sp` | 19.1% | **72.9%** |
| `has_gg` | 10.7% | **50.8%** |
| `has_gsb` | 8.3% | **31.4%** |
| `has_is` | 1.9% | **45.8%** |
| `has_psn` | 12.9% | **61.9%** |
| `has_adg` | 26.1% | **45.8%** |

Every pattern is dramatically more prevalent in Cluster 1. The largest gaps are in `has_is` (+43.9 pp), `has_ip` (+63.3 pp), and `has_cp` (+59.5 pp) — patterns requiring multi-cord indexed or colour-grouped structures only possible in larger khipus.

### Pattern Prevalence Heatmap

`visualizations/phase3/heatmap_cluster_patterns.png`

Cluster 1 shows near-uniform high prevalence across all patterns except GSB (31.4%); Cluster 0 has moderate PP (60.9%) and ADG (26.1%) presence with near-zero IS (1.9%). The heatmap makes the Simple/Complex split immediately visible.

### 2-D Embedding (UMAP)

`visualizations/phase3/umap_by_cluster.png` / `umap_by_n_types.png` / `umap_by_region.png`

The UMAP projection shows a broad main mass (Cluster 0) with a satellite island of large, complex khipus (Cluster 1). The n_pattern_types view confirms the island is the 4–9 pattern zone. Region colouring shows limited geographic signal: only 142/709 khipus have a non-Unknown `region` value. Among those, Inka Late Horizon and Puruchuco objects appear in both clusters, consistent with site heterogeneity rather than site-specific structural types.

### Structural Extremes

**Khipus with all 9 pattern types (n = 4):**

| kfg_id | cluster | n_cords | region |
|--------|---------|---------|--------|
| KH0242 | 1 (Complex) | 874 | Chachapoyas |
| KH0349 | 1 (Complex) | 866 | Unknown |
| KH0433 | 1 (Complex) | 167 | Central Coast, Peru |
| KH0509 | 1 (Complex) | 362 | Unknown |

Three of the four carry over 350 cords; KH0433 (167 cords) is notable for achieving maximum pattern density in a relatively small object. KH0242 and KH0349 are among the largest khipus in the entire corpus.

**Khipus with exactly 1 pattern type (n = 126, all in Cluster 0):**

The dominant singleton is `has_pp` (~55 khipus), followed by `has_adg` (~35), `has_ip`, `has_sp`, and `has_psn`. Most single-pattern khipus are small (typical range 5–50 cords). The 6 PSN-only khipus warrant caution: given the KFG author's assessment of PSN as likely coincidental, these may not reflect deliberate structure.

---

## Cross-tabulation: Clusters vs. Origin Region

Only 142/709 khipus have a non-Unknown `region` value (20%); the remainder are unprovenanced.

| region | Cluster 0 (Simple) | Cluster 1 (Complex) | Total |
|---|---|---|---|
| Unknown | 482 | 87 | 569 |
| Inka, Late Horizon | 23 | 8 | 31 |
| Central Coast, Peru | 22 | 5 | 27 |
| Puruchuco | 23 | 1 | 24 |
| Chachapoyas | 11 | 11 | 22 |
| South Coast, Peru | 15 | 4 | 19 |
| Nazca | 6 | 0 | 6 |
| Other (Ica, Ancon, etc.) | 9 | 2 | 11 |
| **Total** | **591** | **118** | **709** |

Most provenanced khipus (Inka Late Horizon, Central Coast, Puruchuco) fall predominantly in Cluster 0. The clearest exception is **Chachapoyas**, which splits evenly (11:11) between clusters — the large, complex khipus from that site are as common as the simple ones. South Coast Peru also shows a higher Complex fraction (4/19 = 21%).

> **Note**: museum exhibition country (`museum_country`) was dropped from this analysis. It records where a khipu is currently held, not where it was made — unsuitable as a geographic proxy for a corpus displaced from Peru over centuries.

---

## Limitations

1. **Cluster stability.** K-means is sensitive to initialisation and distance metric. The binary pattern columns have equal weight in Euclidean distance; Hamming distance or Jaccard similarity may be more appropriate for binary vectors. Alternative clusterings (hierarchical Ward, DBSCAN) are left for follow-up.

2. **Pattern flag quality.** The flags inherit Phase 2 limitations: PSN is tentative (KFG author considers it likely coincidental); IP has the highest false-positive rate (89 FPs vs KFG). If those flags are noisy, the IP and PSN columns add noise to the clustering input. This is the primary motivation for the "not yet for publication" status.

3. **Provenance sparsity.** `region` is non-Unknown for only 142/709 khipus (20%); `provenance_display` (from `provenance_labels` join) covers a similar subset. Geographic cross-tabulation is therefore indicative, not comprehensive. `museum_country` is intentionally excluded — it records exhibition location, not origin.

4. **No consensus clustering.** A single k-means run is used. Ensemble clustering or stability analysis across multiple seeds and multiple k values would give stronger evidence for the identified cluster structure.

---

## How to Re-run

```bash
# Full run (builds feature matrix, runs clustering, writes outputs)
python scripts/run_phase3_typology.py

# Force-rebuild feature matrix even if cached
python scripts/run_phase3_typology.py --force

# Specify k directly (skip silhouette sweep)
python scripts/run_phase3_typology.py --k 5
```

Outputs:

| File | Description |
|------|-------------|
| `data/processed/phase3_feature_matrix.csv` | Per-khipu feature matrix |
| `data/processed/phase3_clusters.csv` | Feature matrix + cluster + embedding coordinates |
| `data/processed/phase3_silhouette.csv` | Silhouette scores for k=2..10 |
| `visualizations/phase3/` | All PNG figures |

---

## Citations and Acknowledgments

Feature matrix built from K-CAT Phase 2 detector output. Clustering approach follows standard exploratory practice (scikit-learn k-means, UMAP-Learn); no novel algorithmic contributions. Interpretation grounded in Ascher & Ascher (1981) pattern typology and KFG fieldmark definitions.

> Khosla, Ashok. *The Khipu Field Guide*. [khipufieldguide.com](https://khipufieldguide.com), 2020–present.

> Ascher, Marcia and Robert Ascher. *Mathematics of the Incas: Code of the Quipu*. Dover Publications, 1997.

---

*Corpus sweep run 2026-03-02 against K-CAT SQLite database. Re-run with `scripts/run_phase3_typology.py` (add `--force` to rebuild feature matrix from scratch).*
````
