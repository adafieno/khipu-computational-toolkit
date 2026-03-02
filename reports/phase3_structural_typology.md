````markdown
# Phase 3: Structural Typology

**Generated:** [TBD — run `scripts/run_phase3_typology.py` to populate]  
**Database:** K-CAT SQLite database (built from KFG source data)  
**Script:** `scripts/run_phase3_typology.py`  
**Feature matrix:** `src/analysis/feature_matrix.py`  
**Status:** 🔄 Pending (awaiting KFG team responses before publishing — see open questions in Phase 2)

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

`region`, `provenance`, `museum_country`, `creation_date`

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
| Khipus in matrix | [TBD] |
| Columns | [TBD] |
| With any pattern (`n_pattern_types` > 0) | [TBD] ([TBD]%) |

### Silhouette Analysis

| Best k | Silhouette score |
|--------|-----------------|
| [TBD] | [TBD] |

Full curve saved to `visualizations/phase3/silhouette_curve.png`.

### Cluster Summary (k = [TBD])

| Cluster | Size | Top patterns | Mean n_cords | Top countries |
|---------|------|--------------|--------------|---------------|
| 1 | [TBD] | [TBD] | [TBD] | [TBD] |
| 2 | [TBD] | [TBD] | [TBD] | [TBD] |
| … | | | | |

### Pattern Prevalence Heatmap

`visualizations/phase3/heatmap_cluster_patterns.png`

*[Insert findings — e.g. whether any cluster is defined primarily by the absence of PP, or by the combination of CP+IP, etc.]*

### 2-D Embedding

`visualizations/phase3/umap_by_cluster.png` / `umap_by_n_types.png` / `umap_by_country.png`

*[Insert findings — does the embedding show a continuum or discrete islands? Does provenance separate?]*

### Structural Extremes

**Khipus with all [TBD] pattern types (n = [TBD]):**

| kfg_id | cluster | n_cords | region |
|--------|---------|---------|--------|
| [TBD] | | | |

*[Are these structurally exceptional — e.g. unusually large cordage, specific provenance?]*

**Khipus with exactly 1 pattern type (n = [TBD]):**

| kfg_id | cluster | pattern | n_cords |
|--------|---------|---------|---------|
| [TBD] | | | |

*[Which pattern is most common as a singleton? Does single-pattern correlate with small cord count?]*

---

## Cross-tabulation: Clusters vs. Provenance

*[Fill from `print_crosstabs()` console output.]*

| museum_country | Cluster 1 | Cluster 2 | … |
|---------------|-----------|-----------|---|
| [TBD] | | | |

---

## Limitations

1. **Cluster stability.** K-means is sensitive to initialisation and distance metric. The binary pattern columns have equal weight in Euclidean distance; Hamming distance or Jaccard similarity may be more appropriate for binary vectors. Alternative clusterings (hierarchical Ward, DBSCAN) are left for follow-up.

2. **Pattern flag quality.** The flags inherit Phase 2 limitations: PSN is tentative (KFG author considers it likely coincidental); IP has the highest false-positive rate (89 FPs vs KFG). If those flags are noisy, the IP and PSN columns add noise to the clustering input. This is the primary motivation for the "not yet for publication" status.

3. **Metadata sparsity.** `provenance` and `region` are populated for a fraction of the 709 khipus; geographic cross-tabulation is therefore indicative, not comprehensive.

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

*Run `scripts/run_phase3_typology.py` to populate all [TBD] values and regenerate visualisations.*
````
