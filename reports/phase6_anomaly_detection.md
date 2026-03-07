# Phase 6: Anomaly Detection

**Generated:** 2026-03-02 (updated)  
**Database:** K-CAT SQLite database (built from KFG source data)  
**Script:** `scripts/run_phase6_anomaly.py`  
**Inputs:** `data/processed/phase3_clusters.csv` · `data/processed/phase5_color_diversity.csv`  
**Status:** ✅ Complete — anomaly detection is unsupervised; flagged khipus require expert review

---

## Research Question

Which khipus in the KFG corpus are structurally exceptional relative to the bulk of the corpus? Are there consistent structural patterns among outliers?

---

## Methods

Three complementary anomaly detection methods applied to an 11-feature structural matrix per khipu.

**Feature set:**

| Feature | Description |
|---|---|
| `n_cords` | Total cord count |
| `n_pendants` | Pendant (level-0) cord count |
| `n_subsidiaries` | Subsidiary cord count (levels 1+) |
| `n_groups` | Number of cord groups |
| `numeric_coverage` | Fraction of cords with a decoded numeric value |
| `frac_broken` | Fraction of cords marked as broken/damaged |
| `n_colors` | Simple color count (Phase 3) |
| `n_pattern_types` | Count of distinct summation pattern types present |
| `n_unique_colors` | Normalized color code count (Phase 5) |
| `sub_ratio` | Subsidiaries / pendants |
| `group_size` | Pendants / groups |

All features StandardScaler-normalized before model fitting.

| Method | Parameters | Flagging threshold |
|---|---|---|
| **Isolation Forest** | 200 trees, contamination=5%, random_state=42 | Predicted −1 (outlier class) |
| **Local Outlier Factor** | k=20 neighbors, contamination=5% | Predicted −1 (outlier class) |
| **Z-score** | Per-feature, all features | Any feature |z| > 3.0 SD |

**Consensus classification:**
- **High-confidence anomaly**: flagged by ≥ 2 of 3 methods
- **Candidate anomaly**: flagged by exactly 1 method
- **Normal**: flagged by 0 methods

---

## Results

### Method Agreement

`visualizations/phase6/anomaly_method_venn.png`

| Method | Flagged | % corpus |
|---|---|---|
| Isolation Forest | 36 | 5.1% |
| Local Outlier Factor | 36 | 5.1% |
| Z-score (any feature > 3 SD) | 75 | 10.6% |

| Pair | Count |
|---|---|
| IF ∩ LOF | 14 |
| IF ∩ Z-score | 30 |
| LOF ∩ Z-score | 27 |
| **All three** | **14** |

### Consensus

| Class | Count | % corpus |
|---|---|---|
| Normal | 619 | 87.3% |
| Candidate (1 method) | 47 | 6.6% |
| **High-confidence (≥ 2 methods)** | **43** | **6.1%** |

### Leading Flag Features (Z-score)

| Feature | Khipus flagged by this feature as primary |
|---|---|
| `group_size` (pendants/groups) | 14 |
| `numeric_coverage` | 14 |
| `frac_broken` | 13 |
| `n_groups` | 8 |
| `n_subsidiaries` | 7 |

---

### High-confidence Anomaly Catalog (Selected)

`visualizations/phase6/anomaly_scatter.png`  
`visualizations/phase6/anomaly_profiles.png`  
Data: `data/processed/phase6_anomaly_catalog.csv`

| kfg_id | Provenance | Zone | Cluster | n_cords | n_patterns | n_uniq_colors | Numeric cov. | Frac broken | Primary flag | Methods |
|---|---|---|---|---|---|---|---|---|---|---|
| **KH0082** | Lluta Valley | Arica & N. Chile | Complex | **1,831** | 7 | 236 | 22% | 0.2% | n_groups | IF+LOF+Z |
| **KH0329** | Unknown | — | Complex | **1,227** | 8 | 129 | 28% | 7.8% | n_pendants | IF+LOF+Z |
| KH0468 | Ica / Pisco | Ica & Paracas | Complex | 955 | 6 | 54 | 24% | 17.2% | n_cords | IF+LOF+Z |
| KH0242 | Leymebamba | Chachapoyas | Complex | 874 | 9 | 8 | 63% | 0.1% | n_cords | IF+LOF+Z |
| KH0349 | Nazca | Nazca & Far South | Complex | 866 | 9 | 37 | 83% | 6.6% | n_pendants | IF+LOF+Z |
| KH0239 | Leymebamba | Chachapoyas | Complex | 758 | 6 | 36 | 61% | 11.5% | n_pendants | IF+LOF+Z |
| KH0083 | Mollepampa | Chachapoyas | Complex | 591 | 5 | 151 | 60% | 0% | n_subsidiaries | IF+LOF+Z |
| KH0617 | Incahuasi | Cañete–Pisco | Complex | 374 | 7 | 12 | 51% | **77.3%** | n_subsidiaries | IF+Z |
| KH0135 | Pachacamac | Central Coast | Complex | 281 | 2 | 3 | **7.5%** | 0.7% | numeric_coverage | LOF+Z |
| KH0289 | Unknown | — | Complex | 180 | 4 | 4 | **97.8%** | 6.1% | n_groups | LOF+Z |
| KH0384 | Pisco Valley | Cañete–Pisco | Complex | 96 | 3 | 2 | **100%** | 0% | n_groups | LOF+Z |
| KH0271 | Huari | S. Highlands | Complex | 91 | 4 | 2 | **100%** | 2.2% | n_groups | LOF+Z |

All 43 high-confidence anomalies are Complex-cluster khipus. The full catalog is in `data/processed/phase6_anomaly_catalog.csv`.

### Structural Patterns Among Anomalies

Anomalies group into four structural profiles based on their primary flag features:

1. **Large cord count** (KH0082, KH0329, KH0468, KH0242, KH0349, KH0239, KH0083): > 500 cords. These are the largest khipus in the corpus. The Leymebamba cache accounts for several.

2. **Near-complete numeric coverage** (KH0280, KH0289, KH0271, KH0384): numeric coverage > 95%. Most khipus have coverage ~45%; these are well above corpus norms.

3. **High breakage fraction** (KH0617 at 77%, KH0519 at 58%, KH0498 at 47%): heavily damaged but retaining enough structure for analysis.

4. **Extreme structural geometry** (KH0454, KH0453, KH0383): either very few cord groups relative to pendant count or unusually high subsidiary depth.

### Anomaly vs Normal — Feature Distributions

`visualizations/phase6/anomaly_features.png`

High-confidence anomalies separate from the normal corpus primarily on `n_cords` (right tail), `n_unique_colors` (both extremes), `sub_ratio` (high end), and `frac_broken` (0.5–1.0 range).

---

## Limitations

1. **No ground truth for "anomaly."** No externally validated anomalous khipus exist to calibrate against. The 6.1% high-confidence rate is set by the contamination parameter (5%) and Z-score threshold (3 SD) — conventional choices, not empirically determined.
2. **Contamination parameter.** If the true anomaly rate differs from 5%, IF and LOF flag counts will be correspondingly biased.
3. **Leymebamba concentration.** Leymebamba cache khipus constitute a disproportionate share of both the Complex cluster and the anomalies. Removing them would shift the baseline.
4. **Breakage as proxy.** `frac_broken` reflects missing or damaged values in the KFG — a data quality indicator that includes cords where the KFG marks knot data as uncertain.

---

## How to Re-run

```bash
python scripts/run_phase6_anomaly.py
```

Requires Phase 3 and Phase 5 outputs.

| Output | Description |
|---|---|
| `data/processed/phase6_anomaly_scores.csv` | Full corpus with all anomaly scores and flags |
| `data/processed/phase6_anomaly_catalog.csv` | Flagged khipus only (90 total: 43 high-conf + 47 candidates) |
| `visualizations/phase6/anomaly_scatter.png` | n_cords vs coverage and color diversity scatter |
| `visualizations/phase6/anomaly_features.png` | Feature distribution: normal vs high-confidence |
| `visualizations/phase6/anomaly_method_venn.png` | Consensus class distribution + method overlap counts |
| `visualizations/phase6/anomaly_profiles.png` | Normalized feature profiles for top-20 anomalies |

---

*Corpus sweep run against K-CAT SQLite database. Re-run with `scripts/run_phase6_anomaly.py` to refresh.*
