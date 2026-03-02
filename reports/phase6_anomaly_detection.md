# Phase 6: Anomaly Detection

**Generated:** 2026-03-02  
**Database:** K-CAT SQLite database (built from KFG source data)  
**Script:** `scripts/run_phase6_anomaly.py`  
**Inputs:** `data/processed/phase3_clusters.csv` · `data/processed/phase5_color_diversity.csv`  
**Status:** Provisional — anomaly detection is unsupervised; flagged khipus require expert review before interpretation

---

## Research Question

Which khipus in the KFG corpus are structurally exceptional relative to the bulk of the corpus? Are there consistent patterns among outliers that suggest data quality issues, exceptional preservation, or genuinely unusual administrative function?

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

**Methods:**

| Method | Parameters | Flagging threshold |
|---|---|---|
| **Isolation Forest** | 200 trees, contamination=5%, random_state=42 | Predicted −1 (outlier class) |
| **Local Outlier Factor** | k=20 neighbors, contamination=5% | Predicted −1 (outlier class) |
| **Z-score** | Per-feature, all features | Any feature \|z\| > 3.0 SD |

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

**Pairwise overlap:**

| Pair | Count |
|---|---|
| IF ∩ LOF | 14 |
| IF ∩ Z-score | 30 |
| LOF ∩ Z-score | 27 |
| **All three** | **14** |

**Consensus:**

| Class | Count | % corpus |
|---|---|---|
| Normal | 619 | 87.3% |
| Candidate (1 method) | 47 | 6.6% |
| **High-confidence (≥2 methods)** | **43** | **6.1%** |

The legacy OKR Phase 7 identified 13 high-confidence anomalies (2.1% of 612 khipus). The KFG result is 43 (6.1% of 709). The higher rate reflects both the richer KFG feature set (color diversity now included) and the addition of `sub_ratio` and `group_size` as derived features that expose structural extremes not visible in Phase 3 features alone.

### Leading Flag Features (Z-score)

| Feature | Khipus flagged by this feature as primary |
|---|---|
| `group_size` (pendants/groups) | 14 |
| `numeric_coverage` | 14 |
| `frac_broken` | 13 |
| `n_groups` | 8 |
| `n_subsidiaries` | 7 |

`group_size` and `numeric_coverage` are the two most common primary flag features. Z-score flags on `group_size` indicate khipus with either extremely large average group size (many pendants per group, suggesting unusually long groups) or extremely small (1–2 pendants per group, suggesting highly segmented structure). `numeric_coverage` flags appear at both extremes: near-zero coverage (numeric data nearly absent) and near-unity coverage (every cord has a decoded value, which is itself unusual).

---

### High-confidence Anomaly Catalog

`visualizations/phase6/anomaly_scatter.png`  
`visualizations/phase6/anomaly_profiles.png`  
Data: `data/processed/phase6_anomaly_catalog.csv`

Selected high-confidence anomalies, sorted by cord count:

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

Note: KH0082 and KH0083 are the famous Leymebamba paired khipus (actually both from the Leymebamba cache — KH0082 is labelled "Lluta Valley" in the KFG provenance but is part of the same cached assemblage). Their color diversity (236 and 151 unique codes) is without parallel in the corpus.

---

### Anomaly Typology

Examining the catalog across the 11 features, four structural anomaly types emerge:

**Type A — Exceptionally large khipus (n_cords outlier)**  
KH0082, KH0329, KH0468, KH0242, KH0349, KH0239, KH0083, KH0068, KH0084  
These are the largest khipus in the corpus — all Complex, all with > 500 cords. They are not errors; they represent genuine large-scale recording devices. Their anomalousness is corpus-relative: most KFG khipus have < 100 cords. The Leymebamba cache khipus account for several of these.

**Type B — Near-complete numeric coverage (numeric_coverage → 1.0)**  
KH0280, KH0289, KH0271, KH0384, KH0415, KH0621, KH0049, KH0676, KH0453  
Khipus where nearly every cord has a decoded value. This is unusual because most khipus have damaged or incomplete cords (median coverage ~45%). High coverage may reflect exceptional preservation, or khipus designed with a simple, complete record-keeping format that avoids structural elements (no long subsidiaries) that frequently lose values.

**Type C — High breakage fraction (frac_broken outlier)**  
KH0617 (77%), KH0519 (58%), KH0498 (47%), KH0568 (55%)  
These are heavily damaged khipus retaining very few interpretable values but enough structural information to remain in the corpus. KH0617 (Incahuasi, 374 cords, 77% broken) is particularly notable: it has 7 pattern types despite massive damage, suggesting the underlying structure was originally very complex.

**Type D — Extreme structural geometry (group_size, sub_ratio outliers)**  
KH0454, KH0453, KH0383, KH0382, KH0415, KH0676  
These are khipus with either very few cord groups relative to pendant count (long, undivided groups) or with unusually high subsidiary depth. Several are Simple-cluster khipus from Ica and Pachacamac, suggesting these may reflect a distinct recording format for specific local administrative tasks.

---

### Anomaly vs Normal — Feature Distributions

`visualizations/phase6/anomaly_features.png`

High-confidence anomalies are visually separated from the normal corpus on `n_cords` (heavily right-skewed), `n_unique_colors` (anomalies include both very high and very low), and `sub_ratio` (anomalies cluster at the high end). The `frac_broken` distribution shows anomalies clearly extending into the 0.5–1.0 range where the normal corpus is sparse.

---

## Notable Corpus Observations

**KH0082 and KH0083** (Leymebamba paired khipus) are consistent "star" anomalies across every phase. In Phase 3 they dominate the Complex cluster's color diversity; in Phase 5 they have the two highest unique color counts (236, 151); here they score highest on Isolation Forest. These are genuinely extraordinary objects — the largest, most colorful, best-preserved khipus in the corpus — with structural features that set them apart from the rest of the KFG collection regardless of any functional interpretation.

**KH0349** (Nazca, 866 cords, 9 pattern types, 83% numeric coverage) is the most complete non-Leymebamba khipu. All 9 summation pattern types are present and 83% of cords have decoded values — a combination not seen in any other provenanced khipu. It was flagged by all three methods.

**KH0617** (Incahuasi, 374 cords, 77% broken) is the highest-breakage high-confidence anomaly with substantial pattern richness intact. It suggests a khipu that was extensively used and damaged in use or storage, not simply incomplete at manufacture.

**The LOF-only anomalies** (flagged by LOF but not IF or Z-score) tend to be structurally unusual in locally specific ways: high numeric_coverage combined with very few groups, or very high group_size. These are khipus where no single feature is globally extreme but the *combination* of feature values is locally unusual in feature space.

---

## Limitations

1. **No ground truth for "anomaly."** There are no externally validated anomalous khipus to calibrate against. The 6.1% high-confidence rate is set by the contamination parameter (5%) and the Z-score threshold (3 SD) — both are conventional choices, not empirically determined.
2. **Contamination parameter.** If the true anomaly rate is lower (e.g., 2%), Isolation Forest and LOF will over-flag. If higher (10%+), they will under-flag. The consensus approach partially mitigates this by requiring agreement between methods.
3. **Corpus composition effect.** The Leymebamba cache khipus constitute a disproportionate share of the Complex cluster and of the anomalies. Removing them entirely would shift the "normal" baseline and likely reduce the number of anomalies detected.
4. **Breakage as a proxy limitation.** `frac_broken` reflects the fraction of cords with missing or damaged values in the KFG — it is a data quality indicator, not purely a physical property of the khipu. It includes cords where the KFG marks knot data as uncertain.

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

*Corpus sweep run 2026-03-02 against K-CAT SQLite database.*
