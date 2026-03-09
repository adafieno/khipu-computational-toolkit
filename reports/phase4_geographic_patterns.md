# Phase 4: Geographic Patterns

**Generated:** 2026-03-08  
**Database:** K-CAT SQLite database (built from KFG source data)  
**Script:** `scripts/run_phase4_geography.py`  
**Inputs:** `data/processed/phase3_clusters.csv` (Phase 3 feature matrix + cluster assignments)  
**Status:** ✅ Complete

---

## Research Question

Is structural complexity geographically patterned? Do specific summation pattern types associate preferentially with particular regions? And can structural similarity be used to attribute the 265 unprovenanced khipus (37% of corpus) to likely geographic zones?

---

## Data & Methods

**Sample:** 444 provenanced khipus across 8 geographic zones; 265 unprovenanced excluded from statistical tests.

**Zone construction:** `geo_zone` consolidates ~82 `provenance_display` site names into 8 zones (see Phase 3). Unknown / collection-only provenance → Unprovenanced.

**Analyses:**
1. **Zone summary** — complexity rate, mean cord count, mean pattern types per zone; 95% Wilson confidence intervals.
2. **Chi-square tests** — cluster × geo_zone (overall); each of 9 binary pattern flags × geo_zone (largest zones with n ≥ 10).
3. **Pattern prevalence heatmap** — mean `has_*` per zone for all 9 pattern types.
4. **Structural distributions** — box plots of `n_cords` and `n_pattern_types` per zone.
5. **Nearest-neighbor attribution** — for each unprovenanced khipu, 5 nearest neighbors in feature space among provenanced khipus (Euclidean distance over 9 binary pattern flags + scaled n_cords, n_groups, numeric_coverage); plurality-vote zone weighted by 1/distance.

---

## Results

### Complexity Rate by Zone

`visualizations/phase4/complexity_by_zone.png`

| geo_zone | n | Complex | Rate | 95% CI |
|---|---|---|---|---|
| Chachapoyas | 23 | 12 | **52%** | [33–71%] |
| Southern Highlands | 2 | 1 | 50% | — (n too small) |
| Nazca & Far South | 33 | 11 | **33%** | [19–51%] |
| North Peru Coast | 7 | 2 | 29% | — (n too small) |
| Arica & N. Chile | 11 | 3 | 27% | [9–58%] |
| Cañete–Pisco | 82 | 17 | **21%** | [13–31%] |
| Ica & Paracas | 109 | 10 | 9% | [5–16%] |
| Central Coast | 177 | 15 | **8%** | [5–14%] |
| **Provenanced total** | **444** | **71** | **16%** | |

### Overall Geographic Signal (Chi-Square)

| Test | χ² | dof | p | Significant? |
|---|---|---|---|---|
| Cluster × geo_zone | **45.98** | 7 | **< 0.0001** | ✅ Yes |

Geographic zone is a statistically significant predictor of structural complexity class (p < 10⁻⁷).

### Per-Pattern Geographic Signals

Chi-square tests on the 6 zones with n ≥ 10:

| Pattern | χ² | p | Significant? |
|---|---|---|---|
| `has_is` (indexed subsidiary sum) | **37.65** | **< 0.001** | ✅ |
| `has_sp` (subsidiary → pendant) | **32.81** | **< 0.001** | ✅ |
| `has_psn` (pendant sub neighbor) | 25.49 | 0.0001 | ✅ (PSN caveat applies) |
| `has_pp` (pendant → pendant) | 12.47 | 0.029 | ✅ |
| `has_adg` (Ascher decreasing group) | 8.69 | 0.122 | ❌ |
| `has_ip` (indexed pendant) | 6.18 | 0.289 | ❌ |
| `has_cp` (color pendant) | 5.93 | 0.313 | ❌ |
| `has_gg` (group → group) | 7.19 | 0.207 | ❌ |
| `has_gsb` (group sum bands) | 4.30 | 0.507 | ❌ |

Four patterns show significant geographic variation; five do not. IS and SP — both requiring multi-level cord hierarchy (subsidiaries contributing to parents) — are the most geographically concentrated patterns. IP, CP, GG, GSB, and ADG appear at roughly consistent rates across zones.

### Pattern Heatmap by Zone

`visualizations/phase4/pattern_heatmap_by_zone.png`

### Mean Cord Count and Pattern Types by Zone

`visualizations/phase4/structural_by_zone.png`

| geo_zone | mean n_cords | mean pattern types |
|---|---|---|
| Arica & N. Chile | 283 | 3.4 |
| Chachapoyas | 250 | 4.6 |
| North Peru Coast | 177 | 3.9 |
| Nazca & Far South | 148 | 3.1 |
| Cañete–Pisco | 84 | 2.4 |
| Ica & Paracas | 68 | 2.2 |
| Central Coast | 64 | 2.5 |

---

## Nearest-Neighbor Attribution (265 Unprovenanced Khipus)

`visualizations/phase4/nn_attribution.png`  
Data: `data/processed/phase4_nn_attribution.csv`

| Attributed zone | Count | % of unprovenanced |
|---|---|---|
| Central Coast | 147 | 55.5% |
| Ica & Paracas | 58 | 21.9% |
| Cañete–Pisco | 41 | 15.5% |
| Chachapoyas | 10 | 3.8% |
| Nazca & Far South | 8 | 3.0% |
| Arica & N. Chile | 1 | 0.4% |

**High-confidence attributions (top-zone weight ≥ 0.80): 48 khipus.**

**Important caveat:** NN attribution is based purely on structural features. It cannot distinguish between khipus that are similar due to shared origin and those that are similar for other reasons. These attributions are hypothesis-generating, not confirmatory.

> **Note on museum provenance:** `museum_country` is excluded from all analyses. It records current exhibition location, not origin.

---

## Limitations

1. **Small zone sizes.** Southern Highlands (n=2) and North Peru Coast (n=7) are too small for reliable statistics.

2. **Unprovenanced 37%.** Geographic conclusions are limited to the 444 provenanced khipus. If the unprovenanced khipus are systematically different, the provenanced sample may be biased.

3. **PSN open question.** PSN appears geographically significant but the KFG author considers the pattern likely coincidental. The IS and SP geographic signals are not dependent on PSN.

4. **NN attribution.** The nearest-neighbor model is not calibrated — no ground-truth test set is available. The 48 high-confidence attributions are 18% of total unprovenanced.

---

## How to Re-run

```bash
python scripts/run_phase4_geography.py
```

Reads `data/processed/phase3_clusters.csv` (Phase 3 must have run first).

| Output | Description |
|---|---|
| `data/processed/phase4_zone_summary.csv` | Per-zone aggregate stats |
| `data/processed/phase4_chi2_results.csv` | Chi-square test results |
| `data/processed/phase4_nn_attribution.csv` | NN zone attribution for 265 unprovenanced |
| `visualizations/phase4/` | All PNG figures |

---

*See [Citations and Acknowledgments](../README.md#citations-and-acknowledgments) in the project README for primary sources, data attribution, and toolkit provenance.*

---

*Corpus sweep run against K-CAT SQLite database. Re-run with `scripts/run_phase4_geography.py` to refresh.*
