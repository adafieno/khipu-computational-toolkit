# Phase 4: Geographic Patterns

**Generated:** 2026-03-02  
**Database:** K-CAT SQLite database (built from KFG source data)  
**Script:** `scripts/run_phase4_geography.py`  
**Inputs:** `data/processed/phase3_clusters.csv` (Phase 3 feature matrix + cluster assignments)  
**Status:** Provisional — PSN and PP open questions from Phase 2 not yet resolved; findings may shift for ~150 khipus

---

## Research Question

Is structural complexity geographically patterned? Do specific summation pattern types associate preferentially with particular regions? And can structural similarity be used to attribute the 265 unprovenanced khipus (37% of corpus) to likely geographic zones?

---

## Data & Methods

**Sample:** 444 provenanced khipus across 8 geographic zones; 265 unprovenanced excluded from statistical tests.

**Zone construction:** `geo_zone` consolidates ~82 `provenance_display` site names into 8 zones (see Phase 3). Unknown / collection-only provenance → Unprovenanced.

**Analyses:**
1. **Zone summary** — complexity rate, mean cord count, mean pattern types per zone; 95% Wilson confidence intervals.
2. **Chi-square tests** — cluster × geo_zone (overall); each of 9 binary pattern flags × geo_zone (5 largest zones, n ≥ 10, excluding Southern Highlands and North Peru Coast).
3. **Pattern prevalence heatmap** — mean `has_*` per zone for all 9 pattern types.
4. **Structural distributions** — box plots of `n_cords` and `n_pattern_types` per zone.
5. **Nearest-neighbour attribution** — for each unprovenanced khipu, find the 5 nearest neighbours in feature space among provenanced khipus (Euclidean distance over 9 binary pattern flags + scaled n_cords, n_groups, numeric_coverage); assign the plurality-vote zone weighted by 1/distance.

---

## Results

### Complexity Rate by Zone

`visualizations/phase4/complexity_by_zone.png`

| geo_zone | n | Complex | Rate | 95% CI |
|---|---|---|---|---|
| **Chachapoyas** | 23 | 12 | **52%** | [33–71%] |
| Southern Highlands | 2 | 1 | 50% | — (n too small) |
| **Nazca & Far South** | 33 | 11 | **33%** | [19–51%] |
| North Peru Coast | 7 | 2 | 29% | — (n too small) |
| Arica & N. Chile | 11 | 3 | 27% | [9–58%] |
| **Cañete–Pisco** | 82 | 17 | **21%** | [13–31%] |
| Ica & Paracas | 109 | 10 | 9% | [5–16%] |
| **Central Coast** | 177 | 15 | **8%** | [5–14%] |
| **Provenanced total** | **444** | **71** | **16%** | |

Corpus-average complex rate among provenanced khipus: **16%** (71/444).

The gradient is striking: khipus from **Chachapoyas** (Leymebamba cache / Laguna de los Cóndors) are more than 6× more likely to be Complex than those from the **Central Coast** (Pachacamac-dominated). The [33–71%] Wilson CI for Chachapoyas does not include 8%, confirming this is a real structural difference and not sampling noise.

### Overall Geographic Signal (Chi-Square)

| Test | χ² | dof | p | Significant? |
|---|---|---|---|---|
| Cluster × geo_zone | **45.98** | 7 | **< 0.0001** | ✅ Yes |

Geographic zone is a **highly significant** predictor of structural complexity class (p < 10⁻⁷). This is the strongest statistical result in the K-CAT corpus to date.

### Per-Pattern Geographic Signals

Chi-square tests run on the 6 zones with n ≥ 10 (Central Coast, Cañete–Pisco, Ica & Paracas, Nazca & Far South, Chachapoyas, Arica & N. Chile):

| Pattern | χ² | p | Significant? |
|---|---|---|---|
| `has_is` (indexed subsidiary sum) | **37.65** | **< 0.001** | ✅ |
| `has_sp` (subsidiary → pendant) | **32.81** | **< 0.001** | ✅ |
| `has_psn` (pendant sub neighbor) | 25.49 | 0.0001 | ⚠️ (PSN caveat) |
| `has_pp` (pendant → pendant) | 12.47 | 0.029 | ✅ |
| `has_ip` (indexed pendant) | 6.18 | 0.289 | ❌ |
| `has_cp` (color pendant) | 5.93 | 0.313 | ❌ |
| `has_gg` (group → group) | 7.19 | 0.207 | ❌ |
| `has_gsb` (group sum bands) | 4.30 | 0.507 | ❌ |
| `has_adg` (Ascher decreasing group) | 8.69 | 0.122 | ❌ |

**Four patterns show significant geographic variation; five do not.** This is a meaningful structural finding:

- **IS and SP are the geographically concentrated patterns.** Both require multi-level cord hierarchy (subsidiaries contributing to parents). IS in particular is dramatically concentrated in Chachapoyas and Nazca, near-absent in Central Coast. These are precisely the patterns tied to hierarchical accounting structures — consistent with Chachapoyas khipus reflecting a different administrative function than the Pachacamac corpus.

- **IP, CP, GG, GSB, ADG are geographically diffuse.** These patterns appear at roughly consistent rates across zones, suggesting they reflect accounting conventions that were broadly shared across the Inka administrative network regardless of region.

- **PP is marginally significant (p = 0.029)**, driven mainly by the high rate in Chachapoyas (91% PP prevalence) vs. lower rates in smaller zones.

- **PSN** shows apparent significance but should be interpreted cautiously — the KFG author considers PSN likely coincidental (see Phase 2). Its "geographic signal" may partly reflect that larger khipus (which have more cords and therefore more random PSN-adjacent configurations) cluster in specific zones.

### Pattern Heatmap by Zone

`visualizations/phase4/pattern_heatmap_by_zone.png`

Chachapoyas stands out across the board — highest prevalence in PP (91%), IP (83%), SP (74%), IS (26%), PSN (87%). Central Coast is uniformly low except for PP (67%) and ADG (30%). Cañete–Pisco shows a more balanced profile across all 9 patterns.

Notable: **GSB (group sum bands)** is remarkably uniform across all zones (3–17% throughout), consistent with its very low false-positive rate in Phase 2 — it's a precise but rare pattern.

### Mean Cord Count and Pattern Types by Zone

`visualizations/phase4/structural_by_zone.png`

| geo_zone | mean n_cords | median n_cords | mean pattern types |
|---|---|---|---|
| Arica & N. Chile | 283 | — | 3.4 |
| Chachapoyas | 250 | — | 4.6 |
| North Peru Coast | 177 | — | 3.9 |
| Nazca & Far South | 148 | — | 3.1 |
| Cañete–Pisco | 84 | — | 2.4 |
| Ica & Paracas | 68 | — | 2.2 |
| Central Coast | 64 | — | 2.5 |

Arica & N. Chile has the **highest mean cord count** (283) despite only moderate complexity rate (27%). This zone includes large decorative or administrative objects that carry many cords but not necessarily the full summation pattern suite. Chachapoyas combines high cord count (250 avg) with high complexity rate — the most comprehensively complex corpus zone.

Box plots show heavy right skew in all zones (log scale used). The IQR for Central Coast starts below 30 cords and reaches ~100; for Chachapoyas the IQR spans roughly 80–400 cords.

---

## Nearest-Neighbour Attribution (265 Unprovenanced Khipus)

`visualizations/phase4/nn_attribution.png`  
Data: `data/processed/phase4_nn_attribution.csv`

Method: 5-nearest-neighbour vote in 12-dimensional feature space (9 pattern flags + scaled n_cords, n_groups, numeric_coverage), with inverse-distance weighting.

| Attributed zone | Count | % of unprovenanced |
|---|---|---|
| Central Coast | 147 | 55.5% |
| Ica & Paracas | 58 | 21.9% |
| Cañete–Pisco | 41 | 15.5% |
| Chachapoyas | 10 | 3.8% |
| Nazca & Far South | 8 | 3.0% |
| Arica & N. Chile | 1 | 0.4% |

**High-confidence attributions (top-zone weight ≥ 0.80): 48 khipus** — these have nearest neighbours concentrated strongly in one zone.

**Key observations:**

- The majority of structurally unprovenanced khipus (147/265 = 55.5%) look most like Central Coast khipus: small, few pattern types, predominantly PP and ADG. Given that the Pachacamac excavations produced hundreds of similar-looking simple khipus, this distribution is plausible.

- 10 unprovenanced khipus attribute to Chachapoyas (high cord count, multi-pattern profile). Cross-referencing these with collection metadata (Gaffron Collection, etc.) may help verify.

- **Important caveat:** NN attribution is based purely on structural features. It cannot distinguish between khipus that happen to look similar for other reasons (e.g., functional convergence across regions) and those that truly originate in a given zone. These attributions are hypothesis-generating, not confirmatory.

---

## Geographic Interpretation

The K-CAT results are consistent with a model where **khipu structural complexity reflects administrative function**, with function varying systematically by region:

- **Chachapoyas / Nazca** — high complexity rates, large khipus with deep subsidiary hierarchies. Consistent with regional administrative centers managing complex multi-level census or tribute records (Leymebamba cache context).
- **Central Coast (Pachacamac dominant)** — predominantly simple khipus with 1–2 pattern types, small cord counts. May reflect a different recording function — possibly shrine-related tallies, labor quotas, or simpler commodity accounting.
- **Cañete–Pisco** — intermediate; historically an important Inka road junction and textile production zone. Mixed profile may reflect both local administration and through-routes.

These interpretations require expert archaeological validation and should not be taken as conclusions.

> **Note on museum provenance:** `museum_country` is excluded from all analyses. It records current exhibition location, not origin. The Pachacamac corpus spans museums in Lima, Berlin, New York, and Berlin — using museum country as a geographic proxy would scramble all signals.

---

## Limitations

1. **Small zone sizes.** Southern Highlands (n=2) and North Peru Coast (n=7) are too small for reliable statistics; their complex rates (50%, 29%) cannot be interpreted confidently.

2. **Unprovenanced 37%.** Geographic conclusions are limited to the 444 provenanced khipus. If the unprovenanced khipus are systematically collected (e.g., over-representing private collections), the provenanced sample may be biased.

3. **PSN open question.** PSN appears geographically significant but the KFG author considers the pattern likely coincidental. If PSN is removed from the analysis, the IS and SP geographic signals strengthen further (they are not dependent on PSN).

4. **NN attribution.** The nearest-neighbour model is not calibrated — no ground-truth test set is available. The 48 high-confidence (≥ 0.80 weight) attributions are a small fraction (18%) of total unprovenanced; the remaining 82% are attributed to zones with lower confidence.

5. **Phase 2 open questions.** PP threshold and PSN status remain pending KFG response. These affect ~150 binary flags and could shift the per-pattern chi-square results for PP and PSN specifically. The SP and IS geographic signals are robust to this uncertainty.

---

## How to Re-run

```bash
python scripts/run_phase4_geography.py
```

Reads `data/processed/phase3_clusters.csv` (Phase 3 must have run first). All outputs regenerated automatically.

| Output | Description |
|---|---|
| `data/processed/phase4_zone_summary.csv` | Per-zone aggregate stats |
| `data/processed/phase4_chi2_results.csv` | Chi-square test results |
| `data/processed/phase4_nn_attribution.csv` | NN zone attribution for 265 unprovenanced |
| `visualizations/phase4/complexity_by_zone.png` | Ranked bar chart with 95% CI |
| `visualizations/phase4/pattern_heatmap_by_zone.png` | 9 patterns × 8 zones heatmap |
| `visualizations/phase4/structural_by_zone.png` | n_cords and pattern-type box plots |
| `visualizations/phase4/nn_attribution.png` | Unprovenanced attribution summary |

---

## Citations and Acknowledgments

Geographic zone construction based on KFG `provenance_display` field, consolidated from 82 site names. Statistical methods: chi-square contingency test (scipy), Wilson score interval, sklearn nearest-neighbours. Interpretation references Bray (2012) on Leymebamba cache function and Shimada et al. (2004) on Pachacamac administrative structure.

> Khosla, Ashok. *The Khipu Field Guide.* [khipufieldguide.com](https://khipufieldguide.com), 2020–present.

---

*Corpus sweep run 2026-03-02 against K-CAT SQLite database. Re-run with `scripts/run_phase4_geography.py` to refresh.*
