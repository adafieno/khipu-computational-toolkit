# Phase 11: Color × Value Interaction

**Generated:** 2026-03-02 (updated)  
**Database:** K-CAT SQLite database (built from KFG source data)  
**Script:** `scripts/run_phase11_color.py`  
**Inputs:** `data/kfg/khipu_database.db` · Phase 7/8 assignments  
**Status:** ✅ Complete

---

## Research Question

Does cord color associate with value order-of-magnitude, hierarchy level, summation compliance, attachment type, or behavioral cluster?

Five hypotheses tested. All five significant at p < 0.001.

---

## Methods

### Color encoding

The KFG `cord_colors` table stores one or more color codes per cord in `sequence_ord` order. The **primary color** (`sequence_ord = 0`) is the outermost/dominant color. Composite colors (e.g., `W%AB`, `MB-KB`) indicate banded or variegated cords.

Only cords with a primary color code occurring ≥ 50 times in the corpus are retained (59 qualifying codes). Tests use non-zero values only (zero-value cords excluded per Phase 10).

---

## Results

### H1 — Primary color predicts value order-of-magnitude

**Kruskal-Wallis: H = 1,419.0, p = 3.6 × 10⁻²⁶⁷**

| Primary color | n (non-zero) | Median value | ≥ 100 % | ≥ 1,000 % |
|---|---|---|---|---|
| W%AB (banded) | 83 | 126.0 | 54.2% | 10.8% |
| DB-CB (banded) | 44 | 125.0 | 97.7% | 0.0% |
| MB-KB (banded) | 37 | 90.0 | 48.6% | 13.5% |
| YG | 224 | 80.5 | 45.5% | 4.9% |
| W%MB (banded) | 101 | 70.0 | 41.6% | 4.0% |
| NB | 801 | 42.0 | 32.7% | 9.0% |
| W (white) | 14,443 | 13.0 | 17.9% | 4.5% |
| B (brown) | 2,254 | 6.0 | 10.5% | 1.4% |

Composite/banded colors consistently carry higher median values than monochrome codes. The `W%AB`, `DB-CB`, and `W%MB` medians are ≥ 70, compared to plain W at 13. Banded-color sample sizes are small (37–101 cords).

### H2 — Color distribution shifts by hierarchy level

**Chi-square (level-0 vs level-1, top-15 colors): χ² = 1,267, df = 14, p = 7 × 10⁻²⁶²**

| Color | Level-0 % | Level-1 % | Level-2 % |
|---|---|---|---|
| W (white) | 37.0 | 30.3 | 31.2 |
| YB | 8.6 | 4.8 | 3.4 |
| KB | 2.2 | 6.3 | 6.0 |
| AB | 15.4 | 15.8 | 21.4 |
| B | 5.1 | 4.5 | 2.3 |

Color frequencies are not uniform across hierarchy levels. YB decreases from 8.6% at level 0 to 3.4% at level 2. KB increases from 2.2% at level 0 to 6.0% at levels 1–2. AB increases from 15.4% to 21.4% at level 2.

### H3 — Color in summation-compliant groups

The 680 compliant parent-child groups show a parent-child same-color match rate of **30.6%**, compared to 27.7% in the `sub` (non-compliant) class. The difference is modest (2.9 percentage points). W, AB, and MB dominate both compliant and non-compliant groups, reflecting corpus-wide prevalence.

### H4 — Attachment type modulates the color-value signal

**Kruskal-Wallis color × value is significant for all three attachment types:**

| Attachment | H-statistic | p-value |
|---|---|---|
| U (upward) | 1,285.6 | 2.0 × 10⁻²³⁸ |
| R (recto) | 695.0 | 5.4 × 10⁻¹²⁷ |
| V (verso) | 521.1 | 1.3 × 10⁻⁹¹ |

V-attached cords record higher median values than U-attached cords for the same color (e.g., AB: V = 14, R = 11, U = 5).

### H5 — Color composition differs by behavioral cluster

**Chi-square: χ² = 3,482, df = 70, p ≈ 0**

| Cluster | Notable color feature |
|---|---|
| B1 | High LB (13.4% vs corpus 2.0%) |
| B2 | Elevated MB (16.9%) and YB (12.1%) |
| B3 | Elevated B (7.3%) |
| B5 | Highest W (43.8%) |
| B6 | Elevated B (8.1%) and NB (5.1%) |

---

## Limitations

- **Color code transcription** relies on expert visual assessment of aged fibers; some codes may be misidentified due to pigment degradation.
- **Banded-color sample size.** The top banded colors have 37–101 cords each; results are directional.
- **Provenance uncertainty.** ~40% of khipus lack reliable provenance (Phase 4), potentially masking regional color conventions.

---

## Outputs

| File | Description |
|---|---|
| `data/processed/phase11_color_value.csv` | Per-cord: primary color, value, hierarchy level, behavioral label |
| `data/processed/phase11_color_stats.csv` | Per-color: n, median, % ≥ 100, % ≥ 1,000 |
| `visualizations/phase11/color_value_boxplot.png` | H1: color × log-value boxplot |
| `visualizations/phase11/color_by_level.png` | H2: color distribution at levels 0, 1, 2 |
| `visualizations/phase11/color_compliance.png` | H3: color in compliant vs non-compliant groups |
| `visualizations/phase11/attachment_color.png` | H4: color × value by attachment type |
| `visualizations/phase11/color_cluster_heatmap.png` | H5: color composition heatmap across behavioral clusters |

---

*Corpus sweep run against K-CAT SQLite database. Re-run with `scripts/run_phase11_color.py` to refresh.*
