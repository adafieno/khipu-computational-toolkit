# Phase 5: Color Analysis

**Generated:** 2026-03-02 (updated)  
**Database:** K-CAT SQLite database (built from KFG source data)  
**Script:** `scripts/run_phase5_color.py`  
**Inputs:** `data/kfg/khipu_database.db` · `data/processed/phase3_clusters.csv`  
**Status:** ✅ Complete

---

## Research Questions

1. What is the color vocabulary of the KFG corpus? How concentrated is it?
2. Does having a white cord in the first position of a cord group associate with higher summation complexity?
3. Is color diversity associated with structural complexity cluster (Simple vs. Complex)?
4. Do cord value distributions differ by color code?
5. Which colors co-occur across the corpus?

---

## Data

- **62,746 cord records** in the KFG database; **76,258 cord-color entries** (compound cords contribute multiple rows)
- `cord_colors` table: normalized — compound color strings such as `W:MB` are split into individual components (`W` at sequence_ord=0, `MB` at sequence_ord=1)
- `cords.color` retains the original compound string for reference
- **20.3%** of cords have compound colors (two or more components)

---

## Results

### 1. Color Vocabulary

`visualizations/phase5/color_vocab.png`  
Data: `data/processed/phase5_color_vocab.csv`

| color_code | n_entries | % entries | n_khipus | % khipus |
|---|---|---|---|---|
| **W** (white) | 20,936 | 27.5% | 551 | **77.7%** |
| AB (mottled buff) | 11,170 | 14.6% | 397 | 56.0% |
| MB (mottled brown) | 9,291 | 12.2% | 416 | 58.7% |
| YB (yellowish brown) | 4,729 | 6.2% | 208 | 29.3% |
| KB (khaki brown) | 3,921 | 5.1% | 309 | 43.6% |
| B (brown) | 3,251 | 4.3% | 136 | 19.2% |
| GG (grayish green) | 1,559 | 2.0% | 174 | 24.5% |
| LB (light brown) | 1,401 | 1.8% | 70 | 9.9% |
| NB (natural brown) | 1,349 | 1.8% | 45 | 6.3% |
| DB (dark brown) | 1,120 | 1.5% | 74 | 10.4% |

**Total distinct normalized color codes: 2,830.**

The top 10 codes account for approximately 77% of all cord-color entries. White is the most common single code and appears in 77.7% of khipus. The 2,830 distinct codes include many rare compound combinations unique to individual khipus.

**Note on color encoding**: The KFG uses extended Ascher codes (not raw Munsell values) with substantially more granular compound coding than the OKR: `W-KB`, `W-AB`, and other hyphenated codes appear in the `cord_colors` table as compound entries distinct from simple `W` or `KB`. This is a KFG-specific encoding convention and does not reflect greater actual chromatic diversity compared to OKR estimates.

---

### 2. White Cord First-Position Test

`visualizations/phase5/white_cord_analysis.png`

**Operationalization:** A khipu is coded `has_white_first_cord = True` if any pendant cord (`hierarchy_level = 0`) in any of its cord groups has `position_in_group = 1` and a color beginning with `W`.

| Group | n khipus | Mean pattern types | Complex rate |
|---|---|---|---|
| No white first cord | 287 | 2.22 | 14.3% |
| Has white first cord | 422 | **2.92** | 18.2% |

| Test | Result | Significant? |
|---|---|---|
| Pattern types: Mann-Whitney U (greater) | p < 0.0001 | ✅ |
| Cluster (Simple/Complex): chi-square | χ² = 1.66, p = 0.198 | ❌ |

Khipus with white first-position cords have significantly more pattern types on average (+0.70, p < 0.0001), but the difference in Complex classification rate is not statistically significant.

**Caveat**: The KFG `position_in_group` column encodes position within a cord group, not ordinal position across the whole khipu. The original Clindaniel hypothesis was operationalized on OKR's `cord_ordinal` (global position). Results may differ under alternative operationalizations.

---

### 3. Color Diversity by Cluster

`visualizations/phase5/color_diversity_by_cluster.png`  
Data: `data/processed/phase5_color_diversity.csv`

| Cluster | n | Mean unique colors | Median |
|---|---|---|---|
| Simple | 591 | 7.3 | 5 |
| Complex | 118 | **23.6** | 17 |

Mann-Whitney U (Complex > Simple): **p = 6.83 × 10⁻²⁵**

Complex khipus use on average 3.2× as many distinct color codes as Simple khipus. Notable outliers: KH0082 (236 unique colors) and KH0083 (151 unique colors), both from the Leymebamba cache, substantially influence the Complex mean.

---

### 4. Color-Value Association

`visualizations/phase5/color_value_correlation.png`

**Test:** Kruskal-Wallis H-test across 12 most common color codes, restricted to cords with non-zero numeric values.

| Statistic | Value |
|---|---|
| H | **987.18** |
| p | **1.10 × 10⁻²⁰⁴** |

**Median non-zero cord value by color code (top 12):**

| Color | Median value |
|---|---|
| NB (natural brown) | 42 |
| DB (dark brown) | 15 |
| W (white) | 13 |
| AB (mottled buff) | 10 |
| YB (yellowish brown) | 10 |
| B (brown) | 10 |
| HB (hot brown) | 7.5 |
| GG (grayish green) | 6 |
| MB (mottled brown) | 6 |
| LB (light brown) | 6 |
| KB (khaki brown) | 6 |
| RB (reddish brown) | 6 |

The association is highly significant, but multiple confounds are present: cord position (NB and DB appear disproportionately on specific hierarchy levels), corpus composition (NB appears in only 45 khipus), and khipu-level effects (khipus recording large values may use certain colors). Causal direction is not established.

---

### 5. Color Co-occurrence

`visualizations/phase5/color_cooccurrence.png`

The co-occurrence matrix counts khipus containing both color X and color Y. Selected pairings:

| Pair | Co-occurring khipus |
|---|---|
| W + MB | 336 |
| W + AB | 325 |
| AB + MB | 322 |
| W + GG | 158 |
| AB + GG | 150 |
| MB + GG | 148 |

W co-occurs with nearly every other major color, as expected given its 77.7% corpus presence. AB and MB appear together nearly as often as either appears alone (322 joint vs 397/416 individual). LB and NB show more isolated co-occurrence patterns with fewer pairings to the dominant AB/MB group.

---

## Limitations

1. **NB sample size.** Only 45 khipus carry NB cords; the high median value (42) is based on a small sample.
2. **White-first operationalization.** `position_in_group = 1` may not map exactly onto the Clindaniel/Ascher concept.
3. **Color-diversity outliers.** KH0082 (236 unique colors) and KH0083 (151) heavily influence the Complex mean. Both are from the Leymebamba cache and are also outliers in cord count and summation coverage.

---

## How to Re-run

```bash
python scripts/run_phase5_color.py
```

Requires Phase 3 to have run first (reads `data/processed/phase3_clusters.csv`).

| Output | Description |
|---|---|
| `data/processed/phase5_color_vocab.csv` | Color frequency table |
| `data/processed/phase5_color_diversity.csv` | Per-khipu color diversity metrics |
| `data/processed/phase5_stat_results.csv` | Statistical test results |
| `visualizations/phase5/` | All PNG figures |

---

## Citations and Acknowledgments

White-cord hypothesis after Clindaniel (2019) and Ascher & Ascher (1997). Color codes follow KFG extended Ascher notation.

> Khosla, Ashok. *The Khipu Field Guide.* [khipufieldguide.com](https://khipufieldguide.com), 2020–present.

---

*Corpus sweep run against K-CAT SQLite database. Re-run with `scripts/run_phase5_color.py` to refresh.*
