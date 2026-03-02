# Phase 5: Color Analysis

**Generated:** 2026-03-02  
**Database:** K-CAT SQLite database (built from KFG source data)  
**Script:** `scripts/run_phase5_color.py`  
**Inputs:** `data/kfg/khipu_database.db` · `data/processed/phase3_clusters.csv`  
**Status:** Provisional — Phase 2 open questions (PP threshold, PSN interpretation) unresolved; color flags for affected khipus may shift

---

## Research Questions

1. What is the color vocabulary of the KFG corpus? How concentrated is it?
2. Does having a white cord in the first position of a cord group predict higher summation complexity? (The Clindaniel/Ascher white-cord hypothesis)
3. Is color diversity associated with structural complexity cluster (Simple vs. Complex)?
4. Do cord value distributions differ by color code — i.e., does color encode numeric magnitude?
5. Which colors co-occur across the corpus, and what does the pairing structure suggest?

---

## Data

- **62,746 cord records** in the KFG database; **76,258 cord-color entries** (compound cords contribute multiple rows)
- `cord_colors` table: normalised — compound color strings such as `W:MB` are split into individual components (`W` at sequence_ord=0, `MB` at sequence_ord=1)
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

**Total distinct normalised color codes: 2,830.**

The long tail is steep — the top 10 codes account for approximately 77% of all cord-color entries. White is by far the most common single code and appears in 77.7% of khipus. The next tier (AB, MB, KB) reflects the earthy brown palette typical of cotton and camelid fiber khipus. The 2,830 distinct codes includes many rare compound combinations unique to individual khipus.

**Important note on color encoding**: The KFG uses extended Ascher codes (not raw Munsell values), but has substantially more granular compound coding than the OKR: `W-KB`, `W-AB`, and other hyphenated codes appear in the `cord_colors` table as compound entries distinct from simple `W` or `KB`. This is a KFG-specific encoding convention and should not be read as evidence of greater actual chromatic diversity in the corpus compared to OKR estimates.

---

### 2. White Cord First-Position Hypothesis

`visualizations/phase5/white_cord_analysis.png`

**Hypothesis (after Clindaniel / Ascher):** Cords in the first position of a cord group (`position_in_group = 1`) that are white function as summation boundary markers — structurally separating or introducing a group — and should therefore be associated with higher summation-pattern richness.

**Operationalisation:** A khipu is coded `has_white_first_cord = True` if any pendant cord (`hierarchy_level = 0`) in any of its cord groups has `position_in_group = 1` and a color beginning with `W`.

| Group | n khipus | Mean pattern types | Complex rate |
|---|---|---|---|
| No white first cord | 287 | 2.22 | 14.3% |
| Has white first cord | 422 | **2.92** | 18.2% |

**Statistical tests:**

| Test | Result | Significant? |
|---|---|---|
| Pattern types: Mann-Whitney U (greater) | p < 0.0001 | ✅ |
| Cluster (Simple/Complex): chi-square | χ²=1.66, p=0.198 | ❌ |

**Interpretation:** Khipus with white first-position cords are associated with significantly more pattern *types* on average (+0.70 pattern types, p<0.0001), but this does not translate into a statistically significant uplift in the probability of being classified as "Complex" (the binary cluster from Phase 3). This nuance is important:

- White first-position cords may be a marker of more elaborately structured khipus (more diverse pattern usage) without meeting the full threshold of "Complex" (which requires high cord count + deep subsidiary hierarchy in addition to multiple pattern types).
- The effect is consistent with Ascher's original observation on summation boundaries — but the KFG's ground-truth pattern detection means we cannot interpret this as improving *detection accuracy*. Rather, the structural pattern genuinely co-occurs with white-first encoding.
- **Caveat**: The KFG `position_in_group` column encodes position within a cord group, not ordinal position across the whole khipu. The original Clindaniel hypothesis was operationalised on OKR's `cord_ordinal` (global position). Results may differ under alternative operationalisations.

---

### 3. Color Diversity by Cluster and Geographic Zone

`visualizations/phase5/color_diversity_by_cluster.png`  
Data: `data/processed/phase5_color_diversity.csv`

**Unique normalised color codes per khipu (mean values):**

| Cluster | n | Mean unique colors | Median |
|---|---|---|---|
| Simple | 591 | 7.3 | 5 |
| Complex | 118 | **23.6** | 17 |

Mann-Whitney U (Complex > Simple): **p = 6.83 × 10⁻²⁵** — extremely significant.

Complex khipus use on average 3.2× as many distinct color codes as Simple khipus. This is a strong structural signal: the same khipus that have high cord counts, deep subsidiary hierarchies, and multiple summation pattern types also employ a much richer color palette.

Two interpretations are consistent with this:
1. **Color as hierarchy marker**: In Complex khipus with deep cord hierarchies, color may be used systematically to mark different hierarchy levels or cord categories — requiring more codes.
2. **Corpus composition effect**: The Leymebamba (Chachapoyas) cache contributes many Complex khipus; those khipus include KH0082 (236 unique colors) and KH0083 (151 unique colors) which are exceptionally large and colorful. These outliers pull the Complex mean up substantially.

**By geographic zone:** The zone box plots confirm the Chachapoyas and Nazca outlier pattern seen in Phase 4. Chachapoyas khipus dominate the upper tail of color diversity, though the distribution is wide within all zones.

---

### 4. Color-Value Correlation

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

The test is highly significant, but **the practical interpretation is ambiguous**. Several confounding explanations exist:

- **Positional confound**: NB and DB are relatively rare colors that appear disproportionately on deeper subsidiary cords. Subsidiary cords in the KFG often carry smaller values (< 10) in their capacity as summands. But NB has a surprisingly high median (42) — suggesting it may preferentially appear on high-value pendant positions.
- **Corpus composition**: NB is concentrated in a small number of khipus (45 of 709). If those khipus happen to record high-magnitude data (large tribute or census counts), the color-value association is a khipu-level confound, not a cord-level semantic encoding.
- **White as sum cord**: White's median of 13 is above most brown shades (median 6), consistent with white sum cords tending to carry the total of their group — which is larger than any individual summand.

**Conclusion**: Color and value are statistically associated, but the causal direction is not established. There is no compelling evidence that color *encodes* magnitude (the legacy Phase 5 H2 was "NOT SUPPORTED" in OKR; the KFG result is more ambiguous due to the NB finding). This warrants further investigation with larger per-color samples.

---

### 5. Color Co-occurrence

`visualizations/phase5/color_cooccurrence.png`

The co-occurrence matrix counts khipus containing both color X and color Y. Key observations from the lower-triangle heatmap:

- **W + AB** pair in 325 khipus (highest non-diagonal for W); **W + MB** in 336. White co-occurs with nearly every other major color — expected given its 77.7% corpus presence.
- **AB + MB** appear together in 322 khipus — nearly as often as either appears alone. AB and MB are thus almost a "default pair" in the corpus. This is likely a fiber-type signature: the earthy (AB/MB/KB) tones reflect the natural color range of undyed camelid or cotton fiber.
- **GG (grayish green)** shows strong co-occurrence with W (158), AB (150), MB (148), and KB (131), suggesting GG appears in the same broad class of khipus as the earthy palette — not specialised to a restricted set.
- **LB and NB** are more isolated: LB co-occurs with B (19) and YB (19) but rarely with AB (11) or MB (12). NB is similarly isolated. These may represent khipus from a distinct fiber tradition or a distinct geographic origin.
- **PK (pink)** co-occurs heavily with W (127), AB (124), MB (125), KB (101) — suggesting pink cords appear in the same broadly well-preserved, multi-color khipus as the dominant palette. PK has 159 on its own diagonal (159 khipus contain pink), which is surprisingly high and may reflect a specific regional dyeing tradition (Ica/Nazca coastal cotton?).

---

## Synthesis

The five analyses converge on a consistent picture:

1. **Color vocabulary is highly concentrated but with a long tail.** White dominates; the earthy brown–buff palette (AB, MB, KB, YB) comprises the next tier. 2,830 distinct codes exist but the top 10 cover ~77% of entries.

2. **White first-position cords associate with higher summation diversity** (p<0.0001, +0.70 mean pattern types), but do not strongly separate Simple from Complex clusters. This supports a functional role for white boundary cords in structuring group-level summation — but color alone does not predict structural class.

3. **Structural complexity and color richness are tightly coupled.** Complex khipus use 3× more distinct colors than Simple ones (p<10⁻²⁴). Whether color drives complexity or complexity enables richer color use is not determinable from this data alone.

4. **Color and numeric value are statistically associated** (p<10⁻²⁰⁰) but the effect is likely mediated by cord position and khipu-level composition effects. There is no strong evidence for a simple color-encodes-magnitude rule, but NB's anomalously high median (42) merits targeted investigation.

5. **Co-occurrence reveals two partially distinct color worlds:** the ubiquitous earthy palette (W/AB/MB/KB/YB/GG) that spans most of the corpus; and isolated specialists (NB, LB, PK) concentrated in smaller subsets that may correspond to geographic traditions or functional khipu types.

---

## Limitations

1. **NB finding is tentative.** Only 45 khipus carry NB cords. The high median value (42) is driven by a small sample.
2. **White-first operationalisation**: `position_in_group = 1` may not map exactly onto the Clindaniel/Ascher concept. Groups with missing cords at position 1 would be mis-coded.
3. **Color-diversity outliers**: KH0082 (236 unique colors) and KH0083 (151) heavily influence the Complex mean. These are the Leymebamba paired khipus that are also outliers in cord count and summation coverage.
4. **Phase 2 open questions not resolved.** PSN and PP threshold uncertainty does not directly affect color analyses, but `n_pattern_types` values for ~150 khipus may shift slightly.

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
| `visualizations/phase5/color_vocab.png` | Top 30 color codes bar chart |
| `visualizations/phase5/white_cord_analysis.png` | White first-cord hypothesis |
| `visualizations/phase5/color_diversity_by_cluster.png` | Diversity by cluster + zone |
| `visualizations/phase5/color_value_correlation.png` | Value distribution by color |
| `visualizations/phase5/color_cooccurrence.png` | Co-occurrence heatmap |

---

*Corpus sweep run 2026-03-02 against K-CAT SQLite database.*
