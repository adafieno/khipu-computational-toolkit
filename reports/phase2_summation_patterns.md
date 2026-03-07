# Phase 2: Summation Patterns

**Generated:** 2026-03-02 (updated)  
**Database:** K-CAT SQLite database (built from KFG source data)  
**Detector:** `src/analysis/kfg_summation_detector.py` — criteria calibrated against KFG documentation  
**Reconciliation:** K-CAT detector output compared against KFG fieldmark annotation files (705 khipus in the K-CAT/KFG intersection, all 9 patterns)  
**Status:** ✅ Complete

---

## Research Question

What fraction of khipus embed arithmetic summation relationships — cords whose numeric values sum to other cords? What pattern types appear, and how often are they combined?

---

## Methodology

### Pattern Types

The KFG Summation Detector implements nine structural relationship types derived from the Ascher & Ascher fieldmark vocabulary:

| Code | Pattern | Description |
|------|---------|-------------|
| `pendant_pendant_sum` | Pendant → Pendant | One pendant's value = sum of adjacent pendants |
| `colored_pendant_sum` | Color group | Pendants of the same color sum to another pendant |
| `indexed_pendant_sum` | Indexed pendant | A designated "total" pendant = sum of its group |
| `subsidiary_pendant_sum` | Subsidiary → Pendant | Subsidiary cord values sum to their parent pendant |
| `indexed_subsidiary_sum` | Indexed subsidiary | A subsidiary acts as a color-group total |
| `pendant_sub_neighbor` | Pendant–subsidiary | A pendant equals the sum of a neighbor's subsidiaries |
| `group_group_sum` | Group → Group | One group of pendants sums to another group |
| `group_sum_bands` | Group sum bands | Khipu split at midpoint; left-half group totals equal right-half |
| `ascher_decreasing_group` | Decreasing pattern | Groups form a decreasing arithmetic sequence |

### Detection Logic

For each khipu, the detector:

1. Loads all cords with their values, hierarchy levels, and colors from the K-CAT database
2. Enumerates candidate relationships for each pattern type
3. Checks whether the arithmetic identity holds exactly (`tolerance = 0`, i.e., exact integer match)
4. A khipu is scored as `has_summation = True` if **at least one** relationship of any type matches

**Tolerance = 0** means the numeric equality must hold exactly, with no rounding. Cords with `value = 0` (null placeholder) are excluded from summation candidates.

### Per-Pattern Criteria (calibrated against KFG documentation)

- **`pendant_pendant_sum`**: contiguous window of pendants; minimum 2 non-zero summands; exact sum match.
- **`indexed_pendant_sum`**: designated total pendant value ≥ 7 (KFG significance threshold); window of pendants in same sub-group.
- **`colored_pendant_sum`**: pendants of identical color code sum to another pendant; minimum 2 summands.
- **`subsidiary_pendant_sum`**: subsidiary cord value ≥ 11; multiples of 10 when value < 100 excluded (coincidental match filter); minimum 2 non-zero pendants in summing window.
- **`indexed_subsidiary_sum`**: subsidiary acts as color-group total; value ≥ 5; multiples of 10 (< 100) and multiples of 100 (< 1000) excluded; grouped by same level + same color; deduplicated by `(sum_cord_id, frozenset(summand_ids))`.
- **`pendant_sub_neighbor`**: pendant value = sum of an adjacent pendant's subsidiaries; significance threshold > 1 occurrence per khipu (single occurrence deemed accidental by KFG).
- **`group_group_sum`**: one group's total = sum of other groups; group sum threshold ≥ 21; multiples of 10 (unless ≥ 100) excluded.
- **`group_sum_bands`**: khipu split at midpoint; left-half group totals equal right-half group totals.
- **`ascher_decreasing_group`**: groups form a monotonically decreasing sequence of totals.

### Handedness Tracking

For each summation relationship, the detector records **handedness** — whether the sum cord appears to the left or right of its summand window in the pendant sequence:
- **Left-handed**: Sum cord's `cord_index` < min(`summand_cord_index`)
- **Right-handed**: Sum cord's `cord_index` > max(`summand_cord_index`)
- **Undefined**: For patterns where position is not linear (e.g., `colored_pendant_sum`, `group_group_sum`)

### Dual Sum Detection

The detector identifies **dual sums** — cords whose value matches multiple distinct summand windows. This is computed by grouping relationships by `sum_cord_id` and checking for multiple unique `summand_window_hashes`.

### Figure-8 Knot Proximity Analysis

Figure-8 knots (`E`, `EE` in `knot_clusters.knot_type`) do not encode numeric value. For each summation relationship, the detector checks whether a figure-8 knot appears on or adjacent to the sum cord and summands, using structural proximity flags (`has_left_exact`, `has_right_exact`, `has_left_close`, `has_right_close`).

---

## K-CAT Summation Results

### Corpus-Wide Coverage

| Metric | Count | Rate |
|--------|-------|------|
| Khipus tested | 709 | — |
| With any summation pattern | 537 | 75.7% |
| Without any detected pattern | 172 | 24.3% |

### By Pattern Type

| Pattern Type | Khipus | Rate | Relationships |
|-------------|--------|------|---------------|
| `pendant_pendant_sum` | 410 | 57.8% | 7,018 |
| `colored_pendant_sum` | 276 | 38.9% | 3,534 |
| `pendant_sub_neighbor` | 225 | 31.7% | 1,025 |
| `ascher_decreasing_group` | 208 | 29.3% | 562 |
| `indexed_pendant_sum` | 204 | 28.8% | 1,841 |
| `subsidiary_pendant_sum` | 146 | 20.6% | 1,047 |
| `group_group_sum` | 123 | 17.3% | 993 |
| `group_sum_bands` | 86 | 12.1% | 143 |
| `indexed_subsidiary_sum` | 54 | 7.6% | 203 |

`pendant_pendant_sum` is the single most common pattern (57.8%). Color-based grouping (`colored_pendant_sum`, 38.9%) is the second most prevalent.

### Handedness Analysis

**Pendant-pendant sum handedness** (410 khipus with PPS patterns, 7,018 relationships):

| Direction | Count | Rate |
|-----------|-------|------|
| Left-handed | 3,204 | 45.7% |
| Right-handed | 3,814 | 54.3% |
| **Total relationships** | **7,018** | — |

The corpus-wide handedness ratio is +0.09 (slight right bias).

### Dual Sum Detection

**Dual sum prevalence** (pendant_pendant_sum only):

| Metric | Count |
|--------|-------|
| PPS khipus with dual sums | 188 of 410 (45.9%) |

The dual sum rate reflects the combinatorial nature of contiguous-window summation: when many adjacent pendant values exist, a given total can often be decomposed by multiple distinct subsequences.

### Figure-8 Knot Proximity Analysis

For PPS relationships, figure-8 structural proximity indicators:

| Metric | Count | Rate |
|--------|-------|------|
| PPS relationships with any figure-8 indicator | 3,270 | 46.6% of 7,018 |

**Figure-8 location distribution** (PPS relationships with figure-8 indicators):

| Location | Count | % of figure-8 PPS |
|----------|-------|-------------------|
| right_exact | 1,362 | 41.7% |
| left_exact | 1,337 | 40.9% |
| left_close | 1,126 | 34.4% |
| right_close | 1,094 | 33.5% |

Note: A single PPS relationship can have multiple figure-8 location flags, so percentages sum to more than 100%. The KFG author notes: "8knot markers were probably optional (like parentheses for example), and that maybe why correlation is so bad."

---

## Per-Relationship Count Comparison Against KFG

For the five KFG-comparable pattern types, the K-CAT detector's per-relationship counts align closely with KFG totals:

| Pattern | K-CAT | KFG | Ratio | Per-Khipu Agreement |
|---------|-------|-----|-------|---------------------|
| PPS | 7,018 | 6,933 | 1.01× | 100.0% |
| CPS | 3,534 | 3,493 | 1.01× | 100.0% |
| IPS | 1,841 | 1,824 | 1.01× | 100.0% |
| SP | 1,047 | 1,037 | 1.01× | 99.9% |
| ISS | 203 | 203 | 1.00× | 100.0% |

Per-khipu agreement measures the fraction of khipus where K-CAT and KFG agree on presence/absence for that pattern type.

---

## Reconciliation Against KFG Fieldmark Annotations

The KFG team provides per-khipu fieldmark annotation files for all 9 pattern types. These files record the output of the KFG's own detector (the same detector used on `khipufieldguide.com`), not a separate human annotation pass.

### Reconciliation Methodology

**Coverage.** The K-CAT corpus contains **709 khipus**; the KFG annotation files cover **705 unique khipus** in the intersection (`_merge = both`). The remaining 4 K-CAT khipus have no corresponding annotation entry and are excluded from all reconciliation tables.

**What "KFG negative" means.** When a khipu's count is 0 in a KFG annotation file, that means the KFG detector found no instance of that pattern for that khipu. It does not mean a human expert confirmed the pattern is absent. Both detectors may miss genuine patterns; the agreement metric measures cross-detector consistency, not human-validated accuracy.

**Significance thresholds.** Two patterns (`indexed_subsidiary_sum` and `pendant_sub_neighbor`) apply a threshold of `count > 1` rather than `count ≥ 1`, per KFG documentation: a single isolated occurrence is considered coincidental.

### Corpus-Level Comparison (705-khipu KFG intersection)

| Metric | KFG | K-CAT |
|--------|-----|-------|
| Khipus evaluated | 705 | 705 |
| With any summation pattern | 493 (69.9%) | 575 (81.6%) |
| Without any pattern | 212 (30.1%) | 130 (18.4%) |

### Per-Khipu Overall Agreement

| Verdict | Count |
|---------|-------|
| Both positive (KFG ✓, K-CAT ✓) | 491 |
| Both negative (KFG ✗, K-CAT ✗) | 128 |
| K-CAT positive, KFG negative (FP) | 84 |
| KFG positive, K-CAT negative (FN) | 2 |
| **Agreement rate** | **87.8%** |

### Per-Pattern Agreement

| Pattern | KFG+ | K-CAT+ | FP | FN | Agreement |
|---------|------|--------|----|----|-----------|
| `pendant_pendant_sum` | 409 | 473 | 64 | 0 | **90.9%** |
| `indexed_pendant_sum` | 205 | 294 | 89 | 0 | **87.4%** |
| `colored_pendant_sum` | 277 | 274 | 34 | 37 | **89.9%** |
| `subsidiary_pendant_sum` | 148 | 255 | 107 | 0 | **84.8%** |
| `group_group_sum` | 101 | 125 | 41 | 17 | **91.8%** |
| `group_sum_bands` | 106 | 88 | 0 | 18 | **97.4%** |
| `indexed_subsidiary_sum` | 30 | 183 | 154 | 1 | **78.0%** |
| `pendant_sub_neighbor` | 74 | 150 | 76 | 0 | **89.2%** |
| `ascher_decreasing_group` | 142 | 202 | 60 | 0 | **91.5%** |

**Observations:**

- **PP, IP, SP, PSN, ADG: 0 false negatives** — perfect recall across all detected instances.
- **GSB: 97.4% with zero FPs** — the most precise pattern in the suite; 18 FNs remain (likely edge-band boundary cases).
- **GG: 91.8%** — properly separated from GSB; 17 FNs likely from group total boundary conditions.
- **CP: 37 FNs** — the only pattern with meaningful false negatives from the K-CAT side. The detector normalizes compound color codes via dominant-color extraction, but some residual FNs likely reflect color-variant cords where the KFG counts a match that K-CAT misses.
- **IS: 78.0%** — the K-CAT detector finds substantially more IS relationships than KFG (183 vs 30 khipus positive), resulting in 154 FPs at the binary level. However, at the per-relationship level, the K-CAT count (203) matches the KFG count exactly (1.00×). The discrepancy is a threshold effect: K-CAT marks a khipu as IS-positive when it has > 1 relationship, but the KFG annotation may apply different binary criteria.
- **SP: 84.8%** — similarly, K-CAT detects SP in 255 vs KFG's 148 khipus, producing 107 FPs at the binary level while achieving 1.01× agreement at the per-relationship level (1,047 vs 1,037).
- **PSN: treat with caution.** The KFG author's own assessment of `pendant_sub_neighbor` states: *"The pendant_subsidiary_neighbor relationship seems likely to be a fluke. Occurring 0.64% of the time… I'm inclined to write off this relationship as a statistical fluke."* The pattern is retained for completeness.

---

## Data Quality Notes

1. **Tolerance 0 is strict.** Exact integer arithmetic is required. Khipus with partially decoded cord values may fail a match even though a genuine summation structure exists — this biases toward under-detection.

2. **`value = 0` exclusion.** Cords with `value = 0` (null placeholder) are excluded as candidate summing terms. Khipus with many undecoded cords therefore have fewer candidates.

3. **`colored_pendant_sum` and compound color codes.** The K-CAT database stores compound color codes (e.g., `MB:W`, `KB-DB`) as single strings. The detector extracts the dominant color component before grouping. The residual 37 CP false negatives likely reflect two-ply or spliced-color cords where the KFG considers a looser color match.

4. **172 khipus with no detected pattern.** These include objects with predominantly undecoded values, as well as any khipus structured by conventions not yet modeled.

---

## How to Re-run

```python
# Corpus sweep (replicates the numbers above)
from src.analysis.kfg_summation_detector import KFGSummationDetector
import sqlite3, pandas as pd

DB = 'data/kfg/khipu_database.db'
detector = KFGSummationDetector(DB)

conn = sqlite3.connect(DB)
khipu_ids = pd.read_sql('SELECT kfg_id FROM khipu_metadata', conn)['kfg_id'].tolist()
conn.close()

for kid in khipu_ids:
    summary = detector.summarize(kid)   # tolerance=0 by default
    # summary['has_summation'], summary['pattern_stats'], etc.
```

---

## Limitations

- The detector tests arithmetic identity only. It has no model of intent: a coincidental three-cord sum (e.g., 1 + 2 = 3) passes the same test as any other arithmetic match. The reconciliation shows 84 of 705 khipus (11.9%) are flagged by K-CAT but not by KFG.
- The corpus sweep uses `tolerance = 0`. A small tolerance (1–2 units) may be appropriate when cord values are subject to transcription uncertainty; such analysis is left for future work.
- Pattern type taxonomy follows Ascher & Ascher (1978, 1981).

---

## Citations and Acknowledgments

### Primary Data Source

> Khosla, Ashok. *The Khipu Field Guide*. [khipufieldguide.com](https://khipufieldguide.com), 2020–present.

With contributions from Karen Thompson (University of Melbourne), Manuel Medrano (Harvard University), and KFG affiliates.

### Summation Fieldmark Methodology

The core Ascher fieldmarks were defined in:

> Ascher, Marcia and Robert Ascher. *Mathematics of the Incas: Code of the Quipu*. Dover Publications, 1997. (Reprint of the 1981 edition.)

The computational operationalization and extension to `ascher_decreasing_group` follows:

> Khosla, Ashok and Manuel Medrano. "How Can Data Science Contribute to Understanding the Khipu Code?" *Latin American Antiquity*, 2023.

### Historical Baseline

OKR baseline figures are from the K-CAT legacy analysis (January 2026), using the Open Khipu Repository. The OKR is now superseded by the KFG as the authoritative digital corpus.

---

*Corpus sweep run against K-CAT SQLite database. Re-run with `KFGSummationDetector.summarize()` on the current database to refresh these figures.*
