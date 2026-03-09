# Phase 2: Summation Patterns

**Generated:** 2026-03-08  
**Database:** K-CAT SQLite database (built from KFG source data)  
**Detector:** `src/analysis/kfg_summation_detector.py`  
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

### Per-Pattern Criteria

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
| With any summation pattern | 515 | 72.6% |
| Without any detected pattern | 194 | 27.4% |

### By Pattern Type

| Pattern Type | Khipus | Rate | Relationships |
|-------------|--------|------|---------------|
| `pendant_pendant_sum` | 410 | 57.8% | 7,018 |
| `colored_pendant_sum` | 276 | 38.9% | 3,526 |
| `indexed_pendant_sum` | 204 | 28.8% | 1,835 |
| `pendant_sub_neighbor` | 178 | 25.1% | 341 |
| `subsidiary_pendant_sum` | 145 | 20.5% | 1,034 |
| `ascher_decreasing_group` | 144 | 20.3% | 280 |
| `group_sum_bands` | 104 | 14.7% | 175 |
| `group_group_sum` | 102 | 14.4% | 262 |
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
| PPS sum cords with dual decompositions | 1,240 (21.5% of unique sum cords) |

A sum cord has a dual decomposition when its value can be matched by multiple distinct summand windows. 21.5% of unique PPS sum cords have at least two valid windows, reflecting the combinatorial nature of contiguous-window enumeration.

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

## Data Quality Notes

1. **Tolerance 0 is strict.** Exact integer arithmetic is required. Khipus with partially decoded cord values may fail a match even though a genuine summation structure exists — this biases toward under-detection.

2. **`value = 0` exclusion.** Cords with `value = 0` (null placeholder) are excluded as candidate summing terms. Khipus with many undecoded cords therefore have fewer candidates.

3. **`colored_pendant_sum` and compound color codes.** The K-CAT database stores compound color codes (e.g., `MB:W`, `KB-DB`) as single strings. The detector extracts the dominant color component before grouping.

4. **194 khipus with no detected pattern.** These include objects with predominantly undecoded values, as well as any khipus structured by conventions not yet modeled.

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

- The detector tests arithmetic identity only. It has no model of intent: a coincidental three-cord sum (e.g., 1 + 2 = 3) passes the same test as any other arithmetic match.
- The corpus sweep uses `tolerance = 0`. A small tolerance (1–2 units) may be appropriate when cord values are subject to transcription uncertainty; such analysis is left for future work.
- Pattern type taxonomy follows Ascher & Ascher (1978, 1981).

---

*See [Citations and Acknowledgments](../README.md#citations-and-acknowledgments) in the project README for primary sources, data attribution, and toolkit provenance.*

---

*Corpus sweep run against K-CAT SQLite database. Re-run with `KFGSummationDetector.summarize()` on the current database to refresh these figures.*
