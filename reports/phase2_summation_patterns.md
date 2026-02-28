# Phase 2: Summation Patterns

**Generated:** 2026-02-28  
**Database:** Khipu Field Guide (KFG) SQLite database  
**Detector:** `src/analysis/kfg_summation_detector.py`  
**Status:** ✅ Complete

---

## Research Question

What fraction of khipus embed arithmetic summation relationships — cords whose numeric values sum to other cords? What pattern types appear, how often are they combined, and how does the KFG corpus compare with the OKR baseline?

This phase tests the central hypothesis in khipu decipherment: that Inka khipus functioned as accounting devices, with pendant cords recording sub-totals that roll up into group or primary-cord totals.

---

## Methodology

### Pattern Types

The KFG Summation Detector implements eight structural relationship types derived from Ascher & Ascher's fieldmark vocabulary:

| Code | Pattern | Description |
|------|---------|-------------|
| `pendant_pendant_sum` | Pendant → Pendant | One pendant's value = sum of adjacent pendants |
| `colored_pendant_sum` | Color group | Pendants of the same color sum to another pendant |
| `indexed_pendant_sum` | Indexed pendant | A designated "total" pendant = sum of its group |
| `subsidiary_pendant_sum` | Subsidiary → Pendant | Subsidiary cord values sum to their parent pendant |
| `indexed_subsidiary_sum` | Indexed subsidiary | A subsidiary acts as a color-group total |
| `pendant_sub_neighbor` | Pendant–subsidiary | A pendant equals the sum of a neighbor's subsidiaries |
| `group_group_sum` | Group → Group | One group of pendants sums to another group |
| `ascher_decreasing_group` | Decreasing pattern | Groups form a decreasing arithmetic sequence |

### Detection Logic

For each khipu, the detector:

1. Loads all cords with their values, hierarchy levels, and colors from the KFG database
2. Enumerates candidate relationships for each pattern type
3. Checks whether the arithmetic identity holds exactly (`tolerance = 0`, i.e., exact integer match)
4. A khipu is scored as `has_summation = True` if **at least one** relationship of any type matches

**Tolerance = 0** means the numeric equality must hold exactly, with no rounding. Cords with `value = 0` (the KFG null placeholder) are excluded from summation candidates.

### OKR Baseline

The OKR-era detector (`scripts/test_value_computation.py`) implemented three of these pattern types — `contiguous_sums` (equivalent to `pendant_pendant_sum`), `group_totals` (equivalent to `group_group_sum`), and `hierarchical` (which had a known implementation bug and reported 0%). The OKR comparison therefore best maps to the two working OKR types.

---

## Cross-Corpus Comparison

| Metric | OKR (reference) | KFG (this run) |
|--------|----------------|----------------|
| Khipus tested | 619 | 709 |
| With any summation pattern | 430 (69.5%) | 643 (90.7%) |
| Without any pattern | 189 (30.5%) | 66 (9.3%) |
| Contiguous / pendant–pendant | 377 (60.9%) | 601 (84.8%) |
| Group totals / group–group | 331 (53.5%) | 360 (50.8%) |
| Both types combined | 278 (44.9%) | — |
| Hierarchical (OKR) / indexed (KFG) | 0% ⚠️ bug | 440 (62.1%) |

*OKR reference figures from KCAT Phase 3 (summation testing) report, January 2026.*

**Interpreting the increase from 69.5% → 90.7%:** Two factors contribute:

1. **Expanded detector vocabulary.** The OKR detector covered 2–3 pattern types; the KFG detector covers 8. Patterns invisible to the OKR detector (colored sums, indexed subsidiaries, subsidiary→pendant) account for additional khipus newly classified as summation-carrying.

2. **Improved data completeness.** The KFG database has substantially more decoded knot records (238,099 vs 110,677), enabling detection of relationships that had incomplete values in the OKR.

---

## KFG Summation Results

### Corpus-Wide Coverage

| Metric | Count | Rate |
|--------|-------|------|
| Khipus tested | 709 | — |
| With any summation pattern | 643 | 90.7% |
| Without any detected pattern | 66 | 9.3% |

### By Pattern Type

| Pattern Type | Khipus With Pattern | Rate |
|-------------|---------------------|------|
| `pendant_pendant_sum` | 601 | 84.8% |
| `colored_pendant_sum` | 563 | 79.4% |
| `indexed_pendant_sum` | 440 | 62.1% |
| `subsidiary_pendant_sum` | 376 | 53.0% |
| `group_group_sum` | 360 | 50.8% |
| `indexed_subsidiary_sum` | 259 | 36.5% |
| `pendant_sub_neighbor` | 225 | 31.7% |
| `ascher_decreasing_group` | 208 | 29.3% |

`pendant_pendant_sum` is the single most common pattern (84.8%), consistent with the fundamental sequential tallying structure. `colored_pendant_sum` at 79.4% suggests color is used systematically as a grouping axis for arithmetic — a finding relevant to Phase 3 (Color Semantics).

### Complexity: Number of Pattern Types Per Khipu

| Pattern types detected | Khipus |
|------------------------|--------|
| 1 type | 41 |
| 2 types | 63 |
| 3 types | 73 |
| 4 types | 122 |
| 5 types | 105 |
| 6 types | 87 |
| 7 types | 105 |
| 8 types (all) | 47 |

The majority of summation-carrying khipus (357 of 643, 55.5%) show 4 or more distinct pattern types, indicating that Inka record-keepers routinely employed multiple arithmetic encoding strategies in a single object.

### Validation Against Ground Truth

The `scripts/test_all_patterns.py` script tests exact recall on a set of manually annotated ground-truth patterns drawn from the KFG. All ground-truth patterns are recalled at 100%, validating the detector's accuracy on known cases:

```
pendant_pendant_sum       GT annotation  → 100% recall
colored_pendant_sum       GT annotation  → 100% recall
indexed_pendant_sum       GT annotation  → 100% recall
subsidiary_pendant_sum    GT annotation  → 100% recall
indexed_subsidiary_sum    GT annotation  → 100% recall
pendant_sub_neighbor      GT annotation  → 100% recall
group_group_sum           GT annotation  → 100% recall
ascher_decreasing_group   GT annotation  → 100% recall
```

---

## Data Quality Notes

1. **Tolerance 0 is strict.** Exact integer arithmetic is required. Khipus with partially decoded cord values may fail the match even though a genuine summation structure exists — this biases toward under-detection, not over-detection.

2. **`value = 0` exclusion.** Cords with `value = 0` (KFG null placeholder) are excluded as candidate summing terms. This is correct behavior but means khipus with high numbers of undecoded cords will have fewer candidates and lower detection probability.

3. **`colored_pendant_sum` caveat.** The 79.4% color-pattern rate should be read carefully: it reflects khipus where cords of the same color arithmetically sum to another cord in the KFG database as currently transcribed. It does not prove that color was the organizational intent — that question is addressed in Phase 3.

4. **66 khipus with no detected pattern.** These may include: (a) narrative/ceremonial khipus that were never accounting devices; (b) khipus with predominantly undecoded values; (c) khipus using alternative arithmetic conventions not yet modeled. This is consistent with the archaeological consensus that not all khipus were numeric records.

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

Ground truth recall:
```bash
python scripts/test_all_patterns.py
```

---

## Limitations

- The detector tests arithmetic identity only. It has no model of intent: a coincidental three-cord sum (e.g., 1 + 2 = 3) passes the same test as a deliberate accounting entry.
- The corpus sweep uses `tolerance = 0`. A small tolerance (1–2 units) would be appropriate when cord values are subject to transcription uncertainty; such analysis is left for future work.
- Pattern type taxonomy follows Ascher & Ascher (1978, 1981). Other researchers (Urton, Hyland) propose alternative non-numeric interpretations in which these "summation patterns" have a different significance.

---

## Citations and Acknowledgments

### Primary Data Source

> Khosla, Ashok. *The Khipu Field Guide*. [khipufieldguide.com](https://khipufieldguide.com), 2020–present.

With contributions from Karen Thompson (University of Melbourne), Manuel Medrano (Harvard University), and KFG affiliates.

### Summation Fieldmark Methodology

The seven core Ascher fieldmarks were defined in:

> Ascher, Marcia and Robert Ascher. *Mathematics of the Incas: Code of the Quipu*. Dover Publications, 1997. (Reprint of the 1981 edition.)

The computational operationalization and extension to an eighth type (`ascher_decreasing_group`) follow:

> Khosla, Ashok and Manuel Medrano. "How Can Data Science Contribute to Understanding the Khipu Code?" *Latin American Antiquity*, 2023.

### Historical Baseline

OKR baseline figures are from the KCAT Phase 3 legacy analysis (January 2026), using the Open Khipu Repository as the primary dataset. The OKR is now superseded by the KFG as the authoritative digital corpus.

---

*Corpus sweep run 2026-02-28 against KFG SQLite database. Re-run with `KFGSummationDetector.summarize()` on the current database to refresh these figures.*
