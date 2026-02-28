# Phase 2: Summation Patterns

**Generated:** 2026-02-28  
**Database:** KCAT SQLite database (built from KFG source data)  
**Detector:** `src/analysis/kfg_summation_detector.py`  
**Status:** ✅ Complete

---

## Research Question

What fraction of khipus embed arithmetic summation relationships — cords whose numeric values sum to other cords? What pattern types appear, how often are they combined, and how does the KCAT result compare with the OKR baseline?

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

1. Loads all cords with their values, hierarchy levels, and colors from the KCAT database
2. Enumerates candidate relationships for each pattern type
3. Checks whether the arithmetic identity holds exactly (`tolerance = 0`, i.e., exact integer match)
4. A khipu is scored as `has_summation = True` if **at least one** relationship of any type matches

**Tolerance = 0** means the numeric equality must hold exactly, with no rounding. Cords with `value = 0` (null placeholder) are excluded from summation candidates.

### OKR Baseline

The OKR-era detector (`scripts/test_value_computation.py`) implemented three of these pattern types — `contiguous_sums` (equivalent to `pendant_pendant_sum`), `group_totals` (equivalent to `group_group_sum`), and `hierarchical` (which had a known implementation bug and reported 0%). The OKR comparison therefore best maps to the two working OKR types.

---

## Cross-Corpus Comparison

| Metric | OKR (reference) | KCAT (this run) |
|--------|----------------|----------------|
| Khipus tested | 619 | 709 |
| With any summation pattern | 430 (69.5%) | 643 (90.7%) |
| Without any pattern | 189 (30.5%) | 66 (9.3%) |
| Contiguous / pendant–pendant | 377 (60.9%) | 601 (84.8%) |
| Group totals / group–group | 331 (53.5%) | 360 (50.8%) |
| Both types combined | 278 (44.9%) | — |
| Hierarchical (OKR) / indexed (KCAT) | 0% ⚠️ bug | 440 (62.1%) |

*OKR reference figures from KCAT Phase 3 (summation testing) report, January 2026.*

The increase from 69.5% to 90.7% is partly explained by the expanded detector vocabulary: the OKR detector covered 2–3 pattern types while the KCAT detector covers 8, so khipus that carry only the less common pattern types were not detectable by the OKR tool.

The KFG Ascher Sum Browser (khipufieldguide.com/fieldmarks) publishes per-khipu summation fieldmark counts for 702 khipus and shows a "has summation" rate of 69.5% (488/702) — see Reconciliation section below.

---

## KCAT Summation Results

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

`pendant_pendant_sum` is the single most common pattern (84.8%), consistent with the fundamental sequential tallying structure. The prevalence of `colored_pendant_sum` (79.4%) is examined in Phase 3 (Color Semantics) in the context of color code normalization.

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

The majority of summation-carrying khipus (357 of 643, 55.5%) show 4 or more distinct pattern types.

---

## Reconciliation Against KFG Fieldmarks

The KFG Ascher Sum Browser (khipufieldguide.com/fieldmarks) publishes per-khipu counts for 7 Ascher fieldmarks (4 pendant sums + 3 group relationships). A per-khipu comparison was run on 2026-02-28 using `scripts/reconcile_kfg_fieldmarks.py`, matching the 702 KFG-listed khipus against KCAT detector output.

### Corpus-Level Comparison (702-khipu overlap)

| Metric | KFG fieldmarks page | KCAT detector |
|--------|--------------------|--------------|
| Khipus evaluated | 702 | 702 |
| With any summation pattern | 488 (69.5%) | 636 (90.6%) |
| Without any pattern | 214 (30.5%) | 66 (9.4%) |

### Per-Khipu Agreement

| Verdict | Count |
|---------|-------|
| Both positive (KFG ✓, KCAT ✓) | 488 |
| Both negative (KFG ✗, KCAT ✗) | 66 |
| KCAT positive, KFG not flagged | 148 |
| KFG positive, KCAT negative | 0 |
| Agreement rate | 78.9% (554/702) |

KFG does not flag any khipu that KCAT misses. The 148 cases where KCAT is positive but KFG is not flagged are the primary source of divergence.

A structural check on these 148 khipus reveals a size signal: their median cord count is 16, compared to 68 for the 488 khipus both sources agree on. Small khipus at `tolerance = 0` are susceptible to spurious arithmetic coincidences — for example, any three cords with values 1, 2, 3 will pass `pendant_pendant_sum` exactly. KFG fieldmarks represent expert judgment on whether a relationship is structurally meaningful, not merely arithmetically satisfied. The detector does not yet distinguish genuine accounting structure from low-number coincidences.

Of the 148 discordant khipus: 8 are positive only for `ascher_decreasing_group` (a pattern not present in the KFG 7-fieldmark set); the remaining 140 show at least one of the 7 shared pattern types at exact match, but are not flagged by KFG.

### Per-Pattern Agreement (7 Shared Types)

| Pattern | KFG flagged | KCAT flagged | Agreement |
|---------|------------|-------------|----------|
| `pendant_pendant_sum` | 406 | 595 | 73.1% |
| `colored_pendant_sum` | 202 | 557 | 49.1% |
| `indexed_pendant_sum` | 274 | 434 | 68.1% |
| `subsidiary_pendant_sum` | 145 | 372 | 67.7% |
| `group_group_sum` | 103 | 354 | 56.6% |
| `indexed_subsidiary_sum` | 101 | 256 | 64.2% |
| `pendant_sub_neighbor` | 142 | 222 | 65.5% |

`colored_pendant_sum` shows the largest divergence (KFG: 202, KCAT: 557). This is consistent with DQ Note 3: compound color codes in the KCAT database may cause two cords with only a shared color prefix to be grouped as same-color when they are not. Color code normalization is examined in Phase 3.

**The reconciliation indicates that the KCAT 90.7% figure includes detections that KFG expert review does not confirm.** The degree of over-detection attributable to small-khipu coincidences and compound color codes is not yet quantified. Per-pattern figures from this report should be read as detector output, not as validated summation prevalence.

The source data for this reconciliation is saved at `data/processed/kfg_fieldmarks_reconciliation.csv`.

---

## Data Quality Notes

1. **Tolerance 0 is strict.** Exact integer arithmetic is required. Khipus with partially decoded cord values may fail a match even though a genuine summation structure exists — this biases toward under-detection.

2. **`value = 0` exclusion.** Cords with `value = 0` (null placeholder) are excluded as candidate summing terms. Khipus with many undecoded cords therefore have fewer candidates.

3. **`colored_pendant_sum` and compound color codes.** The KCAT database stores compound color codes (e.g., `MB:W`, `KB-DB`) as single strings. Two cords sharing only a color prefix may be counted as same-color when they are not. Color codes should be normalized before drawing conclusions about color-based grouping — see Phase 3.

4. **66 khipus with no detected pattern.** These include objects with predominantly undecoded values, as well as any khipus that may be narrative, ceremonial, or structured by conventions not yet modeled.

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

- The detector tests arithmetic identity only. It has no model of intent: a coincidental three-cord sum (e.g., 1 + 2 = 3) passes the same test as a deliberate accounting entry. The reconciliation against KFG fieldmarks shows this matters in practice: 148 of 702 khipus (21%) are flagged by KCAT but not by KFG, and these skew toward smaller khipus (median 16 cords) where chance coincidences at tolerance = 0 are more probable.
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

*Corpus sweep run 2026-02-28 against KCAT SQLite database. Re-run with `KFGSummationDetector.summarize()` on the current database to refresh these figures.*
