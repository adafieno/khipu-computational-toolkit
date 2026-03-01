# Phase 2: Summation Patterns

**Generated:** 2026-02-28 (revised 2026-03-02, detector v2, reconciliation v3)  
**Database:** K-CAT SQLite database (built from KFG source data)  
**Detector:** `src/analysis/kfg_summation_detector.py` (v2 — criteria verified against KFG documentation)  
**Reconciliation:** v3 — ground truth loaded from `data/kfg/KFG/KFG/checks/*.csv`, not scraped from fieldmarks page  
**Status:** ✅ Complete

---

## Research Question

What fraction of khipus embed arithmetic summation relationships — cords whose numeric values sum to other cords? What pattern types appear, how often are they combined, and how does the K-CAT result compare with the OKR baseline?

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

1. Loads all cords with their values, hierarchy levels, and colors from the K-CAT database
2. Enumerates candidate relationships for each pattern type
3. Checks whether the arithmetic identity holds exactly (`tolerance = 0`, i.e., exact integer match)
4. A khipu is scored as `has_summation = True` if **at least one** relationship of any type matches

**Tolerance = 0** means the numeric equality must hold exactly, with no rounding. Cords with `value = 0` (null placeholder) are excluded from summation candidates.

### OKR Baseline

The OKR-era detector (`scripts/test_value_computation.py`) implemented three of these pattern types — `contiguous_sums` (equivalent to `pendant_pendant_sum`), `group_totals` (equivalent to `group_group_sum`), and `hierarchical` (which had a known implementation bug and reported 0%). The OKR comparison therefore best maps to the two working OKR types.

---

## Cross-Corpus Comparison

| Metric | OKR (reference) | K-CAT v1 (2026-02-28) | K-CAT v2 (2026-03-01) | K-CAT v3 (2026-03-02) |
|--------|----------------|---------------------|---------------------|---------------------|
| Khipus tested | 619 | 702 (KFG overlap) | 702 (KFG overlap) | 703 (KFG checks) |
| With any summation pattern | 430 (69.5%) | 636 (90.6%) | 551 (78.5%) | 573 (81.5%) |
| Without any pattern | 189 (30.5%) | 66 (9.4%) | 151 (21.5%) | 130 (18.5%) |
| Agreement with KFG ground truth | — | ~78.9%† | ~86.5%† | **87.8%** |

†v1 and v2 compared against HTML-scraped fieldmarks page; column order was incorrect (see reconciliation section).

*OKR reference figures from K-CAT Phase 3 (summation testing) report, January 2026.*

**Reconciliation v3 note:** The v1 and v2 reconciliations compared against the KFG Ascher Sum Browser HTML table at `khipufieldguide.com/fieldmarks`. Investigation revealed that the HTML table columns are in the order **PP, IP, CP, SP, GSB, GG, ADG** — differing from the analysis-page narrative order (PP, CP, IP, SP, GG, IS, PSN). This caused columns CP⟷IP and GG⟷IS to be swapped in v1/v2. Additionally, `indexed_subsidiary_sum` and `pendant_sub_neighbor` are not displayed on that page at all; the 7th column is `ascher_decreasing_group`. The v3 reconciliation uses the authoritative `data/kfg/KFG/KFG/checks/*.csv` files instead, yielding unambiguous ground truth for all 9 patterns.

The v2 detector applies KFG-documented criteria learned from a full read of the KFG source documentation (all pattern pages on khipufieldguide.com). The main improvements are:

- **pendant_pendant_sum**: minimum 2 summands (docs confirmed the "3 cords" phrasing refers to physical span including zeros, not 3 non-zero summands)
- **subsidiary_pendant_sum**: subsidiary value ≥ 5, multiples of 10 < 100 excluded  
- **group_group_sum**: sum threshold raised to ≥ 21, not-divisible-by-10 (unless ≥ 100) filter added, part-(b) range sums removed (not in KFG definition), all-subsidiary totals used
- **group_sum_bands**: implemented as a real split-band detector (left half sum = right half sum); previously aliased to group_group_sum  
- **pendant_sub_neighbor**: KFG significance threshold of > 1 occurrence applied (single occurrence is not considered significant by KFG authors)

The KFG Ascher Sum Browser (khipufieldguide.com/fieldmarks) shows 69.5% (488/702). The v2 K-CAT detector at 78.5% is much closer to this ground truth than the v1 at 90.6%.

---

## K-CAT Summation Results

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

## Reconciliation Against KFG Ground Truth (v3)

The authoritative per-khipu ground truth is in `data/kfg/KFG/KFG/checks/*.csv` — one CSV per pattern type, one row per khipu (703 total), with counts computed by the KFG team. `scripts/reconcile_kfg_fieldmarks.py` (v3, 2026-03-02) loads these CSVs directly and compares against the K-CAT v2 detector output.

### Fieldmarks Page Column Order Discovery

The KFG fieldmarks browser (`khipufieldguide.com/fieldmarks`) displays 7 columns in this actual HTML order, which differs from the analysis-page narrative:

| HTML col | Pattern | v1/v2 label |
|----------|---------|------------|
| 1 | `pendant_pendant_sum` (num_sum_cords) | kfg_pp ✓ |
| 2 | `indexed_pendant_sum` (num_sum_cords) | kfg_cp ✗ (was CP) |
| 3 | `colored_pendant_sum` (num_sum_cords) | kfg_ip ✗ (was IP) |
| 4 | `subsidiary_pendant_sum` (num_sum_cords) | kfg_sp ✓ |
| 5 | `group_sum_bands` (num_group_sum_bands) | kfg_gg ✗ (was GG) |
| 6 | `group_group_sum` (num_sum_groups) | kfg_is ✗ (was IS) |
| 7 | `ascher_decreasing_group` (num_decreasing_groups) | kfg_psn ✗ (was PSN) |

`indexed_subsidiary_sum` and `pendant_sub_neighbor` are **not on the fieldmarks page**; their v3 ground truth comes from the checks CSVs.

### Corpus-Level Comparison (703-khipu KFG checks)

| Metric | KFG ground truth | K-CAT v2 detector |
|--------|-----------------|------------------|
| Khipus evaluated | 703 | 703 |
| With any summation pattern | 491 (69.8%) | 573 (81.5%) |
| Without any pattern | 212 (30.2%) | 130 (18.5%) |

### Per-Khipu Overall Agreement

| Verdict | v3 (correct ground truth) |
|---------|--------------------------|
| Both positive (KFG ✓, K-CAT ✓) | 491 |
| Both negative (KFG ✗, K-CAT ✗) | 128 |
| K-CAT positive, KFG negative (FP) | 84 |
| KFG positive, K-CAT negative (FN) | 2 |
| **Agreement rate** | **87.8%** |

With correct ground truth, only **2 FNs** (virtually perfect recall) and **84 FPs** (over-detection in IS and SP patterns).

### Per-Pattern Agreement — v2 Detector vs Checks CSV Ground Truth

| Pattern | Sig | KFG+ | K-CAT+ | FP | FN | Agreement |
|---------|-----|------|-------|----|----|-----------|
| `pendant_pendant_sum` | >=1 | 409 | 473 | 64 | 0 | **90.9%** |
| `indexed_pendant_sum` | >=1 | 205 | 294 | 89 | 0 | **87.4%** |
| `colored_pendant_sum` | >=1 | 277 | 274 | 34 | 37 | **89.9%** |
| `subsidiary_pendant_sum` | >=1 | 148 | 255 | 107 | 0 | **84.8%** |
| `group_group_sum` | >=1 | 101 | 125 | 41 | 17 | **91.8%** |
| `group_sum_bands` | >=1 | 106 | 88 | 0 | 18 | **97.4%** |
| `indexed_subsidiary_sum` | >1 | 30 | 183 | 154 | 1 | 78.0% |
| `pendant_sub_neighbor` | >1 | 74 | 150 | 76 | 0 | **89.2%** |
| `ascher_decreasing_group` | >=1 | 142 | 202 | 60 | 0 | **91.5%** |

**Key observations:**

- **PP, IP, SP, PSN, ADG: 0 false negatives** — perfect recall, all real instances detected.
- **GSB: 97.4% with zero FPs** — the explicit left-sum = right-sum split detector is precise with no over-detection; 18 FNs remain.
- **GG: 91.8%** — properly separated from GSB; 17 FNs likely from group total boundary conditions.
- **IS: 78.0% — largest problem area.** K-CAT detects 183 khipus vs KFG's 30 (>1 significance). The color-index sliding window is too permissive; algorithm revision needed.
- **SP: 84.8%** — 107 FPs; subsidiary pendant sum over-detection; significance or criteria tightening needed.

The full per-khipu comparison is saved at `data/processed/kfg_fieldmarks_reconciliation.csv`.

---

## Data Quality Notes

1. **Tolerance 0 is strict.** Exact integer arithmetic is required. Khipus with partially decoded cord values may fail a match even though a genuine summation structure exists — this biases toward under-detection.

2. **`value = 0` exclusion.** Cords with `value = 0` (null placeholder) are excluded as candidate summing terms. Khipus with many undecoded cords therefore have fewer candidates.

3. **`colored_pendant_sum` and compound color codes.** The K-CAT database stores compound color codes (e.g., `MB:W`, `KB-DB`) as single strings. Two cords sharing only a color prefix may be counted as same-color when they are not. Color codes should be normalized before drawing conclusions about color-based grouping — see Phase 3.

4. **130 khipus with no detected pattern.** These include objects with predominantly undecoded values, as well as any khipus that may be narrative, ceremonial, or structured by conventions not yet modeled.

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

- The detector tests arithmetic identity only. It has no model of intent: a coincidental three-cord sum (e.g., 1 + 2 = 3) passes the same test as a deliberate accounting entry. The reconciliation against KFG ground truth (v3) shows this matters: 84 of 703 khipus (12%) are flagged by K-CAT but not by KFG, concentrated in `indexed_subsidiary_sum` (154 FPs) and `subsidiary_pendant_sum` (107 FPs).
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

OKR baseline figures are from the K-CAT Phase 3 legacy analysis (January 2026), using the Open Khipu Repository as the primary dataset. The OKR is now superseded by the KFG as the authoritative digital corpus.

---

*Corpus sweep run 2026-02-28 against K-CAT SQLite database. Re-run with `KFGSummationDetector.summarize()` on the current database to refresh these figures.*
