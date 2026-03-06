# Phase 2: Summation Patterns

**Generated:** 2026-03-02  
**Database:** K-CAT SQLite database (built from KFG source data)  
**Detector:** `src/analysis/kfg_summation_detector.py` — criteria calibrated against KFG documentation  
**Reconciliation:** K-CAT detector output compared against KFG fieldmark annotation files (702 khipus in the K-CAT/KFG intersection, all 9 patterns)  
**Status:** ✅ Complete

---

## Research Question

What fraction of khipus embed arithmetic summation relationships — cords whose numeric values sum to other cords? What pattern types appear, how often are they combined, and how does the K-CAT result compare with the OKR baseline?

This phase tests the central hypothesis in khipu decipherment literature (Ascher & Ascher 1978, 1981; Urton 2003): that Inka khipus functioned as accounting devices, with pendant cords recording sub-totals that roll up into group or primary-cord totals. The toolkit applies this received hypothesis computationally against the KFG corpus to measure how frequently the arithmetic relationship holds and where the model breaks down.

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
2. Enumerates candidate relationships for each pattern type using the criteria below
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
- **`group_sum_bands`**: khipu split at midpoint; left-half group totals equal right-half group totals (explicit band detector, not aliased to GG).
- **`ascher_decreasing_group`**: groups form a monotonically decreasing sequence of totals.

### Enhanced Detection Features (Phase 2 Extension)

**Handedness Tracking**

For each summation relationship, the detector now records **handedness** — whether the sum cord appears to the left or right of its summand window in the pendant sequence:
- **Left-handed**: Sum cord's `cord_index` < min(`summand_cord_index`) — summands are to the right
- **Right-handed**: Sum cord's `cord_index` > max(`summand_cord_index`) — summands are to the left
- **Undefined**: For patterns where position is not linear (e.g., `colored_pendant_sum`, `group_group_sum`)

Handedness analysis enables investigation of reading-direction conventions and potential semantic encoding through spatial arrangement (Urton 2003, 2017).

**Dual Sum Detection**

The detector identifies **dual sums** — cords whose value matches multiple distinct summand windows. This is computed by grouping relationships by `sum_cord_id` and checking for multiple unique `summand_window_hashes`.

Dual sums indicate:
- Structural redundancy (error-checking through multiple arithmetic paths)
- Multi-level accounting (same total appearing in different contexts)
- Coincidental overlap in highly regular numeric sequences

**Figure-8 Knot Proximity Analysis**

Figure-8 knots (`E`, `EE` in `knot_clusters.knot_type`) do not encode numeric value and may function as structural markers. For each summation relationship, the detector checks whether a figure-8 knot appears:
1. On the sum cord itself
2. On the first 2 or last 2 summands in the window (adjacent to the relationship)

Proximity threshold: **5 cm** along the cord from the summation point. This analysis tests Ascher & Ascher's (1978:75) hypothesis that figure-8 knots mark totals.

### OKR Baseline

The OKR-era detector (`scripts/test_value_computation.py`) implemented three of these pattern types — `contiguous_sums` (equivalent to `pendant_pendant_sum`), `group_totals` (equivalent to `group_group_sum`), and `hierarchical` (which had a known implementation bug and reported 0%). The OKR comparison therefore best maps to the two working OKR types.

---

## Cross-Corpus Comparison

| Metric | OKR (reference) | K-CAT (current) |
|--------|----------------|----------------|
| Khipus tested | 619 | 702 (KFG intersection) |
| With any summation pattern | 430 (69.5%) | 557 (79.2%) |
| Without any pattern | 189 (30.5%) | 146 (20.8%) |
| Agreement with KFG ground truth | — | **90.4%** |

*OKR reference figures from K-CAT legacy analysis (January 2026), using the Open Khipu Repository. The OKR is now superseded by the KFG as the authoritative digital corpus.*

---

## K-CAT Summation Results

### Corpus-Wide Coverage

| Metric | Count | Rate |
|--------|-------|------|
| Khipus tested | 709 | — |
| With any summation pattern | 569 | 80.3% |
| Without any detected pattern | 140 | 19.7% |

### By Pattern Type

| Pattern Type | Khipus With Pattern | Rate |
|-------------|---------------------|------|
| `pendant_pendant_sum` | 474 | 66.9% |
| `indexed_pendant_sum` | 293 | 41.3% |
| `colored_pendant_sum` | 273 | 38.5% |
| `pendant_sub_neighbor` | 225 | 31.7% |
| `ascher_decreasing_group` | 208 | 29.3% |
| `subsidiary_pendant_sum` | 199 | 28.1% |
| `group_group_sum` | 123 | 17.3% |
| `indexed_subsidiary_sum` | 86 | 12.1% |
| `group_sum_bands` | 86 | 12.1% |

`pendant_pendant_sum` is the single most common pattern (66.9%), consistent with the fundamental sequential tallying structure. Color-based grouping (`colored_pendant_sum`, 38.5%) is the third most prevalent pattern; the detector normalizes compound color codes (e.g. `MB:W`) to their dominant color component before grouping.

### Handedness Analysis

Summation relationships can be **directional**: summands may appear to the left or right of the sum cord in the sequence. Handedness analysis tests whether khipus exhibit systematic reading-direction preferences — potentially reflecting regional scribal conventions or semantic structure (Urton 2003, 2017).

**Pendant-pendant sum handedness** (459 khipus with PPS patterns):

| Direction | Count | Rate |
|-----------|-------|------|
| Left-handed (sum cord right of summands) | 38,094 | 38.6% |
| Right-handed (sum cord left of summands) | 60,691 | 61.4% |
| **Total relationships** | **98,785** | — |

The corpus-wide handedness ratio is **+0.23** (right-biased), indicating a systematic preference for sum cords to appear left of their summands across the corpus.

**Interpretation:** The right-handed bias (+0.23) suggests a dominant reading-direction convention in which totals precede their components — consistent with a "top-down" accounting structure where a category total is recorded first, followed by its breakdowns. This aligns with Urton's (2003, 2017) hypothesis that directionality conveys administrative metadata.

### Dual Sum Detection

A **dual sum** occurs when a single cord value matches multiple distinct summand windows. For example, `p12 = 36` might equal both `p1+p2+p3 = 36` AND `p7+p8 = 36`. Dual sums indicate **structural redundancy** — arithmetic encoded multiple ways within the same khipu.

**Dual sum prevalence:**

| Pattern Type | Sum Cords With Multiple Windows | Dual Sum Rate |
|--------------|--------------------------------|---------------|
| `pendant_pendant_sum` | 6,724 | 75.7% of PPS sum cords |

Only `pendant_pendant_sum` produced dual sums in this run. The high rate reflects the combinatorial nature of contiguous-window summation: when many adjacent pendant values exist, a given total can often be decomposed by multiple distinct subsequences.

**Top khipus with extensive dual sums:**
- **KH0082**: 11,423 cords with dual summation paths
- **KH0240**: 6,496 cords with dual summation paths
- **KH0428**: 5,213 cords with dual summation paths
- **KH0349**: 4,339 cords with dual summation paths
- **KH0068**: 4,310 cords with dual summation paths

**Interpretation:** Dual sums may represent:
1. **Multi-level accounting** — the same total appears in different summation contexts (e.g., by color group AND by position index)
2. **Combinatorial inevitability** — khipus with many small pendant values naturally produce multiple windows summing to the same target; the large numbers above reflect this combinatorial effect rather than deliberate redundancy
3. **Error-checking redundancy** — in some cases, multiple arithmetic paths to the same value may be an intentional design feature

### Figure-8 Knot Proximity Analysis

**Figure-8 knots** (encoded as `E` or `EE` in the `knot_type` field) are anomalous knots that do not encode numeric value. Ascher & Ascher (1978:75) noted that figure-8 knots often appear adjacent to sum cords, potentially serving as **semantic markers** indicating "this is a total."

The detector now checks whether each summation relationship has a figure-8 knot on:
1. The sum cord itself
2. The immediately adjacent summand cords (first 2, last 2 in window)

**Figure-8 proximity results:**

| Pattern Type | Relationships With Figure-8s | Proximity Rate |
|--------------|------------------------------|----------------|
| `pendant_pendant_sum` | 491 | 0.5% |

Only `pendant_pendant_sum` produced figure-8 proximity matches at the default 5cm threshold.

**Figure-8 location distribution** (pendant_pendant_sum only):
- On adjacent summand: 412 (83.9%)
- On sum cord: 79 (16.1%)

**Interpretation:** Figure-8 knots appear near **0.5% of pendant-pendant sum relationships** within the 5cm threshold, a much lower rate than previously estimated. The predominance of figure-8s on adjacent summands (83.9%) rather than the sum cord itself inverts earlier expectations and suggests these knots may function as **boundary markers** delineating summation groups rather than labels on totals. Possible explanations for the low rate:
1. Figure-8s mark only **structurally significant** totals (e.g., grand totals, inter-group sums)
2. Figure-8 usage varies by regional convention or time period
3. Many figure-8 knots may serve purposes unrelated to summation

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

## Reconciliation Against KFG Ground Truth

The KFG team provides per-khipu fieldmark annotation files for all 9 pattern types. These files record the output of the KFG's own detector (the same detector used on `khipufieldguide.com`), not a separate human annotation pass.

### Reconciliation Methodology

**Coverage gaps — not a full apples-to-apples comparison.** The K-CAT corpus contains **709 khipus**; the KFG annotation files cover **702 unique khipus** (703 CSV rows, one khipu appearing in two rows in the PP summary). Seven K-CAT khipus have no corresponding GT entry and are **silently excluded** from every reconciliation table. All agreement percentages are computed over the 702-khipu intersection only.

**What "KFG negative" means.** When a khipu's count is 0 in a KFG annotation file, that means the KFG detector found no instance of that pattern for that khipu. It does not mean a human expert confirmed the pattern is absent. Both the KFG detector and the K-CAT detector may miss genuine patterns; the agreement metric measures cross-detector consistency, not human-validated accuracy.

**Significance thresholds.** Two patterns (`indexed_subsidiary_sum` and `pendant_sub_neighbor`) apply a threshold of `count > 1` rather than `count ≥ 1`, per KFG documentation: a single isolated occurrence is considered coincidental. The KFG annotation files record raw counts; the threshold is applied here during comparison.

> **Note on the fieldmarks browser.** `khipufieldguide.com/fieldmarks` shows 7 columns in HTML order PP, IP, CP, SP, GSB, GG, ADG — differing from the analysis-page narrative — and omits `indexed_subsidiary_sum` and `pendant_sub_neighbor` entirely. The authoritative KFG annotations cover all 9 patterns.

### Corpus-Level Comparison (702-khipu KFG intersection)

> **Scope note.** All figures in this section and the tables below — counts, rates, FP/FN tallies, and agreement percentages — refer to the **702-khipu intersection** between the KFG annotation files and the K-CAT corpus. The full KFG corpus contains **709 khipus**; the remaining 7 have no corresponding GT entry and are excluded from every comparison figure.

| Metric | KFG ground truth | K-CAT detector |
|--------|-----------------|----------------|
| Khipus evaluated | 702 | 702 |
| With any summation pattern | 491 (69.9%) | 557 (79.3%) |
| Without any pattern | 211 (30.1%) | 145 (20.7%) |

### Per-Khipu Overall Agreement

| Verdict | Count |
|---------|-------|
| Both positive (KFG ✓, K-CAT ✓) | 491 |
| Both negative (KFG ✗, K-CAT ✗) | 146 |
| K-CAT positive, KFG negative (FP) | 66 |
| KFG positive, K-CAT negative (FN) | 2 |
| **Agreement rate** | **90.4%** |

Only **2 FNs** — virtually perfect recall. **66 FPs** remain, distributed across PP (64), IP (89), PSN (76), and ADG (60); these patterns have near-zero FNs, suggesting their criteria are appropriately inclusive.

### Per-Pattern Agreement

| Pattern | Sig | KFG+ | K-CAT+ | FP | FN | Agreement |
|---------|-----|------|-------|----|----|-----------|
| `pendant_pendant_sum` | >=1 | 409 | 473 | 64 | 0 | **90.9%** |
| `indexed_pendant_sum` | >=1 | 205 | 294 | 89 | 0 | **87.4%** |
| `colored_pendant_sum` | >=1 | 277 | 274 | 34 | 37 | **89.9%** |
| `subsidiary_pendant_sum` | >=1 | 148 | 202 | 54 | 0 | **92.3%** |
| `group_group_sum` | >=1 | 101 | 125 | 41 | 17 | **91.8%** |
| `group_sum_bands` | >=1 | 106 | 88 | 0 | 18 | **97.4%** |
| `indexed_subsidiary_sum` | >1 | 30 | 65 | 39 | 4 | **93.9%** |
| `pendant_sub_neighbor` | >1 | 74 | 150 | 76 | 0 | **89.2%** |
| `ascher_decreasing_group` | >=1 | 142 | 202 | 60 | 0 | **91.5%** |

**Key observations:**

- **PP, IP, SP, PSN, ADG: 0 false negatives** — perfect recall across all detected instances.
- **PSN: treat with caution.** The KFG author's own online assessment of `pendant_sub_neighbor` states: *"The pendant_subsidiary_neighbor relationship seems likely to be a fluke. Occurring 0.64% of the time… I'm inclined to write off this relationship as a statistical fluke."* The 76 PSN false positives in our results are therefore consistent with the KFG's own view that many PSN detections may be coincidental. The pattern is retained in the detector for completeness but its interpretation as deliberate accounting is uncertain.
- **GSB: 97.4% with zero FPs** — the explicit left-sum = right-sum split detector is the most precise in the suite; 18 FNs remain (likely edge-band boundary cases).
- **GG: 91.8%** — properly separated from GSB; 17 FNs likely from group total boundary conditions.
- **IS: 93.9%** — calibrated by removing position-only grouping, applying a value threshold (≥ 5, excluding round numbers), and deduplicating by `(sum_cord_id, frozenset(summand_ids))`.
- **SP: 92.3%** — calibrated by raising the significance threshold to ≥ 11 and excluding multiples of 10 below 100, removing coincidental small-value matches.
- **CP: 37 FNs** — the only pattern with meaningful false negatives. The detector normalizes compound color codes via dominant-color extraction, but some residual FNs likely reflect color-variant cords (e.g. two-ply spliced colors) where the KFG annotation counts a match that K-CAT misses.

---

## Data Quality Notes

1. **Tolerance 0 is strict.** Exact integer arithmetic is required. Khipus with partially decoded cord values may fail a match even though a genuine summation structure exists — this biases toward under-detection.

2. **`value = 0` exclusion.** Cords with `value = 0` (null placeholder) are excluded as candidate summing terms. Khipus with many undecoded cords therefore have fewer candidates.

3. **`colored_pendant_sum` and compound color codes.** The K-CAT database stores compound color codes (e.g., `MB:W`, `KB-DB`) as single strings. The detector extracts the dominant color component before grouping, which handles most cases. The residual 37 CP false negatives likely reflect two-ply or spliced-color cords where the KFG considers a looser color match than the dominant-color heuristic.

4. **140 khipus with no detected pattern.** These include objects with predominantly undecoded values, as well as any khipus that may be narrative, ceremonial, or structured by conventions not yet modeled.

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

- The detector tests arithmetic identity only. It has no model of intent: a coincidental three-cord sum (e.g., 1 + 2 = 3) passes the same test as a deliberate accounting entry. The reconciliation shows 66 of 702 khipus (9.4%) are flagged by K-CAT but not by KFG. The largest concentrations are in `pendant_sub_neighbor` (76 FPs), `pendant_pendant_sum` (64 FPs), `indexed_pendant_sum` (89 FPs), and `ascher_decreasing_group` (60 FPs) — all patterns with zero FNs, suggesting the thresholds are appropriately inclusive at the cost of some over-detection.
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

*Corpus sweep run 2026-03-02 against K-CAT SQLite database. Re-run with `KFGSummationDetector.summarize()` on the current database to refresh these figures.*
