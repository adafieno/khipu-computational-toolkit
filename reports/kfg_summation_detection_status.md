# KFG Summation Pattern Detection — Current Status
*March 1, 2026 · kfg-integration branch · commit c29fad4+*

---

## 1. Approach

### 1.1 The Key Insight

Rather than trying to algorithmically *re-detect* patterns that the KFG has already
identified, we load the authoritative `checks/*_relation.csv` files directly.
These files record every summation relationship already confirmed by the KFG author,
at full cord-level resolution (cord name, value, summand cords, handedness, etc.).

For khipus **not** in the KFG corpus, the algorithmic detector serves as a fallback.

### 1.2 Two-Path Architecture

```
detect_all_patterns(kfg_id, loader=loader)
          │
          ├─ in_kfg(kfg_id) ? ──YES──► KFGRelationLoader.build_all_matches()
          │                               reads *_relation.csv directly
          │                               → exact KFG annotation
          │
          └─ NO ──────────────────────► KFGSummationDetector algorithms
                                         → approximate re-detection
```

### 1.3 An Important Discovery: Multiple Patterns, Same Cord

The KFG relation CSVs **intentionally** record the same cord in multiple pattern
tables.  For example, a single cord can appear in all three of:
- `pendant_pendant_sum_relation.csv` — the cord = sum of a contiguous pendant window
- `indexed_pendant_sum_relation.csv` — the cord = sum of same-index cords across groups
- `colored_pendant_sum_relation.csv` — the cord = sum of all cords with the same color

These are independent relationships with different summand sets.  The KFG author
records all of them because they are all simultaneously true.

This means **mutual exclusivity should not be applied to detection** — it is only
relevant if you need to assign a single *classification label* to a cord.  When
exclusivity was incorrectly applied, agreement with the KFG summary dropped from
**99.4% to 98.6%** because valid PP and CP relationships were stripped after
IP claimed the same cords first.

### 1.4 Mutual Exclusivity (Classification Only)

The `apply_exclusivity(results)` function remains available for use cases where you
need one primary pattern label per cord.  Priority follows structural specificity:

| Priority | Pattern | Rationale |
|:--------:|---------|-----------|
| 1 (highest) | `indexed_subsidiary_sum` (IS) | subsidiary + color-index + cross-group |
| 2 | `subsidiary_pendant_sum` (SP) | subsidiary + pendant window |
| 3 | `indexed_pendant_sum` (IP) | position-indexed cross-group |
| 4 | `colored_pendant_sum` (CP) | color-indexed cross-group |
| 5 (lowest) | `pendant_pendant_sum` (PP) | contiguous window, any color |

Group-level patterns (GG, GSB, ADG, PSN) annotate whole groups, not individual cords.

---

## 2. Results — 702 KFG Khipus

### 2.1 Per-Pattern Agreement (apply_excl=False — correct mode)

| Pattern | KFG+ | CAT+ | FP | FN | Agreement |
|---------|-----:|-----:|---:|---:|----------:|
| `pendant_pendant_sum` | 406 | 406 | 0 | 0 | **100.0%** |
| `indexed_pendant_sum` | 202 | 202 | 0 | 0 | **100.0%** |
| `colored_pendant_sum` | 274 | 274 | 0 | 0 | **100.0%** |
| `subsidiary_pendant_sum` | 145 | 145 | 0 | 0 | **100.0%** |
| `group_group_sum` | 101 | 101 | 0 | 0 | **100.0%** |
| `group_sum_bands` | 103 | 103 | 0 | 0 | **100.0%** |
| `indexed_subsidiary_sum` | 30 | 30 | 0 | 0 | **100.0%** |
| `pendant_sub_neighbor` | 71 | 66 | 16 | 21 | 94.7% |
| `ascher_decreasing_group` | 142 | 142 | 0 | 0 | **100.0%** |

**Grand mean agreement: 99.4%** — eight of nine patterns at 100.0%.

> KFG+ = khipus the KFG summary CSV marks as having this pattern.
> CAT+ = khipus our system detects (using the relation CSV as source).
> FP = we say YES, KFG summary says NO.
> FN = KFG summary says YES, we find nothing in the relation CSV.

### 2.2 Corpus Coverage

- 702 khipus covered by the KFG summary CSVs
- 503 khipus with at least one relation CSV row (algorithmic fallback for ~199)
- 464 / 654 KFG khipus (71%) exhibit at least one summation pattern

---

## 3. The `ascher_sums_overview.csv` File (Added 2026-03-01)

This file is a **single-table aggregation** of all 7 core Ascher fieldmarks for all
703 khipus.  Assessment:

| Property | Value |
|----------|-------|
| Rows | 703 (full KFG corpus) |
| Columns | `kfg_name` + 7 pattern counts + `num_ascher_sums` total |
| Patterns covered | PP, IP, CP, SP, GSB, GG, ADG |
| NOT included | IS (`indexed_subsidiary_sum`) and PSN (`pendant_sub_neighbor`) |

**Relationship to individual summary CSVs:** The per-pattern columns match exactly
the `num_sum_cords` / `num_sum_groups` / `num_group_sum_bands` / `num_decreasing_groups`
columns in the individual summary CSVs — verified across all 703 khipus with zero
discrepancies.  The `num_ascher_sums` total equals the arithmetic sum of the 7
per-pattern columns for every khipu.

**Assessment:** This file is a convenience aggregation.  It adds no new data beyond
the individual summary CSVs, but is useful as:
- A single source for checking overall pattern coverage per khipu
- The only file providing a cross-pattern total (`num_ascher_sums`)
- A compact alternative when only binary presence is needed, not full statistics

---

## 4. Open Question for the KFG Team

The only remaining disagreement is `pendant_sub_neighbor` (94.7%, 37 khipus).
This is a **counting-unit mismatch** between the two KFG files for this pattern.

### Q — What is a "group" in `num_pendant_sub_neighbor_groups`?

The summary CSV column is `num_pendant_sub_neighbor_groups`, but the relation CSV
stores individual **cord triplets** (pendant P, its subsidiary P.s, neighbor P±1).
These are different counting units and diverge substantially:

| Khipu | summary (groups) | relation (triplets) |
|-------|----------------:|----------------:|
| KH0252 | 3 | 19 |
| KH0028 | 5 | 10 |
| KH0145 | 1 | 5 |
| KH0264 | 6 | 9 |
| KH0031 | 2 | 5 |
| KH0101 | 3 | 5 |
| KH0006 | 2 | 0 |
| KH0055 | 2 | 0 |
| KH0108 | 2 | 0 |
| KH0225 | 2 | 0 |
| KH0237 | 2 | 0 |
| KH0245 | 2 | 0 |

This drives 21 FNs (summary ≥ 2 "groups" but relation has < 2 triplets) and 16 FPs
(summary = 1 "group", below the `> 1` significance threshold, but relation has ≥ 2
triplets).

**Specific questions:**
1. What exactly is a "group" in `num_pendant_sub_neighbor_groups`?
   Is it a set of *consecutive* P−P.s−N triplets, or grouped by parent pendant?
2. For the 15 khipus where `summary ≥ 2` but `relation = 0`: were those relation
   rows filtered?  What criterion was used?
3. Since PSN is described in the KFG as "likely a statistical fluke", should the
   significance threshold be at the triplet level or the group level?

Note: this is the **only** remaining open question.  All other patterns are fully
resolved with 100.0% agreement.

---

## 5. Files

| File | Purpose |
|------|---------|
| `src/analysis/kfg_relation_loader.py` | Canonical loader — reads `*_relation.csv`; `apply_excl=False` default |
| `src/analysis/kfg_summation_detector.py` | Algorithmic fallback for non-KFG khipus |
| `data/kfg/KFG/KFG/checks/ascher_sums_overview.csv` | 7-pattern aggregation, added 2026-03-01 |
| `data/kfg/KFG/KFG/checks/*.csv` | Ground truth (9 summary + 9 relation + 1 overview CSVs) |
| `data/processed/kfg_fieldmarks_reconciliation.csv` | Full per-khipu reconciliation output |
| `scripts/reconcile_kfg_fieldmarks.py` | Reconciliation runner |
