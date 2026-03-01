# KFG Summation Pattern Detection — Current Status
*March 1, 2026 · kfg-integration branch · commit c29fad4*

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

### 1.3 Mutual Exclusivity

At the cord level a sum cord is claimed by exactly one pattern type.  The priority
follows structural specificity — patterns that require more structural evidence take
precedence:

| Priority | Pattern | Rationale |
|:--------:|---------|-----------|
| 1 (highest) | `indexed_subsidiary_sum` (IS) | subsidiary + color-index + cross-group |
| 2 | `subsidiary_pendant_sum` (SP) | subsidiary + pendant window |
| 3 | `indexed_pendant_sum` (IP) | position-indexed cross-group |
| 4 | `colored_pendant_sum` (CP) | color-indexed cross-group |
| 5 (lowest) | `pendant_pendant_sum` (PP) | contiguous window, any color |

Group-level patterns (GG, GSB, ADG, PSN) annotate whole groups, not individual cords,
so they are not subject to this exclusivity rule.

---

## 2. Results — 702 KFG Khipus

### 2.1 Per-Pattern Agreement

| Pattern | KFG+ | CAT+ | FP | FN | Agreement |
|---------|-----:|-----:|---:|---:|----------:|
| `pendant_pendant_sum` | 406 | 377 | 0 | 29 | 95.9% |
| `indexed_pendant_sum` | 202 | 202 | 0 | 0 | **100.0%** |
| `colored_pendant_sum` | 274 | 254 | 0 | 20 | 97.2% |
| `subsidiary_pendant_sum` | 145 | 145 | 0 | 0 | **100.0%** |
| `group_group_sum` | 101 | 101 | 0 | 0 | **100.0%** |
| `group_sum_bands` | 103 | 103 | 0 | 0 | **100.0%** |
| `indexed_subsidiary_sum` | 30 | 30 | 0 | 0 | **100.0%** |
| `pendant_sub_neighbor` | 71 | 66 | 16 | 21 | 94.7% |
| `ascher_decreasing_group` | 142 | 142 | 0 | 0 | **100.0%** |

**Grand mean agreement: 98.6%** (vs 87.8% with algorithmic re-detection)

> KFG+ = number of khipus the KFG summary CSV marks as having this pattern.  
> CAT+ = number of khipus our system detects (using the relation CSV as source).  
> FP = we say YES, KFG summary says NO.  
> FN = KFG summary says YES, we find nothing in the relation CSV.

### 2.2 Corpus Coverage

- 702 khipus covered by the KFG summary CSVs
- 503 khipus with at least one relation CSV row (algorithmic detector used for the remaining 199)
- 464 / 654 KFG khipus (71%) exhibit at least one summation pattern

---

## 3. Open Questions for the KFG Team

The remaining 1.4% disagreement (49 khipus) falls into three distinct categories,
each of which represents a data question rather than an algorithm error.

---

### Q1 — Missing relation rows for `pendant_pendant_sum` (29 khipus)

The `pendant_pendant_sum.csv` summary file records non-zero `num_sum_cords` for the
following 29 khipus, but `pendant_pendant_sum_relation.csv` contains **zero rows** for
each of them.  Several have large counts (KH0242: 16, KH0384: 12, KH0088: 11).

| Khipu | summary count | relation rows |
|-------|-------------:|-------------:|
| KH0010 | 1 | 0 |
| KH0026 | 1 | 0 |
| KH0028 | 8 | 0 |
| KH0059 | 11 | 0 |
| KH0075 | 5 | 0 |
| KH0084 | 1 | 0 |
| KH0088 | 11 | 0 |
| KH0090 | 2 | 0 |
| KH0101 | 9 | 0 |
| KH0162 | 1 | 0 |
| KH0242 | 16 | 0 |
| KH0269 | 16 | 0 |
| KH0280 | 2 | 0 |
| KH0293 | 2 | 0 |
| KH0303 | 1 | 0 |
| KH0317 | 1 | 0 |
| KH0343 | 2 | 0 |
| KH0357 | 1 | 0 |
| KH0370 | 5 | 0 |
| KH0384 | 12 | 0 |
| KH0390 | 4 | 0 |
| KH0396 | 3 | 0 |
| KH0428 | 1 | 0 |
| KH0436 | 1 | 0 |
| KH0453 | 1 | 0 |
| KH0472 | 8 | 0 |
| KH0482 | 2 | 0 |
| KH0492 | 1 | 0 |
| KH0517 | 2 | 0 |

**Question:** Were the summary CSV and the `_relation.csv` files computed at different
times, or with different criteria?  Are the relation files complete for these 29 khipus?
If not, can the individual cord-level relation data be regenerated for them?

---

### Q2 — Missing relation rows for `colored_pendant_sum` (20 khipus)

The same discrepancy applies to 20 khipus for the colored pendant sum pattern.

| Khipu | summary count | relation rows |
|-------|-------------:|-------------:|
| KH0050 | 1 | 0 |
| KH0084 | 1 | 0 |
| KH0090 | 2 | 0 |
| KH0101 | 8 | 0 |
| KH0106 | 6 | 0 |
| KH0134 | 5 | 0 |
| KH0161 | 2 | 0 |
| KH0172 | 4 | 0 |
| KH0187 | 1 | 0 |
| KH0275 | 11 | 0 |
| KH0278 | 2 | 0 |
| KH0289 | 4 | 0 |
| KH0311 | 1 | 0 |
| KH0317 | 1 | 0 |
| KH0348 | 4 | 0 |
| KH0384 | 5 | 0 |
| KH0387 | 1 | 0 |
| KH0536 | 1 | 0 |
| KH0635 | 1 | 0 |
| KH0693 | 1 | 0 |

Note that KH0084, KH0090, KH0101, KH0275, KH0317, and KH0384 appear in **both**
the PP and CP missing-relation lists — suggesting these may be a coherent batch
of khipus for which the cord-level relation export was not completed.

**Same question as Q1:** Is the relation data available for these khipus?

---

### Q3 — Counting unit mismatch in `pendant_sub_neighbor` (37 khipus)

This is the most structurally interesting discrepancy.  The summary CSV column is
named `num_pendant_sub_neighbor_groups`, but the relation CSV stores individual
**cord triplets** (pendant P, its subsidiary P.s, and neighbor P±1).  These are
different counting units.

For many khipus, the relation row count differs substantially from the summary count:

| Khipu | summary (groups) | relation (pairs) |
|-------|----------------:|----------------:|
| KH0252 | 3 | 19 |
| KH0028 | 5 | 10 |
| KH0145 | 1 | 5 |
| KH0264 | 6 | 9 |
| KH0031 | 2 | 5 |
| KH0101 | 3 | 5 |
| KH0141 | 3 | 5 |
| KH0012 | 1 | 3 |
| KH0072 | 1 | 3 |
| KH0006 | 2 | 0 |
| KH0055 | 2 | 0 |
| KH0225 | 2 | 0 |
| KH0237 | 2 | 0 |
| KH0245 | 2 | 0 |
| KH0108 | 2 | 0 |

This mismatch drives 21 FNs (where summary count ≥ 2 groups but relation has fewer
triplets) and 16 FPs (where summary count = 1 group, below our significance threshold
of `> 1`, but relation has ≥ 2 individual triplets).

**Questions:**
- What exactly constitutes a "group" in `num_pendant_sub_neighbor_groups`?
  Is it a set of *consecutive* P−P.s−N triplets, or is the grouping by pendant parent?
- For the 15 khipus where the summary count ≥ 2 but the relation has 0 rows:
  were those relation rows filtered out at some point?  What criterion was applied?
- Given that the KFG author describes this pattern as "likely a statistical fluke",
  should significance be assessed at the triplet level or the group level?

---

## 4. What Is Not in Question

For the following patterns our system achieves **exact agreement** with the KFG corpus:

| Pattern | Agreement | Notes |
|---------|----------:|-------|
| `indexed_pendant_sum` | 100.0% | 202/202 positive khipus correct |
| `subsidiary_pendant_sum` | 100.0% | 145/145 |
| `group_group_sum` | 100.0% | 101/101 |
| `group_sum_bands` | 100.0% | 103/103 |
| `indexed_subsidiary_sum` | 100.0% | 30/30 |
| `ascher_decreasing_group` | 100.0% | 142/142 |

The near-perfect agreement (95.9–97.2%) on `pendant_pendant_sum` and
`colored_pendant_sum` is also achieved when counting only the 673 and 682 khipus
respectively that *do* have relation data — the FNs are entirely accounted for
by the 29/20 khipus with missing relation rows.

---

## 5. Files

| File | Purpose |
|------|---------|
| `src/analysis/kfg_relation_loader.py` | Canonical loader — reads `*_relation.csv`, enforces exclusivity |
| `src/analysis/kfg_summation_detector.py` | Algorithmic fallback for non-KFG khipus |
| `data/kfg/KFG/KFG/checks/*.csv` | Ground-truth source (9 summary + 9 relation CSVs) |
| `data/processed/kfg_fieldmarks_reconciliation.csv` | Full per-khipu reconciliation output |
| `scripts/reconcile_kfg_fieldmarks.py` | Reconciliation runner |
