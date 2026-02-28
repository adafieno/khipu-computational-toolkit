# Phase 1: Corpus Foundation

**Generated:** 2026-02-28  
**Database:** KCAT SQLite database (built from KFG source data)  
**Script:** `scripts/corpus_statistics.py`  
**Status:** ✅ Complete

---

## Research Question

What does the khipu corpus look like at baseline? How many objects are represented, how complete are the data, and what are the distributional properties across cords, knots, colors, and provenance? This phase establishes the empirical foundation that all subsequent phases build on.

---

## Methodology

The baseline statistics script queries the KFG SQLite database and reports:

1. **Corpus size** — total khipus, cords, knots, and color records
2. **Cord hierarchy** — breakdown by structural level (group cord L0, pendant L1, subsidiary L2+)
3. **Numeric coverage** — fraction of cords with decoded non-zero values; treatment of `value=0` as a null placeholder
4. **Knot type distribution** — S (single/hundreds), L (long/tens), E (figure-eight/units), and special types
5. **Geographic distribution** — provenance, museum country, institution counts

**Numeric decoding convention (Ascher & Ascher positional notation):**
- `S` knot = hundreds position = 100
- `L` knot = tens position = NUM_TURNS × 10
- `E` knot = units position = 1
- Cord value = sum of all knot values on that cord
- `value = 0` is used as a null/missing-value placeholder in the KFG database

This methodology is identical to the OKR baseline and is described fully in Ascher & Ascher (1978, 1981).

---

## Cross-Corpus Comparison

| Metric | OKR (reference) | KFG (this run) |
|--------|----------------|----------------|
| Khipus in dataset | 619 | 709 |
| Khipus with cord data | 612 | 709 |
| Total cords | 54,403 | 62,746 |
| Decoded knot records | 110,677 | 238,099 |
| Color records | 56,306 | 76,258 |
| Numeric coverage (value > 0) | 68.2% | 69.7% |
| Khipus with any decoded value | 95.8% | 98.2% |

*OKR reference figures from KCAT Phase 0–1 reports, December 2025–January 2026.*

---

## KFG Corpus Results

### Corpus Size

| Component | Count |
|-----------|-------|
| Khipus | 709 |
| Total cords | 62,746 |
| — Group / top cords (L0) | 45,096 |
| — Pendant cords (L1) | 15,465 |
| — Subsidiary cords (L2+) | 2,162 |
| Knot clusters | 70,143 |
| Decoded knot instances | 238,099 |
| Color records | 76,258 |
| Unique color codes | 2,830 |

### Numeric Coverage

- **43,706** cords have a decoded value > 0 (69.7% of total)
- **19,040** cords have `value = 0` (30.3%) — treated as null/undecoded
- **696** of 709 khipus (98.2%) have at least one decoded cord
- **381** khipus have ≥ 80% of cords with non-zero values

**Size distribution across khipus:**

| Metric | Value |
|--------|-------|
| Mean cords per khipu | 88.5 |
| Median cords per khipu | 42 |
| Minimum cords | 1 |
| Maximum cords | 1831 |

The wide gap between mean (88.5) and median (42) indicates a right-skewed distribution — most khipus are modest in size but a small number are very large.

### Knot Type Distribution

| Type | Count | Interpretation |
|------|-------|----------------|
| L (long) | 30,651 | Tens position (value = turns × 10) |
| S (single) | 29,278 | Hundreds position (value = 100) |
| E (figure-eight) | 9,580 | Units position (value = 1) |
| SP (special/pendant) | 217 | Non-numeric marker |
| U (unknown) | 152 | Not yet classified |
| EE (double figure-eight) | 117 | Variant units |
| TF | 110 | Terminal figure-eight |
| LL | 27 | Double long |
| BL | 11 | Blank / spacer |

The L:S ratio (1.05) closely matches the OKR corpus, consistent with the predominance of multi-digit values in accounting khipus.

### Geographic Distribution

| Metric | Count |
|--------|-------|
| Unique provenances | 89 |
| Museum countries | 12 |
| Institutions | 73 |

**Top provenances:**

| Provenance | Khipus |
|-----------|--------|
| Unknown | 236 |
| Pachacamac | 86 |
| Ica | 52 |
| Incahuasi | 52 |
| Leymebamba | 22 |
| Huaquerones | 19 |
| Nazca | 13 |
| Huacones | 11 |
| Armatambo, Huaca San Pedro | 11 |
| Eduard Gaffron | 10 |

**Top museum countries:**

| Country | Khipus |
|---------|--------|
| Peru | 105 |
| Germany | 70 |
| USA | 64 |
| France | 8 |
| Israel | 4 |
| Great Britain | 4 |
| Switzerland | 4 |
| Holland | 1 |

---

## Data Quality Notes

1. **value = 0 ambiguity.** The KFG database stores `value = 0` for cords where no knot value was decoded. This is a null placeholder, not a true zero. Downstream analyses must account for this before computing numeric statistics. See the summation patterns phase (Phase 2) for how this is handled.

2. **level = 0 cords.** 45,096 cords have `hierarchy_level = 0`. In KFG structure these represent group/top-level cords that organize pendants but do not themselves carry knot values. They should be excluded from pendant-level numeric analyses.

3. **Unknown provenance.** 236 khipus (33.3%) have no recorded provenance. Geographic analyses should note this coverage gap.

4. **Unique color codes: 2,830.** This is higher than expected from the ~30 Ascher base codes. The KFG database stores compound codes (e.g., `MB:W`, `KB-DB`) as single strings; downstream color analyses should normalize these before counting distinct colors.

---

## How to Re-run

```bash
python scripts/corpus_statistics.py             # console output only
python scripts/corpus_statistics.py --report    # also writes this report
python scripts/corpus_statistics.py --db path/to/other.db --report  # any corpus
```

---

## Limitations

- Statistics reflect the state of the KCAT database at time of generation. The KCAT database is rebuilt from KFG source XLS files by `scripts/build_kfg_database.py`; re-run that script first to incorporate any upstream KFG updates.
- Numeric decoding uses the Ascher positional system. Alternative decoding proposals (e.g., Urton's binary system) would produce different value distributions.
- Provenance data quality depends on KFG source records. The 236 unknown-provenance khipus represent a real gap in the archaeological record, not a data entry error.

---

## Citations and Acknowledgments

### Primary Data Source

This report analyzes the **Khipu Field Guide (KFG)** database, the world's most accurate and largest digital database of Inka khipus.

> Khosla, Ashok. *The Khipu Field Guide*. [khipufieldguide.com](https://khipufieldguide.com), 2020–present.

The KFG was created and is edited by **Ashok Khosla**. Substantial database curation and correction work was contributed by **Karen Thompson** (Senior Research Data Specialist, University of Melbourne), along with KFG affiliates **Manuel Medrano** (Harvard University), **Kylie Quave** (George Washington University), **Mack FitzPatrick** (Harvard University), **Saoirse Byrne**, and **Andrés Chirinos**. Per Ashok Khosla: "Karen Thompson and I both have invested at least 3 or 4 person-years of effort in improving and correcting the database."

### Numeric Decoding Methodology

The positional notation system used to decode cord values from knot type and position was established by:

> Ascher, Marcia and Robert Ascher. *Mathematics of the Incas: Code of the Quipu*. Dover Publications, 1997. (Reprint of the 1981 edition.)

> Ascher, Marcia and Robert Ascher. "Code of the Quipu: Databook." Cornell University, 1978.

### Published KFG Research

The primary publication applying data science methodology to KFG khipu data:

> Khosla, Ashok and Manuel Medrano. "How Can Data Science Contribute to Understanding the Khipu Code?" *Latin American Antiquity*, 2023.

Karen Thompson's work on KFG Ascher khipus (including the relationship between KH0082 and KH0083) has been published in *Nawpa Pacha* (Journal of Andean Archaeology).

### Historical Reference Corpus

Cross-corpus comparison figures in this report use the **Open Khipu Repository (OKR)** as a historical baseline. The OKR is the earlier community-maintained dataset that preceded the KFG. Per the KFG team, the KFG is now the authoritative source for computational khipu research.

> Open Khipu Repository. Harvard Dataverse. (Historical reference only — superseded by the KFG.)

### About This Toolkit

The Khipu Computational Analysis Toolkit (KCAT) is an independent computational companion to the KFG, providing reproducible Python scripts for corpus-scale methods research. It does not modify or redistribute the underlying KFG data. If you use these analyses, please cite the KFG primary source and the Ascher & Ascher reference for the decoding methodology.

---

*Report generated automatically by `scripts/corpus_statistics.py`. Re-run to refresh with the latest database state.*
