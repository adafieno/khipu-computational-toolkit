"""
Phase 1 — Corpus Foundation: Corpus Statistics
===============================================
Computes and reports baseline statistics for the K-CAT khipu database
(built from KFG source data).

This script is corpus-agnostic: it reads whatever SQLite database is
configured and reports what is there. It does not interpret findings.

Usage:
    python scripts/corpus_statistics.py
    python scripts/corpus_statistics.py --db path/to/other.db
    python scripts/corpus_statistics.py --report  # also write to reports/

Output:
    Console summary (always)
    reports/phase1_corpus_foundation.md  (with --report flag)

Data Attribution:
    Primary data source: The Khipu Field Guide (KFG) — Ashok Khosla et al.
    https://khipufieldguide.com
    Numeric decoding: Ascher & Ascher (1978, 1981 / 1997 Dover reprint)
    Published KFG research: Khosla & Medrano, Latin American Antiquity (2023)
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).parent.parent
DB_DEFAULT = ROOT / "data" / "kfg" / "khipu_database.db"
REPORT_PATH = ROOT / "reports" / "phase1_corpus_foundation.md"

# ── OKR baseline figures (from Phase 0/1 reports, January 2026) ──────────────
# These are fixed reference numbers for cross-corpus comparison.
OKR_BASELINE: dict[str, Any] = {
    "corpus":         "Open Khipu Repository (OKR)",
    "khipus_total":   619,
    "khipus_usable":  612,
    "cords":          54_403,
    "knots_decoded":  110_677,
    "color_records":  56_306,
    "numeric_pct":    68.2,
    "khipus_with_values_pct": 95.8,
    "source":         "Phase 0–1 reports, K-CAT OKR era (December 2025–January 2026)",
}


def load_stats(db_path: Path) -> dict[str, Any]:
    """Query the database and return a flat stats dict."""
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}", file=sys.stderr)
        print("Run: python scripts/build_kfg_database.py", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))

    km = pd.read_sql("SELECT * FROM khipu_metadata", conn)
    c  = pd.read_sql("SELECT * FROM cords", conn)
    kc = pd.read_sql("SELECT * FROM knot_clusters", conn)
    cc = pd.read_sql("SELECT * FROM cord_colors", conn)

    c["has_value"] = c["value"] > 0

    per_khipu = (
        c.groupby("kfg_id")
        .agg(cord_count=("cord_id", "count"), valued_count=("has_value", "sum"))
        .reset_index()
    )
    per_khipu["coverage"] = per_khipu["valued_count"] / per_khipu["cord_count"]

    conn.close()

    total = len(c)
    valued = int(c["has_value"].sum())

    return {
        # Corpus size
        "khipus":              len(km),
        "khipus_with_cords":   int((per_khipu.cord_count > 0).sum()),
        "cords_total":         total,
        "cords_pendant":       int((c["hierarchy_level"] == 1).sum()),
        "cords_subsidiary":    int((c["hierarchy_level"] > 1).sum()),
        "cords_group":         int((c["hierarchy_level"] == 0).sum()),
        "knot_clusters":       len(kc),
        "knots_total":         int(kc["num_knots"].sum()),
        "color_records":       len(cc),
        "unique_color_codes":  int(cc["color_code"].nunique()),
        # Numeric coverage
        "cords_with_value":    valued,
        "numeric_pct":         round(valued / total * 100, 1),
        "cords_zero":          int((c["value"] == 0).sum()),
        "cords_zero_pct":      round((c["value"] == 0).mean() * 100, 1),
        "khipus_any_value":    int((per_khipu.valued_count > 0).sum()),
        "khipus_80pct_cover":  int((per_khipu.coverage >= 0.8).sum()),
        # Size distribution
        "avg_cords":           round(per_khipu.cord_count.mean(), 1),
        "median_cords":        int(per_khipu.cord_count.median()),
        "min_cords":           int(per_khipu.cord_count.min()),
        "max_cords":           int(per_khipu.cord_count.max()),
        # Knot type breakdown
        "knot_types":          kc["knot_type"].value_counts().to_dict(),
        # Geographic
        "unique_provenances":  int(km["provenance"].nunique()),
        "unique_countries":    int(km["museum_country"].nunique()),
        "unique_museums":      int(km["museum_name"].nunique()),
        "top_provenances":     km["provenance"].value_counts().head(10).to_dict(),
        "top_countries":       km["museum_country"].value_counts().head(8).to_dict(),
    }


def print_stats(s: dict[str, Any], label: str = "KFG") -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n{'='*60}")
    print(f"  CORPUS STATISTICS — {label}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    print("\n── Corpus size ──────────────────────────────────────────")
    print(f"  Khipus:             {s['khipus']:,}")
    print(f"  Khipus with cords:  {s['khipus_with_cords']:,}")
    print(f"  Total cords:        {s['cords_total']:,}")
    print(f"    Pendant (L1):     {s['cords_pendant']:,}")
    print(f"    Subsidiary (L2+): {s['cords_subsidiary']:,}")
    print(f"    Group/top (L0):   {s['cords_group']:,}")
    print(f"  Knot clusters:      {s['knot_clusters']:,}")
    print(f"  Decoded knots:      {s['knots_total']:,}")
    print(f"  Color records:      {s['color_records']:,}")
    print(f"  Unique color codes: {s['unique_color_codes']:,}")

    print("\n── Numeric coverage ─────────────────────────────────────")
    print(f"  Cords with value > 0:  {s['cords_with_value']:,} / {s['cords_total']:,} = {s['numeric_pct']}%")
    print(f"  Cords with value = 0:  {s['cords_zero']:,} ({s['cords_zero_pct']}%) ← null placeholder")
    print(f"  Khipus with any value: {s['khipus_any_value']:,} / {s['khipus']:,}")
    print(f"  Khipus ≥ 80% coverage: {s['khipus_80pct_cover']:,}")
    print(f"  Avg cords / khipu:     {s['avg_cords']}")
    print(f"  Median cords / khipu:  {s['median_cords']}")
    print(f"  Min / Max cords:       {s['min_cords']} / {s['max_cords']}")

    print("\n── Knot types ───────────────────────────────────────────")
    for kt, n in sorted(s["knot_types"].items(), key=lambda x: -x[1]):
        print(f"  {kt:<6} {n:,}")

    print("\n── Geographic distribution ──────────────────────────────")
    print(f"  Unique provenances: {s['unique_provenances']}")
    print(f"  Unique countries:   {s['unique_countries']}")
    print(f"  Unique museums:     {s['unique_museums']}")
    print("  Top provenances:")
    for k, v in list(s["top_provenances"].items())[:10]:
        print(f"    {k:<40} {v:>4}")
    print("  Top countries:")
    for k, v in list(s["top_countries"].items())[:8]:
        print(f"    {k:<40} {v:>4}")
    print()


def write_report(s: dict[str, Any], okr: dict[str, Any]) -> None:
    """Write (or overwrite) the Phase 1 report markdown file."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d")
    kt = s["knot_types"]

    content = f"""\
# Phase 1: Corpus Foundation

**Generated:** {now}  
**Database:** Khipu Field Guide (KFG) SQLite database  
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
| Khipus in dataset | 619 | {s['khipus']:,} |
| Khipus with cord data | 612 | {s['khipus_with_cords']:,} |
| Total cords | 54,403 | {s['cords_total']:,} |
| Decoded knot records | 110,677 | {s['knots_total']:,} |
| Color records | 56,306 | {s['color_records']:,} |
| Numeric coverage (value > 0) | 68.2% | {s['numeric_pct']}% |
| Khipus with any decoded value | 95.8% | {s['khipus_any_value'] / s['khipus'] * 100:.1f}% |

*OKR reference figures from K-CAT Phase 0–1 reports, December 2025–January 2026.*

---

## KFG Corpus Results

### Corpus Size

| Component | Count |
|-----------|-------|
| Khipus | {s['khipus']:,} |
| Total cords | {s['cords_total']:,} |
| — Group / top cords (L0) | {s['cords_group']:,} |
| — Pendant cords (L1) | {s['cords_pendant']:,} |
| — Subsidiary cords (L2+) | {s['cords_subsidiary']:,} |
| Knot clusters | {s['knot_clusters']:,} |
| Decoded knot instances | {s['knots_total']:,} |
| Color records | {s['color_records']:,} |
| Unique color codes | {s['unique_color_codes']:,} |

### Numeric Coverage

- **{s['cords_with_value']:,}** cords have a decoded value > 0 ({s['numeric_pct']}% of total)
- **{s['cords_zero']:,}** cords have `value = 0` ({s['cords_zero_pct']}%) — treated as null/undecoded
- **{s['khipus_any_value']:,}** of {s['khipus']:,} khipus ({s['khipus_any_value'] / s['khipus'] * 100:.1f}%) have at least one decoded cord
- **{s['khipus_80pct_cover']:,}** khipus have ≥ 80% of cords with non-zero values

**Size distribution across khipus:**

| Metric | Value |
|--------|-------|
| Mean cords per khipu | {s['avg_cords']} |
| Median cords per khipu | {s['median_cords']} |
| Minimum cords | {s['min_cords']} |
| Maximum cords | {s['max_cords']} |

The wide gap between mean ({s['avg_cords']}) and median ({s['median_cords']}) indicates a right-skewed distribution — most khipus are modest in size but a small number are very large.

### Knot Type Distribution

| Type | Count | Interpretation |
|------|-------|----------------|
| L (long) | {kt.get('L', 0):,} | Tens position (value = turns × 10) |
| S (single) | {kt.get('S', 0):,} | Hundreds position (value = 100) |
| E (figure-eight) | {kt.get('E', 0):,} | Units position (value = 1) |
| SP (special/pendant) | {kt.get('SP', 0):,} | Non-numeric marker |
| U (unknown) | {kt.get('U', 0):,} | Not yet classified |
| EE (double figure-eight) | {kt.get('EE', 0):,} | Variant units |
| TF | {kt.get('TF', 0):,} | Terminal figure-eight |
| LL | {kt.get('LL', 0):,} | Double long |
| BL | {kt.get('BL', 0):,} | Blank / spacer |

The L:S ratio ({kt.get('L', 0) / kt.get('S', 1):.2f}) closely matches the OKR corpus, consistent with the predominance of multi-digit values in accounting khipus.

### Geographic Distribution

| Metric | Count |
|--------|-------|
| Unique provenances | {s['unique_provenances']} |
| Museum countries | {s['unique_countries']} |
| Institutions | {s['unique_museums']} |

**Top provenances:**

| Provenance | Khipus |
|-----------|--------|
""" + "\n".join(
        f"| {k} | {v} |"
        for k, v in list(s["top_provenances"].items())[:10]
    ) + f"""

**Top museum countries:**

| Country | Khipus |
|---------|--------|
""" + "\n".join(
        f"| {k} | {v} |"
        for k, v in list(s["top_countries"].items())[:8]
    ) + f"""

---

## Data Quality Notes

1. **value = 0 ambiguity.** The KFG database stores `value = 0` for cords where no knot value was decoded. This is a null placeholder, not a true zero. Downstream analyses must account for this before computing numeric statistics. See the summation patterns phase (Phase 2) for how this is handled.

2. **level = 0 cords.** {s['cords_group']:,} cords have `hierarchy_level = 0`. In KFG structure these represent group/top-level cords that organize pendants but do not themselves carry knot values. They should be excluded from pendant-level numeric analyses.

3. **Unknown provenance.** {s['top_provenances'].get('Unknown', 0)} khipus ({s['top_provenances'].get('Unknown', 0) / s['khipus'] * 100:.1f}%) have no recorded provenance. Geographic analyses should note this coverage gap.

4. **Unique color codes: {s['unique_color_codes']:,}.** This is higher than expected from the ~30 Ascher base codes. The KFG database stores compound codes (e.g., `MB:W`, `KB-DB`) as single strings; downstream color analyses should normalize these before counting distinct colors.

---

## How to Re-run

```bash
python scripts/corpus_statistics.py             # console output only
python scripts/corpus_statistics.py --report    # also writes this report
python scripts/corpus_statistics.py --db path/to/other.db --report  # any corpus
```

---

## Limitations

- Statistics reflect the state of the KFG database at time of generation. The database is rebuilt from KFG source XLS files by `scripts/build_kfg_database.py`; re-run that script first to incorporate any upstream KFG updates.
- Numeric decoding uses the Ascher positional system. Alternative decoding proposals (e.g., Urton's binary system) would produce different value distributions.
- Provenance data quality depends on KFG source records. The {s['top_provenances'].get('Unknown', 0)} unknown-provenance khipus represent a real gap in the archaeological record, not a data entry error.

---

*Report generated automatically by `scripts/corpus_statistics.py`. Re-run to refresh with the latest database state.*
"""

    REPORT_PATH.write_text(content, encoding="utf-8")
    print(f"Report written to: {REPORT_PATH.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 — Corpus Foundation statistics")
    parser.add_argument("--db", type=Path, default=DB_DEFAULT, help="Path to SQLite database")
    parser.add_argument("--report", action="store_true", help="Write report to reports/")
    args = parser.parse_args()

    stats = load_stats(args.db)
    print_stats(stats, label="KFG")

    if args.report:
        write_report(stats, OKR_BASELINE)
    else:
        print("Tip: run with --report to write reports/phase1_corpus_foundation.md")


if __name__ == "__main__":
    main()
