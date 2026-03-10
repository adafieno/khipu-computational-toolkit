"""
Reconciliation: KFG Ascher Sum Ground-Truth vs K-CAT detector output (v2).

GROUND-TRUTH SOURCE: data/kfg/KFG/KFG/checks/*.csv
  These files contain one row per khipu (all 703), with per-pattern statistics
  computed by the KFG team.  They are the authoritative source, avoiding the
  HTML-column-order ambiguity of the fieldmarks browser page.

NOTE ON FIELDMARKS PAGE COLUMN ORDER
  The https://khipufieldguide.com/fieldmarks page shows 7 columns in this
  order (DIFFERENT from the analysis-page narrative):
      1. pendant_pendant_sum      (num_sum_cords)
      2. indexed_pendant_sum      (num_sum_cords)   <- NOT colored
      3. colored_pendant_sum      (num_sum_cords)   <- NOT indexed
      4. subsidiary_pendant_sum   (num_sum_cords)
      5. group_sum_bands          (num_group_sum_bands)
      6. group_group_sum          (num_sum_groups)
      7. ascher_decreasing_group  (num_decreasing_groups)
  indexed_subsidiary_sum and pendant_sub_neighbor are NOT shown on that page.

SIGNIFICANCE THRESHOLDS (from individual analysis pages)
  pendant_pendant_sum    : > 0  (any match)
  indexed_pendant_sum    : > 0  (significance > mean 7, but fieldmarks uses any)
  colored_pendant_sum    : > 0
  subsidiary_pendant_sum : > 0
  group_group_sum        : > 0
  group_sum_bands        : > 0
  indexed_subsidiary_sum : > 1  ("1 deemed possibly accidental")
  pendant_sub_neighbor   : > 1  (same reasoning)
  ascher_decreasing_group: any  (has_decreasing_groups == True)

K-CAT detector implements all 9 patterns.  The reconciler compares each one
against the KFG ground truth.

Usage:
    python scripts/reconcile_kfg_fieldmarks.py
"""

import csv
import sys
import sqlite3
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent.parent
DB      = ROOT / "data" / "kfg" / "khipu_database.db"
CHKS    = ROOT / "data" / "kfg" / "KFG" / "KFG" / "checks"
OUT_CSV = ROOT / "data" / "processed" / "kfg_fieldmarks_reconciliation.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from src.analysis.kfg_summation_detector import KFGSummationDetector


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------
# Each entry: (short_key, csv_file, value_column, significance_threshold)
# significance_threshold = minimum *exclusive* value that counts as a match.
# E.g. thresh=0 means any value > 0; thresh=1 means count must be > 1.

PATTERNS = [
    ("pp",  "pendant_pendant_sum.csv",      "num_sum_cords",                   0),
    ("ip",  "indexed_pendant_sum.csv",       "num_sum_cords",                   0),
    ("cp",  "colored_pendant_sum.csv",       "num_sum_cords",                   0),
    ("sp",  "subsidiary_pendant_sum.csv",    "num_sum_cords",                   0),
    ("gg",  "group_group_sum.csv",           "num_sum_groups",                  0),
    ("gsb", "group_sum_bands.csv",           "num_group_sum_bands",             0),
    ("is",  "indexed_subsidiary_sum.csv",    "num_sum_cords",                   1),
    ("psn", "pendant_sub_neighbor.csv",      "num_pendant_sub_neighbor_groups", 1),
    ("adg", "ascher_decreasing_group.csv",   "num_decreasing_groups",           0),
]


# ---------------------------------------------------------------------------
# Step 1: Load KFG ground-truth from checks CSVs
# ---------------------------------------------------------------------------

def load_kfg_ground_truth():
    """Load per-khipu binary flags from the KFG checks CSV files.

    Returns a DataFrame with columns:
        khipu_id, kfg_pp, kfg_ip, kfg_cp, kfg_sp, kfg_gg, kfg_gsb,
        kfg_is, kfg_psn, kfg_adg, kfg_any
    """
    roster_path = CHKS / "pendant_pendant_sum.csv"
    with open(roster_path, newline="", encoding="utf-8") as f:
        roster_rows = list(csv.reader(f))
    khipu_ids = [row[0] for row in roster_rows[1:]]

    result = {kid: {} for kid in khipu_ids}

    for key, fname, val_col, thresh in PATTERNS:
        path = CHKS / fname
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        header = rows[0]
        col_idx = header.index(val_col) if val_col in header else None

        for row in rows[1:]:
            kid = row[0]
            if kid not in result:
                continue
            raw = row[col_idx] if col_idx is not None else "0"
            try:
                val = int(float(raw))
            except (ValueError, TypeError):
                val = 1 if raw == "True" else 0
            result[kid][f"kfg_{key}"] = 1 if val > thresh else 0

    records = []
    kfg_cols = [f"kfg_{k}" for k, *_ in PATTERNS]
    for kid in khipu_ids:
        row = {"khipu_id": kid}
        row.update(result[kid])
        row["kfg_any"] = int(any(result[kid].get(c, 0) for c in kfg_cols))
        records.append(row)

    df = pd.DataFrame(records)
    n_any = df["kfg_any"].sum()
    print(f"KFG ground truth loaded: {len(df)} khipus, "
          f"{n_any} ({n_any/len(df)*100:.1f}%) have any summation.")
    print("\nKFG pattern counts (khipus with flag=1):")
    for key, *_ in PATTERNS:
        col = f"kfg_{key}"
        n = df[col].sum()
        print(f"  {col:12s}: {n:4d} ({n/len(df)*100:.1f}%)")
    return df


# ---------------------------------------------------------------------------
# Step 2: Run K-CAT detector
# ---------------------------------------------------------------------------

def run_kcat_detector(kfg_ids):
    """Run KFGSummationDetector on every khipu in kfg_ids.

    Returns a DataFrame with columns:
        khipu_id, kcat_any, kcat_pp, kcat_ip, kcat_cp, kcat_sp,
        kcat_gg, kcat_gsb, kcat_is, kcat_psn, kcat_adg
    """
    detector = KFGSummationDetector(str(DB))

    conn = sqlite3.connect(str(DB))
    all_kcat_ids = set(
        pd.read_sql("SELECT kfg_id FROM khipu_metadata", conn)["kfg_id"].tolist()
    )
    conn.close()

    # Map K-CAT detector pattern keys -> our short keys
    detector_key_map = {
        "pendant_pendant_sum":      "pp",
        "indexed_pendant_sum":      "ip",
        "colored_pendant_sum":      "cp",
        "subsidiary_pendant_sum":   "sp",
        "group_group_sum":          "gg",
        "group_sum_bands":          "gsb",
        "indexed_subsidiary_sum":   "is",
        "pendant_sub_neighbor":     "psn",
        "ascher_decreasing_group":  "adg",
    }

    # Significance thresholds for K-CAT detector match counts
    kcat_thresholds = {
        "is":  1,   # > 1 occurrence
        "psn": 1,   # > 1 group
    }

    rows = []
    none_row = {"kcat_any": None, **{f"kcat_{k}": None for k, *_ in PATTERNS}}

    for kid in kfg_ids:
        if kid not in all_kcat_ids:
            rows.append({"khipu_id": kid, **none_row})
            continue
        try:
            summary = detector.summarize(kid)
            stats = summary.get("pattern_stats", {})
            row = {"khipu_id": kid}
            for dkey, skey in detector_key_map.items():
                n = stats.get(dkey, {}).get("matches", 0)
                th = kcat_thresholds.get(skey, 0)
                row[f"kcat_{skey}"] = 1 if n > th else 0
            kcat_cols = [f"kcat_{k}" for k, *_ in PATTERNS]
            row["kcat_any"] = int(any(row.get(c, 0) for c in kcat_cols))
            rows.append(row)
        except Exception as exc:
            print(f"  WARNING: detector failed on {kid}: {exc}")
            rows.append({"khipu_id": kid, **none_row})

    df = pd.DataFrame(rows)
    valid = df["kcat_any"].notna()
    n_valid = valid.sum()
    n_any = df.loc[valid, "kcat_any"].sum()
    print(f"\nKCAT detector run on {len(kfg_ids)} khipus.")
    print(f"  K-CAT 'has summation': {n_any} / {n_valid} "
          f"({n_any/n_valid*100:.1f}%)")
    return df


# ---------------------------------------------------------------------------
# Step 3: Merge and print summary
# ---------------------------------------------------------------------------

def reconcile(kfg, kcat):
    return pd.merge(kfg, kcat, on="khipu_id", how="outer", indicator=True)


def print_summary(merged):
    both  = merged.dropna(subset=["kfg_any", "kcat_any"])
    total = len(both)

    agree_p = ((both["kfg_any"] == 1) & (both["kcat_any"] == 1)).sum()
    agree_n = ((both["kfg_any"] == 0) & (both["kcat_any"] == 0)).sum()
    kcat_fp = ((both["kfg_any"] == 0) & (both["kcat_any"] == 1)).sum()
    kfg_fn  = ((both["kfg_any"] == 1) & (both["kcat_any"] == 0)).sum()
    agreement_rate = (agree_p + agree_n) / total * 100

    kfg_only  = (merged["_merge"] == "left_only").sum()
    kcat_only = (merged["_merge"] == "right_only").sum()

    print("\n" + "=" * 70)
    print("RECONCILIATION SUMMARY  (KFG checks CSVs vs K-CAT detector v2)")
    print("=" * 70)
    print(f"Total rows merged                           : {len(merged)}")
    print(f"  Khipus in both KFG checks + K-CAT DB       : {total}")
    print(f"  KFG checks only (not in K-CAT DB)          : {kfg_only}")
    print(f"  K-CAT DB only    (not in KFG checks)       : {kcat_only}")
    print()
    print("Overall 'has summation' agreement:")
    print(f"  Both positive (agree +)                   : {agree_p}")
    print(f"  Both negative (agree -)                   : {agree_n}")
    print(f"  K-CAT positive, KFG negative (FP)          : {kcat_fp}")
    print(f"  KFG positive,  K-CAT negative (FN)         : {kfg_fn}")
    print(f"\n  Agreement rate (in-both set)              : {agreement_rate:.1f}%")

    # Per-pattern breakdown
    print("\nPer-pattern (KFG checks vs K-CAT, in-both set):")
    header = (f"  {'Pattern':<8}  {'Sig':>3}  {'KFG+':>5}  {'K-CAT+':>5}  "
              f"{'Agree+':>6}  {'Agree-':>6}  {'FP':>4}  {'FN':>4}  {'Agr%':>6}")
    print(header)
    print("  " + "-" * 68)

    for key, _, _, thresh in PATTERNS:
        kfg_col  = f"kfg_{key}"
        kcat_col = f"kcat_{key}"
        sub = both.dropna(subset=[kfg_col, kcat_col])
        if sub.empty:
            continue
        kfg_pos  = (sub[kfg_col]  == 1).sum()
        kcat_pos = (sub[kcat_col] == 1).sum()
        ap = ((sub[kfg_col] == 1) & (sub[kcat_col] == 1)).sum()
        an = ((sub[kfg_col] == 0) & (sub[kcat_col] == 0)).sum()
        fp = ((sub[kfg_col] == 0) & (sub[kcat_col] == 1)).sum()
        fn = ((sub[kfg_col] == 1) & (sub[kcat_col] == 0)).sum()
        agr = (ap + an) / len(sub) * 100
        sig = f">{thresh}" if thresh else ">=1"
        print(f"  {key:<8}  {sig:>3}  {kfg_pos:>5}  {kcat_pos:>5}  "
              f"{ap:>6}  {an:>6}  {fp:>4}  {fn:>4}  {agr:>5.1f}%")

    print("=" * 70)
    print()
    print("Notes:")
    print("  PP, IP, CP, SP, GSB, GG, ADG = 7 cols on KFG fieldmarks page")
    print("  IS, PSN  = verified via checks CSVs; not on fieldmarks page")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading KFG ground truth from checks CSVs ...")
    kfg_df  = load_kfg_ground_truth()
    print("\nRunning K-CAT detector ...")
    kcat_df = run_kcat_detector(kfg_df["khipu_id"].tolist())
    merged  = reconcile(kfg_df, kcat_df)
    print_summary(merged)

    merged.to_csv(OUT_CSV, index=False)
    print(f"\nFull per-khipu comparison saved to:\n  {OUT_CSV}")
