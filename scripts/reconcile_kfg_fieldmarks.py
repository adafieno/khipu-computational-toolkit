"""
Reconciliation: KFG Ascher Sum Browser vs KCAT detector output.

Fetches the per-khipu summation fieldmark table from khipufieldguide.com/fieldmarks
and compares it against KFGSummationDetector.summarize() on the KCAT database.

KFG publishes 7 Ascher fieldmarks (4 pendant sums + 3 group relationships):
  col1: pendant_pendant_sum
  col2: colored_pendant_sum
  col3: indexed_pendant_sum
  col4: subsidiary_pendant_sum
  col5: group_group_sum
  col6: indexed_subsidiary_sum
  col7: pendant_sub_neighbor  (or ascher_decreasing_group)

KCAT detector implements 8 types (above + ascher_decreasing_group).

Usage:
    python scripts/reconcile_kfg_fieldmarks.py
"""

import re
import sys
import sqlite3
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "kfg" / "khipu_database.db"
OUT_CSV = ROOT / "data" / "processed" / "kfg_fieldmarks_reconciliation.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from src.analysis.kfg_summation_detector import KFGSummationDetector

KFG_URL = "https://khipufieldguide.com/fieldmarks"

# ---------------------------------------------------------------------------
# Step 1: Fetch & parse KFG Ascher Sum Browser table
# ---------------------------------------------------------------------------

def fetch_kfg_table() -> pd.DataFrame:
    """Fetch the KFG fieldmarks page and parse the summation table.

    Returns a DataFrame indexed by khipu_id with columns:
        kfg_pp, kfg_cp, kfg_ip, kfg_sp, kfg_gg, kfg_is, kfg_psn
        kfg_any  (True if any of the 7 counts > 0)
    """
    print(f"Fetching {KFG_URL} ...")
    resp = requests.get(KFG_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # The fieldmarks table has alternating icon/count column pairs.
    # Columns: [index, khipu_id, icon1, count1, icon2, count2, ..., icon7, count7]
    rows = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 16:
            continue  # skip header / malformed
        try:
            khipu_id = tds[1].get_text(strip=True)
            if not khipu_id:
                continue
            # extract the 7 count columns (odd positions 3,5,7,9,11,13,15)
            counts = []
            for i in [3, 5, 7, 9, 11, 13, 15]:
                txt = tds[i].get_text(strip=True)
                counts.append(int(txt) if txt.isdigit() else 0)
            rows.append([khipu_id] + counts)
        except (IndexError, ValueError):
            continue

    if not rows:
        raise RuntimeError("Failed to parse any rows from the KFG fieldmarks page. "
                           "The page layout may have changed.")

    df = pd.DataFrame(
        rows,
        columns=["khipu_id", "kfg_pp", "kfg_cp", "kfg_ip",
                 "kfg_sp", "kfg_gg", "kfg_is", "kfg_psn"]
    )
    # Deduplicate: KH0350 appears twice in the fetched data
    df = df.drop_duplicates(subset="khipu_id", keep="first")
    df["kfg_any"] = (df[["kfg_pp", "kfg_cp", "kfg_ip",
                          "kfg_sp", "kfg_gg", "kfg_is", "kfg_psn"]] > 0).any(axis=1)
    print(f"  Parsed {len(df)} khipus from KFG fieldmarks page.")
    print(f"  KFG 'has summation': {df['kfg_any'].sum()} / {len(df)} "
          f"({df['kfg_any'].mean()*100:.1f}%)")
    return df


# ---------------------------------------------------------------------------
# Step 2: Run KCAT detector on the same set
# ---------------------------------------------------------------------------

def run_kcat_detector(kfg_ids: list[str]) -> pd.DataFrame:
    """Run KFGSummationDetector on every khipu in kfg_ids.

    Returns a DataFrame with columns:
        khipu_id, kcat_any, kcat_pp, kcat_cp, kcat_ip, kcat_sp,
        kcat_gg, kcat_is, kcat_psn, kcat_adg
    """
    detector = KFGSummationDetector(str(DB))

    # All IDs in KCAT database
    conn = sqlite3.connect(str(DB))
    all_kcat_ids = set(
        pd.read_sql("SELECT kfg_id FROM khipu_metadata", conn)["kfg_id"].tolist()
    )
    conn.close()

    rows = []
    pattern_keys = [
        "pendant_pendant_sum", "colored_pendant_sum", "indexed_pendant_sum",
        "subsidiary_pendant_sum", "group_group_sum", "indexed_subsidiary_sum",
        "pendant_sub_neighbor", "ascher_decreasing_group"
    ]
    short_keys = ["kcat_pp", "kcat_cp", "kcat_ip", "kcat_sp",
                  "kcat_gg", "kcat_is", "kcat_psn", "kcat_adg"]

    for kid in kfg_ids:
        if kid not in all_kcat_ids:
            # KFG has this khipu but KCAT doesn't - record as missing
            rows.append([kid, None] + [None] * 8)
            continue
        try:
            summary = detector.summarize(kid)
            stats = summary.get("pattern_stats", {})
            has_any = summary.get("has_summation", False)
            counts = [1 if stats.get(pk, {}).get("matches", 0) > 0 else 0 for pk in pattern_keys]
            rows.append([kid, has_any] + counts)
        except Exception as exc:
            print(f"  WARNING: detector failed on {kid}: {exc}")
            rows.append([kid, None] + [None] * 8)

    df = pd.DataFrame(rows, columns=["khipu_id", "kcat_any"] + short_keys)
    print(f"\nKCAT detector run on {len(kfg_ids)} khipus.")
    valid = df["kcat_any"].notna()
    n_valid = valid.sum()
    n_any = df.loc[valid, "kcat_any"].sum()
    print(f"  KCAT 'has summation' (KFG-overlap set): {n_any} / {n_valid} "
          f"({n_any/n_valid*100:.1f}%)")
    return df


# ---------------------------------------------------------------------------
# Step 3: Merge and analyse
# ---------------------------------------------------------------------------

def reconcile(kfg: pd.DataFrame, kcat: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(kfg, kcat, on="khipu_id", how="outer", indicator=True)

    # Classification
    def classify(row):
        if pd.isna(row["kcat_any"]):
            return "kfg_only"        # in KFG, not in KCAT DB
        if pd.isna(row["kfg_any"]):
            return "kcat_only"       # in KCAT DB, not on KFG fieldmarks page
        if row["kfg_any"] and row["kcat_any"]:
            return "agree_positive"  # both say summation
        if not row["kfg_any"] and not row["kcat_any"]:
            return "agree_negative"  # both say no summation
        if not row["kfg_any"] and row["kcat_any"]:
            return "kcat_only_pos"   # KCAT positive, KFG negative
        if row["kfg_any"] and not row["kcat_any"]:
            return "kfg_only_pos"    # KFG positive, KCAT negative
        return "unknown"

    merged["verdict"] = merged.apply(classify, axis=1)
    return merged


def print_summary(merged: pd.DataFrame):
    counts = merged["verdict"].value_counts()
    total = len(merged)

    in_both  = merged["verdict"].isin(
        ["agree_positive", "agree_negative", "kcat_only_pos", "kfg_only_pos"]
    ).sum()

    agree_p  = counts.get("agree_positive", 0)
    agree_n  = counts.get("agree_negative", 0)
    kcat_p   = counts.get("kcat_only_pos",  0)
    kfg_p    = counts.get("kfg_only_pos",   0)
    kfg_o    = counts.get("kfg_only",       0)
    kcat_o   = counts.get("kcat_only",      0)

    print("\n" + "="*60)
    print("RECONCILIATION SUMMARY")
    print("="*60)
    print(f"Total rows in merged table         : {total}")
    print(f"  Khipus in both KFG + KCAT        : {in_both}")
    print(f"  KFG page only (not in KCAT DB)   : {kfg_o}")
    print(f"  KCAT DB only (not on KFG page)   : {kcat_o}")
    print()
    print(f"Agreement on summation presence:")
    print(f"  Both positive (agree +)          : {agree_p}")
    print(f"  Both negative (agree -)          : {agree_n}")
    print(f"  KCAT positive, KFG negative      : {kcat_p}")
    print(f"  KFG positive, KCAT negative      : {kfg_p}")
    if in_both:
        agreement_rate = (agree_p + agree_n) / in_both * 100
        print(f"\n  Agreement rate (in_both set)     : {agreement_rate:.1f}%")

    # Per-pattern type agreement (for the 7 shared patterns)
    shared_pairs = [
        ("kfg_pp",  "kcat_pp",  "pendant_pendant"),
        ("kfg_cp",  "kcat_cp",  "colored_pendant"),
        ("kfg_ip",  "kcat_ip",  "indexed_pendant"),
        ("kfg_sp",  "kcat_sp",  "subsidiary_pendant"),
        ("kfg_gg",  "kcat_gg",  "group_group"),
        ("kfg_is",  "kcat_is",  "indexed_subsidiary"),
        ("kfg_psn", "kcat_psn", "pendant_sub_neighbor"),
    ]
    print("\nPer-pattern agreement (KFG col vs KCAT col):")
    print(f"  {'Pattern':<26} {'KFG+':>6} {'KCAT+':>6} {'Agree+':>7} {'Agree-':>7} {'Agr%':>6}")
    print("  " + "-"*62)
    for kfg_col, kcat_col, label in shared_pairs:
        sub = merged.dropna(subset=[kfg_col, kcat_col])
        if sub.empty:
            continue
        kfg_pos  = (sub[kfg_col]  > 0).sum()
        kcat_pos = (sub[kcat_col] > 0).sum()
        ap = ((sub[kfg_col] > 0) & (sub[kcat_col] > 0)).sum()
        an = ((sub[kfg_col] == 0) & (sub[kcat_col] == 0)).sum()
        agr = (ap + an) / len(sub) * 100
        print(f"  {label:<26} {kfg_pos:>6} {kcat_pos:>6} {ap:>7} {an:>7} {agr:>5.1f}%")

    print("="*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    kfg_df  = fetch_kfg_table()
    kcat_df = run_kcat_detector(kfg_df["khipu_id"].tolist())
    merged  = reconcile(kfg_df, kcat_df)
    print_summary(merged)

    merged.to_csv(OUT_CSV, index=False)
    print(f"\nFull per-khipu comparison saved to:\n  {OUT_CSV}")
