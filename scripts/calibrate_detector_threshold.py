"""
Calibration sweep: find the min_matches threshold + color normalization setting
that best reproduces KFG fieldmarks summation judgments (488/702 = 69.5%).

Two knobs:
  min_matches    : require total_matches >= N for has_summation = True
  normalize_color: strip color code suffixes before running colored_pendant_sum
                   e.g. "MB:W" -> "MB", "KB-DB" -> "KB"

For each combination, reports K-CAT positive rate, FP, FN, and agreement vs KFG.

Usage:
    python scripts/calibrate_detector_threshold.py
"""

import os, re, sqlite3, sys, tempfile, gc
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB   = ROOT / "data" / "kfg" / "khipu_database.db"
sys.path.insert(0, str(ROOT))
from src.analysis.kfg_summation_detector import KFGSummationDetector

# ── KFG ground truth ──────────────────────────────────────────────────────────
RECON   = pd.read_csv(ROOT / "data" / "processed" / "kfg_fieldmarks_reconciliation.csv")
KFG_IDS = RECON["khipu_id"].tolist()
KFG_POS = set(RECON.loc[RECON["kfg_any"] == True,  "khipu_id"])
KFG_NEG = set(RECON.loc[RECON["kfg_any"] == False, "khipu_id"])

_COLOR_LEAD = re.compile(r"^([A-Za-z]+)")

def normalize_color(raw):
    if not raw:
        return raw
    m = _COLOR_LEAD.match(raw)
    return m.group(1).upper() if m else raw.upper()


def build_temp_db(normalize: bool) -> str:
    """Copy the 702-khipu overlap into a temp DB, optionally normalizing colors."""
    src = sqlite3.connect(str(DB))
    f   = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp = f.name
    f.close()
    dst = sqlite3.connect(tmp)

    # khipu_metadata
    ph   = ",".join("?" * len(KFG_IDS))
    meta = src.execute(f"SELECT * FROM khipu_metadata WHERE kfg_id IN ({ph})", KFG_IDS).fetchall()
    cols = [d[1] for d in src.execute("PRAGMA table_info(khipu_metadata)").fetchall()]
    dst.execute(f"CREATE TABLE khipu_metadata ({','.join(cols)})")
    dst.executemany(f"INSERT INTO khipu_metadata VALUES ({','.join('?'*len(cols))})", meta)

    # cords
    cords = src.execute(f"SELECT * FROM cords WHERE kfg_id IN ({ph})", KFG_IDS).fetchall()
    ccols = [d[1] for d in src.execute("PRAGMA table_info(cords)").fetchall()]
    cidx  = ccols.index("color") if "color" in ccols else None
    if normalize and cidx is not None:
        cords = [
            tuple(normalize_color(v) if i == cidx else v for i, v in enumerate(row))
            for row in cords
        ]
    dst.execute(f"CREATE TABLE cords ({','.join(ccols)})")
    dst.executemany(f"INSERT INTO cords VALUES ({','.join('?'*len(ccols))})", cords)

    # optional extra tables
    for tbl in ["cord_color_mapping", "knot_clusters"]:
        try:
            rows = src.execute(f"SELECT * FROM {tbl}").fetchall()
            tc   = [d[1] for d in src.execute(f"PRAGMA table_info({tbl})").fetchall()]
            dst.execute(f"CREATE TABLE {tbl} ({','.join(tc)})")
            dst.executemany(f"INSERT INTO {tbl} VALUES ({','.join('?'*len(tc))})", rows)
        except Exception:
            pass

    dst.commit(); dst.close(); src.close()
    return tmp


def run_detector(db_path: str) -> dict:
    """Returns {khipu_id: total_matches} for all KFG_IDS."""
    det = KFGSummationDetector(db_path)
    out = {}
    for kid in KFG_IDS:
        try:
            out[kid] = det.summarize(kid).get("total_matches", 0)
        except Exception:
            out[kid] = 0
    return out


def evaluate(results: dict, min_matches: int) -> dict:
    pos  = {k for k, m in results.items() if m >= min_matches}
    neg  = set(results) - pos
    total = len(results)
    return {
        "n_pos":     len(pos),
        "rate":      len(pos) / total * 100,
        "fp":        len(pos & KFG_NEG),
        "fn":        len(neg & KFG_POS),
        "agreement": (len(pos & KFG_POS) + len(neg & KFG_NEG)) / total * 100,
    }


if __name__ == "__main__":
    kfg_rate = len(KFG_POS) / len(KFG_IDS) * 100
    print(f"KFG reference : {len(KFG_POS)}/{len(KFG_IDS)} = {kfg_rate:.1f}%\n")

    runs = {}
    for normalize, label in [(False, "no"), (True, "yes")]:
        print(f"Building DB (color_norm={label}) and running detector...")
        tmp = build_temp_db(normalize)
        runs[label] = run_detector(tmp)
        gc.collect()
        try:
            os.unlink(tmp)
        except PermissionError:
            pass  # Windows: let OS clean up on exit
        print(f"  Done.\n")

    hdr = (f"{'min_matches':>11}  {'color_norm':>10}  "
           f"{'K-CAT+':>6}  {'rate':>6}  {'FP':>5}  {'FN':>5}  {'Agr%':>6}")
    print(hdr)
    print("-" * len(hdr))

    best = None
    for min_m in range(1, 20):
        for label in ["no", "yes"]:
            ev   = evaluate(runs[label], min_m)
            diff = abs(ev["n_pos"] - len(KFG_POS))
            note = ""
            if best is None or diff < best[0]:
                best = (diff, min_m, label, ev)
                note = "  ← best so far"
            print(f"{min_m:>11}  {label:>10}  "
                  f"{ev['n_pos']:>6}  {ev['rate']:>5.1f}%  "
                  f"{ev['fp']:>5}  {ev['fn']:>5}  "
                  f"{ev['agreement']:>5.1f}%{note}")

    diff, best_m, best_norm, best_ev = best
    print(f"\n{'='*60}")
    print(f"Best match to KFG {kfg_rate:.1f}%:")
    print(f"  min_matches    = {best_m}")
    print(f"  color_norm     = {best_norm}")
    print(f"  K-CAT positive  : {best_ev['n_pos']}/702 ({best_ev['rate']:.1f}%)")
    print(f"  FP (K-CAT+/KFG-): {best_ev['fp']}")
    print(f"  FN (K-CAT-/KFG+): {best_ev['fn']}")
    print(f"  Agreement rate : {best_ev['agreement']:.1f}%")
    print(f"{'='*60}")

