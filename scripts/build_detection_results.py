"""
Build Pattern Detection Results
================================
Runs the KFG Summation Detector on every khipu in the database and stores
per-khipu per-pattern statistics in the ``pattern_results`` table.

This makes the explorer and all analytics fully self-contained: no external
checks/*.csv ground-truth files are needed at runtime.  The detector is the
algorithmic implementation in src/analysis/kfg_summation_detector.py.

Schema written to the database
-------------------------------
    pattern_results (
        kfg_id            TEXT,
        pattern           TEXT,
        num_instances     INTEGER,   -- total summation matches found
        num_left_sums     INTEGER,   -- matches where summands are left of sum cord
        num_right_sums    INTEGER,   -- matches where summands are right of sum cord
        mean_sum          REAL,      -- mean value of sum cords
        num_dual_sums     INTEGER,   -- sum cords appearing in >=2 relationships
        num_multisummands INTEGER,   -- matches with 3+ distinct summands
        PRIMARY KEY (kfg_id, pattern)
    )

Usage
-----
    cd c:/code/khipu-computational-toolkit
    python scripts/build_detection_results.py              # full corpus (~5 min)
    python scripts/build_detection_results.py --limit 20   # quick test
    python scripts/build_detection_results.py --db path/to/other.db
    python scripts/build_detection_results.py --verbose
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.kfg_summation_detector import KFGSummationDetector

ROOT = Path(__file__).parent.parent
DB_DEFAULT = ROOT / "data" / "kfg" / "khipu_database.db"

PATTERNS = [
    "pendant_pendant_sum",
    "colored_pendant_sum",
    "indexed_pendant_sum",
    "subsidiary_pendant_sum",
    "indexed_subsidiary_sum",
    "group_group_sum",
    "group_sum_bands",
    "ascher_decreasing_group",
    "pendant_sub_neighbor",
]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS pattern_results (
    kfg_id            TEXT    NOT NULL,
    pattern           TEXT    NOT NULL,
    num_instances     INTEGER NOT NULL DEFAULT 0,
    num_left_sums     INTEGER NOT NULL DEFAULT 0,
    num_right_sums    INTEGER NOT NULL DEFAULT 0,
    mean_sum          REAL,
    num_dual_sums     INTEGER NOT NULL DEFAULT 0,
    num_multisummands INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (kfg_id, pattern)
)
"""

_CREATE_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_pr_kfg ON pattern_results(kfg_id)"
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _flat_order(cord) -> float:
    """Flat group-ordered sort key used to determine handedness."""
    g = cord.group_idx if cord.group_idx is not None else 0
    p = cord.position_in_group if cord.position_in_group is not None else 0
    return float(g) * 1_000_000 + float(p)


def _aggregate(matches: list) -> dict[str, Any]:
    """Summarise a list of SummationMatch objects into per-pattern stats."""
    if not matches:
        return {}

    sum_cord_hits: dict[int, int] = defaultdict(int)
    left = right = n_multi = 0
    total_val = 0.0

    for m in matches:
        sc = m.sum_cord
        sum_cord_hits[sc.cord_id] += 1
        total_val += float(m.expected_sum or 0)

        # Handedness: use pre-computed value from detector if available,
        # otherwise fall back to flat-order comparison.
        if m.handedness is not None and m.handedness != 0:
            if m.handedness < 0:
                left += 1
            else:
                right += 1
        else:
            sc_ord = _flat_order(sc)
            s_ords = [_flat_order(s) for s in m.summand_cords]
            if s_ords:
                if max(s_ords) < sc_ord:
                    left += 1
                elif min(s_ords) > sc_ord:
                    right += 1

        # Multi-summand: 3+ distinct non-zero summands in this relationship
        n_nonzero = sum(1 for s in m.summand_cords if (s.value or 0) != 0)
        if n_nonzero >= 3:
            n_multi += 1

    n = len(matches)
    # Dual: sum cords that appear as the sum cord in 2+ distinct relationships
    n_dual = sum(1 for cnt in sum_cord_hits.values() if cnt >= 2)

    return {
        "num_instances":     n,
        "num_left_sums":     left,
        "num_right_sums":    right,
        "mean_sum":          round(total_val / n, 4) if n else None,
        "num_dual_sums":     n_dual,
        "num_multisummands": n_multi,
    }


# ── Core function ─────────────────────────────────────────────────────────────

def build_detection_results(
    db_path: Path,
    limit: int = 0,
    verbose: bool = False,
) -> None:
    """
    ----------
    db_path : Path
        SQLite database built by build_kfg_database.py.
    limit : int
        If > 0, process only the first *limit* khipus (for testing).
    verbose : bool
        Print per-khipu progress.
    """
    if not db_path.exists():
        print(f"ERROR: database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    detector = KFGSummationDetector(str(db_path))

    conn = sqlite3.connect(str(db_path))
    conn.execute(_CREATE_TABLE)
    conn.execute(_CREATE_INDEX)
    # Full rebuild on every run so results stay in sync with any detector changes
    conn.execute("DELETE FROM pattern_results")
    conn.commit()

    cur = conn.cursor()
    cur.execute("SELECT kfg_id FROM khipu_metadata ORDER BY kfg_id")
    all_ids = [r[0] for r in cur.fetchall()]
    if limit:
        all_ids = all_ids[:limit]

    total  = len(all_ids)
    errors = 0

    print(f"Detecting summation patterns across {total} khipus…")

    for i, kfg_id in enumerate(all_ids, 1):
        if verbose or i % 50 == 0:
            print(f"  [{i:>3}/{total}] {kfg_id}")

        try:
            results = detector.detect_all_patterns(kfg_id)
        except Exception as exc:
            if verbose:
                print(f"    ERROR {kfg_id}: {exc}")
            errors += 1
            continue

        rows = []
        for pattern in PATTERNS:
            stats = _aggregate(results.get(pattern) or [])
            if not stats:
                continue
            rows.append((
                kfg_id,
                pattern,
                stats["num_instances"],
                stats["num_left_sums"],
                stats["num_right_sums"],
                stats["mean_sum"],
                stats["num_dual_sums"],
                stats["num_multisummands"],
            ))

        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO pattern_results VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()

    conn.close()

    done = total - errors
    print(f"\nDone — {done}/{total} khipus processed, {errors} errors.")
    print(f"pattern_results table written to {db_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build pattern_results table in the KFG database",
    )
    parser.add_argument(
        "--db", type=Path, default=DB_DEFAULT,
        help="Path to the SQLite database (default: data/kfg/khipu_database.db)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only the first N khipus (0 = full corpus)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    build_detection_results(args.db, args.limit, args.verbose)


if __name__ == "__main__":
    main()
