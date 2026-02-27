"""
Test KFG Summation Detector -- Precision / Recall against Ground Truth

For each KFG ground-truth row we parse the exact summands from summand_string,
then ask: does our detector find AT LEAST ONE match for that sum cord whose
summand set exactly matches the ground-truth summands?

Recall    = fraction of GT relationships our detector finds exactly.
"""

from pathlib import Path
import sys
import re
import pandas as pd
import sqlite3
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.kfg_summation_detector import KFGSummationDetector
from src.config_kfg import get_kfg_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUMMAND_TOKEN = re.compile(
    r'[A-Z][A-Z0-9:]*@\[(\d+),\s*(\d+)\]:(-?\d+)'
)


def parse_summand_indices(summand_string):
    """Return frozenset of (group_idx, pos_in_group) from summand_string."""
    if not isinstance(summand_string, str) or not summand_string.strip():
        return None
    tokens = SUMMAND_TOKEN.findall(summand_string)
    if not tokens:
        return None
    return frozenset((int(g), int(p)) for g, p, _ in tokens)


def load_gt(checks_dir, pattern, kfg_id=None):
    f = Path(checks_dir) / f"{pattern}_relation.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    if kfg_id and 'kfg_name' in df.columns:
        return df[df['kfg_name'] == kfg_id].copy()
    return df


def cord_index_map(db_path, kfg_id):
    """Return {(group_idx, pos_in_group): cord_name} for top-level pendants."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT cord_name, group_idx, position_in_group
            FROM cords
            WHERE kfg_id=? AND hierarchy_level=0 AND group_idx IS NOT NULL
        """, (kfg_id,))
        return {(int(r[1]), int(r[2])): r[0] for r in cur.fetchall()}


# ---------------------------------------------------------------------------
# Per-khipu test
# ---------------------------------------------------------------------------

CORD_BASED_PATTERNS = [
    'pendant_pendant_sum',
    'colored_pendant_sum',
    'indexed_pendant_sum',
    'subsidiary_pendant_sum',
    'indexed_subsidiary_sum',
]


def test_khipu(detector, kfg_id, checks_dir, db_path, verbose=False):
    print(f"\n{'='*72}")
    print(f"  {kfg_id}")
    print('='*72)

    # Run detector
    all_det = detector.detect_all_patterns(kfg_id, tolerance=1)

    # Build lookup: sum_cord_name -> set of frozensets(summand_cord_names)
    idx_lk = cord_index_map(db_path, kfg_id)
    pp_windows = defaultdict(set)
    for m in all_det.get('pendant_pendant_sum', []):
        sname = frozenset(s.cord_name for s in m.summand_cords)
        pp_windows[m.sum_cord.cord_name].add(sname)

    grand_found = grand_missed = 0

    for pattern in CORD_BASED_PATTERNS:
        gt = load_gt(checks_dir, pattern, kfg_id)
        our = all_det.get(pattern, [])
        gn = len(gt)
        on = len(our)

        if gn == 0 and on == 0:
            continue

        if pattern == 'pendant_pendant_sum' and gn > 0:
            # Exact recall: check each GT row
            found = missed = 0
            miss_ex = []
            for _, row in gt.iterrows():
                si = parse_summand_indices(row.get('summand_string', ''))
                if si is None:
                    continue
                snames = frozenset(idx_lk.get(k) for k in si)
                if None in snames:
                    continue  # unresolved cord index
                if snames in pp_windows.get(row['cord_name'], set()):
                    found += 1
                else:
                    missed += 1
                    if verbose and len(miss_ex) < 3:
                        miss_ex.append(row)
            grand_found += found
            grand_missed += missed
            r = found / gn if gn else 0
            print(f"  {pattern:35} GT={gn:4}  ours={on:6}  "
                  f"exact-recall={found}/{gn} ({r:.0%})")
            if verbose:
                for row in miss_ex:
                    print(f"      MISS sum={row['cord_name']}"
                          f"  ss={row.get('summand_string','')[:60]}")
        else:
            diff = on - gn
            sign = '+' if diff > 0 else ''
            status = 'OK' if diff == 0 else f"diff={sign}{diff}"
            print(f"  {pattern:35} GT={gn:4}  ours={on:6}  {status}")

    total = grand_found + grand_missed
    pct = grand_found / total if total else 0
    print(f"\n  pendant_pendant_sum exact recall: {grand_found}/{total} ({pct:.0%})")

    for p in ('group_group_sum', 'ascher_decreasing_group'):
        gt = load_gt(checks_dir, p, kfg_id)
        our = all_det.get(p, [])
        if len(gt) or len(our):
            print(f"  {p:35} GT={len(gt):4}  ours={len(our):6}")

    return grand_found, grand_missed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Measure recall/precision of KFG summation detector')
    parser.add_argument('--khipus', nargs='+', help='KFG IDs to test')
    parser.add_argument('--sample', type=int, default=5,
                        help='Random sample size when --khipus not given')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    config = get_kfg_config()
    db_path = config.get_database_path()
    checks_dir = Path('data/kfg/KFG/KFG/checks')

    print('='*72)
    print('KFG SUMMATION DETECTOR  --  PRECISION / RECALL')
    print('='*72)
    print(f"DB:     {db_path}")
    print(f"Checks: {checks_dir}")

    detector = KFGSummationDetector(db_path)

    if args.khipus:
        targets = args.khipus
    else:
        gt = load_gt(checks_dir, 'pendant_pendant_sum')
        if not gt.empty:
            counts = gt['kfg_name'].value_counts()
            targets = counts.index[:args.sample].tolist()
        else:
            targets = ['KH0001', 'CM009']

    agg_found = agg_missed = 0
    for kfg_id in targets:
        try:
            f, m = test_khipu(detector, kfg_id, checks_dir, db_path,
                              verbose=args.verbose)
            agg_found += f
            agg_missed += m
        except Exception as e:
            import traceback
            print(f"\n  ERROR {kfg_id}: {e}")
            traceback.print_exc()

    total = agg_found + agg_missed
    if total:
        print(f"\n{'='*72}")
        print(f"AGGREGATE pendant_pendant_sum recall: {agg_found}/{total} "
              f"({agg_found/total:.1%})")
        print('='*72)


if __name__ == '__main__':
    main()
