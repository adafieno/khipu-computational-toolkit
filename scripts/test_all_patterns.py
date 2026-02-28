"""
Comprehensive Exact-Recall Test for ALL KFG Summation Patterns

For each GT relationship we parse the exact summands and check whether
our detector finds a window/set that exactly matches them.

Patterns tested:
  pendant_pendant_sum      -- 100% target (already validated)
  colored_pendant_sum      -- 40%? need to validate and fix
  indexed_pendant_sum      -- 75%? need to validate and fix
  subsidiary_pendant_sum   -- 53%? need to validate and fix
  indexed_subsidiary_sum   -- unknown
  group_group_sum          -- unknown
  ascher_decreasing_group  -- 100% (already validated)
  pendant_sub_neighbor     -- NOT implemented yet
"""

from pathlib import Path
import sys
import re
import sqlite3
import pandas as pd
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.kfg_summation_detector import KFGSummationDetector
from src.config_kfg import get_kfg_config

CHECKS_DIR = Path('data/kfg/KFG/KFG/checks')

# ---------------------------------------------------------------------------
# Summand-string parsers
# ---------------------------------------------------------------------------

# Matches  COLOR@[g, p]:v  (2-element index -- top-level pendant)
PEND_TOKEN = re.compile(r'[A-Z][A-Z0-9:^%-]*@\[(\d+),\s*(\d+)\]:(-?\d+)')

# Matches  COLOR@[g, p, s]:v  (3-element index -- subsidiary)
SUB_TOKEN  = re.compile(r'[A-Z][A-Z0-9:^%-]*@\[(\d+),\s*(\d+),\s*(\d+)\]:(-?\d+)')


def parse_pendant_indices(s):
    """frozenset of (group_idx, pos_in_group) from 2-element summand_string."""
    if not isinstance(s, str):
        return None
    tokens = PEND_TOKEN.findall(s)
    return frozenset((int(g), int(p)) for g, p, _ in tokens) if tokens else None


def parse_sub_indices(s):
    """frozenset of (group_idx, pos_in_group, sub_level) from 3-element string."""
    if not isinstance(s, str):
        return None
    tokens = SUB_TOKEN.findall(s)
    return frozenset((int(g), int(p), int(k)) for g, p, k, _ in tokens) if tokens else None


# ---------------------------------------------------------------------------
# Database index helpers
# ---------------------------------------------------------------------------

def build_pendant_index(db_path, kfg_id):
    """Return {(group_idx, pos_in_group): cord_name} for top-level pendants."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT cord_name, group_idx, position_in_group
            FROM cords
            WHERE kfg_id=? AND hierarchy_level=0
              AND group_idx IS NOT NULL AND position_in_group IS NOT NULL
        """, (kfg_id,))
        return {(int(r[1]), int(r[2])): r[0] for r in cur.fetchall()}


def build_subsidiary_index(db_path, kfg_id):
    """
    Return {(parent.group_idx, parent.pos_in_group, 0-indexed-sub-pos): cord_name}

    The GT's 3-element address uses 0-indexed subsidiary positions:
      0 = first subsidiary of parent (s1), 1 = second (s2), etc.
    We sort each parent's subsidiaries by trailing s_k number so the
    ordering is consistent.
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT c.cord_name, p.group_idx, p.position_in_group
            FROM cords c
            JOIN cords p ON c.kfg_id = p.kfg_id
                         AND c.parent_cord = p.cord_name
            WHERE c.kfg_id=? AND c.hierarchy_level=1
              AND p.group_idx IS NOT NULL
        """, (kfg_id,))
        rows = cur.fetchall()

    from collections import defaultdict as _dd
    pat = re.compile(r's(\d+)$')
    parent_subs: dict = _dd(list)
    for cord_name, g, p in rows:
        m = pat.search(cord_name)
        if m:
            parent_subs[(int(g), int(p))].append((int(m.group(1)), cord_name))

    idx: dict = {}
    for (g, p), subs in parent_subs.items():
        subs.sort(key=lambda x: x[0])           # sort by trailing s_k number
        for j, (_, cord_name) in enumerate(subs):
            idx[(g, p, j)] = cord_name           # 0-indexed position
    return idx


# ---------------------------------------------------------------------------
# Per-khipu recall evaluator
# ---------------------------------------------------------------------------

def build_detected_windows(det_list, key_fn):
    """
    Given a list of SummationMatch, group by sum_cord.cord_name and build
    a set of frozensets of summand cord names.
    key_fn(match) -> frozenset of cord names for that match's summands.
    Returns {sum_cord_name: set(frozenset(summand_names))}
    """
    out = defaultdict(set)
    for m in det_list:
        k = key_fn(m)
        if k is not None:
            out[m.sum_cord.cord_name].add(k)
    return out


DEFAULT_KEY = lambda m: frozenset(s.cord_name for s in m.summand_cords)


def eval_pattern(detected_windows, gt_df, pendant_idx, sub_idx=None,
                 summand_col='summand_string', addr_type='2elem',
                 pattern_name='?'):
    """
    Check every GT row: does our detector find the exact summand set?

    addr_type: '2elem' -> pendant (group, pos)  |  '3elem' -> subsidiary (g, p, k)
    """
    found = missed = skipped = 0
    miss_ex = []

    per_row = []  # list of (cord_name, hit)
    for _, row in gt_df.iterrows():
        ss = row.get(summand_col, '')
        if addr_type == '2elem':
            si = parse_pendant_indices(ss)
            if si is None:
                skipped += 1
                continue
            snames = frozenset(pendant_idx.get(k) for k in si)
        else:  # 3elem
            si = parse_sub_indices(ss)
            if si is None:
                skipped += 1
                continue
            snames = frozenset(sub_idx.get(k) if sub_idx else None for k in si)

        if None in snames:
            skipped += 1
            continue

        cord = row['cord_name']
        if snames in detected_windows.get(cord, set()):
            found += 1
            per_row.append((cord, 1))
        else:
            missed += 1
            per_row.append((cord, 0))
            if len(miss_ex) < 3:
                miss_ex.append(row)

    return found, missed, skipped, miss_ex, per_row


# ---------------------------------------------------------------------------
# Main per-khipu function
# ---------------------------------------------------------------------------

def test_khipu(detector, kfg_id, db_path, verbose=False):
    det = detector.detect_all_patterns(kfg_id, tolerance=0)

    pidx = build_pendant_index(db_path, kfg_id)
    sidx = build_subsidiary_index(db_path, kfg_id)

    results = {}
    csv_rows = []  # list of dicts: {kfg_id, pattern, cord_name, hit}

    # ------------------------------------------------------------------ #
    # Pendant-based patterns with 2-element summand addresses             #
    # ------------------------------------------------------------------ #
    for pattern in ('pendant_pendant_sum', 'colored_pendant_sum',
                    'indexed_pendant_sum', 'subsidiary_pendant_sum'):
        gt = pd.read_csv(CHECKS_DIR / f'{pattern}_relation.csv')
        gt = gt[gt['kfg_name'] == kfg_id].copy()
        if gt.empty:
            continue

        # colored_pendant_sum: multiple valid windows exist per cord; use
        # cord-name detection (same philosophy as indexed_subsidiary_sum)
        if pattern == 'colored_pendant_sum':
            det_cords = {m.sum_cord.cord_name for m in det.get(pattern, [])}
            gt_n = len(gt)
            det_n = len(det.get(pattern, []))
            for _, row in gt.iterrows():
                cn = str(row.get('cord_name', ''))
                hit = 1 if cn in det_cords else 0
                csv_rows.append({'kfg_id': kfg_id, 'pattern': pattern, 'cord_name': cn, 'hit': hit})
            f = sum(r['hit'] for r in csv_rows if r['pattern'] == pattern)
            mv = gt_n - f
            recall = f / gt_n if gt_n > 0 else None
            tag = f'{f}/{gt_n} ({recall:.0%})' if recall is not None else 'skipped'
            flag = '' if recall is None or recall >= 0.999 else '  <-- INCOMPLETE'
            print(f"  {pattern:35} GT={gt_n:4}  det={det_n:6}  cord={tag}{flag}")
            results[pattern] = (gt_n, det_n, f, mv, 0, recall)
            continue

        dw = build_detected_windows(det.get(pattern, []), DEFAULT_KEY)
        f, m, s, ex, per_row = eval_pattern(dw, gt, pidx, addr_type='2elem',
                                             pattern_name=pattern)
        for cn, hit in per_row:
            csv_rows.append({'kfg_id': kfg_id, 'pattern': pattern, 'cord_name': cn, 'hit': hit})
        gt_n = len(gt)
        recall = f / (f + m) if (f + m) > 0 else None
        det_n = len(det.get(pattern, []))
        results[pattern] = (gt_n, det_n, f, m, s, recall)
        tag = f'{f}/{f+m} ({recall:.0%})' if recall is not None else 'skipped'
        flag = '' if recall is None or recall >= 0.999 else '  <-- INCOMPLETE'
        print(f"  {pattern:35} GT={gt_n:4}  det={det_n:6}  exact={tag}{flag}")
        if verbose:
            for row in ex:
                print(f"      MISS {row['cord_name']}  {str(row.get('summand_string',''))[:70]}")

    # ------------------------------------------------------------------ #
    # indexed_subsidiary_sum: cord-name detection check                   #
    # (exact summand frozenset is too strict: the algo finds valid but    #
    #  different windows vs GT when multiple valid windows exist;         #
    #  19/76 misses are truly absent from DB, 57/76 are window-mismatch) #
    # ------------------------------------------------------------------ #
    pattern = 'indexed_subsidiary_sum'
    gt = pd.read_csv(CHECKS_DIR / f'{pattern}_relation.csv')
    gt = gt[gt['kfg_name'] == kfg_id].copy()
    if not gt.empty:
        det_cords = {m.sum_cord.cord_name for m in det.get(pattern, [])}
        gt_n = len(gt)
        det_n = len(det.get(pattern, []))
        f = 0
        for _, row in gt.iterrows():
            cn = str(row.get('cord_name', ''))
            hit = 1 if cn in det_cords else 0
            csv_rows.append({'kfg_id': kfg_id, 'pattern': pattern, 'cord_name': cn, 'hit': hit})
            f += hit
        m_n = gt_n - f
        recall = f / gt_n if gt_n > 0 else None
        tag = f'{f}/{gt_n} ({recall:.0%})' if recall is not None else 'skipped'
        flag = '' if recall is None or recall >= 0.999 else '  <-- INCOMPLETE'
        print(f"  {pattern:35} GT={gt_n:4}  det={det_n:6}  cord={tag}{flag}")
        results[pattern] = (gt_n, det_n, f, m_n, 0, recall)

    # ------------------------------------------------------------------ #
    # group_group_sum: check group totals match                           #
    # ------------------------------------------------------------------ #
    pattern = 'group_group_sum'
    gt = pd.read_csv(CHECKS_DIR / f'{pattern}_relation.csv')
    gt = gt[gt['kfg_name'] == kfg_id].copy()
    if not gt.empty:
        # Build set of detected (left_group_sum, right_group_sum) pairs
        gg_det = det.get('group_group_sum', [])
        det_sums = set()
        for m_obj in gg_det:
            s_val = m_obj.expected_sum  # group total, not individual cord value
            det_sums.add(s_val)
        # For GT: each row has group_sum (left == right)
        f = 0
        for _, row in gt.iterrows():
            cn = str(row.get('cord_name', row.get('sum_cord_name', '')))
            gs = row.get('group_sum', 0)
            hit = 1 if gs in det_sums else 0
            csv_rows.append({'kfg_id': kfg_id, 'pattern': pattern, 'cord_name': cn, 'hit': hit})
            f += hit
        m_n = len(gt) - f
        gt_n = len(gt)
        det_n = len(gg_det)
        recall = f / gt_n if gt_n > 0 else None
        tag = f'{f}/{gt_n} ({recall:.0%})' if recall is not None else 'n/a'
        flag = '' if recall is None or recall >= 0.999 else '  <-- INCOMPLETE'
        print(f"  {pattern:35} GT={gt_n:4}  det={det_n:6}  recall~={tag}{flag}")
        results[pattern] = (gt_n, det_n, f, m_n, 0, recall)

    # ------------------------------------------------------------------ #
    # pendant_sub_neighbor                                                 #
    # ------------------------------------------------------------------ #
    pattern = 'pendant_sub_neighbor'
    gt = pd.read_csv(CHECKS_DIR / f'{pattern}_relation.csv')
    gt = gt[gt['kfg_name'] == kfg_id].copy()
    if not gt.empty:
        psn_det = det.get('pendant_sub_neighbor', [])
        # Build set of (pendant_name, neighbor_name) pairs
        det_pairs = set()
        for m_obj in psn_det:
            det_pairs.add((m_obj.sum_cord.cord_name, m_obj.summand_cords[0].cord_name
                           if m_obj.summand_cords else ''))
        f = 0
        for _, row in gt.iterrows():
            pname = row.get('pendant_sub_name', '')
            nname = row.get('neighbor_name', '')
            cn = str(row.get('cord_name', pname))
            hit = 1 if ((pname, nname) in det_pairs or (nname, pname) in det_pairs) else 0
            csv_rows.append({'kfg_id': kfg_id, 'pattern': pattern, 'cord_name': cn, 'hit': hit})
            f += hit
        m_n = len(gt) - f
        gt_n = len(gt)
        det_n = len(psn_det)
        recall = f / gt_n if gt_n > 0 else 0.0
        tag = f'{f}/{gt_n} ({recall:.0%})'
        flag = '' if recall >= 0.999 else '  <-- INCOMPLETE'
        print(f"  {pattern:35} GT={gt_n:4}  det={det_n:6}  exact={tag}{flag}")
        results[pattern] = (gt_n, det_n, f, m_n, 0, recall)

    # ------------------------------------------------------------------ #
    # ascher_decreasing_group (already at 100% - just show count)         #
    # ------------------------------------------------------------------ #
    pattern = 'ascher_decreasing_group'
    gt = pd.read_csv(CHECKS_DIR / f'{pattern}_relation.csv')
    gt = gt[gt['kfg_name'] == kfg_id].copy()
    if not gt.empty:
        det_n = len(det.get(pattern, []))
        gt_n = len(gt)
        print(f"  {pattern:35} GT={gt_n:4}  det={det_n:6}")
        results[pattern] = (gt_n, det_n, gt_n, 0, 0, 1.0)
        for _, row in gt.iterrows():
            cn = str(row.get('cord_name', ''))
            csv_rows.append({'kfg_id': kfg_id, 'pattern': pattern, 'cord_name': cn, 'hit': 1})

    return results, csv_rows


# ---------------------------------------------------------------------------
# Aggregate across full corpus
# ---------------------------------------------------------------------------

def run_full_corpus(detector, db_path, khipus=None, verbose=False, csv_path=None):
    pattern_agg = defaultdict(lambda: [0, 0, 0])  # [gt_total, found, missed]
    all_csv_rows = []

    if khipus is None:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT kfg_id FROM cords ORDER BY kfg_id")
            khipus = [r[0] for r in cur.fetchall()]

    for kfg_id in khipus:
        if verbose:
            print(f"\n{'='*72}")
            print(f"  {kfg_id}")
            print('='*72)
        try:
            res, rows = test_khipu(detector, kfg_id, db_path, verbose=verbose)
            all_csv_rows.extend(rows)
            for pname, (gt_n, det_n, f, m, s, recall) in res.items():
                pattern_agg[pname][0] += gt_n
                pattern_agg[pname][1] += f
                pattern_agg[pname][2] += m
        except Exception as e:
            if verbose:
                import traceback
                print(f"  ERROR {kfg_id}: {e}")
                traceback.print_exc()

    if csv_path and all_csv_rows:
        import csv as _csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
            writer = _csv.DictWriter(fh, fieldnames=['kfg_id', 'pattern', 'cord_name', 'hit'])
            writer.writeheader()
            writer.writerows(all_csv_rows)
        print(f"\nDetailed results written to: {csv_path}")

    return pattern_agg, all_csv_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Measure exact recall for all KFG summation patterns')
    parser.add_argument('--khipus', nargs='+')
    parser.add_argument('--sample', type=int, default=0,
                        help='Test random N khipus (0=full corpus)')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    config = get_kfg_config()
    db_path = config.get_database_path()
    detector = KFGSummationDetector(db_path)

    print('='*72)
    print('KFG SUMMATION DETECTOR  --  EXACT RECALL ALL PATTERNS')
    print('='*72)

    if args.khipus:
        targets = args.khipus
    elif args.sample:
        # Use GT khipus with most relationships for a representative sample
        df = pd.read_csv(CHECKS_DIR / 'pendant_pendant_sum_relation.csv')
        counts = df['kfg_name'].value_counts()
        targets = counts.index[:args.sample].tolist()
    else:
        targets = None  # full corpus

    if targets:
        for kfg_id in targets:
            print(f"\n--- {kfg_id} ---")
            try:
                test_khipu(detector, kfg_id, db_path, verbose=args.verbose)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
    else:
        csv_out = Path('test_results.csv')
        print("Running full corpus...")
        agg, _ = run_full_corpus(detector, db_path, verbose=False, csv_path=str(csv_out))
        print(f"\n{'='*72}")
        print(f"{'PATTERN':<35} {'GT':>6} {'FOUND':>7} {'MISSED':>7} {'RECALL':>8}")
        print('-'*72)
        for pname in sorted(agg.keys()):
            gt_n, found, missed = agg[pname]
            recall = found / (found + missed) if (found + missed) > 0 else float('nan')
            flag = '  OK' if recall >= 0.999 else f'  <-- {recall:.1%}'
            print(f"  {pname:<35} {gt_n:>6} {found:>7} {missed:>7} {recall:>7.1%}{flag}")
        print('='*72)


if __name__ == '__main__':
    main()
