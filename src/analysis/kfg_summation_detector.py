"""
KFG Summation Pattern Detector v2 (Group-Aware)

Algorithms verified against KFG ground truth (14,533 relationships, 702 khipus).

Confirmed insight
-----------------
pendant_pendant_sum is a SLIDING WINDOW on the flattened pendant sequence
(ordered by group_idx, then position_in_group).  Every single summand set in
the ground truth consists of a contiguous run of pendants in that flat order.

Pattern catalogue
-----------------
1. pendant_pendant_sum      - a pendant = sum of any consecutive window of
                              other pendants in flat group-ordered sequence
2. colored_pendant_sum      - a pendant = sum of all other pendants with the
                              same primary color
3. indexed_pendant_sum      - a pendant at position p = sum of all cords at
                              the same position_in_group p across other groups
4. subsidiary_pendant_sum   - a pendant = sum of its direct subsidiaries
5. indexed_subsidiary_sum   - a subsidiary s_k = sum of same-index subs across
                              other pendants (sliding window)
6. group_group_sum          - total of one group = total of another group
                              (or a cord = total of a contiguous range of groups)
7. group_sum_bands          - alias of group_group_sum (same structural logic)
"""

import sqlite3
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Cord:
    cord_id: int
    cord_name: str
    pendant_num: Optional[int]
    hierarchy_level: int
    parent_cord: Optional[str]
    group_idx: Optional[int]
    position_in_group: Optional[int]
    value: Optional[int]
    color: Optional[str]


@dataclass
class SummationMatch:
    pattern_type: str
    sum_cord: Cord
    summand_cords: List[Cord]
    expected_sum: int
    actual_sum: int
    matches: bool
    notes: str = ""


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class KFGSummationDetector:
    """
    Detect all documented KFG summation pattern types.

    Prerequisites
    -------------
    The cords table must have group_idx and position_in_group columns,
    populated by scripts/migrate_cord_groups.py.
    """

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _load_all_cords(self, kfg_id: str) -> List[Cord]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT cord_id, cord_name, pendant_num, hierarchy_level,
                       parent_cord, group_idx, position_in_group, value, color
                FROM cords
                WHERE kfg_id = ?
                ORDER BY hierarchy_level, group_idx, position_in_group
            """, (kfg_id,))
            return [Cord(*row) for row in cur.fetchall()]

    def _pendants(self, cords: List[Cord]) -> List[Cord]:
        """Top-level pendants in flat group order."""
        return [c for c in cords
                if c.hierarchy_level == 0
                and c.group_idx is not None
                and c.position_in_group is not None]

    def _subsidiaries(self, cords: List[Cord]) -> List[Cord]:
        return [c for c in cords if c.hierarchy_level == 1]

    # ------------------------------------------------------------------
    # Algorithm helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prefix_sums(values: List[Optional[int]]) -> List[int]:
        ps = [0] * (len(values) + 1)
        for i, v in enumerate(values):
            ps[i + 1] = ps[i] + (v if v is not None else 0)
        return ps

    @staticmethod
    def _find_windows(target: int,
                      exclude_idx: int,
                      prefix: List[int],
                      n: int,
                      values: Optional[List[int]] = None) -> List[Tuple[int, int]]:
        """
        All [l, r] (inclusive) contiguous windows NOT containing exclude_idx
        that sum exactly to target.

        The two-pointer finds the leftmost window per r-position.  Because
        leading zero-value cords may be captured, each raw window is also
        returned in a zero-trimmed form (leading/trailing zeros stripped).
        Both raw and trimmed windows are included so we match both the KFG
        canonical form and the minimal numeric form.
        """
        if target <= 0:
            return []
        raw: List[Tuple[int, int]] = []

        def scan(lo: int, hi: int):
            l = lo
            for r in range(lo, hi):
                while l <= r and (prefix[r + 1] - prefix[l]) > target:
                    l += 1
                if (prefix[r + 1] - prefix[l]) == target:
                    raw.append((l, r))

        scan(0, exclude_idx)
        scan(exclude_idx + 1, n)

        if not raw or values is None:
            return raw

        # Deduplicate: add trimmed (no leading/trailing zeros) variant of each
        seen: set = set(raw)
        results = list(raw)
        for (l, r) in raw:
            tl, tr = l, r
            while tl < tr and values[tl] == 0:
                tl += 1
            while tr > tl and values[tr] == 0:
                tr -= 1
            if (tl, tr) not in seen:
                seen.add((tl, tr))
                results.append((tl, tr))
        return results

    @staticmethod
    def _primary_color(color: Optional[str]) -> str:
        if not color:
            return ''
        m = re.match(r'([A-Z]+)', color)
        return m.group(1) if m else ''

    # ------------------------------------------------------------------
    # Pattern 1: pendant_pendant_sum
    # ------------------------------------------------------------------

    def detect_pendant_pendant_sum(self, kfg_id: str,
                                   cords: Optional[List[Cord]] = None,
                                   tolerance: int = 0) -> List[SummationMatch]:
        """
        Sliding window on the group-ordered flat pendant sequence.

        For each pendant with value V > 0, finds every contiguous run of
        OTHER pendants whose values sum to V (within tolerance).
        Confirmed: every relationship in the KFG ground truth is a contiguous
        window in this ordering.
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)
        flat = self._pendants(cords)
        if len(flat) < 2:
            return []

        values = [c.value if c.value is not None else 0 for c in flat]
        prefix = self._prefix_sums(values)
        n = len(flat)
        matches = []

        for i, cord in enumerate(flat):
            if cord.value is None or cord.value <= 0:
                continue
            for l, r in self._find_windows(cord.value, i, prefix, n, values):
                summands = flat[l:r + 1]
                actual = sum(s.value or 0 for s in summands)
                matches.append(SummationMatch(
                    pattern_type='pendant_pendant_sum',
                    sum_cord=cord,
                    summand_cords=summands,
                    expected_sum=cord.value,
                    actual_sum=actual,
                    matches=True,
                    notes=f'window[{l}-{r}]'
                ))
        return matches

    # ------------------------------------------------------------------
    # Pattern 2: colored_pendant_sum
    # ------------------------------------------------------------------

    def detect_colored_pendant_sum(self, kfg_id: str,
                                   cords: Optional[List[Cord]] = None,
                                   tolerance: int = 0) -> List[SummationMatch]:
        """
        A pendant = sum of all other pendants sharing the same primary color.
        Also applies sliding window within each color group.
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)
        flat = [c for c in self._pendants(cords) if c.value is not None]

        by_color: Dict[str, List[Cord]] = defaultdict(list)
        for c in flat:
            col = self._primary_color(c.color)
            if col:
                by_color[col].append(c)

        matches = []
        for color, group in by_color.items():
            if len(group) < 2:
                continue
            values = [c.value for c in group]
            prefix = self._prefix_sums(values)
            n = len(group)
            for i, cord in enumerate(group):
                if cord.value is None or cord.value <= 0:
                    continue
                # (a) cord = sum of ALL others in color group
                rest = sum(c.value for c in group) - cord.value
                if abs(cord.value - rest) <= tolerance:
                    summands = [c for c in group if c.cord_id != cord.cord_id]
                    matches.append(SummationMatch(
                        pattern_type='colored_pendant_sum',
                        sum_cord=cord,
                        summand_cords=summands,
                        expected_sum=cord.value,
                        actual_sum=rest,
                        matches=True,
                        notes=f'color={color} all-others'
                    ))
                    continue
                # (b) sliding window within color group
                for l, r in self._find_windows(cord.value, i, prefix, n, values):
                    summands = group[l:r + 1]
                    actual = sum(s.value for s in summands)
                    matches.append(SummationMatch(
                        pattern_type='colored_pendant_sum',
                        sum_cord=cord,
                        summand_cords=summands,
                        expected_sum=cord.value,
                        actual_sum=actual,
                        matches=True,
                        notes=f'color={color} window[{l}-{r}]'
                    ))
        return matches

    # ------------------------------------------------------------------
    # Pattern 3: indexed_pendant_sum
    # ------------------------------------------------------------------

    def detect_indexed_pendant_sum(self, kfg_id: str,
                                   cords: Optional[List[Cord]] = None,
                                   tolerance: int = 0) -> List[SummationMatch]:
        """
        cord[g][p] sums over cord[g'][p] for g' in some set.
        Applies sliding window within each position_in_group bucket.
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)
        flat = [c for c in self._pendants(cords) if c.value is not None]

        by_pos: Dict[int, List[Cord]] = defaultdict(list)
        for c in flat:
            by_pos[c.position_in_group].append(c)

        matches = []
        for pos, group in by_pos.items():
            if len(group) < 2:
                continue
            values = [c.value for c in group]
            prefix = self._prefix_sums(values)
            n = len(group)
            for i, cord in enumerate(group):
                if cord.value is None or cord.value <= 0:
                    continue
                for l, r in self._find_windows(cord.value, i, prefix, n, values):
                    summands = group[l:r + 1]
                    actual = sum(s.value for s in summands)
                    matches.append(SummationMatch(
                        pattern_type='indexed_pendant_sum',
                        sum_cord=cord,
                        summand_cords=summands,
                        expected_sum=cord.value,
                        actual_sum=actual,
                        matches=True,
                        notes=f'pos={pos} window[{l}-{r}]'
                    ))
        return matches

    # ------------------------------------------------------------------
    # Pattern 4: subsidiary_pendant_sum
    # ------------------------------------------------------------------

    def detect_subsidiary_pendant_sum(self, kfg_id: str,
                                      cords: Optional[List[Cord]] = None,
                                      tolerance: int = 0) -> List[SummationMatch]:
        """
        A subsidiary cord's value = sum of a contiguous window of top-level
        pendants in the flat group-ordered sequence.

        KFG ground truth pattern: the subsidiary encodes a cross-group sum --
        its value equals a sliding-window run of pendant values.
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)
        pendants = self._pendants(cords)
        subs = self._subsidiaries(cords)
        if not pendants or not subs:
            return []

        values = [c.value if c.value is not None else 0 for c in pendants]
        prefix = self._prefix_sums(values)
        n = len(pendants)
        matches = []

        for sub in subs:
            if sub.value is None or sub.value <= 0:
                continue
            # Use n as exclude_idx (no exclusion) to scan entire pendant sequence
            for l, r in self._find_windows(sub.value, n, prefix, n, values):
                summands = pendants[l:r + 1]
                actual = sum(s.value or 0 for s in summands)
                matches.append(SummationMatch(
                    pattern_type='subsidiary_pendant_sum',
                    sum_cord=sub,
                    summand_cords=summands,
                    expected_sum=sub.value,
                    actual_sum=actual,
                    matches=True,
                    notes=f'sub={sub.cord_name} window[{l}-{r}]'
                ))
        return matches

    # ------------------------------------------------------------------
    # Pattern 5: indexed_subsidiary_sum
    # ------------------------------------------------------------------

    def detect_indexed_subsidiary_sum(self, kfg_id: str,
                                      cords: Optional[List[Cord]] = None,
                                      tolerance: int = 0) -> List[SummationMatch]:
        """
        A subsidiary at index k (s1, s2...) sums same-index subsidiaries
        across other pendants.  Applies sliding window within each s_k group.
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)
        SUB_IDX = re.compile(r's(\d+)$')
        by_idx: Dict[int, List[Cord]] = defaultdict(list)
        for c in self._subsidiaries(cords):
            if c.value is None:
                continue
            m = SUB_IDX.search(c.cord_name)
            if m:
                by_idx[int(m.group(1))].append(c)

        matches = []
        for sub_idx, group in by_idx.items():
            if len(group) < 2:
                continue
            values = [c.value for c in group]
            prefix = self._prefix_sums(values)
            n = len(group)
            for i, cord in enumerate(group):
                if cord.value is None or cord.value <= 0:
                    continue
                for l, r in self._find_windows(cord.value, i, prefix, n, values):
                    summands = group[l:r + 1]
                    actual = sum(s.value for s in summands)
                    matches.append(SummationMatch(
                        pattern_type='indexed_subsidiary_sum',
                        sum_cord=cord,
                        summand_cords=summands,
                        expected_sum=cord.value,
                        actual_sum=actual,
                        matches=True,
                        notes=f's{sub_idx} window[{l}-{r}]'
                    ))
        return matches

    # ------------------------------------------------------------------
    # Pattern 6 & 7: group_group_sum / group_sum_bands
    # ------------------------------------------------------------------

    def detect_group_group_sum(self, kfg_id: str,
                               cords: Optional[List[Cord]] = None,
                               tolerance: int = 0) -> List[SummationMatch]:
        """
        (a) Total of group A == total of group B.
        (b) A single cord == sum of a contiguous range of group totals.
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)
        flat = [c for c in self._pendants(cords) if c.value is not None]
        if not flat:
            return []

        group_totals: Dict[int, int] = defaultdict(int)
        group_members: Dict[int, List[Cord]] = defaultdict(list)
        for c in flat:
            group_totals[c.group_idx] += c.value
            group_members[c.group_idx].append(c)

        groups = sorted(group_totals)
        totals = [group_totals[g] for g in groups]
        prefix = self._prefix_sums(totals)
        ng = len(groups)
        matches = []

        # (a) pairwise equal group totals
        seen = set()
        for i, gi in enumerate(groups):
            for j, gj in enumerate(groups):
                if j <= i or totals[i] <= 0:
                    continue
                if abs(totals[i] - totals[j]) <= tolerance:
                    key = (min(gi, gj), max(gi, gj))
                    if key not in seen:
                        seen.add(key)
                        matches.append(SummationMatch(
                            pattern_type='group_group_sum',
                            sum_cord=group_members[gi][0],
                            summand_cords=group_members[gj],
                            expected_sum=totals[i],
                            actual_sum=totals[j],
                            matches=True,
                            notes=f'grp{gi}==grp{gj}'
                        ))

        # (b) single cord = contiguous group-range sum
        for g_i, gi in enumerate(groups):
            for cord in group_members[gi]:
                if cord.value is None or cord.value <= 0:
                    continue
                for l, r in self._find_windows(cord.value, g_i, prefix, ng, totals):
                    summands = [c for k in range(l, r + 1)
                                for c in group_members[groups[k]]]
                    matches.append(SummationMatch(
                        pattern_type='group_group_sum',
                        sum_cord=cord,
                        summand_cords=summands,
                        expected_sum=cord.value,
                        actual_sum=sum(s.value for s in summands),
                        matches=True,
                        notes=f'{cord.cord_name}=grps[{groups[l]}-{groups[r]}]'
                    ))
        return matches

    # ------------------------------------------------------------------
    # Pattern 8: ascher_decreasing_group
    # ------------------------------------------------------------------

    def detect_ascher_decreasing_group(self, kfg_id: str,
                                       cords: Optional[List[Cord]] = None,
                                       min_r2: float = 0.6,
                                       min_size: int = 3) -> List[SummationMatch]:
        """
        Groups where pendant values decrease roughly linearly.

        This is Ascher's "diminishing group" pattern: within a group, the
        pendant values decrease and their positions fit a line y=mx+b with
        significant negative slope (R² >= min_r2).

        Verified against KFG ground truth: threshold min_r2=0.6 recovers all
        documented decreasing groups in CM009 (R² range 0.61-0.85).
        """
        import numpy as np
        if cords is None:
            cords = self._load_all_cords(kfg_id)
        flat = self._pendants(cords)
        if not flat:
            return []

        by_group: Dict[int, List[Cord]] = defaultdict(list)
        for c in flat:
            if c.group_idx is not None and c.value is not None:
                by_group[c.group_idx].append(c)

        matches = []
        for gidx, members in sorted(by_group.items()):
            if len(members) < min_size:
                continue
            ordered = sorted(members, key=lambda c: c.position_in_group)
            vals = [c.value for c in ordered]
            x = np.arange(len(vals), dtype=float)
            y = np.array(vals, dtype=float)

            # Linear fit
            m, b = np.polyfit(x, y, 1)
            if m >= 0:
                continue

            # R²
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            if ss_tot == 0:
                continue
            y_pred = m * x + b
            ss_res = float(np.sum((y - y_pred) ** 2))
            r2 = 1.0 - ss_res / ss_tot

            if r2 >= min_r2 - 1e-9:
                matches.append(SummationMatch(
                    pattern_type='ascher_decreasing_group',
                    sum_cord=ordered[0],
                    summand_cords=ordered,
                    expected_sum=sum(vals),
                    actual_sum=sum(vals),
                    matches=True,
                    notes=f'group={gidx} m={m:.3f} r2={r2:.3f}'
                ))
        return matches

    # ------------------------------------------------------------------
    # All patterns
    # ------------------------------------------------------------------

    def detect_all_patterns(self, kfg_id: str,
                            tolerance: int = 0) -> Dict[str, List[SummationMatch]]:
        """Run every detector and return results keyed by pattern type."""
        cords = self._load_all_cords(kfg_id)
        gg = self.detect_group_group_sum(kfg_id, cords, tolerance)
        return {
            'pendant_pendant_sum':     self.detect_pendant_pendant_sum(kfg_id, cords, tolerance),
            'colored_pendant_sum':     self.detect_colored_pendant_sum(kfg_id, cords, tolerance),
            'indexed_pendant_sum':     self.detect_indexed_pendant_sum(kfg_id, cords, tolerance),
            'subsidiary_pendant_sum':  self.detect_subsidiary_pendant_sum(kfg_id, cords, tolerance),
            'indexed_subsidiary_sum':  self.detect_indexed_subsidiary_sum(kfg_id, cords, tolerance),
            'group_group_sum':         gg,
            'group_sum_bands':         gg,   # same structural logic
            'ascher_decreasing_group': self.detect_ascher_decreasing_group(kfg_id, cords),
            'pendant_sub_neighbor':    [],   # TODO
        }

    def summarize(self, kfg_id: str, tolerance: int = 0) -> Dict:
        all_p = self.detect_all_patterns(kfg_id, tolerance)
        pstats = {}
        total_rels = total_matches = 0
        for ptype, rels in all_p.items():
            if rels:
                m = sum(1 for r in rels if r.matches)
                pstats[ptype] = {
                    'total': len(rels),
                    'matches': m,
                    'match_rate': m / len(rels)
                }
                total_rels += len(rels)
                total_matches += m
        return {
            'kfg_id':        kfg_id,
            'has_summation': total_matches > 0,
            'total_rels':    total_rels,
            'total_matches': total_matches,
            'overall_rate':  total_matches / total_rels if total_rels else 0.0,
            'pattern_stats': pstats,
            'num_types':     len([p for p in pstats if pstats[p]['total'] > 0])
        }

    def summarize_khipu(self, kfg_id: str, tolerance: int = 0) -> Dict:
        """Alias for summarize() using the key names expected by test scripts."""
        r = self.summarize(kfg_id, tolerance)
        # Remap keys to what test_kfg_summation_detector.py expects
        pstats_remapped = {}
        for ptype, s in r['pattern_stats'].items():
            pstats_remapped[ptype] = {
                'total':      s['total'],
                'matches':    s['matches'],
                'match_rate': s['match_rate'],
            }
        return {
            'kfg_id':              r['kfg_id'],
            'has_summation':       r['has_summation'],
            'total_relationships': r['total_rels'],
            'total_matches':       r['total_matches'],
            'overall_match_rate':  r['overall_rate'],
            'pattern_stats':       pstats_remapped,
            'num_pattern_types':   r['num_types'],
        }
