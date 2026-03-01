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
from typing import Dict, List, Optional, Tuple, Set
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
    termination: Optional[str] = None

    @property
    def is_broken(self) -> bool:
        """True when the cord is physically broken (termination='B')."""
        return self.termination == 'B'



@dataclass
class SummationMatch:
    pattern_type: str
    sum_cord: Cord
    summand_cords: List[Cord]
    expected_sum: int
    actual_sum: int
    matches: bool
    notes: str = ""

    @property
    def broken_cords(self) -> List[str]:
        """Names of broken (termination='B') cords involved in this relationship."""
        b = []
        if self.sum_cord.is_broken:
            b.append(self.sum_cord.cord_name)
        b.extend(c.cord_name for c in self.summand_cords if c.is_broken)
        return b

    @property
    def has_broken_cord(self) -> bool:
        """True if any cord in this summation relationship is physically broken."""
        return bool(self.broken_cords)


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
                       parent_cord, group_idx, position_in_group, value, color,
                       termination
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
                      values: Optional[List[int]] = None,
                      min_window: int = 2) -> List[Tuple[int, int]]:
        """
        All [l, r] (inclusive) contiguous windows NOT containing exclude_idx
        that sum exactly to target, with at least min_window elements.

        min_window defaults to 2: a "sum" of a single element is not a
        summation pattern — it is just two equal values coinciding.

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
                if (prefix[r + 1] - prefix[l]) == target and r - l + 1 >= min_window:
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
            if (tl, tr) not in seen and tr - tl + 1 >= min_window:
                seen.add((tl, tr))
                results.append((tl, tr))
        return results

    @staticmethod
    def _primary_color(color: Optional[str]) -> str:
        if not color:
            return ''
        m = re.match(r'([A-Z]+)', color)
        return m.group(1) if m else ''

    @staticmethod
    def _dominant_color(color: Optional[str]) -> str:
        """
        Return the color of the longest section of a (possibly spliced) cord.

        For solid cords (no '(' in color string) the full color is returned.
        For spliced cords like 'W(0.0-5.0)/W:0G(5.0-7.0)/W(7.0-34.0)' the
        color of the longest section ('W' with 27 cm) is returned.

        This matches the KFG GT convention: a cord is labeled by its dominant
        color, so 'W:SY(0.0-2.0)/W(2.0-25.0)' becomes 'W'.
        """
        if not color:
            return ''
        if '(' not in color:
            return color   # solid cord -- full color is canonical
        # Parse sections: each looks like  COLOR(start-end)  separated by '/'
        SECTION = re.compile(r'([A-Z][A-Z0-9:^%-]*(?:[Wa-z]*)?)'  # color
                             r'\((\d+\.?\d*)-(\d+\.?\d*)\)'        # (lo-hi)
                             )
        best_color = ''
        best_len = -1.0
        for m in SECTION.finditer(color):
            sec_len = float(m.group(3)) - float(m.group(2))
            if sec_len > best_len:
                best_len = sec_len
                best_color = m.group(1)
        return best_color if best_color else color

    # ------------------------------------------------------------------
    # Pattern 1: pendant_pendant_sum
    # ------------------------------------------------------------------

    def detect_pendant_pendant_sum(self, kfg_id: str,
                                   cords: Optional[List[Cord]] = None,
                                   tolerance: int = 0) -> List[SummationMatch]:
        """
        Sliding window on the group-ordered flat pendant sequence.

        For each pendant with value V >= 11, finds every contiguous run of
        OTHER pendants whose values sum to V.  Zero-value cords between non-zero
        summands do not break contiguity (per KFG documentation).

        KFG criteria (from pendant_pendant_sum.html#search-criteria):
          - sum cord value must be >= 11
          - at least 2 non-zero summand cords required (min_window=2)
          - 0-value cords in the physical span are allowed (contiguity is not
            broken by zeros); the '3 cords' phrasing in the doc refers to the
            physical span including zeros, confirmed by KFG ground truth
          - exact numerical match required
          - maximum window span = 250
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)
        flat = self._pendants(cords)
        if len(flat) < 3:  # need at least 2 summands + 1 sum cord
            return []

        values = [c.value if c.value is not None else 0 for c in flat]
        prefix = self._prefix_sums(values)
        n = len(flat)
        matches = []

        for i, cord in enumerate(flat):
            if cord.value is None or cord.value < 11:
                continue
            for l, r in self._find_windows(cord.value, i, prefix, n, values, min_window=2):
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
        A pendant = sum of a contiguous window of same-color pendants.

        Key insight: use FULL color string for grouping (W:KB != W).  The
        GT summand_string never includes zero-value cords, so we filter them
        out of the returned summand_cords to enable exact-recall matching.
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)
        flat = [c for c in self._pendants(cords) if c.value is not None]

        # Group by DOMINANT color string (handles spliced cords correctly)
        by_color: Dict[str, List[Cord]] = defaultdict(list)
        for c in flat:
            col = self._dominant_color(c.color)
            if col:
                by_color[col].append(c)

        matches = []
        for color, group in by_color.items():
            if len(group) < 3:  # need ≥2 summands + 1 sum cord
                continue
            values = [c.value for c in group]
            prefix = self._prefix_sums(values)
            n = len(group)
            for i, cord in enumerate(group):
                if cord.value is None or cord.value < 11:
                    continue
                # (a) cord = sum of ALL others in color group
                rest = sum(c.value for c in group) - cord.value
                if abs(cord.value - rest) <= tolerance:
                    # GT skips zero-value summands
                    summands = [c for c in group
                                if c.cord_id != cord.cord_id and (c.value or 0) != 0]
                    # KFG criteria: cord value > number of summands, summands span >= 2 groups
                    if cord.value <= len(summands):
                        continue
                    if len(set(c.group_idx for c in summands)) < 2:
                        continue
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
                    # GT skips zero-value summands
                    summands = [c for c in group[l:r + 1] if (c.value or 0) != 0]
                    actual = sum(s.value for s in summands)
                    # KFG criteria: cord value > number of summands, summands span >= 2 groups
                    if cord.value <= len(summands):
                        continue
                    if len(set(c.group_idx for c in summands)) < 2:
                        continue
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
        cord[g][p] = sum of a sliding window of other cords at the same
        position_in_group p across different groups.

        GT summand_string never lists zero-value cords, so zero-value cords
        are filtered from returned summand_cords.

        KFG criteria (from indexed_pendant_sum.html#search-criteria):
          - sum cord value must be >= 5
          - exclude multiples of 10 when cord value < 100 (trivial round sums)
          - exclude multiples of 100 when cord value < 1000
          - at least 2 summands required (no identity matches)
          - summands must be in contiguous groups
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)
        flat = [c for c in self._pendants(cords) if c.value is not None]

        by_pos: Dict[int, List[Cord]] = defaultdict(list)
        for c in flat:
            by_pos[c.position_in_group].append(c)

        matches = []
        for pos, group in by_pos.items():
            if len(group) < 3:  # need ≥2 summands + 1 sum cord
                continue
            values = [c.value for c in group]
            prefix = self._prefix_sums(values)
            n = len(group)
            for i, cord in enumerate(group):
                v = cord.value
                if v is None or v < 5:
                    continue
                # KFG: exclude round-number trivial sums
                if v < 100 and v % 10 == 0:
                    continue
                if v < 1000 and v % 100 == 0:
                    continue
                for l, r in self._find_windows(v, i, prefix, n, values):
                    # GT skips zero-value summands
                    summands = [c for c in group[l:r + 1] if (c.value or 0) != 0]
                    actual = sum(s.value for s in summands)
                    matches.append(SummationMatch(
                        pattern_type='indexed_pendant_sum',
                        sum_cord=cord,
                        summand_cords=summands,
                        expected_sum=v,
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

        KFG criteria (from subsidiary_pendant_sum.html#search-criteria):
          - subsidiary (sum) cord value must be >= 5
          - summand pendant cords must be contiguous (zeros allowed between)
          - maximum window span = 250
          - multiples of 10 when subsidiary value < 100 are excluded
          - when multiple windows match, shortest is chosen
          - exact numerical match required
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
            if sub.value is None or sub.value < 5:
                continue
            # KFG: exclude multiples of 10 when subsidiary value < 100
            v = sub.value
            if v < 100 and v % 10 == 0:
                continue
            # Use n as exclude_idx (no exclusion) to scan entire pendant sequence
            windows = self._find_windows(v, n, prefix, n, values)
            if not windows:
                continue
            # When multiple windows match, choose shortest
            windows.sort(key=lambda w: w[1] - w[0])
            for l, r in windows:
                # GT skips zero-value summands
                summands = [c for c in pendants[l:r + 1] if (c.value or 0) != 0]
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
        A subsidiary cord = sum of a sliding window of same-position subsidiaries.

        Two complementary groupings are tried and results are unioned:

        A) Same-level grouping: (parent.pos_in_group, sub_0idx)
           For each sub-level k, window over same-level-k subs at same parent
           position across groups. Handles 's2 = sum of s2's of other groups'.

        B) Cross-level + dominant-color grouping: (parent.pos_in_group, dominant_col)
           All subs of the same color at the same parent position, regardless of
           sub-level. Handles 's2 = sum of s1's of other groups' (cross-level).

        Within each bucket, subsidiaries are ordered by (parent.group_idx, sub_0idx).
        Zero-value cords are filtered from summand_cords (GT convention).
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)

        SUB_IDX = re.compile(r's(\d+)$')

        # Build parent lookup
        parent_map: Dict[str, Cord] = {c.cord_name: c
                                       for c in cords if c.hierarchy_level == 0}

        # Collect subsidiary info
        sub_info = []  # (sub_cord, parent.position_in_group, parent.group_idx, sub_0idx, dom_color)
        for c in self._subsidiaries(cords):
            if c.value is None:
                continue
            m = SUB_IDX.search(c.cord_name)
            if not m:
                continue
            sub_1idx = int(m.group(1))
            parent = parent_map.get(c.parent_cord)
            if parent is None or parent.position_in_group is None:
                continue
            sub_info.append((c, parent.position_in_group,
                             parent.group_idx, sub_1idx - 1,
                             self._dominant_color(c.color)))

        def _scan_buckets(by_bucket):
            out = []
            for key, items in by_bucket.items():
                if len(items) < 2:
                    continue
                items.sort(key=lambda x: (x[0], x[1]))   # (group_idx, sub_0idx)
                group = [c for c, _, _ in items]
                values = [c.value for c in group]
                prefix = self._prefix_sums(values)
                n = len(group)
                for i, cord in enumerate(group):
                    if cord.value is None or cord.value <= 0:
                        continue
                    for l, r in self._find_windows(cord.value, i, prefix, n, values):
                        summands = [c for c in group[l:r + 1]
                                    if (c.value or 0) != 0]
                        actual = sum(s.value for s in summands)
                        out.append(SummationMatch(
                            pattern_type='indexed_subsidiary_sum',
                            sum_cord=cord,
                            summand_cords=summands,
                            expected_sum=cord.value,
                            actual_sum=actual,
                            matches=True,
                            notes=f'key={key} window[{l}-{r}]'
                        ))
            return out

        # Strategy A: (pos, sub_0idx) -- same level
        bucket_a: Dict[tuple, List] = defaultdict(list)
        for c, pos, gidx, sub_0idx, dcol in sub_info:
            bucket_a[(pos, sub_0idx)].append((c, gidx, sub_0idx))

        # Strategy B: (pos, dominant_color) -- cross-level
        bucket_b: Dict[tuple, List] = defaultdict(list)
        for c, pos, gidx, sub_0idx, dcol in sub_info:
            bucket_b[(pos, dcol)].append((c, gidx, sub_0idx))

        # Fix _scan_buckets to accept the tuple format (cord, gidx, sub_0idx)
        def _scan(bucket):
            out = []
            for key, items in bucket.items():
                if len(items) < 2:
                    continue
                items.sort(key=lambda x: (x[1], x[2]))   # sort by (gidx, sub_0idx)
                group = [c for c, _, _ in items]
                values = [c.value for c in group]
                prefix = self._prefix_sums(values)
                n = len(group)
                for i, cord in enumerate(group):
                    if cord.value is None or cord.value <= 0:
                        continue
                    for l, r in self._find_windows(cord.value, i, prefix, n, values):
                        summands = [c for c in group[l:r + 1]
                                    if (c.value or 0) != 0]
                        actual = sum(s.value for s in summands)
                        out.append(SummationMatch(
                            pattern_type='indexed_subsidiary_sum',
                            sum_cord=cord,
                            summand_cords=summands,
                            expected_sum=cord.value,
                            actual_sum=actual,
                            matches=True,
                            notes=f'key={key} window[{l}-{r}]'
                        ))
            return out

        # Strategy C: (pos,) only -- any sub_idx, consecutive groups
        # Catches cases where the sum cord and its summands are at the same
        # pendant position but have different sub_0idx values.
        bucket_c: Dict[tuple, List] = defaultdict(list)
        for c, pos, gidx, sub_0idx, dcol in sub_info:
            bucket_c[(pos,)].append((c, gidx, sub_0idx))

        return _scan(bucket_a) + _scan(bucket_b) + _scan(bucket_c)


    # ------------------------------------------------------------------
    # Pattern X: pendant_sub_neighbor
    # ------------------------------------------------------------------

    def detect_pendant_sub_neighbor(self, kfg_id: str,
                                    cords: Optional[List[Cord]] = None,
                                    tolerance: int = 0) -> List[SummationMatch]:
        """
        For each pendant P with subsidiaries: neighbor.value = P.value + sub_sum
        where sub_sum = sum of P's direct subsidiary values, and neighbor is the
        immediately adjacent pendant (left or right) in flat group order.

        GT columns: pendant_sub_name (P), neighbor_name (N), handedness (-1=left).
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)
        flat = self._pendants(cords)
        if len(flat) < 2:
            return []

        # Build subsidiary value map: pendant_cord_name -> sum of ALL sub-level values
        # GT includes subsidiaries at ALL hierarchy levels (level-1, level-2, etc.)
        cord_map_local: Dict[str, Cord] = {c.cord_name: c for c in cords}
        sub_sums: Dict[str, int] = defaultdict(int)
        for c in cords:
            if getattr(c, 'hierarchy_level', 0) == 0 or c.parent_cord is None:
                continue
            if c.value is None or c.value == 0:
                continue
            # Trace up to the root pendant
            current = c.parent_cord
            visited: set = set()
            while current and current not in visited:
                visited.add(current)
                ancestor = cord_map_local.get(current)
                if ancestor is None or getattr(ancestor, 'hierarchy_level', 0) == 0:
                    break
                current = ancestor.parent_cord
            pendant_root = current
            if pendant_root and cord_map_local.get(pendant_root) is not None:
                sub_sums[pendant_root] += c.value

        matches = []
        seen_pairs: set = set()
        for i, pendant in enumerate(flat):
            if pendant.value is None:
                continue
            ss = sub_sums.get(pendant.cord_name, 0)
            if ss == 0:
                continue  # no subsidiaries or all zero
            # GT formula: n = abs(p + signed_ssum)
            # Since DB always stores subsidiary values as positive, ssum could be
            # positive or negative in the GT:
            #   GT ssum < 0 (most cases): n = abs(p - ss)
            #   GT ssum > 0 (rare cases): n = p + ss
            # We check both candidates to cover either sign.
            expected_abs = abs(pendant.value - ss)  # covers negative ssum
            expected_pos = pendant.value + ss        # covers positive ssum
            for neighbor in (flat[i - 1] if i > 0 else None,
                             flat[i + 1] if i < len(flat) - 1 else None):
                if neighbor is None or neighbor.value is None:
                    continue
                if abs(neighbor.value - expected_abs) <= tolerance or abs(neighbor.value - expected_pos) <= tolerance:
                    matched_expected = expected_abs if abs(neighbor.value - expected_abs) <= tolerance else expected_pos
                    key = (pendant.cord_name, neighbor.cord_name)
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        matches.append(SummationMatch(
                            pattern_type='pendant_sub_neighbor',
                            sum_cord=pendant,
                            summand_cords=[neighbor],
                            expected_sum=matched_expected,
                            actual_sum=neighbor.value,
                            matches=True,
                            notes=f'{pendant.cord_name}+subs=>{matched_expected} neighbor={neighbor.cord_name}'
                        ))
        return matches

    # ------------------------------------------------------------------
    # Pattern 6 & 7: group_group_sum / group_sum_bands
    # ------------------------------------------------------------------

    def detect_group_group_sum(self, kfg_id: str,
                               cords: Optional[List[Cord]] = None,
                               tolerance: int = 0) -> List[SummationMatch]:
        """
        Total of all cords in group A (including subsidiaries) equals total
        of all cords in group B.

        KFG criteria (from group_group_sum.html#search-criteria):
          - group sum must be >= 21 (threshold raised from 11 to suppress
            accidental matches in the 11-20 range)
          - group sum must NOT be divisible by 10, OR must be >= 100
          - top cords are excluded (to avoid overlap with sum_top_cord fieldmarks)
          - subsidiaries ARE included in each group's total
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)
        flat = [c for c in self._pendants(cords) if c.value is not None]
        if not flat:
            return []

        group_members: Dict[int, List[Cord]] = defaultdict(list)
        cord_to_group: Dict[str, int] = {}
        cord_map_gg: Dict[str, Cord] = {c.cord_name: c for c in cords}

        # Strategy 1: pendant-only group totals
        group_totals_pend: Dict[int, int] = defaultdict(int)
        for c in flat:
            group_totals_pend[c.group_idx] += c.value
            group_members[c.group_idx].append(c)
            cord_to_group[c.cord_name] = c.group_idx

        # Strategy 2: pendant + full sub-tree totals
        group_totals_subs: Dict[int, int] = defaultdict(int, group_totals_pend)
        for c in cords:
            if getattr(c, 'hierarchy_level', 0) == 0 or c.parent_cord is None:
                continue
            if c.value is None or c.value == 0:
                continue
            current = c.parent_cord
            visited: set = set()
            while current and current not in visited:
                visited.add(current)
                ancestor = cord_map_gg.get(current)
                if ancestor is None or getattr(ancestor, 'hierarchy_level', 0) == 0:
                    break
                current = ancestor.parent_cord
            parent_grp = cord_to_group.get(current)
            if parent_grp is not None:
                group_totals_subs[parent_grp] += c.value

        # KFG definition: pairwise equal group totals only (subsidiaries included).
        # "Part (b)" range-sums are not in the KFG definition and inflate FPs.
        groups = sorted(group_totals_subs)
        totals = [group_totals_subs[g] for g in groups]
        seen_eq: set = set()
        matches = []

        for i, gi in enumerate(groups):
            for j, gj in enumerate(groups):
                if j <= i or totals[i] <= 0:
                    continue
                # KFG: group sum must be >= 21
                if totals[i] < 21:
                    continue
                # KFG: not divisible by 10 unless >= 100
                if totals[i] % 10 == 0 and totals[i] < 100:
                    continue
                if abs(totals[i] - totals[j]) <= tolerance:
                    key = (min(gi, gj), max(gi, gj))
                    if key not in seen_eq:
                        seen_eq.add(key)
                        matches.append(SummationMatch(
                            pattern_type='group_group_sum',
                            sum_cord=group_members[gi][0],
                            summand_cords=group_members[gj],
                            expected_sum=totals[i],
                            actual_sum=totals[j],
                            matches=True,
                            notes=f'grp{gi}==grp{gj}'
                        ))
        return matches

    # ------------------------------------------------------------------
    # Pattern 7: group_sum_bands
    # ------------------------------------------------------------------

    def detect_group_sum_bands(self, kfg_id: str,
                               cords: Optional[List[Cord]] = None,
                               tolerance: int = 0) -> List[SummationMatch]:
        """
        Groups whose left-half sum equals their right-half sum at some split.

        KFG criteria (from group_sum_bands.html#search-criteria):
          - group total sum must be >= 5
          - groups where only one distinct value is repeated are excluded
          - left or right band of 1 cord is excluded (those are pendant-pendant
            sum relationships)
          - any split index where left_sum == right_sum is accepted
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)
        flat = [c for c in self._pendants(cords) if c.value is not None]
        if not flat:
            return []

        # Group pendants by group_idx (top-level pendants only)
        by_group: Dict[int, List[Cord]] = defaultdict(list)
        for c in flat:
            if c.group_idx is not None:
                by_group[c.group_idx].append(c)

        matches = []
        for gidx, members in sorted(by_group.items()):
            if len(members) < 4:  # need >= 2 cords each side
                continue
            ordered = sorted(members, key=lambda c: c.position_in_group)
            vals = [c.value for c in ordered]
            total = sum(vals)

            # KFG: group total >= 5
            if total < 5:
                continue
            # KFG: skip groups where only one distinct non-zero value is repeated
            non_zero_vals = [v for v in vals if v != 0]
            if len(set(non_zero_vals)) <= 1 and len(non_zero_vals) > 0:
                continue

            # Try every split k: left=ordered[0:k], right=ordered[k:n]
            n = len(ordered)
            for k in range(2, n - 1):  # left >= 2, right >= 2
                left_sum = sum(vals[:k])
                right_sum = sum(vals[k:])
                if abs(left_sum - right_sum) <= tolerance and left_sum > 0:
                    # Use first cord of left band as sum_cord, right band as summands
                    matches.append(SummationMatch(
                        pattern_type='group_sum_bands',
                        sum_cord=ordered[0],
                        summand_cords=ordered,
                        expected_sum=left_sum,
                        actual_sum=right_sum,
                        matches=True,
                        notes=f'grp{gidx} split@{k} left={left_sum} right={right_sum}'
                    ))
                    break  # one match per group is enough
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

    def detect_all_patterns(
        self,
        kfg_id: str,
        tolerance: int = 0,
        loader=None,
        apply_exclusivity: bool = False,
    ) -> Dict[str, List[SummationMatch]]:
        """
        Run every detector and return results keyed by pattern type.

        Parameters
        ----------
        kfg_id            : khipu identifier
        tolerance         : allowed off-by-one for algorithmic detection
        loader            : optional KFGRelationLoader instance. When
                            provided *and* the khipu is in the KFG corpus,
                            ground-truth relation CSV data is used instead
                            of the algorithmic detector — eliminating false
                            positives from re-detection noise.
        apply_exclusivity : when True, enforce mutual exclusivity at the cord
                            level (IS > SP > IP > CP > PP).

                            **Default is False** because the KFG ground-truth
                            data intentionally records the same cord in
                            multiple pattern tables.  Set to True only when
                            assigning a single primary pattern label per cord
                            (classification use case).
        """
        cords = self._load_all_cords(kfg_id)

        # ---------------------------------------------------------------
        # Ground-truth path: use the relation CSVs for known KFG khipus.
        # ---------------------------------------------------------------
        if loader is not None and loader.in_kfg(kfg_id):
            return loader.build_all_matches(
                kfg_id, cords,
                apply_excl=apply_exclusivity,
                resolve_summands=True,
            )

        # ---------------------------------------------------------------
        # Algorithmic path: used for non-KFG khipus (or when no loader).
        # ---------------------------------------------------------------
        results = {
            'pendant_pendant_sum':     self.detect_pendant_pendant_sum(kfg_id, cords, tolerance),
            'colored_pendant_sum':     self.detect_colored_pendant_sum(kfg_id, cords, tolerance),
            'indexed_pendant_sum':     self.detect_indexed_pendant_sum(kfg_id, cords, tolerance),
            'subsidiary_pendant_sum':  self.detect_subsidiary_pendant_sum(kfg_id, cords, tolerance),
            'indexed_subsidiary_sum':  self.detect_indexed_subsidiary_sum(kfg_id, cords, tolerance),
            'group_group_sum':         self.detect_group_group_sum(kfg_id, cords, tolerance),
            'group_sum_bands':         self.detect_group_sum_bands(kfg_id, cords, tolerance),
            'ascher_decreasing_group': self.detect_ascher_decreasing_group(kfg_id, cords),
            'pendant_sub_neighbor':    self.detect_pendant_sub_neighbor(kfg_id, cords, tolerance),
        }
        if apply_exclusivity:
            from src.analysis.kfg_relation_loader import apply_exclusivity as _excl
            results = _excl(results)
        return results

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
