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
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import sys
from functools import wraps

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
    attachment: Optional[str] = None

    @property
    def is_broken(self) -> bool:
        """True when the cord is physically broken (termination='B')."""
        return self.termination == 'B'

    @property
    def is_top_cord(self) -> bool:
        """True when the cord has top-cord attachment (attachment='T')."""
        return self.attachment == 'T'



@dataclass
class SummationMatch:
    pattern_type: str
    sum_cord: Cord
    summand_cords: List[Cord]
    expected_sum: int
    actual_sum: int
    matches: bool
    notes: str = ""
    handedness: Optional[float] = None  # signed: negative=left, positive/zero=right (KFG convention)
    is_dual_sum: bool = False  # True if sum cord has multiple summand windows
    figure8_proximity: Optional[Dict] = None  # Figure-8 knot proximity info

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
    
    @property
    def summand_range(self) -> Optional[Tuple[int, int]]:
        """Return (min_group_idx, max_group_idx) or (min_position, max_position) of summands."""
        if not self.summand_cords:
            return None
        positions: list[int] = [c.group_idx if c.group_idx is not None else c.position_in_group 
                     for c in self.summand_cords 
                     if c.group_idx is not None or c.position_in_group is not None]  # type: ignore[misc]
        if not positions:
            return None
        return (min(positions), max(positions))


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def time_pattern_detection(func):
    """Decorator to time pattern detection methods."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_timing_enabled') or not self._timing_enabled:
            return func(self, *args, **kwargs)
        
        start = time.perf_counter()
        result = func(self, *args, **kwargs)
        elapsed = time.perf_counter() - start
        
        pattern_name = func.__name__.replace('detect_', '')
        if not hasattr(self, '_timing_data'):
            self._timing_data = {}
        self._timing_data[pattern_name] = elapsed
        
        return result
    return wrapper


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
    
    PATTERN_TYPES = [
        'pendant_pendant_sum',
        'colored_pendant_sum',
        'indexed_pendant_sum',
        'subsidiary_pendant_sum',
        'indexed_subsidiary_sum',
        'pendant_sub_neighbor',
        'group_group_sum',
        'group_sum_bands',
        'ascher_decreasing_group'
    ]

    def __init__(self, db_path, enable_timing=False):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self._timing_enabled = enable_timing
        self._timing_data = {}

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
                       termination, attachment
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
    
    def _load_figure8_knots(self, kfg_id: str) -> Dict[str, List[float]]:
        """Load figure-8 knot positions for each cord.
        Returns: {cord_name: [position_cm, ...]} for cords with E/EE knots.
        """
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT c.cord_name, kc.position_cm
                FROM cords c
                JOIN knot_clusters kc ON c.cord_id = kc.cord_id
                WHERE c.kfg_id = ? 
                  AND kc.knot_type IN ('E', 'EE')
                  AND kc.position_cm IS NOT NULL
                ORDER BY c.cord_name, kc.position_cm
            """, (kfg_id,))
            
            figure8_map = defaultdict(list)
            for cord_name, pos_cm in cur.fetchall():
                figure8_map[cord_name].append(float(pos_cm))
            
            return dict(figure8_map)
    
    def _build_eight_knot_cord_ids(self, kfg_id: str) -> Set[int]:
        """Return the set of cord_ids that have at least one figure-8 knot (E/EE)."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT DISTINCT kc.cord_id
                FROM knot_clusters kc
                JOIN cords c ON c.cord_id = kc.cord_id
                WHERE c.kfg_id = ? AND kc.knot_type IN ('E', 'EE')
            """, (kfg_id,))
            return {row[0] for row in cur.fetchall()}

    # ------------------------------------------------------------------
    # Algorithm helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prefix_sums(values: List[int] | List[Optional[int]]) -> List[int]:
        ps = [0] * (len(values) + 1)
        for i, v in enumerate(values):
            ps[i + 1] = ps[i] + (v if v is not None else 0)
        return ps

    @staticmethod
    def _find_windows_optimized(target: int,
                                exclude_idx: int,
                                values: List[int],
                                min_window: int = 2,
                                first_only: bool = True,
                                filter_all_ones: bool = True) -> List[Tuple[int, int, str]]:
        """Find contiguous windows that sum to target using hash-based lookup.
        
        Returns list of (start, end, side) tuples where:
        - start, end: window bounds (inclusive)
        - side: 'left' if window is left of exclude_idx, 'right' if right
        
        Parameters
        ----------
        first_only : bool
            When True (default), return at most ONE window per side, matching
            KFG CordSummer behavior where search_sum_at() returns the first
            valid match and stops.
        filter_all_ones : bool
            When True (default), reject windows where every value is 1
            (matching KFG filter_one_summands=True).
        """
        if target <= 0:
            return []
        
        n = len(values)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + values[i]
        
        results: List[Tuple[int, int, str]] = []
        
        # Scan left of exclude_idx
        if exclude_idx > 0:
            seen_prefix: Dict[int, int] = {0: -1}
            for r in range(exclude_idx):
                current_sum = prefix[r + 1]
                need = current_sum - target
                
                if need in seen_prefix:
                    l = seen_prefix[need] + 1
                    if r - l + 1 >= min_window:
                        # Trim leading/trailing zeros
                        tl, tr = l, r
                        while tl < tr and values[tl] == 0:
                            tl += 1
                        while tr > tl and values[tr] == 0:
                            tr -= 1
                        if tr - tl + 1 >= min_window:
                            # Filter all-ones windows
                            if filter_all_ones and all(values[k] == 1 for k in range(tl, tr + 1)):
                                pass  # skip
                            else:
                                results.append((tl, tr, 'left'))
                                if first_only:
                                    break
                
                seen_prefix[current_sum] = r
        
        # Scan right of exclude_idx
        if exclude_idx < n - 1:
            seen_prefix = {prefix[exclude_idx + 1]: exclude_idx}
            for r in range(exclude_idx + 1, n):
                current_sum = prefix[r + 1]
                need = current_sum - target
                
                if need in seen_prefix:
                    l = seen_prefix[need] + 1
                    if l > exclude_idx and r - l + 1 >= min_window:
                        # Trim leading/trailing zeros
                        tl, tr = l, r
                        while tl < tr and values[tl] == 0:
                            tl += 1
                        while tr > tl and values[tr] == 0:
                            tr -= 1
                        if tr - tl + 1 >= min_window:
                            # Filter all-ones windows
                            if filter_all_ones and all(values[k] == 1 for k in range(tl, tr + 1)):
                                pass  # skip
                            else:
                                results.append((tl, tr, 'right'))
                                if first_only:
                                    break
                
                seen_prefix[current_sum] = r
        
        return results

    @staticmethod
    def _find_first_window_nearest_start(
        target: int,
        exclude_idx: int,
        values: List[int],
        min_window: int = 2,
        filter_all_ones: bool = True,
    ) -> List[Tuple[int, int, str]]:
        """Find at most one window per side using nearest-start-first ordering.

        Matches KFG CordSummer.search_sum_at() behavior:
        - Iterates start positions from nearest to sum cord outward
        - Skips zero-value start positions
        - For each start, finds the first (shortest) window ending there
        - Returns immediately on first valid match (size >= min_window)
        - If first cumsum match is too short (< min_window), stops entirely

        Returns at most 2 tuples: one 'right', one 'left'.
        """
        if target <= 0:
            return []

        n = len(values)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + values[i]

        results: List[Tuple[int, int, str]] = []

        # --- Right scan (start positions after exclude_idx) ---
        for start in range(exclude_idx + 1, n):
            if values[start] == 0:
                continue
            needed = prefix[start] + target
            found_cumsum_hit = False
            # Search for first end where cumsum matches target
            for end in range(start, n):
                if prefix[end + 1] == needed:
                    found_cumsum_hit = True
                    if end - start + 1 >= min_window:
                        if not (filter_all_ones and all(values[k] == 1 for k in range(start, end + 1))):
                            results.append((start, end, 'right'))
                    break  # KFG: stop at first cumsum match regardless of window size
                if prefix[end + 1] > needed:
                    break  # prefix is non-decreasing
            # KFG: if cumsum matched but window was too short, give up entirely
            if found_cumsum_hit:
                break

        # --- Left scan (reverse order, then map indices back) ---
        rev_values = values[:exclude_idx][::-1]
        n_left = len(rev_values)
        if n_left >= min_window:
            rev_prefix = [0] * (n_left + 1)
            for i in range(n_left):
                rev_prefix[i + 1] = rev_prefix[i] + rev_values[i]

            for start in range(0, n_left):
                if rev_values[start] == 0:
                    continue
                needed = rev_prefix[start] + target
                found_cumsum_hit = False
                for end in range(start, n_left):
                    if rev_prefix[end + 1] == needed:
                        found_cumsum_hit = True
                        if end - start + 1 >= min_window:
                            orig_start = exclude_idx - 1 - end
                            orig_end = exclude_idx - 1 - start
                            if not (filter_all_ones and all(values[k] == 1 for k in range(orig_start, orig_end + 1))):
                                results.append((orig_start, orig_end, 'left'))
                        break
                    if rev_prefix[end + 1] > needed:
                        break
                if found_cumsum_hit:
                    break

        return results

    @staticmethod
    def _find_windows(target: int,
                      exclude_idx: int,
                      prefix: List[int],
                      n: int,
                      values: Optional[List[int]] = None,
                      min_window: int = 2) -> List[Tuple[int, int]]:
        """Legacy two-pointer window finder (kept for backward compatibility).
        
        Used by CPS, IPS, ISS, SP pattern detectors.
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
    def _dedup_shortest_per_cord(
        matches: List['SummationMatch'],
        is_trivial: Optional[Callable[['SummationMatch'], bool]] = None,
    ) -> List['SummationMatch']:
        """Keep only the shortest summand window per sum cord per side.

        Mirrors KFG's ``filter_results`` + ``fix_handedness`` pipeline:
        1. Split matches into left (handedness < 0) and right (handedness >= 0).
        2. For each side, iterate matches in ascending summand-count order.
        3. For each sum cord: if first (shortest) match is trivial, block the cord
           entirely; otherwise keep the shortest match.
        4. Merge left + right.

        Parameters
        ----------
        matches : list of SummationMatch
            Raw matches (multiple windows per cord allowed).
        is_trivial : callable, optional
            ``is_trivial(match) -> bool``.  When *None*, no triviality filter
            is applied.
        """
        if not matches:
            return []

        from collections import defaultdict

        def _filter_side(side_matches: List['SummationMatch']) -> List['SummationMatch']:
            # Sort by (sum_cord_id, number of summands) — shortest first
            side_matches.sort(key=lambda m: (m.sum_cord.cord_id, len(m.summand_cords)))
            best: Dict[int, Optional['SummationMatch']] = {}  # cord_id → match or None (blocked)
            for m in side_matches:
                cid = m.sum_cord.cord_id
                if cid in best:
                    continue  # already resolved (kept or blocked)
                if is_trivial and is_trivial(m):
                    best[cid] = None  # block this cord
                else:
                    best[cid] = m
            return [m for m in best.values() if m is not None]

        left = [m for m in matches if m.handedness is not None and m.handedness < 0]
        right = [m for m in matches if m.handedness is None or m.handedness >= 0]

        return _filter_side(left) + _filter_side(right)

    @staticmethod
    def _obo_equal(a_value: int, test_value: int) -> bool:
        """Off-by-one equality: exact for values < 100, else allow one digit off."""
        if a_value < 100:
            return a_value == test_value
        digits = list(str(test_value))
        for i, d in enumerate(digits):
            di = int(d)
            for near in range(max(0, di - 1), min(9, di + 1) + 1):
                candidate = int(''.join(digits[:i]) + str(near) + ''.join(digits[i + 1:]))
                if candidate == a_value:
                    return True
        return False

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

    @staticmethod
    def _build_child_map(cords: List[Cord]) -> Dict[int, List[int]]:
        """Map pendant cord_id -> list of subsidiary cord_ids."""
        name_to_cord: Dict[str, Cord] = {c.cord_name: c for c in cords}
        child_map: Dict[int, List[int]] = defaultdict(list)
        for c in cords:
            if c.hierarchy_level is not None and c.hierarchy_level >= 1 and c.parent_cord:
                parent = name_to_cord.get(c.parent_cord)
                if parent is not None and parent.hierarchy_level == 0:
                    child_map[parent.cord_id].append(c.cord_id)
        return dict(child_map)

    @staticmethod
    def _check_figure8_structural(
        summand_cords: List[Cord],
        flat_pendants: List[Cord],
        flat_idx_map: Dict[int, int],
        eight_knot_ids: Set[int],
        child_map: Dict[int, List[int]],
    ) -> Optional[Dict]:
        """Check figure-8 knot markers using KFG structural adjacency rules.

        KFG checks first and last summand cords for eight_knots in 4 categories:
        - exact pendant: the summand cord itself has an eight_knot
        - exact subsidiary: a subsidiary of the summand cord has an eight_knot
        - close pendant: left/right neighbor pendant has an eight_knot
        - close subsidiary: a subsidiary of the neighbor has an eight_knot
        """
        if not summand_cords or not eight_knot_ids:
            return None

        first = summand_cords[0]
        last = summand_cords[-1]

        def cord_has_8knot(c: 'Cord') -> bool:
            return c.cord_id in eight_knot_ids

        def subs_have_8knot(c: 'Cord') -> bool:
            return any(sid in eight_knot_ids for sid in child_map.get(c.cord_id, []))

        # Get neighbors in flat pendant list
        first_idx = flat_idx_map.get(first.cord_id)
        last_idx = flat_idx_map.get(last.cord_id)
        left_neighbor = flat_pendants[first_idx - 1] if first_idx is not None and first_idx > 0 else None
        right_neighbor = flat_pendants[last_idx + 1] if last_idx is not None and last_idx < len(flat_pendants) - 1 else None

        left_exact = cord_has_8knot(first) or subs_have_8knot(first)
        right_exact = cord_has_8knot(last) or subs_have_8knot(last)
        left_close = False
        right_close = False
        if left_neighbor is not None:
            left_close = cord_has_8knot(left_neighbor) or subs_have_8knot(left_neighbor)
        if right_neighbor is not None:
            right_close = cord_has_8knot(right_neighbor) or subs_have_8knot(right_neighbor)

        has_indicator = left_exact or right_exact or left_close or right_close
        if not has_indicator:
            return None

        return {
            'has_figure8knot_indicator': True,
            'has_left_exact': left_exact,
            'has_right_exact': right_exact,
            'has_left_close': left_close,
            'has_right_close': right_close,
        }

    # ------------------------------------------------------------------
    # Pattern 1: pendant_pendant_sum
    # ------------------------------------------------------------------

    @time_pattern_detection
    def detect_pendant_pendant_sum(self, kfg_id: str,
                                   cords: Optional[List[Cord]] = None,
                                   tolerance: int = 0) -> List[SummationMatch]:
        """
        Sliding window on the group-ordered flat pendant sequence.

        For each pendant with value V >= 11, finds the first contiguous window
        of OTHER pendants (per side) whose values sum to V.

        KFG constraints (matched for consistency):
          - sum cord value must be >= 11  (CordSummer min_sum_val)
          - at least 2 non-zero summand cords required (min_window=2)
          - first match only per side per sum cord (CordSummer early termination)
          - all-ones windows rejected (CordSummer filter_one_summands)
          - top cords excluded (PPS is_trivial_match)
          - at least one summand in a different cord group (PPS cross-group check)
          - handedness as signed integer: -(sum_index - mean(summand_indices))
          - figure-8 via structural adjacency (not distance)
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)
        flat = self._pendants(cords)
        if len(flat) < 3:
            return []

        values = [c.value if c.value is not None else 0 for c in flat]

        # Build indices for figure-8 structural checks
        eight_knot_ids = self._build_eight_knot_cord_ids(kfg_id)
        child_map = self._build_child_map(cords)
        flat_idx_map: Dict[int, int] = {c.cord_id: i for i, c in enumerate(flat)}

        # Track windows per sum cord to detect dual sums
        sum_cord_windows: Dict[int, int] = defaultdict(int)
        matches: List[SummationMatch] = []

        for i, cord in enumerate(flat):
            if cord.value is None or cord.value < 11:
                continue
            # KFG: skip top cords
            if cord.is_top_cord:
                continue

            windows = self._find_first_window_nearest_start(
                cord.value, i, values,
                min_window=2,
                filter_all_ones=False,  # KFG has `or True` bug that disables this
            )

            # Filter: at least one summand in a different group than sum cord
            valid_windows: List[Tuple[int, int, str]] = []
            for l, r, side in windows:
                summands = flat[l:r + 1]
                if any(s.group_idx != cord.group_idx for s in summands):
                    valid_windows.append((l, r, side))

            sum_cord_windows[cord.cord_id] = len(valid_windows)

            for l, r, side in valid_windows:
                summands = flat[l:r + 1]
                actual = sum(s.value or 0 for s in summands)

                # KFG handedness: -(sum_index - mean(summand_indices))
                summand_mean = sum(flat_idx_map.get(s.cord_id, 0) for s in summands) / len(summands)
                handedness_val = -(i - summand_mean)

                # Structural figure-8 check
                f8 = self._check_figure8_structural(
                    summands, flat, flat_idx_map, eight_knot_ids, child_map)

                matches.append(SummationMatch(
                    pattern_type='pendant_pendant_sum',
                    sum_cord=cord,
                    summand_cords=summands,
                    expected_sum=cord.value,
                    actual_sum=actual,
                    matches=True,
                    handedness=handedness_val,
                    is_dual_sum=(len(valid_windows) > 1),
                    figure8_proximity=f8,
                    notes=f'window[{l}-{r}] {side}'
                ))

        return matches

    # ------------------------------------------------------------------
    # Pattern 2: colored_pendant_sum
    # ------------------------------------------------------------------

    @time_pattern_detection
    def detect_colored_pendant_sum(self, kfg_id: str,
                                   cords: Optional[List[Cord]] = None,
                                   tolerance: int = 0) -> List[SummationMatch]:
        """
        A pendant = sum of a contiguous window of same-color pendants.

        KFG algorithm (fieldmark_ascher_colored_pendant_sum.py):
        - Group pendants by main_color; for each sum cord (value > 10), split
          same-color cords into left/right halves, remove zeros, find all
          contiguous windows of size >= 3 that sum to cord value.
        - filter_results: keep only shortest window per sum cord per side.
        - is_trivial_match: value <= len(summands) OR value <= 10 OR len < 2.
        - fix_handedness: recalculate from pendant indices.
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)
        flat = [c for c in self._pendants(cords) if c.value is not None]

        # Assign flat-order index for handedness calculation
        flat_order = {c.cord_id: idx for idx, c in enumerate(flat)}

        # Group by DOMINANT color string (handles spliced cords correctly)
        by_color: Dict[str, List[Cord]] = defaultdict(list)
        for c in flat:
            col = self._dominant_color(c.color)
            if col:
                by_color[col].append(c)

        raw_matches: List[SummationMatch] = []
        for color, group in by_color.items():
            if len(group) < 4:  # need >= 3 summands + 1 sum cord
                continue
            for i, cord in enumerate(group):
                if cord.value is None or cord.value < 11:
                    continue

                # KFG: split into left/right, filter zeros, find combos >= 3
                for side_label, side_cords in [
                    ('left', group[:i]),
                    ('right', group[i + 1:]),
                ]:
                    nz_cords = [c for c in side_cords if (c.value or 0) > 0]
                    if len(nz_cords) < 3:
                        continue
                    nz_values: List[int] = [c.value or 0 for c in nz_cords]
                    nz_prefix = self._prefix_sums(nz_values)
                    nz_n = len(nz_cords)
                    # Scan all windows of size >= 3 (KFG contiguous_combinations range(3,...))
                    for l, r in self._find_windows(
                        cord.value, nz_n, nz_prefix, nz_n, nz_values, min_window=3
                    ):
                        summands = nz_cords[l:r + 1]
                        actual = sum(s.value or 0 for s in summands)
                        # Signed handedness from flat pendant indices
                        sum_idx = flat_order.get(cord.cord_id, 0)
                        summand_mean = round(sum(flat_order.get(s.cord_id, 0) for s in summands) / len(summands))
                        handedness = -(sum_idx - summand_mean)
                        raw_matches.append(SummationMatch(
                            pattern_type='colored_pendant_sum',
                            sum_cord=cord,
                            summand_cords=summands,
                            expected_sum=cord.value,
                            actual_sum=actual,
                            matches=True,
                            handedness=float(handedness),
                            notes=f'color={color} {side_label} window[{l}-{r}]'
                        ))

        def _cps_trivial(m: SummationMatch) -> bool:
            v = m.sum_cord.value or 0
            return (
                v <= len(m.summand_cords)
                or v <= 10
                or len(m.summand_cords) < 2
            )

        return self._dedup_shortest_per_cord(raw_matches, is_trivial=_cps_trivial)

    # ------------------------------------------------------------------
    # Pattern 3: indexed_pendant_sum
    # ------------------------------------------------------------------

    @time_pattern_detection
    def detect_indexed_pendant_sum(self, kfg_id: str,
                                   cords: Optional[List[Cord]] = None,
                                   tolerance: int = 0) -> List[SummationMatch]:
        """
        cord[g][p] = sum of a sliding window of other cords at the same
        position_in_group p across different groups.

        KFG algorithm (fieldmark_ascher_indexed_pendant_sum.py):
        - For each group's pendants with value > 5: skip top cords, get
          same-position cords from left and right groups, filter zeros,
          find all contiguous combos of size >= 2 that sum to cord value.
        - find_sum_matches filter: value > 1, not tens (< 100), not hundreds
          (100-999), and len > 1.
        - filter_results: keep shortest per sum cord per side + is_trivial_match.
        - is_trivial_match: value <= len OR value < 11 OR (2 summands w/ any
          value=1) OR len < 2.
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)
        flat = [c for c in self._pendants(cords) if c.value is not None]

        # Assign flat-order index for handedness calculation
        flat_order = {c.cord_id: idx for idx, c in enumerate(flat)}

        by_pos: Dict[int, List[Cord]] = defaultdict(list)
        for c in flat:
            if c.position_in_group is not None:
                by_pos[c.position_in_group].append(c)

        raw_matches: List[SummationMatch] = []
        for pos, group in by_pos.items():
            if len(group) < 3:  # need >= 2 summands + 1 sum cord
                continue
            for i, cord in enumerate(group):
                v = cord.value
                if v is None or v < 6:  # KFG: > 5
                    continue
                # KFG: skip top cords
                if cord.is_top_cord:
                    continue
                # KFG find_sum_matches: pre-filter on sum cord value
                if v <= 1:
                    continue
                if v < 100 and v % 10 == 0:
                    continue
                if v >= 100 and v < 1000 and v % 100 == 0:
                    continue

                # KFG: split left/right from cord position, filter zeros
                for side_label, side_cords in [
                    ('left', group[:i]),
                    ('right', group[i + 1:]),
                ]:
                    nz_cords = [c for c in side_cords if (c.value or 0) > 0]
                    if len(nz_cords) < 2:
                        continue
                    nz_values: List[int] = [c.value or 0 for c in nz_cords]
                    nz_prefix = self._prefix_sums(nz_values)
                    nz_n = len(nz_cords)
                    for l, r in self._find_windows(
                        v, nz_n, nz_prefix, nz_n, nz_values, min_window=2
                    ):
                        summands = nz_cords[l:r + 1]
                        actual = sum(s.value or 0 for s in summands)
                        sum_idx = flat_order.get(cord.cord_id, 0)
                        summand_mean = round(sum(flat_order.get(s.cord_id, 0) for s in summands) / len(summands))
                        handedness = -(sum_idx - summand_mean)
                        raw_matches.append(SummationMatch(
                            pattern_type='indexed_pendant_sum',
                            sum_cord=cord,
                            summand_cords=summands,
                            expected_sum=v,
                            actual_sum=actual,
                            matches=True,
                            handedness=float(handedness),
                            notes=f'pos={pos} {side_label} window[{l}-{r}]'
                        ))

        def _ips_trivial(m: SummationMatch) -> bool:
            v = m.sum_cord.value or 0
            return (
                v <= len(m.summand_cords)
                or v < 11
                or (len(m.summand_cords) == 2 and any((s.value or 0) == 1 for s in m.summand_cords))
                or len(m.summand_cords) < 2
            )

        return self._dedup_shortest_per_cord(raw_matches, is_trivial=_ips_trivial)

    # ------------------------------------------------------------------
    # Pattern 4: subsidiary_pendant_sum
    # ------------------------------------------------------------------
    @time_pattern_detection
    def detect_subsidiary_pendant_sum(self, kfg_id: str,
                                      cords: Optional[List[Cord]] = None,
                                      tolerance: int = 0) -> List[SummationMatch]:
        """
        A subsidiary cord's value = sum of a contiguous window of top-level
        pendants.

        KFG algorithm (fieldmark_ascher_subsidiary_pendant_sum.py):
        - For each subsidiary with value > 10: find parent pendant, split
          pendant list into left (up to and including parent) and right
          (after parent). Zero-filter, find contiguous combos >= 3.
        - filter_results: keep shortest per sum cord per side.
        - is_trivial_match: value <= len OR value <= 10 OR (%10==0 when <100)
          OR len < 2.
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)
        pendants = self._pendants(cords)
        subs = self._subsidiaries(cords)
        if not pendants or not subs:
            return []

        # Build parent name → pendant position lookup
        pendant_by_name: Dict[str, int] = {}
        for idx, p in enumerate(pendants):
            pendant_by_name[p.cord_name] = idx

        # Flat-order index for handedness
        flat_order = {c.cord_id: idx for idx, c in enumerate(pendants)}

        raw_matches: List[SummationMatch] = []

        for sub in subs:
            v = sub.value
            if v is None or v < 11:
                continue
            # Find parent pendant position
            parent_name = sub.parent_cord
            if not parent_name or parent_name not in pendant_by_name:
                continue
            parent_pos = pendant_by_name[parent_name]

            # KFG: left includes parent (0..parent_pos+1), right is after parent
            for side_label, side_pendants in [
                ('left', pendants[:parent_pos + 1]),
                ('right', pendants[parent_pos + 1:]),
            ]:
                nz_cords = [c for c in side_pendants if (c.value or 0) > 0]
                if len(nz_cords) < 3:
                    continue
                nz_values: List[int] = [c.value or 0 for c in nz_cords]
                nz_prefix = self._prefix_sums(nz_values)
                nz_n = len(nz_cords)
                for l, r in self._find_windows(
                    v, nz_n, nz_prefix, nz_n, nz_values, min_window=3
                ):
                    summands = nz_cords[l:r + 1]
                    actual = sum(s.value or 0 for s in summands)
                    # Handedness: use pendant_index of parent and summands
                    sum_idx = flat_order.get(sub.cord_id, parent_pos)
                    summand_mean = round(sum(flat_order.get(s.cord_id, 0) for s in summands) / len(summands))
                    handedness = -(sum_idx - summand_mean)
                    raw_matches.append(SummationMatch(
                        pattern_type='subsidiary_pendant_sum',
                        sum_cord=sub,
                        summand_cords=summands,
                        expected_sum=v,
                        actual_sum=actual,
                        matches=True,
                        handedness=float(handedness),
                        notes=f'sub={sub.cord_name} {side_label} window[{l}-{r}]'
                    ))

        def _sp_trivial(m: SummationMatch) -> bool:
            v = m.sum_cord.value or 0
            return (
                v <= len(m.summand_cords)
                or v <= 10
                or (v < 100 and v % 10 == 0)
                or len(m.summand_cords) < 2
            )

        return self._dedup_shortest_per_cord(raw_matches, is_trivial=_sp_trivial)

    # ------------------------------------------------------------------
    # Pattern 5: indexed_subsidiary_sum
    # ------------------------------------------------------------------
    @time_pattern_detection
    def detect_indexed_subsidiary_sum(self, kfg_id: str,
                                      cords: Optional[List[Cord]] = None,
                                      tolerance: int = 0) -> List[SummationMatch]:
        """
        A subsidiary cord = sum of similarly-indexed (by color) subsidiary
        cords at the same pendant position in contiguous groups.

        KFG algorithm (fieldmark_ascher_indexed_subsidiary_sum.py):
        - For each pendant P with value > 5, for each subsidiary S of P:
          find S's main color and position among P's same-color subsidiaries.
          In left/right groups at the same pendant position, find subsidiaries
          with same color at same position-in-color-group.
        - Use contiguous_combinations (min size 2, no cap) over left/right
          indexed cords.  Equality test uses off-by-one for values >= 100.
        - remove_duplicates: keep shortest per sum cord; block if value <= len
          or value < 5 or len < 2.
        - fix_handedness from pendant_index.
        """
        if cords is None:
            cords = self._load_all_cords(kfg_id)

        pendants = self._pendants(cords)
        subs = self._subsidiaries(cords)
        if not pendants or not subs:
            return []

        SUB_IDX = re.compile(r's(\d+)$')

        # pendant lookup: (group_idx, position_in_group) → Cord
        pendant_by_gp: Dict[tuple, Cord] = {}
        for c in pendants:
            if c.group_idx is not None and c.position_in_group is not None:
                pendant_by_gp[(c.group_idx, c.position_in_group)] = c

        # Flat-order index for handedness (pendant cord_id → idx)
        flat_order = {c.cord_id: idx for idx, c in enumerate(pendants)}
        # Map subsidiary cord_id → parent pendant's flat index (for handedness)
        pendant_name_to_id: Dict[str, int] = {c.cord_name: c.cord_id for c in pendants}
        sub_flat_order: Dict[int, int] = {}
        for c in subs:
            if c.parent_cord and c.parent_cord in pendant_name_to_id:
                parent_cid = pendant_name_to_id[c.parent_cord]
                if parent_cid in flat_order:
                    sub_flat_order[c.cord_id] = flat_order[parent_cid]

        # Group subsidiaries by parent name → {color → [(sub_idx, Cord)]}
        sub_by_parent_color: Dict[str, Dict[str, List[Tuple[int, Cord]]]] = defaultdict(lambda: defaultdict(list))
        for c in subs:
            if c.value is None or c.parent_cord is None:
                continue
            m = SUB_IDX.search(c.cord_name)
            sub_idx = int(m.group(1)) if m else 0
            dcol = self._dominant_color(c.color)
            if dcol:
                sub_by_parent_color[c.parent_cord][dcol].append((sub_idx, c))
        # Sort each color group by sub_idx
        for parent_name in sub_by_parent_color:
            for color in sub_by_parent_color[parent_name]:
                sub_by_parent_color[parent_name][color].sort(key=lambda x: x[0])

        all_group_idxs = sorted(set(c.group_idx for c in pendants
                                     if c.group_idx is not None))

        raw_matches: List[SummationMatch] = []

        for pendant in pendants:
            if (pendant.value or 0) <= 5:
                continue
            if pendant.group_idx is None or pendant.position_in_group is None:
                continue
            gidx = pendant.group_idx
            pos = pendant.position_in_group

            color_groups = sub_by_parent_color.get(pendant.cord_name, {})
            for color, sub_list in color_groups.items():
                for pos_in_color, (_, sub_cord) in enumerate(sub_list):
                    if sub_cord.value is None:
                        continue
                    v = sub_cord.value

                    # Gather indexed cords from left/right groups
                    left_subs: List[Cord] = []
                    right_subs: List[Cord] = []
                    for other_gidx in all_group_idxs:
                        if other_gidx == gidx:
                            continue
                        other_pendant = pendant_by_gp.get((other_gidx, pos))
                        if other_pendant is None:
                            continue
                        other_color_subs = sub_by_parent_color.get(
                            other_pendant.cord_name, {}).get(color, [])
                        if not other_color_subs:
                            continue
                        # KFG: if >1 same-color → pick by position index; if 1 → take it
                        target: Optional[Cord] = None
                        if len(other_color_subs) > 1:
                            if pos_in_color < len(other_color_subs):
                                target = other_color_subs[pos_in_color][1]
                        elif len(other_color_subs) == 1:
                            target = other_color_subs[0][1]
                        if target is not None:
                            if other_gidx < gidx:
                                left_subs.append(target)
                            else:
                                right_subs.append(target)

                    # Test contiguous combinations on each side
                    for side_label, side_cords in [('left', left_subs),
                                                    ('right', right_subs)]:
                        if len(side_cords) < 2:
                            continue
                        for size in range(2, len(side_cords) + 1):
                            for start in range(len(side_cords) - size + 1):
                                combo = side_cords[start:start + size]
                                test_sum = sum(c.value or 0 for c in combo)
                                if not self._obo_equal(v, test_sum):
                                    continue
                                # Inline trivial filter from find_sum_matches
                                if v <= 1 and len(combo) <= 1:
                                    continue
                                sum_idx = sub_flat_order.get(sub_cord.cord_id, 0)
                                summand_mean = round(
                                    sum(sub_flat_order.get(s.cord_id, 0)
                                        for s in combo) / len(combo))
                                handedness = -(sum_idx - summand_mean)
                                raw_matches.append(SummationMatch(
                                    pattern_type='indexed_subsidiary_sum',
                                    sum_cord=sub_cord,
                                    summand_cords=list(combo),
                                    expected_sum=v,
                                    actual_sum=test_sum,
                                    matches=True,
                                    handedness=float(handedness),
                                    notes=f'color={color} pos={pos} {side_label}'
                                ))

        def _iss_trivial(m: SummationMatch) -> bool:
            v = m.sum_cord.value or 0
            return (
                v <= len(m.summand_cords)
                or v < 5
                or len(m.summand_cords) < 2
            )

        return self._dedup_shortest_per_cord(raw_matches, is_trivial=_iss_trivial)


    # ------------------------------------------------------------------
    # Pattern X: pendant_sub_neighbor
    @time_pattern_detection
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
    @time_pattern_detection
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
            if c.group_idx is None or c.value is None:
                continue
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
            current: Optional[str] = c.parent_cord
            visited: set = set()
            while current and current not in visited:
                visited.add(current)
                ancestor = cord_map_gg.get(current)
                if ancestor is None or getattr(ancestor, 'hierarchy_level', 0) == 0:
                    break
                current = ancestor.parent_cord
            parent_grp = cord_to_group.get(current) if current else None
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
    @time_pattern_detection
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
            ordered = sorted(members, key=lambda c: c.position_in_group or 0)
            vals: list[int] = [c.value or 0 for c in ordered]
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
    @time_pattern_detection
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
            ordered = sorted(members, key=lambda c: c.position_in_group or 0)
            vals: list[int] = [c.value or 0 for c in ordered]
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
    
    def get_timing_stats(self) -> Dict[str, float]:
        """Return timing data for pattern detection methods.
        
        Returns dict mapping pattern name to elapsed seconds.
        Only available when enable_timing=True in constructor.
        """
        if not hasattr(self, '_timing_data'):
            return {}
        return dict(self._timing_data)
    
    def reset_timing(self):
        """Clear accumulated timing data."""
        self._timing_data = {}
    
    def analyze_handedness(self, results: Dict[str, List[SummationMatch]]) -> Dict[str, Dict]:
        """Analyze handedness distribution for patterns that support it.
        
        Uses KFG convention: handedness < 0 = left, handedness >= 0 = right.
        
        Returns dict mapping pattern_type to:
            - num_left: count of left-handed relationships
            - num_right: count of right-handed relationships
            - handedness_ratio: (right - left) / total
            - is_asymmetric: True if |ratio| > 0.2
        """
        handedness_stats = {}
        
        for pattern_type, matches in results.items():
            if not matches:
                continue
            
            left_count = sum(1 for m in matches if m.handedness is not None and m.handedness < 0)
            right_count = sum(1 for m in matches if m.handedness is not None and m.handedness >= 0)
            total = left_count + right_count
            
            if total == 0:
                continue
            
            ratio = (right_count - left_count) / total
            
            handedness_stats[pattern_type] = {
                'num_left': left_count,
                'num_right': right_count,
                'total': total,
                'handedness_ratio': ratio,
                'is_asymmetric': abs(ratio) > 0.2
            }
        
        return handedness_stats
    
    def analyze_dual_sums(self, results: Dict[str, List[SummationMatch]]) -> Dict[str, Dict]:
        """Analyze dual sum patterns (cords with multiple summand windows).
        
        Returns dict mapping pattern_type to:
            - num_dual_sums: count of sum cords with multiple windows
            - dual_sum_cords: list of cord names
            - total_sum_cords: total unique sum cords
            - dual_sum_rate: fraction of sum cords that are dual
        """
        dual_sum_stats = {}
        
        for pattern_type, matches in results.items():
            if not matches:
                continue
            
            # Group by sum cord using cord_id (globally unique DB row id)
            sum_cord_matches: Dict[int, List[SummationMatch]] = defaultdict(list)
            for m in matches:
                sum_cord_matches[m.sum_cord.cord_id].append(m)
            
            dual_sum_ids = [cid for cid, ms in sum_cord_matches.items() 
                           if ms[0].is_dual_sum]
            
            if dual_sum_ids:
                dual_sum_stats[pattern_type] = {
                    'num_dual_sums': len(dual_sum_ids),
                    'dual_sum_cord_ids': sorted(dual_sum_ids),
                    'total_sum_cords': len(sum_cord_matches),
                    'dual_sum_rate': len(dual_sum_ids) / len(sum_cord_matches)
                }
        
        return dual_sum_stats
    
    def analyze_figure8_markers(self, results: Dict[str, List[SummationMatch]]) -> Dict[str, Dict]:
        """Analyze figure-8 knot proximity to summation relationships.
        
        Returns dict mapping pattern_type to:
            - num_with_figure8: count of relationships near figure-8 knots
            - figure8_rate: fraction with figure-8 markers
            - locations: distribution of where figure-8s appear
        """
        figure8_stats = {}
        
        for pattern_type, matches in results.items():
            if not matches:
                continue
            
            with_figure8 = [m for m in matches if m.figure8_proximity is not None]
            
            if with_figure8:
                location_counts = defaultdict(int)
                for m in with_figure8:
                    if m.figure8_proximity:
                        location_counts[m.figure8_proximity['location']] += 1
                
                figure8_stats[pattern_type] = {
                    'num_with_figure8': len(with_figure8),
                    'total_relationships': len(matches),
                    'figure8_rate': len(with_figure8) / len(matches),
                    'locations': dict(location_counts)
                }
        
        return figure8_stats

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
