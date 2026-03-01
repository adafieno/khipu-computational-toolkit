"""
KFG Relation Loader — Ground-Truth Cord Annotations

Replaces algorithmic re-detection for known KFG khipus by loading the
authoritative `*_relation.csv` files from the KFG checks directory.

Architecture
------------
For the 703 khipus in the KFG corpus, every summation relationship has
already been identified by the KFG author.  The `*_relation.csv` files
record each relationship at cord or group level with full detail.  Re-running
the pattern detector on these khipus only introduces noise (false positives
from the approximation algorithm vs. the expert annotations).

Two components are provided:

1.  KFGRelationLoader
        Loads all nine `*_relation.csv` files and exposes methods to retrieve
        canonical SummationMatch objects for any known KFG khipu.

2.  apply_exclusivity(results)
        Enforces mutual exclusivity at the *cord* level: once a cord is
        claimed by a higher-specificity pattern, the same cord is removed
        from lower-specificity pattern results.  Group-level patterns (GG,
        GSB, ADG, PSN) operate on groups, not individual cords, so they are
        not filtered.

Exclusivity hierarchy (most → least specific):
    indexed_subsidiary_sum   IS  subsidiary + color-indexed cross-group
    subsidiary_pendant_sum   SP  subsidiary summing pendants
    indexed_pendant_sum      IP  pendant position-indexed cross-group
    colored_pendant_sum      CP  pendant color-indexed cross-group
    pendant_pendant_sum      PP  pendant contiguous window (most general)

Group-level (no cord-level exclusion):
    group_group_sum          GG  equal group totals
    group_sum_bands          GSB equal band halves within a group
    ascher_decreasing_group  ADG monotone decreasing group
    pendant_sub_neighbor     PSN difference relation (rare, likely fluke)

Usage
-----
    from src.analysis.kfg_relation_loader import KFGRelationLoader, apply_exclusivity

    loader = KFGRelationLoader("data/kfg/KFG/KFG/checks")

    # Check if a khipu is in the KFG corpus
    if loader.in_kfg("KH0049"):
        cords = detector._load_all_cords("KH0049")
        results = loader.build_all_matches("KH0049", cords)
        # results["pendant_pendant_sum"] → List[SummationMatch]

    # For non-KFG khipus, fall back to the algorithmic detector:
    else:
        results = detector.detect_all_patterns("MY_KHIPU")
        results = apply_exclusivity(results)   # always enforce exclusivity
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Import the data-structures defined in the detector so that both paths
# (ground-truth and algorithmic) return identically-typed objects.
from src.analysis.kfg_summation_detector import Cord, SummationMatch


# ---------------------------------------------------------------------------
# Exclusivity constants
# ---------------------------------------------------------------------------

# Cord-level patterns in priority order (most specific → least specific).
CORD_EXCLUSIVITY_ORDER: List[str] = [
    "indexed_subsidiary_sum",   # subsidiary summing color-indexed subs
    "subsidiary_pendant_sum",   # subsidiary summing pendant window
    "indexed_pendant_sum",      # pendant summing position-matched cross-group
    "colored_pendant_sum",      # pendant summing color-matched cross-group
    "pendant_pendant_sum",      # pendant summing contiguous window
]

# Group-level patterns that are *not* subject to cord-level exclusivity.
GROUP_LEVEL_PATTERNS: List[str] = [
    "group_group_sum",
    "group_sum_bands",
    "ascher_decreasing_group",
    "pendant_sub_neighbor",
]

ALL_PATTERNS: List[str] = CORD_EXCLUSIVITY_ORDER + GROUP_LEVEL_PATTERNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_summand_coords(summand_string: str) -> List[tuple]:
    """
    Parse a summand_string such as
        "GG@[6, 2, 1]:1 + MB@[8, 2, 1]:2 + MB@[10, 2, 1]:3"
    into a list of (group_idx, position_in_group, sub_level) tuples.

    For pendant-level entries (2-element index) sub_level is 0.
    """
    if not isinstance(summand_string, str):
        return []
    coords = []
    # Match both 2-element [g, p] and 3-element [g, p, s] index notations.
    for m in re.finditer(r'@\[(\d+),\s*(\d+)(?:,\s*(\d+))?\]', summand_string):
        g = int(m.group(1))
        p = int(m.group(2))
        s = int(m.group(3)) if m.group(3) is not None else 0
        coords.append((g, p, s))
    return coords


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class KFGRelationLoader:
    """
    Canonical pattern annotations loaded from the KFG `*_relation.csv` files.

    Parameters
    ----------
    checks_dir : str | Path
        Path to the directory that contains the nine `*_relation.csv` files,
        e.g. ``"data/kfg/KFG/KFG/checks"``.
    """

    # Map pattern_type → (relation_csv_name, relation_kind)
    _PATTERN_META: Dict[str, tuple] = {
        "pendant_pendant_sum":     ("pendant_pendant_sum_relation.csv",     "cord_sum"),
        "colored_pendant_sum":     ("colored_pendant_sum_relation.csv",     "cord_sum"),
        "indexed_pendant_sum":     ("indexed_pendant_sum_relation.csv",     "cord_sum"),
        "subsidiary_pendant_sum":  ("subsidiary_pendant_sum_relation.csv",  "cord_sum"),
        "indexed_subsidiary_sum":  ("indexed_subsidiary_sum_relation.csv",  "cord_sum"),
        "pendant_sub_neighbor":    ("pendant_sub_neighbor_relation.csv",    "psn"),
        "group_group_sum":         ("group_group_sum_relation.csv",         "group_pair"),
        "group_sum_bands":         ("group_sum_bands_relation.csv",         "group_band"),
        "ascher_decreasing_group": ("ascher_decreasing_group_relation.csv", "group_decrease"),
    }

    def __init__(self, checks_dir: str | Path) -> None:
        self.checks_dir = Path(checks_dir)
        self._dfs: Dict[str, pd.DataFrame] = {}
        self._known_cache: Optional[set] = None
        self._load_all()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        for ptype, (fname, _) in self._PATTERN_META.items():
            path = self.checks_dir / fname
            if path.exists():
                self._dfs[ptype] = pd.read_csv(path)

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------

    @property
    def known_khipus(self) -> set:
        """Set of all kfg_name values that appear in at least one relation CSV."""
        if self._known_cache is None:
            s: set = set()
            for df in self._dfs.values():
                s.update(df["kfg_name"].unique())
            self._known_cache = s
        return self._known_cache

    def in_kfg(self, kfg_id: str) -> bool:
        """True if *kfg_id* has ground-truth relation data in the checks CSVs."""
        return kfg_id in self.known_khipus

    # ------------------------------------------------------------------
    # Raw data access
    # ------------------------------------------------------------------

    def get_relations(self, pattern: str, kfg_id: str) -> pd.DataFrame:
        """Return the raw relation rows for *kfg_id* under *pattern*."""
        df = self._dfs.get(pattern)
        if df is None or df.empty:
            return pd.DataFrame()
        return df[df["kfg_name"] == kfg_id].copy()

    def get_sum_cord_names(self, kfg_id: str, pattern: str) -> set:
        """
        For cord-level patterns (PP, CP, IP, SP, IS): return the set of
        ``cord_name`` values that the KFG identifies as sum cords.

        Returns an empty set for group-level patterns.
        """
        kind = self._PATTERN_META.get(pattern, (None, None))[1]
        if kind not in ("cord_sum",):
            return set()
        rows = self.get_relations(pattern, kfg_id)
        if rows.empty or "cord_name" not in rows.columns:
            return set()
        return set(rows["cord_name"].dropna())

    # ------------------------------------------------------------------
    # SummationMatch construction
    # ------------------------------------------------------------------

    def build_matches(
        self,
        kfg_id: str,
        pattern: str,
        cord_map: Dict[str, Cord],
        coord_lookup: Optional[Dict[tuple, Cord]] = None,
    ) -> List[SummationMatch]:
        """
        Construct SummationMatch objects for *kfg_id*/*pattern* from the CSV.

        Parameters
        ----------
        kfg_id     : the khipu identifier
        pattern    : one of the nine pattern type strings
        cord_map   : ``cord_name → Cord`` from :meth:`_build_cord_map`
        coord_lookup : optional ``(group_idx, position_in_group, sub_level) → Cord``
                       used to resolve summands from summand_string coordinates.
                       If None, summand_cords will be empty lists.
        """
        meta = self._PATTERN_META.get(pattern)
        if meta is None:
            return []
        kind = meta[1]
        rows = self.get_relations(pattern, kfg_id)
        if rows.empty:
            return []

        matches: List[SummationMatch] = []

        if kind == "cord_sum":
            # Every row is one sum cord with a summand_string.
            for _, row in rows.iterrows():
                cord_name = row.get("cord_name")
                cord = cord_map.get(cord_name)
                if cord is None:
                    continue

                # Optionally resolve summands
                summand_cords: List[Cord] = []
                if coord_lookup is not None:
                    ss = row.get("summand_string", "")
                    for (g, p, s) in _parse_summand_coords(ss):
                        summand = coord_lookup.get((g, p, s))
                        if summand is not None:
                            summand_cords.append(summand)

                val = cord.value if cord.value is not None else 0
                matches.append(SummationMatch(
                    pattern_type=pattern,
                    sum_cord=cord,
                    summand_cords=summand_cords,
                    expected_sum=val,
                    actual_sum=val,
                    matches=True,
                    notes="kfg_ground_truth",
                ))

        elif kind == "psn":
            # pendant_sub_neighbor: (pendant_sub_name, neighbor_name) pair
            for _, row in rows.iterrows():
                sub_name  = row.get("pendant_sub_name")
                nbr_name  = row.get("neighbor_name")
                cord      = cord_map.get(sub_name)
                neighbor  = cord_map.get(nbr_name)
                if cord is None:
                    continue
                nbr_val = int(row.get("neighbor_value", 0))
                matches.append(SummationMatch(
                    pattern_type=pattern,
                    sum_cord=cord,
                    summand_cords=[neighbor] if neighbor is not None else [],
                    expected_sum=nbr_val,
                    actual_sum=nbr_val,
                    matches=True,
                    notes="kfg_ground_truth",
                ))

        elif kind == "group_pair":
            # group_group_sum: left/right group pair — no single sum cord.
            # Represent using the first cord of the left group as proxy.
            for _, row in rows.iterrows():
                g_left = int(row.get("left_group_index", -1))
                rep = _first_cord_in_group(cord_map, g_left)
                if rep is None:
                    continue
                g_sum = int(row.get("group_sum", 0))
                matches.append(SummationMatch(
                    pattern_type=pattern,
                    sum_cord=rep,
                    summand_cords=[],
                    expected_sum=g_sum,
                    actual_sum=g_sum,
                    matches=True,
                    notes="kfg_ground_truth|group_level",
                ))

        elif kind == "group_band":
            # group_sum_bands: a single group with split halves equal.
            for _, row in rows.iterrows():
                g_idx = int(row.get("group_index", -1))
                rep = _first_cord_in_group(cord_map, g_idx)
                if rep is None:
                    continue
                split_sum = int(row.get("split_sum", 0))
                matches.append(SummationMatch(
                    pattern_type=pattern,
                    sum_cord=rep,
                    summand_cords=[],
                    expected_sum=split_sum,
                    actual_sum=split_sum,
                    matches=True,
                    notes="kfg_ground_truth|group_level",
                ))

        elif kind == "group_decrease":
            # ascher_decreasing_group: decreasing values along a group.
            for _, row in rows.iterrows():
                g_idx = int(row.get("group_index", -1))
                rep = _first_cord_in_group(cord_map, g_idx)
                if rep is None:
                    continue
                matches.append(SummationMatch(
                    pattern_type=pattern,
                    sum_cord=rep,
                    summand_cords=[],
                    expected_sum=0,
                    actual_sum=0,
                    matches=True,
                    notes="kfg_ground_truth|group_level",
                ))

        return matches

    def build_all_matches(
        self,
        kfg_id: str,
        cords: List[Cord],
        apply_excl: bool = False,
        resolve_summands: bool = True,
    ) -> Dict[str, List[SummationMatch]]:
        """
        Build all nine pattern match lists for *kfg_id* from ground-truth data.

        Parameters
        ----------
        kfg_id          : khipu identifier (must be in the KFG corpus)
        cords           : list of Cord objects from the database
        apply_excl      : if True, enforce mutual exclusivity at cord level
                          (IS > SP > IP > CP > PP priority order).

                          **Default is False** because the KFG ground-truth
                          relation CSVs intentionally record the same cord in
                          multiple pattern tables — a cord can simultaneously
                          be a PP sum cord, an IP sum cord, and a CP sum cord,
                          each with a different set of summand cords.  Applying
                          exclusivity removes those valid relationships and
                          reduces agreement from 99.4% to 98.6%.

                          Set apply_excl=True only when you need a SINGLE
                          primary pattern label per cord (classification use).
        resolve_summands: if True, try to populate summand_cords from
                          coordinate lookups (slightly slower)
        """
        cord_map = _build_cord_map(cords)
        coord_lut = _build_coord_lookup(cords) if resolve_summands else None

        results: Dict[str, List[SummationMatch]] = {}
        for ptype in ALL_PATTERNS:
            results[ptype] = self.build_matches(kfg_id, ptype, cord_map, coord_lut)

        if apply_excl:
            results = apply_exclusivity(results)

        return results


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def _build_cord_map(cords: List[Cord]) -> Dict[str, Cord]:
    """cord_name → Cord lookup."""
    return {c.cord_name: c for c in cords if c.cord_name}


def _build_coord_lookup(cords: List[Cord]) -> Dict[tuple, Cord]:
    """
    (group_idx, position_in_group, hierarchy_level) → Cord.

    hierarchy_level == 0 for pendants, 1 for subsidiaries.
    For subsidiaries the third tuple element is their position_in_group value
    (which corresponds to the sub_position in the summand_string coordinate).
    """
    lut: Dict[tuple, Cord] = {}
    for c in cords:
        if c.group_idx is not None and c.position_in_group is not None:
            key = (c.group_idx, c.position_in_group, c.hierarchy_level)
            lut[key] = c
    return lut


def _first_cord_in_group(cord_map: Dict[str, Cord], group_idx: int) -> Optional[Cord]:
    """Return the first (lowest position_in_group) cord in *group_idx*."""
    candidates = [
        c for c in cord_map.values()
        if c.group_idx == group_idx and c.hierarchy_level == 0
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda c: c.position_in_group or 0)


def apply_exclusivity(
    results: Dict[str, List[SummationMatch]],
) -> Dict[str, List[SummationMatch]]:
    """
    Enforce mutual exclusivity at the cord level.

    A cord (identified by ``sum_cord.cord_name``) can only be the sum cord of
    ONE pattern type.  When the same cord appears in multiple patterns, only
    the highest-specificity pattern retains the match — lower-specificity
    patterns have that cord's match removed.

    Priority (highest first):
        IS → SP → IP → CP → PP

    Group-level patterns (GG, GSB, ADG, PSN) are not filtered because they
    annotate groups, not individual cords.

    Parameters
    ----------
    results : dict mapping pattern_type → List[SummationMatch]

    Returns
    -------
    New dict with the same keys but cord-exclusive match lists.
    """
    claimed: set = set()
    out: Dict[str, List[SummationMatch]] = {}

    # Apply exclusivity only to cord-level patterns, in priority order.
    for ptype in CORD_EXCLUSIVITY_ORDER:
        matches = results.get(ptype, [])
        filtered = [m for m in matches if m.sum_cord.cord_name not in claimed]
        claimed.update(m.sum_cord.cord_name for m in filtered)
        out[ptype] = filtered

    # Group-level patterns pass through unchanged.
    for ptype in GROUP_LEVEL_PATTERNS:
        out[ptype] = results.get(ptype, [])

    return out
