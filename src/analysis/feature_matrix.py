"""
Per-khipu feature matrix for Phase 3 structural typology.

Builds a flat DataFrame with one row per K-CAT khipu containing:
  - 9 binary pattern flags (from KFGSummationDetector)
  - 3 pattern-count columns (raw match counts for PP, IP, CP as continuous signal)
  - Structural features derived from the cords table
  - Metadata fields from khipu_metadata for cluster enrichment

Usage
-----
    from src.analysis.feature_matrix import build_feature_matrix

    df = build_feature_matrix("data/kfg/khipu_database.db")
    # df.shape = (709, ~30 columns)

Column catalogue
----------------
Binary pattern flags (0/1):
    has_pp, has_ip, has_cp, has_sp, has_gg, has_gsb, has_is, has_psn, has_adg

Pattern counts (raw integer, useful as continuous features):
    n_pp, n_ip, n_cp       — num_sum_cords for PP, IP, CP respectively

Structural:
    n_cords               — total cord count
    n_pendants            — hierarchy_level == 1 cords
    n_subsidiaries        — hierarchy_level >= 2 cords
    n_groups              — distinct group_idx values (excludes None)
    numeric_coverage      — fraction of cords with value > 0
    frac_broken           — fraction of cords with termination == 'B'
    n_colors              — distinct primary color codes
    n_pattern_types       — count of has_* columns that equal 1

Metadata (string / nullable — geographic provenance only, NOT museum location):
    region            — KFG region grouping (e.g. "Central Coast, Peru", "Chachapoyas")
    provenance_display — cleaned site name from provenance_labels table
                         (many are "Unknown"; do not use museum fields as provenance proxy)
    creation_date
    geo_zone          — consolidated geographic zone derived from provenance_display:
                         "Central Coast" | "Cañete–Pisco" | "Ica & Paracas" |
                         "Nazca & Far South" | "Chachapoyas" | "North Peru Coast" |
                         "Arica & N. Chile" | "Southern Highlands"
                         None for Unknown, collection names, and unresolvable labels
"""

import sqlite3
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Allow running from repo root or from src/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.kfg_summation_detector import KFGSummationDetector

# Pattern keys in canonical order
PATTERN_KEYS = ["pp", "ip", "cp", "sp", "gg", "gsb", "is", "psn", "adg"]
BINARY_COLS  = [f"has_{k}" for k in PATTERN_KEYS]

# Map from pattern key -> detector ptype string
_PTYPE = {
    "pp":  "pendant_pendant_sum",
    "ip":  "indexed_pendant_sum",
    "cp":  "colored_pendant_sum",
    "sp":  "subsidiary_pendant_sum",
    "gg":  "group_group_sum",
    "gsb": "group_sum_bands",
    "is":  "indexed_subsidiary_sum",
    "psn": "pendant_sub_neighbor",
    "adg": "ascher_decreasing_group",
}

# Significance thresholds (> thresh counts as positive), matching reconciler
_THRESH = {k: 0 for k in PATTERN_KEYS}
_THRESH["is"]  = 1
_THRESH["psn"] = 1

# ---------------------------------------------------------------------------
# Geographic zone consolidation
# ---------------------------------------------------------------------------
# Maps provenance_display -> 8-zone label.  Entries not present in this dict
# (collections, "Unknown", or ambiguous cross-site labels) receive None.
GEO_ZONE_MAP: dict = {
    # ── Central Coast (Lima + Huacho / Chancay merged) ──────────────────────
    "Ancon (prob.)":                        "Central Coast",
    "Armatambo (Huaca San Pedro)":           "Central Coast",
    "Armatambo / Huaca San Pedro":           "Central Coast",
    "Armatambo, Lima":                       "Central Coast",
    "Cajamarquilla":                         "Central Coast",
    "Central Coast (Late Period, prob.)":    "Central Coast",
    "Cerro Solar (foothills)":               "Central Coast",
    "Chancay":                               "Central Coast",
    "Chancay (Central Coast)":               "Central Coast",
    "Chancay (Hda. Huando)":                 "Central Coast",
    "Chancay / Huando (Gaffron)":            "Central Coast",
    "Chuquitanta":                           "Central Coast",
    "Cieneguilla (Lurin Valley)":            "Central Coast",
    "Huaca San Marco":                       "Central Coast",
    "Huacho":                                "Central Coast",
    "Huacho (?)":                            "Central Coast",
    "Huacho (Central Coast)":                "Central Coast",
    "Huacho / Pachacamac":                   "Central Coast",
    "Huaquerones":                           "Central Coast",
    "La Molina":                             "Central Coast",
    "Lima":                                  "Central Coast",
    "Lima (Huaca Pérez)":                    "Central Coast",
    "Lima (Maranga, Huaca 1)":               "Central Coast",
    "Lima (Pueblo Libre)":                   "Central Coast",
    "Marquez":                               "Central Coast",
    "Near Lima":                             "Central Coast",
    "Near Lima (prob.)":                     "Central Coast",
    "Pachacamac":                            "Central Coast",
    "Pachacamac (Casa de los Quipus)":       "Central Coast",
    "Pachacamac (Casa del Quipu)":           "Central Coast",
    "Pachacamac (Fundort)":                  "Central Coast",
    "Pachacamac (Nordenskiöld)":             "Central Coast",
    "Purucucho":                             "Central Coast",
    "Rimac Valley":                          "Central Coast",
    # ── Cañete – Pisco (~150–350 km S of Lima) ──────────────────────────────
    "Hda. Ullujalla / Callengo":             "Cañete–Pisco",
    "Huacones":                              "Cañete–Pisco",
    "Incahuasi":                             "Cañete–Pisco",
    "La Centinela / Tambo de Mora":          "Cañete–Pisco",
    "La Puntilla (Paracas/Pisco)":           "Cañete–Pisco",
    "Pisco":                                 "Cañete–Pisco",
    "Pisco Valley":                          "Cañete–Pisco",
    "Tambo Colorado":                        "Cañete–Pisco",
    # ── Ica Valley & Paracas (~300–400 km S of Lima) ─────────────────────────
    "Atarco":                                "Ica & Paracas",
    "Between Ica and Pisco":                 "Ica & Paracas",
    "Ica":                                   "Ica & Paracas",
    "Ica (Coast)":                           "Ica & Paracas",
    "Ica (Site T, Grave M — Uhle)":          "Ica & Paracas",
    "Ica / Cajamarquilla":                   "Ica & Paracas",
    "Ica / Pisco":                           "Ica & Paracas",
    "Ica Valley (Hda. Callango / Ocucaje)":  "Ica & Paracas",
    "Ica Valley (Rancho San Juan)":          "Ica & Paracas",
    "Ica Valley (Site T, Grave K)":          "Ica & Paracas",
    "Ica Valley (Site T, Grave M)":          "Ica & Paracas",
    "Ica Valley (near Callengo)":            "Ica & Paracas",
    "Ica Valley (near Callango)":            "Ica & Paracas",
    "Ocucaje":                               "Ica & Paracas",
    "Ocucaje / Ullujaya (Ica)":              "Ica & Paracas",
    "Paracas":                               "Ica & Paracas",
    # ── Nazca & Far South (~400–600 km S of Lima) ────────────────────────────
    "Acari":                                 "Nazca & Far South",
    "Chala":                                 "Nazca & Far South",
    "Nazca":                                 "Nazca & Far South",
    "Nazca (Hda. Copara)":                   "Nazca & Far South",
    "Nazca (Monte de Cacatilla)":            "Nazca & Far South",
    "Nazca (Santa Clara)":                   "Nazca & Far South",
    "South Coast":                           "Nazca & Far South",
    "South Peru":                            "Nazca & Far South",
    "Southern Coast":                        "Nazca & Far South",
    # ── Chachapoyas (northern highlands, ~700 km N of Lima) ──────────────────
    "Leymebamba":                            "Chachapoyas",
    "Mollepampa":                            "Chachapoyas",
    # ── North Peru Coast (~400–700 km N of Lima) ─────────────────────────────
    "Pacasmayo":                             "North Peru Coast",
    "Santa":                                 "North Peru Coast",
    # ── Arica & Northern Chile (outside Peru) ────────────────────────────────
    "Arica, Chile (Playa Miller 6)":         "Arica & N. Chile",
    "Lluta Valley":                          "Arica & N. Chile",
    "Quillagua, Valle de Loa":               "Arica & N. Chile",
    # ── Southern Highlands (Cusco / Ayacucho) ────────────────────────────────
    "Cuzco":                                 "Southern Highlands",
    "Huari":                                 "Southern Highlands",
    # Excluded from GEO_ZONE_MAP (→ None):
    #   Gaffron Collection, Gaffron Estate, Belli Collection, Goodspeed Collection,
    #   Stanford Collection (prob. 1905), Aankoop, Peru (unknown), Nazca / Ancon,
    #   Unknown, Unknown (non-Gaffron)
}


def _apply_geo_zone(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a geo_zone column by mapping provenance_display through GEO_ZONE_MAP.
    Rows not in the map (unknowns, collections, ambiguous) receive None."""
    df = df.copy()
    df["geo_zone"] = df["provenance_display"].map(GEO_ZONE_MAP)
    return df


def _primary_color(color_str: Optional[str]) -> Optional[str]:
    """Extract the dominant component from a compound color code."""
    if not color_str:
        return None
    return color_str.split(":")[0].split("-")[0].strip()


def _load_structural_features(db_path: str) -> pd.DataFrame:
    """Load per-khipu structural statistics from the cords table."""
    conn = sqlite3.connect(db_path)
    cords = pd.read_sql(
        """
        SELECT kfg_id, cord_id, hierarchy_level, termination,
               value, color, group_idx
        FROM cords
        """,
        conn,
    )
    conn.close()

    # Numeric coverage: fraction with value > 0
    cords["has_value"] = (cords["value"].fillna(0) > 0).astype(int)
    cords["is_broken"] = (cords["termination"] == "B").astype(int)
    cords["primary_color"] = cords["color"].apply(_primary_color)

    stats = (
        cords.groupby("kfg_id")
        .agg(
            n_cords=("cord_id", "count"),
            n_pendants=("hierarchy_level", lambda s: (s == 1).sum()),
            n_subsidiaries=("hierarchy_level", lambda s: (s >= 2).sum()),
            n_groups=("group_idx", lambda s: s.dropna().nunique()),
            numeric_coverage=("has_value", "mean"),
            frac_broken=("is_broken", "mean"),
            n_colors=("primary_color", lambda s: s.dropna().nunique()),
        )
        .reset_index()
    )
    return stats


def _load_metadata(db_path: str) -> pd.DataFrame:
    """Load origin provenance for enrichment.

    Uses region (KFG grouping) and provenance_labels (site-level display name).
    museum_country / museum_name are deliberately excluded — they record where
    an object is currently held, not where it was made or used.
    """
    conn = sqlite3.connect(db_path)
    meta = pd.read_sql(
        """
        SELECT kfg_id, region, provenance, creation_date
        FROM khipu_metadata
        """,
        conn,
    )
    labels = pd.read_sql("SELECT raw, display_name FROM provenance_labels", conn)
    conn.close()

    # Normalise join key: strip whitespace and collapse internal spaces
    def _norm(s):
        if not isinstance(s, str):
            return ""
        return " ".join(s.strip().split())

    meta["_prov_key"]   = meta["provenance"].apply(_norm)
    labels["_prov_key"] = labels["raw"].apply(_norm)
    # Keep only first match per raw string (table may have duplicates for same site)
    labels_dedup = labels.drop_duplicates("_prov_key")[["_prov_key", "display_name"]]

    meta = meta.merge(labels_dedup, on="_prov_key", how="left")
    meta["provenance_display"] = meta["display_name"].fillna(
        meta["provenance"].apply(lambda v: v if isinstance(v, str) and v.strip() not in ("", "Unknown") else None)
    )
    return meta.drop(columns=["provenance", "_prov_key", "display_name"])


def build_feature_matrix(
    db_path: str,
    tolerance: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the complete per-khipu feature matrix.

    Parameters
    ----------
    db_path : str
        Path to K-CAT SQLite database.
    tolerance : int
        Arithmetic tolerance passed to KFGSummationDetector.summarize().
        Default 0 (exact integer match, matching Phase 2 methodology).
    verbose : bool
        Print progress every 100 khipus.

    Returns
    -------
    pd.DataFrame
        One row per khipu, columns as described in the module docstring.
    """
    detector = KFGSummationDetector(db_path)

    conn = sqlite3.connect(db_path)
    khipu_ids = pd.read_sql(
        "SELECT kfg_id FROM khipu_metadata ORDER BY kfg_id", conn
    )["kfg_id"].tolist()
    conn.close()

    records = []
    for i, kid in enumerate(khipu_ids):
        if verbose and i % 100 == 0:
            print(f"  [{i}/{len(khipu_ids)}] {kid}")

        summary = detector.summarize(kid, tolerance=tolerance)
        pstats = summary.get("pattern_stats", {})

        row = {"kfg_id": kid}

        # Binary flags using per-pattern significance thresholds
        for key in PATTERN_KEYS:
            ptype = _PTYPE[key]
            thresh = _THRESH[key]
            matches = pstats.get(ptype, {}).get("matches", 0)
            row[f"has_{key}"] = int(matches > thresh)

        # Raw match counts for the three highest-volume continuous signals
        for key in ("pp", "ip", "cp"):
            ptype = _PTYPE[key]
            row[f"n_{key}"] = pstats.get(ptype, {}).get("matches", 0)

        row["n_pattern_types"] = sum(row[f"has_{k}"] for k in PATTERN_KEYS)
        records.append(row)

    pattern_df = pd.DataFrame(records)

    structural_df = _load_structural_features(db_path)
    metadata_df = _load_metadata(db_path)

    df = (
        pattern_df
        .merge(structural_df, on="kfg_id", how="left")
        .merge(metadata_df,   on="kfg_id", how="left")
    )

    df = _apply_geo_zone(df)

    if verbose:
        n = len(df)
        n_any = df[[f"has_{k}" for k in PATTERN_KEYS]].any(axis=1).sum()
        print(f"\nFeature matrix: {n} khipus, {len(df.columns)} columns")
        print(f"  With any pattern: {n_any} ({100*n_any/n:.1f}%)")
        print(f"  Pattern-type distribution:")
        print(df["n_pattern_types"].value_counts().sort_index().to_string())

    return df


if __name__ == "__main__":
    import sys

    db = sys.argv[1] if len(sys.argv) > 1 else "data/kfg/khipu_database.db"
    df = build_feature_matrix(db)
    out = Path("data/processed/phase3_feature_matrix.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
