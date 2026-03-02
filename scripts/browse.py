"""
Khipu Explorer — Standalone Local Browser
==========================================
Interactive Streamlit app for exploring the KFG khipu database.

Views:
    - Corpus Browser  : filterable / sortable table of all 709 khipus
    - Analytics       : 4-tab analytics dashboard
                          Overview   — prevalence · co-occurrence · complexity
                          Deep Dive  — handedness · counts · magnitudes · dual/multi
                          Geography  — provenance × pattern heatmap
                          Pattern Space — PCA scatter · detail table
    - 3D Viewer       : interactive Plotly 3D cord structure for a selected khipu
    - X-Ray View      : cord group color map + summation arc overlays (PP/IP/CP/SP/IS)

Usage:
    streamlit run scripts/browse.py

Requirements (already in requirements.txt):
    pip install streamlit plotly pandas numpy
"""

import ast
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sqlite3
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent.parent
DB_PATH     = ROOT / "data" / "kfg" / "khipu_database.db"
CHECKS_PATH = ROOT / "data" / "kfg" / "KFG" / "KFG" / "checks"

sys.path.insert(0, str(ROOT / "src"))
try:
    from analysis.kfg_relation_loader import KFGRelationLoader as _KFGLoader
    _LOADER_INSTANCE: Optional["_KFGLoader"] = None

    def _get_loader() -> Optional["_KFGLoader"]:
        global _LOADER_INSTANCE
        if _LOADER_INSTANCE is None and CHECKS_PATH.exists():
            _LOADER_INSTANCE = _KFGLoader(str(CHECKS_PATH))
        return _LOADER_INSTANCE
except ImportError:
    def _get_loader():  # type: ignore[misc]
        return None

# ── Per-pattern config ─────────────────────────────────────────────────────────
# Each entry: (short_key, display_name, csv_filename, positive_col)
PATTERN_CONFIG = [
    ("pp",  "PP — Pendant-Pendant Sum",       "pendant_pendant_sum.csv",       "num_sum_cords"),
    ("ip",  "IP — Indexed Pendant Sum",        "indexed_pendant_sum.csv",        "num_sum_cords"),
    ("cp",  "CP — Colored Pendant Sum",        "colored_pendant_sum.csv",        "num_sum_cords"),
    ("sp",  "SP — Subsidiary Pendant Sum",     "subsidiary_pendant_sum.csv",     "num_sum_cords"),
    ("is",  "IS — Indexed Subsidiary Sum",     "indexed_subsidiary_sum.csv",     "num_sum_cords"),
    ("gg",  "GG — Group-Group Sum",            "group_group_sum.csv",            "num_sum_groups"),
    ("gsb", "GSB — Group Sum Bands",           "group_sum_bands.csv",            "num_group_sum_bands"),
    ("adg", "ADG — Ascher Decreasing Groups",  "ascher_decreasing_group.csv",    "num_decreasing_groups"),
    ("psn", "PSN — Pendant-Sub-Neighbor",      "pendant_sub_neighbor.csv",       "num_pendant_sub_neighbor_groups"),
]

# Cord-level patterns that have relation CSVs with cord_index + summand_string
ARC_PATTERNS = {
    "pendant_pendant_sum":    ("#3b82f6", "PP"),
    "indexed_pendant_sum":    ("#f97316", "IP"),
    "colored_pendant_sum":    ("#22c55e", "CP"),
    "subsidiary_pendant_sum": ("#a855f7", "SP"),
    "indexed_subsidiary_sum": ("#f43f5e", "IS"),
}

# ── Ascher color map ───────────────────────────────────────────────────────────

_COLOR_MAP: dict[str, str] = {
    "W":   "#F5F5F5",   # White
    "MB":  "#8B4513",   # Medium Brown
    "KB":  "#A0522D",   # Khaki Brown
    "DB":  "#5C3317",   # Dark Brown
    "LB":  "#C8A882",   # Light Brown
    "B":   "#8B6914",   # Brown
    "GG":  "#228B22",   # Green
    "LG":  "#90EE90",   # Light Green
    "YG":  "#9ACD32",   # Yellow-Green
    "Y":   "#FFD700",   # Yellow
    "R":   "#DC143C",   # Red
    "LR":  "#FF6B6B",   # Light Red
    "O":   "#FF8C00",   # Orange
    "P":   "#800080",   # Purple
    "LP":  "#DDA0DD",   # Light Purple
    "BL":  "#4169E1",   # Blue
    "LBL": "#87CEEB",   # Light Blue
    "GR":  "#808080",   # Gray
    "DG":  "#A9A9A9",   # Dark Gray
    "LGR": "#D3D3D3",   # Light Gray
    "BK":  "#1C1C1C",   # Black
    "MG":  "#FFD700",   # Mottled Gold
    "AB":  "#CD853F",   # Auburn Brown
    "RL":  "#B22222",   # Reddish
    "OB":  "#FF7F50",   # Orange-Brown
}

_FALLBACK = "#9B7B5A"

# ── Friendly provenance labels ─────────────────────────────────────────────────
# ── Provenance label helpers ───────────────────────────────────────────────────
# Friendly labels are stored in the provenance_labels table in the DB.
# Run scripts/migrate_provenance_labels.py to (re-)seed the table.
# To add or edit a label, update that script and the DB — not this file.


@st.cache_data(ttl=3600)
def _load_prov_labels() -> dict[str, str]:
    """Load raw→display_name mapping from the provenance_labels DB table."""
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT raw, display_name FROM provenance_labels"
        ).fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}


def _fmt_prov(raw: str | None) -> str:
    """Return a short display label for a raw provenance string."""
    if not raw or str(raw).strip() in ("", "nan", "None"):
        return "—"
    s = str(raw).strip()
    label = _load_prov_labels().get(s, s)
    # Fallback truncation for any unlisted long strings
    return label if len(label) <= 50 else label[:47] + "…"


def color_to_hex(code: str) -> str:
    """Resolve a potentially compound Ascher color code (e.g. 'MB:W') to hex."""
    if not code or not code.strip():
        return _FALLBACK
    # Take first component of compound codes like "MB:W" or "KB-DB"
    base = code.split(":")[0].split("-")[0].strip().upper()
    return _COLOR_MAP.get(base, _FALLBACK)


# ── Database helpers ───────────────────────────────────────────────────────────

@st.cache_resource
def _get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        st.error(
            f"Database not found at `{DB_PATH}`. "
            "Run `python scripts/build_kfg_database.py` first."
        )
        st.stop()
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=3600)
def load_corpus() -> pd.DataFrame:
    conn = _get_conn()
    return pd.read_sql_query(
        """
        SELECT
            km.kfg_id,
            km.kfg_name,
            km.provenance,
            km.region,
            km.museum_country,
            km.museum_name,
            km.kfg_url,
            COUNT(c.cord_id) AS cord_count
        FROM khipu_metadata km
        LEFT JOIN cords c ON c.kfg_id = km.kfg_id
        GROUP BY km.kfg_id
        ORDER BY km.kfg_id
        """,
        conn,
    )


@st.cache_data(ttl=3600)
def load_cords(kfg_id: str) -> pd.DataFrame:
    conn = _get_conn()
    return pd.read_sql_query(
        """
        SELECT
            c.cord_id,
            c.cord_name,
            c.hierarchy_level,
            c.parent_cord,
            c.color,
            c.value,
            c.length,
            c.position,
            c.position_in_group,
            c.group_idx,
            c.knots,
            c.attachment,
            c.twist,
            c.termination,
            GROUP_CONCAT(cc.color_code, ':') AS multi_colors
        FROM cords c
        LEFT JOIN cord_colors cc ON cc.cord_id = c.cord_id
        WHERE c.kfg_id = ?
        GROUP BY c.cord_id
        ORDER BY c.group_idx, c.position_in_group, c.position
        """,
        conn,
        params=(kfg_id,),
    )


@st.cache_data(ttl=3600)
def load_meta(kfg_id: str) -> dict:
    conn = _get_conn()
    row = conn.execute(
        """
        SELECT km.*, pc.length AS primary_length, pc.color AS primary_color,
               pc.structure AS primary_structure
        FROM khipu_metadata km
        LEFT JOIN primary_cord pc ON pc.kfg_id = km.kfg_id
        WHERE km.kfg_id = ?
        """,
        (kfg_id,),
    ).fetchone()
    return dict(row) if row else {}


# ── 3D figure builder ──────────────────────────────────────────────────────────

def build_3d_figure(kfg_id: str) -> Optional[go.Figure]:
    df = load_cords(kfg_id)
    if df.empty:
        return None

    # hierarchy_level: 0 = pendant, 1 = first sub, 2 = second sub, …
    GROUP_SPACING = 10.0   # x-gap between groups
    PEND_SPACING  = 1.6    # x-gap between pendants within a group
    SUB_X_OFF     = 0.5    # x-offset per subsidiary level

    pendants = df[df["hierarchy_level"] == 0].reset_index(drop=True)
    subs     = df[df["hierarchy_level"] > 0].sort_values("hierarchy_level").reset_index(drop=True)

    # position lookup: cord_name → (x, y, z)
    pos: dict[str, tuple[float, float, float]] = {}

    xs, ys, zs, colors, texts = [], [], [], [], []
    edge_x, edge_y, edge_z = [], [], []

    # Pendants hang vertically from the primary cord (y=0 plane)
    # x = group_idx * GROUP_SPACING + position_in_group * PEND_SPACING
    for i, row in pendants.iterrows():
        g = float(row["group_idx"]) if pd.notna(row["group_idx"]) else float(i)
        p = float(row["position_in_group"]) if pd.notna(row["position_in_group"]) else 0.0
        x = g * GROUP_SPACING + p * PEND_SPACING
        z = float(row["length"] or 10.0) * 0.3
        xs.append(x); ys.append(0.0); zs.append(z)
        colors.append(color_to_hex(str(row["color"] or "")))
        texts.append(
            f"<b>{row['cord_name']}</b><br>"
            f"Color: {row['color']}<br>"
            f"Value: {row['value']}<br>"
            f"Length: {row['length']} cm<br>"
            f"Knots: {row['knots']}"
        )
        pos[str(row["cord_name"])] = (x, 0.0, z)
        # vertical line down from primary cord
        edge_x += [x, x, None]; edge_y += [0.0, 0.0, None]; edge_z += [0.0, z, None]

    # Subsidiaries branch off their parent
    for _, row in subs.iterrows():
        parent_name = str(row["parent_cord"] or "")
        if parent_name not in pos:
            continue
        px, py, pz = pos[parent_name]
        depth = float(row["hierarchy_level"])   # level 1 = first sub
        sx = px + SUB_X_OFF * depth
        sy = py + depth * 0.5
        sz = pz - float(row["length"] or 5.0) * 0.1

        xs.append(sx); ys.append(sy); zs.append(sz)
        colors.append(color_to_hex(str(row["color"] or "")))
        texts.append(
            f"<b>{row['cord_name']}</b> (sub L{row['hierarchy_level']})<br>"
            f"Color: {row['color']}<br>"
            f"Value: {row['value']}<br>"
            f"Length: {row['length']} cm"
        )
        pos[str(row["cord_name"])] = (sx, sy, sz)
        edge_x += [px, sx, None]; edge_y += [py, sy, None]; edge_z += [pz, sz, None]

    n_pend = len(pendants)

    fig = go.Figure()

    # Structural edges
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="#94a3b8", width=1.5),
        hoverinfo="none",
        showlegend=False,
        name="",
    ))

    # Cord nodes
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(
            size=[7 if i < n_pend else 5 for i in range(len(xs))],
            color=colors,
            line=dict(color="#1e293b", width=0.8),
            opacity=0.92,
        ),
        text=texts,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
        name="",
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Position",
            yaxis_title="Subsidiary depth",
            zaxis_title="Cord length",
            bgcolor="#0f172a",
            xaxis=dict(showgrid=True, gridcolor="#334155", color="#94a3b8"),
            yaxis=dict(showgrid=True, gridcolor="#334155", color="#94a3b8"),
            zaxis=dict(showgrid=True, gridcolor="#334155", color="#94a3b8"),
        ),
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        margin=dict(l=0, r=0, t=0, b=0),
        height=620,
    )
    return fig


# ── X-Ray color grid ───────────────────────────────────────────────────────────

def build_xray_figure(cords_df: pd.DataFrame) -> go.Figure:
    """Flat 2D color grid: one square per pendant, grouped by group_idx."""
    # hierarchy_level 0 = pendant; fall back to all cords if no level-0 rows
    pendants = cords_df[cords_df["hierarchy_level"] == 0].copy()
    if pendants.empty:
        pendants = cords_df.copy()

    xs, ys, cols, texts = [], [], [], []
    groups = sorted(pendants["group_idx"].dropna().unique())

    for g in groups:
        group_cords = pendants[pendants["group_idx"] == g].sort_values("position_in_group")
        for j, (_, row) in enumerate(group_cords.iterrows()):
            xs.append(float(g or 0))
            ys.append(float(j))
            cols.append(color_to_hex(str(row["color"] or "")))
            val = row["value"]
            texts.append(
                f"<b>{row['cord_name']}</b><br>"
                f"Group {int(g) if g == g else '?'} · pos {j}<br>"
                f"Color: {row['color']}<br>"
                f"Value: {val if pd.notna(val) else '—'}"
            )

    fig = go.Figure(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(
            size=16,
            color=cols,
            symbol="square",
            line=dict(color="#334155", width=1),
        ),
        text=texts,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        xaxis_title="Group index",
        yaxis_title="Position in group",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#334155", color="#94a3b8"),
        yaxis=dict(gridcolor="#334155", color="#94a3b8", autorange="reversed"),
        height=max(300, len(ys) * 22 + 80) if ys else 200,
        margin=dict(l=0, r=0, t=10, b=40),
    )
    return fig


# ── Analytics helpers ──────────────────────────────────────────────────────────

def _parse_coord_index(s: str) -> Optional[tuple[float, float]]:
    """Extract (group, position) from a cord coordinate string.

    Handles all formats found in the checks/ CSVs:
      '[4, 2]'           → (4.0, 2.0)   bare list (PP / IP / CP relations)
      'GG@[2, 6]:3'      → (2.0, 6.0)   embedded with color + value
      'W@[4, 0, 0]:138'  → (4.0, 0.0)   3-element subsidiary form
    """
    m = re.search(r"\[(\d+),\s*(\d+)", str(s))
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def _parse_summand_grid_coords(summand_string: str) -> list[tuple[float, float]]:
    """Parse all (group, position) coords from a summand_string column value.

    e.g. 'GG@[2, 6]:3 + W:KB@[2, 7]:1 + W@[3, 0]:56'
      → [(2.0, 6.0), (2.0, 7.0), (3.0, 0.0)]
    """
    return [
        (float(m.group(1)), float(m.group(2)))
        for m in re.finditer(r"\[(\d+),\s*(\d+)", str(summand_string))
    ]


def _bezier_arc(
    x1: float, y1: float, x2: float, y2: float, n: int = 18
) -> tuple[list, list]:
    """Quadratic Bézier arc from (x1,y1) to (x2,y2) for Plotly overlay.

    The control point bows *upward* (lower y value — grid is y-reversed).
    Returns (xs, ys) with a trailing None so multiple arcs can be concatenated
    into a single Plotly trace.
    """
    cx = (x1 + x2) / 2
    cy = min(y1, y2) - max(1.0, abs(x2 - x1) * 0.35)
    xs: list = []
    ys: list = []
    for i in range(n + 1):
        t = i / n
        xs.append((1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2)
        ys.append((1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2)
    return xs + [None], ys + [None]


@st.cache_data(ttl=3600)
def load_analytics_data() -> pd.DataFrame:
    """Read all 9 summary CSVs; return one boolean-flag row per KFG khipu.

    Columns: 'kfg_id' + one column per PATTERN_CONFIG short key (True / False).
    Returns an empty DataFrame when CHECKS_PATH is missing.
    """
    if not CHECKS_PATH.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for key, _name, csv_file, pos_col in PATTERN_CONFIG:
        path = CHECKS_PATH / csv_file
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["kfg_name", pos_col])
        except Exception:
            continue
        df = df.rename(columns={"kfg_name": "kfg_id", pos_col: key})
        df[key] = pd.to_numeric(df[key], errors="coerce").fillna(0) > 0
        # Collapse any duplicate kfg_id rows by taking the max (True wins over False)
        df = df.groupby("kfg_id", as_index=False)[key].max()
        frames.append(df.reset_index(drop=True))

    if not frames:
        return pd.DataFrame()

    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on="kfg_id", how="outer")
    pat_cols = [k for k, *_ in PATTERN_CONFIG if k in result.columns]
    result[pat_cols] = result[pat_cols].fillna(False)

    # Ensure every khipu in the DB corpus is represented (7 khipus have no CSV
    # entries at all; they should appear as all-False rather than being absent).
    all_ids = load_corpus()[["kfg_id"]]
    result = all_ids.merge(result, on="kfg_id", how="left")
    result[pat_cols] = result[pat_cols].fillna(False)

    return result.reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_full_analytics() -> pd.DataFrame:
    """Load ALL numeric columns from every summary CSV, plus provenance.

    Returns one row per KFG khipu with columns like pp_num_sum_cords,
    pp_num_left_sums, …, provenance, region.
    """
    if not CHECKS_PATH.exists():
        return pd.DataFrame()

    result: Optional[pd.DataFrame] = None
    for key, _name, csv_file, _pos_col in PATTERN_CONFIG:
        path = CHECKS_PATH / csv_file
        if not path.exists():
            continue
        df = pd.read_csv(path)
        renames = {c: f"{key}_{c}" for c in df.columns if c != "kfg_name"}
        df = df.rename(columns={"kfg_name": "kfg_id", **renames})
        for c in df.columns:
            if c != "kfg_id":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # Collapse duplicate kfg_id rows before merging to prevent cartesian explosion
        num_cols = [c for c in df.columns if c != "kfg_id"]
        df = df.groupby("kfg_id", as_index=False)[num_cols].max()
        result = df if result is None else result.merge(df, on="kfg_id", how="outer")

    if result is None:
        return pd.DataFrame()

    corp = load_corpus()[["kfg_id", "provenance", "region"]]
    # Use right-join so all 709 DB khipus appear; the 7 with no CSV data get NaN numerics.
    result = result.merge(corp, on="kfg_id", how="right")
    # outer-merging multiple CSVs can introduce duplicates; keep first occurrence
    result = result.drop_duplicates(subset=["kfg_id"]).reset_index(drop=True)
    return result


def build_prevalence_figure(flags_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: # khipus with ≥1 of each summation pattern."""
    pat_cols = [k for k, *_ in PATTERN_CONFIG if k in flags_df.columns]
    short    = {k: name.split("—")[0].strip() for k, name, *_ in PATTERN_CONFIG}

    sorted_items = sorted(
        ((k, int(flags_df[k].sum())) for k in pat_cols),
        key=lambda x: x[1],
        reverse=True,
    )
    keys   = [k for k, _ in sorted_items]
    names  = [short[k] for k in keys]
    values = [cnt for _, cnt in sorted_items]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color="#3b82f6",
        text=values,
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Khipus",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#334155", color="#94a3b8"),
        yaxis=dict(gridcolor="#334155", color="#e2e8f0"),
        margin=dict(l=0, r=50, t=10, b=40),
        height=370,
    )
    return fig


def build_cooccurrence_figure(flags_df: pd.DataFrame) -> go.Figure:
    """9×9 pattern co-occurrence heatmap (diagonal = single-pattern count)."""
    pat_cols = [k for k, *_ in PATTERN_CONFIG if k in flags_df.columns]
    labels   = [k.upper() for k in pat_cols]

    mat  = flags_df[pat_cols].astype(int).values
    cooc = mat.T @ mat

    fig = go.Figure(go.Heatmap(
        z=cooc, x=labels, y=labels,
        colorscale="Blues",
        hovertemplate="%{y} ∩ %{x}: %{z}<extra></extra>",
        showscale=True,
    ))

    # Per-cell annotations with adaptive text colour (Plotly textfont.color
    # does not accept a 2-D array, so we annotate each cell individually).
    threshold = cooc.max() * 0.45
    n = len(labels)
    for i in range(n):
        for j in range(n):
            v = int(cooc[i, j])
            fc = "#0f172a" if v < threshold else "#e2e8f0"
            fig.add_annotation(
                x=labels[j], y=labels[i],
                text=str(v),
                showarrow=False,
                font=dict(size=11, color=fc),
            )

    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        xaxis=dict(color="#94a3b8", tickangle=-35),
        yaxis=dict(color="#94a3b8", autorange="reversed"),
        margin=dict(l=0, r=0, t=10, b=40),
        height=420,
    )
    return fig


def build_complexity_figure(flags_df: pd.DataFrame) -> go.Figure:
    """Histogram: how many distinct patterns does each khipu exhibit?"""
    pat_cols = [k for k, *_ in PATTERN_CONFIG if k in flags_df.columns]
    counts   = flags_df[pat_cols].astype(int).sum(axis=1)

    bins  = list(range(0, int(counts.max()) + 2))
    freqs = [int((counts == b).sum()) for b in bins[:-1]]

    fig = go.Figure(go.Bar(
        x=bins[:-1], y=freqs,
        marker_color="#06b6d4",
        text=freqs, textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Distinct patterns per khipu",
        yaxis_title="Khipus",
        xaxis=dict(tickmode="linear", dtick=1, gridcolor="#334155", color="#94a3b8"),
        yaxis=dict(gridcolor="#334155", color="#94a3b8"),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        margin=dict(l=0, r=30, t=10, b=40),
        height=340,
    )
    return fig


def build_handedness_figure(full_df: pd.DataFrame) -> go.Figure:
    """Grouped bar: total left-sum vs right-sum cords per pattern type."""
    HAND_PATTERNS = [
        ("pp",  "PP"),
        ("ip",  "IP"),
        ("cp",  "CP"),
        ("sp",  "SP"),
        ("is",  "IS"),
    ]
    names, lefts, rights = [], [], []
    for key, label in HAND_PATTERNS:
        lc = f"{key}_num_left_sums"
        rc = f"{key}_num_right_sums"
        if lc in full_df.columns and rc in full_df.columns:
            names.append(label)
            lefts.append(int(full_df[lc].sum(skipna=True)))
            rights.append(int(full_df[rc].sum(skipna=True)))

    fig = go.Figure([
        go.Bar(name="Left (←)", x=names, y=lefts,  marker_color="#3b82f6"),
        go.Bar(name="Right (→)", x=names, y=rights, marker_color="#f97316"),
    ])
    fig.update_layout(
        barmode="group",
        xaxis_title="Pattern",
        yaxis_title="Total sum-cord instances",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#334155", color="#94a3b8"),
        yaxis=dict(gridcolor="#334155", color="#94a3b8"),
        margin=dict(l=0, r=0, t=10, b=40),
        height=340,
    )
    return fig


def build_count_dist_figure(full_df: pd.DataFrame) -> go.Figure:
    """Box plots: distribution of the summation-instance count per pattern."""
    COUNT_COLS = [
        ("pp",  "pp_num_sum_cords",         "PP"),
        ("ip",  "ip_num_sum_cords",          "IP"),
        ("cp",  "cp_num_sum_cords",          "CP"),
        ("sp",  "sp_num_sum_cords",          "SP"),
        ("is",  "is_num_sum_cords",          "IS"),
        ("gg",  "gg_num_sum_groups",         "GG"),
        ("gsb", "gsb_num_group_sum_bands",   "GSB"),
        ("adg", "adg_num_decreasing_groups", "ADG"),
        ("psn", "psn_num_pendant_sub_neighbor_groups", "PSN"),
    ]
    fig = go.Figure()
    palette = ["#3b82f6","#f97316","#22c55e","#a855f7","#f43f5e",
               "#eab308","#06b6d4","#ec4899","#10b981"]
    for i, (_key, col, label) in enumerate(COUNT_COLS):
        if col not in full_df.columns:
            continue
        vals = full_df[col].dropna()
        vals = vals[vals > 0]
        if vals.empty:
            continue
        fig.add_trace(go.Box(
            y=vals, name=label,
            marker_color=palette[i % len(palette)],
            boxpoints="outliers",
            line_color="#e2e8f0",
        ))
    fig.update_layout(
        yaxis_title="Count per khipu (positive cases only)",
        showlegend=False,
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#334155", color="#94a3b8"),
        yaxis=dict(gridcolor="#334155", color="#94a3b8"),
        margin=dict(l=0, r=0, t=10, b=40),
        height=380,
    )
    return fig


def build_magnitude_figure(full_df: pd.DataFrame) -> go.Figure:
    """Box plots: distribution of the *mean sum value* per pattern."""
    MAG_COLS = [
        ("pp",  "pp_mean_sum",  "PP"),
        ("ip",  "ip_mean_sum",  "IP"),
        ("cp",  "cp_mean_sum",  "CP"),
        ("sp",  "sp_mean_sum",  "SP"),
        ("is",  "is_mean_sum",  "IS"),
        ("gg",  "gg_mean_sum",  "GG"),
    ]
    fig = go.Figure()
    palette = ["#3b82f6","#f97316","#22c55e","#a855f7","#f43f5e","#eab308"]
    for i, (_key, col, label) in enumerate(MAG_COLS):
        if col not in full_df.columns:
            continue
        vals = full_df[col].dropna()
        vals = vals[vals > 0]
        if vals.empty:
            continue
        fig.add_trace(go.Box(
            y=vals, name=label,
            marker_color=palette[i % len(palette)],
            boxpoints="outliers",
            line_color="#e2e8f0",
        ))
    fig.update_layout(
        yaxis_title="Mean sum value (positive cases only)",
        showlegend=False,
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#334155", color="#94a3b8"),
        yaxis=dict(gridcolor="#334155", color="#94a3b8"),
        margin=dict(l=0, r=0, t=10, b=40),
        height=380,
    )
    return fig


def build_dual_multi_figure(full_df: pd.DataFrame) -> go.Figure:
    """Stacked bar: for PP/IP/CP — regular vs dual-summand vs multi-summand."""
    DM_PATTERNS = [
        ("pp", "PP", "pp_num_sum_cords", "pp_num_dual_sums", "pp_num_multisummands"),
        ("ip", "IP", "ip_num_sum_cords", "ip_num_dual_sums", "ip_num_multisummands"),
        ("cp", "CP", "cp_num_sum_cords", "cp_num_dual_sums", "cp_num_multisummands"),
    ]
    labels, regulars, duals, multis = [], [], [], []
    for _key, label, total_col, dual_col, multi_col in DM_PATTERNS:
        if total_col not in full_df.columns:
            continue
        total = float(full_df[total_col].sum(skipna=True))
        dual  = float(full_df[dual_col].sum(skipna=True)) if dual_col in full_df.columns else 0.0
        multi = float(full_df[multi_col].sum(skipna=True)) if multi_col in full_df.columns else 0.0
        labels.append(label)
        duals.append(dual)
        multis.append(multi)
        regulars.append(max(0.0, total - dual - multi))

    fig = go.Figure([
        go.Bar(name="Regular",        x=labels, y=regulars, marker_color="#3b82f6"),
        go.Bar(name="Dual-summand",   x=labels, y=duals,    marker_color="#f97316"),
        go.Bar(name="Multi-summand",  x=labels, y=multis,   marker_color="#a855f7"),
    ])
    fig.update_layout(
        barmode="stack",
        yaxis_title="Total cord instances",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#334155", color="#94a3b8"),
        yaxis=dict(gridcolor="#334155", color="#94a3b8"),
        margin=dict(l=0, r=0, t=10, b=40),
        height=340,
    )
    return fig


def build_geo_heatmap(full_df: pd.DataFrame, flags_df: pd.DataFrame) -> go.Figure:
    """Heatmap: provenance (rows) × pattern (cols) = % of khipus with that pattern."""
    pat_cols  = [k for k, *_ in PATTERN_CONFIG]
    short     = {k: k.upper() for k, *_ in PATTERN_CONFIG}

    # join provenance into flags
    if "provenance" in full_df.columns:
        # Drop duplicates before building the lookup to avoid non-unique index
        prov_map = (
            full_df[["kfg_id", "provenance"]]
            .dropna(subset=["provenance"])
            .drop_duplicates(subset=["kfg_id"])
            .set_index("kfg_id")["provenance"]
            .to_dict()
        )
        df = flags_df.copy()
        df["provenance"] = df["kfg_id"].map(prov_map).map(
            lambda v: _fmt_prov(v) if pd.notna(v) else None
        )
    else:
        return go.Figure()

    df = df.dropna(subset=["provenance"])
    top_provs = (
        df.groupby("provenance").size()
        .sort_values(ascending=False)
        .head(25).index.tolist()
    )
    df = df[df["provenance"].isin(top_provs)]

    available = [k for k in pat_cols if k in df.columns]
    z_vals, text_vals = [], []
    for prov in top_provs:
        sub = df[df["provenance"] == prov]
        n   = len(sub)
        row_z, row_t = [], []
        for k in available:
            pct = sub[k].mean() * 100 if n > 0 else 0
            row_z.append(round(pct, 1))
            row_t.append(f"{pct:.0f}%<br>(n={n})")
        z_vals.append(row_z)
        text_vals.append(row_t)

    col_labels = [short[k] for k in available]

    fig = go.Figure(go.Heatmap(
        z=z_vals, x=col_labels, y=top_provs,
        colorscale="YlOrRd",
        hovertemplate="%{y} · %{x}: %{text}<extra></extra>",
        text=text_vals,
        showscale=True,
        colorbar=dict(title="%", ticksuffix="%"),
    ))

    # Per-cell annotations with adaptive text colour.
    max_z = max(v for row in z_vals for v in row) or 1
    threshold = max_z * 0.45
    for i, prov in enumerate(top_provs):
        for j, col in enumerate(col_labels):
            v = z_vals[i][j]
            fc = "#0f172a" if v < threshold else "#f8fafc"
            fig.add_annotation(
                x=col, y=prov,
                text=text_vals[i][j].replace("<br>", "\n"),
                showarrow=False,
                font=dict(size=8, color=fc),
            )
    fig.update_layout(
        xaxis_title="Pattern",
        yaxis_title="Provenance",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        xaxis=dict(color="#94a3b8"),
        yaxis=dict(color="#94a3b8", dtick=1),
        margin=dict(l=0, r=0, t=10, b=40),
        height=max(400, len(top_provs) * 22 + 80),
    )
    return fig


def build_pca_figure(flags_df: pd.DataFrame) -> go.Figure:
    """2D PCA scatter of khipus in 9-dimensional pattern-flag space."""
    pat_cols = [k for k, *_ in PATTERN_CONFIG if k in flags_df.columns]
    X = flags_df[pat_cols].fillna(False).astype(float).values

    # Center then SVD (= PCA)
    Xc  = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:2].T  # (n, 2)

    n_patterns = X.sum(axis=1).astype(int)
    palette    = ["#475569","#3b82f6","#f97316","#22c55e","#a855f7",
                  "#f43f5e","#eab308","#06b6d4","#ec4899","#10b981"]
    colors_ = [palette[min(n, len(palette) - 1)] for n in n_patterns]

    hover = [
        f"<b>{row['kfg_id']}</b><br>{int(n_patterns[i])} pattern(s)"
        for i, (_, row) in enumerate(flags_df.iterrows())
    ]

    fig = go.Figure(go.Scatter(
        x=proj[:, 0], y=proj[:, 1],
        mode="markers",
        marker=dict(size=5, color=colors_, opacity=0.7,
                    line=dict(width=0)),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    # Add invisible legend traces for colour meaning
    for cnt, col in enumerate(palette[:5]):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=8, color=col),
            name=f"{cnt} pattern{'s' if cnt != 1 else ''}",
            showlegend=True,
        ))
    fig.update_layout(
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        legend=dict(bgcolor="rgba(0,0,0,0)", title="# patterns"),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#334155", color="#94a3b8", zeroline=False),
        yaxis=dict(gridcolor="#334155", color="#94a3b8", zeroline=False),
        margin=dict(l=0, r=0, t=10, b=40),
        height=480,
    )
    return fig


@st.cache_data(ttl=3600)
def load_arc_data(kfg_id: str) -> dict:
    """Return cord-level arc data for the 5 pendant-level summation patterns.

    Result shape:
        { pattern_stem: [(sum_coord, [summand_coords]), ...] }
    where each coord is (group_float, position_float).
    """
    arc_csvs = {
        "pendant_pendant_sum":    "pendant_pendant_sum_relation.csv",
        "indexed_pendant_sum":    "indexed_pendant_sum_relation.csv",
        "colored_pendant_sum":    "colored_pendant_sum_relation.csv",
        "subsidiary_pendant_sum": "subsidiary_pendant_sum_relation.csv",
        "indexed_subsidiary_sum": "indexed_subsidiary_sum_relation.csv",
    }
    result: dict = {}
    for pattern, csv_file in arc_csvs.items():
        path = CHECKS_PATH / csv_file
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "kfg_name" not in df.columns:
            continue
        rows = df[df["kfg_name"] == kfg_id]
        if rows.empty:
            continue
        arcs: list = []
        for _, row in rows.iterrows():
            sum_coord = _parse_coord_index(str(row.get("cord_index", "") or ""))
            summand_coords = _parse_summand_grid_coords(
                str(row.get("summand_string", "") or "")
            )
            if sum_coord and summand_coords:
                arcs.append((sum_coord, summand_coords))
        if arcs:
            result[pattern] = arcs
    return result


def build_arc_traces(
    arc_data: dict, enabled_patterns: set
) -> list:
    """Build one Plotly Scatter trace per enabled arc pattern."""
    traces: list = []
    for pattern, arcs in arc_data.items():
        if pattern not in enabled_patterns:
            continue
        color, label = ARC_PATTERNS.get(pattern, ("#888888", pattern))
        all_x: list = []
        all_y: list = []
        for sum_coord, summand_coords in arcs:
            sx, sy = sum_coord
            for tx, ty in summand_coords:
                arc_xs, arc_ys = _bezier_arc(sx, sy, tx, ty)
                all_x.extend(arc_xs)
                all_y.extend(arc_ys)
        if all_x:
            traces.append(go.Scatter(
                x=all_x, y=all_y,
                mode="lines",
                line=dict(color=color, width=1.5),
                opacity=0.75,
                name=label,
                hoverinfo="skip",
            ))
    return traces


# ── K-CAT metadata ─────────────────────────────────────────────────────────────
KCAT_GITHUB = "https://github.com/adafieno/khipu-computational-toolkit"

_CUSTOM_CSS = """<style>
/* ── hide Streamlit chrome ──────────────────────────────────────────────────  */
header[data-testid="stHeader"]   { display: none !important; }
[data-testid="stToolbar"]        { display: none !important; }
[data-testid="stDecoration"]     { display: none !important; }
[data-testid="stFooterDefault"]  { display: none !important; }
#MainMenu                        { display: none !important; }
/* hide the native Streamlit footer bar (matches any emotion class) */
[data-testid="stBottom"]         { display: none !important; }

/* ── global layout ──────────────────────────────────────────────────────────  */
[data-testid="stApp"]            { background: #0b1120; }
/* push content below fixed header (64 px) and above fixed footer (32 px) */
[data-testid="stMainBlockContainer"],
.main .block-container           { padding-top: 72px !important; padding-bottom: 44px !important; }

/* ── fixed header bar ───────────────────────────────────────────────────────  */
.kcat-header {
    position: fixed; top: 0; left: 0; right: 0; z-index: 999999;
    display: flex; align-items: center; gap: 16px;
    height: 64px; padding: 0 28px;
    background: #0f172a; border-bottom: 2px solid #1e3a5f;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
.kcat-app-name { font-size: 1.25rem; font-weight: 700; color: #e2e8f0; white-space: nowrap; }
.kcat-badge {
    font-size: 0.67rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #64748b; background: #1e293b;
    border: 1px solid #334155; border-radius: 20px;
    padding: 3px 11px; white-space: nowrap;
}
.kcat-stat    { font-size: 0.8rem; color: #475569; white-space: nowrap; }
.kcat-spacer  { flex: 1; }
.kcat-gh-link { font-size: 0.82rem; color: #3b82f6 !important; text-decoration: none; white-space: nowrap; }
.kcat-gh-link:hover { color: #60a5fa !important; }

/* ── fixed footer ───────────────────────────────────────────────────────────  */
.kcat-footer {
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 999999;
    padding: 6px 28px; background: #070f1c;
    border-top: 1px solid #1e293b;
    font-size: 0.72rem; color: #94a3b8; text-align: center;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    pointer-events: none;
}

/* ── sidebar ────────────────────────────────────────────────────────────────  */
[data-testid="stSidebar"]          { background: #070f1c !important; }
[data-testid="stSidebar"] section  { padding-top: 4px !important; }

/* nav section header */
.nav-section-label {
    padding: 18px 16px 6px; font-size: 0.6rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase; color: #475569;
}
/* sidebar stats chip */
.sidebar-stats {
    padding: 10px 16px 8px; font-size: 0.73rem; color: #475569;
    border-top: 1px solid #111c2e; margin-top: 8px;
}

/* ── nav radio items ────────────────────────────────────────────────────────  */
/* hide the auto-generated widget label */
[data-testid="stSidebar"] .stRadio > label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { display: none !important; }

/* each radio option row ── target data-baseweb="radio" (BaseWeb component) */
[data-testid="stSidebar"] [data-baseweb="radio"] {
    border-radius: 8px !important; padding: 10px 14px !important;
    margin: 2px 6px !important; background: transparent !important;
    transition: background 0.1s !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"]:hover {
    background: #111c2e !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) {
    background: #1e3a5f !important;
}
/* the text inside each option */
[data-testid="stSidebar"] [data-baseweb="radio"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-baseweb="radio"] span {
    color: #94a3b8 !important; font-size: 0.93rem !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) span {
    color: #3b82f6 !important; font-weight: 600 !important;
}
/* hide the radio circle indicator */
[data-testid="stSidebar"] [data-baseweb="radio"] [data-testid="stRadioOptionLabel"] > span:first-child,
[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child { display: none !important; }

/* ── in-section khipu picker card ───────────────────────────────────────────  */
.picker-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 12px 16px; margin-bottom: 18px;
}
</style>"""




# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Khipu Explorer",
        page_icon="🧶",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    corpus = load_corpus()

    # ── Header bar ─────────────────────────────────────────────────────────────
    st.markdown(
        f"""<div class="kcat-header">
          <span class="kcat-app-name">Khipu Explorer</span>
          <span class="kcat-badge">Part of K-CAT</span>
          <span class="kcat-stat">KFG &nbsp;·&nbsp; {len(corpus):,} khipus &nbsp;·&nbsp; {corpus['cord_count'].sum():,} cords</span>
          <span class="kcat-spacer"></span>
          <a class="kcat-gh-link" href="{KCAT_GITHUB}" target="_blank">K-CAT on GitHub ↗</a>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Sidebar nav ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="nav-section-label">Navigate</div>', unsafe_allow_html=True)
        _view_raw = st.radio(
            "navigation",
            ["🗂  Corpus Browser", "📊  Analytics", "🧶  3D Viewer", "🔬  X-Ray View"],
            index=0,
            label_visibility="collapsed",
        )
        st.markdown(
            f'<div class="sidebar-stats">{len(corpus):,} khipus'
            f'<br>{corpus["cord_count"].sum():,} cords</div>',
            unsafe_allow_html=True,
        )

    # Strip icon prefix to get the plain view name
    view = _view_raw.split("  ", 1)[-1]

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="kcat-footer">© 2026 Agustín Da Fieno Delucchi</div>',
        unsafe_allow_html=True,
    )

    # ── Khipu picker helper — used by 3D Viewer and X-Ray View ─────────────────
    def _khipu_picker(key_prefix: str) -> Optional[str]:
        """Render a provenance filter + khipu selector styled as a card.
        Returns the selected kfg_id, or None if no match."""
        st.markdown('<div class="picker-card">', unsafe_allow_html=True)
        provenances = sorted(corpus["provenance"].dropna().unique())
        prov_options = ["All"] + sorted(set(_fmt_prov(p) for p in provenances))
        _prov_raw_map: dict[str, list[str]] = {}
        for p in provenances:
            _prov_raw_map.setdefault(_fmt_prov(p), []).append(p)

        c1, c2 = st.columns([1, 2])
        prov_label = c1.selectbox("Provenance", prov_options, key=f"{key_prefix}_prov")
        if prov_label == "All":
            pool = corpus
        else:
            raw_vals = _prov_raw_map.get(prov_label, [])
            pool = corpus[corpus["provenance"].isin(raw_vals)]

        khipu_ids = pool["kfg_id"].tolist()
        sel: Optional[str] = None
        if khipu_ids:
            k_labels = {
                row["kfg_id"]: (
                    f"{row['kfg_id']}  {row['kfg_name'] or ''}  "
                    f"[{_fmt_prov(row['provenance'])}]"
                )
                for _, row in pool.iterrows()
            }
            sel = c2.selectbox(
                "Khipu",
                khipu_ids,
                format_func=lambda k: k_labels.get(k, k),
                key=f"{key_prefix}_khipu",
            )
        else:
            c2.warning("No khipus match this filter.")
        st.markdown("</div>", unsafe_allow_html=True)
        return sel

    # ── Corpus Browser ─────────────────────────────────────────────────────────
    if view == "Corpus Browser":
        st.header("Corpus Browser")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Khipus", f"{len(corpus):,}")
        m2.metric("Cords", f"{corpus['cord_count'].sum():,}")
        m3.metric("Provenances", str(corpus["provenance"].nunique()))
        m4.metric("Countries", str(corpus["museum_country"].nunique()))

        st.divider()

        # Filters
        f1, f2, f3 = st.columns([2, 1, 1])
        search = f1.text_input("Search (ID, name, provenance, museum)")
        countries = ["All"] + sorted(str(c) for c in corpus["museum_country"].dropna().unique())
        sel_country = f2.selectbox("Country", countries, key="cb_country")
        max_cords = int(corpus["cord_count"].max() or 1)
        min_cords = f3.slider("Min cords", 0, max_cords, 0, key="cb_min_cords")

        display = corpus.copy()
        if search:
            mask = display.apply(
                lambda r: search.lower() in " ".join(str(v) for v in r).lower(),
                axis=1,
            )
            display = display[mask]
        if sel_country != "All":
            display = display[display["museum_country"] == sel_country]
        display = display[display["cord_count"] >= min_cords]

        st.caption(f"Showing **{len(display):,}** of {len(corpus):,} khipus")

        # Clickable links in the KFG URL column
        def make_link(row: pd.Series) -> str:
            url = row.get("kfg_url", "")
            kid = row.get("kfg_id", "")
            if url:
                return f'<a href="{url}" target="_blank">{kid}</a>'
            return str(kid)

        st.dataframe(
            display.rename(columns={
                "kfg_id": "KFG ID",
                "kfg_name": "Name",
                "provenance": "Provenance",
                "region": "Region",
                "museum_country": "Country",
                "museum_name": "Museum",
                "cord_count": "Cords",
            }).drop(columns=["kfg_url"], errors="ignore"),
            width='stretch',
            hide_index=True,
            height=600,
        )
    # ── Analytics ───────────────────────────────────────────────────────────────────────
    elif view == "Analytics":
        flags_df = load_analytics_data()
        full_df  = load_full_analytics()
        n        = len(flags_df)

        st.header("Corpus Analytics")
        if flags_df.empty:
            st.warning(
                "KFG checks/ directory not found. "
                "Run `python scripts/build_kfg_database.py` and ensure the "
                "checks/ directory is present."
            )
            return

        pat_cols    = [k for k, *_ in PATTERN_CONFIG if k in flags_df.columns]
        any_pattern = (flags_df[pat_cols].astype(int).sum(axis=1) > 0)
        no_pattern  = (~any_pattern).sum()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("KFG khipus", f"{n:,}")
        m2.metric("With ≥1 pattern", f"{int(any_pattern.sum()):,}")
        m3.metric("Pattern coverage", f"{any_pattern.mean() * 100:.1f}%")
        m4.metric("No pattern", f"{int(no_pattern):,}")
        # Most common pattern
        best_k = max(pat_cols, key=lambda k: int(flags_df[k].sum()))
        best_n = int(flags_df[best_k].sum())
        best_label = best_k.upper()
        m5.metric("Most common", f"{best_label} ({best_n:,})")

        st.divider()

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview",
            "🔬 Deep Dive",
            "🌍 Geography",
            "🔭 Pattern Space",
        ])

        # ── Tab 1: Overview ────────────────────────────────────────────────────
        with tab1:
            with st.expander("ℹ️ What do the pattern codes mean?", expanded=False):
                st.markdown(
                    "| Code | Full name | What it detects |\n"
                    "|------|-----------|-----------------|\n"
                    "| **PP** | Pendant-Pendant Sum | Pendant = sum of other pendants in same/adjacent group |\n"
                    "| **IP** | Indexed Pendant Sum | Pendant = sum of pendants at the same position index across groups |\n"
                    "| **CP** | Colored Pendant Sum | Pendant = sum of pendants sharing its Ascher color code |\n"
                    "| **SP** | Subsidiary Pendant Sum | Pendant = sum of its own subsidiary (child) cords |\n"
                    "| **IS** | Indexed Subsidiary Sum | Subsidiary = sum of subsidiaries at the same index across groups |\n"
                    "| **GG** | Group-Group Sum | A whole cord group = sum of two other groups |\n"
                    "| **GSB** | Group Sum Bands | Alternating groups sum adjacent ones, forming bands |\n"
                    "| **ADG** | Ascher Decreasing Groups | Groups with monotonically decreasing cord values (Ascher 1981) |\n"
                    "| **PSN** | Pendant-Sub-Neighbor | Pendant = neighboring pendant + that pendant's subsidiary |"
                )
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Pattern Prevalence")
                st.caption("Khipus exhibiting ≥1 instance of each summation pattern")
                with st.expander("ℹ️ How to read this chart", expanded=False):
                    st.markdown(
                        "Each bar = one pattern code. Bar length = number of khipus that have "
                        "at least one confirmed instance of that pattern.  \n\n"
                        "The chart is sorted by prevalence. A long bar means the pattern is "
                        "common across the corpus; a short bar means it is rare or harder to detect."
                    )
                st.plotly_chart(build_prevalence_figure(flags_df), width="stretch")
            with c2:
                st.subheader("Pattern Co-occurrence")
                st.caption("Khipus that simultaneously exhibit both patterns (diagonal = single-pattern count).")
                with st.expander("ℹ️ How to read this heatmap", expanded=False):
                    st.markdown(
                        "**Cells** show how many khipus express *both* the row pattern and the column pattern simultaneously.  \n"
                        "**Diagonal** = khipus that have that single pattern (same as the prevalence bar chart).  \n"
                        "**Darker off-diagonal** = the two patterns frequently co-occur — suggesting they belong to "
                        "the same scribe tradition or accounting layer.  \n"
                        "**White/blank cells** = the pair rarely or never co-occurs."
                    )
                st.plotly_chart(build_cooccurrence_figure(flags_df), width="stretch")

            st.subheader("Pattern Complexity")
            st.caption(
                "How many *distinct* summation patterns does a single khipu exhibit? "
                "0 = no detected summation structure; higher = richer accounting system."
            )
            with st.expander("ℹ️ How to read this chart", expanded=False):
                st.markdown(
                    "Each khipu is scored by counting how many of the 9 pattern flags are True.  \n"
                    "**0** = no summation structure detected (or below detection threshold).  \n"
                    "**1–2** = most common — khipus focused on one or two accounting conventions.  \n"
                    "**3+** = complex khipus that combine multiple summation strategies simultaneously.  \n\n"
                    "The long tail toward higher complexities is where the most analytically interesting khipus sit."
                )
            st.plotly_chart(build_complexity_figure(flags_df), width="stretch")

        # ── Tab 2: Deep Dive ───────────────────────────────────────────────────
        with tab2:
            st.subheader("Handedness (Left vs Right Sums)")
            st.caption("Total left- vs right-oriented summation instances across the corpus (cord-level patterns: PP · IP · CP · SP · IS).")
            with st.expander("ℹ️ How to read this chart", expanded=False):
                st.markdown(
                    "**Left (←)** — the summand cords lie to the *left* of the sum cord along the primary cord.  \n"
                    "**Right (→)** — the summand cords lie to the *right*.  \n\n"
                    "Handedness reflects the direction in which the scribe accumulated the running total. "
                    "A strong imbalance toward one side may indicate a preferred reading convention "
                    "within a khipu tradition or find-site group."
                )
            st.plotly_chart(build_handedness_figure(full_df), width="stretch")

            st.divider()
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("Instance-Count Distribution")
                st.caption("Box plots of *how many* summation instances each khipu has (positive cases only).")
                with st.expander("ℹ️ How to read this chart", expanded=False):
                    st.markdown(
                        "Each box covers the **interquartile range** (25th–75th percentile) of instance counts "
                        "across all khipus that have at least one of that pattern.  \n"
                        "The line inside the box = median; whiskers extend to 1.5× IQR.  \n"
                        "Dots beyond the whiskers = outlier khipus with unusually many instances.  \n\n"
                        "A tall box means high variability — some khipus use that pattern extensively, "
                        "others only sparingly."
                    )
                st.plotly_chart(build_count_dist_figure(full_df), width="stretch")
            with c4:
                st.subheader("Sum Magnitude Distribution")
                st.caption("Box plots of the mean cord-value sum per khipu, for patterns that report numeric magnitudes.")
                with st.expander("ℹ️ How to read this chart", expanded=False):
                    st.markdown(
                        "For each khipu that has a given pattern, the **mean sum value** is computed as the "
                        "average of all individual cord sums attributed to that pattern.  \n"
                        "Box layout is the same as the Instance-Count chart: IQR box, median line, "
                        "1.5× IQR whiskers, individual outlier dots.  \n\n"
                        "High magnitudes for IP or CP can indicate that those patterns were used to "
                        "aggregate large commodity totals, while low-magnitude patterns may represent "
                        "fine-grained sub-unit accounting."
                    )
                st.plotly_chart(build_magnitude_figure(full_df), width="stretch")

            st.divider()
            st.subheader("Dual- & Multi-Summand Breakdown")
            st.caption("PP, IP, and CP cord instances split by summation complexity.")
            with st.expander("ℹ️ Summation complexity types", expanded=False):
                st.markdown(
                    "**Regular** — sum cord = A + B (the standard, most common form).  \n"
                    "**Dual-summand** — the same cord independently participates in *two* separate summation relationships.  \n"
                    "**Multi-summand** — sum cord = A + B + C + … (three or more addends).  \n\n"
                    "Dual- and multi-summand instances suggest a cord was used as a pivot point "
                    "in overlapping accounting structures."
                )
            st.plotly_chart(build_dual_multi_figure(full_df), width="stretch")

        # ── Tab 3: Geography ───────────────────────────────────────────────────
        with tab3:
            st.subheader("Pattern Rate by Provenance")
            st.caption("Percentage of khipus from each provenance (top 25 by count) that exhibit each summation pattern.")
            with st.expander("ℹ️ How to read this heatmap", expanded=False):
                st.markdown(
                    "**Columns** = pattern codes (PP, IP, CP, SP, IS, GG, GSB, ADG, PSN).  \n"
                    "**Rows** = archaeological find sites, sorted by total khipu count.  \n"
                    "**Cell colour** — darker = higher rate; lighter = lower or absent.  \n"
                    "**Cell label** — shows the percentage and sample size (n=…) for that site.  \n\n"
                    "A consistently dark column across many sites indicates a corpus-wide pattern; "
                    "a dark cell in just one row suggests a pattern that may be regionally specific "
                    "or linked to a particular administrative tradition."
                )
            st.plotly_chart(build_geo_heatmap(full_df, flags_df), width="stretch")

        # ── Tab 4: Pattern Space ───────────────────────────────────────────────
        with tab4:
            st.subheader("Khipu Pattern-Space (PCA)")
            st.caption(
                "Each dot = one khipu, projected from a 9-dimensional boolean pattern-flag space "
                "onto the first two principal components."
            )
            with st.expander("ℹ️ How to read this plot", expanded=False):
                st.markdown(
                    "**What is PCA?**  \n"
                    "Principal Component Analysis finds the directions of maximum variance in the data. "
                    "Here the 9 input dimensions are the binary flags for PP, IP, CP, SP, IS, GG, GSB, ADG, PSN.\n\n"
                    "**Axes (PC 1, PC 2)** are linear combinations of those 9 flags. "
                    "They don't have a simple physical meaning, but khipus that sit close together "
                    "are *similar in their summation profiles*.\n\n"
                    "**Colour** = number of distinct patterns expressed (0 = grey, 1 = blue, 2 = orange, …).  \n"
                    "**Clusters** in the scatter suggest structural families or regional schools of accounting."
                )
            st.plotly_chart(build_pca_figure(flags_df), width="stretch")

            st.divider()
            st.subheader("Pattern Detail Table")
            st.caption("Per-pattern corpus-wide statistics from the KFG checks/ ground truth.")
            with st.expander("ℹ️ Column definitions", expanded=False):
                st.markdown(
                    "**Khipus** — number of khipus with at least one confirmed instance of this pattern.  \n"
                    "**Coverage %** — Khipus ÷ total corpus size.  \n"
                    "**Avg count/khipu** — mean number of individual summation instances per positive khipu "
                    "(e.g. how many PP sum-cords a khipu typically has).  \n"
                    "**Avg mean-sum** — mean of the per-khipu average cord-value sum; gives a sense of "
                    "the numeric magnitude of the values involved."
                )

            rows = []
            for key, name, _csv, pos_col in PATTERN_CONFIG:
                if key not in flags_df.columns:
                    continue
                n_pos  = int(flags_df[key].sum())
                pct    = n_pos / n * 100
                mean_col = f"{key}_mean_sum"
                mean_val = (
                    f"{full_df[mean_col].mean(skipna=True):.1f}"
                    if not full_df.empty and mean_col in full_df.columns
                    else "—"
                )
                count_col = next(
                    (f"{key}_{c}" for c in ["num_sum_cords","num_sum_groups",
                                             "num_group_sum_bands","num_decreasing_groups",
                                             "num_pendant_sub_neighbor_groups"]
                     if not full_df.empty and f"{key}_{c}" in full_df.columns),
                    None,
                )
                mean_count = (
                    f"{full_df[count_col].mean(skipna=True):.1f}"
                    if count_col else "—"
                )
                rows.append({
                    "Pattern": name,
                    "Khipus": n_pos,
                    "Coverage %": f"{pct:.1f}%",
                    "Avg count/khipu": mean_count,
                    "Avg mean-sum": mean_val,
                })
            st.dataframe(
                pd.DataFrame(rows),
                width="stretch",
                hide_index=True,
            )

    # ── 3D Viewer ──────────────────────────────────────────────────────────────
    elif view == "3D Viewer":
        st.header("3D Viewer")
        selected_id = _khipu_picker("3dv")
        if not selected_id:
            st.info("Select a khipu above.")
            return

        meta = load_meta(selected_id)
        st.subheader(meta.get('kfg_name') or selected_id)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("KFG ID", selected_id)
        c2.metric("Provenance", _fmt_prov(meta.get("provenance")))
        c3.metric("Museum", str(meta.get("museum_name") or "—")[:30])
        c4.metric("Primary cord", f"{meta.get('primary_length') or '?'} cm")

        url = meta.get("kfg_url", "")
        if url:
            st.markdown(f"[View on KFG ↗]({url})")

        with st.spinner("Building 3D visualization…"):
            fig = build_3d_figure(selected_id)

        if fig:
            st.plotly_chart(fig, width='stretch')
            with st.expander("Raw cord data"):
                cords_df = load_cords(selected_id)
                st.dataframe(
                    cords_df[[
                        "cord_name", "hierarchy_level", "parent_cord",
                        "color", "value", "length", "knots",
                        "group_idx", "position_in_group",
                    ]],
                    width='stretch',
                    hide_index=True,
                )
        else:
            st.warning("No cord data found for this khipu.")

    # ── X-Ray View ─────────────────────────────────────────────────────────────
    elif view == "X-Ray View":
        st.header("X-Ray View")
        selected_id = _khipu_picker("xray")
        if not selected_id:
            st.info("Select a khipu above.")
            return

        meta = load_meta(selected_id)
        st.subheader(meta.get('kfg_name') or selected_id)

        url = meta.get("kfg_url", "")
        if url:
            st.markdown(f"[View on KFG ↗]({url})")

        cords_df = load_cords(selected_id)
        if cords_df.empty:
            st.warning("No cord data found for this khipu.")
            return

        pendants    = cords_df[cords_df["hierarchy_level"] == 0]
        subs        = cords_df[cords_df["hierarchy_level"] > 0]
        n_groups    = cords_df["group_idx"].nunique()
        valued      = cords_df[pd.to_numeric(cords_df["value"], errors="coerce").notna()]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total cords", len(cords_df))
        m2.metric("Pendants", len(pendants))
        m3.metric("Subsidiaries", len(subs))
        m4.metric("Cord groups", n_groups)

        # ─ Arc overlay controls
        st.subheader("Summation Arc Overlay")
        loader_inst = _get_loader()
        arc_traces: list = []
        if loader_inst and loader_inst.in_kfg(selected_id):
            arc_data  = load_arc_data(selected_id)
            available = [p for p in ARC_PATTERNS if arc_data.get(p)]
            if available:
                st.caption("Toggle pattern types to show/hide arcs on the color grid.")
                toggle_cols     = st.columns(max(1, len(available)))
                enabled_patterns: set = set()
                for i, p in enumerate(available):
                    _arc_color, label = ARC_PATTERNS[p]
                    if toggle_cols[i].checkbox(label, value=True, key=f"arc_{p}"):
                        enabled_patterns.add(p)
                arc_traces = build_arc_traces(arc_data, enabled_patterns)
            else:
                st.caption("No cord-level summation patterns found for this khipu.")
        else:
            if loader_inst:
                st.caption("This khipu is not in the KFG corpus — arc overlays unavailable.")
            else:
                st.caption("KFG checks/ directory not found — arc overlays unavailable.")

        # ─ Color grid
        st.subheader("Color grid (pendants by group)")
        st.caption(
            "Each square is one pendant cord, colored by Ascher color code. "
            "Arcs connect sum cords to their summands; arc color = pattern type."
        )
        xray_fig = build_xray_figure(cords_df)
        for trace in arc_traces:
            xray_fig.add_trace(trace)
        if arc_traces:
            xray_fig.update_layout(showlegend=True)
        st.plotly_chart(xray_fig, width='stretch')

        # Group summary table
        st.subheader("Group summary")
        groups_df = (
            cords_df.groupby("group_idx")
            .agg(
                cords=("cord_id", "count"),
                colors=("color", lambda x: " · ".join(sorted(set(
                    str(v) for v in x if v and str(v).strip()
                )))),
                numeric_count=("value", lambda x: pd.to_numeric(x, errors="coerce").notna().sum()),
                total_value=(
                    "value",
                    lambda x: round(pd.to_numeric(x, errors="coerce").sum(), 3),
                ),
            )
            .reset_index()
            .rename(columns={"group_idx": "Group", "cords": "Cords",
                              "colors": "Colors present",
                              "numeric_count": "With values",
                              "total_value": "Sum of values"})
        )
        st.dataframe(groups_df, width='stretch', hide_index=True)




if __name__ == "__main__":
    main()
