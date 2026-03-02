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
    - Summation Arcs  : cord group color map + summation arc overlays (PP/IP/CP/SP/IS)

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


def _chart_title(text: str) -> None:
    """Render a chart section title without Streamlit's heading wrapper machinery."""
    st.markdown(
        f'<p style="font-size:1.2rem;font-weight:600;color:#cbd5e1;'
        f'margin:8px 0 2px 0;padding:0;line-height:1.3">{text}</p>',
        unsafe_allow_html=True,
    )


@st.dialog("Khipu Details", width="large")
def _khipu_detail_modal(kfg_id: str) -> None:
    """Full-detail modal for a single khipu."""
    meta     = load_meta(kfg_id)
    cords_df = load_cords(kfg_id)
    url      = meta.get("kfg_url", "")

    # ── Header ────────────────────────────────────────────────────────────────
    h_col, link_col = st.columns([4, 1])
    h_col.markdown(
        f'<p style="font-size:1.4rem;font-weight:700;color:#e2e8f0;margin:0">{kfg_id}</p>',
        unsafe_allow_html=True,
    )
    if url:
        link_col.markdown(
            f'<div style="text-align:right;padding-top:4px">'
            f'<a href="{url}" target="_blank" '
            f'style="font-size:0.85rem;color:#3b82f6;text-decoration:none">'
            f'View on KFG ↗</a></div>',
            unsafe_allow_html=True,
        )
    name = meta.get("kfg_name") or ""
    if name and str(name).strip() not in ("", "nan", "None"):
        st.caption(str(name))

    st.markdown("---")

    # ── Key metrics ───────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Provenance",    _fmt_prov(meta.get("provenance")))
    c2.metric("Region",        str(meta.get("region")        or "—"))
    c3.metric("Country",       str(meta.get("museum_country") or "—"))

    c4, c5, c6 = st.columns(3)
    c4.metric("Museum",        str(meta.get("museum_name")   or "—")[:40])
    c5.metric("Primary cord",  f"{meta.get('primary_length') or '?'} cm")
    c6.metric("Primary color", str(meta.get("primary_color") or "—"))

    # ── Cord summary ─────────────────────────────────────────────────────────
    if not cords_df.empty:
        st.markdown("---")
        pendants = cords_df[cords_df["hierarchy_level"] == 0]
        subs     = cords_df[cords_df["hierarchy_level"] > 0]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total cords",   len(cords_df))
        d2.metric("Pendants",      len(pendants))
        d3.metric("Subsidiaries",  len(subs))
        d4.metric("Cord groups",   cords_df["group_idx"].nunique())

        with st.expander("Cord data", expanded=False):
            st.dataframe(
                cords_df[[
                    "cord_name", "hierarchy_level", "parent_cord",
                    "color", "value", "length",
                ]],
                hide_index=True,
                width="stretch",
            )

    # ── Remaining metadata fields ─────────────────────────────────────────────
    skip = {
        "kfg_id", "kfg_name", "kfg_url", "provenance", "region",
        "museum_name", "museum_country", "primary_length",
        "primary_color", "primary_structure",
    }
    extra = {
        k: v for k, v in meta.items()
        if k not in skip and v is not None and str(v).strip() not in ("", "nan", "None")
    }
    if extra:
        st.markdown("---")
        st.caption("Additional metadata")
        rows_extra = [[k.replace("_", " ").title(), str(v)] for k, v in extra.items()]
        st.dataframe(
            pd.DataFrame(rows_extra, columns=["Field", "Value"]),
            hide_index=True,
            width="stretch",
        )


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

def _load_knot_clusters(kfg_id: str) -> pd.DataFrame:
    """Load per-cord knot cluster data from knot_clusters table."""
    conn = _get_conn()
    return pd.read_sql_query(
        """
        SELECT kc.cord_id, c.cord_name, kc.knot_type, kc.num_knots, kc.position_cm
        FROM knot_clusters kc
        JOIN cords c ON c.cord_id = kc.cord_id
        WHERE c.kfg_id = ?
        ORDER BY c.cord_id, kc.cluster_ordinal
        """,
        conn,
        params=(kfg_id,),
    )


def _parse_knots(knot_str: str) -> list[dict]:
    """Parse a cord's knots column string into a list of knot dicts.

    Format in DB: one or more tokens like  '1S(33.0,U)'  or  '2L(18.5,Z)'
    separated by commas or semicolons.  Returns list of
    {'type': 'S'|'L'|'E', 'pos_cm': float, 'turns': int}.
    """
    if not knot_str or knot_str in ("-", "None", ""):
        return []
    knots = []
    # match patterns like  1S(33.0,U)  or  2L(18.5,Z)  or  E(12,U)
    for m in re.finditer(r"(\d*)([SLE])\(([0-9.]+)", str(knot_str)):
        turns = int(m.group(1)) if m.group(1) else 1
        ktype = m.group(2)
        pos   = float(m.group(3))
        knots.append({"type": ktype, "turns": turns, "pos_cm": pos})
    return knots


def build_3d_figure(kfg_id: str) -> Optional[go.Figure]:
    """Interactive 3D cord-structure viewer, modelled after the OKR implementation.

    Layout:
      • Primary cord  — thick brown horizontal line along x-axis at z=0
      • Pendants      — thick colored lines hanging down (negative z)
      • Subsidiaries  — elbow-offset branches from their parent cord
      • Knots         — shaped markers along each cord (all 8 types from knot_clusters)
                        S/L/E/EE/SP/LL/TF/BL/U — each with distinct symbol + colour
    """
    df = load_cords(kfg_id)
    if df.empty:
        return None

    pendants = df[df["hierarchy_level"] == 0].sort_values(
        ["group_idx", "position_in_group"]
    ).reset_index(drop=True)
    subs = df[df["hierarchy_level"] > 0].sort_values("hierarchy_level").reset_index(drop=True)

    if pendants.empty:
        return None

    n_pend = len(pendants)
    # Adaptive spacing — wider gaps so cords don't crowd each other
    spacing = 2.0 if n_pend > 100 else (2.5 if n_pend > 50 else 3.0)

    # Proportional length scaling — normalise to this khipu's own range
    all_lengths = df["length"].replace(0, None).dropna()
    max_len = float(all_lengths.max()) if not all_lengths.empty else 50.0
    depth_scale = 5.0 / max(max_len, 1.0)   # fill ~5 depth units

    # Assign each pendant a sequential x position
    # (group_idx can be 0..457 so we cannot use it directly as x)
    pos: dict[str, tuple[float, float, float]] = {}
    for i, row in pendants.iterrows():
        x = i * spacing
        cord_length = float(row["length"] or 30.0)
        z = -(cord_length * depth_scale)   # hang downward (negative z); proportional
        pos[str(row["cord_name"])] = (x, 0.0, z)

    for _, row in subs.iterrows():
        parent_name = str(row["parent_cord"] or "")
        if parent_name not in pos:
            continue
        px, py, pz = pos[parent_name]
        depth  = float(row["hierarchy_level"])
        sub_len = float(row["length"] or 20.0)
        sub_z   = pz - (sub_len * depth_scale * 0.6)   # scale relative to parent
        off_x   = 0.5 * (1 if depth % 2 == 1 else -1)   # alternate elbow side
        pos[str(row["cord_name"])] = (px + off_x, py, sub_z)

    fig = go.Figure()

    # ── Primary cord ────────────────────────────────────────────────────────
    px_all = [pos[str(r["cord_name"])][0] for _, r in pendants.iterrows()
              if str(r["cord_name"]) in pos]
    if px_all:
        fig.add_trace(go.Scatter3d(
            x=[min(px_all) - spacing, max(px_all) + spacing],
            y=[0.0, 0.0], z=[0.0, 0.0],
            mode="lines",
            line=dict(color="#8B7355", width=12),
            name="Primary cord",
            hoverinfo="skip",
            showlegend=False,
        ))

    # ── Pendant cords ────────────────────────────────────────────────────────
    for _, row in pendants.iterrows():
        name = str(row["cord_name"])
        if name not in pos:
            continue
        cx, cy, cz = pos[name]
        cord_hex   = color_to_hex(str(row["color"] or ""))
        cord_len   = float(row["length"] or 30.0)
        fig.add_trace(go.Scatter3d(
            x=[cx, cx], y=[0.0, cy], z=[0.0, cz],
            mode="lines",
            line=dict(color=cord_hex, width=8),
            hovertext=(
                f"<b>{name}</b><br>"
                f"Color: {row['color']}<br>"
                f"Value: {row['value']}<br>"
                f"Length: {cord_len:.1f} cm<br>"
                f"Knots: {row['knots'] or '—'}"
            ),
            hoverinfo="text",
            showlegend=False,
        ))

    # ── Subsidiary cords ─────────────────────────────────────────────────────
    for _, row in subs.iterrows():
        name        = str(row["cord_name"])
        parent_name = str(row["parent_cord"] or "")
        if name not in pos or parent_name not in pos:
            continue
        px, py, pz   = pos[parent_name]
        cx, cy, cz   = pos[name]
        cord_hex     = color_to_hex(str(row["color"] or ""))
        cord_len     = float(row["length"] or 20.0)
        elbow_x      = cx
        # Draw: parent → elbow (horizontal) → subsidiary end (vertical drop)
        fig.add_trace(go.Scatter3d(
            x=[px, elbow_x, cx], y=[py, py, cy], z=[pz, pz, cz],
            mode="lines",
            line=dict(color=cord_hex, width=6),
            hovertext=(
                f"<b>{name}</b> (sub L{row['hierarchy_level']})<br>"
                f"Color: {row['color']}<br>"
                f"Value: {row['value']}<br>"
                f"Length: {cord_len:.1f} cm"
            ),
            hoverinfo="text",
            showlegend=False,
        ))

    # ── Knot markers (from knot_clusters — all 8 types) ─────────────────────
    knot_styles = {
        "S":  dict(symbol="circle",        color="#c2410c", size=8,  name="S – single"),
        "L":  dict(symbol="diamond",       color="#ca8a04", size=10, name="L – long"),
        "E":  dict(symbol="square",        color="#2563eb", size=8,  name="E – figure-8"),
        "EE": dict(symbol="square-open",   color="#7c3aed", size=9,  name="EE – dbl figure-8"),
        "SP": dict(symbol="circle-open",   color="#06b6d4", size=8,  name="SP – space"),
        "LL": dict(symbol="diamond-open",  color="#f59e0b", size=9,  name="LL – dbl long"),
        "TF": dict(symbol="cross",         color="#10b981", size=8,  name="TF – triple figure-8"),
        "BL": dict(symbol="triangle-up",   color="#f43f5e", size=8,  name="BL – back long"),
        "U":  dict(symbol="circle-open",   color="#94a3b8", size=7,  name="U – unknown"),
    }
    knot_buckets: dict[str, dict] = {
        kt: {"x": [], "y": [], "z": [], "hover": []}
        for kt in knot_styles
    }

    knots_df = _load_knot_clusters(kfg_id)
    # Build a fast lookup: cord_name → length
    cord_len_map = {str(r["cord_name"]): float(r["length"] or 30.0) for _, r in df.iterrows()}

    for _, krow in knots_df.iterrows():
        name   = str(krow["cord_name"])
        if name not in pos:
            continue
        ktype  = str(krow["knot_type"]).strip().upper()
        if ktype not in knot_buckets:
            continue
        cx, cy, cz = pos[name]
        cord_len   = cord_len_map.get(name, 30.0)
        pos_cm     = float(krow["position_cm"] or 0.0)
        # fraction along cord (clamp 0.05–0.95 to stay visually on the line)
        t = max(0.05, min(0.95, pos_cm / cord_len)) if cord_len > 0 else 0.5
        kx = cx
        ky = cy * t
        kz = cz * t
        knot_buckets[ktype]["x"].append(kx)
        knot_buckets[ktype]["y"].append(ky)
        knot_buckets[ktype]["z"].append(kz)
        n_knots = int(krow["num_knots"] or 1)
        knot_buckets[ktype]["hover"].append(
            f"<b>Cord {name}</b><br>"
            f"Type: {ktype}  ×{n_knots}<br>"
            f"@ {pos_cm:.1f} cm"
        )

    for ktype, bkt in knot_buckets.items():
        if not bkt["x"]:
            continue
        sty = knot_styles[ktype]
        fig.add_trace(go.Scatter3d(
            x=bkt["x"], y=bkt["y"], z=bkt["z"],
            mode="markers",
            marker=dict(size=sty["size"], symbol=sty["symbol"],
                        color=sty["color"], line=dict(width=1, color="white")),
            hovertext=bkt["hover"],
            hoverinfo="text",
            name=sty["name"],
            showlegend=True,
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Pendant position",
            yaxis_title="",
            zaxis_title="Cord depth",
            bgcolor="#0f172a",
            xaxis=dict(showgrid=True, gridcolor="#334155", color="#94a3b8",
                       title=dict(font=dict(color="#94a3b8"))),
            yaxis=dict(showgrid=False, showticklabels=False,
                       title=dict(text="", font=dict(color="#94a3b8"))),
            zaxis=dict(showgrid=True, gridcolor="#334155", color="#94a3b8",
                       title=dict(font=dict(color="#94a3b8"))),
            # Wide aspect: x (pendants) gets 4× the rendered space vs depth/y
            aspectmode="manual",
            aspectratio=dict(x=4, y=0.3, z=1),
            # Low frontal camera — looks along the pendant row from slight angle
            camera=dict(
                eye=dict(x=1.5, y=-1.2, z=0.5),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        legend=dict(
            x=1.0, y=0.5, yanchor="middle",
            bgcolor="rgba(15,23,42,0.85)",
            bordercolor="#334155", borderwidth=1,
            font=dict(color="#e2e8f0", size=11),
        ),
        margin=dict(l=0, r=120, t=10, b=0),
        autosize=True,
        height=700,
        showlegend=True,
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


@st.cache_data(ttl=3600)
def compute_summation_arcs(kfg_id: str) -> dict:
    """Compute PP / IP / CP / SP summation relations directly from the cord DB.

    Returns: { pattern_key: [(sum_coord, [summand_coords]), ...] }
    where each coord is (group_float, position_in_group_float).

    Patterns:
      PP  - pendant == sum of other pendants in the same cord group
      IP  - pendant == sum of same-index pendants across groups
      CP  - pendant == sum of same-color pendants across the whole khipu
      SP  - pendant == sum of its direct subsidiary cord values
             (summand_coords is empty; the pendant itself is the annotated node)
    """
    df = load_cords(kfg_id)
    if df.empty:
        return {}

    df["_v"] = pd.to_numeric(df["value"], errors="coerce")
    pendants = df[df["hierarchy_level"] == 0].copy()
    subs     = df[df["hierarchy_level"] > 0].copy()
    tol      = 0.001

    def _coord(row) -> tuple:
        return (float(row["group_idx"]), float(row["position_in_group"] or 0))

    result: dict = {}

    # ── PP: pendant == sum of other pendants in the same group ──────────────
    pp_arcs: list = []
    for _g, grp in pendants.groupby("group_idx"):
        vals = {str(r["cord_name"]): (r["_v"], _coord(r))
                for _, r in grp.iterrows() if pd.notna(r["_v"])}
        if len(vals) < 2:
            continue
        total = sum(v for v, _ in vals.values())
        for name, (val, coord) in vals.items():
            if val > 0 and abs(val - (total - val)) < tol:
                pp_arcs.append((coord, [c for n, (_, c) in vals.items() if n != name]))
    if pp_arcs:
        result["pendant_pendant_sum"] = pp_arcs

    # ── IP: pendant == sum of same-position pendants in other groups ─────────
    ip_arcs: list = []
    for _pos, grp in pendants.groupby("position_in_group"):
        vals = {str(r["cord_name"]): (r["_v"], _coord(r))
                for _, r in grp.iterrows() if pd.notna(r["_v"])}
        if len(vals) < 2:
            continue
        total = sum(v for v, _ in vals.values())
        for name, (val, coord) in vals.items():
            if val > 0 and abs(val - (total - val)) < tol:
                ip_arcs.append((coord, [c for n, (_, c) in vals.items() if n != name]))
    if ip_arcs:
        result["indexed_pendant_sum"] = ip_arcs

    # ── CP: pendant == sum of same-color pendants across the full khipu ──────
    cp_arcs: list = []
    for _color, grp in pendants.groupby("color"):
        vals = {str(r["cord_name"]): (r["_v"], _coord(r))
                for _, r in grp.iterrows() if pd.notna(r["_v"])}
        if len(vals) < 2:
            continue
        total = sum(v for v, _ in vals.values())
        for name, (val, coord) in vals.items():
            if val > 0 and abs(val - (total - val)) < tol:
                cp_arcs.append((coord, [c for n, (_, c) in vals.items() if n != name]))
    if cp_arcs:
        result["colored_pendant_sum"] = cp_arcs

    # ── SP: pendant == sum of its direct subsidiary cord values ──────────────
    if not subs.empty:
        sub_by_parent = subs.groupby("parent_cord")
        sp_arcs: list = []
        for _, prow in pendants.iterrows():
            pname = str(prow["cord_name"])
            pval  = prow["_v"]
            if pd.isna(pval) or pval <= 0:
                continue
            if pname not in sub_by_parent.groups:
                continue
            child_sum = sub_by_parent.get_group(pname)["_v"].dropna().sum()
            if abs(pval - child_sum) < tol:
                sp_arcs.append((_coord(prow), []))
        if sp_arcs:
            result["subsidiary_pendant_sum"] = sp_arcs

    return result


def build_summation_figure(
    cords_df: pd.DataFrame,
    arc_data: dict,
    enabled_patterns: set,
) -> go.Figure:
    """Cord-node layout with summation arcs bowing above the grid.

    x  = evenly spaced group position (0, 1, 2 …)
    y  = −position_in_group  (pos 0 at top = y = 0; deeper positions go down)
    Arcs bow upward (positive y, above the cord grid), coloured by pattern.
    Sum cords get a gold outer ring; summand cords get a cyan outer ring.
    """
    pendants = cords_df[cords_df["hierarchy_level"] == 0].copy()
    if pendants.empty:
        pendants = cords_df.copy()

    groups = sorted(pendants["group_idx"].dropna().unique())
    group_x: dict[float, float] = {float(g): float(i) for i, g in enumerate(groups)}

    # Build per-cord lookup: (group_float, pos_float) → {x, y, color, name, value}
    coord_map: dict[tuple, dict] = {}
    node_x, node_y, node_color, node_hover = [], [], [], []

    for _, row in pendants.sort_values(["group_idx", "position_in_group"]).iterrows():
        g = row["group_idx"]
        if pd.isna(g):
            continue
        g_f = float(g)
        p_f = float(row["position_in_group"] or 0)
        x   = group_x.get(g_f, g_f)
        y   = -p_f                        # pos 0 at top; deeper positions below
        hex_c = color_to_hex(str(row["color"] or ""))
        val   = row["value"]
        name  = str(row["cord_name"])
        coord_map[(g_f, p_f)] = dict(x=x, y=y, color=hex_c, name=name, value=val)
        node_x.append(x)
        node_y.append(y)
        node_color.append(hex_c)
        node_hover.append(
            f"<b>{name}</b><br>"
            f"Group {int(g)} · pos {int(p_f)}<br>"
            f"Color: {row['color']}<br>"
            f"Value: {val if pd.notna(val) else '—'}"
        )

    # Identify which nodes participate in enabled arcs
    sum_keys:     set = set()
    summand_keys: set = set()
    for pattern, arcs in arc_data.items():
        if pattern not in enabled_patterns:
            continue
        for sum_coord, summand_list in arcs:
            sum_keys.add(sum_coord)
            for sc in summand_list:
                summand_keys.add(sc)

    # Build value labels: Σ prefix for sum cords, plain for all others with a value
    def _fmt_val(v) -> str:
        if pd.isna(v):
            return ""
        try:
            f = float(v)
            return str(int(f)) if f == int(f) else f"{f:.2g}"
        except (ValueError, TypeError):
            return str(v)

    lbl_x, lbl_y, lbl_text, lbl_color = [], [], [], []
    for (g_f, p_f), info in coord_map.items():
        raw_lbl = _fmt_val(info["value"])
        if not raw_lbl:
            continue
        coord = (g_f, p_f)
        if coord in sum_keys:
            display = f"Σ{raw_lbl}"
            lbl_c   = "#fbbf24"           # amber-300, bright gold for sum
        else:
            display = raw_lbl
            lbl_c   = "#f1f5f9"           # near-white for all others
        lbl_x.append(info["x"])
        lbl_y.append(info["y"])
        lbl_text.append(display)
        lbl_color.append(lbl_c)

    fig = go.Figure()

    # Layer 1: all cord nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=22, color=node_color, symbol="circle",
                    line=dict(color="#334155", width=1.5)),
        text=node_hover, hovertemplate="%{text}<extra></extra>",
        showlegend=False, name="Cords",
    ))

    # Layer 2: summand ring (cyan)
    sk_x = [coord_map[k]["x"] for k in summand_keys if k in coord_map]
    sk_y = [coord_map[k]["y"] for k in summand_keys if k in coord_map]
    if sk_x:
        fig.add_trace(go.Scatter(
            x=sk_x, y=sk_y, mode="markers",
            marker=dict(size=32, color="rgba(0,0,0,0)", symbol="circle",
                        line=dict(color="#06b6d4", width=2.5)),
            hoverinfo="skip", showlegend=True, name="Summand cord",
        ))

    # Layer 3: sum cord ring (gold)
    sc_x = [coord_map[k]["x"] for k in sum_keys if k in coord_map]
    sc_y = [coord_map[k]["y"] for k in sum_keys if k in coord_map]
    if sc_x:
        fig.add_trace(go.Scatter(
            x=sc_x, y=sc_y, mode="markers",
            marker=dict(size=36, color="rgba(0,0,0,0)", symbol="circle",
                        line=dict(color="#f59e0b", width=3)),
            hoverinfo="skip", showlegend=True, name="Sum cord",
        ))

    # Layer 4: value labels — pixel-offset annotations so they clear the ring edge
    for xi, yi, txt, clr in zip(lbl_x, lbl_y, lbl_text, lbl_color):
        fig.add_annotation(
            x=xi, y=yi,
            text=txt,
            showarrow=False,
            xshift=26,          # pixels right of the data point (clears size-36 ring)
            xanchor="left",
            yanchor="middle",
            font=dict(size=11, color=clr, family="monospace"),
            bgcolor="rgba(0,0,0,0)",
        )

    # Layer 5: arc traces per pattern (bow upward) + arc equation annotations
    for pattern, arcs in arc_data.items():
        if pattern not in enabled_patterns:
            continue
        arc_color, abbr = ARC_PATTERNS.get(pattern, ("#888888", pattern))
        all_ax: list = []
        all_ay: list = []
        for sum_coord, summand_list in arcs:
            if sum_coord not in coord_map:
                continue
            sx   = coord_map[sum_coord]["x"]
            sy   = coord_map[sum_coord]["y"]
            sval = _fmt_val(coord_map[sum_coord]["value"])
            summand_vals = []
            for sc in summand_list:
                if sc not in coord_map:
                    continue
                tx = coord_map[sc]["x"]
                ty = coord_map[sc]["y"]
                ax, ay = _bezier_arc_up(sx, sy, tx, ty)
                all_ax.extend(ax)
                all_ay.extend(ay)
                summand_vals.append(_fmt_val(coord_map[sc]["value"]))
            # Annotate arc group with equation exactly at the Bezier apex (t=0.5)
            if summand_vals and summand_list and summand_list[0] in coord_map:
                tx0 = coord_map[summand_list[0]]["x"]
                ty0 = coord_map[summand_list[0]]["y"]
                # Replicate _bezier_arc_up control point then evaluate at t=0.5
                _cx  = (sx + tx0) / 2
                _cy  = max(sy, ty0) + max(1.5, abs(tx0 - sx) * 0.45)
                mid_x = _cx                                   # t=0.5 x == midpoint
                mid_y = 0.25 * sy + 0.5 * _cy + 0.25 * ty0  # t=0.5 y on quadratic
                eq    = " + ".join(summand_vals)
                if sval:
                    eq = f"{' + '.join(summand_vals)} = {sval}"
                fig.add_annotation(
                    x=mid_x, y=mid_y, text=eq,
                    showarrow=False,
                    font=dict(size=9, color=arc_color),
                    xanchor="center", yanchor="middle",
                    bgcolor="rgba(15,23,42,0.82)",
                    borderpad=3,
                )
        if all_ax:
            fig.add_trace(go.Scatter(
                x=all_ax, y=all_ay, mode="lines",
                line=dict(color=arc_color, width=2.5),
                opacity=0.85, name=abbr,
                hoverinfo="skip", showlegend=True,
            ))

    # Group index labels below the cord grid
    min_y = min(node_y) if node_y else 0
    for g, xi in group_x.items():
        fig.add_annotation(
            x=xi, y=min_y - 0.6, text=str(int(g)),
            showarrow=False, font=dict(size=9, color="#64748b"),
            xanchor="center",
        )

    n_depth   = max(abs(min_y) + 1, 1) if node_y else 1
    arc_space = 3.5 if (arc_data and enabled_patterns) else 0
    x_max     = max(group_x.values(), default=0)

    fig.update_layout(
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        xaxis=dict(
            showgrid=False, showticklabels=False, zeroline=False,
            title=dict(text="← group index (numbers below) →",
                       font=dict(size=10, color="#64748b")),
            range=[-0.7, x_max + 1.2],
        ),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, title=""),
        height=max(380, int(n_depth) * 58 + int(arc_space * 55) + 90),
        margin=dict(l=10, r=12, t=20, b=50),
        legend=dict(
            x=1.01, y=0.5, yanchor="middle",
            bgcolor="rgba(15,23,42,0.85)",
            bordercolor="#334155", borderwidth=1,
            font=dict(color="#e2e8f0", size=11),
        ),
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


def _bezier_arc_up(
    x1: float, y1: float, x2: float, y2: float, n: int = 22
) -> tuple[list, list]:
    """Quadratic Bézier arc that bows *upward* (positive y direction).

    Used by build_summation_figure where y=0 is the top of the grid and arcs
    arch above it into positive-y space.
    """
    cx = (x1 + x2) / 2
    cy = max(y1, y2) + max(1.5, abs(x2 - x1) * 0.45)
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
        colorscale="Turbo",
        hovertemplate="%{y} ∩ %{x}: %{z}<extra></extra>",
        showscale=True,
    ))

    # Per-cell annotations with adaptive text colour.
    # Turbo peaks in brightness at ~40-80% of range (yellow-green zone);
    # low end (deep blue) and high end (dark red) are both dark → need light text.
    max_v = int(cooc.max()) or 1
    n = len(labels)
    for i in range(n):
        for j in range(n):
            v = int(cooc[i, j])
            ratio = v / max_v
            fc = "#0f172a" if 0.25 < ratio < 0.82 else "#e2e8f0"
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
            row_t.append(f"{pct:.0f}%")
        z_vals.append(row_z)
        text_vals.append(row_t)

    col_labels = [short[k] for k in available]

    fig = go.Figure(go.Heatmap(
        z=z_vals, x=col_labels, y=top_provs,
        colorscale="Turbo",
        hovertemplate="%{y} · %{x}: %{text}<extra></extra>",
        text=text_vals,
        showscale=True,
        colorbar=dict(title="%", ticksuffix="%"),
    ))

    # Per-cell annotations — single line.
    # Turbo peaks in brightness ~40-80% of range; use dark text there, light elsewhere.
    max_z = max(v for row in z_vals for v in row) or 1
    for i, prov in enumerate(top_provs):
        for j, col in enumerate(col_labels):
            v = z_vals[i][j]
            ratio = v / max_z
            fc = "#0f172a" if 0.25 < ratio < 0.82 else "#e2e8f0"
            fig.add_annotation(
                x=col, y=prov,
                text=text_vals[i][j],
                showarrow=False,
                font=dict(size=10, color=fc),
            )
    # Column labels pinned to the top of the plot area so they're visible
    # when the chart is scrolled — more reliable than axis mirroring.
    for col in col_labels:
        fig.add_annotation(
            x=col, xref="x",
            y=1.0, yref="paper",
            text=f"<b>{col}</b>",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=11, color="#94a3b8"),
        )
    fig.update_layout(
        xaxis_title="Pattern",
        yaxis_title="Provenance",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        xaxis=dict(color="#94a3b8", side="bottom"),
        yaxis=dict(color="#94a3b8", dtick=1),
        margin=dict(l=0, r=0, t=30, b=40),
        height=max(350, len(top_provs) * 28 + 80),
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

    kfg_ids = flags_df["kfg_id"].values if "kfg_id" in flags_df.columns else [str(i) for i in range(len(flags_df))]

    fig = go.Figure()
    # One trace per pattern-count so legend click/double-click shows/hides points
    for cnt in sorted(set(n_patterns.tolist())):
        mask = n_patterns == cnt
        col  = palette[min(cnt, len(palette) - 1)]
        hover = [
            f"<b>{kfg_ids[i]}</b><br>{cnt} pattern{'s' if cnt != 1 else ''}"
            for i in range(len(flags_df)) if mask[i]
        ]
        fig.add_trace(go.Scatter(
            x=proj[mask, 0], y=proj[mask, 1],
            mode="markers",
            marker=dict(size=9, color=col, opacity=0.75, line=dict(width=0)),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
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
/* ── hide ALL Streamlit chrome ──────────────────────────────────────────────  */
header[data-testid="stHeader"]         { display: none !important; }
[data-testid="stToolbar"]              { display: none !important; }
[data-testid="stDecoration"]           { display: none !important; }
[data-testid="stFooterDefault"]        { display: none !important; }
[data-testid="stBottom"]               { display: none !important; }
[data-testid="stSidebar"]              { display: none !important; }
[data-testid="stSidebarCollapseButton"]{ display: none !important; }
#MainMenu                              { display: none !important; }

/* ── global background ────────────────────────────────────────────────────────  */
[data-testid="stApp"]                  { background: #0b1120; }
/* inset main content: clear 64px fixed header, 150px left nav, 44px footer */
[data-testid="stMainBlockContainer"],
.main .block-container {
    padding-top: 68px !important;
    padding-left: 150px !important;
    padding-bottom: 44px !important;
    max-width: 100% !important;
}
/* Pull ALL heading wrappers up to clear Streamlit's injected top-gap */
[data-testid="stHeadingWithActionElements"] {
    margin-top: -48px !important;
    padding-top: 0 !important;
}
/* Hide all horizontal divider lines */
[data-testid="stDivider"] { display: none !important; }
hr { display: none !important; }

/* ── fixed header bar ─────────────────────────────────────────────────────  */
.kcat-header {
    position: fixed; top: 0; left: 0; right: 0; z-index: 9000;
    display: flex; align-items: center; gap: 16px;
    height: 64px; padding: 0 24px;
    background: #0f172a; border-bottom: 2px solid #1e3a5f;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
.kcat-app-name { font-size: 1.2rem; font-weight: 700; color: #e2e8f0; white-space: nowrap; }
.kcat-badge {
    font-size: 0.67rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #64748b; background: #1e293b;
    border: 1px solid #334155; border-radius: 20px; padding: 3px 11px;
}
.kcat-stat   { font-size: 0.78rem; color: #475569; white-space: nowrap; }
.kcat-spacer { flex: 1; }
.kcat-gh-link { font-size: 0.8rem; color: #3b82f6 !important; text-decoration: none; }
.kcat-gh-link:hover { color: #60a5fa !important; }

/* ── fixed left nav bar ────────────────────────────────────────────────────  */
.kcat-nav {
    position: fixed; top: 64px; left: 0; bottom: 36px; width: 80px;
    z-index: 8999;
    background: #070f1c; border-right: 1px solid #1e293b;
    display: flex; flex-direction: column; align-items: center;
    padding-top: 16px; gap: 6px;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
.kcat-nav a {
    width: 54px; height: 54px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 12px; font-size: 1.9rem;
    text-decoration: none; color: #64748b;
    transition: background 0.12s, color 0.12s;
}
.kcat-nav a:hover  { background: #1e293b; color: #e2e8f0; }
.kcat-nav a.active { background: #1e3a5f; color: #3b82f6; }

/* ── fixed footer ───────────────────────────────────────────────────────────  */
.kcat-footer {
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 9000;
    height: 36px; padding: 0 24px;
    background: #070f1c; border-top: 1px solid #1e293b;
    display: flex; align-items: center;
    font-size: 0.72rem; color: #475569;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}

/* ── in-section khipu picker card ───────────────────────────────────────────  */
.picker-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 12px 16px; margin-bottom: 18px;
}

/* ── KPI metric cards ────────────────────────────────────────────────────────  */
[data-testid="stMetric"] {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 10px 14px 8px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #64748b !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #e2e8f0 !important;
    line-height: 1.2;
}

/* ── Analytics tabs ──────────────────────────────────────────────────────────  */
[data-baseweb="tab-list"] {
    background: #0b1120 !important;
    border-bottom: 2px solid #1e293b !important;
    gap: 4px !important;
    padding-bottom: 0 !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-bottom: none !important;
    border-radius: 8px 8px 0 0 !important;
    color: #64748b !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 10px 22px !important;
    transition: background 0.15s, color 0.15s, border-color 0.15s !important;
}
[data-baseweb="tab"]:hover {
    background: #1e293b !important;
    color: #cbd5e1 !important;
    border-color: #334155 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #1e3a5f !important;
    color: #e2e8f0 !important;
    border-color: #3b82f6 !important;
}
[data-baseweb="tab-highlight"] { background: transparent !important; }
[data-baseweb="tab-border"]    { background: transparent !important; }

/* ── Subheader section labels (st.subheader → h3) ───────────────────────  */
[data-testid="stHeadingWithActionElements"] h3 {
    font-size: 1.0rem !important;
    font-weight: 600 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    color: #cbd5e1 !important;
    border-bottom: none !important;
    padding-bottom: 0 !important;
    margin-top: 0 !important;
    margin-bottom: 4px !important;
}
</style>"""




# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Khipu Explorer",
        page_icon="🪢",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    with st.spinner("Loading corpus…"):
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

    # ── Routing via query param (‘v’) ───────────────────────────────────────────
    _VIEW_MAP = {
        "corpus":    "Corpus Browser",
        "analytics": "Analytics",
        "3dviewer":  "3D Viewer",
        "arcs":      "Summation Arcs",
    }
    _PARAM_MAP = {v: k for k, v in _VIEW_MAP.items()}
    _NAV_ITEMS = [
        ("corpus",    "🔎", "Corpus Browser"),
        ("analytics", "💡", "Analytics"),
        ("3dviewer",  "🧊", "3D Viewer"),
        ("arcs",      "Σ", "Summation Arcs"),
    ]
    _vp = st.query_params.get("v", "corpus")
    view: str = _VIEW_MAP.get(_vp, "Corpus Browser")

    # ── Fixed left nav (pure HTML — never hidden by Streamlit JS) ───────────
    _nav_links = "".join(
        f'<a href="?v={_key}" target="_self" class="{"active" if view == _name else ""}" '
        f'title="{_name}">{_icon}</a>'
        for _key, _icon, _name in _NAV_ITEMS
    )
    st.markdown(f'<div class="kcat-nav">{_nav_links}</div>', unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="kcat-footer">© 2026 Agustín Da Fieno Delucchi</div>',
        unsafe_allow_html=True,
    )

    # ── Khipu picker helper — used by 3D Viewer and Summation Arcs ───────────────
    def _khipu_picker(key_prefix: str, page_title: str = "") -> Optional[str]:
        """Render provenance filter + khipu selector in a compact 2-column row.
        Returns the selected kfg_id, or None if no match."""
        provenances = sorted(corpus["provenance"].dropna().unique())
        prov_options = ["All"] + sorted(set(_fmt_prov(p) for p in provenances))
        _prov_raw_map: dict[str, list[str]] = {}
        for p in provenances:
            _prov_raw_map.setdefault(_fmt_prov(p), []).append(p)

        p_col, k_col = st.columns([1, 2])
        prov_label = p_col.selectbox(
            "Provenance", prov_options,
            key=f"{key_prefix}_prov", label_visibility="collapsed",
        )
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
            sel = k_col.selectbox(
                "Khipu", khipu_ids,
                format_func=lambda k: k_labels.get(k, k),
                key=f"{key_prefix}_khipu", label_visibility="collapsed",
            )
        else:
            k_col.warning("No khipus match this filter.")
        return sel

    # ── Corpus Browser ─────────────────────────────────────────────────────────
    if view == "Corpus Browser":
        st.header("Corpus Browser")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Khipus", f"{len(corpus):,}")
        m2.metric("Cords", f"{corpus['cord_count'].sum():,}")
        m3.metric("Provenances", str(corpus["provenance"].nunique()))
        m4.metric("Countries", str(corpus["museum_country"].nunique()))

        display = corpus.copy()
        display["provenance"] = display["provenance"].apply(
            lambda v: _fmt_prov(v) if pd.notna(v) else "—"
        )

        st.caption(f"**{len(display):,}** khipus — select any row to open its details.")

        display_table = display.rename(columns={
            "kfg_id": "KFG ID",
            "provenance": "Provenance",
            "region": "Region",
            "museum_country": "Country",
            "museum_name": "Museum",
            "cord_count": "Cords",
        }).drop(columns=["kfg_url", "kfg_name"], errors="ignore")

        sel = st.dataframe(
            display_table,
            width="stretch",
            hide_index=True,
            height=600,
            on_select="rerun",
            selection_mode="single-row",
        )
        rows = sel.selection.rows if sel and hasattr(sel, "selection") else []
        if rows:
            _khipu_detail_modal(display_table.iloc[rows[0]]["KFG ID"])
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

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview",
            "🔬 Deep Dive",
            "🌍 Geography",
            "🧮 Pattern Space",
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
                _chart_title("Pattern Prevalence")
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
                _chart_title("Pattern Co-occurrence")
                st.caption("Khipus that simultaneously exhibit both patterns (diagonal = single-pattern count).")
                with st.expander("ℹ️ How to read this heatmap", expanded=False):
                    st.markdown(
                        "**Cells** show how many khipus express *both* the row pattern and the column pattern simultaneously.  \n"
                        "**Diagonal** = khipus that have that single pattern (same as the prevalence bar chart).  \n"
                        "**Cell colour** uses the Turbo scale: deep blue = low count, cyan/yellow-green = mid, dark red = high. "
                        "A bright or warm off-diagonal cell signals two patterns that frequently co-occur — suggesting they belong to "
                        "the same scribe tradition or accounting layer.  \n"
                        "**Cool/dark off-diagonal cells** = the pair rarely or never co-occurs."
                    )
                st.plotly_chart(build_cooccurrence_figure(flags_df), width="stretch")

            _chart_title("Pattern Complexity")
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
            _chart_title("Handedness (Left vs Right Sums)")
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

            c3, c4 = st.columns(2)
            with c3:
                _chart_title("Instance-Count Distribution")
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
                _chart_title("Sum Magnitude Distribution")
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

            _chart_title("Dual- & Multi-Summand Breakdown")
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
            _chart_title("Pattern Rate by Provenance")
            st.caption("Percentage of khipus from each provenance (top 25 by count) that exhibit each summation pattern.")
            with st.expander("ℹ️ How to read this heatmap", expanded=False):
                st.markdown(
                    "**Columns** = pattern codes (PP, IP, CP, SP, IS, GG, GSB, ADG, PSN).  \n"
                    "**Rows** = archaeological find sites, sorted by total khipu count.  \n"
                    "**Cell colour** uses the Turbo scale: deep blue = low rate, cyan/yellow-green = mid, dark red = high rate.  \n"
                    "**Cell label** — shows the percentage of khipus from that site that exhibit the pattern.  \n\n"
                    "A consistently warm/bright column across many sites indicates a corpus-wide pattern; "
                    "a warm cell in just one row suggests a pattern that may be regionally specific "
                    "or linked to a particular administrative tradition."
                )
            st.plotly_chart(build_geo_heatmap(full_df, flags_df), width="stretch")

        # ── Tab 4: Pattern Space ───────────────────────────────────────────────
        with tab4:
            _chart_title("Khipu Pattern-Space (PCA)")
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

            _chart_title("Pattern Detail Table")
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
        # Top bar: title on the left (link sits below it), filters on the right
        _h_col, _prov_col, _k_col = st.columns([3, 1, 2])
        _h_col.header("3D Viewer")

        # Provenance filter (no label shown)
        provenances = sorted(corpus["provenance"].dropna().unique())
        prov_options = ["All"] + sorted(set(_fmt_prov(p) for p in provenances))
        _prov_raw_map: dict[str, list[str]] = {}
        for p in provenances:
            _prov_raw_map.setdefault(_fmt_prov(p), []).append(p)
        prov_label = _prov_col.selectbox(
            "Provenance", prov_options,
            key="3dv_prov", label_visibility="collapsed",
        )
        pool = corpus if prov_label == "All" else corpus[
            corpus["provenance"].isin(_prov_raw_map.get(prov_label, []))
        ]

        # Khipu selector (no label shown)
        khipu_ids = pool["kfg_id"].tolist()
        selected_id: Optional[str] = None
        if khipu_ids:
            k_labels = {
                row["kfg_id"]: (
                    f"{row['kfg_id']}  {row['kfg_name'] or ''}  "
                    f"[{_fmt_prov(row['provenance'])}]"
                )
                for _, row in pool.iterrows()
            }
            selected_id = _k_col.selectbox(
                "Khipu", khipu_ids,
                format_func=lambda k: k_labels.get(k, k),
                key="3dv_khipu", label_visibility="collapsed",
            )
        else:
            _k_col.warning("No khipus match this filter.")

        if not selected_id:
            st.info("Select a khipu above.")
            return

        meta     = load_meta(selected_id)
        cords_df = load_cords(selected_id)
        n_pend   = int((cords_df["hierarchy_level"] == 0).sum())
        n_subs   = int((cords_df["hierarchy_level"] > 0).sum())
        n_knots  = sum(len(_parse_knots(str(k))) for k in cords_df["knots"] if k)

        # "View on KFG" link sits below the title in the same left column
        url = meta.get("kfg_url", "")
        if url:
            _h_col.markdown(
                f'<a href="{url}" target="_blank" '
                f'style="font-size:0.82rem;color:#3b82f6;text-decoration:none">'
                f'View on KFG ↗</a>',
                unsafe_allow_html=True,
            )

        # Compact stat cards (shorter than st.metric)
        def _stat_card(label: str, value: str) -> str:
            return (
                f'<div style="background:#1e293b;border-radius:6px;'
                f'padding:5px 12px;margin:0">'
                f'<div style="font-size:0.6rem;color:#94a3b8;text-transform:uppercase;'
                f'letter-spacing:0.06em;margin-bottom:2px">{label}</div>'
                f'<div style="font-size:1.05rem;font-weight:600;color:#e2e8f0;'
                f'line-height:1.2">{value}</div>'
                f'</div>'
            )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(_stat_card("KFG ID",       selected_id),                                   unsafe_allow_html=True)
        c2.markdown(_stat_card("Pendants",     str(n_pend)),                                   unsafe_allow_html=True)
        c3.markdown(_stat_card("Subsidiaries", str(n_subs)),                                   unsafe_allow_html=True)
        c4.markdown(_stat_card("Knots",        str(n_knots)),                                  unsafe_allow_html=True)
        c5.markdown(_stat_card("Primary cord", f"{meta.get('primary_length') or '?'} cm"),    unsafe_allow_html=True)

        st.markdown(
            '<div style="margin-top:14px;font-size:0.8rem;color:#94a3b8">'
            "Cord colours reflect the Ascher colour code. "
            "● S knot · ◆ L knot · ■ E knot"
            "</div>",
            unsafe_allow_html=True,
        )

        with st.spinner("Building 3D visualization…"):
            fig = build_3d_figure(selected_id)

        if fig:
            st.plotly_chart(fig, width='stretch')
            with st.expander("Raw cord data"):
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

    # ── Summation Arcs ─────────────────────────────────────────────────────────
    elif view == "Summation Arcs":
        # Top bar — same pattern as 3D Viewer
        _h_col, _prov_col, _k_col = st.columns([3, 1, 2])
        _h_col.header("Summation Arcs")

        provenances = sorted(corpus["provenance"].dropna().unique())
        prov_options = ["All"] + sorted(set(_fmt_prov(p) for p in provenances))
        _prov_raw_map: dict[str, list[str]] = {}
        for p in provenances:
            _prov_raw_map.setdefault(_fmt_prov(p), []).append(p)
        prov_label = _prov_col.selectbox(
            "Provenance", prov_options,
            key="arcs_prov", label_visibility="collapsed",
        )
        pool = corpus if prov_label == "All" else corpus[
            corpus["provenance"].isin(_prov_raw_map.get(prov_label, []))
        ]

        khipu_ids = pool["kfg_id"].tolist()
        selected_id: Optional[str] = None
        if khipu_ids:
            k_labels = {
                row["kfg_id"]: (
                    f"{row['kfg_id']}  {row['kfg_name'] or ''}  "
                    f"[{_fmt_prov(row['provenance'])}]"
                )
                for _, row in pool.iterrows()
            }
            selected_id = _k_col.selectbox(
                "Khipu", khipu_ids,
                format_func=lambda k: k_labels.get(k, k),
                key="arcs_khipu", label_visibility="collapsed",
            )
        else:
            _k_col.warning("No khipus match this filter.")

        if not selected_id:
            st.info("Select a khipu above.")
            return

        meta = load_meta(selected_id)
        url  = meta.get("kfg_url", "")
        if url:
            _h_col.markdown(
                f'<a href="{url}" target="_blank" '
                f'style="font-size:0.82rem;color:#3b82f6;text-decoration:none">'
                f'View on KFG ↗</a>',
                unsafe_allow_html=True,
            )

        cords_df = load_cords(selected_id)
        if cords_df.empty:
            st.warning("No cord data found for this khipu.")
            return

        pendants = cords_df[cords_df["hierarchy_level"] == 0]
        subs     = cords_df[cords_df["hierarchy_level"] > 0]
        n_groups = cords_df["group_idx"].nunique()

        # Compact stat cards
        def _stat_card_arcs(label: str, value: str) -> str:
            return (
                f'<div style="background:#1e293b;border-radius:6px;'
                f'padding:5px 12px;margin:0">'
                f'<div style="font-size:0.6rem;color:#94a3b8;text-transform:uppercase;'
                f'letter-spacing:0.06em;margin-bottom:2px">{label}</div>'
                f'<div style="font-size:1.05rem;font-weight:600;color:#e2e8f0;'
                f'line-height:1.2">{value}</div>'
                f'</div>'
            )
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(_stat_card_arcs("KFG ID",       selected_id),        unsafe_allow_html=True)
        c2.markdown(_stat_card_arcs("Pendants",     str(len(pendants))), unsafe_allow_html=True)
        c3.markdown(_stat_card_arcs("Subsidiaries", str(len(subs))),     unsafe_allow_html=True)
        c4.markdown(_stat_card_arcs("Cord groups",  str(n_groups)),      unsafe_allow_html=True)

        # ── Compute summation patterns directly from DB ─────────────────────────
        _PATTERN_FULL = {
            "pendant_pendant_sum":    "Pendant → Pendant (PP): same group",
            "indexed_pendant_sum":    "Indexed Pendant (IP): same position across groups",
            "colored_pendant_sum":    "Color-Grouped (CP): same Ascher color",
            "subsidiary_pendant_sum": "Subsidiary → Pendant (SP): pendant = sum of its subs",
        }
        st.markdown(
            '<div style="margin-top:18px;margin-bottom:8px;font-size:0.95rem;'
            'font-weight:600;color:#e2e8f0">Summation patterns</div>',
            unsafe_allow_html=True,
        )
        arc_data         = compute_summation_arcs(selected_id)
        enabled_patterns: set = set()
        available = [p for p in ARC_PATTERNS if arc_data.get(p)]

        if available:
            tog_cols = st.columns(max(1, len(available)))
            for i, p in enumerate(available):
                arc_color, abbr = ARC_PATTERNS[p]
                n_arcs = len(arc_data[p])
                full   = _PATTERN_FULL.get(p, abbr)
                tog_cols[i].markdown(
                    f'<div style="padding:8px 12px;background:#1e293b;border-radius:6px;'
                    f'border-left:3px solid {arc_color};margin-bottom:4px">'
                    f'<span style="font-size:0.88rem;font-weight:700;color:{arc_color}">'
                    f'{abbr}</span>'
                    f'<br><span style="font-size:0.82rem;color:#cbd5e1">{full}</span><br>'
                    f'<span style="font-size:0.78rem;color:#64748b">'
                    f'{n_arcs} relation{"s" if n_arcs != 1 else ""}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if tog_cols[i].checkbox("Show", value=True, key=f"arc_{p}",
                                        label_visibility="collapsed"):
                    enabled_patterns.add(p)
        else:
            st.info("🔍 No summation patterns detected for this khipu.")

        # ── Cord layout + arc figure ──────────────────────────────────────────────
        st.markdown(
            '<div style="margin-top:20px;margin-bottom:6px;font-size:0.95rem;'
            'font-weight:600;color:#e2e8f0">Cord map</div>'
            '<div style="font-size:0.82rem;color:#94a3b8;margin-bottom:6px">'
            'Each circle is a cord, coloured by Ascher code. '
            'Gold ring = sum cord &nbsp;·&nbsp; Cyan ring = summand cord. '
            'Arcs bow above the grid and connect each sum cord to its summands.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_summation_figure(cords_df, arc_data, enabled_patterns),
            width='stretch',
        )

        # ── Summation relations table ────────────────────────────────────────────
        if arc_data and enabled_patterns:
            # Build a fast lookup: (group_float, pos_float) → row
            _pos_lookup: dict[tuple, pd.Series] = {}
            for _, _r in cords_df.iterrows():
                _g = _r["group_idx"]
                _p = float(_r["position_in_group"] or 0)
                if not pd.isna(_g):
                    _pos_lookup[(float(_g), _p)] = _r

            rel_rows: list[dict] = []
            for pattern, arcs in arc_data.items():
                if pattern not in enabled_patterns:
                    continue
                arc_color, abbr = ARC_PATTERNS[pattern]
                for sum_coord, summand_list in arcs:
                    s_row = _pos_lookup.get(sum_coord)
                    s_name  = s_row["cord_name"] if s_row is not None else str(sum_coord)
                    s_val   = s_row["value"]     if s_row is not None else "—"
                    summand_parts = []
                    for sc in summand_list:
                        sc_row = _pos_lookup.get(sc)
                        sc_n   = sc_row["cord_name"] if sc_row is not None else str(sc)
                        sc_v   = sc_row["value"]     if sc_row is not None else "—"
                        summand_parts.append(f"{sc_n} ({sc_v})")
                    rel_rows.append({
                        "Pattern":  abbr,
                        "Sum cord": f"{s_name} ( {s_val} )",
                        "Summands": " + ".join(summand_parts),
                    })

            if rel_rows:
                st.markdown(
                    '<div style="margin-top:20px;margin-bottom:8px;font-size:0.95rem;'
                    'font-weight:600;color:#e2e8f0">Summation relations</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    pd.DataFrame(rel_rows), width='stretch', hide_index=True,
                )

        # ── Group summary ──────────────────────────────────────────────────────
        with st.expander("Group summary table"):
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
