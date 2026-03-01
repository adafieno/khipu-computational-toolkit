"""
Khipu Explorer — Standalone Local Browser
==========================================
Interactive Streamlit app for exploring the KFG khipu database.

Three views:
    - Corpus Browser  : filterable / sortable table of all 709 khipus
    - 3D Viewer       : interactive Plotly 3D cord structure for a selected khipu
    - X-Ray View      : cord group color map + summation preview (full arc view coming)

Usage:
    streamlit run scripts/browse.py

Requirements (already in requirements.txt):
    pip install streamlit plotly pandas
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

    SPACING   = 1.4   # horizontal gap between pendants
    SUB_X_OFF = 0.4   # x offset for each subsidiary level

    pendants = df[df["hierarchy_level"] == 1].reset_index(drop=True)
    subs     = df[df["hierarchy_level"] > 1].sort_values("hierarchy_level").reset_index(drop=True)

    # position lookup: cord_name → (x, y, z)
    pos: dict[str, tuple[float, float, float]] = {}

    xs, ys, zs, colors, texts = [], [], [], [], []
    edge_x, edge_y, edge_z = [], [], []

    # Pendants hang vertically from the primary cord (y=0 plane)
    for i, row in pendants.iterrows():
        x = float(row["position"] or i) * SPACING
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
        depth = float(row["hierarchy_level"] - 1)
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
    pendants = cords_df[cords_df["hierarchy_level"] == 1].copy()
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Khipu Explorer",
        page_icon="🧶",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        "<style>.block-container{padding-top:1.5rem}</style>",
        unsafe_allow_html=True,
    )

    corpus = load_corpus()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🧶 Khipu Explorer")
        st.caption(f"KFG · {len(corpus):,} khipus · {corpus['cord_count'].sum():,} cords")

        view = st.radio(
            "View",
            ["Corpus Browser", "Analytics", "3D Viewer", "X-Ray View"],
            index=0,
        )

        st.divider()
        st.subheader("Select khipu")

        # Provenance filter
        provenances = ["All"] + sorted(
            str(p) for p in corpus["provenance"].dropna().unique()
        )
        prov = st.selectbox("Provenance", provenances, key="prov_filter")
        pool = corpus if prov == "All" else corpus[corpus["provenance"] == prov]

        khipu_ids = pool["kfg_id"].tolist()
        labels = {
            row["kfg_id"]: f"{row['kfg_id']}  {row['kfg_name'] or ''}  [{row['provenance'] or '—'}]"
            for _, row in pool.iterrows()
        }

        selected_id: Optional[str] = None
        if khipu_ids:
            selected_id = st.selectbox(
                "Khipu",
                khipu_ids,
                format_func=lambda k: labels.get(k, k),
                key="khipu_select",
            )
        else:
            st.warning("No khipus for this filter.")

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
            use_container_width=True,
            hide_index=True,
            height=600,
        )
    # ── Analytics ───────────────────────────────────────────────────────────────────────
    elif view == "Analytics":
        flags_df = load_analytics_data()
        n        = len(flags_df)
        st.header("Corpus Analytics")
        st.caption(
            f"Pattern prevalence and co-occurrence across {n:,} KFG khipus · "
            "sourced from authoritative KFG checks/ ground-truth files"
        )

        if flags_df.empty:
            st.warning(
                "KFG checks/ directory not found. "
                "Run `python scripts/build_kfg_database.py` and ensure the "
                "checks/ directory is present."
            )
        else:
            pat_cols    = [k for k, *_ in PATTERN_CONFIG]
            any_pattern = (flags_df[pat_cols].sum(axis=1) > 0)
            m1, m2, m3  = st.columns(3)
            m1.metric("KFG khipus", f"{n:,}")
            m2.metric("With ≥1 pattern", f"{int(any_pattern.sum()):,}")
            m3.metric("Pattern coverage", f"{any_pattern.mean() * 100:.1f}%")

            st.divider()

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Pattern Prevalence")
                st.caption("Khipus with ≥1 instance of each Ascher summation pattern type")
                st.plotly_chart(build_prevalence_figure(flags_df), use_container_width=True)
            with c2:
                st.subheader("Pattern Co-occurrence")
                st.caption(
                    "Number of khipus that exhibit both patterns simultaneously. "
                    "Diagonal = khipus with that single pattern."
                )
                st.plotly_chart(build_cooccurrence_figure(flags_df), use_container_width=True)
    # ── 3D Viewer ──────────────────────────────────────────────────────────────
    elif view == "3D Viewer":
        if not selected_id:
            st.info("Select a khipu in the sidebar.")
            return

        meta = load_meta(selected_id)
        st.header(f"3D Viewer — {meta.get('kfg_name') or selected_id}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("KFG ID", selected_id)
        c2.metric("Provenance", str(meta.get("provenance") or "—"))
        c3.metric("Museum", str(meta.get("museum_name") or "—")[:30])
        c4.metric("Primary cord", f"{meta.get('primary_length') or '?'} cm")

        url = meta.get("kfg_url", "")
        if url:
            st.markdown(f"[View on KFG ↗]({url})")

        with st.spinner("Building 3D visualization…"):
            fig = build_3d_figure(selected_id)

        if fig:
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Raw cord data"):
                cords_df = load_cords(selected_id)
                st.dataframe(
                    cords_df[[
                        "cord_name", "hierarchy_level", "parent_cord",
                        "color", "value", "length", "knots",
                        "group_idx", "position_in_group",
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.warning("No cord data found for this khipu.")

    # ── X-Ray View ─────────────────────────────────────────────────────────────
    elif view == "X-Ray View":
        if not selected_id:
            st.info("Select a khipu in the sidebar.")
            return

        meta = load_meta(selected_id)
        st.header(f"X-Ray View — {meta.get('kfg_name') or selected_id}")

        url = meta.get("kfg_url", "")
        if url:
            st.markdown(f"[View on KFG ↗]({url})")

        cords_df = load_cords(selected_id)
        if cords_df.empty:
            st.warning("No cord data found for this khipu.")
            return

        pendants    = cords_df[cords_df["hierarchy_level"] == 1]
        subs        = cords_df[cords_df["hierarchy_level"] > 1]
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
        st.plotly_chart(xray_fig, use_container_width=True)

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
        st.dataframe(groups_df, use_container_width=True, hide_index=True)




if __name__ == "__main__":
    main()
