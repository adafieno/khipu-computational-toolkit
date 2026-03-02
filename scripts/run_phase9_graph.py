"""
Phase 9: Graph-Based Structural Analysis
==========================================
Builds a NetworkX directed tree for each khipu from the cord parent-child
hierarchy, then computes per-khipu topology metrics that are orthogonal to
both the structural (T1/T2) and behavioral (B1–B6) features already in hand.

Tree construction:
  Each khipu becomes a rooted directed tree:
    virtual_root → pendant cords (level 0) → subsidiary cords (level 1) → …
  The virtual root represents the primary cord (spine). Pendant cords with
  parent_cord = NULL attach directly to it.

Topology metrics computed per khipu:
  depth            — maximum level in the tree (same as max_hier_level but
                     recomputed independently from graph structure)
  n_leaves         — cords with no children (terminal value-recorders)
  leaf_ratio       — n_leaves / n_cords
  mean_branching   — mean out-degree of non-leaf nodes (excl. virtual root)
  branching_cv     — coefficient of variation of out-degrees (structural
                     regularity: low CV = uniform groups, high CV = irregular)
  branching_entropy — Shannon entropy of out-degree distribution
  widest_level     — level index where cord count is greatest (primary width)
  balance_score    — 1 - (std of subtree sizes / mean subtree size) at level 0
                     (1 = perfectly balanced, lower = more lopsided)
  subtree_size_cv  — CV of level-0 subtree sizes (pendant group evenness)

Motif analysis:
  For each pendant (level-0) cord, the "pendant motif" is simply its out-degree
  (number of subsidiaries it carries). The corpus-wide distribution of pendant
  motifs reveals what branching patterns were structurally preferred.
  Common motifs (≥50 occurrences) are catalogued.

Statistical tests:
  T1: B4 (two-tier hierarchical) vs B5 (flat high-variety) — Mann-Whitney U
      on branching_entropy, branching_cv, leaf_ratio, depth
  T2: Geographic zone vs branching_entropy — Kruskal-Wallis
  T3: T1 vs T2 structural group vs topology metrics

Inputs:
  data/kfg/khipu_database.db
  data/processed/phase8_behavioral_clusters.csv

Outputs:
  data/processed/phase9_graph_metrics.csv     (per-khipu topology metrics)
  data/processed/phase9_motif_catalog.csv     (pendant motif frequencies)
  visualizations/phase9/topology_heatmap.png
  visualizations/phase9/branching_distribution.png
  visualizations/phase9/motif_bar.png
  visualizations/phase9/b4_vs_b5_topology.png
  visualizations/phase9/zone_topology.png
"""

from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import sqlite3
import networkx as nx
from scipy.stats import entropy as sci_ent, mannwhitneyu, kruskal

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "kfg" / "khipu_database.db"
P8_CSV  = ROOT / "data" / "processed" / "phase8_behavioral_clusters.csv"
OUT_DATA = ROOT / "data" / "processed"
OUT_VIZ  = ROOT / "visualizations" / "phase9"
OUT_VIZ.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ── 1. Load cord hierarchy ─────────────────────────────────────────────────
print("Loading cord hierarchy …")
conn = sqlite3.connect(DB_PATH)
cords = pd.read_sql(
    "SELECT kfg_id, cord_id, cord_name, parent_cord, hierarchy_level, value FROM cords",
    conn
)
conn.close()

cords["hierarchy_level"] = cords["hierarchy_level"].fillna(0).astype(int)
cords["parent_cord"]     = cords["parent_cord"].fillna("").str.strip()

print(f"  {len(cords):,} cords across {cords['kfg_id'].nunique()} khipus")

# ── 2. Build trees and compute metrics ────────────────────────────────────
print("Building NetworkX trees and computing topology metrics …")

VIRTUAL_ROOT = "__root__"
metrics      = []
pendant_motifs = []   # (kfg_id, pendant_cord_name, n_subsidiaries)

for kfg_id, grp in cords.groupby("kfg_id"):
    grp = grp.copy().reset_index(drop=True)

    # Build a (cord_name → cord_id) lookup for this khipu
    name_set = set(grp["cord_name"])

    # Build directed graph: root → level-0 cords → level-1+ cords
    G = nx.DiGraph()
    G.add_node(VIRTUAL_ROOT)

    for _, row in grp.iterrows():
        G.add_node(row["cord_name"], level=row["hierarchy_level"],
                   value=row["value"], cord_id=row["cord_id"])

        if row["parent_cord"] == "" or row["parent_cord"] not in name_set:
            # Attaches to virtual root (pendant or orphan)
            G.add_edge(VIRTUAL_ROOT, row["cord_name"])
        else:
            G.add_edge(row["parent_cord"], row["cord_name"])

    n_cords = len(grp)

    # Out-degrees (excl. virtual root and leaves)
    out_degrees = {n: d for n, d in G.out_degree()
                   if n != VIRTUAL_ROOT and d > 0}
    all_out     = list(out_degrees.values())     # non-zero out-degrees only

    # Level-0 (pendant) cords
    level0_nodes = grp[grp["hierarchy_level"] == 0]["cord_name"].tolist()

    # depth = max hierarchy_level
    depth = int(grp["hierarchy_level"].max())

    # leaf_ratio
    leaf_nodes = [n for n in G.nodes()
                  if n != VIRTUAL_ROOT and G.out_degree(n) == 0]
    leaf_ratio = len(leaf_nodes) / n_cords if n_cords > 0 else 0.0

    # branching stats on pendant-level (level-0) out-degrees
    # (how many subsidiaries each pendant carries)
    pendant_out = []
    for pn in level0_nodes:
        od = G.out_degree(pn)
        pendant_out.append(od)
        pendant_motifs.append({"kfg_id": kfg_id, "cord_name": pn, "n_subsidiaries": od})

    # mean/CV/entropy of pendant out-degrees
    pendant_arr = np.array(pendant_out, dtype=float)
    mean_branching   = float(pendant_arr.mean()) if len(pendant_arr) > 0 else 0.0
    branching_cv     = float(pendant_arr.std() / pendant_arr.mean()
                             if pendant_arr.mean() > 0 else 0.0)
    if len(pendant_arr) > 1:
        vc = pd.Series(pendant_arr).value_counts()
        p  = vc / vc.sum()
        branching_entropy = float(sci_ent(p, base=2))
    else:
        branching_entropy = 0.0

    # widest_level — level with most cords
    level_counts = grp["hierarchy_level"].value_counts()
    widest_level = int(level_counts.idxmax())

    # balance_score: among level-0 subtrees, how even are their sizes?
    subtree_sizes = []
    for pn in level0_nodes:
        # size of subtree rooted at this pendant (including itself)
        if nx.has_path(G, pn, pn):
            sz = 1
        try:
            desc = nx.descendants(G, pn)
            sz   = 1 + len(desc)
        except Exception:
            sz = 1
        subtree_sizes.append(sz)

    if len(subtree_sizes) > 1 and np.mean(subtree_sizes) > 0:
        subtree_size_cv  = float(np.std(subtree_sizes) / np.mean(subtree_sizes))
        balance_score    = float(1.0 - min(subtree_size_cv, 1.0))
    else:
        subtree_size_cv = 0.0
        balance_score   = 1.0

    metrics.append({
        "kfg_id":           kfg_id,
        "n_cords":          n_cords,
        "n_pendants":       len(level0_nodes),
        "n_leaves":         len(leaf_nodes),
        "leaf_ratio":       leaf_ratio,
        "depth":            depth,
        "mean_branching":   mean_branching,
        "branching_cv":     branching_cv,
        "branching_entropy": branching_entropy,
        "widest_level":     widest_level,
        "balance_score":    balance_score,
        "subtree_size_cv":  subtree_size_cv,
    })

metrics_df = pd.DataFrame(metrics)
motif_df   = pd.DataFrame(pendant_motifs)

print(f"  topology metrics for {len(metrics_df)} khipus")
print(f"  {len(motif_df):,} pendant-motif records")

# ── 3. Motif catalog ───────────────────────────────────────────────────────
print("\nBuilding motif catalog …")
motif_counts = motif_df["n_subsidiaries"].value_counts().sort_index()

# Label motifs
def motif_label(n):
    if n == 0:  return "pure pendant (no subs)"
    if n == 1:  return "1 subsidiary"
    return f"{n} subsidiaries"

motif_catalog = pd.DataFrame({
    "n_subsidiaries": motif_counts.index,
    "count":          motif_counts.values,
    "label":          [motif_label(n) for n in motif_counts.index],
    "pct_pendants":   motif_counts.values / len(motif_df) * 100,
})
motif_catalog.to_csv(OUT_DATA / "phase9_motif_catalog.csv", index=False)
print("  → phase9_motif_catalog.csv saved")
print(motif_catalog[motif_catalog["count"] >= 20].to_string(index=False))

# ── 4. Merge with Phase 8 behavioral clusters ─────────────────────────────
p8 = pd.read_csv(P8_CSV)[["kfg_id", "beh_label", "typology_label", "geo_zone"]]
metrics_df = metrics_df.merge(p8, on="kfg_id", how="left")
metrics_df.to_csv(OUT_DATA / "phase9_graph_metrics.csv", index=False)
print("  → phase9_graph_metrics.csv saved")

# ── 5. Topology summary by behavioral cluster ──────────────────────────────
print("\n── Topology by behavioral cluster ───────────────────────────────────")
TOPO_FEATURES = ["depth", "leaf_ratio", "mean_branching",
                 "branching_cv", "branching_entropy",
                 "balance_score", "subtree_size_cv"]

topo_profile = metrics_df.groupby("beh_label")[TOPO_FEATURES].mean()
topo_profile["n"] = metrics_df.groupby("beh_label").size()
print(topo_profile.to_string())

# ── 6. Topology heatmap ────────────────────────────────────────────────────
beh_labels = sorted(metrics_df["beh_label"].dropna().unique())

heat_df   = topo_profile[TOPO_FEATURES].T
heat_norm = heat_df.subtract(heat_df.min(axis=1), axis=0)
denom     = (heat_df.max(axis=1) - heat_df.min(axis=1)).replace(0, 1)
heat_norm = heat_norm.divide(denom, axis=0)

feat_labels = [
    "Tree depth (max level)",
    "Leaf ratio",
    "Mean branching (pendants)",
    "Branching CV (regularity)",
    "Branching entropy (bits)",
    "Balance score (0–1)",
    "Subtree size CV",
]

fig, ax = plt.subplots(figsize=(max(8, len(beh_labels) * 1.5), 6))
im = ax.imshow(heat_norm.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(beh_labels)))
ax.set_xticklabels(beh_labels, fontsize=11, fontweight="bold")
ax.set_yticks(range(len(TOPO_FEATURES)))
ax.set_yticklabels(feat_labels, fontsize=9)
for i in range(len(TOPO_FEATURES)):
    for j, bl in enumerate(beh_labels):
        val = heat_df.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                color="black" if heat_norm.values[i, j] < 0.6 else "white")
ax.set_title(
    "Phase 9: Tree Topology Profile by Behavioral Cluster\n"
    "(cell = raw mean; colour = row-normalised 0–1)",
    fontsize=11,
)
plt.colorbar(im, ax=ax, label="Relative value (0=min, 1=max)")
plt.tight_layout()
fig.savefig(OUT_VIZ / "topology_heatmap.png", dpi=150)
plt.close(fig)
print("\n  → topology_heatmap.png saved")

# ── 7. Branching distribution ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Phase 9: Pendant Branching Factor Distribution", fontsize=12)

ax = axes[0]
# Histogram of pendant out-degrees, capped at 20
clipped = motif_df["n_subsidiaries"].clip(upper=20)
bins = np.arange(-0.5, 21.5, 1)
ax.hist(clipped, bins=bins, color="#2c7bb6", edgecolor="white", alpha=0.85)
ax.set_xlabel("Number of subsidiaries per pendant (capped at 20)")
ax.set_ylabel("Pendant-cord count")
ax.set_title("Corpus-wide pendant motif distribution")
# Mark most common
top_motif = motif_catalog.nlargest(1, "count").iloc[0]
ax.axvline(top_motif["n_subsidiaries"], color="red", linestyle="--",
           label=f"Mode = {top_motif['n_subsidiaries']} subs ({top_motif['pct_pendants']:.1f}%)")
ax.legend(fontsize=8)

ax = axes[1]
# Top 12 motifs as bar chart
top12 = motif_catalog.nlargest(12, "count").sort_values("n_subsidiaries")
ax.barh(top12["label"], top12["count"], color="#2c7bb6", edgecolor="white", alpha=0.85)
ax.set_xlabel("Number of pendant cords")
ax.set_title("Top 12 pendant motifs")
for i, (cnt, pct) in enumerate(zip(top12["count"], top12["pct_pendants"])):
    ax.text(cnt + 10, i, f"{pct:.1f}%", va="center", fontsize=8)

plt.tight_layout()
fig.savefig(OUT_VIZ / "branching_distribution.png", dpi=150)
plt.close(fig)
print("  → branching_distribution.png saved")

# ── 8. B4 vs B5 direct comparison ─────────────────────────────────────────
print("\n── Statistical tests ─────────────────────────────────────────────────")

b4 = metrics_df[metrics_df["beh_label"] == "B4"]
b5 = metrics_df[metrics_df["beh_label"] == "B5"]
print(f"\nB4 (two-tier hierarchical, n={len(b4)}) vs B5 (flat high-variety, n={len(b5)})")
print(f"{'Metric':<25} {'B4 median':>10} {'B5 median':>10} {'U-stat':>10} {'p':>10} {'sig':>5}")
print("-" * 75)

compare_metrics = ["depth", "leaf_ratio", "mean_branching",
                   "branching_cv", "branching_entropy", "balance_score"]
b4_vs_b5_results = []
for m in compare_metrics:
    a = b4[m].dropna().values
    b = b5[m].dropna().values
    if len(a) < 3 or len(b) < 3:
        continue
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"{m:<25} {np.median(a):>10.3f} {np.median(b):>10.3f} {u:>10.1f} {p:>10.4f} {sig:>5}")
    b4_vs_b5_results.append((m, np.median(a), np.median(b), p, sig))

# T2: geographic zone vs branching_entropy
zone_groups = [grp["branching_entropy"].dropna().values
               for _, grp in metrics_df[metrics_df["geo_zone"].notna()].groupby("geo_zone")
               if len(grp) >= 3]
if len(zone_groups) >= 2:
    kw_stat, kw_p = kruskal(*zone_groups)
    print(f"\nKruskal-Wallis branching_entropy × geo_zone: stat={kw_stat:.3f}  p={kw_p:.4f}  "
          f"{'✅' if kw_p < 0.05 else '❌'}")

# T3: T1 vs T2
t1 = metrics_df[metrics_df["typology_label"] == "T1"]
t2 = metrics_df[metrics_df["typology_label"] == "T2"]
print(f"\nT1 (n={len(t1)}) vs T2 (n={len(t2)}) topology:")
for m in ["depth", "mean_branching", "branching_entropy", "balance_score"]:
    a = t1[m].dropna().values
    b = t2[m].dropna().values
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"  {m:<25} T1={np.median(a):.3f}  T2={np.median(b):.3f}  p={p:.4f} {sig}")

# ── 9. B4 vs B5 visualisation ─────────────────────────────────────────────
compare_viz = [("depth",            "Tree depth",         True),
               ("branching_entropy","Branching entropy",  True),
               ("leaf_ratio",       "Leaf ratio",         False),
               ("balance_score",    "Balance score",      False)]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Phase 9: B4 (Two-tier Hierarchical) vs B5 (Flat High-variety) — Tree Topology",
             fontsize=11)

palette_b4b5 = {"B4": "#e66101", "B5": "#5e3c99"}
for ax, (metric, label, higher_b4) in zip(axes, compare_viz):
    data_b4 = b4[metric].dropna().values
    data_b5 = b5[metric].dropna().values
    bp = ax.boxplot([data_b4, data_b5],
                    tick_labels=["B4", "B5"], patch_artist=True)
    bp["boxes"][0].set_facecolor(palette_b4b5["B4"]); bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor(palette_b4b5["B5"]); bp["boxes"][1].set_alpha(0.75)
    ax.set_title(label, fontsize=10)
    ax.set_ylabel(metric)
    # Add p-value annotation
    matched = [r for r in b4_vs_b5_results if r[0] == metric]
    if matched:
        p_val = matched[0][3]
        sig   = matched[0][4]
        ax.text(1.5, ax.get_ylim()[1] * 0.95,
                f"p={p_val:.4f} {sig}", ha="center", fontsize=8, color="darkred")

plt.tight_layout()
fig.savefig(OUT_VIZ / "b4_vs_b5_topology.png", dpi=150)
plt.close(fig)
print("  → b4_vs_b5_topology.png saved")

# ── 10. Zone topology ──────────────────────────────────────────────────────
zone_df = metrics_df[metrics_df["geo_zone"].notna()].copy()
zone_means = zone_df.groupby("geo_zone")[["branching_entropy", "depth",
                                          "mean_branching", "balance_score"]].mean()

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Phase 9: Tree Topology by Geographic Zone", fontsize=12)

zone_palette = plt.cm.Set2.colors
metrics_viz  = [("branching_entropy", "Mean branching entropy (bits)"),
                ("depth",             "Mean tree depth"),
                ("mean_branching",    "Mean pendant branching factor"),
                ("balance_score",     "Mean balance score")]

for ax, (metric, label) in zip(axes.flat, metrics_viz):
    vals   = zone_means[metric].sort_values()
    colors = [zone_palette[i % len(zone_palette)] for i in range(len(vals))]
    ax.barh(range(len(vals)), vals.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(vals.index, fontsize=8)
    ax.set_xlabel(label)
    corpus_mean = metrics_df[metric].mean()
    ax.axvline(corpus_mean, color="red", linestyle="--", linewidth=1,
               label=f"Corpus mean={corpus_mean:.2f}")
    ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig(OUT_VIZ / "zone_topology.png", dpi=150)
plt.close(fig)
print("  → zone_topology.png saved")

# ── 11. Summary ───────────────────────────────────────────────────────────
print(f"\n══ Phase 9 Complete ══════════════════════════════════════════════════")
print(metrics_df.groupby("beh_label")[TOPO_FEATURES].median().to_string())
print()
for f in sorted(OUT_VIZ.glob("*.png")):
    print(f"    {f.relative_to(ROOT)}")
for f in sorted(OUT_DATA.glob("phase9_*.csv")):
    print(f"    {f.relative_to(ROOT)}")
