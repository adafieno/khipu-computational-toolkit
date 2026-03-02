"""
Phase 5: Color Analysis
=======================
Five analyses on KFG cord color data:

1. Color vocabulary — frequency table, simple vs compound colors
2. White cord first-position hypothesis — do khipus with white cords in first
   group position have higher summation-pattern rates?
3. Color diversity by cluster (Simple/Complex) and geo_zone
4. Color-value correlation — do value distributions differ by color code?
5. Color co-occurrence heatmap — which color pairs co-occur most across khipus?

Inputs:
  data/kfg/khipu_database.db
  data/processed/phase3_clusters.csv   (cluster + geo_zone labels)

Outputs:
  data/processed/phase5_color_vocab.csv
  data/processed/phase5_color_diversity.csv
  data/processed/phase5_stat_results.csv
  visualizations/phase5/color_vocab.png
  visualizations/phase5/white_cord_analysis.png
  visualizations/phase5/color_diversity_by_cluster.png
  visualizations/phase5/color_value_correlation.png
  visualizations/phase5/color_cooccurrence.png
"""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "kfg" / "khipu_database.db"
PHASE3_CSV = ROOT / "data" / "processed" / "phase3_clusters.csv"
OUT_DATA = ROOT / "data" / "processed"
OUT_VIZ = ROOT / "visualizations" / "phase5"
OUT_VIZ.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading KFG database …")
conn = sqlite3.connect(DB_PATH)

# Cord-level data: color, value, position, hierarchy
cords = pd.read_sql(
    """
    SELECT cord_id, kfg_id, hierarchy_level, position_in_group,
           group_idx, color, value
    FROM cords
    """,
    conn,
)

# Normalised per-component colors (e.g. "W:MB" → rows W, MB)
cord_colors = pd.read_sql(
    "SELECT cord_id, color_code, sequence_ord FROM cord_colors", conn
)
conn.close()

# Phase-3 feature matrix (cluster + geo_zone)
phase3 = pd.read_csv(PHASE3_CSV)
phase3["cluster_label"] = phase3["cluster"].map({0: "Simple", 1: "Complex"})

print(f"  {len(cords):,} cords | {len(cord_colors):,} color entries | "
      f"{len(phase3)} khipus in phase3")

# Numeric value: coerce to float
cords["value_num"] = pd.to_numeric(cords["value"], errors="coerce")

# Compound flag: original color string contains ":"
cords["is_compound"] = cords["color"].str.contains(":", na=False)

# ============================================================================
# 1. Color vocabulary
# ============================================================================
print("\n── 1. Color vocabulary ──────────────────────────────────────────────")

# Frequency of each normalised color code across all cord entries
vocab = (
    cord_colors.groupby("color_code")
    .size()
    .reset_index(name="n_entries")
    .sort_values("n_entries", ascending=False)
    .reset_index(drop=True)
)
vocab["pct_entries"] = vocab["n_entries"] / len(cord_colors) * 100

# How many unique khipus carry each color?
cord_to_khipu = cords[["cord_id", "kfg_id"]].drop_duplicates()
color_khipu = cord_colors.merge(cord_to_khipu, on="cord_id")
vocab_khipus = (
    color_khipu.groupby("color_code")["kfg_id"]
    .nunique()
    .reset_index(name="n_khipus")
)
vocab = vocab.merge(vocab_khipus, on="color_code")
vocab["pct_khipus"] = vocab["n_khipus"] / phase3["kfg_id"].nunique() * 100

n_simple = (~cords["is_compound"]).sum()
n_compound = cords["is_compound"].sum()
pct_compound = n_compound / len(cords) * 100
print(f"  Total cord records: {len(cords):,}")
print(f"  Simple color cords: {n_simple:,} ({100-pct_compound:.1f}%)")
print(f"  Compound color cords: {n_compound:,} ({pct_compound:.1f}%)")
print(f"  Distinct color codes (normalised): {len(vocab)}")
print(f"  Top 10 codes:")
print(vocab.head(10).to_string(index=False))

vocab.to_csv(OUT_DATA / "phase5_color_vocab.csv", index=False)
print("  → phase5_color_vocab.csv saved")

# Visualisation: horizontal bar, top 30 codes
fig, ax = plt.subplots(figsize=(8, 10))
top30 = vocab.head(30).copy()
colors_bar = ["#d95f02" if c == "W" else "#1b9e77" for c in top30["color_code"]]
ax.barh(top30["color_code"][::-1], top30["n_entries"][::-1], color=colors_bar[::-1])
ax.set_xlabel("Number of cord-color entries", fontsize=11)
ax.set_title("KFG Color Vocabulary — Top 30 Codes\n(normalised; compound cords split by component)", fontsize=12)
ax.axvline(0, color="black", linewidth=0.5)
w_patch = mpatches.Patch(color="#d95f02", label="White (W)")
other_patch = mpatches.Patch(color="#1b9e77", label="Other")
ax.legend(handles=[w_patch, other_patch], fontsize=9)
plt.tight_layout()
fig.savefig(OUT_VIZ / "color_vocab.png", dpi=150)
plt.close(fig)
print("  → color_vocab.png saved")

# ============================================================================
# 2. White cord first-position hypothesis
# ============================================================================
print("\n── 2. White cord first-position hypothesis ──────────────────────────")

# Focus on pendant-level cords (hierarchy_level == 0)
pendants = cords[cords["hierarchy_level"] == 0].copy()

# For each khipu: does any cord group have position_in_group==1 with color starting "W"?
first_pos = pendants[pendants["position_in_group"] == 1].copy()
first_pos["is_white_first"] = first_pos["color"].str.startswith("W")

khipu_white_first = (
    first_pos.groupby("kfg_id")["is_white_first"]
    .any()
    .reset_index(name="has_white_first_cord")
)

# Fraction of cord groups with white first cord (more granular)
group_first = first_pos.groupby("kfg_id").agg(
    n_groups_with_first=("position_in_group", "count"),
    n_groups_white_first=("is_white_first", "sum"),
).reset_index()
group_first["frac_groups_white_first"] = (
    group_first["n_groups_white_first"] / group_first["n_groups_with_first"]
)

# Also: fraction of white cords overall (all levels)
white_overall = (
    cords.assign(is_white=cords["color"].str.startswith("W", na=False))
    .groupby("kfg_id")["is_white"]
    .agg(["sum", "count"])
    .reset_index()
    .rename(columns={"sum": "n_white", "count": "n_cords_total"})
)
white_overall["frac_white"] = white_overall["n_white"] / white_overall["n_cords_total"]

# Merge with phase3
df_white = (
    phase3[["kfg_id", "cluster_label", "n_pattern_types",
            "has_pp", "has_ip", "has_cp", "has_sp",
            "has_gg", "has_gsb", "has_is", "has_psn", "has_adg",
            "geo_zone", "n_cords"]]
    .merge(khipu_white_first, on="kfg_id", how="left")
    .merge(group_first[["kfg_id", "frac_groups_white_first"]], on="kfg_id", how="left")
    .merge(white_overall[["kfg_id", "frac_white"]], on="kfg_id", how="left")
)
df_white["has_white_first_cord"] = df_white["has_white_first_cord"].fillna(False)
df_white["any_pattern"] = (df_white["n_pattern_types"] > 0).astype(int)

n_has_wf = df_white["has_white_first_cord"].sum()
pct_has_wf = n_has_wf / len(df_white) * 100
print(f"  Khipus with ≥1 white first-group cord: {n_has_wf} ({pct_has_wf:.1f}%)")

# Compare n_pattern_types between groups
wf_yes = df_white[df_white["has_white_first_cord"]]["n_pattern_types"]
wf_no = df_white[~df_white["has_white_first_cord"]]["n_pattern_types"]
u_stat, p_val = stats.mannwhitneyu(wf_yes, wf_no, alternative="greater")
print(f"  Mean pattern types — with white first: {wf_yes.mean():.2f}, "
      f"without: {wf_no.mean():.2f}")
print(f"  Mann-Whitney U (greater): U={u_stat:.0f}, p={p_val:.4f}")

# Compare fraction complex
complex_rate_with = df_white[df_white["has_white_first_cord"]]["cluster_label"].eq("Complex").mean()
complex_rate_without = df_white[~df_white["has_white_first_cord"]]["cluster_label"].eq("Complex").mean()
print(f"  Complex rate — with: {complex_rate_with:.1%}, without: {complex_rate_without:.1%}")

# Chi-square: white_first × complex
ct = pd.crosstab(df_white["has_white_first_cord"], df_white["cluster_label"])
chi2_wf, p_chi2_wf, dof_wf, _ = stats.chi2_contingency(ct)
print(f"  Chi-square (white_first × cluster): χ²={chi2_wf:.2f}, p={p_chi2_wf:.4f}")

# Visualisation: side-by-side bars for mean n_pattern_types + complex rate
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("White Cord First-Position Hypothesis\n"
             "(Does having a white cord at position 1 of any group predict summation complexity?)",
             fontsize=11)

# Left: mean n_pattern_types
grp_means = df_white.groupby("has_white_first_cord")["n_pattern_types"].agg(["mean", "sem"]).reset_index()
ax = axes[0]
labels = ["No white\nfirst cord", "Has white\nfirst cord"]
vals_means = grp_means["mean"].values
vals_sems = grp_means["sem"].values
bars = ax.bar(labels, vals_means, yerr=vals_sems, capsize=5,
              color=["#7570b3", "#d95f02"], edgecolor="black", width=0.5)
ax.set_ylabel("Mean pattern types (±SEM)")
ax.set_title(f"Pattern Types\n(Mann-Whitney p={p_val:.4f})")
for bar, v in zip(bars, vals_means):
    ax.text(bar.get_x() + bar.get_width() / 2, v + vals_sems[vals_means.tolist().index(v)] + 0.02,
            f"{v:.2f}", ha="center", va="bottom", fontsize=10)

# Right: complex rate
ax = axes[1]
rates = [complex_rate_without * 100, complex_rate_with * 100]
bars = ax.bar(labels, rates, color=["#7570b3", "#d95f02"], edgecolor="black", width=0.5)
ax.set_ylabel("Complex cluster rate (%)")
ax.set_title(f"Fraction Complex Khipus\n(χ²={chi2_wf:.2f}, p={p_chi2_wf:.4f})")
for bar, v in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.1f}%",
            ha="center", va="bottom", fontsize=10)

plt.tight_layout()
fig.savefig(OUT_VIZ / "white_cord_analysis.png", dpi=150)
plt.close(fig)
print("  → white_cord_analysis.png saved")

# ============================================================================
# 3. Color diversity by cluster and geo_zone
# ============================================================================
print("\n── 3. Color diversity ───────────────────────────────────────────────")

# Per-khipu: number of distinct normalised color codes
color_div = (
    cord_colors.merge(cord_to_khipu, on="cord_id")
    .groupby("kfg_id")["color_code"]
    .nunique()
    .reset_index(name="n_unique_colors")
)

df_div = phase3[["kfg_id", "cluster_label", "geo_zone", "n_cords"]].merge(
    color_div, on="kfg_id", how="left"
)
df_div["n_unique_colors"] = df_div["n_unique_colors"].fillna(0)

# Also save full diversity table
df_div.to_csv(OUT_DATA / "phase5_color_diversity.csv", index=False)
print("  → phase5_color_diversity.csv saved")

print(f"  Mean unique colors — Simple: {df_div[df_div['cluster_label']=='Simple']['n_unique_colors'].mean():.1f}, "
      f"Complex: {df_div[df_div['cluster_label']=='Complex']['n_unique_colors'].mean():.1f}")
u2, p2 = stats.mannwhitneyu(
    df_div[df_div["cluster_label"] == "Complex"]["n_unique_colors"],
    df_div[df_div["cluster_label"] == "Simple"]["n_unique_colors"],
    alternative="greater",
)
print(f"  Mann-Whitney U (Complex > Simple): U={u2:.0f}, p={p2:.6f}")

# Zone order (drop zones with n<5 for display)
zone_counts = df_div["geo_zone"].value_counts()
valid_zones = zone_counts[zone_counts >= 5].index.tolist()
ZONE_ORDER = [z for z in [
    "Central Coast", "Cañete–Pisco", "Ica & Paracas",
    "Nazca & Far South", "Arica & N. Chile",
    "North Peru Coast", "Southern Highlands", "Chachapoyas"
] if z in valid_zones]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Color Diversity (Unique Normalised Color Codes per Khipu)", fontsize=12)

# Left: by cluster
ax = axes[0]
cluster_order = ["Simple", "Complex"]
data_to_plot = [df_div[df_div["cluster_label"] == c]["n_unique_colors"].values for c in cluster_order]
bp = ax.boxplot(data_to_plot, patch_artist=True, notch=False,
                medianprops={"color": "black", "linewidth": 2})
colors_box = ["#7570b3", "#d95f02"]
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"Simple\n(n={df_div[df_div['cluster_label']=='Simple'].shape[0]})",
                    f"Complex\n(n={df_div[df_div['cluster_label']=='Complex'].shape[0]})"])
ax.set_ylabel("Unique color codes")
ax.set_title(f"By Cluster\n(Mann-Whitney p={p2:.1e})")

# Right: by geo_zone
ax = axes[1]
zone_data = [df_div[df_div["geo_zone"] == z]["n_unique_colors"].values for z in ZONE_ORDER]
zone_ns = [len(d) for d in zone_data]
bp2 = ax.boxplot(zone_data, patch_artist=True, notch=False,
                 medianprops={"color": "black", "linewidth": 2})
for patch in bp2["boxes"]:
    patch.set_facecolor("#2ca25f")
    patch.set_alpha(0.7)
ax.set_xticks(range(1, len(ZONE_ORDER) + 1))
ax.set_xticklabels([f"{z}\n(n={n})" for z, n in zip(ZONE_ORDER, zone_ns)],
                   rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Unique color codes")
ax.set_title("By Geographic Zone")

plt.tight_layout()
fig.savefig(OUT_VIZ / "color_diversity_by_cluster.png", dpi=150)
plt.close(fig)
print("  → color_diversity_by_cluster.png saved")

# ============================================================================
# 4. Color-value correlation
# ============================================================================
print("\n── 4. Color-value correlation ───────────────────────────────────────")

# Merge cord values with primary (first) color component (sequence_ord == 0)
primary_color = cord_colors[cord_colors["sequence_ord"] == 0][["cord_id", "color_code"]]
cord_val_color = (
    cords[["cord_id", "value_num"]]
    .dropna(subset=["value_num"])
    .merge(primary_color, on="cord_id")
)
cord_val_color = cord_val_color[cord_val_color["value_num"] > 0]  # exclude zeros

# Top 12 color codes by frequency among valued cords
top_colors = cord_val_color["color_code"].value_counts().head(12).index.tolist()
cv_subset = cord_val_color[cord_val_color["color_code"].isin(top_colors)]

# Kruskal-Wallis test
groups = [grp["value_num"].values for _, grp in cv_subset.groupby("color_code")]
kw_stat, kw_p = stats.kruskal(*groups)
print(f"  Non-zero cord values by color: Kruskal-Wallis H={kw_stat:.2f}, p={kw_p:.4e}")
print(f"  Top 12 color codes tested: {top_colors}")

# Medians per color
medians = cv_subset.groupby("color_code")["value_num"].median().sort_values(ascending=False)
print("  Median non-zero value by color:")
print(medians.to_string())

# Visualisation: box plot on log scale
ordered_colors = medians.index.tolist()
fig, ax = plt.subplots(figsize=(12, 5))
data_boxes = [cv_subset[cv_subset["color_code"] == c]["value_num"].values for c in ordered_colors]
ns = [len(d) for d in data_boxes]
bp = ax.boxplot(data_boxes, patch_artist=True, showfliers=False,
                medianprops={"color": "black", "linewidth": 2})
palette = plt.cm.tab20.colors
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(palette[i % len(palette)])
    patch.set_alpha(0.8)
ax.set_yscale("log")
ax.set_xticks(range(1, len(ordered_colors) + 1))
ax.set_xticklabels([f"{c}\n(n={n:,})" for c, n in zip(ordered_colors, ns)],
                   rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Cord value (log scale, zeros excluded)")
ax.set_title(f"Cord Value Distribution by Primary Color Code (top 12)\n"
             f"Kruskal-Wallis: H={kw_stat:.1f}, p={kw_p:.1e} — "
             f"{'significant' if kw_p < 0.05 else 'NOT significant'}")
plt.tight_layout()
fig.savefig(OUT_VIZ / "color_value_correlation.png", dpi=150)
plt.close(fig)
print("  → color_value_correlation.png saved")

# ============================================================================
# 5. Color co-occurrence across khipus
# ============================================================================
print("\n── 5. Color co-occurrence heatmap ───────────────────────────────────")

# Top 15 color codes by number of khipus that contain them
top15_colors = vocab.head(15)["color_code"].tolist()

# Binary matrix: khipu × color (1 if that khipu contains that color)
cv_top = color_khipu[color_khipu["color_code"].isin(top15_colors)]
khipu_color_matrix = (
    cv_top.groupby(["kfg_id", "color_code"])
    .size()
    .unstack(fill_value=0)
    .clip(upper=1)
    .reindex(columns=top15_colors, fill_value=0)
)

# Co-occurrence matrix: (X^T X) where X is binary
mat = khipu_color_matrix.values
cooc = mat.T @ mat  # [n_colors × n_colors]
cooc_df = pd.DataFrame(cooc, index=top15_colors, columns=top15_colors)

print("  Co-occurrence matrix (khipus containing both):")
print(cooc_df.to_string())

fig, ax = plt.subplots(figsize=(9, 8))
mask = np.zeros_like(cooc, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True  # show lower triangle only
sns.heatmap(
    cooc_df,
    mask=mask,
    annot=True,
    fmt="d",
    cmap="YlOrRd",
    square=True,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "Khipus with both colors"},
)
ax.set_title("Color Co-occurrence Heatmap (Top 15 Color Codes)\n"
             "Cell value = number of khipus containing both colors", fontsize=11)
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
fig.savefig(OUT_VIZ / "color_cooccurrence.png", dpi=150)
plt.close(fig)
print("  → color_cooccurrence.png saved")

# ============================================================================
# Statistical results summary
# ============================================================================
stat_results = pd.DataFrame([
    {
        "test": "White-first-cord × n_pattern_types (Mann-Whitney)",
        "statistic": u_stat,
        "p": p_val,
        "significant": p_val < 0.05,
        "note": f"With={wf_yes.mean():.2f} vs Without={wf_no.mean():.2f} mean pattern types",
    },
    {
        "test": "White-first-cord × cluster (chi-square)",
        "statistic": chi2_wf,
        "p": p_chi2_wf,
        "significant": p_chi2_wf < 0.05,
        "note": f"Complex rate with={complex_rate_with:.1%}, without={complex_rate_without:.1%}",
    },
    {
        "test": "Color diversity Complex > Simple (Mann-Whitney)",
        "statistic": u2,
        "p": p2,
        "significant": p2 < 0.05,
        "note": (f"Mean unique colors: Complex={df_div[df_div['cluster_label']=='Complex']['n_unique_colors'].mean():.1f}, "
                 f"Simple={df_div[df_div['cluster_label']=='Simple']['n_unique_colors'].mean():.1f}"),
    },
    {
        "test": "Color code × cord value (Kruskal-Wallis)",
        "statistic": kw_stat,
        "p": kw_p,
        "significant": kw_p < 0.05,
        "note": f"Top 12 colors, non-zero values only",
    },
])
stat_results.to_csv(OUT_DATA / "phase5_stat_results.csv", index=False)
print("\n  → phase5_stat_results.csv saved")

# ============================================================================
# Summary
# ============================================================================
print("\n══ Phase 5 Complete ══════════════════════════════════════════════════")
print(f"  Distinct color codes (normalised):  {len(vocab)}")
print(f"  Compound cord fraction:             {pct_compound:.1f}%")
print(f"  Khipus with white-first cord:       {n_has_wf} ({pct_has_wf:.1f}%)")
print(f"  White-first × pattern types p:      {p_val:.4f}  {'✅' if p_val < 0.05 else '❌'}")
print(f"  White-first × cluster chi2 p:       {p_chi2_wf:.4f}  {'✅' if p_chi2_wf < 0.05 else '❌'}")
print(f"  Color diversity Complex vs Simple p: {p2:.2e}  {'✅' if p2 < 0.05 else '❌'}")
print(f"  Color-value Kruskal-Wallis p:       {kw_p:.2e}  {'✅' if kw_p < 0.05 else '❌'}")
print()
print("  Outputs:")
for f in sorted(OUT_VIZ.glob("*.png")):
    print(f"    {f.relative_to(ROOT)}")
for f in sorted(OUT_DATA.glob("phase5_*.csv")):
    print(f"    {f.relative_to(ROOT)}")
