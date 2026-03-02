"""
Phase 11: Color × Value Interaction
=====================================
Tests whether cord color encodes the unit scale of recorded values — the
leading structural hypothesis in khipu studies.  If the Inca decimal
administration used color to signal magnitude tiers (units, tens, hundreds,
thousands), then:

  H1  Primary cord color predicts value order-of-magnitude.
      Prediction: at least one color has a median value ≥ 10× another.

  H2  Color distribution shifts by hierarchy level.
      Prediction: colors common at level-1 subsidiaries differ from those
      dominant at level-0 pendants.

  H3  Within summation-compliant parent-child groups, parent and child share
      a systematic color *difference*, not a random one.
      Prediction: the parent's color is skewed toward higher-value colors
      compared to its children's colors.

  H4  Cord attachment type (U / R / V) modulates the color-value relationship.
      Prediction: the attachment type interacts with color to explain value
      variance better than color alone.

  H5  Color × behavioral cluster interaction.
      Prediction: B4 (hierarchical aggregation) uses a different color palette
      than B5 (flat multi-commodity) because they record different unit scales.

Color schema
────────────
  cords.color     — composite label (e.g. "W", "W:AB", "AB:MB")
  cord_colors     — normalised: (cord_id, color_code, sequence_ord)
                    sequence_ord=0 is the primary (outermost) color
  primary color   — color_code where sequence_ord=0
  banded          — cord has sequence_ord=1+ (stripe / barber-pole pattern)

Top color codes (by corpus frequency, sequence_ord=0):
  W  = white          AB = amber-brown     MB = medium-brown
  YB = yellow-brown   KB = khaki-brown     B  = brown
  GG = gray-green     LB = light-brown     NB = natural-brown
  DB = dark-brown     HB = heavy-brown     RB = reddish-brown
  PK = pink           BG = blue-gray       LK = light-khaki
  BS = bluish         RL = reddish-light   CB = cream-brown

Inputs
──────
  data/kfg/khipu_database.db
  data/processed/phase8_behavioral_clusters.csv
  data/processed/phase10_summation_groups.csv

Outputs
───────
  data/processed/phase11_color_value.csv          (per-cord color+value+metadata)
  data/processed/phase11_color_stats.csv          (per-color summary statistics)
  visualizations/phase11/color_value_boxplot.png  (H1)
  visualizations/phase11/color_by_level.png       (H2)
  visualizations/phase11/color_compliance.png     (H3)
  visualizations/phase11/attachment_color.png     (H4)
  visualizations/phase11/color_cluster_heatmap.png (H5)
"""

from pathlib import Path
import sqlite3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
from sklearn.preprocessing import LabelEncoder

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DB_PATH  = ROOT / "data" / "kfg" / "khipu_database.db"
P8_CSV   = ROOT / "data" / "processed" / "phase8_behavioral_clusters.csv"
P10_CSV  = ROOT / "data" / "processed" / "phase10_summation_groups.csv"
OUT_DATA = ROOT / "data" / "processed"
OUT_VIZ  = ROOT / "visualizations" / "phase11"
OUT_VIZ.mkdir(parents=True, exist_ok=True)

# Minimum cord count to include a color in analyses
MIN_COLOR_N = 50

# ── 1. Load data ─────────────────────────────────────────────────────────
print("Loading cord and color data …")
conn = sqlite3.connect(DB_PATH)

cords = pd.read_sql(
    "SELECT cord_id, kfg_id, cord_name, hierarchy_level, parent_cord, "
    "value, color, attachment, twist FROM cords",
    conn
)
cc = pd.read_sql("SELECT cord_id, color_code, sequence_ord FROM cord_colors", conn)
conn.close()

cords["hierarchy_level"] = cords["hierarchy_level"].fillna(0).astype(int)
cords["parent_cord"]     = cords["parent_cord"].fillna("").str.strip()
cords["value_num"]       = pd.to_numeric(cords["value"], errors="coerce").fillna(0.0)
cords["attachment"]      = cords["attachment"].fillna("Unknown")

print(f"  {len(cords):,} cords  /  {cords['kfg_id'].nunique()} khipus")
print(f"  cord_colors rows: {len(cc):,}")

# Primary color = sequence_ord == 0
primary_col = cc[cc["sequence_ord"] == 0][["cord_id","color_code"]].rename(
    columns={"color_code": "primary_color"}
)
# Banded flag = any sequence_ord >= 1
banded_ids  = set(cc[cc["sequence_ord"] >= 1]["cord_id"].unique())

cords = cords.merge(primary_col, on="cord_id", how="left")
cords["primary_color"] = cords["primary_color"].fillna("unknown")
cords["is_banded"]     = cords["cord_id"].isin(banded_ids)

# Merge behavioral labels
p8 = pd.read_csv(P8_CSV)[["kfg_id","beh_label","typology_label"]]
cords = cords.merge(p8, on="kfg_id", how="left")

print(f"  Primary colors (≥{MIN_COLOR_N} cords): ", end="")
top_colors = cords["primary_color"].value_counts()
top_colors = top_colors[top_colors >= MIN_COLOR_N].index.tolist()
print(len(top_colors))

# Restrict to cords with a qualifying primary color and non-zero value
cords_nz = cords[(cords["value_num"] > 0) & (cords["primary_color"].isin(top_colors))].copy()
print(f"  Non-zero value cords with qualifying primary color: {len(cords_nz):,}")

# ── 2. Per-color value statistics ─────────────────────────────────────────
print("\nComputing per-color value statistics …")

color_stats = (cords_nz.groupby("primary_color")["value_num"]
               .agg(n="count",
                    median="median",
                    mean="mean",
                    std="std",
                    p25=lambda x: x.quantile(0.25),
                    p75=lambda x: x.quantile(0.75),
                    pct_ge10=lambda x: (x >= 10).mean() * 100,
                    pct_ge100=lambda x: (x >= 100).mean() * 100,
                    pct_ge1000=lambda x: (x >= 1000).mean() * 100)
               .reset_index()
               .sort_values("median", ascending=False))

color_stats.to_csv(OUT_DATA / "phase11_color_stats.csv", index=False)
print("  → phase11_color_stats.csv saved")
print("\n  Per-color median value (top 20, non-zero cords):")
print(color_stats[["primary_color","n","median","pct_ge100","pct_ge1000"]]
      .head(20).to_string(index=False))

# Kruskal-Wallis test: value_num × primary_color
kw_groups = [grp["value_num"].values
             for _, grp in cords_nz.groupby("primary_color") if len(grp) >= MIN_COLOR_N]
kw_stat, kw_p = kruskal(*kw_groups)
print(f"\n  Kruskal-Wallis value × primary_color:  H={kw_stat:.2f}  p={kw_p:.2e}  "
      f"{'✅ H1 supported' if kw_p < 0.05 else '❌ H1 not supported'}")

# Ratio: highest vs lowest color median
hi_val = color_stats["median"].max()
lo_val = color_stats["median"].min()
print(f"  Highest color median: {hi_val:.1f}  /  Lowest: {lo_val:.1f}  "
      f"(ratio {hi_val/max(lo_val,0.01):.1f}×)")

# ── 3. Plot H1: color × value boxplot ────────────────────────────────────
# Order colors by median value
ordered = color_stats.sort_values("median", ascending=False)["primary_color"].tolist()
# Cap to top-25 colors by n for readability
top_n_colors = color_stats.nlargest(25, "n")["primary_color"].tolist()
plot_order = [c for c in ordered if c in top_n_colors]

fig, ax = plt.subplots(figsize=(16, 6))
bp_data = [np.log10(cords_nz[cords_nz["primary_color"] == c]["value_num"].clip(lower=0.1))
           for c in plot_order]
bp = ax.boxplot(bp_data, tick_labels=plot_order, patch_artist=True, showfliers=False)

# Colour boxes by median value (spectral palette)
medians = [color_stats[color_stats["primary_color"]==c]["median"].values[0] for c in plot_order]
med_norm = plt.Normalize(vmin=min(medians), vmax=max(medians))
cmap = plt.cm.RdYlGn
for box, med in zip(bp["boxes"], medians):
    box.set_facecolor(cmap(med_norm(med)))
    box.set_alpha(0.80)

ax.set_xlabel("Primary cord color code")
ax.set_ylabel("log₁₀(cord value)")
ax.set_title(
    "Phase 11 H1: Primary Color vs Value Magnitude (non-zero cords)\n"
    "Box colour = median value (green=high, red=low)  |  whiskers=5–95th pct",
    fontsize=10
)
ax.axhline(1, color="grey", linestyle=":", linewidth=0.8, label="value=10")
ax.axhline(2, color="grey", linestyle="--", linewidth=0.8, label="value=100")
ax.axhline(3, color="grey", linestyle="-.", linewidth=0.8, label="value=1000")
ax.legend(fontsize=8, loc="upper right")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
fig.savefig(OUT_VIZ / "color_value_boxplot.png", dpi=150)
plt.close(fig)
print("\n  → color_value_boxplot.png saved")

# ── 4. H2: Color distribution by hierarchy level ─────────────────────────
print("\n── H2: Color distribution by hierarchy level ────────────────────────")

level_color = (cords[cords["primary_color"].isin(top_colors)]
               .groupby(["hierarchy_level","primary_color"])
               .size()
               .unstack(fill_value=0))

# Normalise to fraction within each level
level_color_pct = level_color.div(level_color.sum(axis=1), axis=0) * 100

# Focus on levels 0–2 (>99% of cords)
lc2 = level_color_pct.loc[level_color_pct.index.isin([0,1,2])]

# Top 15 colors by total count for readability
top15 = cords[cords["primary_color"].isin(top_colors)]["primary_color"].value_counts().head(15).index

print("  Color % by level (top 10 colors):")
print(lc2[top15[:10]].round(1).to_string())

# Chi-square test: level (0 vs 1) × top-15 color
ct = pd.crosstab(
    cords[cords["hierarchy_level"].isin([0,1]) & cords["primary_color"].isin(top15)]["hierarchy_level"],
    cords[cords["hierarchy_level"].isin([0,1]) & cords["primary_color"].isin(top15)]["primary_color"]
)
chi2, p_chi, dof, _ = chi2_contingency(ct)
print(f"\n  Chi-square level-0 vs level-1 × color: χ²={chi2:.1f}  df={dof}  p={p_chi:.2e}  "
      f"{'✅ H2 supported' if p_chi < 0.05 else '❌'}")

# Plot H2
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig.suptitle("Phase 11 H2: Color Distribution Shifts by Hierarchy Level", fontsize=11)

level_labels = {0: "Level 0 (pendants)", 1: "Level 1 (subsidiaries)", 2: "Level 2 (sub-sub)"}
level_colors_palette = plt.cm.tab20.colors

for ax, lvl in zip(axes, [0, 1, 2]):
    if lvl not in lc2.index:
        ax.set_visible(False)
        continue
    row = lc2.loc[lvl, top15].sort_values(ascending=True)
    clrs = [level_colors_palette[i % 20] for i in range(len(row))]
    ax.barh(row.index, row.values, color=clrs, edgecolor="white")
    ax.set_xlabel("% of cords at this level")
    ax.set_title(level_labels[lvl])

plt.tight_layout()
fig.savefig(OUT_VIZ / "color_by_level.png", dpi=150)
plt.close(fig)
print("  → color_by_level.png saved")

# ── 5. H3: Color in summation-compliant groups ────────────────────────────
print("\n── H3: Color in summation-compliant groups ──────────────────────────")

p10 = pd.read_csv(P10_CSV)[["kfg_id","parent_name","parent_val","children_sum","compliance_class"]]
compliant = p10[p10["compliance_class"] == "compliant"].copy()

# Get the primary color for each parent cord
cord_color_map = cords.set_index(["kfg_id","cord_name"])["primary_color"].to_dict()
cords["full_key"] = list(zip(cords["kfg_id"], cords["cord_name"]))
compliant["parent_color"] = compliant.apply(
    lambda r: cord_color_map.get((r["kfg_id"], r["parent_name"]), "unknown"), axis=1
)

# Build child-cord colors for compliant groups too
# Match parent-child from cords table
cords["parent_key"] = list(zip(cords["kfg_id"], cords["parent_cord"]))
child_colors = (cords[cords["parent_cord"] != ""]
                .copy()
                .assign(parent_key=lambda d: list(zip(d["kfg_id"], d["parent_cord"]))))

# For compliant groups, find the most common child color
comp_key_set = set(zip(compliant["kfg_id"], compliant["parent_name"]))
child_in_comp = child_colors[
    child_colors.apply(lambda r: (r["kfg_id"], r["parent_cord"]) in comp_key_set, axis=1)
]

print(f"  Compliant groups: {len(compliant):,}")
print(f"  Child cords in compliant groups: {len(child_in_comp):,}")

print("\n  Parent color distribution (compliant):")
pc = compliant["parent_color"].value_counts().head(12)
print(pc.to_string())

print("\n  Child primary color distribution (compliant groups):")
cc2 = child_in_comp["primary_color"].value_counts().head(12)
print(cc2.to_string())

# Color match rate: does parent color == child color?
child_in_comp = child_in_comp.copy()
child_in_comp["parent_color"] = child_in_comp.apply(
    lambda r: cord_color_map.get((r["kfg_id"], r["parent_cord"]), "unknown"), axis=1
)
match_rate = (child_in_comp["primary_color"] == child_in_comp["parent_color"]).mean()
print(f"\n  Parent-child same primary color in compliant groups: {match_rate*100:.1f}%")

# Compare to non-compliant groups
non_comp = p10[p10["compliance_class"] == "sub"].copy()
non_key = set(zip(non_comp["kfg_id"], non_comp["parent_name"]))
child_in_nc = child_colors[
    child_colors.apply(lambda r: (r["kfg_id"], r["parent_cord"]) in non_key, axis=1)
].copy()
child_in_nc["parent_color"] = child_in_nc.apply(
    lambda r: cord_color_map.get((r["kfg_id"], r["parent_cord"]), "unknown"), axis=1
)
nc_match_rate = (child_in_nc["primary_color"] == child_in_nc["parent_color"]).mean()
print(f"  Parent-child same primary color in NON-compliant groups: {nc_match_rate*100:.1f}%")

# Plot H3
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Phase 11 H3: Color Patterns in Summation-Compliant vs Non-Compliant Groups",
             fontsize=11)

ax = axes[0]
comp_colors = compliant["parent_color"].value_counts().head(12)
non_comp_colors = non_comp.apply(
    lambda r: cord_color_map.get((r["kfg_id"], r["parent_name"]), "unknown"), axis=1
).value_counts().head(12)
all_c = sorted(set(comp_colors.index) | set(non_comp_colors.index))
x = np.arange(len(all_c))
width = 0.4
ax.bar(x - width/2,
       [comp_colors.get(c, 0) / len(compliant) * 100 for c in all_c],
       width, label="Compliant", color="#2ca02c", alpha=0.8)
ax.bar(x + width/2,
       [non_comp_colors.get(c, 0) / len(non_comp) * 100 for c in all_c],
       width, label="Sub (non-compliant)", color="#d62728", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(all_c, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("% of parent groups with this color")
ax.set_title("Parent cord color: compliant vs non-compliant")
ax.legend(fontsize=8)

ax = axes[1]
labels = ["Compliant", "Non-compliant (sub)"]
match_rates = [match_rate * 100, nc_match_rate * 100]
bars = ax.bar(labels, match_rates, color=["#2ca02c","#d62728"], edgecolor="white", alpha=0.85)
ax.set_ylabel("% parent-child same primary color")
ax.set_title("Parent=Child color match rate\n(summation compliance groups)")
ax.set_ylim(0, max(match_rates) * 1.25)
for bar, val in zip(bars, match_rates):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}%",
            ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
fig.savefig(OUT_VIZ / "color_compliance.png", dpi=150)
plt.close(fig)
print("  → color_compliance.png saved")

# ── 6. H4: Attachment type × color × value ───────────────────────────────
print("\n── H4: Attachment type modulation ───────────────────────────────────")

attach_main = ["U","R","V"]   # ignore rare types
cords_att = cords_nz[cords_nz["attachment"].isin(attach_main)].copy()

print("  Median value by attachment + top-5 colors:")
att_color_pivot = (cords_att[cords_att["primary_color"].isin(top15[:5])]
                   .groupby(["attachment","primary_color"])["value_num"]
                   .median()
                   .unstack(fill_value=0))
print(att_color_pivot.round(1).to_string())

# Kruskal-Wallis for each attachment type separately
for att in attach_main:
    sub = cords_att[cords_att["attachment"] == att]
    grps = [g["value_num"].values for _, g in sub.groupby("primary_color") if len(g) >= 20]
    if len(grps) >= 2:
        st, pv = kruskal(*grps)
        print(f"  KW value × color ({att}):  H={st:.1f}  p={pv:.2e}  "
              f"({'✅' if pv<0.05 else '❌'})")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Phase 11 H4: Cord Attachment Type × Color × Value", fontsize=11)
att_palette = {"U":"#1f77b4","R":"#ff7f0e","V":"#2ca02c"}

for ax, att in zip(axes, attach_main):
    sub = cords_att[cords_att["attachment"] == att]
    col_med = sub.groupby("primary_color")["value_num"].median().sort_values(ascending=False)
    col_med = col_med[col_med.index.isin(top15[:12])]
    ax.bar(range(len(col_med)), np.log10(col_med.values.clip(min=0.1)),
           color=att_palette.get(att, "grey"), edgecolor="white", alpha=0.85)
    ax.set_xticks(range(len(col_med)))
    ax.set_xticklabels(col_med.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("log₁₀(median value)")
    ax.set_title(f"Attachment={att}  (n={len(sub):,})")
    ax.set_ylim(0, 4)

plt.tight_layout()
fig.savefig(OUT_VIZ / "attachment_color.png", dpi=150)
plt.close(fig)
print("  → attachment_color.png saved")

# ── 7. H5: Color cluster heatmap ─────────────────────────────────────────
print("\n── H5: Color × behavioral cluster interaction ───────────────────────")

beh_col = (cords[cords["primary_color"].isin(top15) & cords["beh_label"].notna()]
           .groupby(["beh_label","primary_color"])
           .size()
           .unstack(fill_value=0))
beh_col_pct = beh_col.div(beh_col.sum(axis=1), axis=0) * 100

beh_labels = sorted(beh_col_pct.index)

print("  Color % by behavioral cluster (top 10 colors):")
print(beh_col_pct[top15[:10]].round(1).to_string())

fig, ax = plt.subplots(figsize=(14, 5))
data_mat = beh_col_pct[top15].values
im = ax.imshow(data_mat, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(top15)))
ax.set_xticklabels(top15, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(beh_labels)))
ax.set_yticklabels(beh_labels, fontsize=11, fontweight="bold")
for i in range(data_mat.shape[0]):
    for j in range(data_mat.shape[1]):
        ax.text(j, i, f"{data_mat[i,j]:.1f}",
                ha="center", va="center", fontsize=7,
                color="black" if data_mat[i,j] < data_mat.max()*0.6 else "white")
plt.colorbar(im, ax=ax, label="% of cords in cluster with this color")
ax.set_title(
    "Phase 11 H5: Primary Color Distribution by Behavioral Cluster\n"
    "(cell = % of all cords in that cluster carrying the color)",
    fontsize=10
)
plt.tight_layout()
fig.savefig(OUT_VIZ / "color_cluster_heatmap.png", dpi=150)
plt.close(fig)
print("  → color_cluster_heatmap.png saved")

# Chi-square: beh_label × primary_color
ct5 = pd.crosstab(
    cords[cords["beh_label"].notna() & cords["primary_color"].isin(top15)]["beh_label"],
    cords[cords["beh_label"].notna() & cords["primary_color"].isin(top15)]["primary_color"]
)
chi5, p5, dof5, _ = chi2_contingency(ct5)
print(f"\n  Chi-square beh_label × primary_color: χ²={chi5:.1f}  df={dof5}  p={p5:.2e}  "
      f"{'✅ H5 supported' if p5 < 0.05 else '❌'}")

# ── 8. Save extended cord-level CSV ──────────────────────────────────────
cords[["cord_id","kfg_id","cord_name","hierarchy_level","value_num",
       "primary_color","is_banded","attachment","twist",
       "beh_label","typology_label"]].to_csv(
    OUT_DATA / "phase11_color_value.csv", index=False)
print("\n  → phase11_color_value.csv saved")

# ── 9. Summary ────────────────────────────────────────────────────────────
print(f"\n══ Phase 11 Complete ══════════════════════════════════════════════════")
print("\nTop 10 colors by median non-zero value:")
print(color_stats[["primary_color","n","median","pct_ge100","pct_ge1000"]].head(10).to_string(index=False))
print()
for f in sorted(OUT_VIZ.glob("*.png")):
    print(f"    {f.relative_to(ROOT)}")
for f in sorted(OUT_DATA.glob("phase11_*.csv")):
    print(f"    {f.relative_to(ROOT)}")
