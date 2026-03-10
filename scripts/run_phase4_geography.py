"""
Phase 4: Geographic Patterns Analysis
======================================
Analyses the geographic distribution of structural types across the corpus.

Inputs (cached from Phase 3):
  data/processed/phase3_clusters.csv   — per-khipu feature matrix + cluster

Outputs:
  data/processed/phase4_zone_summary.csv       — per-zone aggregate stats
  data/processed/phase4_chi2_results.csv       — chi-square test results
  data/processed/phase4_nn_attribution.csv     — nearest-neighbour geo guesses for unprovenanced khipus
  visualizations/phase4/
    complexity_by_zone.png       — ranked bar chart with 95 % CI
    pattern_heatmap_by_zone.png  — 9 patterns × 8 zones prevalence heatmap
    structural_by_zone.png       — n_cords and n_pattern_types distributions by zone
    nn_attribution_heatmap.png   — unprovenanced → most similar zone (top-5 NN vote)

Usage:
  python scripts/run_phase4_geography.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
CLUSTERS_CSV = ROOT / "data" / "processed" / "phase3_clusters.csv"
OUT_DIR = ROOT / "visualizations" / "phase4"
PROCESSED = ROOT / "data" / "processed"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ZONE_ORDER = [
    "Central Coast",
    "Cañete–Pisco",
    "Ica & Paracas",
    "Nazca & Far South",
    "Arica & N. Chile",
    "North Peru Coast",
    "Southern Highlands",
    "Chachapoyas",
]

PATTERN_COLS = ["has_pp", "has_ip", "has_cp", "has_sp", "has_gg",
                "has_gsb", "has_is", "has_psn", "has_adg"]
PATTERN_LABELS = ["PP", "IP", "CP", "SP", "GG", "GSB", "IS", "PSN", "ADG"]

# Features used for nearest-neighbour matching (same as clustering input)
FEATURE_COLS_NN = PATTERN_COLS + ["n_cords", "n_groups", "numeric_coverage"]

# Colour palette — one colour per zone (+ grey for Unprovenanced)
ZONE_PALETTE = {
    "Central Coast":      "#4C9BE8",
    "Cañete–Pisco":       "#5DBB8A",
    "Ica & Paracas":      "#F4A460",
    "Nazca & Far South":  "#D95F5F",
    "Chachapoyas":        "#9B59B6",
    "Arica & N. Chile":   "#E67E22",
    "North Peru Coast":   "#1ABC9C",
    "Southern Highlands": "#C0392B",
    "Unprovenanced":      "#AAAAAA",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(f"Loading {CLUSTERS_CSV.name} …")
df = pd.read_csv(CLUSTERS_CSV)
df["geo_zone"] = df["geo_zone"].fillna("Unprovenanced")

prov = df[df["geo_zone"] != "Unprovenanced"].copy()
unprov = df[df["geo_zone"] == "Unprovenanced"].copy()

print(f"  {len(prov)} provenanced / {len(unprov)} unprovenanced")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wilson_ci(k, n, z=1.96):
    """Wilson score interval."""
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, centre - margin), min(1, centre + margin)


# ---------------------------------------------------------------------------
# 1. Zone summary table
# ---------------------------------------------------------------------------
print("\nComputing zone summary …")

# Recompute with function available
summary_rows = []
for zone in ZONE_ORDER:
    zdf = prov[prov["geo_zone"] == zone]
    n = len(zdf)
    n_complex = int((zdf["cluster"] == 1).sum())
    rate = n_complex / n if n else np.nan
    lo, hi = _wilson_ci(n_complex, n) if n >= 5 else (np.nan, np.nan)
    summary_rows.append({
        "geo_zone": zone,
        "n": n,
        "n_complex": n_complex,
        "complex_rate": rate,
        "ci_lo": lo,
        "ci_hi": hi,
        "mean_n_cords": zdf["n_cords"].mean(),
        "median_n_cords": zdf["n_cords"].median(),
        "mean_n_pattern_types": zdf["n_pattern_types"].mean(),
        "mean_numeric_coverage": zdf["numeric_coverage"].mean(),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(PROCESSED / "phase4_zone_summary.csv", index=False)
print(f"  saved -> data/processed/phase4_zone_summary.csv")

# ---------------------------------------------------------------------------
# 2. Chi-square tests
# ---------------------------------------------------------------------------
print("\nChi-square tests …")

chi2_rows = []

# 2a. Cluster (simple vs complex) × geo_zone
ct = pd.crosstab(prov["geo_zone"], prov["cluster"])
ct = ct.reindex(ZONE_ORDER).fillna(0)
chi2, p, dof, _ = stats.chi2_contingency(ct.values)
chi2_rows.append({"test": "cluster_x_geo_zone", "chi2": chi2, "dof": dof, "p": p,
                  "significant_p05": p < 0.05})
print(f"  cluster × geo_zone:  χ²={chi2:.2f}  dof={dof}  p={p:.4f}")

# 2b. Each pattern × geo_zone (exclude tiny zones with <5 khipus for both cells)
eligible_zones = summary[summary["n"] >= 10]["geo_zone"].tolist()
prov_elig = prov[prov["geo_zone"].isin(eligible_zones)]

for col, label in zip(PATTERN_COLS, PATTERN_LABELS):
    ct2 = pd.crosstab(prov_elig["geo_zone"], prov_elig[col])
    if ct2.shape[1] < 2:
        continue
    chi2_p, p_p, dof_p, _ = stats.chi2_contingency(ct2.values)
    chi2_rows.append({"test": f"pattern_{label}_x_geo_zone",
                      "chi2": chi2_p, "dof": dof_p, "p": p_p,
                      "significant_p05": p_p < 0.05})
    print(f"  {label} × geo_zone:  χ²={chi2_p:.2f}  p={p_p:.4f}")

chi2_df = pd.DataFrame(chi2_rows)
chi2_df.to_csv(PROCESSED / "phase4_chi2_results.csv", index=False)
print(f"  saved -> data/processed/phase4_chi2_results.csv")

# ---------------------------------------------------------------------------
# 3. Visualisations
# ---------------------------------------------------------------------------
print("\nGenerating visualisations …")

# --- 3a. Complexity rate bar chart with CI ---
fig, ax = plt.subplots(figsize=(10, 5))
s = summary.dropna(subset=["complex_rate"]).sort_values("complex_rate", ascending=False)
colours = [ZONE_PALETTE.get(z, "#888888") for z in s["geo_zone"]]
bars = ax.bar(s["geo_zone"], s["complex_rate"] * 100,
              color=colours, edgecolor="white", linewidth=0.8)
# Error bars
for _, row in s.iterrows():
    if not np.isnan(row["ci_lo"]):
        ax.errorbar(row["geo_zone"], row["complex_rate"] * 100,
                    yerr=[[( row["complex_rate"] - row["ci_lo"]) * 100],
                           [(row["ci_hi"] - row["complex_rate"]) * 100]],
                    fmt="none", color="#333333", capsize=4, linewidth=1.5)
# Corpus average
avg = prov["cluster"].eq(1).mean() * 100
ax.axhline(avg, color="#555555", linestyle="--", linewidth=1, label=f"Corpus avg {avg:.0f}%")

ax.set_ylabel("Complex khipus (%)", fontsize=12)
ax.set_title("Structural complexity rate by geographic zone\n(Cluster 1 = Complex; 95% Wilson CI; n ≥ 5 zones only)", fontsize=12)
ax.tick_params(axis="x", rotation=25)
ax.set_ylim(0, 75)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
for bar, rate in zip(bars, s["complex_rate"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{rate*100:.0f}%", ha="center", va="bottom", fontsize=9, color="#333333")
plt.tight_layout()
plt.savefig(OUT_DIR / "complexity_by_zone.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved -> visualizations/phase4/complexity_by_zone.png")

# --- 3b. Pattern prevalence heatmap by zone ---
heatmap_data = []
for zone in ZONE_ORDER:
    zdf = prov[prov["geo_zone"] == zone]
    row = {col: zdf[col].mean() for col in PATTERN_COLS}
    row["n"] = len(zdf)
    heatmap_data.append(row)

hm = pd.DataFrame(heatmap_data, index=ZONE_ORDER)
hm_plot = hm[PATTERN_COLS].copy()
hm_plot.columns = PATTERN_LABELS
hm_plot.index = [f"{z}  (n={hm.loc[z,'n']:.0f})" for z in ZONE_ORDER]

fig, ax = plt.subplots(figsize=(11, 5.5))
sns.heatmap(hm_plot, ax=ax, annot=True, fmt=".0%", cmap="YlOrRd",
            vmin=0, vmax=1, linewidths=0.5, linecolor="#dddddd",
            cbar_kws={"label": "Prevalence", "shrink": 0.7})
ax.set_title("Pattern prevalence by geographic zone (provenanced khipus only)", fontsize=12)
ax.set_xlabel("Pattern type", fontsize=11)
ax.set_ylabel("")
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.savefig(OUT_DIR / "pattern_heatmap_by_zone.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved -> visualizations/phase4/pattern_heatmap_by_zone.png")

# --- 3c. Structural feature distributions by zone ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
zone_data_cords = [prov[prov["geo_zone"] == z]["n_cords"].values for z in ZONE_ORDER]
zone_data_ptypes = [prov[prov["geo_zone"] == z]["n_pattern_types"].values for z in ZONE_ORDER]
zone_ns = [len(prov[prov["geo_zone"] == z]) for z in ZONE_ORDER]
zone_labels = [f"{z}\n(n={n})" for z, n in zip(ZONE_ORDER, zone_ns)]
zone_colours = [ZONE_PALETTE[z] for z in ZONE_ORDER]

bp1 = axes[0].boxplot(zone_data_cords, patch_artist=True, medianprops=dict(color="black", linewidth=1.5),
                      flierprops=dict(marker="o", markersize=2, alpha=0.3))
for patch, colour in zip(bp1["boxes"], zone_colours):
    patch.set_facecolor(colour)
    patch.set_alpha(0.75)
axes[0].set_yscale("log")
axes[0].set_ylabel("Total cord count (log scale)", fontsize=11)
axes[0].set_title("Cord count by zone", fontsize=11)
axes[0].set_xticklabels(zone_labels, fontsize=7.5, rotation=30, ha="right")
axes[0].spines[["top", "right"]].set_visible(False)

bp2 = axes[1].boxplot(zone_data_ptypes, patch_artist=True, medianprops=dict(color="black", linewidth=1.5),
                      flierprops=dict(marker="o", markersize=2, alpha=0.3))
for patch, colour in zip(bp2["boxes"], zone_colours):
    patch.set_facecolor(colour)
    patch.set_alpha(0.75)
axes[1].set_ylabel("Number of pattern types (0–9)", fontsize=11)
axes[1].set_title("Pattern-type count by zone", fontsize=11)
axes[1].set_xticklabels(zone_labels, fontsize=7.5, rotation=30, ha="right")
axes[1].spines[["top", "right"]].set_visible(False)

plt.suptitle("Structural feature distributions by geographic zone", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "structural_by_zone.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved -> visualizations/phase4/structural_by_zone.png")

# ---------------------------------------------------------------------------
# 4. Nearest-neighbour attribution for unprovenanced khipus
# ---------------------------------------------------------------------------
print("\nNearest-neighbour attribution for unprovenanced khipus …")

# Build feature matrix; scale structural cols
scaler = StandardScaler()
prov_feats = prov[FEATURE_COLS_NN].copy()
# Scale the 3 continuous cols in-place (same as Phase 3 clustering)
prov_feats[["n_cords", "n_groups", "numeric_coverage"]] = scaler.fit_transform(
    prov[["n_cords", "n_groups", "numeric_coverage"]]
)

unprov_feats = unprov[FEATURE_COLS_NN].copy()
unprov_feats[["n_cords", "n_groups", "numeric_coverage"]] = scaler.transform(
    unprov[["n_cords", "n_groups", "numeric_coverage"]]
)

dists = cdist(unprov_feats.values, prov_feats.values, metric="euclidean")
K = 5  # top-K neighbours

nn_rows = []
for i, (idx, row) in enumerate(unprov.iterrows()):
    nearest_idx = np.argsort(dists[i])[:K]
    nn_zones = prov.iloc[nearest_idx]["geo_zone"].values
    nn_dists = dists[i][nearest_idx]
    # Plurality vote (weighted by 1/distance)
    votes = {}
    for z, d in zip(nn_zones, nn_dists):
        w = 1 / (d + 1e-9)
        votes[z] = votes.get(z, 0) + w
    top_zone = max(votes, key=votes.get)
    top_weight = votes[top_zone] / sum(votes.values())
    nn_rows.append({
        "kfg_id": row["kfg_id"],
        "provenance_display": row["provenance_display"],
        "n_cords": int(row["n_cords"]),
        "n_pattern_types": int(row["n_pattern_types"]),
        "cluster": int(row["cluster"]),
        "nn_top_zone": top_zone,
        "nn_top_weight": round(top_weight, 3),
        "nn_zones": "|".join(nn_zones),
        "nn_dists": "|".join(f"{d:.3f}" for d in nn_dists),
    })

nn_df = pd.DataFrame(nn_rows)
nn_df.to_csv(PROCESSED / "phase4_nn_attribution.csv", index=False)
print(f"  {len(nn_df)} unprovenanced khipus attributed")
print(f"  saved -> data/processed/phase4_nn_attribution.csv")
print("\n  Top-zone vote distribution:")
print(nn_df["nn_top_zone"].value_counts().to_string())
print(f"\n  High-confidence attributions (weight ≥ 0.8): {nn_df['nn_top_weight'].ge(0.8).sum()}")

# --- 3d. Attribution heatmap: unprovenanced → zone ---
vote_matrix = nn_df.groupby("nn_top_zone").size().reindex(ZONE_ORDER, fill_value=0)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1.8, 1]})

# Left: bar of attributed zones
colours_bar = [ZONE_PALETTE[z] for z in vote_matrix.index]
axes[0].barh(vote_matrix.index, vote_matrix.values, color=colours_bar, edgecolor="white")
axes[0].set_xlabel("Unprovenanced khipus attributed to zone", fontsize=11)
axes[0].set_title("Top-zone NN vote for 265 unprovenanced khipus", fontsize=11)
axes[0].spines[["top", "right"]].set_visible(False)
for i, v in enumerate(vote_matrix.values):
    if v > 0:
        axes[0].text(v + 0.5, i, str(v), va="center", fontsize=9)

# Right: scatter of n_cords vs n_pattern_types coloured by attributed zone
sc_colours = [ZONE_PALETTE.get(z, "#888888") for z in nn_df["nn_top_zone"]]
axes[1].scatter(nn_df["n_pattern_types"], nn_df["n_cords"],
                c=sc_colours, alpha=0.55, s=22, edgecolors="none")
axes[1].set_yscale("log")
axes[1].set_xlabel("Pattern types (0–9)", fontsize=11)
axes[1].set_ylabel("Cord count (log scale)", fontsize=11)
axes[1].set_title("Unprovenanced by inferred zone", fontsize=11)
axes[1].spines[["top", "right"]].set_visible(False)
legend_handles = [mpatches.Patch(color=ZONE_PALETTE[z], label=z) for z in ZONE_ORDER]
axes[1].legend(handles=legend_handles, fontsize=7, loc="upper left",
               framealpha=0.6, ncol=1)

plt.tight_layout()
plt.savefig(OUT_DIR / "nn_attribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved -> visualizations/phase4/nn_attribution.png")

# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------
print("\n=== Zone summary ===")
print(summary[["geo_zone", "n", "n_complex", "complex_rate",
               "mean_n_cords", "mean_n_pattern_types"]].to_string(index=False))

print("\n=== Chi-square results ===")
print(chi2_df.to_string(index=False))
