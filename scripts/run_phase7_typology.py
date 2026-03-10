"""
Phase 7: Multi-feature Typology
================================
An evidence-grounded khipu typology that combines all features accumulated
across Phases 3–6: structural (cord counts, hierarchy), summation patterns,
color diversity, and anomaly scores.

The analysis deliberately avoids pre-assigning administrative function labels.
Clusters receive neutral alphanumeric IDs (T1–TK) and are described solely
by their empirical feature profiles. Any interpretive labels require external
expert validation.

Approach:
  1. Build enriched feature matrix (Phase 3 + 5 + 6 features)
  2. K-means sweep k=2..8, silhouette selection
  3. Profile each cluster: feature means, pattern prevalence, zone distribution
  4. Visualise: silhouette curve, profile heatmap, UMAP projection,
     cluster × zone stacked bar, cluster × Simple/Complex grouped bar

Inputs:
  data/processed/phase3_clusters.csv
  data/processed/phase5_color_diversity.csv
  data/processed/phase6_anomaly_scores.csv

Outputs:
  data/processed/phase7_typology.csv          (per-khipu cluster assignment)
  data/processed/phase7_cluster_profiles.csv  (per-cluster feature means)
  visualizations/phase7/silhouette_curve.png
  visualizations/phase7/profile_heatmap.png
  visualizations/phase7/umap_typology.png
  visualizations/phase7/cluster_zone.png
  visualizations/phase7/cluster_complexity.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent
P3_CSV    = ROOT / "data" / "processed" / "phase3_clusters.csv"
P5_CSV    = ROOT / "data" / "processed" / "phase5_color_diversity.csv"
P6_CSV    = ROOT / "data" / "processed" / "phase6_anomaly_scores.csv"
OUT_DATA  = ROOT / "data" / "processed"
OUT_VIZ   = ROOT / "visualizations" / "phase7"
OUT_VIZ.mkdir(parents=True, exist_ok=True)

ZONE_ORDER = [
    "Central Coast", "Cañete–Pisco", "Ica & Paracas",
    "Nazca & Far South", "Arica & N. Chile",
    "North Peru Coast", "Southern Highlands", "Chachapoyas",
]

PATTERN_COLS = ["has_pp", "has_ip", "has_cp", "has_sp",
                "has_gg", "has_gsb", "has_is", "has_psn", "has_adg"]

# Features used in clustering (pre-scaling)
CLUSTER_FEATURES = [
    "n_cords",
    "n_pendants",
    "n_subsidiaries",
    "n_groups",
    "numeric_coverage",
    "frac_broken",
    "n_pattern_types",
    "n_unique_colors",
    "sub_ratio",
    "group_size",
]

K_RANGE = range(2, 9)   # k = 2 … 8
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load & merge
# ---------------------------------------------------------------------------
print("Loading data …")
df = pd.read_csv(P3_CSV)
div = pd.read_csv(P5_CSV)[["kfg_id", "n_unique_colors"]]
ano = pd.read_csv(P6_CSV)[["kfg_id", "iso_score", "lof_score",
                            "z_max", "n_methods_flagged", "anomaly_class"]]

df = df.merge(div, on="kfg_id", how="left")
df = df.merge(ano, on="kfg_id", how="left")
df["n_unique_colors"]  = df["n_unique_colors"].fillna(0)
df["cluster_label"]    = df["cluster"].map({0: "Simple", 1: "Complex"})

# Derived structural features
df["sub_ratio"]  = df["n_subsidiaries"] / df["n_pendants"].replace(0, np.nan)
df["group_size"] = df["n_pendants"]     / df["n_groups"].replace(0, np.nan)
df["sub_ratio"]  = df["sub_ratio"].fillna(0)
df["group_size"] = df["group_size"].fillna(df["n_cords"])

print(f"  {len(df)} khipus | {len(CLUSTER_FEATURES)} clustering features")

# Feature matrix
X_raw = df[CLUSTER_FEATURES].copy().fillna(0)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ---------------------------------------------------------------------------
# K-means sweep
# ---------------------------------------------------------------------------
print("\n── K-means silhouette sweep ─────────────────────────────────────────")
sil_scores = {}
inertias   = {}

for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    sil_scores[k] = sil
    inertias[k] = km.inertia_
    print(f"  k={k}  silhouette={sil:.4f}  inertia={km.inertia_:,.0f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"\n  Best k = {best_k} (silhouette={sil_scores[best_k]:.4f})")

# Final clustering at best_k
km_final = KMeans(n_clusters=best_k, n_init=30, random_state=RANDOM_STATE)
df["typology_raw"] = km_final.fit_predict(X)

# Re-label clusters T1…TK ordered by median n_cords (ascending) for stability
order = (
    df.groupby("typology_raw")["n_cords"]
    .median()
    .sort_values()
    .index.tolist()
)
remap = {old: new for new, old in enumerate(order)}
df["typology"] = df["typology_raw"].map(remap)
df["typology_label"] = "T" + (df["typology"] + 1).astype(str)

# ---------------------------------------------------------------------------
# Silhouette curve visualisation
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Phase 7: K-means Sweep — Silhouette and Inertia", fontsize=12)

ax = axes[0]
ks = list(K_RANGE)
sils = [sil_scores[k] for k in ks]
ax.plot(ks, sils, marker="o", color="#1b9e77", linewidth=2)
ax.axvline(best_k, color="#d95f02", linestyle="--", label=f"Best k={best_k}")
ax.set_xlabel("k (number of clusters)")
ax.set_ylabel("Silhouette score")
ax.set_title("Silhouette Score")
ax.set_xticks(ks)
ax.legend()

ax = axes[1]
inert = [inertias[k] for k in ks]
ax.plot(ks, inert, marker="s", color="#7570b3", linewidth=2)
ax.axvline(best_k, color="#d95f02", linestyle="--", label=f"Best k={best_k}")
ax.set_xlabel("k (number of clusters)")
ax.set_ylabel("Inertia (within-cluster sum of squares)")
ax.set_title("Elbow Curve")
ax.set_xticks(ks)
ax.legend()

plt.tight_layout()
fig.savefig(OUT_VIZ / "silhouette_curve.png", dpi=150)
plt.close(fig)
print("\n  → silhouette_curve.png saved")

# ---------------------------------------------------------------------------
# Cluster profiles
# ---------------------------------------------------------------------------
print(f"\n── Cluster profiles (k={best_k}) ───────────────────────────────────")

profile_features = CLUSTER_FEATURES + PATTERN_COLS
profile = (
    df.groupby("typology_label")[profile_features + ["cluster_label"]]
    .agg(lambda s: s.mean() if s.dtype != object else s.value_counts().index[0])
)
profile["n_khipus"] = df.groupby("typology_label").size()
profile["pct_complex"] = df.groupby("typology_label")["cluster_label"].apply(
    lambda s: (s == "Complex").mean() * 100
)
profile["pct_provenanced"] = df.groupby("typology_label")["geo_zone"].apply(
    lambda s: s.notna().mean() * 100
)

profile.to_csv(OUT_DATA / "phase7_cluster_profiles.csv")
print("  → phase7_cluster_profiles.csv saved")
print(profile[["n_khipus", "pct_complex", "n_cords", "n_pattern_types",
               "n_unique_colors", "numeric_coverage", "frac_broken"]].to_string())

# ---------------------------------------------------------------------------
# Profile heatmap
# ---------------------------------------------------------------------------
# Normalise feature means to 0-1 across clusters for heatmap display
heat_feats = ["n_cords", "n_pendants", "n_subsidiaries", "n_groups",
              "n_pattern_types", "n_unique_colors",
              "numeric_coverage", "frac_broken", "sub_ratio", "group_size",
              "has_pp", "has_ip", "has_cp", "has_sp",
              "has_gg", "has_gsb", "has_is", "has_psn", "has_adg"]

heat_df = df.groupby("typology_label")[heat_feats].mean().T
# row-wise min-max normalisation
heat_norm = heat_df.subtract(heat_df.min(axis=1), axis=0)
denom = (heat_df.max(axis=1) - heat_df.min(axis=1)).replace(0, 1)
heat_norm = heat_norm.divide(denom, axis=0)

import matplotlib.colors as mcolors
fig, ax = plt.subplots(figsize=(max(7, best_k * 1.3), 10))
im = ax.imshow(heat_norm.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(best_k))
ax.set_xticklabels(heat_norm.columns, fontsize=11, fontweight="bold")
ax.set_yticks(range(len(heat_feats)))
ax.set_yticklabels(heat_feats, fontsize=9)
for i in range(len(heat_feats)):
    for j in range(best_k):
        val = heat_df.values[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                color="black" if heat_norm.values[i, j] < 0.7 else "white")
ax.set_title(
    f"Cluster Profile Heatmap — {best_k} Typology Groups\n"
    f"(cell = raw mean; colour = row-normalised 0–1)",
    fontsize=11,
)
plt.colorbar(im, ax=ax, label="Relative value (0=min, 1=max in row)")
# Divider line between structural and pattern features
ax.axhline(9.5, color="white", linewidth=2)
ax.text(best_k - 0.5, 9.7, "▼ pattern flags", fontsize=8, ha="right", color="gray")
plt.tight_layout()
fig.savefig(OUT_VIZ / "profile_heatmap.png", dpi=150)
plt.close(fig)
print("  → profile_heatmap.png saved")

# ---------------------------------------------------------------------------
# UMAP projection (or PCA fallback)
# ---------------------------------------------------------------------------
if HAS_UMAP:
    print("\n── UMAP projection ──────────────────────────────────────────────────")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE, n_jobs=1)
    embedding = reducer.fit_transform(X)
    method = "UMAP"
else:
    from sklearn.decomposition import PCA
    print("\n── PCA projection (umap-learn not installed) ─────────────────────────")
    embedding = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    method = "PCA"
df["umap_x"] = embedding[:, 0]
df["umap_y"] = embedding[:, 1]

palette = plt.cm.tab10.colors
typology_labels = sorted(df["typology_label"].unique())

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f"Phase 7: {method} Projection — Multi-feature Typology", fontsize=13)

# Left: coloured by typology
ax = axes[0]
for i, tl in enumerate(typology_labels):
    sub = df[df["typology_label"] == tl]
    ax.scatter(sub["umap_x"], sub["umap_y"],
               c=[palette[i % len(palette)]], s=18, alpha=0.7,
               label=f"{tl} (n={len(sub)})", edgecolors="none")
ax.set_title("Coloured by Typology Group")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.legend(fontsize=8, markerscale=1.5)

# Right: coloured by Simple/Complex (Phase 3)
ax = axes[1]
color_map_c = {"Simple": "#7570b3", "Complex": "#d95f02"}
for cls, color in color_map_c.items():
    sub = df[df["cluster_label"] == cls]
    ax.scatter(sub["umap_x"], sub["umap_y"],
               c=color, s=18, alpha=0.6,
               label=f"{cls} (n={len(sub)})", edgecolors="none")
# Overlay anomalies as stars
hc = df[df["anomaly_class"] == "High-confidence"]
ax.scatter(hc["umap_x"], hc["umap_y"],
           marker="*", s=90, c="gold", edgecolors="black",
           linewidths=0.4, label=f"High-conf anomaly (n={len(hc)})", zorder=5)
ax.set_title("Coloured by Phase-3 Cluster + Anomalies")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.legend(fontsize=8, markerscale=1.0)

plt.tight_layout()
fig.savefig(OUT_VIZ / "umap_typology.png", dpi=150)
plt.close(fig)
print("  → umap_typology.png saved")

# ---------------------------------------------------------------------------
# Cluster × Geographic zone (stacked bar)
# ---------------------------------------------------------------------------
zone_counts = (
    df[df["geo_zone"].notna()]
    .groupby(["typology_label", "geo_zone"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=[z for z in ZONE_ORDER if z in df["geo_zone"].unique()],
             fill_value=0)
)
zone_pct = zone_counts.div(zone_counts.sum(axis=1), axis=0) * 100

zone_palette = plt.cm.Set2.colors
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Phase 7: Geographic Distribution by Typology Group", fontsize=12)

# Left: counts
ax = axes[0]
bottom = np.zeros(len(zone_counts))
for i, zone in enumerate(zone_pct.columns):
    vals = zone_counts[zone].values
    ax.bar(zone_counts.index, vals, bottom=bottom,
           color=zone_palette[i % len(zone_palette)], label=zone)
    bottom += vals
ax.set_ylabel("Khipu count")
ax.set_title("Provenanced khipus per group")
ax.legend(fontsize=7, loc="upper right")

# Right: percentage
ax = axes[1]
bottom = np.zeros(len(zone_pct))
for i, zone in enumerate(zone_pct.columns):
    vals = zone_pct[zone].values
    ax.bar(zone_pct.index, vals, bottom=bottom,
           color=zone_palette[i % len(zone_palette)], label=zone)
    bottom += vals
ax.set_ylabel("% of provenanced khipus in group")
ax.set_title("Zone composition (%) per group")
ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
fig.savefig(OUT_VIZ / "cluster_zone.png", dpi=150)
plt.close(fig)
print("  → cluster_zone.png saved")

# ---------------------------------------------------------------------------
# Cluster × Simple/Complex + anomaly (grouped bar)
# ---------------------------------------------------------------------------
complexity_counts = (
    df.groupby(["typology_label", "cluster_label"])
    .size()
    .unstack(fill_value=0)
    [["Simple", "Complex"]]
)
anomaly_counts = (
    df[df["anomaly_class"] == "High-confidence"]
    .groupby("typology_label")
    .size()
    .reindex(complexity_counts.index, fill_value=0)
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Phase 7: Cluster Composition", fontsize=12)

# Left: Simple/Complex stacked
ax = axes[0]
x = np.arange(len(complexity_counts))
width = 0.5
ax.bar(x, complexity_counts["Simple"], width, label="Simple", color="#7570b3", alpha=0.85)
ax.bar(x, complexity_counts["Complex"], width,
       bottom=complexity_counts["Simple"], label="Complex", color="#d95f02", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(complexity_counts.index)
ax.set_ylabel("Khipu count")
ax.set_title("Simple / Complex composition per typology group")
ax.legend(fontsize=9)
# Label total
totals = complexity_counts.sum(axis=1)
for xi, tot in zip(x, totals):
    ax.text(xi, tot + 1, str(tot), ha="center", va="bottom", fontsize=9, fontweight="bold")

# Right: anomaly rate per typology
ax = axes[1]
totals_arr = df.groupby("typology_label").size().reindex(complexity_counts.index)
anom_rate = (anomaly_counts / totals_arr * 100).values
bars = ax.bar(x, anom_rate, width, color="#fc8d59", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(complexity_counts.index)
ax.set_ylabel("High-confidence anomaly rate (%)")
ax.set_title("Anomaly rate per typology group")
for bar, v in zip(bars, anom_rate):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
fig.savefig(OUT_VIZ / "cluster_complexity.png", dpi=150)
plt.close(fig)
print("  → cluster_complexity.png saved")

# ---------------------------------------------------------------------------
# Save per-khipu typology
# ---------------------------------------------------------------------------
out_cols = [
    "kfg_id", "provenance_display", "geo_zone", "cluster_label",
    "typology", "typology_label",
    "n_cords", "n_pattern_types", "n_unique_colors",
    "numeric_coverage", "frac_broken",
    "iso_score", "anomaly_class",
] + PATTERN_COLS
df[out_cols].to_csv(OUT_DATA / "phase7_typology.csv", index=False)
print("\n  → phase7_typology.csv saved")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print(f"\n══ Phase 7 Complete  (k={best_k}) ══════════════════════════════════")
summary = (
    df.groupby("typology_label")
    .agg(
        n=("kfg_id", "count"),
        pct_complex=("cluster_label", lambda s: (s == "Complex").mean() * 100),
        median_cords=("n_cords", "median"),
        mean_patterns=("n_pattern_types", "mean"),
        mean_colors=("n_unique_colors", "mean"),
        mean_numeric_cov=("numeric_coverage", "mean"),
        mean_frac_broken=("frac_broken", "mean"),
        n_anomaly=("anomaly_class", lambda s: (s == "High-confidence").sum()),
    )
    .reset_index()
)
print(summary.to_string(index=False))
print()
print("  Outputs:")
for f in sorted(OUT_VIZ.glob("*.png")):
    print(f"    {f.relative_to(ROOT)}")
for f in sorted(OUT_DATA.glob("phase7_*.csv")):
    print(f"    {f.relative_to(ROOT)}")
