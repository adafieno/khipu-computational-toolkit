"""
Phase 6: Anomaly Detection
==========================
Identifies structurally unusual khipus in the KFG corpus using three
complementary approaches, then cross-validates results.

Methods:
  1. Isolation Forest  — unsupervised ML; scores each khipu's structural
                         novelty against the full corpus
  2. Local Outlier Factor — density-based; flags khipus whose local
                            neighbourhood is much denser than the khipu itself
  3. Z-score flagging   — per-feature; flags khipus >3 SD from mean on any
                          single structural feature

A khipu flagged by ≥2 methods is classified as a HIGH-CONFIDENCE anomaly.
A khipu flagged by exactly 1 method is a CANDIDATE anomaly.

Inputs:
  data/processed/phase3_clusters.csv
  data/processed/phase5_color_diversity.csv  (n_unique_colors)

Outputs:
  data/processed/phase6_anomaly_scores.csv
  data/processed/phase6_anomaly_catalog.csv      (flagged khipus only)
  visualizations/phase6/anomaly_scatter.png
  visualizations/phase6/anomaly_features.png
  visualizations/phase6/anomaly_method_venn.png
  visualizations/phase6/anomaly_profiles.png
"""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PHASE3_CSV  = ROOT / "data" / "processed" / "phase3_clusters.csv"
PHASE5_DIV  = ROOT / "data" / "processed" / "phase5_color_diversity.csv"
OUT_DATA    = ROOT / "data" / "processed"
OUT_VIZ     = ROOT / "visualizations" / "phase6"
OUT_VIZ.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature set
# ---------------------------------------------------------------------------
STRUCT_FEATURES = [
    "n_cords",
    "n_pendants",
    "n_subsidiaries",
    "n_groups",
    "numeric_coverage",
    "frac_broken",
    "n_colors",           # Phase 3 simple color count
    "n_pattern_types",
    "n_unique_colors",    # Phase 5 normalised color-code count (richer)
    "sub_ratio",          # derived: subsidiaries / pendants
    "group_size",         # derived: pendants / groups
]

IF_CONTAMINATION  = 0.05   # expect ~5% outliers (35 of 709)
LOF_N_NEIGHBORS   = 20
ZSCORE_THRESHOLD  = 3.0    # per-feature |z| > 3
HIGH_CONF_METHODS = 2       # flagged by ≥N methods → high-confidence

# ---------------------------------------------------------------------------
# Load and merge data
# ---------------------------------------------------------------------------
print("Loading data …")
df = pd.read_csv(PHASE3_CSV)
div = pd.read_csv(PHASE5_DIV)[["kfg_id", "n_unique_colors"]]
df = df.merge(div, on="kfg_id", how="left")
df["n_unique_colors"] = df["n_unique_colors"].fillna(0)

# Derived features
df["sub_ratio"]  = df["n_subsidiaries"] / df["n_pendants"].replace(0, np.nan)
df["group_size"] = df["n_pendants"]     / df["n_groups"].replace(0, np.nan)
df["sub_ratio"]  = df["sub_ratio"].fillna(0)
df["group_size"] = df["group_size"].fillna(df["n_cords"])  # fallback for ungrouped

df["cluster_label"] = df["cluster"].map({0: "Simple", 1: "Complex"})
print(f"  {len(df)} khipus loaded, {len(STRUCT_FEATURES)} features")

# Feature matrix for anomaly detection
X_raw = df[STRUCT_FEATURES].copy()
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ============================================================================
# Method 1: Isolation Forest
# ============================================================================
print("\n── Method 1: Isolation Forest ───────────────────────────────────────")
iso = IsolationForest(
    contamination=IF_CONTAMINATION,
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)
iso_pred = iso.fit_predict(X)           # -1 = anomaly, 1 = normal
iso_score = -iso.score_samples(X)       # higher = more anomalous (invert)
df["iso_score"]   = iso_score
df["iso_flagged"] = (iso_pred == -1)

n_iso = df["iso_flagged"].sum()
print(f"  Flagged: {n_iso} ({n_iso/len(df)*100:.1f}%)")
print(f"  Score range: {iso_score.min():.3f} – {iso_score.max():.3f}")

# ============================================================================
# Method 2: Local Outlier Factor
# ============================================================================
print("\n── Method 2: Local Outlier Factor ───────────────────────────────────")
lof = LocalOutlierFactor(
    n_neighbors=LOF_N_NEIGHBORS,
    contamination=IF_CONTAMINATION,
    n_jobs=-1,
)
lof_pred  = lof.fit_predict(X)          # -1 = anomaly
lof_score = -lof.negative_outlier_factor_  # higher = more anomalous
df["lof_score"]   = lof_score
df["lof_flagged"] = (lof_pred == -1)

n_lof = df["lof_flagged"].sum()
print(f"  Flagged: {n_lof} ({n_lof/len(df)*100:.1f}%)")

# ============================================================================
# Method 3: Z-score (per feature)
# ============================================================================
print("\n── Method 3: Z-score per feature ────────────────────────────────────")
z_scores = pd.DataFrame(
    np.abs(stats.zscore(X_raw, nan_policy="omit")),
    columns=[f"z_{c}" for c in STRUCT_FEATURES],
    index=df.index,
)
df = pd.concat([df, z_scores], axis=1)
df["z_max"]     = z_scores.max(axis=1)
df["z_flagged"] = df["z_max"] > ZSCORE_THRESHOLD
df["z_flag_feature"] = z_scores.idxmax(axis=1).str.replace("z_", "")

n_z = df["z_flagged"].sum()
print(f"  Flagged (any feature > {ZSCORE_THRESHOLD} SD): {n_z} ({n_z/len(df)*100:.1f}%)")

# most commonly flagged features
flagged_features = df[df["z_flagged"]]["z_flag_feature"].value_counts()
print("  Leading flag feature:")
print(flagged_features.head(5).to_string())

# ============================================================================
# Consensus
# ============================================================================
print("\n── Consensus ────────────────────────────────────────────────────────")
df["n_methods_flagged"] = df["iso_flagged"].astype(int) + \
                          df["lof_flagged"].astype(int) + \
                          df["z_flagged"].astype(int)

df["anomaly_class"] = "Normal"
df.loc[df["n_methods_flagged"] == 1, "anomaly_class"] = "Candidate"
df.loc[df["n_methods_flagged"] >= HIGH_CONF_METHODS, "anomaly_class"] = "High-confidence"

counts = df["anomaly_class"].value_counts()
print(f"  Normal:          {counts.get('Normal', 0)}")
print(f"  Candidate:       {counts.get('Candidate', 0)}")
print(f"  High-confidence: {counts.get('High-confidence', 0)}")

# Method overlap (Venn-style counts)
both_iso_lof = (df["iso_flagged"] & df["lof_flagged"]).sum()
both_iso_z   = (df["iso_flagged"] & df["z_flagged"]).sum()
both_lof_z   = (df["lof_flagged"] & df["z_flagged"]).sum()
all_three    = (df["iso_flagged"] & df["lof_flagged"] & df["z_flagged"]).sum()
print(f"\n  IF ∩ LOF:      {both_iso_lof}")
print(f"  IF ∩ Z-score:  {both_iso_z}")
print(f"  LOF ∩ Z-score: {both_lof_z}")
print(f"  All three:     {all_three}")

# ============================================================================
# Save outputs
# ============================================================================
score_cols = (
    ["kfg_id", "provenance_display", "geo_zone", "cluster_label",
     "n_cords", "n_pattern_types", "n_unique_colors",
     "numeric_coverage", "frac_broken", "sub_ratio", "group_size",
     "iso_score", "lof_score", "z_max", "z_flag_feature",
     "iso_flagged", "lof_flagged", "z_flagged",
     "n_methods_flagged", "anomaly_class"]
)
df[score_cols].to_csv(OUT_DATA / "phase6_anomaly_scores.csv", index=False)
print("\n  → phase6_anomaly_scores.csv saved")

catalog = (
    df[df["anomaly_class"] != "Normal"][score_cols]
    .sort_values(["anomaly_class", "n_methods_flagged", "iso_score"],
                 ascending=[True, False, False])
    .reset_index(drop=True)
)
catalog.to_csv(OUT_DATA / "phase6_anomaly_catalog.csv", index=False)
print(f"  → phase6_anomaly_catalog.csv saved ({len(catalog)} anomalies)")

# ============================================================================
# Visualization 1: n_cords vs numeric_coverage scatter
# ============================================================================
print("\n── Visualizations ───────────────────────────────────────────────────")
COLOR_MAP = {"Normal": "#aaaaaa", "Candidate": "#fdae61", "High-confidence": "#d7191c"}
MARKER_MAP = {"Normal": "o", "Candidate": "^", "High-confidence": "*"}
SIZE_MAP = {"Normal": 18, "Candidate": 50, "High-confidence": 120}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Phase 6: Anomaly Detection — Structural Scatter Plots", fontsize=13)

for ax, (xcol, ycol, xlab, ylab, xlog) in zip(axes, [
    ("n_cords", "numeric_coverage", "Cord count (log)", "Numeric coverage", True),
    ("n_cords", "n_unique_colors",  "Cord count (log)", "Unique color codes (log)", True),
]):
    for cls in ["Normal", "Candidate", "High-confidence"]:
        sub = df[df["anomaly_class"] == cls]
        ax.scatter(
            sub[xcol], sub[ycol],
            c=COLOR_MAP[cls], marker=MARKER_MAP[cls],
            s=SIZE_MAP[cls], alpha=0.7, label=cls, edgecolors="none",
            zorder=3 if cls == "High-confidence" else 1,
        )
    if xlog:
        ax.set_xscale("log")
    if ycol == "n_unique_colors":
        ax.set_yscale("log")
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(f"{xlab} vs {ylab}")

axes[0].legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUT_VIZ / "anomaly_scatter.png", dpi=150)
plt.close(fig)
print("  → anomaly_scatter.png saved")

# ============================================================================
# Visualization 2: Feature distributions — anomaly vs normal
# ============================================================================
feat_display = ["n_cords", "n_pattern_types", "numeric_coverage",
                "frac_broken", "n_unique_colors", "sub_ratio"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Feature Distributions: Normal vs High-confidence Anomalies", fontsize=12)

hc = df[df["anomaly_class"] == "High-confidence"]
nm = df[df["anomaly_class"] == "Normal"]

for ax, feat in zip(axes.flat, feat_display):
    vals_nm = nm[feat].dropna()
    vals_hc = hc[feat].dropna()
    ax.hist(vals_nm, bins=30, color="#aaaaaa", alpha=0.6, label=f"Normal (n={len(vals_nm)})",
            density=True)
    ax.hist(vals_hc, bins=15, color="#d7191c", alpha=0.7, label=f"High-conf (n={len(vals_hc)})",
            density=True)
    if feat in ("n_cords", "n_unique_colors", "sub_ratio"):
        ax.set_xscale("symlog", linthresh=1)
    ax.set_title(feat, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylabel("Density", fontsize=8)

plt.tight_layout()
fig.savefig(OUT_VIZ / "anomaly_features.png", dpi=150)
plt.close(fig)
print("  → anomaly_features.png saved")

# ============================================================================
# Visualization 3: Method overlap bar chart
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Phase 6: Anomaly Method Agreement", fontsize=12)

# Left: consensus class distribution
ax = axes[0]
class_order = ["Normal", "Candidate", "High-confidence"]
class_counts = [counts.get(c, 0) for c in class_order]
colors_bar = [COLOR_MAP[c] for c in class_order]
bars = ax.bar(class_order, class_counts, color=colors_bar, edgecolor="black", width=0.5)
ax.set_ylabel("Number of khipus")
ax.set_title("Anomaly Classification (consensus of 3 methods)")
for bar, v in zip(bars, class_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1, str(v),
            ha="center", va="bottom", fontsize=11, fontweight="bold")

# Right: method overlap matrix (method flagged count + overlap)
ax = axes[1]
method_names = ["Isolation\nForest", "Local Outlier\nFactor", "Z-score\n(>3 SD)"]
method_flags = [df["iso_flagged"].sum(), df["lof_flagged"].sum(), df["z_flagged"].sum()]
pair_names = ["IF ∩ LOF", "IF ∩ Z", "LOF ∩ Z", "All 3"]
pair_counts = [both_iso_lof, both_iso_z, both_lof_z, all_three]
pair_colors = ["#fc8d59", "#fc8d59", "#fc8d59", "#d7191c"]

x = np.arange(len(method_names) + len(pair_names))
labels_all = method_names + pair_names
counts_all = method_flags + pair_counts
colors_all = ["#74add1"] * 3 + pair_colors
bars2 = ax.bar(x, counts_all, color=colors_all, edgecolor="black", width=0.6)
ax.set_xticks(x)
ax.set_xticklabels(labels_all, fontsize=9)
ax.set_ylabel("Khipus flagged")
ax.set_title("Per-method and Overlap Counts")
for bar, v in zip(bars2, counts_all):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, str(v),
            ha="center", va="bottom", fontsize=9)

plt.tight_layout()
fig.savefig(OUT_VIZ / "anomaly_method_venn.png", dpi=150)
plt.close(fig)
print("  → anomaly_method_venn.png saved")

# ============================================================================
# Visualization 4: High-confidence anomaly profiles (horizontal bars)
# ============================================================================
hc_sorted = hc.sort_values("iso_score", ascending=False).head(20)

feat_profile = ["n_cords", "n_pattern_types", "n_unique_colors",
                "numeric_coverage", "frac_broken", "sub_ratio"]

# Normalise each feature to 0-1 range across full corpus for display
norm_df = df[feat_profile].copy()
for col in feat_profile:
    col_min, col_max = norm_df[col].min(), norm_df[col].max()
    norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min + 1e-9)

hc_norm = norm_df.loc[hc_sorted.index]
hc_norm.index = hc_sorted["kfg_id"].values

fig, ax = plt.subplots(figsize=(10, max(6, len(hc_norm) * 0.45)))
y_pos = np.arange(len(hc_norm))
bar_height = 0.12
palette = plt.cm.Set2.colors

for i, feat in enumerate(feat_profile):
    ax.barh(y_pos + i * bar_height, hc_norm[feat],
            height=bar_height, label=feat, color=palette[i], alpha=0.85)

ax.set_yticks(y_pos + bar_height * len(feat_profile) / 2)
ax.set_yticklabels(hc_norm.index, fontsize=8)
ax.set_xlabel("Normalised feature value (0 = min, 1 = max in corpus)")
ax.set_title("High-confidence Anomaly Profiles\n(top 20 by Isolation Forest score)", fontsize=11)
ax.legend(loc="lower right", fontsize=8)
ax.set_xlim(0, 1.05)
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(OUT_VIZ / "anomaly_profiles.png", dpi=150)
plt.close(fig)
print("  → anomaly_profiles.png saved")

# ============================================================================
# Console summary of high-confidence anomalies
# ============================================================================
print("\n══ High-confidence anomalies ═════════════════════════════════════════")
display_cols = ["kfg_id", "provenance_display", "geo_zone", "cluster_label",
                "n_cords", "n_pattern_types", "n_unique_colors",
                "numeric_coverage", "frac_broken",
                "iso_flagged", "lof_flagged", "z_flagged", "z_flag_feature"]
hc_display = (
    df[df["anomaly_class"] == "High-confidence"][display_cols]
    .sort_values("n_cords", ascending=False)
)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
print(hc_display.to_string(index=False))

# ============================================================================
# Summary
# ============================================================================
print("\n══ Phase 6 Complete ══════════════════════════════════════════════════")
print(f"  Isolation Forest flagged:   {n_iso}")
print(f"  Local Outlier Factor flagged: {n_lof}")
print(f"  Z-score flagged:            {n_z}")
print(f"  High-confidence anomalies:  {counts.get('High-confidence', 0)} "
      f"({counts.get('High-confidence', 0)/len(df)*100:.1f}%)")
print(f"  Candidate anomalies:        {counts.get('Candidate', 0)}")
print()
print("  Outputs:")
for f in sorted(OUT_VIZ.glob("*.png")):
    print(f"    {f.relative_to(ROOT)}")
for f in sorted(OUT_DATA.glob("phase6_*.csv")):
    print(f"    {f.relative_to(ROOT)}")
