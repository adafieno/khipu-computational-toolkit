"""
Phase 8: Behavioral Recording Analysis
=======================================
Moves beyond structural morphology to ask: *what kind of accounting behaviour
does each khipu exhibit?*  

The analysis derives seven per-khipu behavioral signals that are largely
orthogonal to structural size (cord count / hierarchy):

  A. value_register       — median non-zero cord value (accounting scale)
  B. pct_nonzero          — fraction of cords carrying any non-zero value
  C. pct_round5           — fraction of non-zero values divisible by 5
                            (proxy: quota/tribute standardisation vs. census counting)
  D. entropy_per_cord     — Shannon entropy of cord values / n recorded cords
                            (information density: economical vs. redundant encoding)
  E. max_hier_level       — deepest hierarchy level (aggregation tier depth)
  F. knot_L_ratio         — fraction of long-knots among all knot clusters
                            (encodes digits 5-9; more L = recording larger per-cord values)
  G. knot_E_ratio         — fraction of figure-eight knots (E / E-E)
                            (encodes 10s terminals; high E = lots of round-ten values)

These features are then used to:
  1. Cluster khipus into behavioral types (k-sweep 2–6, silhouette selection)
  2. Profile each type against structural typology, summation patterns, geography
  3. Test statistical hypotheses about accounting function

Key hypotheses tested:
  H1: Round-number affinity varies by geographic zone (tribute-quota areas vs.
      census-count areas).
  H2: Hierarchy depth ≥ 3 khipus are geographically concentrated, suggesting
      site-of-aggregation effects.
  H3: Behavioral clusters cross-cut the T1/T2 structural partition — meaning
      the same structural type can exhibit different accounting flavours.

Inputs:
  data/kfg/khipu_database.db          (cords, knot_clusters, knot_clusters tables)
  data/processed/phase7_typology.csv  (T1/T2 assignments + pattern flags)
  data/processed/phase4_geography.csv (geo_zone)

Outputs:
  data/processed/phase8_behavioral_features.csv   (7 features + metadata per khipu)
  data/processed/phase8_behavioral_clusters.csv   (per-khipu cluster assignment)
  data/processed/phase8_behavioral_profiles.csv   (per-cluster feature means)
  visualizations/phase8/silhouette_curve.png
  visualizations/phase8/behavioral_heatmap.png
  visualizations/phase8/value_register.png
  visualizations/phase8/round_number_zone.png
  visualizations/phase8/cross_structural.png
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
from scipy.stats import entropy as sci_ent, kruskal, mannwhitneyu, chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DB_PATH  = ROOT / "data" / "kfg" / "khipu_database.db"
P7_CSV   = ROOT / "data" / "processed" / "phase7_typology.csv"
P4_CSV   = ROOT / "data" / "processed" / "phase4_geography.csv"
OUT_DATA = ROOT / "data" / "processed"
OUT_VIZ  = ROOT / "visualizations" / "phase8"
OUT_VIZ.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
K_RANGE      = range(2, 7)

PATTERN_COLS = ["has_pp", "has_ip", "has_cp", "has_sp",
                "has_gg", "has_gsb", "has_is", "has_psn", "has_adg"]

# ── 1. Pull raw data from DB ────────────────────────────────────────────────
print("Loading raw KFG data …")
conn = sqlite3.connect(DB_PATH)

cords  = pd.read_sql("SELECT kfg_id, cord_id, hierarchy_level, value FROM cords", conn)
knots  = pd.read_sql("""
    SELECT c.kfg_id, k.knot_type
    FROM knot_clusters k JOIN cords c ON k.cord_id = c.cord_id
""", conn)
meta   = pd.read_sql("SELECT kfg_id FROM khipu_metadata", conn)
conn.close()

print(f"  cords:  {len(cords):,}")
print(f"  knots:  {len(knots):,}")
print(f"  khipus: {len(meta):,}")

# ── 2. Compute per-khipu behavioral features ────────────────────────────────
print("\nComputing behavioral features …")

features = []
for kfg_id, grp in cords.groupby("kfg_id"):
    n_total   = len(grp)
    n_nonzero = (grp["value"] > 0).sum()
    vals_nz   = grp.loc[grp["value"] > 0, "value"]

    # A. value_register — median non-zero value (0 if no values at all)
    value_register = float(vals_nz.median()) if len(vals_nz) > 0 else 0.0

    # B. pct_nonzero
    pct_nonzero = n_nonzero / n_total if n_total > 0 else 0.0

    # C. pct_round5 — round-number affinity
    if len(vals_nz) > 0:
        pct_round5 = (vals_nz % 5 == 0).sum() / len(vals_nz)
    else:
        pct_round5 = 0.0

    # D. entropy_per_cord — Shannon entropy / n cords with values
    if len(vals_nz) > 1:
        vc   = vals_nz.value_counts()
        p    = vc / vc.sum()
        h    = float(sci_ent(p, base=2))
        entropy_per_cord = h / len(vals_nz)
    else:
        entropy_per_cord = 0.0

    # E. max_hier_level
    max_hier = int(grp["hierarchy_level"].max()) if grp["hierarchy_level"].notna().any() else 0

    features.append({
        "kfg_id":          kfg_id,
        "value_register":  value_register,
        "pct_nonzero":     pct_nonzero,
        "pct_round5":      pct_round5,
        "entropy_per_cord": entropy_per_cord,
        "max_hier_level":  max_hier,
    })

feat_df = pd.DataFrame(features)

# F & G. Knot L-ratio and E-ratio per khipu
knot_counts = knots.groupby(["kfg_id", "knot_type"]).size().unstack(fill_value=0)
# Ensure L and E columns exist
for col in ["L", "S", "E"]:
    if col not in knot_counts.columns:
        knot_counts[col] = 0
knot_counts["n_knots_total"] = knot_counts.sum(axis=1)
knot_counts["knot_L_ratio"]  = knot_counts["L"]  / knot_counts["n_knots_total"].replace(0, np.nan)
knot_counts["knot_E_ratio"]  = knot_counts["E"]  / knot_counts["n_knots_total"].replace(0, np.nan)
knot_counts = knot_counts[["knot_L_ratio", "knot_E_ratio"]].reset_index()

feat_df = feat_df.merge(knot_counts, on="kfg_id", how="left")
feat_df["knot_L_ratio"]  = feat_df["knot_L_ratio"].fillna(0)
feat_df["knot_E_ratio"]  = feat_df["knot_E_ratio"].fillna(0)

# Load Phase 7 typology and geography
p7   = pd.read_csv(P7_CSV)[["kfg_id", "typology_label", "geo_zone",
                              "provenance_display"] + PATTERN_COLS]

# Load Phase 4 geo if Phase 7 doesn't have it (should already be there)
feat_df = feat_df.merge(p7, on="kfg_id", how="left")

print(f"  behavioral features computed for {len(feat_df)} khipus")

# Log-transform value_register and entropy (right-skewed)
feat_df["log_value_register"]  = np.log1p(feat_df["value_register"])
feat_df["log_entropy_per_cord"] = np.log1p(feat_df["entropy_per_cord"])

BEHAV_FEATURES = [
    "log_value_register",
    "pct_nonzero",
    "pct_round5",
    "log_entropy_per_cord",
    "max_hier_level",
    "knot_L_ratio",
    "knot_E_ratio",
]

feat_df.to_csv(OUT_DATA / "phase8_behavioral_features.csv", index=False)
print(f"  → phase8_behavioral_features.csv saved")

# ── 3. K-means sweep ────────────────────────────────────────────────────────
print("\n── K-means sweep ────────────────────────────────────────────────────")
X_raw = feat_df[BEHAV_FEATURES].fillna(0).values
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

sil_scores = {}
inertias   = {}

for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    sil_scores[k] = sil
    inertias[k]   = km.inertia_
    print(f"  k={k}  silhouette={sil:.4f}  inertia={km.inertia_:,.0f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"\n  Best k = {best_k}  (silhouette={sil_scores[best_k]:.4f})")

km_final = KMeans(n_clusters=best_k, n_init=30, random_state=RANDOM_STATE)
feat_df["beh_raw"] = km_final.fit_predict(X)

# Order clusters by ascending median log_value_register
order = (
    feat_df.groupby("beh_raw")["log_value_register"]
    .median().sort_values().index.tolist()
)
remap = {old: new for new, old in enumerate(order)}
feat_df["beh_cluster"]    = feat_df["beh_raw"].map(remap)
feat_df["beh_label"]      = "B" + (feat_df["beh_cluster"] + 1).astype(str)

# ── Silhouette curve ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Phase 8: Behavioral Clustering — K-means Sweep", fontsize=12)

ks    = list(K_RANGE)
sils  = [sil_scores[k] for k in ks]
inert = [inertias[k] for k in ks]

ax = axes[0]
ax.plot(ks, sils, marker="o", color="#1b9e77", linewidth=2)
ax.axvline(best_k, color="#d95f02", linestyle="--", label=f"Best k={best_k}")
ax.set_xlabel("k"); ax.set_ylabel("Silhouette"); ax.set_title("Silhouette Score")
ax.set_xticks(ks); ax.legend()

ax = axes[1]
ax.plot(ks, inert, marker="s", color="#7570b3", linewidth=2)
ax.axvline(best_k, color="#d95f02", linestyle="--", label=f"Best k={best_k}")
ax.set_xlabel("k"); ax.set_ylabel("Inertia"); ax.set_title("Elbow Curve")
ax.set_xticks(ks); ax.legend()

plt.tight_layout()
fig.savefig(OUT_VIZ / "silhouette_curve.png", dpi=150)
plt.close(fig)
print("  → silhouette_curve.png saved")

# ── 4. Cluster profiles ──────────────────────────────────────────────────────
print(f"\n── Behavioral cluster profiles (k={best_k}) ─────────────────────────")

beh_labels = sorted(feat_df["beh_label"].unique())
raw_features = ["value_register", "pct_nonzero", "pct_round5",
                "entropy_per_cord", "max_hier_level",
                "knot_L_ratio", "knot_E_ratio"]

profile = feat_df.groupby("beh_label")[raw_features].mean()
profile["n_khipus"]     = feat_df.groupby("beh_label").size()
profile["pct_T2"]       = (feat_df.groupby("beh_label")["typology_label"]
                           .apply(lambda s: (s == "T2").mean() * 100))
profile.to_csv(OUT_DATA / "phase8_behavioral_profiles.csv")
print("  → phase8_behavioral_profiles.csv saved")
print(profile[["n_khipus", "pct_T2", "value_register", "pct_round5",
               "entropy_per_cord", "max_hier_level"]].to_string())

# ── 5. Behavioral heatmap ────────────────────────────────────────────────────
heat_df   = profile[raw_features].T
heat_norm = heat_df.subtract(heat_df.min(axis=1), axis=0)
denom     = (heat_df.max(axis=1) - heat_df.min(axis=1)).replace(0, 1)
heat_norm = heat_norm.divide(denom, axis=0)

feature_labels = [
    "Median value (register)",
    "% cords with values",
    "% values divisible by 5",
    "Shannon entropy / cord",
    "Max hierarchy level",
    "L-knot ratio (digits 5-9)",
    "E-knot ratio (figure-eights)",
]

fig, ax = plt.subplots(figsize=(max(7, best_k * 1.5), 6))
im = ax.imshow(heat_norm.values, aspect="auto", cmap="PuBu", vmin=0, vmax=1)
ax.set_xticks(range(best_k))
ax.set_xticklabels(heat_norm.columns, fontsize=11, fontweight="bold")
ax.set_yticks(range(len(raw_features)))
ax.set_yticklabels(feature_labels, fontsize=9)
for i in range(len(raw_features)):
    for j in range(best_k):
        val = heat_df.values[i, j]
        fmt = f"{val:.0f}" if val >= 10 else f"{val:.2f}"
        ax.text(j, i, fmt, ha="center", va="center", fontsize=8,
                color="black" if heat_norm.values[i, j] < 0.6 else "white")
ax.set_title(
    f"Behavioral Recording Profile — {best_k} Cluster Heatmap\n"
    "(cell = raw mean; colour = row-normalised 0–1)",
    fontsize=11,
)
plt.colorbar(im, ax=ax, label="Relative value (0=min, 1=max across clusters)")
plt.tight_layout()
fig.savefig(OUT_VIZ / "behavioral_heatmap.png", dpi=150)
plt.close(fig)
print("  → behavioral_heatmap.png saved")

# ── 6. Value register distribution ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Phase 8: Value Register — Accounting Scale per Behavioral Cluster", fontsize=12)
palette = plt.cm.Set1.colors

ax = axes[0]
for i, bl in enumerate(beh_labels):
    sub = feat_df[feat_df["beh_label"] == bl]["value_register"].replace(0, np.nan).dropna()
    ax.hist(np.log1p(sub), bins=25, alpha=0.6,
            color=palette[i % len(palette)], label=f"{bl} (n={len(sub)})")
ax.set_xlabel("log(1 + median cord value)")
ax.set_ylabel("Khipu count")
ax.set_title("Value Register Distribution (log scale)")
ax.legend(fontsize=8)

# Round-number affinity boxplot by cluster
ax = axes[1]
data_box = [feat_df[feat_df["beh_label"] == bl]["pct_round5"].values for bl in beh_labels]
bp = ax.boxplot(data_box, labels=beh_labels, patch_artist=True)
for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("Fraction of values divisible by 5")
ax.set_title("Round-number Affinity (quota/tribute vs. census style)")
ax.axhline(feat_df["pct_round5"].mean(), color="black", linestyle="--",
           linewidth=1, label=f"Corpus mean = {feat_df['pct_round5'].mean():.2f}")
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT_VIZ / "value_register.png", dpi=150)
plt.close(fig)
print("  → value_register.png saved")

# ── 7. Round-number affinity × geographic zone ───────────────────────────────
# Test H1: round-number affinity varies by geo_zone
zone_df = feat_df[feat_df["geo_zone"].notna()].copy()
zone_groups = [grp["pct_round5"].values
               for _, grp in zone_df.groupby("geo_zone")]
kw_stat, kw_p = kruskal(*zone_groups) if len(zone_groups) >= 2 else (np.nan, np.nan)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle(
    f"Phase 8: Round-number Affinity by Geography  "
    f"(Kruskal-Wallis p={kw_p:.4f})",
    fontsize=12,
)

ax = axes[0]
zone_means = zone_df.groupby("geo_zone")["pct_round5"].median().sort_values()
colors_z   = plt.cm.viridis(np.linspace(0.1, 0.9, len(zone_means)))
ax.barh(range(len(zone_means)), zone_means.values, color=colors_z, edgecolor="white")
ax.set_yticks(range(len(zone_means)))
ax.set_yticklabels(zone_means.index, fontsize=8)
ax.set_xlabel("Median fraction divisible by 5")
ax.set_title("Median Round-number Affinity per Zone")
ax.axvline(feat_df["pct_round5"].median(), color="red", linestyle="--", linewidth=1, label="Corpus median")
ax.legend(fontsize=8)

ax = axes[1]
zone_means2 = zone_df.groupby("geo_zone")["value_register"].median().sort_values()
colors_z2   = plt.cm.plasma(np.linspace(0.1, 0.9, len(zone_means2)))
ax.barh(range(len(zone_means2)), zone_means2.values, color=colors_z2, edgecolor="white")
ax.set_yticks(range(len(zone_means2)))
ax.set_yticklabels(zone_means2.index, fontsize=8)
ax.set_xlabel("Median cord value ($)")
ax.set_title("Median Accounting Scale per Zone")
ax.axvline(feat_df["value_register"].median(), color="red", linestyle="--", linewidth=1, label="Corpus median")
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT_VIZ / "round_number_zone.png", dpi=150)
plt.close(fig)
print("  → round_number_zone.png saved")

# ── 8. Cross-structural plot ─────────────────────────────────────────────────
# Show how behavioral clusters cut across T1/T2 structural partition
cross = (
    feat_df.groupby(["beh_label", "typology_label"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["T1", "T2"], fill_value=0)
)
cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

# Hierarchy depth distribution by cluster
hier_box = [feat_df[feat_df["beh_label"] == bl]["max_hier_level"].values for bl in beh_labels]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Phase 8: Behavioral Clusters vs. Structural Typology", fontsize=12)

# Left: cross-tab stacked bar T1/T2
ax = axes[0]
x = np.arange(len(cross))
ax.bar(x, cross_pct["T1"], label="T1 (Compact)", color="#7570b3", alpha=0.85)
ax.bar(x, cross_pct["T2"], bottom=cross_pct["T1"], label="T2 (Extended)", color="#d95f02", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(cross.index)
ax.set_ylabel("%"); ax.set_title("T1/T2 composition per behavioral cluster")
ax.legend(fontsize=8)
for xi, (t1v, t2v) in enumerate(zip(cross["T1"], cross["T2"])):
    ax.text(xi, 103, f"n={t1v+t2v}", ha="center", fontsize=8)

# Middle: hierarchy depth boxplot
ax = axes[1]
bp = ax.boxplot(hier_box, labels=beh_labels, patch_artist=True)
for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_ylabel("Max hierarchy level")
ax.set_title("Aggregation Depth per Behavioral Cluster")
# Annotate deep-hierarchy khipus
n_deep = feat_df.groupby("beh_label").apply(lambda d: (d["max_hier_level"] >= 3).sum())
for xi, (bl, nd) in enumerate(n_deep.items()):
    ax.text(xi + 1, ax.get_ylim()[1] * 0.9, f"≥3: {nd}", ha="center", fontsize=8, color="darkred")

# Right: entropy per cord boxplot
entr_box = [feat_df[feat_df["beh_label"] == bl]["entropy_per_cord"].values for bl in beh_labels]
ax = axes[2]
bp = ax.boxplot(entr_box, labels=beh_labels, patch_artist=True)
for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_ylabel("Shannon entropy / n cords (bits)")
ax.set_title("Information Density per Behavioral Cluster")

plt.tight_layout()
fig.savefig(OUT_VIZ / "cross_structural.png", dpi=150)
plt.close(fig)
print("  → cross_structural.png saved")

# ── 9. Statistical tests ─────────────────────────────────────────────────────
print("\n── Statistical Hypotheses ───────────────────────────────────────────")

# H1: round-number affinity varies by geo_zone
print(f"\nH1 (Kruskal-Wallis round-number affinity × geo_zone):")
print(f"  stat={kw_stat:.3f}  p={kw_p:.4f}  {'✅ REJECT H0' if kw_p < 0.05 else '❌ CANNOT REJECT H0'}")

# H2: hierarchy depth ≥3 — chi-square test vs. geo zone
if feat_df["geo_zone"].notna().sum() > 10:
    zone_df2 = feat_df[feat_df["geo_zone"].notna()].copy()
    zone_df2["deep"] = (zone_df2["max_hier_level"] >= 3).astype(int)
    ct_deep = pd.crosstab(zone_df2["geo_zone"], zone_df2["deep"])
    if ct_deep.shape[1] == 2:
        chi2_h2, p_h2, _, _ = chi2_contingency(ct_deep)
        print(f"\nH2 (Chi-square deep-hierarchy ≥3 × geo_zone):")
        print(f"  chi2={chi2_h2:.3f}  p={p_h2:.4f}  {'✅ REJECT H0' if p_h2 < 0.05 else '❌ CANNOT REJECT H0'}")
        deep_by_zone = zone_df2[zone_df2["max_hier_level"] >= 3].groupby("geo_zone").size().sort_values(ascending=False)
        print(f"  Deep-hierarchy khipus by zone:\n{deep_by_zone.to_string()}")

# H3: behavioral clusters cross-cut T1/T2 (chi-square)
ct_beh_struct = pd.crosstab(feat_df["beh_label"], feat_df["typology_label"])
chi2_h3, p_h3, _, _ = chi2_contingency(ct_beh_struct)
print(f"\nH3 (Chi-square behavioral cluster × structural typology):")
print(f"  chi2={chi2_h3:.3f}  p={p_h3:.4f}  {'✅ REJECT H0' if p_h3 < 0.05 else '❌ CANNOT REJECT H0'}")
print(f"  → Behavioral clusters {'DO' if p_h3 < 0.05 else 'do NOT'} significantly cross-cut T1/T2")

# Bonus: pattern types × round-number affinity
print(f"\nBonus: Kruskal-Wallis pct_round5 × summation pattern type:")
feat_df["dominant_pattern"] = feat_df[PATTERN_COLS].idxmax(axis=1)
patt_groups = [grp["pct_round5"].values
               for _, grp in feat_df.groupby("dominant_pattern")
               if len(grp) > 5]
if len(patt_groups) > 1:
    k_stat, k_p = kruskal(*patt_groups)
    print(f"  stat={k_stat:.3f}  p={k_p:.4f}  {'✅' if k_p < 0.05 else '❌'}")
    means = feat_df.groupby("dominant_pattern")["pct_round5"].mean().sort_values(ascending=False)
    print(f"  Pattern means:\n{means.to_string()}")

# ── 10. Save per-khipu assignments ──────────────────────────────────────────
out_cols = [
    "kfg_id", "typology_label", "geo_zone", "provenance_display",
    "beh_cluster", "beh_label",
    "value_register", "pct_nonzero", "pct_round5",
    "entropy_per_cord", "max_hier_level",
    "knot_L_ratio", "knot_E_ratio",
] + PATTERN_COLS
feat_df[out_cols].to_csv(OUT_DATA / "phase8_behavioral_clusters.csv", index=False)
print("\n  → phase8_behavioral_clusters.csv saved")

# ── 11. Summary ──────────────────────────────────────────────────────────────
print(f"\n══ Phase 8 Complete (k={best_k}) ════════════════════════════════════")
summary = (
    feat_df.groupby("beh_label")
    .agg(
        n=("kfg_id", "count"),
        median_value=("value_register", "median"),
        pct_round5_mean=("pct_round5", "mean"),
        entropy_mean=("entropy_per_cord", "mean"),
        max_hier_median=("max_hier_level", "median"),
        L_ratio_mean=("knot_L_ratio", "mean"),
        E_ratio_mean=("knot_E_ratio", "mean"),
    )
    .reset_index()
)
print(summary.to_string(index=False))
print()
for f in sorted(OUT_VIZ.glob("*.png")):
    print(f"    {f.relative_to(ROOT)}")
