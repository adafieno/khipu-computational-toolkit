"""
Phase 10: Summation Structure & Zero-Cord Analysis
====================================================
Phase 10 was originally designed as missing-value prediction.  A preliminary
survey revealed that the KFG cord table has *no* missing values (all 62,746 cords
carry a recorded value).  However, 30.3 % of cords have value = 0, and only 6.6 %
of parent-child groups satisfy the additive summation constraint
(sum of children ≈ parent value).  These two facts define a richer investigative
agenda:

  Q1. What arithmetic relationship *actually* holds between parent and child
      cords when simple summation fails?
  Q2. Are zero-value cords placeholders / structural slots, or genuine zero counts?
  Q3. Which khipu features predict summation compliance?

Summary of analyses
───────────────────
  A. Summation compliance audit
     For every parent-child group, compute children_sum / parent_value (the
     "summation ratio").  Classify groups as:
       • compliant     ratio in [0.95, 1.05]
       • partial       ratio in [0.50, 0.95) or (1.05, 1.50]
       • sub           ratio < 0.50   (children summed << parent)
       • supra         ratio > 1.50   (children summed >> parent)
       • trivial       parent = 0

  B. Ratio landmark analysis
     Test whether the distribution of summation ratios clusters near simple
     fractions: 1/10, 1/3, 1/2, 1, 2, 10 etc. — a signature of decimal or
     fractional accounting conventions.

  C. Zero-cord profile
     Per khipu: pct_zero, are zeros concentrated at pendants or subsidiaries,
     are zeros associated with particular behavioral clusters?

  D. Summation compliance predictor (logistic regression)
     Target: is this khipu's predominant summation class "compliant"?
     Features: beh_label (OHE), typology_label, depth, branching_entropy,
               pct_round5, pct_zero, geo_zone (OHE).

Inputs
──────
  data/kfg/khipu_database.db
  data/processed/phase9_graph_metrics.csv   (depth, branching_entropy, etc.)
  data/processed/phase8_behavioral_clusters.csv (beh_label, pct_round5, typology, zone)

Outputs
───────
  data/processed/phase10_summation_groups.csv
  data/processed/phase10_zero_analysis.csv
  visualizations/phase10/ratio_distribution.png
  visualizations/phase10/compliance_by_cluster.png
  visualizations/phase10/zero_cord_patterns.png
  visualizations/phase10/compliance_predictors.png
"""

from pathlib import Path
import sqlite3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DB_PATH  = ROOT / "data" / "kfg" / "khipu_database.db"
P8_CSV   = ROOT / "data" / "processed" / "phase8_behavioral_clusters.csv"
P9_CSV   = ROOT / "data" / "processed" / "phase9_graph_metrics.csv"
OUT_DATA = ROOT / "data" / "processed"
OUT_VIZ  = ROOT / "visualizations" / "phase10"
OUT_VIZ.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ────────────────────────────────────────────────────────────
print("Loading cord data …")
conn  = sqlite3.connect(DB_PATH)
cords = pd.read_sql(
    "SELECT kfg_id, cord_id, cord_name, parent_cord, hierarchy_level, value FROM cords",
    conn
)
conn.close()

cords["hierarchy_level"] = cords["hierarchy_level"].fillna(0).astype(int)
cords["parent_cord"]     = cords["parent_cord"].fillna("").str.strip()
cords["value_num"]       = pd.to_numeric(cords["value"], errors="coerce").fillna(0.0)

print(f"  {len(cords):,} cords  /  {cords['kfg_id'].nunique()} khipus")
print(f"  Zero-value cords: {(cords['value_num']==0).sum():,} ({(cords['value_num']==0).mean()*100:.1f}%)")

# ── 2. Build parent-child summation table ──────────────────────────────────
print("\nBuilding parent-child summation groups …")

sum_rows = []
for kfg_id, grp in cords.groupby("kfg_id"):
    name2val = dict(zip(grp["cord_name"], grp["value_num"]))
    for _, row in grp.iterrows():
        if row["parent_cord"] and row["parent_cord"] in name2val:
            pval = name2val[row["parent_cord"]]
            cval = row["value_num"]
            sum_rows.append({
                "kfg_id":        kfg_id,
                "parent_name":   row["parent_cord"],
                "child_name":    row["cord_name"],
                "parent_val":    pval,
                "child_val":     cval,
                "child_level":   row["hierarchy_level"],
            })

rel = pd.DataFrame(sum_rows)

# Aggregate to parent-group level
pg = (rel.groupby(["kfg_id", "parent_name", "parent_val"])
         .agg(children_sum=("child_val","sum"),
              n_children=("child_val","count"),
              n_nonzero_children=("child_val", lambda x: (x>0).sum()),
              n_zero_children=("child_val", lambda x: (x==0).sum()),
              children_mean=("child_val","mean"),
              children_max=("child_val","max"))
         .reset_index())

# Summation ratio
pg["ratio"] = np.where(
    pg["parent_val"] > 0,
    pg["children_sum"] / pg["parent_val"],
    np.nan
)

def classify_ratio(r):
    if pd.isna(r):       return "trivial_parent_zero"
    if r < 0.05:         return "children_all_zero"
    if r < 0.50:         return "sub"
    if r < 0.95:         return "partial_sub"
    if r <= 1.05:        return "compliant"
    if r <= 1.50:        return "partial_supra"
    return "supra"

pg["compliance_class"] = pg["ratio"].apply(classify_ratio)
pg.to_csv(OUT_DATA / "phase10_summation_groups.csv", index=False)
print(f"  {len(pg):,} parent groups  →  phase10_summation_groups.csv")

# Summary
cc = pg["compliance_class"].value_counts()
print("\n  Summation compliance breakdown:")
for k, v in cc.items():
    print(f"    {k:<28} {v:5,}  ({v/len(pg)*100:.1f}%)")

# ── 3. Per-khipu summation profile ────────────────────────────────────────
print("\nBuilding per-khipu summation profile …")

khipu_sum = pg.groupby("kfg_id").apply(
    lambda d: pd.Series({
        "n_groups":         len(d),
        "n_compliant":      (d["compliance_class"] == "compliant").sum(),
        "n_sub":            (d["compliance_class"] == "sub").sum(),
        "n_supra":          (d["compliance_class"] == "supra").sum(),
        "n_trivial":        (d["compliance_class"] == "trivial_parent_zero").sum(),
        "pct_compliant":    (d["compliance_class"] == "compliant").mean() * 100,
        "median_ratio":     d["ratio"].median(),
        "mean_ratio":       d["ratio"].mean(),
    })
).reset_index()

# Zero cord profile per khipu
zero_profile = cords.groupby("kfg_id").apply(
    lambda d: pd.Series({
        "n_cords":          len(d),
        "n_zero":           (d["value_num"] == 0).sum(),
        "pct_zero":         (d["value_num"] == 0).mean() * 100,
        "n_zero_pendant":   ((d["value_num"] == 0) & (d["hierarchy_level"] == 0)).sum(),
        "n_zero_sub":       ((d["value_num"] == 0) & (d["hierarchy_level"] > 0)).sum(),
        "pct_zero_pendant": ((d["value_num"] == 0) & (d["hierarchy_level"] == 0)).sum() / max((d["hierarchy_level"] == 0).sum(), 1) * 100,
        "pct_zero_sub":     ((d["value_num"] == 0) & (d["hierarchy_level"] > 0)).sum() / max((d["hierarchy_level"] > 0).sum(), 1) * 100,
    })
).reset_index()

zero_profile = zero_profile.merge(khipu_sum, on="kfg_id", how="left")

# Merge with behavioral labels
p8 = pd.read_csv(P8_CSV)[["kfg_id","beh_label","typology_label","geo_zone","pct_round5"]]
p9 = pd.read_csv(P9_CSV)[["kfg_id","depth","branching_entropy","balance_score"]]

zero_profile = zero_profile.merge(p8, on="kfg_id", how="left")
zero_profile = zero_profile.merge(p9, on="kfg_id", how="left")
zero_profile.to_csv(OUT_DATA / "phase10_zero_analysis.csv", index=False)
print(f"  {len(zero_profile):,} khipus  →  phase10_zero_analysis.csv")

# Print zero distribution by cluster
print("\n  Zero-cord % by behavioral cluster (median):")
print(zero_profile.groupby("beh_label")["pct_zero"].median().round(1).to_string())

# ── 4. Ratio landmark analysis ─────────────────────────────────────────────
print("\n── Ratio landmark analysis ───────────────────────────────────────────")
ratios = pg["ratio"].dropna()
landmarks = {
    "1/10 (0.10)": (0.08, 0.12),
    "1/5  (0.20)": (0.18, 0.22),
    "1/3  (0.33)": (0.31, 0.36),
    "1/2  (0.50)": (0.48, 0.52),
    "2/3  (0.67)": (0.64, 0.70),
    "1/1  (1.00)": (0.95, 1.05),
    "2/1  (2.00)": (1.90, 2.10),
    "10/1 (10.0)": (9.0, 11.0),
}
print(f"  Total ratios: {len(ratios):,}")
for name, (lo, hi) in landmarks.items():
    n = ratios.between(lo, hi).sum()
    print(f"  {name:<16}  {n:5,}  ({n/len(ratios)*100:.1f}%)")

# ── 5. Plot: ratio distribution ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Phase 10: Parent-Child Summation Ratio Distribution", fontsize=12)

ax = axes[0]
# Log-scaled ratio distribution
log_r = np.log10(ratios.clip(lower=0.001, upper=100))
ax.hist(log_r, bins=80, color="#2c7bb6", edgecolor="white", alpha=0.8)
ax.set_xlabel("log₁₀(children_sum / parent_value)")
ax.set_ylabel("Parent-group count")
ax.set_title("Full ratio distribution (log scale)")
for lm_name, (lo, hi) in landmarks.items():
    lm_val = (lo + hi) / 2
    ax.axvline(np.log10(lm_val), color="red", linestyle=":", linewidth=0.8)
ax.axvline(np.log10(1.0), color="red", linestyle="--", linewidth=1.5, label="Ratio=1 (compliant)")
ax.legend(fontsize=8)

ax = axes[1]
cc_plot = cc.reindex(["compliant","partial_sub","partial_supra","sub","supra",
                       "children_all_zero","trivial_parent_zero"]).fillna(0)
colors = ["#2ca02c","#98df8a","#aec7e8","#d62728","#ff9896","#c5b0d5","#7f7f7f"]
ax.bar(range(len(cc_plot)), cc_plot.values, color=colors[:len(cc_plot)], edgecolor="white")
ax.set_xticks(range(len(cc_plot)))
ax.set_xticklabels(cc_plot.index, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Parent-group count")
ax.set_title("Compliance class distribution")
for i, v in enumerate(cc_plot.values):
    ax.text(i, v + 20, str(int(v)), ha="center", fontsize=7)

plt.tight_layout()
fig.savefig(OUT_VIZ / "ratio_distribution.png", dpi=150)
plt.close(fig)
print("\n  → ratio_distribution.png saved")

# ── 6. Plot: compliance by behavioral cluster ─────────────────────────────
beh_labels = sorted(zero_profile["beh_label"].dropna().unique())

comp_by_beh = (zero_profile.groupby("beh_label")
               .agg(pct_compliant_mean=("pct_compliant","mean"),
                    pct_compliant_med=("pct_compliant","median"),
                    pct_zero_med=("pct_zero","median"),
                    n=("kfg_id","count"))
               .reindex(beh_labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Phase 10: Summation Compliance and Zero-Cord Rate by Behavioral Cluster",
             fontsize=11)

ax = axes[0]
ax.bar(beh_labels, comp_by_beh["pct_compliant_med"], color="#2c7bb6", edgecolor="white")
ax.set_ylabel("Median % summation-compliant groups per khipu")
ax.set_xlabel("Behavioral cluster")
ax.set_title("Summation compliance")
corpus_comp = zero_profile["pct_compliant"].median()
ax.axhline(corpus_comp, color="red", linestyle="--", label=f"Corpus median={corpus_comp:.1f}%")
ax.legend(fontsize=8)
for i, (bl, row) in enumerate(comp_by_beh.iterrows()):
    ax.text(i, row["pct_compliant_med"] + 0.5, f"n={int(row['n'])}", ha="center", fontsize=8)

ax = axes[1]
ax.bar(beh_labels, comp_by_beh["pct_zero_med"], color="#d62728", edgecolor="white", alpha=0.85)
ax.set_ylabel("Median % zero-value cords per khipu")
ax.set_xlabel("Behavioral cluster")
ax.set_title("Zero-value cord rate")
corpus_zero = zero_profile["pct_zero"].median()
ax.axhline(corpus_zero, color="black", linestyle="--", label=f"Corpus median={corpus_zero:.1f}%")
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT_VIZ / "compliance_by_cluster.png", dpi=150)
plt.close(fig)
print("  → compliance_by_cluster.png saved")

# ── 7. Plot: zero-cord patterns ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Phase 10: Zero-Value Cord Distribution Patterns", fontsize=11)

ax = axes[0]
ax.hist(zero_profile["pct_zero"], bins=40, color="#7f7f7f", edgecolor="white", alpha=0.85)
ax.set_xlabel("% zero-value cords in khipu")
ax.set_ylabel("Khipu count")
ax.set_title("Distribution of zero-cord rates")
ax.axvline(zero_profile["pct_zero"].median(), color="red", linestyle="--",
           label=f"Median={zero_profile['pct_zero'].median():.1f}%")
ax.legend(fontsize=8)

ax = axes[1]
ax.scatter(zero_profile["pct_zero"], zero_profile["pct_round5"],
           c=zero_profile["beh_label"].map({"B1":0,"B2":1,"B3":2,"B4":3,"B5":4,"B6":5}),
           cmap="tab10", alpha=0.5, s=20)
ax.set_xlabel("% zero-value cords")
ax.set_ylabel("% round-5 values (Phase 8 feature)")
ax.set_title("Zero rate vs round-number affinity")

ax = axes[2]
# Pendant vs subsidiary zero rates
ax.scatter(zero_profile["pct_zero_pendant"], zero_profile["pct_zero_sub"],
           c=zero_profile["depth"], cmap="viridis", alpha=0.6, s=20)
ax.set_xlabel("% zeros among pendants")
ax.set_ylabel("% zeros among subsidiaries")
ax.set_title("Zero rate: pendants vs subsidiaries")
sm = plt.cm.ScalarMappable(cmap="viridis",
     norm=plt.Normalize(vmin=zero_profile["depth"].min(), vmax=zero_profile["depth"].max()))
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Tree depth")

plt.tight_layout()
fig.savefig(OUT_VIZ / "zero_cord_patterns.png", dpi=150)
plt.close(fig)
print("  → zero_cord_patterns.png saved")

# ── 8. Predictive model: compliance ───────────────────────────────────────
print("\n── Logistic regression: predict high-compliance khipus ───────────────")

model_df = zero_profile.dropna(subset=["beh_label","typology_label",
                                        "depth","branching_entropy",
                                        "pct_round5","pct_zero"]).copy()
model_df["y"] = (model_df["pct_compliant"] >= 10).astype(int)
print(f"  Target: pct_compliant >= 10%  →  positive={model_df['y'].sum()}  "
      f"negative={len(model_df)-model_df['y'].sum()}")

# One-hot encode categoricals
feature_cols = ["depth","branching_entropy","pct_round5","pct_zero","balance_score"]
X_df = model_df[feature_cols].fillna(0)
beh_dummies = pd.get_dummies(model_df["beh_label"], prefix="beh")
typ_dummies = pd.get_dummies(model_df["typology_label"], prefix="typ")
X = pd.concat([X_df, beh_dummies, typ_dummies], axis=1).astype(float)
y = model_df["y"].values

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
cv_scores = cross_val_score(lr, X_sc, y, cv=5, scoring="roc_auc")
print(f"  5-fold CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

lr.fit(X_sc, y)
feat_imp = pd.Series(np.abs(lr.coef_[0]), index=X.columns).sort_values(ascending=False)
print("\n  Top 10 features by |coefficient|:")
print(feat_imp.head(10).to_string())

# ── 9. Compliance predictor plot ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Phase 10: Predicting Summation-Compliant Khipus", fontsize=11)

ax = axes[0]
top_feats = feat_imp.head(10)
ax.barh(range(len(top_feats)), top_feats.values, color="#2c7bb6", edgecolor="white")
ax.set_yticks(range(len(top_feats)))
ax.set_yticklabels(top_feats.index, fontsize=9)
ax.set_xlabel("|Logistic regression coefficient|")
ax.set_title(f"Feature importance\n(CV AUC={cv_scores.mean():.3f})")
ax.invert_yaxis()

ax = axes[1]
# pct_round5 vs pct_compliant scatter with compliance coloring
compliant_mask = (zero_profile["pct_compliant"].fillna(0) >= 10)
ax.scatter(zero_profile[~compliant_mask]["pct_round5"],
           zero_profile[~compliant_mask]["pct_compliant"],
           c="#d62728", alpha=0.35, s=18, label="Low compliance (<10%)")
ax.scatter(zero_profile[compliant_mask]["pct_round5"],
           zero_profile[compliant_mask]["pct_compliant"],
           c="#2ca02c", alpha=0.55, s=25, label="High compliance (≥10%)")
ax.set_xlabel("% round-5 values (quota signal)")
ax.set_ylabel("% summation-compliant groups")
ax.set_title("Round-number affinity vs summation compliance")
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT_VIZ / "compliance_predictors.png", dpi=150)
plt.close(fig)
print("  → compliance_predictors.png saved")

# ── 10. Statistical tests ──────────────────────────────────────────────────
print("\n── Statistical tests ─────────────────────────────────────────────────")

# Kruskal-Wallis: pct_compliant across behavioral clusters
kw_groups = [grp["pct_compliant"].dropna().values
             for _, grp in zero_profile.groupby("beh_label") if len(grp) >= 3]
if len(kw_groups) >= 2:
    stat, p = kruskal(*kw_groups)
    print(f"\nKruskal-Wallis pct_compliant × beh_label: H={stat:.2f}  p={p:.4f}  "
          f"{'✅' if p < 0.05 else '❌'}")

# Kruskal-Wallis: pct_zero across behavioral clusters
kw_z = [grp["pct_zero"].dropna().values
        for _, grp in zero_profile.groupby("beh_label") if len(grp) >= 3]
if len(kw_z) >= 2:
    stat_z, p_z = kruskal(*kw_z)
    print(f"Kruskal-Wallis pct_zero × beh_label:       H={stat_z:.2f}  p={p_z:.4f}  "
          f"{'✅' if p_z < 0.05 else '❌'}")

# Mann-Whitney: T1 vs T2 on pct_compliant and pct_zero
t1z = zero_profile[zero_profile["typology_label"]=="T1"]["pct_compliant"].dropna()
t2z = zero_profile[zero_profile["typology_label"]=="T2"]["pct_compliant"].dropna()
u, p = mannwhitneyu(t1z, t2z, alternative="two-sided")
print(f"\nMann-Whitney pct_compliant T1 vs T2: U={u:.1f}  p={p:.4f}  "
      f"T1_med={t1z.median():.1f}%  T2_med={t2z.median():.1f}%  "
      f"{'***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ''))}")

t1z2 = zero_profile[zero_profile["typology_label"]=="T1"]["pct_zero"].dropna()
t2z2 = zero_profile[zero_profile["typology_label"]=="T2"]["pct_zero"].dropna()
u2, p2 = mannwhitneyu(t1z2, t2z2, alternative="two-sided")
print(f"Mann-Whitney pct_zero     T1 vs T2: U={u2:.1f}  p={p2:.4f}  "
      f"T1_med={t1z2.median():.1f}%  T2_med={t2z2.median():.1f}%  "
      f"{'***' if p2<0.001 else ('**' if p2<0.01 else ('*' if p2<0.05 else ''))}")

# ── 11. Final summary ──────────────────────────────────────────────────────
print(f"\n══ Phase 10 Complete ══════════════════════════════════════════════════")
print("\nPer-cluster median summary:")
print(zero_profile.groupby("beh_label")[["pct_zero","pct_compliant","median_ratio"]].median().round(2))
print()
for f in sorted(OUT_VIZ.glob("*.png")):
    print(f"    {f.relative_to(ROOT)}")
for f in sorted(OUT_DATA.glob("phase10_*.csv")):
    print(f"    {f.relative_to(ROOT)}")
