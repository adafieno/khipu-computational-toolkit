"""
Phase 3: Structural Typology via Clustering

Clusters 709 K-CAT khipus on their 9-dimensional summation pattern vector,
enriches with structural features, and cross-tabulates against available
provenance / institutional metadata.

Outputs (all written to data/processed/phase3_*)
-------------------------------------------------
phase3_feature_matrix.csv   — full per-khipu feature matrix (from feature_matrix.py)
phase3_clusters.csv         — feature matrix + cluster assignments
phase3_silhouette.csv       — silhouette scores for k=2..10

Visualizations (visualizations/phase3/)
---------------------------------------
umap_by_cluster.png         — UMAP 2-D embedding coloured by k-means cluster
umap_by_n_types.png         — UMAP coloured by n_pattern_types (0-9)
umap_by_region.png          — UMAP coloured by origin region (provenance_display / region)
heatmap_cluster_patterns.png— cluster × pattern heatmap (mean has_* per cluster)
silhouette_curve.png        — silhouette score vs k

Usage
-----
    python scripts/run_phase3_typology.py [--db PATH] [--k INT] [--force]

    --db     path to K-CAT SQLite database  (default: data/kfg/khipu_database.db)
    --k      number of k-means clusters (default: auto-select via silhouette)
    --force  rebuild feature matrix even if cached CSV exists
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Optional UMAP — graceful fallback to PCA if not installed
try:
    from umap import UMAP
    _HAS_UMAP = True
except ImportError:
    from sklearn.decomposition import PCA
    _HAS_UMAP = False
    warnings.warn(
        "umap-learn not installed — using PCA for 2-D projection. "
        "Install with: pip install umap-learn",
        stacklevel=2,
    )

# Resolve repo root regardless of where script is called from
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from src.analysis.feature_matrix import build_feature_matrix, PATTERN_KEYS, BINARY_COLS

DB_DEFAULT       = str(_REPO / "data" / "kfg" / "khipu_database.db")
PROCESSED_DIR    = _REPO / "data" / "processed"
VIZ_DIR          = _REPO / "visualizations" / "phase3"
MATRIX_CSV       = PROCESSED_DIR / "phase3_feature_matrix.csv"
CLUSTERS_CSV     = PROCESSED_DIR / "phase3_clusters.csv"
SILHOUETTE_CSV   = PROCESSED_DIR / "phase3_silhouette.csv"

STRUCT_COLS  = ["n_cords", "n_groups", "numeric_coverage"]
FEATURE_COLS = BINARY_COLS + STRUCT_COLS   # input to clustering

# ── Colour palettes ──────────────────────────────────────────────────────────
CLUSTER_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#8dd3c7", "#fb9a99",
]
NTYPE_CMAP = "viridis"


# ---------------------------------------------------------------------------
# 1. Load / build feature matrix
# ---------------------------------------------------------------------------

def load_or_build_matrix(db: str, force: bool) -> pd.DataFrame:
    if not force and MATRIX_CSV.exists():
        print(f"Loading cached feature matrix -> {MATRIX_CSV}")
        return pd.read_csv(MATRIX_CSV)
    print("Building feature matrix (this takes ~2 min first time) …")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = build_feature_matrix(db, verbose=True)
    df.to_csv(MATRIX_CSV, index=False)
    print(f"Saved -> {MATRIX_CSV}")
    return df


# ---------------------------------------------------------------------------
# 2. Clustering
# ---------------------------------------------------------------------------

def select_k(X: np.ndarray, k_range=(2, 10)) -> tuple[int, pd.DataFrame]:
    """Return best k (highest silhouette) and the full score table."""
    scores = []
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels)
        scores.append({"k": k, "silhouette": round(s, 4)})
        print(f"  k={k:2d}  silhouette={s:.4f}")
    sil_df = pd.DataFrame(scores)
    best_k = int(sil_df.loc[sil_df["silhouette"].idxmax(), "k"])
    print(f"\nBest k = {best_k}  (silhouette = {sil_df['silhouette'].max():.4f})")
    return best_k, sil_df


def run_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=42, n_init=30)
    return km.fit_predict(X)


def run_embedding(X: np.ndarray) -> np.ndarray:
    """Return a (n, 2) 2-D projection for visualisation."""
    if _HAS_UMAP:
        print("Computing UMAP embedding …")
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    else:
        print("Computing PCA embedding (umap-learn not available) …")
        reducer = PCA(n_components=2, random_state=42)
    return reducer.fit_transform(X)


# ---------------------------------------------------------------------------
# 3. Visualisation helpers
# ---------------------------------------------------------------------------

VIZ_DIR.mkdir(parents=True, exist_ok=True)


def _savefig(name: str):
    path = VIZ_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {path.relative_to(_REPO)}")


def plot_embedding_by_cluster(emb: np.ndarray, labels: np.ndarray, k: int, method: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    for c in range(k):
        mask = labels == c
        ax.scatter(emb[mask, 0], emb[mask, 1], s=14, alpha=0.7,
                   color=CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)],
                   label=f"Cluster {c+1} (n={mask.sum()})")
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(f"K-CAT Khipus — {method} coloured by k-means cluster (k={k})")
    ax.legend(markerscale=2, fontsize=8)
    _savefig(f"{'umap' if _HAS_UMAP else 'pca'}_by_cluster.png")


def plot_embedding_by_ntype(emb: np.ndarray, n_types: np.ndarray, method: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=n_types, cmap=NTYPE_CMAP,
                    s=14, alpha=0.7, vmin=0, vmax=9)
    plt.colorbar(sc, ax=ax, label="Number of pattern types")
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(f"K-CAT Khipus — {method} coloured by pattern-type count")
    _savefig(f"{'umap' if _HAS_UMAP else 'pca'}_by_n_types.png")


def plot_embedding_by_region(emb: np.ndarray, regions: pd.Series, method: str):
    """Colour UMAP/PCA by origin region (provenance_display, falling back to region)."""
    top = regions.value_counts().head(6).index.tolist()
    cats = [c if c in top else "Other / unknown" for c in regions.fillna("Unknown")]
    unique_cats = sorted(set(cats))
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(unique_cats))
    cat_to_color = {c: cmap(i) for i, c in enumerate(unique_cats)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for cat in unique_cats:
        mask = np.array([c == cat for c in cats])
        ax.scatter(emb[mask, 0], emb[mask, 1], s=14, alpha=0.7,
                   color=cat_to_color[cat], label=f"{cat} ({mask.sum()})")
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(f"K-CAT Khipus — {method} coloured by origin region")
    ax.legend(markerscale=2, fontsize=8)
    _savefig(f"{'umap' if _HAS_UMAP else 'pca'}_by_region.png")


def plot_cluster_pattern_heatmap(df: pd.DataFrame, k: int):
    """Mean has_* per cluster, displayed as a heatmap."""
    cluster_means = df.groupby("cluster")[BINARY_COLS].mean()
    fig, ax = plt.subplots(figsize=(11, max(3, k * 0.7 + 1)))
    im = ax.imshow(cluster_means.values, aspect="auto", vmin=0, vmax=1, cmap="YlOrRd")
    ax.set_xticks(range(len(BINARY_COLS)))
    ax.set_xticklabels([c.replace("has_", "").upper() for c in BINARY_COLS], fontsize=8)
    ax.set_yticks(range(k))
    ax.set_yticklabels(
        [f"Cluster {c+1} (n={int((df['cluster']==c).sum())})" for c in range(k)],
        fontsize=8,
    )
    for y in range(k):
        for x in range(len(BINARY_COLS)):
            val = cluster_means.values[y, x]
            ax.text(x, y, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if val < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="Fraction of khipus in cluster with pattern")
    ax.set_title(f"Pattern prevalence per cluster (k={k})")
    _savefig("heatmap_cluster_patterns.png")


def plot_silhouette_curve(sil_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sil_df["k"], sil_df["silhouette"], marker="o", color="#377eb8")
    best = sil_df.loc[sil_df["silhouette"].idxmax()]
    ax.axvline(best["k"], color="#e41a1c", linestyle="--", alpha=0.6,
               label=f"Best k={int(best['k'])} ({best['silhouette']:.4f})")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("K-means silhouette score vs k")
    ax.legend()
    _savefig("silhouette_curve.png")


# ---------------------------------------------------------------------------
# 4. Cross-tabulation summaries
# ---------------------------------------------------------------------------

def print_crosstabs(df: pd.DataFrame, k: int):
    print("\n=== Cluster sizes ===")
    print(df["cluster"].value_counts().sort_index().to_string())

    print("\n=== Pattern prevalence per cluster (mean has_*) ===")
    print(df.groupby("cluster")[BINARY_COLS].mean().round(3).to_string())

    print("\n=== Mean n_pattern_types per cluster ===")
    print(df.groupby("cluster")["n_pattern_types"].mean().round(2).to_string())

    print("\n=== Mean n_cords per cluster ===")
    print(df.groupby("cluster")["n_cords"].mean().round(0).to_string())

    print("\n=== Origin region per cluster ===")
    region_col = "provenance_display" if "provenance_display" in df.columns else "region"
    ct = pd.crosstab(df["cluster"], df[region_col].fillna("Unknown"))
    print(ct.to_string())

    print("\n=== Extremes: all 9 patterns ===")
    all9 = df[df["n_pattern_types"] == len(PATTERN_KEYS)]
    print(f"  {len(all9)} khipus carry all {len(PATTERN_KEYS)} patterns")
    if len(all9):
        print(all9[["kfg_id", "cluster", "n_cords", "region"]].to_string())

    print("\n=== Extremes: exactly 1 pattern ===")
    one = df[df["n_pattern_types"] == 1]
    print(f"  {len(one)} khipus carry exactly 1 pattern")
    if len(one):
        dominant = [BINARY_COLS[x] for x in df.loc[one.index, BINARY_COLS].values.argmax(axis=1)]
        print(one[["kfg_id", "cluster", "n_cords"]].assign(pattern=dominant).to_string())


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 3 structural typology")
    parser.add_argument("--db",    default=DB_DEFAULT, help="K-CAT SQLite database")
    parser.add_argument("--k",     type=int, default=None,
                        help="Number of k-means clusters (default: auto-select)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild feature matrix even if cached CSV exists")
    args = parser.parse_args()

    # 1. Feature matrix
    df = load_or_build_matrix(args.db, force=args.force)

    # 2. Prepare feature array
    feature_df = df[FEATURE_COLS].copy()
    # Scale structural columns, leave binary columns as-is
    scaler = StandardScaler()
    X_struct = scaler.fit_transform(feature_df[STRUCT_COLS])
    X = np.hstack([feature_df[BINARY_COLS].values.astype(float), X_struct])

    # 3. Silhouette sweep
    print("\nSilhouette sweep (k=2..10) …")
    if args.k is None:
        best_k, sil_df = select_k(X)
    else:
        best_k = args.k
        sil_df = pd.DataFrame({"k": [best_k], "silhouette": [
            silhouette_score(X, KMeans(n_clusters=best_k, random_state=42, n_init=20).fit_predict(X))
        ]})
        print(f"Using user-specified k={best_k}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sil_df.to_csv(SILHOUETTE_CSV, index=False)
    print(f"Saved silhouette scores -> {SILHOUETTE_CSV.relative_to(_REPO)}")

    # 4. Final clustering
    print(f"\nRunning k-means with k={best_k} …")
    labels = run_kmeans(X, best_k)
    df["cluster"] = labels

    df.to_csv(CLUSTERS_CSV, index=False)
    print(f"Saved cluster assignments -> {CLUSTERS_CSV.relative_to(_REPO)}")

    # 5. Embedding
    emb = run_embedding(X)
    df["emb_x"] = emb[:, 0]
    df["emb_y"] = emb[:, 1]
    method = "UMAP" if _HAS_UMAP else "PCA"

    # 6. Plots
    print("\nGenerating visualisations …")
    plot_silhouette_curve(sil_df)
    plot_embedding_by_cluster(emb, labels, best_k, method)
    plot_embedding_by_ntype(emb, df["n_pattern_types"].values, method)
    region_col = "provenance_display" if "provenance_display" in df.columns else "region"
    plot_embedding_by_region(emb, df[region_col], method)
    plot_cluster_pattern_heatmap(df, best_k)

    # 7. Console summaries
    print_crosstabs(df, best_k)

    print("\nPhase 3 complete.")
    print(f"  Clusters CSV  : {CLUSTERS_CSV.relative_to(_REPO)}")
    print(f"  Visualisations: {VIZ_DIR.relative_to(_REPO)}/")


if __name__ == "__main__":
    main()
