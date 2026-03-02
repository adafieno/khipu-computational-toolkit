"""
Cluster Khipus by Structural Patterns

This script performs clustering analysis on khipus based on their structural
features and similarity metrics. Uses multiple clustering approaches:
1. K-means clustering on structural features
2. Hierarchical clustering
3. DBSCAN for density-based clusters

Analyzes resulting clusters for common patterns and characteristics.
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import sqlite3

# Add src to path for runtime
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore


class KhipuClusterer:
    """Cluster khipus by structural patterns."""

    def __init__(self, features_path: Path, db_path: Path):
        print(f"Loading structural features from {features_path}...")
        self.features_df = pd.read_csv(features_path)
        print(f"✓ Loaded features for {len(self.features_df)} khipus")

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def prepare_features(self) -> tuple:
        """Prepare and normalize features for clustering."""
        # Select features for clustering
        feature_cols = [
            'num_nodes', 'num_edges', 'avg_degree', 'max_degree', 'density',
            'depth', 'width', 'avg_branching', 'num_roots', 'num_leaves',
            'has_numeric', 'has_color'
        ]

        # Filter out khipus with no nodes
        valid_df = self.features_df[self.features_df['num_nodes'] > 0].copy()

        X = valid_df[feature_cols].values
        khipu_ids = valid_df['khipu_id'].values

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, khipu_ids, valid_df, scaler

    def perform_kmeans(
        self,
        X: np.ndarray,
        khipu_ids: np.ndarray,
        k_range: range = range(
            3,
            11)) -> dict:
        """Perform K-means clustering with multiple k values."""
        print("\nPerforming K-means clustering...")

        results = {}
        scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            # Compute quality metrics
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)

            scores.append({
                'k': k,
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'inertia': kmeans.inertia_
            })

            print(f"  k={k}: silhouette={silhouette:.4f}, CH={calinski:.2f}")

        # Choose best k based on silhouette score
        best_idx = max(range(len(scores)),
                       key=lambda i: scores[i]['silhouette'])
        best_k = scores[best_idx]['k']

        print(
            f"\n✓ Best k={best_k} (silhouette={scores[best_idx]['silhouette']:.4f})")

        # Perform final clustering with best k
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        results = {
            'method': 'kmeans',
            'best_k': best_k,
            'labels': labels,
            'khipu_ids': khipu_ids,
            'scores': scores,
            'centroids': kmeans.cluster_centers_
        }

        return results

    def perform_hierarchical(
            self,
            X: np.ndarray,
            khipu_ids: np.ndarray,
            n_clusters: int = 5) -> dict:
        """Perform hierarchical (agglomerative) clustering."""
        print(
            f"\nPerforming hierarchical clustering (n_clusters={n_clusters})...")

        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters, linkage='ward')
        labels = clusterer.fit_predict(X)

        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)

        print(f"✓ Silhouette score: {silhouette:.4f}")
        print(f"✓ Calinski-Harabasz score: {calinski:.2f}")

        return {
            'method': 'hierarchical',
            'n_clusters': n_clusters,
            'labels': labels,
            'khipu_ids': khipu_ids,
            'silhouette': silhouette,
            'calinski_harabasz': calinski
        }

    def perform_dbscan(
            self,
            X: np.ndarray,
            khipu_ids: np.ndarray,
            eps: float = 1.0,
            min_samples: int = 5) -> dict:
        """Perform DBSCAN density-based clustering."""
        print(
            f"\nPerforming DBSCAN clustering (eps={eps}, min_samples={min_samples})...")

        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        print(f"✓ Found {n_clusters} clusters")
        print(f"✓ Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")

        # Compute silhouette only if we have valid clusters
        silhouette = None
        if n_clusters > 1 and n_noise < len(labels):
            try:
                # Exclude noise points for silhouette calculation
                valid_mask = labels != -1
                if sum(valid_mask) > n_clusters:
                    silhouette = silhouette_score(
                        X[valid_mask], labels[valid_mask])
                    print(f"✓ Silhouette score: {silhouette:.4f}")
            except Exception:
                pass

        return {
            'method': 'dbscan',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'labels': labels,
            'khipu_ids': khipu_ids,
            'silhouette': silhouette
        }

    def analyze_clusters(
            self,
            clustering_result: dict,
            features_df: pd.DataFrame) -> dict:
        """Analyze characteristics of each cluster."""
        print("\nAnalyzing cluster characteristics...")

        labels = clustering_result['labels']
        khipu_ids = clustering_result['khipu_ids']

        # Create dataframe with cluster assignments
        cluster_df = pd.DataFrame({
            'khipu_id': khipu_ids,
            'cluster': labels
        })

        # Merge with features
        cluster_df = cluster_df.merge(features_df, on='khipu_id', how='left')

        # Compute cluster statistics
        cluster_stats = []
        unique_clusters = sorted([c for c in set(labels) if c != -1])

        for cluster_id in unique_clusters:
            cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]

            stats = {
                'cluster_id': int(cluster_id),
                'size': len(cluster_data),
                'avg_num_nodes': float(cluster_data['num_nodes'].mean()),
                'avg_depth': float(cluster_data['depth'].mean()),
                'avg_branching': float(cluster_data['avg_branching'].mean()),
                'avg_numeric_coverage': float(cluster_data['has_numeric'].mean()),
                'avg_color_coverage': float(cluster_data['has_color'].mean()),
                'std_num_nodes': float(cluster_data['num_nodes'].std()),
                'min_num_nodes': int(cluster_data['num_nodes'].min()),
                'max_num_nodes': int(cluster_data['num_nodes'].max())
            }

            cluster_stats.append(stats)

            print(f"\nCluster {cluster_id} (n={stats['size']}):")
            print(
                f"  Avg nodes: {stats['avg_num_nodes']:.1f} (std={stats['std_num_nodes']:.1f})")
            print(f"  Avg depth: {stats['avg_depth']:.2f}")
            print(f"  Avg branching: {stats['avg_branching']:.2f}")
            print(f"  Numeric coverage: {stats['avg_numeric_coverage']:.2%}")

        # Add noise cluster if present
        if -1 in labels:
            noise_data = cluster_df[cluster_df['cluster'] == -1]
            print(f"\nNoise points (n={len(noise_data)}):")
            print(f"  Avg nodes: {noise_data['num_nodes'].mean():.1f}")

        return {
            'cluster_assignments': cluster_df,
            'cluster_statistics': cluster_stats
        }

    def get_cluster_provenances(
            self, cluster_df: pd.DataFrame) -> pd.DataFrame:
        """Get provenance information for each cluster."""
        print("\nAnalyzing cluster provenance distributions...")

        # Get provenance from database
        khipu_ids = cluster_df['khipu_id'].tolist()
        placeholders = ','.join(['?'] * len(khipu_ids))
        query = f"""
        SELECT KHIPU_ID, PROVENANCE, REGION
        FROM khipu_main
        WHERE KHIPU_ID IN ({placeholders})
        """

        prov_df = pd.read_sql_query(query, self.conn, params=khipu_ids)

        # Merge with cluster assignments
        result_df = cluster_df.merge(
            prov_df,
            left_on='khipu_id',
            right_on='KHIPU_ID',
            how='left')

        # Analyze provenance distribution per cluster
        for cluster_id in sorted(result_df['cluster'].unique()):
            if cluster_id == -1:
                continue

            cluster_data = result_df[result_df['cluster'] == cluster_id]
            prov_counts = cluster_data['PROVENANCE'].value_counts().head(5)

            if len(prov_counts) > 0:
                print(f"\nCluster {cluster_id} - Top provenances:")
                for prov, count in prov_counts.items():
                    pct = count / len(cluster_data) * 100
                    print(f"  {prov}: {count} ({pct:.1f}%)")

        return result_df

    def perform_pca(self, X: np.ndarray, n_components: int = 2) -> tuple:
        """Perform PCA for visualization."""
        print(f"\nPerforming PCA (n_components={n_components})...")

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        explained_var = pca.explained_variance_ratio_
        print(f"✓ Explained variance: {explained_var}")
        print(f"✓ Total: {sum(explained_var):.2%}")

        return X_pca, pca

    def export_results(
            self,
            clustering_results: dict,
            analysis: dict,
            output_dir: Path):
        """Export clustering results."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export cluster assignments
        assignments_csv = Path(
            output_dir) / f"cluster_assignments_{clustering_results['method']}.csv"
        analysis['cluster_assignments'].to_csv(assignments_csv, index=False)
        print(f"\n✓ Exported cluster assignments to {assignments_csv}")

        # Export cluster statistics
        stats_json = Path(
            output_dir) / f"cluster_statistics_{clustering_results['method']}.json"
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'method': clustering_results['method'],
            'n_clusters': len(analysis['cluster_statistics']),
            'cluster_statistics': analysis['cluster_statistics']
        }

        with open(stats_json, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"✓ Exported cluster statistics to {stats_json}")

    def run_analysis(self, output_dir: Path, pca_csv_path: Path):
        """Run complete clustering analysis."""
        print("=" * 80)
        print("KHIPU CLUSTERING ANALYSIS")
        print("=" * 80)

        # Prepare features
        X, khipu_ids, valid_df, scaler = self.prepare_features()
        print(f"\nPrepared features for {len(khipu_ids)} valid khipus")

        # Perform K-means clustering
        kmeans_results = self.perform_kmeans(X, khipu_ids)
        kmeans_analysis = self.analyze_clusters(kmeans_results, valid_df)
        self.get_cluster_provenances(kmeans_analysis['cluster_assignments'])

        # Export K-means results
        self.export_results(kmeans_results, kmeans_analysis, output_dir)

        # Perform hierarchical clustering
        hier_results = self.perform_hierarchical(
            X, khipu_ids, n_clusters=kmeans_results['best_k'])
        hier_analysis = self.analyze_clusters(hier_results, valid_df)
        self.export_results(hier_results, hier_analysis, output_dir)

        # Perform PCA for visualization
        X_pca, pca = self.perform_pca(X, n_components=2)

        # Export PCA coordinates
        pca_df = pd.DataFrame({
            'khipu_id': khipu_ids,
            'pc1': X_pca[:, 0],
            'pc2': X_pca[:, 1],
            'cluster_kmeans': kmeans_results['labels'],
            'cluster_hierarchical': hier_results['labels']
        })
        pca_df.to_csv(pca_csv_path, index=False)
        print(f"\n✓ Exported PCA coordinates to {pca_csv_path}")

        print("\n" + "=" * 80)
        print("CLUSTERING ANALYSIS COMPLETE")
        print("=" * 80)

        return kmeans_results, hier_results, pca_df

    def __del__(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()


if __name__ == "__main__":
    # Get configuration
    config = get_config()

    # Validate setup
    validation = config.validate_setup()
    if not validation['valid']:
        print("\nConfiguration errors:")
        for error in validation['errors']:
            print(f"  • {error}")
        sys.exit(1)

    # Set up paths
    features_path = config.get_processed_file(
        'graph_structural_features.csv', phase=4)
    db_path = config.get_database_path()
    output_dir = config.processed_dir / 'phase4'
    pca_csv_path = output_dir / 'cluster_pca_coordinates.csv'

    print(f"Database: {db_path}")
    print(f"Features: {features_path}")
    print(f"Output: {output_dir}")
    print()

    clusterer = KhipuClusterer(features_path, db_path)
    clusterer.run_analysis(output_dir, pca_csv_path)
