"""
Phase 9.9: Stability & Stress Testing

Tests robustness of discovered patterns under:
- Feature ablation (remove features one at a time)
- Data masking (randomly mask 10-30% of data)
- Re-clustering stability (multiple runs with different parameters)
- Cross-validation (train/test splits)

Integrates with:
- Phase 4: Clustering results
- Phase 8: ML classification
- Phase 9: All previous robustness analyses
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class StabilityTester:
    """Tests stability of analyses under data perturbation."""

    def __init__(self, data_dir: Path = Path("data/processed")):
        """Initialize with data directory."""
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "phase9" / "9.9_stability"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("PHASE 9.9: STABILITY & STRESS TESTING")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}\n")

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load required datasets."""
        print("Loading datasets...")

        data = {}

        data['structural'] = pd.read_csv(
            self.data_dir / "graph_structural_features.csv"
        )
        print(f"  ✓ Structural features: {len(data['structural'])} khipus")

        data['numeric'] = pd.read_csv(
            self.data_dir / "cord_numeric_values.csv"
        )
        print(f"  ✓ Numeric values: {len(data['numeric'])} records")

        data['color'] = pd.read_csv(self.data_dir / "color_data.csv")
        print(f"  ✓ Color data: {len(data['color'])} records")

        data['hierarchy'] = pd.read_csv(self.data_dir / "cord_hierarchy.csv")
        print(f"  ✓ Hierarchy: {len(data['hierarchy'])} cords")

        # Load Phase 4 clustering results
        cluster_file = self.data_dir / "cluster_assignments_kmeans.csv"
        if cluster_file.exists():
            data['clusters'] = pd.read_csv(cluster_file)
            print(f"  ✓ Cluster assignments: {len(data['clusters'])} khipus")
        else:
            data['clusters'] = pd.DataFrame()
            print("  ⚠ Cluster assignments not found")

        # Load Phase 8 classifications
        class_file = self.data_dir / "khipu_function_classification.csv"
        if class_file.exists():
            data['classifications'] = pd.read_csv(class_file)
            print(f"  ✓ Classifications: {len(data['classifications'])} khipus")
        else:
            data['classifications'] = pd.DataFrame()
            print("  ⚠ Classifications not found")

        print()
        return data

    def test_feature_ablation(
        self, structural: pd.DataFrame, original_clusters: pd.DataFrame
    ) -> pd.DataFrame:
        """Test clustering stability when features are removed."""
        print("Testing feature ablation...")

        features = ['num_nodes', 'depth', 'avg_branching', 'width', 'density']
        results = []

        # Merge structural data with cluster assignments
        merged = structural.merge(
            original_clusters[['khipu_id', 'cluster']],
            on='khipu_id',
            how='inner'
        )

        # Get original cluster assignments
        original_labels = merged['cluster'].values

        # Test removing each feature
        for ablate_feature in features:
            remaining_features = [f for f in features if f in merged.columns and f != ablate_feature]

            if not remaining_features:
                continue

            # Cluster with remaining features
            X = merged[remaining_features].fillna(0).values
            X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

            n_clusters = merged['cluster'].nunique()
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            ablated_labels = kmeans.fit_predict(X_normalized)

            # Measure agreement
            ari = adjusted_rand_score(original_labels, ablated_labels)
            nmi = normalized_mutual_info_score(original_labels, ablated_labels)

            results.append({
                'ablated_feature': ablate_feature,
                'remaining_features': len(remaining_features),
                'adjusted_rand_index': ari,
                'normalized_mutual_info': nmi,
                'stability': ari  # ARI as primary stability metric
            })

            print(f"  - Ablated {ablate_feature}: ARI={ari:.3f}, NMI={nmi:.3f}")

        results_df = pd.DataFrame(results)

        # Summary statistics
        mean_stability = results_df['stability'].mean()
        min_stability = results_df['stability'].min()
        critical_feature = results_df.loc[results_df['stability'].idxmin(), 'ablated_feature']

        print("\nAblation Summary:")
        print(f"  Mean stability: {mean_stability:.3f}")
        print(f"  Minimum stability: {min_stability:.3f}")
        print(f"  Most critical feature: {critical_feature}")

        return results_df

    def test_data_masking(
        self, structural: pd.DataFrame, numeric: pd.DataFrame,
        color: pd.DataFrame, original_clusters: pd.DataFrame
    ) -> pd.DataFrame:
        """Test clustering stability when data is randomly masked."""
        print("\nTesting data masking...")

        masking_levels = [0.10, 0.20, 0.30]
        results = []

        # Merge structural data with cluster assignments
        merged = structural.merge(
            original_clusters[['khipu_id', 'cluster']],
            on='khipu_id',
            how='inner'
        )

        original_labels = merged['cluster'].values
        features = ['num_nodes', 'depth', 'avg_branching', 'width', 'density']

        for mask_pct in masking_levels:
            # Mask random subset of khipus
            n_mask = int(len(merged) * mask_pct)
            mask_indices = np.random.choice(len(merged), n_mask, replace=False)

            # Create masked structural features (set to mean)
            masked_structural = merged.copy()
            for feat in features:
                if feat in masked_structural.columns:
                    feat_mean = masked_structural[feat].mean()
                    masked_structural.loc[merged.index[mask_indices], feat] = feat_mean

            # Re-cluster
            X = masked_structural[features].fillna(0).values
            X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

            n_clusters = merged['cluster'].nunique()
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            masked_labels = kmeans.fit_predict(X_normalized)

            # Measure drift
            ari = adjusted_rand_score(original_labels, masked_labels)
            nmi = normalized_mutual_info_score(original_labels, masked_labels)

            # Calculate drift (percent changed)
            drift = 1 - ari

            results.append({
                'masking_level': mask_pct,
                'n_masked': n_mask,
                'adjusted_rand_index': ari,
                'normalized_mutual_info': nmi,
                'drift': drift
            })

            print(f"  - Masked {mask_pct*100:.0f}%: ARI={ari:.3f}, Drift={drift:.3f}")

        results_df = pd.DataFrame(results)

        return results_df

    def test_reclustering_stability(
        self, structural: pd.DataFrame, original_clusters: pd.DataFrame
    ) -> Dict:
        """Test clustering consensus across multiple runs and algorithms."""
        print("\nTesting re-clustering stability...")

        # Merge structural data with cluster assignments
        merged = structural.merge(
            original_clusters[['khipu_id', 'cluster']],
            on='khipu_id',
            how='inner'
        )

        features = ['num_nodes', 'depth', 'avg_branching', 'width', 'density']
        X = merged[[f for f in features if f in merged.columns]].fillna(0).values
        X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

        n_clusters = merged['cluster'].nunique()
        n_runs = 50  # Reduced from 100 for performance

        # Test different random seeds
        print("  Testing different random seeds...")
        seed_results = []
        for seed in range(n_runs):
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            labels = kmeans.fit_predict(X_normalized)
            seed_results.append(labels)

        # Calculate consensus (pairwise NMI)
        nmi_scores = []
        for i in range(len(seed_results)):
            for j in range(i + 1, len(seed_results)):
                nmi = normalized_mutual_info_score(seed_results[i], seed_results[j])
                nmi_scores.append(nmi)

        mean_nmi = np.mean(nmi_scores)
        print(f"    Mean pairwise NMI: {mean_nmi:.3f}")

        # Test different k values
        print("  Testing different k values...")
        k_values = [n_clusters - 1, n_clusters, n_clusters + 1]
        k_results = {}

        for k in k_values:
            if k < 2:
                continue
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_normalized)

            ari = adjusted_rand_score(merged['cluster'].values, labels)
            k_results[k] = {
                'adjusted_rand_index': ari,
                'n_clusters': k
            }
            print(f"    k={k}: ARI={ari:.3f}")

        # Test different algorithms
        print("  Testing different algorithms...")
        algo_results = {}

        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_normalized)

        # Hierarchical
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        hierarchical_labels = hierarchical.fit_predict(X_normalized)

        ari_kmeans_hierarchical = adjusted_rand_score(kmeans_labels, hierarchical_labels)
        algo_results['kmeans_vs_hierarchical'] = ari_kmeans_hierarchical

        print(f"    K-means vs Hierarchical: ARI={ari_kmeans_hierarchical:.3f}")

        return {
            'seed_consensus_nmi': mean_nmi,
            'k_value_tests': k_results,
            'algorithm_agreement': algo_results
        }

    def test_cross_validation(
        self, structural: pd.DataFrame, classifications: pd.DataFrame
    ) -> Dict:
        """Test classification stability with train/test splits."""
        print("\nTesting cross-validation stability...")

        if classifications.empty or 'predicted_function' not in classifications.columns:
            print("  ⚠ No classification data available")
            return {}

        # Merge structural features with classifications
        merged = structural.merge(
            classifications[['khipu_id', 'predicted_function']],
            on='khipu_id',
            how='inner'
        )

        if len(merged) < 50:
            print("  ⚠ Insufficient data for cross-validation")
            return {}

        features = ['num_nodes', 'depth', 'avg_branching', 'width', 'density']
        X = merged[[f for f in features if f in merged.columns]].fillna(0).values
        y = merged['predicted_function'].values

        n_runs = 20
        scores = []

        print(f"  Running {n_runs} train/test splits...")

        for i in range(n_runs):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=i
            )

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scores.append(score)

        mean_score = np.mean(scores)
        std_score = np.std(scores)

        print(f"    Mean accuracy: {mean_score:.3f} ± {std_score:.3f}")

        return {
            'mean_accuracy': mean_score,
            'std_accuracy': std_score,
            'n_runs': n_runs,
            'scores': scores
        }

    def save_results(
        self, ablation: pd.DataFrame, masking: pd.DataFrame,
        stability: Dict, cross_val: Dict
    ) -> None:
        """Save all results to CSV and JSON."""
        print("\nSaving results...")

        ablation.to_csv(
            self.output_dir / "feature_ablation_results.csv",
            index=False
        )
        print("  ✓ Saved: feature_ablation_results.csv")

        masking.to_csv(
            self.output_dir / "data_masking_results.csv",
            index=False
        )
        print("  ✓ Saved: data_masking_results.csv")

        with open(self.output_dir / "clustering_stability.json", 'w') as f:
            json.dump(stability, f, indent=2, default=str)
        print("  ✓ Saved: clustering_stability.json")

        if cross_val:
            with open(self.output_dir / "cross_validation_results.json", 'w') as f:
                json.dump(cross_val, f, indent=2, default=str)
            print("  ✓ Saved: cross_validation_results.json")

        # Summary report
        summary = {
            "ablation": {
                "mean_stability": float(ablation['stability'].mean()),
                "min_stability": float(ablation['stability'].min()),
                "critical_feature": ablation.loc[
                    ablation['stability'].idxmin(), 'ablated_feature'
                ]
            },
            "masking": {
                "max_drift": float(masking['drift'].max()),
                "drift_at_10pct": float(masking[masking['masking_level'] == 0.10]['drift'].values[0])
                if len(masking[masking['masking_level'] == 0.10]) > 0 else None
            },
            "clustering_stability": {
                "seed_consensus": stability.get('seed_consensus_nmi', 0),
                "algorithm_agreement": stability.get('algorithm_agreement', {})
            },
            "cross_validation": cross_val
        }

        with open(self.output_dir / "stability_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print("  ✓ Saved: stability_summary.json")

    def run_analysis(self) -> None:
        """Execute complete stability testing pipeline."""
        print("=" * 80)
        print("STABILITY & STRESS TESTING PIPELINE")
        print("=" * 80)

        # Load data
        data = self.load_data()

        structural = data['structural']
        numeric = data['numeric']
        color = data['color']
        original_clusters = data['clusters']
        classifications = data['classifications']

        if original_clusters.empty:
            print("\n⚠ Cannot run stability tests without clustering results")
            print("Run Phase 4 clustering first!")
            return

        # Test feature ablation
        ablation_results = self.test_feature_ablation(structural, original_clusters)

        # Test data masking
        masking_results = self.test_data_masking(
            structural, numeric, color, original_clusters
        )

        # Test re-clustering stability
        stability_results = self.test_reclustering_stability(
            structural, original_clusters
        )

        # Test cross-validation
        cv_results = self.test_cross_validation(structural, classifications)

        # Save all results
        self.save_results(
            ablation_results, masking_results, stability_results, cv_results
        )

        print("\n" + "=" * 80)
        print("PHASE 9.9 COMPLETE")
        print("=" * 80)
        print("\nKey Findings:")
        print(f"  - Mean ablation stability: {ablation_results['stability'].mean():.3f}")
        print(f"  - Max masking drift: {masking_results['drift'].max():.3f}")
        print(f"  - Seed consensus (NMI): {stability_results['seed_consensus_nmi']:.3f}")

        # Interpretation
        if ablation_results['stability'].mean() > 0.8:
            print("\n✓ Patterns are STABLE - no single feature dominates")
        elif ablation_results['stability'].mean() > 0.6:
            print("\n⚠ Patterns are MODERATELY STABLE - some feature dependence")
        else:
            print("\n✗ Patterns are UNSTABLE - heavily dependent on specific features")


def run_stability_testing():
    """Main entry point for Phase 9.9."""
    tester = StabilityTester()
    tester.run_analysis()
    return tester


if __name__ == "__main__":
    tester = run_stability_testing()
