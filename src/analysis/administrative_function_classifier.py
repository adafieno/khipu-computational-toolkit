"""
Phase 8: Administrative Function Classifier - Core Module

This module implements administrative function classification using:
1. Structural features (color-agnostic baseline)
2. Chromatic encoding features (administrative affordances)
3. Integrated multi-modal classification

Key Principles:
- No semantic decoding (operational features only)
- Function before interpretation (how used, not what said)
- Expert-in-the-loop validation (probabilistic assignments)
"""

import sys
from pathlib import Path

# Add src directory to path for config import
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
from typing import Dict, List  # noqa: E402
import json  # noqa: E402
from datetime import datetime  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import silhouette_score, calinski_harabasz_score  # noqa: E402
from sklearn.model_selection import cross_val_score, StratifiedKFold  # noqa: E402
from scipy.stats import entropy  # noqa: E402


class AdministrativeFunctionClassifier:
    """
    Classify khipus by administrative function using structural, chromatic,
    and numeric affordances.

    Implements three-stage analysis:
    1. Structural typology (color-agnostic baseline)
    2. Chromatic encoding analysis
    3. Integrated classification
    """

    def __init__(self):
        """Initialize classifier with data paths from config."""
        config = get_config()
        self.data_dir = config.processed_dir
        self.output_dir = config.processed_dir / "phase8"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store results
        self.structural_features = None
        self.chromatic_features = None
        self.numeric_features = None
        self.integrated_features = None

        self.structural_clusters = None
        self.final_typology = None

        print("=" * 80)
        print("PHASE 8: Administrative Function & Encoding Strategies")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print()

    def load_base_data(self) -> pd.DataFrame:
        """Load and merge all relevant datasets from previous phases."""
        print("Loading base datasets...")

        # Load structural features (Phase 4)
        structural = pd.read_csv(
            self.data_dir / "phase4" /
            "graph_structural_features.csv")
        print(f"  ✓ Structural features: {len(structural)} khipus")

        # Load summation results (Phase 3)
        summation = pd.read_csv(self.data_dir / "phase3" / "summation_test_results.csv")
        print(f"  ✓ Summation results: {len(summation)} khipus")

        # Load color data (Phase 2)
        color_data = pd.read_csv(self.data_dir / "phase2" / "color_data.csv")
        print(f"  ✓ Color data: {len(color_data)} records")

        # Load numeric values (Phase 1)
        numeric_data = pd.read_csv(self.data_dir / "phase1" / "cord_numeric_values.csv")
        print(f"  ✓ Numeric values: {len(numeric_data)} cords")

        # Load existing clusters (Phase 4)
        clusters = pd.read_csv(
            self.data_dir / "phase4" /
            "cluster_assignments_kmeans.csv")
        print(f"  ✓ Existing clusters: {len(clusters)} khipus")

        # Merge datasets
        base_df = structural.merge(
            summation[['khipu_id', 'pendant_match_rate', 'has_pendant_summation', 'has_white_boundaries']],
            on='khipu_id', how='left'
        ).merge(
            clusters[['khipu_id', 'cluster']],
            on='khipu_id', how='left'
        )

        # Add color diversity
        color_diversity = color_data.groupby('khipu_id').agg({
            'color_cd_1': 'nunique',
            'full_color': 'count'
        }).reset_index()
        color_diversity.columns = [
            'khipu_id',
            'unique_colors',
            'total_color_records']

        base_df = base_df.merge(color_diversity, on='khipu_id', how='left')

        # Fill NaN values
        base_df['pendant_match_rate'] = base_df['pendant_match_rate'].fillna(0)
        base_df['has_pendant_summation'] = base_df['has_pendant_summation'].fillna(
            0)
        base_df['unique_colors'] = base_df['unique_colors'].fillna(0)
        base_df['total_color_records'] = base_df['total_color_records'].fillna(
            0)

        print(
            f"\n✓ Merged dataset: {len(base_df)} khipus with {len(base_df.columns)} features")
        return base_df

    def extract_structural_features(
            self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        8.1: Extract structural features (color-agnostic baseline).

        Features:
        - Hierarchy depth
        - Branching factor
        - Cord count
        - Summation reliability
        - Numeric density and coverage
        - Presence of aggregation layers
        - Structural entropy
        """
        print("\n" + "=" * 80)
        print("8.1: STRUCTURAL TYPOLOGY (Color-Agnostic Baseline)")
        print("=" * 80)

        structural_features = pd.DataFrame()
        structural_features['khipu_id'] = base_df['khipu_id']

        # Hierarchy metrics
        structural_features['hierarchy_depth'] = base_df['depth']
        structural_features['branching_factor'] = base_df['avg_branching']
        structural_features['cord_count'] = base_df['num_nodes']

        # Summation reliability
        structural_features['summation_match_rate'] = base_df['pendant_match_rate']
        structural_features['has_summation'] = (
            base_df['pendant_match_rate'] > 0.5).astype(int)

        # Numeric behavior
        structural_features['numeric_coverage'] = base_df['has_numeric'].fillna(
            0)
        structural_features['avg_numeric_value'] = base_df['avg_numeric_value'].fillna(
            0)

        # Structural complexity
        structural_features['node_density'] = base_df['density']
        structural_features['leaf_ratio'] = base_df['num_leaves'] / \
            (base_df['num_nodes'] + 1)

        # Aggregation layers (depth > 2 indicates multi-level aggregation)
        structural_features['has_aggregation'] = (
            base_df['depth'] >= 3).astype(int)

        # Structural entropy (diversity of branching)
        # Use coefficient of variation of branching as proxy
        structural_features['structural_complexity'] = base_df['max_degree'] / \
            (base_df['avg_branching'] + 1)

        print(
            f"\nExtracted {len(structural_features.columns)-1} structural features:")
        for col in structural_features.columns:
            if col != 'khipu_id':
                print(f"  - {col}")

        self.structural_features = structural_features
        return structural_features

    def extract_chromatic_features(self, base_df: pd.DataFrame,
                                   color_data: pd.DataFrame) -> pd.DataFrame:
        """
        8.2: Extract chromatic encoding features (administrative affordances).

        Features:
        - Color entropy per khipu
        - Color count vs cord count ratio
        - Color position matrix (primary, pendant, subsidiary)
        - Color transition frequency
        - Boundary alignment
        """
        print("\n" + "=" * 80)
        print("8.2: CHROMATIC ENCODING AS ADMINISTRATIVE AFFORDANCE")
        print("=" * 80)

        # Load full color and hierarchy data
        color_data = pd.read_csv(self.data_dir / "phase2" / "color_data.csv")
        hierarchy_data = pd.read_csv(self.data_dir / "phase2" / "cord_hierarchy.csv")

        chromatic_features = pd.DataFrame()
        chromatic_features['khipu_id'] = base_df['khipu_id']

        # Process each khipu
        for idx, khipu_id in enumerate(base_df['khipu_id']):
            khipu_colors = color_data[color_data['khipu_id'] == khipu_id]
            khipu_hierarchy = hierarchy_data[hierarchy_data['KHIPU_ID'] == khipu_id]

            if len(khipu_colors) == 0:
                continue

            # Color entropy
            color_counts = khipu_colors['color_cd_1'].value_counts()
            color_probs = color_counts / color_counts.sum()
            chromatic_features.loc[idx, 'color_entropy'] = entropy(color_probs)

            # Color/cord ratio
            cord_count = len(khipu_hierarchy)
            chromatic_features.loc[idx, 'color_cord_ratio'] = len(
                khipu_colors) / max(cord_count, 1)

            # Color diversity
            chromatic_features.loc[idx,
                                   'unique_color_count'] = khipu_colors['color_cd_1'].nunique()

            # Multi-color complexity (cords with multiple colors)
            multi_color_count = (khipu_colors['color_cd_2'].notna()).sum()
            chromatic_features.loc[idx, 'multi_color_ratio'] = multi_color_count / max(
                len(khipu_colors), 1)

            # Color position analysis (if hierarchy available)
            if len(khipu_hierarchy) > 0:
                merged = khipu_hierarchy.merge(
                    khipu_colors[['cord_id', 'color_cd_1']],
                    left_on='CORD_ID',
                    right_on='cord_id',
                    how='left'
                )

                # Primary cord colors
                primary_colors = merged[merged['CORD_LEVEL']
                                        == 1]['color_cd_1'].nunique()
                chromatic_features.loc[idx,
                                       'primary_color_diversity'] = primary_colors

                # Pendant colors (level 2)
                pendant_colors = merged[merged['CORD_LEVEL']
                                        == 2]['color_cd_1'].nunique()
                chromatic_features.loc[idx,
                                       'pendant_color_diversity'] = pendant_colors

                # Color transitions (changes between parent-child)
                chromatic_features.loc[idx, 'color_transitions'] = self._count_color_transitions(
                    merged)

        # Fill NaN values
        chromatic_features = chromatic_features.fillna(0)

        print(
            f"\nExtracted {len(chromatic_features.columns)-1} chromatic features:")
        for col in chromatic_features.columns:
            if col != 'khipu_id':
                print(f"  - {col}")

        self.chromatic_features = chromatic_features
        return chromatic_features

    def _count_color_transitions(
            self, hierarchy_with_colors: pd.DataFrame) -> int:
        """Count color changes between parent-child relationships."""
        transitions = 0
        for _, row in hierarchy_with_colors.iterrows():
            if pd.notna(row['ATTACHED_TO']):
                parent = hierarchy_with_colors[
                    hierarchy_with_colors['CORD_ID'] == row['ATTACHED_TO']
                ]
                if len(parent) > 0 and pd.notna(
                        row['color_cd_1']) and pd.notna(
                        parent.iloc[0]['color_cd_1']):
                    if row['color_cd_1'] != parent.iloc[0]['color_cd_1']:
                        transitions += 1
        return transitions

    def perform_structural_clustering(self, structural_features: pd.DataFrame,
                                      n_clusters: int = 7) -> Dict:
        """
        Perform unsupervised clustering on structural features only.

        Returns cluster assignments and statistics.
        """
        print("\n" + "-" * 80)
        print("Performing structural clustering...")
        print("-" * 80)

        # Prepare features (exclude khipu_id)
        feature_cols = [
            col for col in structural_features.columns if col != 'khipu_id']
        X = structural_features[feature_cols].values

        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Dimensionality reduction
        pca = PCA(n_components=min(10, len(feature_cols)))
        X_pca = pca.fit_transform(X_scaled)

        print(
            f"PCA: {pca.n_components_} components explain {pca.explained_variance_ratio_.sum():.1%} variance")

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(X_pca)

        # Compute cluster quality metrics
        silhouette = silhouette_score(X_pca, cluster_labels)
        calinski = calinski_harabasz_score(X_pca, cluster_labels)

        print("\nCluster Quality:")
        print(f"  Silhouette Score: {silhouette:.3f}")
        print(f"  Calinski-Harabasz: {calinski:.1f}")

        # Analyze clusters
        cluster_stats = self._analyze_clusters(
            structural_features, cluster_labels, feature_cols)

        # Store results
        self.structural_clusters = {
            'labels': cluster_labels,
            'centroids': kmeans.cluster_centers_,
            'pca_components': X_pca,
            'scaler': scaler,
            'pca': pca,
            'stats': cluster_stats,
            'silhouette': silhouette,
            'calinski': calinski
        }

        return self.structural_clusters

    def _analyze_clusters(self, features_df: pd.DataFrame,
                          labels: np.ndarray,
                          feature_cols: List[str]) -> pd.DataFrame:
        """Compute statistics for each cluster."""
        features_df = features_df.copy()
        features_df['cluster'] = labels

        stats = []
        for cluster_id in range(labels.max() + 1):
            cluster_data = features_df[features_df['cluster'] == cluster_id]

            stat = {
                'cluster_id': cluster_id,
                'count': len(cluster_data),
                'pct': len(cluster_data) / len(features_df) * 100
            }

            # Add mean values for each feature
            for col in feature_cols:
                stat[f'mean_{col}'] = cluster_data[col].mean()

            stats.append(stat)

        stats_df = pd.DataFrame(stats)

        print("\nCluster Distribution:")
        for _, row in stats_df.iterrows():
            print(
                f"  Cluster {int(row['cluster_id'])}: {int(row['count'])} khipus ({row['pct']:.1f}%)")

        return stats_df

    def build_integrated_classifier(self,
                                    structural_features: pd.DataFrame,
                                    chromatic_features: pd.DataFrame,
                                    base_df: pd.DataFrame) -> Dict:
        """
        8.3: Build integrated classifier using structure + color + numeric features.

        Compares three feature sets:
        1. Structure only
        2. Structure + numeric
        3. Structure + numeric + color
        """
        print("\n" + "=" * 80)
        print("8.3: INTEGRATED ADMINISTRATIVE FUNCTION CLASSIFIER")
        print("=" * 80)

        # Merge all features
        integrated = structural_features.merge(
            chromatic_features,
            on='khipu_id',
            how='left'
        )

        # Create target variable from existing function classification
        # Use Phase 5 function predictions as ground truth
        try:
            function_data = pd.read_csv(
                self.data_dir / "phase8" / "khipu_function_classification.csv")
            integrated = integrated.merge(
                function_data[['khipu_id', 'predicted_function']],
                on='khipu_id',
                how='left'
            )
        except FileNotFoundError:
            print(
                "  Warning: No ground truth labels found, using cluster-based heuristic")
            # Use cluster + summation rate as heuristic
            integrated['predicted_function'] = (
                (integrated['summation_match_rate'] > 0.5) &
                (integrated['unique_color_count'] <= 5)
            ).map({True: 'Accounting', False: 'Narrative'})

        # Ensure predicted_function exists and has valid values
        if 'predicted_function' not in integrated.columns or integrated['predicted_function'].isna(
        ).all():
            print("  Warning: Creating synthetic labels from cluster assignments")
            integrated['predicted_function'] = integrated['cluster'].astype(
                str)

        # Fill missing values
        integrated = integrated.fillna(0)

        # Define feature sets
        structural_cols = [
            col for col in structural_features.columns if col != 'khipu_id']
        chromatic_cols = [
            col for col in chromatic_features.columns if col != 'khipu_id']

        feature_sets = {
            'structure_only': structural_cols,
            'structure_numeric': structural_cols,  # numeric already in structural
            'structure_numeric_color': structural_cols + chromatic_cols
        }

        results = {}

        for set_name, feature_cols in feature_sets.items():
            print(f"\n{'-' * 80}")
            print(f"Feature Set: {set_name} ({len(feature_cols)} features)")
            print(f"{'-' * 80}")

            X = integrated[feature_cols].values
            y = integrated['predicted_function'].astype(str).values

            # Train Random Forest
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                class_weight='balanced'
            )

            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')

            # Train on full data
            rf.fit(X, y)

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)

            print(
                f"\nCross-Validation Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            print("\nTop 10 Most Important Features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']:30s} {row['importance']:.4f}")

            results[set_name] = {
                'model': rf,
                'cv_scores': cv_scores,
                'feature_importance': feature_importance,
                'features': feature_cols
            }

        self.integrated_features = integrated
        return results

    def generate_final_typology(self,
                                structural_clusters: Dict,
                                classifier_results: Dict) -> pd.DataFrame:
        """
        Generate final administrative typology combining clustering and classification.
        """
        print("\n" + "=" * 80)
        print("GENERATING FINAL ADMINISTRATIVE TYPOLOGY")
        print("=" * 80)

        typology = pd.DataFrame()
        typology['khipu_id'] = self.integrated_features['khipu_id']
        typology['structural_cluster'] = structural_clusters['labels']
        typology['predicted_function'] = self.integrated_features['predicted_function'].astype(
            str)

        # Get predictions from best model
        best_model = classifier_results['structure_numeric_color']['model']
        best_features = classifier_results['structure_numeric_color']['features']

        X = self.integrated_features[best_features].values
        typology['confidence'] = best_model.predict_proba(X).max(axis=1)

        # Assign administrative type based on cluster + function
        typology['administrative_type'] = typology.apply(
            self._assign_administrative_type, axis=1
        )

        # Add key characteristics
        typology = typology.merge(
            self.structural_features[[
                'khipu_id', 'cord_count', 'hierarchy_depth',
                'summation_match_rate', 'numeric_coverage'
            ]],
            on='khipu_id'
        )

        typology = typology.merge(
            self.chromatic_features[[
                'khipu_id', 'unique_color_count', 'color_entropy'
            ]],
            on='khipu_id'
        )

        # Summary statistics
        print("\nAdministrative Type Distribution:")
        type_counts = typology['administrative_type'].value_counts()
        for admin_type, count in type_counts.items():
            pct = count / len(typology) * 100
            print(f"  {admin_type:30s}: {count:4d} ({pct:5.1f}%)")

        print(f"\nAverage Confidence: {typology['confidence'].mean():.3f}")
        print(
            f"High Confidence (>0.8): {(typology['confidence'] > 0.8).sum()} khipus")

        self.final_typology = typology
        return typology

    def _assign_administrative_type(self, row: pd.Series) -> str:
        """
        Assign administrative type based on cluster and function.

        Types (following Phase 4 archetypes + function):
        - Local Operational Record (small, accounting, high summation)
        - Aggregated Summary (large, accounting, hierarchical)
        - Inspection/Audit Record (medium, accounting, low color)
        - Multi-Category Record (accounting, high color diversity)
        - Narrative/Ceremonial (low summation, high color)
        - Exceptional/Anomalous (outliers)
        """
        cluster = row['structural_cluster']
        function = row['predicted_function']

        # Map clusters to types (based on Phase 4 analysis)
        if cluster == 3:  # Minimal cluster
            return "Local Operational Record"
        elif cluster == 1:  # Large Hierarchical
            return "Aggregated Summary"
        elif cluster == 2:  # Medium Standard
            if function == 'Accounting':
                return "Standard Administrative Record"
            else:
                return "Multi-Category Record"
        elif cluster == 0:  # Small Dense
            return "Compact Operational Record"
        elif cluster == 4:  # Wide Shallow
            return "Lateral Category Tracking"
        elif cluster == 5:  # Deep Complex
            return "Multi-Level Aggregation"
        elif cluster == 6:  # Exceptional
            return "Exceptional/Anomalous"
        else:
            return "Unclassified"

    def save_results(self, structural_clusters: Dict,
                     classifier_results: Dict,
                     final_typology: pd.DataFrame):
        """Save all Phase 8 results to disk."""
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        # Save structural features
        self.structural_features.to_csv(
            self.output_dir / "structural_features.csv", index=False
        )
        print("  ✓ Structural features")

        # Save chromatic features
        self.chromatic_features.to_csv(
            self.output_dir / "chromatic_features.csv", index=False
        )
        print("  ✓ Chromatic features")

        # Save cluster assignments
        cluster_assignments = pd.DataFrame({
            'khipu_id': self.structural_features['khipu_id'],
            'structural_cluster': structural_clusters['labels'],
            'silhouette_score': structural_clusters['silhouette']
        })
        cluster_assignments.to_csv(
            self.output_dir / "structural_cluster_assignments.csv", index=False
        )
        print("  ✓ Cluster assignments")

        # Save cluster statistics
        structural_clusters['stats'].to_csv(
            self.output_dir / "structural_cluster_statistics.csv", index=False
        )
        print("  ✓ Cluster statistics")

        # Save final typology
        final_typology.to_csv(
            self.output_dir / "administrative_typology.csv", index=False
        )
        print("  ✓ Administrative typology")

        # Save feature importance for all models
        for set_name, results in classifier_results.items():
            results['feature_importance'].to_csv(
                self.output_dir / f"feature_importance_{set_name}.csv", index=False)
        print("  ✓ Feature importance (3 models)")

        # Save metadata
        metadata = {
            'generated': datetime.now().isoformat(),
            'phase': 'Phase 8: Administrative Function & Encoding Strategies',
            'total_khipus': len(final_typology),
            'n_clusters': len(structural_clusters['stats']),
            'cluster_quality': {
                'silhouette_score': float(structural_clusters['silhouette']),
                'calinski_harabasz': float(structural_clusters['calinski'])
            },
            'model_performance': {
                set_name: {
                    'cv_mean': float(results['cv_scores'].mean()),
                    'cv_std': float(results['cv_scores'].std()),
                    'n_features': len(results['features'])
                }
                for set_name, results in classifier_results.items()
            },
            'administrative_types': final_typology['administrative_type'].value_counts().to_dict()
        }

        with open(self.output_dir / "phase8_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print("  ✓ Metadata")

        print(f"\n✓ All results saved to: {self.output_dir}")


def run_phase8_analysis():
    """Run complete Phase 8 analysis pipeline."""
    # Initialize classifier
    classifier = AdministrativeFunctionClassifier()

    # Load data
    base_df = classifier.load_base_data()

    # 8.1: Extract structural features
    structural_features = classifier.extract_structural_features(base_df)

    # 8.2: Extract chromatic features
    config = get_config()
    color_data = pd.read_csv(config.get_processed_file("color_data.csv", phase=2))
    chromatic_features = classifier.extract_chromatic_features(
        base_df, color_data)

    # Perform structural clustering
    structural_clusters = classifier.perform_structural_clustering(
        structural_features)

    # 8.3: Build integrated classifier
    classifier_results = classifier.build_integrated_classifier(
        structural_features, chromatic_features, base_df
    )

    # Generate final typology
    final_typology = classifier.generate_final_typology(
        structural_clusters, classifier_results
    )

    # Save all results
    classifier.save_results(
        structural_clusters,
        classifier_results,
        final_typology)

    return classifier, final_typology


if __name__ == "__main__":
    classifier, typology = run_phase8_analysis()
    print("\n" + "=" * 80)
    print("PHASE 8 COMPLETE")
    print("=" * 80)
