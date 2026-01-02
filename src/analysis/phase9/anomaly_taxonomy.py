"""
Phase 9.7: Anomaly Taxonomy (Pre-Interpretive)

Classifies anomalies by deviation type without causal explanation:
- Numeric deviations: Extreme values, impossible summations
- Structural deviations: Unusual depth, extreme branching
- Chromatic deviations: Rare colors, extreme entropy
- Hybrid deviations: Multiple deviation types
- Coherence testing and robustness correlation

Integrates with:
- Phase 7: Anomaly detection results
- Phase 9.2: Robustness scores
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, davies_bouldin_score


class AnomalyTaxonomyAnalyzer:
    """Analyzes and classifies anomalies by deviation type."""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "phase9" / "9.7_anomaly_taxonomy"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> None:
        """Load required datasets."""
        print("Loading datasets...")
        
        # Load Phase 7 anomaly detection results (khipu_id, anomaly_score, is_anomaly_*)
        anomaly_file = self.data_dir / "anomaly_detection_results.csv"
        self.anomalies = pd.read_csv(anomaly_file)
        print(f"  ✓ Anomaly detection: {len(self.anomalies)} khipus")
        
        # Load detailed anomaly data (with outlier flags)
        detailed_file = self.data_dir / "anomaly_detection_detailed.csv"
        self.anomaly_details = pd.read_csv(detailed_file)
        print(f"  ✓ Detailed anomalies: {len(self.anomaly_details)} khipus")
        
        # Load Phase 9.2 robustness metrics (khipu_id, robustness_score)
        robustness_file = self.data_dir / "phase9" / "9.2_robustness" / "robustness_metrics.csv"
        if robustness_file.exists():
            self.robustness = pd.read_csv(robustness_file)
            print(f"  ✓ Robustness metrics: {len(self.robustness)} khipus")
        else:
            self.robustness = pd.DataFrame()
            print("  ⚠ Robustness metrics not found")
        
        # Load structural features for context (khipu_id, num_nodes, depth, etc.)
        struct_file = self.data_dir / "graph_structural_features.csv"
        self.structural = pd.read_csv(struct_file)
        print(f"  ✓ Structural features: {len(self.structural)} khipus")
        
        # Load color data for chromatic analysis
        color_file = self.data_dir / "color_data.csv"
        self.colors = pd.read_csv(color_file)
        print(f"  ✓ Color data: {len(self.colors)} records")
        
    def classify_deviation_types(self) -> pd.DataFrame:
        """Classify anomalies by deviation type."""
        print("\nClassifying deviation types...")
        
        classifications = []
        
        for _, row in self.anomaly_details.iterrows():
            khipu_id = row['khipu_id']
            
            # Initialize deviation flags
            numeric_deviation = False
            structural_deviation = False
            chromatic_deviation = False
            
            # Numeric deviations
            if pd.notna(row.get('avg_numeric_value_outlier')):
                numeric_deviation = row['avg_numeric_value_outlier']
            
            # Structural deviations (outlier flags)
            structural_flags = [
                row.get('num_nodes_outlier', False),
                row.get('depth_outlier', False),
                row.get('avg_branching_outlier', False),
                row.get('density_outlier', False)
            ]
            structural_deviation = any(structural_flags)
            
            # Topology anomalies also count as structural
            if row.get('is_anomaly_topology', False):
                structural_deviation = True
            
            # Chromatic deviations (calculate from color data)
            khipu_colors = self.colors[self.colors['khipu_id'] == khipu_id]
            if len(khipu_colors) > 0:
                n_colors = khipu_colors['color_cd_1'].nunique()
                # Rare colors: > 95th percentile
                color_counts_all = self.colors['color_cd_1'].value_counts()
                rare_threshold = color_counts_all.quantile(0.05)
                has_rare_colors = any(
                    color_counts_all.get(c, 0) < rare_threshold 
                    for c in khipu_colors['color_cd_1'].unique()
                )
                # Extreme entropy: > 95th percentile
                if n_colors > 1:
                    color_counts = khipu_colors['color_cd_1'].value_counts()
                    probs = color_counts / color_counts.sum()
                    entropy = -sum(probs * np.log2(probs + 1e-10))
                else:
                    entropy = 0
                
                # Calculate entropy threshold from all khipus
                all_entropies = []
                for kid in self.colors['khipu_id'].unique():
                    kc = self.colors[self.colors['khipu_id'] == kid]
                    if len(kc) > 0:
                        nc = kc['color_cd_1'].nunique()
                        if nc > 1:
                            cc = kc['color_cd_1'].value_counts()
                            p = cc / cc.sum()
                            e = -sum(p * np.log2(p + 1e-10))
                            all_entropies.append(e)
                
                entropy_threshold = np.percentile(all_entropies, 95) if len(all_entropies) > 0 else 999
                extreme_entropy = entropy > entropy_threshold
                
                chromatic_deviation = has_rare_colors or extreme_entropy
            else:
                chromatic_deviation = False
                entropy = 0
                n_colors = 0
            
            # Determine deviation type
            deviation_count = sum([numeric_deviation, structural_deviation, chromatic_deviation])
            
            if deviation_count == 0:
                deviation_type = 'none'
            elif deviation_count == 1:
                if numeric_deviation:
                    deviation_type = 'numeric'
                elif structural_deviation:
                    deviation_type = 'structural'
                else:
                    deviation_type = 'chromatic'
            else:
                deviation_type = 'hybrid'
            
            classifications.append({
                'khipu_id': khipu_id,
                'deviation_type': deviation_type,
                'numeric_deviation': numeric_deviation,
                'structural_deviation': structural_deviation,
                'chromatic_deviation': chromatic_deviation,
                'deviation_count': deviation_count,
                'anomaly_score': row['anomaly_score'],
                'is_anomaly': row.get('high_confidence_anomaly', False),
                'n_colors': n_colors,
                'color_entropy': entropy
            })
        
        class_df = pd.DataFrame(classifications)
        
        # Statistics
        type_counts = class_df['deviation_type'].value_counts()
        
        print(f"  ✓ Classified {len(class_df)} khipus")
        print(f"  ✓ Deviation type distribution:")
        for dtype, count in type_counts.items():
            pct = 100 * count / len(class_df)
            print(f"    - {dtype}: {count} ({pct:.1f}%)")
        
        return class_df
    
    def cluster_anomalies(self, class_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Apply hierarchical clustering to anomalies."""
        print("\nClustering anomalies...")
        
        # Filter to deviations (exclude 'none' type)
        anomalies_only = class_df[class_df['deviation_type'] != 'none'].copy()
        
        if len(anomalies_only) < 3:
            print("  ⚠ Too few deviations for clustering")
            return pd.DataFrame(), {}
        
        # Create feature vector for clustering
        # Merge with structural data
        cluster_data = anomalies_only.merge(
            self.structural[['khipu_id', 'num_nodes', 'depth', 'avg_branching', 'density']],
            on='khipu_id'
        )
        
        # Feature vector: numeric score, structural features, chromatic entropy
        features = cluster_data[[
            'anomaly_score',
            'num_nodes',
            'depth',
            'avg_branching',
            'density',
            'color_entropy'
        ]].fillna(0)
        
        # Normalize features
        features_normalized = (features - features.mean()) / (features.std() + 1e-10)
        
        # Hierarchical clustering
        linkage_matrix = linkage(features_normalized, method='ward')
        
        # Determine optimal number of clusters using silhouette score
        best_k = 3
        best_silhouette = -1
        
        for k in range(3, min(8, len(anomalies_only) // 2)):
            labels = fcluster(linkage_matrix, k, criterion='maxclust')
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(features_normalized, labels)
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_k = k
        
        # Apply best clustering
        cluster_labels = fcluster(linkage_matrix, best_k, criterion='maxclust')
        cluster_data['anomaly_cluster'] = cluster_labels
        
        # Calculate cluster statistics
        db_score = davies_bouldin_score(features_normalized, cluster_labels)
        
        print(f"  ✓ Clustered {len(anomalies_only)} deviations")
        print(f"  ✓ Optimal clusters: {best_k}")
        print(f"  ✓ Silhouette score: {best_silhouette:.3f}")
        print(f"  ✓ Davies-Bouldin score: {db_score:.3f}")
        
        # Cluster characteristics
        cluster_stats = []
        for cluster_id in range(1, best_k + 1):
            cluster_members = cluster_data[cluster_data['anomaly_cluster'] == cluster_id]
            
            cluster_stats.append({
                'cluster_id': int(cluster_id),
                'size': int(len(cluster_members)),
                'mean_anomaly_score': float(cluster_members['anomaly_score'].mean()),
                'dominant_deviation': cluster_members['deviation_type'].mode()[0] if len(cluster_members) > 0 else 'unknown',
                'numeric_rate': float(cluster_members['numeric_deviation'].mean()),
                'structural_rate': float(cluster_members['structural_deviation'].mean()),
                'chromatic_rate': float(cluster_members['chromatic_deviation'].mean()),
                'mean_nodes': float(cluster_members['num_nodes'].mean()),
                'mean_depth': float(cluster_members['depth'].mean())
            })
        
        stats_df = pd.DataFrame(cluster_stats).sort_values('size', ascending=False)
        
        print(f"  ✓ Cluster characteristics:")
        for _, row in stats_df.iterrows():
            print(f"    - Cluster {row['cluster_id']}: {row['size']} anomalies, dominant: {row['dominant_deviation']}")
        
        clustering_summary = {
            'n_anomalies': int(len(anomalies_only)),
            'n_clusters': int(best_k),
            'silhouette_score': float(best_silhouette),
            'davies_bouldin_score': float(db_score),
            'linkage_matrix': linkage_matrix.tolist()
        }
        
        return cluster_data, clustering_summary
    
    def test_coherence(self, cluster_data: pd.DataFrame) -> Dict:
        """Test if anomalies form coherent subgroups."""
        print("\nTesting anomaly coherence...")
        
        if len(cluster_data) == 0:
            print("  ⚠ No cluster data available")
            return {}
        
        # Intra-cluster vs inter-cluster distance
        features = cluster_data[[
            'anomaly_score',
            'num_nodes',
            'depth',
            'avg_branching',
            'density',
            'color_entropy'
        ]].fillna(0)
        
        features_normalized = (features - features.mean()) / (features.std() + 1e-10)
        
        # Calculate intra-cluster distances
        intra_distances = []
        for cluster_id in cluster_data['anomaly_cluster'].unique():
            cluster_members = cluster_data[cluster_data['anomaly_cluster'] == cluster_id]
            if len(cluster_members) > 1:
                cluster_features = features_normalized.loc[cluster_members.index]
                distances = pdist(cluster_features, metric='euclidean')
                intra_distances.extend(distances)
        
        # Calculate inter-cluster distances (sample)
        inter_distances = []
        clusters = cluster_data['anomaly_cluster'].unique()
        for i, c1 in enumerate(clusters):
            for c2 in clusters[i+1:]:
                c1_members = cluster_data[cluster_data['anomaly_cluster'] == c1]
                c2_members = cluster_data[cluster_data['anomaly_cluster'] == c2]
                
                # Sample up to 10 pairs
                for _ in range(min(10, len(c1_members) * len(c2_members))):
                    if len(c1_members) > 0 and len(c2_members) > 0:
                        idx1 = np.random.choice(c1_members.index)
                        idx2 = np.random.choice(c2_members.index)
                        dist = np.linalg.norm(
                            features_normalized.loc[idx1].values - 
                            features_normalized.loc[idx2].values
                        )
                        inter_distances.append(dist)
        
        # Coherence ratio: inter / intra (higher = more coherent clusters)
        mean_intra = np.mean(intra_distances) if len(intra_distances) > 0 else 0
        mean_inter = np.mean(inter_distances) if len(inter_distances) > 0 else 0
        coherence_ratio = mean_inter / mean_intra if mean_intra > 0 else 0
        
        # Statistical test
        if len(intra_distances) > 0 and len(inter_distances) > 0:
            t_stat, p_val = stats.ttest_ind(inter_distances, intra_distances)
        else:
            t_stat, p_val = 0, 1
        
        results = {
            'mean_intra_distance': float(mean_intra),
            'mean_inter_distance': float(mean_inter),
            'coherence_ratio': float(coherence_ratio),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'clusters_coherent': bool(p_val < 0.05 and coherence_ratio > 1.5)
        }
        
        print(f"  ✓ Mean intra-cluster distance: {mean_intra:.3f}")
        print(f"  ✓ Mean inter-cluster distance: {mean_inter:.3f}")
        print(f"  ✓ Coherence ratio: {coherence_ratio:.3f}")
        print(f"  ✓ Clusters coherent: {results['clusters_coherent']}")
        
        return results
    
    def identify_transitional_artifacts(self, class_df: pd.DataFrame) -> pd.DataFrame:
        """Identify borderline cases between normal and anomalous."""
        print("\nIdentifying transitional artifacts...")
        
        # Transitional: anomaly score near decision boundary
        # Typically around 0 or near threshold
        anomaly_threshold = class_df[class_df['is_anomaly'] == True]['anomaly_score'].min()
        normal_max = class_df[class_df['is_anomaly'] == False]['anomaly_score'].max()
        
        # Transitional zone: between normal_max and anomaly_threshold
        transitional = class_df[
            (class_df['anomaly_score'] >= normal_max - 0.1) &
            (class_df['anomaly_score'] <= anomaly_threshold + 0.1)
        ].copy()
        
        # Sort by score
        transitional = transitional.sort_values('anomaly_score', ascending=False)
        
        print(f"  ✓ Identified {len(transitional)} transitional artifacts")
        print(f"  ✓ Score range: {transitional['anomaly_score'].min():.3f} to {transitional['anomaly_score'].max():.3f}")
        
        return transitional
    
    def correlate_with_robustness(self, class_df: pd.DataFrame) -> Dict:
        """Correlate anomaly types with robustness scores."""
        print("\nCorrelating anomalies with robustness...")
        
        if self.robustness.empty:
            print("  ⚠ No robustness data available")
            return {}
        
        # Merge classifications with robustness
        analysis_data = class_df.merge(
            self.robustness[['khipu_id', 'robustness_score']],
            on='khipu_id',
            how='inner'
        )
        
        # Overall correlation
        corr, p_val = stats.pearsonr(
            analysis_data['anomaly_score'],
            analysis_data['robustness_score']
        )
        
        # Robustness by deviation type
        type_robustness = {}
        for dtype in analysis_data['deviation_type'].unique():
            type_data = analysis_data[analysis_data['deviation_type'] == dtype]
            type_robustness[dtype] = {
                'mean_robustness': float(type_data['robustness_score'].mean()),
                'std_robustness': float(type_data['robustness_score'].std()),
                'n': int(len(type_data))
            }
        
        # Test if anomalies are more fragile (lower robustness)
        anomalies = analysis_data[analysis_data['is_anomaly'] == True]['robustness_score']
        normal = analysis_data[analysis_data['is_anomaly'] == False]['robustness_score']
        
        if len(anomalies) > 0 and len(normal) > 0:
            t_stat, p_val_robust = stats.ttest_ind(anomalies, normal)
            anomalies_fragile = anomalies.mean() < normal.mean() and p_val_robust < 0.05
        else:
            t_stat, p_val_robust = 0, 1
            anomalies_fragile = False
        
        results = {
            'overall_correlation': float(corr),
            'correlation_p_value': float(p_val),
            'significant_correlation': bool(p_val < 0.05),
            'robustness_by_type': type_robustness,
            'anomalies_mean_robustness': float(anomalies.mean()) if len(anomalies) > 0 else 0,
            'normal_mean_robustness': float(normal.mean()) if len(normal) > 0 else 0,
            't_statistic': float(t_stat),
            'fragility_p_value': float(p_val_robust),
            'anomalies_more_fragile': bool(anomalies_fragile)
        }
        
        print(f"  ✓ Correlation: r={corr:.3f}, p={p_val:.4f}")
        print(f"  ✓ Anomalies mean robustness: {results['anomalies_mean_robustness']:.3f}")
        print(f"  ✓ Normal mean robustness: {results['normal_mean_robustness']:.3f}")
        print(f"  ✓ Anomalies more fragile: {anomalies_fragile}")
        
        return results
    
    def save_results(self, class_df: pd.DataFrame, cluster_data: pd.DataFrame,
                    clustering_summary: Dict, coherence_results: Dict,
                    transitional: pd.DataFrame, robustness_correlation: Dict) -> None:
        """Save all analysis results."""
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        
        # Save deviation classifications
        class_df.to_csv(self.output_dir / "deviation_classifications.csv", index=False)
        print(f"  ✓ deviation_classifications.csv ({len(class_df)} khipus)")
        
        # Save anomaly subclusters
        if len(cluster_data) > 0:
            cluster_data.to_csv(self.output_dir / "anomaly_subclusters.csv", index=False)
            print(f"  ✓ anomaly_subclusters.csv ({len(cluster_data)} anomalies)")
        
        # Save transitional artifacts
        transitional.to_csv(self.output_dir / "transitional_artifacts.csv", index=False)
        print(f"  ✓ transitional_artifacts.csv ({len(transitional)} artifacts)")
        
        # Save deviation taxonomy
        taxonomy = {
            'deviation_types': {
                'numeric': 'Extreme values, impossible summations, missing values',
                'structural': 'Unusual depth, extreme branching, topology anomalies',
                'chromatic': 'Rare colors, extreme entropy, missing color data',
                'hybrid': 'Multiple deviation types present'
            },
            'distribution': class_df['deviation_type'].value_counts().to_dict(),
            'clustering': clustering_summary,
            'coherence': coherence_results,
            'robustness_correlation': robustness_correlation
        }
        
        with open(self.output_dir / "deviation_taxonomy.json", 'w') as f:
            json.dump(taxonomy, f, indent=2)
        print("  ✓ deviation_taxonomy.json")
        
        # Summary report
        summary = {
            'analysis': 'Phase 9.7: Anomaly Taxonomy (Pre-Interpretive)',
            'date': '2026-01-01',
            'total_khipus': int(len(class_df)),
            'n_anomalies': int(class_df['is_anomaly'].sum()),
            'anomaly_rate': float(class_df['is_anomaly'].mean()),
            'deviation_distribution': {
                k: int(v) for k, v in class_df['deviation_type'].value_counts().items()
            },
            'clustering_summary': clustering_summary,
            'coherence_results': coherence_results,
            'n_transitional': int(len(transitional)),
            'robustness_correlation': robustness_correlation
        }
        
        with open(self.output_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print("  ✓ analysis_summary.json")
        
        print(f"\n✓ All results saved to: {self.output_dir}")
    
    def run_analysis(self) -> None:
        """Execute complete anomaly taxonomy analysis."""
        print("=" * 80)
        print("PHASE 9.7: ANOMALY TAXONOMY (PRE-INTERPRETIVE)")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Load data
        self.load_data()
        
        # Classify deviation types
        class_df = self.classify_deviation_types()
        
        # Cluster anomalies
        cluster_data, clustering_summary = self.cluster_anomalies(class_df)
        
        # Test coherence
        coherence_results = self.test_coherence(cluster_data)
        
        # Identify transitional artifacts
        transitional = self.identify_transitional_artifacts(class_df)
        
        # Correlate with robustness
        robustness_correlation = self.correlate_with_robustness(class_df)
        
        # Save results
        self.save_results(class_df, cluster_data, clustering_summary,
                         coherence_results, transitional, robustness_correlation)
        
        print("\n" + "=" * 80)
        print("PHASE 9.7 COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    analyzer = AnomalyTaxonomyAnalyzer()
    analyzer.run_analysis()
