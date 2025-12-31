"""
Anomaly Detection for Khipu Dataset

Identifies outlier khipus that don't fit established structural patterns.
Uses Isolation Forest and statistical methods to flag potential:
- Transcription errors
- Unique/rare specimens
- Data quality issues

Three approaches:
1. Isolation Forest on structural features
2. Statistical outliers (Z-score method)
3. Graph-based anomalies (unusual topologies)

Usage: python scripts/detect_anomalies.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import sqlite3
import json
from pathlib import Path

def load_data():
    """Load all necessary data for anomaly detection."""
    print("Loading khipu data...")
    
    # Load features and clustering
    features = pd.read_csv("data/processed/graph_structural_features.csv")
    clusters = pd.read_csv("data/processed/cluster_assignments_kmeans.csv")
    summation = pd.read_csv("data/processed/summation_test_results.csv")
    
    # Load provenance
    conn = sqlite3.connect("khipu.db")
    provenance = pd.read_sql_query("SELECT KHIPU_ID, PROVENANCE FROM khipu_main", conn)
    conn.close()
    
    # Merge
    data = features.merge(clusters[['khipu_id', 'cluster']], on='khipu_id')
    data = data.merge(summation[['khipu_id', 'has_pendant_summation', 'pendant_match_rate']], on='khipu_id')
    data = data.merge(provenance, left_on='khipu_id', right_on='KHIPU_ID', how='left')
    
    print(f"Loaded {len(data)} khipus")
    return data

def isolation_forest_anomalies(data, contamination=0.05):
    """
    Detect anomalies using Isolation Forest.
    
    Args:
        data: DataFrame with khipu features
        contamination: Expected proportion of outliers (default 5%)
    
    Returns:
        DataFrame with anomaly scores and labels
    """
    print(f"\n{'='*60}")
    print("ISOLATION FOREST ANOMALY DETECTION")
    print(f"{'='*60}\n")
    
    # Select features for anomaly detection
    feature_cols = [
        'num_nodes', 'num_edges', 'avg_degree', 'max_degree', 'density',
        'depth', 'width', 'avg_branching', 'num_roots', 'num_leaves',
        'has_numeric', 'has_color', 'avg_numeric_value', 'std_numeric_value'
    ]
    
    X = data[feature_cols].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200
    )
    
    predictions = iso_forest.fit_predict(X_scaled)
    scores = iso_forest.score_samples(X_scaled)
    
    # Create results DataFrame
    results = data[['khipu_id', 'PROVENANCE', 'cluster']].copy()
    results['anomaly_score'] = scores
    results['is_anomaly_isolation'] = (predictions == -1)
    
    # Add feature values for context
    for col in feature_cols:
        results[col] = data[col]
    
    n_anomalies = results['is_anomaly_isolation'].sum()
    print(f"Expected outliers ({contamination*100}%): {int(len(data) * contamination)}")
    print(f"Detected anomalies: {n_anomalies} ({n_anomalies/len(data)*100:.1f}%)")
    
    # Show most anomalous
    print(f"\nTop 10 most anomalous khipus:")
    print("-" * 60)
    anomalous = results.sort_values('anomaly_score').head(10)
    for _, row in anomalous.iterrows():
        print(f"Khipu {row['khipu_id']:7d} | Score: {row['anomaly_score']:7.3f} | "
              f"Cluster: {row['cluster']} | {row['PROVENANCE']}")
        print(f"  Size: {row['num_nodes']:3.0f} nodes | Depth: {row['depth']:3.0f} | "
              f"Branching: {row['avg_branching']:.2f} | Density: {row['density']:.3f}")
    
    return results

def statistical_outliers(data, z_threshold=3):
    """
    Detect outliers using Z-score method on key features.
    
    Args:
        data: DataFrame with khipu features
        z_threshold: Number of standard deviations for outlier (default 3)
    
    Returns:
        DataFrame with outlier flags for each feature
    """
    print(f"\n{'='*60}")
    print("STATISTICAL OUTLIER DETECTION (Z-SCORE METHOD)")
    print(f"{'='*60}\n")
    
    results = data[['khipu_id', 'PROVENANCE', 'cluster']].copy()
    
    # Features to check
    check_features = ['num_nodes', 'depth', 'avg_branching', 'density', 'avg_numeric_value']
    
    outlier_counts = []
    
    for feat in check_features:
        values = data[feat].dropna()
        mean = values.mean()
        std = values.std()
        z_scores = np.abs((data[feat] - mean) / std)
        
        is_outlier = z_scores > z_threshold
        results[f'{feat}_outlier'] = is_outlier
        
        n_outliers = is_outlier.sum()
        outlier_counts.append(n_outliers)
        
        print(f"{feat:20s}: {n_outliers:3d} outliers ({n_outliers/len(data)*100:5.1f}%) | "
              f"Mean: {mean:8.2f} | Std: {std:8.2f}")
    
    # Count total outlier flags per khipu
    outlier_cols = [c for c in results.columns if c.endswith('_outlier')]
    results['num_outlier_flags'] = results[outlier_cols].sum(axis=1)
    results['is_anomaly_statistical'] = results['num_outlier_flags'] >= 2
    
    n_multi_outliers = results['is_anomaly_statistical'].sum()
    print(f"\nKhipus with 2+ outlier flags: {n_multi_outliers} ({n_multi_outliers/len(data)*100:.1f}%)")
    
    # Show top anomalies
    print(f"\nTop 10 khipus with most outlier flags:")
    print("-" * 60)
    top_anomalies = results.sort_values('num_outlier_flags', ascending=False).head(10)
    for _, row in top_anomalies.iterrows():
        outlier_features = [col.replace('_outlier', '') for col in outlier_cols if row[col]]
        print(f"Khipu {row['khipu_id']:7d} | {row['num_outlier_flags']} flags | "
              f"Cluster: {row['cluster']} | {row['PROVENANCE']}")
        print(f"  Outlier in: {', '.join(outlier_features)}")
    
    return results

def graph_topology_anomalies(data):
    """
    Detect unusual graph topologies.
    
    Flags khipus with:
    - Unusual depth-to-width ratios
    - Anomalous branching patterns
    - Disconnected components
    """
    print(f"\n{'='*60}")
    print("GRAPH TOPOLOGY ANOMALY DETECTION")
    print(f"{'='*60}\n")
    
    results = data[['khipu_id', 'PROVENANCE', 'cluster', 'num_nodes', 'depth', 'width', 
                    'avg_branching', 'num_roots', 'num_leaves', 'density']].copy()
    
    # Check for unusual patterns
    
    # 1. Multiple roots (should typically be 1)
    results['multi_root_anomaly'] = results['num_roots'] > 1
    n_multi_root = results['multi_root_anomaly'].sum()
    print(f"Multiple roots: {n_multi_root} khipus ({n_multi_root/len(data)*100:.1f}%)")
    
    # 2. Depth/width ratio anomalies
    results['depth_width_ratio'] = results['depth'] / (results['width'] + 1)
    ratio_mean = results['depth_width_ratio'].mean()
    ratio_std = results['depth_width_ratio'].std()
    results['extreme_ratio_anomaly'] = np.abs(results['depth_width_ratio'] - ratio_mean) > 3 * ratio_std
    n_ratio = results['extreme_ratio_anomaly'].sum()
    print(f"Extreme depth/width ratio: {n_ratio} khipus ({n_ratio/len(data)*100:.1f}%)")
    
    # 3. Unusual branching (very high or very low)
    branching_q99 = results['avg_branching'].quantile(0.99)
    branching_q01 = results['avg_branching'].quantile(0.01)
    results['extreme_branching_anomaly'] = (
        (results['avg_branching'] > branching_q99) |
        (results['avg_branching'] < branching_q01)
    )
    n_branching = results['extreme_branching_anomaly'].sum()
    print(f"Extreme branching: {n_branching} khipus ({n_branching/len(data)*100:.1f}%)")
    print(f"  99th percentile: {branching_q99:.2f} | 1st percentile: {branching_q01:.2f}")
    
    # 4. Star topology (single level, many leaves)
    results['star_topology'] = (results['depth'] == 1) & (results['num_leaves'] > 20)
    n_star = results['star_topology'].sum()
    print(f"Star topologies: {n_star} khipus ({n_star/len(data)*100:.1f}%)")
    
    # Overall topology anomaly flag
    topology_cols = ['multi_root_anomaly', 'extreme_ratio_anomaly', 
                     'extreme_branching_anomaly', 'star_topology']
    results['num_topology_flags'] = results[topology_cols].sum(axis=1)
    results['is_anomaly_topology'] = results['num_topology_flags'] >= 1
    
    n_topology_anomalies = results['is_anomaly_topology'].sum()
    print(f"\nTotal topology anomalies: {n_topology_anomalies} ({n_topology_anomalies/len(data)*100:.1f}%)")
    
    # Show examples
    print(f"\nTop 10 topology anomalies:")
    print("-" * 60)
    top_topo = results[results['is_anomaly_topology']].sort_values('num_topology_flags', ascending=False).head(10)
    for _, row in top_topo.iterrows():
        flags = [col.replace('_anomaly', '').replace('_', ' ') for col in topology_cols if row[col]]
        print(f"Khipu {row['khipu_id']:7d} | {row['num_topology_flags']} flags | "
              f"Cluster: {row['cluster']} | {row['PROVENANCE']}")
        print(f"  Flags: {', '.join(flags)}")
        print(f"  Stats: {row['num_nodes']:.0f} nodes | Depth: {row['depth']:.0f} | "
              f"Width: {row['width']:.0f} | Branching: {row['avg_branching']:.2f}")
    
    return results

def combine_anomaly_detections(iso_results, stat_results, topo_results):
    """Combine all anomaly detection methods into a unified report."""
    print(f"\n{'='*60}")
    print("COMBINED ANOMALY REPORT")
    print(f"{'='*60}\n")
    
    # Merge results
    combined = iso_results[['khipu_id', 'PROVENANCE', 'cluster', 'anomaly_score', 'is_anomaly_isolation']].copy()
    combined = combined.merge(
        stat_results[['khipu_id', 'is_anomaly_statistical', 'num_outlier_flags']],
        on='khipu_id'
    )
    combined = combined.merge(
        topo_results[['khipu_id', 'is_anomaly_topology', 'num_topology_flags']],
        on='khipu_id'
    )
    
    # Count methods that flagged each khipu
    combined['num_methods_flagged'] = (
        combined['is_anomaly_isolation'].astype(int) +
        combined['is_anomaly_statistical'].astype(int) +
        combined['is_anomaly_topology'].astype(int)
    )
    
    # High-confidence anomalies (flagged by 2+ methods)
    combined['high_confidence_anomaly'] = combined['num_methods_flagged'] >= 2
    
    # Summary statistics
    n_high_conf = combined['high_confidence_anomaly'].sum()
    n_any = (combined['num_methods_flagged'] > 0).sum()
    
    print(f"Flagged by at least 1 method: {n_any} khipus ({n_any/len(combined)*100:.1f}%)")
    print(f"HIGH CONFIDENCE (2+ methods): {n_high_conf} khipus ({n_high_conf/len(combined)*100:.1f}%)")
    
    # Method agreement
    print(f"\nMethod agreement:")
    print(f"  Isolation Forest only:  {(combined['is_anomaly_isolation'] & ~combined['is_anomaly_statistical'] & ~combined['is_anomaly_topology']).sum()}")
    print(f"  Statistical only:        {(~combined['is_anomaly_isolation'] & combined['is_anomaly_statistical'] & ~combined['is_anomaly_topology']).sum()}")
    print(f"  Topology only:           {(~combined['is_anomaly_isolation'] & ~combined['is_anomaly_statistical'] & combined['is_anomaly_topology']).sum()}")
    print(f"  Isolation + Statistical: {(combined['is_anomaly_isolation'] & combined['is_anomaly_statistical'] & ~combined['is_anomaly_topology']).sum()}")
    print(f"  Isolation + Topology:    {(combined['is_anomaly_isolation'] & ~combined['is_anomaly_statistical'] & combined['is_anomaly_topology']).sum()}")
    print(f"  Statistical + Topology:  {(~combined['is_anomaly_isolation'] & combined['is_anomaly_statistical'] & combined['is_anomaly_topology']).sum()}")
    print(f"  All three methods:       {(combined['is_anomaly_isolation'] & combined['is_anomaly_statistical'] & combined['is_anomaly_topology']).sum()}")
    
    # High-confidence anomalies list
    print(f"\nHIGH CONFIDENCE ANOMALIES (flagged by 2+ methods):")
    print("-" * 80)
    high_conf = combined[combined['high_confidence_anomaly']].sort_values('num_methods_flagged', ascending=False)
    
    if len(high_conf) > 0:
        for _, row in high_conf.head(20).iterrows():
            methods = []
            if row['is_anomaly_isolation']:
                methods.append('Isolation')
            if row['is_anomaly_statistical']:
                methods.append(f'Statistical({row["num_outlier_flags"]:.0f})')
            if row['is_anomaly_topology']:
                methods.append(f'Topology({row["num_topology_flags"]:.0f})')
            
            print(f"Khipu {row['khipu_id']:7d} | Cluster: {row['cluster']} | {row['PROVENANCE']}")
            print(f"  Methods: {', '.join(methods)} | Anomaly score: {row['anomaly_score']:.3f}")
    else:
        print("  No khipus flagged by multiple methods")
    
    # Cluster distribution of anomalies
    print(f"\nAnomaly distribution by cluster:")
    print("-" * 60)
    cluster_dist = combined.groupby('cluster').agg({
        'khipu_id': 'count',
        'high_confidence_anomaly': 'sum',
        'is_anomaly_isolation': 'sum',
        'is_anomaly_statistical': 'sum',
        'is_anomaly_topology': 'sum'
    })
    cluster_dist.columns = ['Total', 'High_Conf', 'Isolation', 'Statistical', 'Topology']
    cluster_dist['Anomaly_Rate'] = cluster_dist['High_Conf'] / cluster_dist['Total'] * 100
    print(cluster_dist.to_string())
    
    return combined

def save_results(combined, iso_results, stat_results, topo_results):
    """Save anomaly detection results."""
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}\n")
    
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    # Save combined results
    output_file = output_dir / "anomaly_detection_results.csv"
    combined.to_csv(output_file, index=False)
    print(f"✓ Saved combined results: {output_file}")
    
    # Save high-confidence anomalies
    high_conf = combined[combined['high_confidence_anomaly']]
    hc_file = output_dir / "high_confidence_anomalies.csv"
    high_conf.to_csv(hc_file, index=False)
    print(f"✓ Saved high-confidence anomalies: {hc_file} ({len(high_conf)} khipus)")
    
    # Save detailed results with all flags
    detailed = iso_results.merge(stat_results, on='khipu_id')
    detailed = detailed.merge(topo_results, on='khipu_id')
    detailed_file = output_dir / "anomaly_detection_detailed.csv"
    detailed.to_csv(detailed_file, index=False)
    print(f"✓ Saved detailed results: {detailed_file}")
    
    # Save summary statistics
    summary = {
        'total_khipus': len(combined),
        'anomaly_methods': {
            'isolation_forest': {
                'count': int(combined['is_anomaly_isolation'].sum()),
                'percentage': float(combined['is_anomaly_isolation'].mean() * 100)
            },
            'statistical_outliers': {
                'count': int(combined['is_anomaly_statistical'].sum()),
                'percentage': float(combined['is_anomaly_statistical'].mean() * 100)
            },
            'topology_anomalies': {
                'count': int(combined['is_anomaly_topology'].sum()),
                'percentage': float(combined['is_anomaly_topology'].mean() * 100)
            }
        },
        'high_confidence_anomalies': {
            'count': int(combined['high_confidence_anomaly'].sum()),
            'percentage': float(combined['high_confidence_anomaly'].mean() * 100),
            'khipu_ids': combined[combined['high_confidence_anomaly']]['khipu_id'].tolist()
        },
        'by_cluster': combined.groupby('cluster')['high_confidence_anomaly'].agg(['sum', 'count', 'mean']).to_dict()
    }
    
    summary_file = output_dir / "anomaly_detection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary statistics: {summary_file}")

def main():
    """Main anomaly detection pipeline."""
    print(f"\n{'='*70}")
    print(" KHIPU ANOMALY DETECTION ")
    print(f"{'='*70}\n")
    
    # Load data
    data = load_data()
    
    # Run three detection methods
    iso_results = isolation_forest_anomalies(data, contamination=0.05)
    stat_results = statistical_outliers(data, z_threshold=3)
    topo_results = graph_topology_anomalies(data)
    
    # Combine results
    combined = combine_anomaly_detections(iso_results, stat_results, topo_results)
    
    # Save results
    save_results(combined, iso_results, stat_results, topo_results)
    
    print(f"\n{'='*70}")
    print(" ANOMALY DETECTION COMPLETE ")
    print(f"{'='*70}\n")
    
    print("Review the following files:")
    print("  • data/processed/anomaly_detection_results.csv - All khipus with anomaly flags")
    print("  • data/processed/high_confidence_anomalies.csv - Khipus flagged by 2+ methods")
    print("  • data/processed/anomaly_detection_detailed.csv - Full feature details")
    print("  • data/processed/anomaly_detection_summary.json - Statistical summary")

if __name__ == "__main__":
    main()
