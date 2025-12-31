"""
Cluster Visualization

Generate visualizations of khipu clustering results:
1. PCA scatter plot colored by cluster
2. PCA scatter plot colored by provenance
3. Cluster size distribution
4. Feature distributions by cluster
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlite3

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_data():
    """Load clustering and PCA data."""
    print("Loading data...")
    
    clusters = pd.read_csv("data/processed/cluster_assignments_kmeans.csv")
    pca = pd.read_csv("data/processed/cluster_pca_coordinates.csv")
    features = pd.read_csv("data/processed/graph_structural_features.csv")
    
    # Load provenance
    conn = sqlite3.connect("khipu.db")
    provenance = pd.read_sql_query(
        "SELECT KHIPU_ID, PROVENANCE FROM khipu_main", 
        conn
    )
    conn.close()
    
    # Merge - use only necessary columns
    data = clusters[['khipu_id', 'cluster']].merge(
        pca[['khipu_id', 'pc1', 'pc2']], on='khipu_id', how='left'
    )
    data = data.merge(
        provenance, left_on='khipu_id', right_on='KHIPU_ID', how='left'
    )
    data = data.merge(
        features[['khipu_id', 'num_nodes', 'depth', 'avg_branching', 'has_numeric']], 
        on='khipu_id', how='left'
    )
    
    # Clean provenance
    data['PROVENANCE'] = data['PROVENANCE'].fillna('Unknown')
    data['PROVENANCE'] = data['PROVENANCE'].replace(['', ' '], 'Unknown')
    
    print(f"✓ Loaded {len(data)} khipus")
    print(f"  Columns: {data.columns.tolist()}")
    return data


def plot_pca_by_cluster(data, output_dir):
    """Create PCA scatter plot colored by cluster."""
    print("\nCreating PCA cluster plot...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot each cluster
    clusters = sorted(data['cluster'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    
    for cluster, color in zip(clusters, colors):
        cluster_data = data[data['cluster'] == cluster]
        ax.scatter(
            cluster_data['pc1'], 
            cluster_data['pc2'],
            c=[color],
            label=f'Cluster {cluster} (n={len(cluster_data)})',
            alpha=0.6,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )
    
    ax.set_xlabel('PC1 (45.7% variance)', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2 (16.1% variance)', fontsize=12, fontweight='bold')
    ax.set_title('Khipu Clustering: PCA Visualization', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "cluster_pca_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


def plot_pca_by_provenance(data, output_dir):
    """Create PCA scatter plot colored by provenance."""
    print("\nCreating PCA provenance plot...")
    
    # Get top provenances
    prov_counts = data['PROVENANCE'].value_counts()
    top_provs = prov_counts.head(8).index.tolist()
    
    data_plot = data.copy()
    data_plot['prov_category'] = data_plot['PROVENANCE'].apply(
        lambda x: x if x in top_provs else 'Other'
    )
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot each provenance
    provs = sorted(data_plot['prov_category'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(provs)))
    
    for prov, color in zip(provs, colors):
        prov_data = data_plot[data_plot['prov_category'] == prov]
        ax.scatter(
            prov_data['pc1'], 
            prov_data['pc2'],
            c=[color],
            label=f'{prov} (n={len(prov_data)})',
            alpha=0.6,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )
    
    ax.set_xlabel('PC1 (45.7% variance)', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2 (16.1% variance)', fontsize=12, fontweight='bold')
    ax.set_title('Khipu Provenance: PCA Visualization', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "provenance_pca_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


def plot_cluster_sizes(data, output_dir):
    """Create bar chart of cluster sizes."""
    print("\nCreating cluster size plot...")
    
    cluster_sizes = data['cluster'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(
        cluster_sizes.index, 
        cluster_sizes.values,
        color=plt.cm.tab10(np.linspace(0, 1, len(cluster_sizes))),
        edgecolor='white',
        linewidth=1.5
    )
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height,
            f'{int(height)}',
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
    
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Khipus', fontsize=12, fontweight='bold')
    ax.set_title('Khipu Distribution Across Clusters', fontsize=14, fontweight='bold')
    ax.set_xticks(cluster_sizes.index)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "cluster_sizes.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


def plot_feature_distributions(data, output_dir):
    """Create violin plots of key features by cluster."""
    print("\nCreating feature distribution plots...")
    
    features_to_plot = ['num_nodes', 'depth', 'avg_branching', 'has_numeric']
    feature_labels = ['Number of Nodes', 'Depth', 'Avg Branching Factor', 'Has Numeric Values']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (feature, label) in enumerate(zip(features_to_plot, feature_labels)):
        ax = axes[i]
        
        # Prepare data
        plot_data = data[['cluster', feature]].dropna()
        
        # Violin plot
        parts = ax.violinplot(
            [plot_data[plot_data['cluster'] == c][feature].values 
             for c in sorted(plot_data['cluster'].unique())],
            positions=sorted(plot_data['cluster'].unique()),
            showmeans=True,
            showmedians=True
        )
        
        # Color violins
        colors = plt.cm.tab10(np.linspace(0, 1, len(parts['bodies'])))
        for body, color in zip(parts['bodies'], colors):
            body.set_facecolor(color)
            body.set_alpha(0.7)
        
        ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(f'{label} by Cluster', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Feature Distributions Across Clusters', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = Path(output_dir) / "feature_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


def create_cluster_summary_table(data, output_dir):
    """Create summary statistics table for clusters."""
    print("\nCreating cluster summary table...")
    
    summary_stats = []
    
    for cluster in sorted(data['cluster'].unique()):
        cluster_data = data[data['cluster'] == cluster]
        
        stats = {
            'Cluster': int(cluster),
            'Count': len(cluster_data),
            'Avg Nodes': f"{cluster_data['num_nodes'].mean():.1f}",
            'Avg Depth': f"{cluster_data['depth'].mean():.2f}",
            'Avg Branching': f"{cluster_data['avg_branching'].mean():.2f}",
            'Numeric %': f"{cluster_data['has_numeric'].mean()*100:.1f}%",
            'Top Provenance': cluster_data['PROVENANCE'].mode()[0] if len(cluster_data) > 0 else 'N/A'
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Save as CSV
    output_path = Path(output_dir) / "cluster_summary_table.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Saved to {output_path}")
    
    return summary_df


def main():
    """Generate all cluster visualizations."""
    print("="*80)
    print("CLUSTER VISUALIZATION SUITE")
    print("="*80)
    
    # Create output directory
    output_dir = Path("visualizations/clusters")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_data()
    
    # Generate visualizations
    plot_pca_by_cluster(data, output_dir)
    plot_pca_by_provenance(data, output_dir)
    plot_cluster_sizes(data, output_dir)
    plot_feature_distributions(data, output_dir)
    summary_df = create_cluster_summary_table(data, output_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"Generated {len(list(output_dir.glob('*.png')))} PNG files")
    print(f"Generated {len(list(output_dir.glob('*.csv')))} CSV file")
    
    print("\nCluster Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
