"""
Geographic and Motif Visualizations

Generate visualizations for:
1. Geographic heatmap of summation rates
2. Motif frequency charts by cluster
3. Provenance comparison charts
"""

import sys
from pathlib import Path

# Add src directory to path for config import
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import sqlite3  # noqa: E402
import json  # noqa: E402

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_geographic_data():
    """Load geographic correlation data."""
    print("Loading geographic data...")
    
    config = get_config()
    with open(config.get_processed_file("geographic_correlation_analysis.json"), "r") as f:
        geo_data = json.load(f)
    
    summation_data = pd.read_csv(config.get_processed_file("summation_test_results.csv"))
    
    conn = sqlite3.connect(config.get_database_path())
    provenance = pd.read_sql_query(
        "SELECT KHIPU_ID, PROVENANCE FROM khipu_main", 
        conn
    )
    conn.close()
    
    # Merge
    data = summation_data.merge(provenance, left_on='khipu_id', right_on='KHIPU_ID', how='left')
    data['PROVENANCE'] = data['PROVENANCE'].fillna('Unknown')
    
    print(f"✓ Loaded summation data for {len(data)} khipus")
    return data, geo_data


def plot_summation_by_provenance(data, output_dir):
    """Create bar chart of summation rates by provenance."""
    print("\nCreating summation by provenance plot...")
    
    # Filter out empty provenances and get top provenances
    data_filtered = data[data['PROVENANCE'].str.strip() != ''].copy()
    prov_counts = data_filtered['PROVENANCE'].value_counts()
    top_provs = prov_counts[prov_counts >= 10].index.tolist()
    
    data_top = data_filtered[data_filtered['PROVENANCE'].isin(top_provs)]
    
    # Abbreviate long provenance names
    abbrev_map = {
        'Armatambo, Huaca San Pedro': 'Armatambo/HSP',
        'Hacienda Ullujalla y Callengo': 'Ullujalla/Callengo'
    }
    
    # Calculate summation rate and match rate by provenance
    prov_stats = []
    for prov in top_provs:
        prov_data = data_top[data_top['PROVENANCE'] == prov]
        display_name = abbrev_map.get(prov, prov)
        prov_stats.append({
            'Provenance': display_name,
            'Count': len(prov_data),
            'Summation Rate': prov_data['has_pendant_summation'].mean() * 100,
            'Avg Match Rate': prov_data['pendant_match_rate'].mean() * 100
        })
    
    stats_df = pd.DataFrame(prov_stats).sort_values('Summation Rate', ascending=False)
    
    # Create plot with dynamic height based on number of provenances
    n_provs = len(stats_df)
    fig_height = max(6, n_provs * 0.5)  # At least 0.5 inches per provenance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, fig_height))
    
    # Summation rate
    bars1 = ax1.barh(
        stats_df['Provenance'], 
        stats_df['Summation Rate'],
        color=plt.cm.viridis(stats_df['Summation Rate']/stats_df['Summation Rate'].max()),
        edgecolor='white',
        linewidth=1.5
    )
    ax1.set_xlabel('Summation Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Khipus with Summation Patterns by Provenance', fontsize=13, fontweight='bold')
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, stats_df['Summation Rate']):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontweight='bold', fontsize=9)
    
    # Match rate
    bars2 = ax2.barh(
        stats_df['Provenance'], 
        stats_df['Avg Match Rate'],
        color=plt.cm.plasma(stats_df['Avg Match Rate']/stats_df['Avg Match Rate'].max()),
        edgecolor='white',
        linewidth=1.5
    )
    ax2.set_xlabel('Average Match Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Summation Accuracy by Provenance', fontsize=13, fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, stats_df['Avg Match Rate']):
        ax2.text(val + 0.2, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontweight='bold', fontsize=9)
    
    # Ensure all y-axis labels are visible
    ax1.tick_params(axis='y', labelsize=9)
    ax2.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15)  # Add space for y-axis labels
    output_path = Path(output_dir) / "summation_by_provenance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


def plot_provenance_feature_comparison(data, output_dir):
    """Create comparison of structural features by provenance."""
    print("\nCreating provenance feature comparison...")
    
    # Load features
    config = get_config()
    features = pd.read_csv(config.get_processed_file("graph_structural_features.csv"))
    conn = sqlite3.connect(config.get_database_path())
    provenance = pd.read_sql_query("SELECT KHIPU_ID, PROVENANCE FROM khipu_main", conn)
    conn.close()
    
    merged = features.merge(provenance, left_on='khipu_id', right_on='KHIPU_ID', how='left')
    merged['PROVENANCE'] = merged['PROVENANCE'].fillna('Unknown')
    
    # Filter out empty provenances and get top provenances
    merged = merged[merged['PROVENANCE'].str.strip() != ''].copy()
    prov_counts = merged['PROVENANCE'].value_counts()
    top_provs = prov_counts[prov_counts >= 10].index.tolist()[:8]
    
    merged_top = merged[merged['PROVENANCE'].isin(top_provs)]
    
    # Abbreviate long provenance names
    abbrev_map = {
        'Armatambo, Huaca San Pedro': 'Armatambo/HSP',
        'Hacienda Ullujalla y Callengo': 'Ullujalla/Callengo'
    }
    
    # Calculate stats
    prov_stats = []
    for prov in top_provs:
        prov_data = merged_top[merged_top['PROVENANCE'] == prov]
        display_name = abbrev_map.get(prov, prov)
        prov_stats.append({
            'Provenance': display_name,
            'Avg Size': prov_data['num_nodes'].mean(),
            'Avg Depth': prov_data['depth'].mean(),
            'Avg Branching': prov_data['avg_branching'].mean()
        })
    
    stats_df = pd.DataFrame(prov_stats)
    
    # Create plot with dynamic height
    n_provs = len(stats_df)
    fig_height = max(6, n_provs * 0.5)  # At least 0.5 inches per provenance
    fig, axes = plt.subplots(1, 3, figsize=(18, fig_height))
    
    metrics = ['Avg Size', 'Avg Depth', 'Avg Branching']
    titles = ['Average Khipu Size (Nodes)', 'Average Hierarchy Depth', 'Average Branching Factor']
    
    for ax, metric, title in zip(axes, metrics, titles):
        sorted_df = stats_df.sort_values(metric, ascending=False)
        
        bars = ax.barh(
            sorted_df['Provenance'], 
            sorted_df[metric],
            color=plt.cm.coolwarm(sorted_df[metric]/sorted_df[metric].max()),
            edgecolor='white',
            linewidth=1.5
        )
        
        ax.set_xlabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, sorted_df[metric]):
            ax.text(val + val*0.02, bar.get_y() + bar.get_height()/2, 
                   f'{val:.1f}', va='center', fontweight='bold', fontsize=9)
        
        # Ensure y-axis labels are visible
        ax.tick_params(axis='y', labelsize=9)
    
    plt.suptitle('Structural Features by Provenance', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    plt.subplots_adjust(left=0.12)  # Add space for y-axis labels
    output_path = Path(output_dir) / "provenance_features.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


def plot_motif_frequencies(output_dir):
    """Create motif frequency charts."""
    print("\nCreating motif frequency plots...")
    
    config = get_config()
    with open(config.get_processed_file("motif_mining_results.json"), "r") as f:
        motif_data = json.load(f)
    
    # Extract cluster motif counts
    cluster_stats = []
    for cluster_id, cluster_info in motif_data['cluster_motifs'].items():
        cluster_stats.append({
            'Cluster': int(cluster_id),
            'Total Motifs': cluster_info['branching_motifs']['total'],
            'Unique Motifs': cluster_info['branching_motifs']['unique']
        })
    
    stats_df = pd.DataFrame(cluster_stats).sort_values('Cluster')
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Total motifs
    bars1 = ax1.bar(
        stats_df['Cluster'].astype(str), 
        stats_df['Total Motifs'],
        color=plt.cm.viridis(np.linspace(0, 1, len(stats_df))),
        edgecolor='white',
        linewidth=1.5
    )
    ax1.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Branching Motifs', fontsize=12, fontweight='bold')
    ax1.set_title('Total Branching Motifs by Cluster', fontsize=13, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, stats_df['Total Motifs']):
        ax1.text(bar.get_x() + bar.get_width()/2, val + val*0.02, 
                f'{val}', ha='center', fontweight='bold', fontsize=9)
    
    # Unique motifs
    bars2 = ax2.bar(
        stats_df['Cluster'].astype(str), 
        stats_df['Unique Motifs'],
        color=plt.cm.plasma(np.linspace(0, 1, len(stats_df))),
        edgecolor='white',
        linewidth=1.5
    )
    ax2.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Unique Branching Motifs', fontsize=12, fontweight='bold')
    ax2.set_title('Unique Branching Motifs by Cluster', fontsize=13, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, stats_df['Unique Motifs']):
        ax2.text(bar.get_x() + bar.get_width()/2, val + val*0.02, 
                f'{val}', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "motif_frequencies.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


def plot_universal_motifs(output_dir):
    """Create visualization of universal motifs."""
    print("\nCreating universal motifs plot...")
    
    config = get_config()
    with open(config.get_processed_file("motif_mining_results.json"), "r") as f:
        motif_data = json.load(f)
    
    universal = motif_data['universal_motifs']['universal_branching']
    
    if not universal:
        print("  No universal motifs to plot")
        return
    
    # Parse motif data
    motif_info = []
    for motif_str, clusters in universal.items():
        motif_info.append({
            'Motif': motif_str[:30] + '...' if len(motif_str) > 30 else motif_str,
            'Num Clusters': len(clusters),
            'Clusters': str(clusters)
        })
    
    info_df = pd.DataFrame(motif_info).sort_values('Num Clusters', ascending=False)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.barh(
        range(len(info_df)),
        info_df['Num Clusters'],
        color=plt.cm.coolwarm(info_df['Num Clusters']/7),
        edgecolor='white',
        linewidth=1.5
    )
    
    ax.set_yticks(range(len(info_df)))
    ax.set_yticklabels(info_df['Motif'], fontsize=9)
    ax.set_xlabel('Number of Clusters Containing Motif', fontsize=12, fontweight='bold')
    ax.set_title('Universal Branching Motifs (Present in ≥3 Clusters)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 7.5)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, info_df['Num Clusters'])):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
               f'{val}/7', va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "universal_motifs.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


def main():
    """Generate all geographic and motif visualizations."""
    print("="*80)
    print("GEOGRAPHIC & MOTIF VISUALIZATION SUITE")
    print("="*80)
    
    # Create output directories
    geo_dir = Path("visualizations/geographic")
    motif_dir = Path("visualizations/motifs")
    geo_dir.mkdir(parents=True, exist_ok=True)
    motif_dir.mkdir(parents=True, exist_ok=True)
    
    # Geographic visualizations
    data, geo_data = load_geographic_data()
    plot_summation_by_provenance(data, geo_dir)
    plot_provenance_feature_comparison(data, geo_dir)
    
    # Motif visualizations
    plot_motif_frequencies(motif_dir)
    plot_universal_motifs(motif_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nGeographic output: {geo_dir.absolute()}")
    print(f"  Generated {len(list(geo_dir.glob('*.png')))} PNG files")
    print(f"\nMotif output: {motif_dir.absolute()}")
    print(f"  Generated {len(list(motif_dir.glob('*.png')))} PNG files")


if __name__ == "__main__":
    main()
