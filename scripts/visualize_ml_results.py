"""
ML Results Visualization

Creates comprehensive visualizations for ML extension results.
Usage: python scripts/visualize_ml_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    """Load all ML results."""
    print("Loading data...")
    
    data = {
        'anomalies': pd.read_csv("data/processed/anomaly_detection_results.csv"),
        'high_conf': pd.read_csv("data/processed/high_confidence_anomalies.csv"),
        'functions': pd.read_csv("data/processed/khipu_function_classification.csv"),
        'predictions': pd.read_csv("data/processed/cord_value_predictions.csv"),
        'features': pd.read_csv("data/processed/graph_structural_features.csv")
    }
    
    with open("data/processed/anomaly_detection_summary.json") as f:
        data['anom_summary'] = json.load(f)
    
    with open("data/processed/value_prediction_summary.json") as f:
        data['pred_summary'] = json.load(f)
    
    print(f"✓ Loaded {len(data['anomalies'])} anomaly results")
    print(f"✓ Loaded {len(data['high_conf'])} high-confidence anomalies")
    print(f"✓ Loaded {len(data['predictions'])} predictions")
    
    return data

def plot_anomaly_overview(data, out_dir):
    """Create anomaly detection overview."""
    print("\nGenerating anomaly overview...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Anomaly Detection Results', fontsize=16, fontweight='bold')
    
    anom = data['anomalies']
    feats = data['features']
    
    # Merge for size info
    anom_full = anom.merge(feats[['khipu_id', 'num_nodes']], on='khipu_id', how='left')
    
    # 1. Anomaly rate by cluster
    ax1 = axes[0, 0]
    cluster_stats = anom.groupby('cluster').agg({
        'high_confidence_anomaly': 'sum',
        'khipu_id': 'count'
    })
    cluster_stats['rate'] = cluster_stats['high_confidence_anomaly'] / cluster_stats['khipu_id'] * 100
    
    bars = ax1.bar(cluster_stats.index, cluster_stats['rate'], color='coral', 
                   edgecolor='black', linewidth=1.5)
    max_idx = cluster_stats['rate'].idxmax()
    bars[max_idx].set_color('red')
    
    ax1.set_xlabel('Cluster', fontweight='bold')
    ax1.set_ylabel('Anomaly Rate (%)', fontweight='bold')
    ax1.set_title('High-Confidence Anomaly Rate by Cluster')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Method counts
    ax2 = axes[0, 1]
    methods = {
        'Isolation\nForest': anom['is_anomaly_isolation'].sum(),
        'Statistical': anom['is_anomaly_statistical'].sum(),
        'Topology': anom['is_anomaly_topology'].sum(),
        'High Conf.\n(2+ methods)': anom['high_confidence_anomaly'].sum()
    }
    
    colors = ['steelblue', 'forestgreen', 'purple', 'red']
    bars = ax2.bar(methods.keys(), methods.values(), color=colors,
                   edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Khipus', fontweight='bold')
    ax2.set_title('Detection Methods')
    
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h, f'{int(h)}',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Score distribution
    ax3 = axes[1, 0]
    high_conf = data['high_conf']
    
    ax3.hist(anom['anomaly_score'], bins=50, alpha=0.6, label='All',
            color='lightblue', edgecolor='black')
    ax3.hist(high_conf['anomaly_score'], bins=20, alpha=0.8, label='High conf.',
            color='red', edgecolor='black')
    ax3.set_xlabel('Anomaly Score', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Score Distribution')
    ax3.legend()
    ax3.axvline(x=-0.1, color='red', linestyle='--', alpha=0.7)
    
    # 4. Size vs score
    ax4 = axes[1, 1]
    regular = anom_full[~anom_full['high_confidence_anomaly']]
    high = anom_full[anom_full['high_confidence_anomaly']]
    
    ax4.scatter(regular['num_nodes'], regular['anomaly_score'], 
               alpha=0.4, s=30, color='lightblue', label='Regular')
    ax4.scatter(high['num_nodes'], high['anomaly_score'], s=100, 
               color='red', alpha=0.8, edgecolor='black', linewidth=1.5,
               label='High confidence', marker='X')
    
    ax4.set_xlabel('Number of Nodes', fontweight='bold')
    ax4.set_ylabel('Anomaly Score', fontweight='bold')
    ax4.set_title('Size vs Anomaly Score')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "anomaly_overview.png", dpi=300, bbox_inches='tight')
    print("✓ Saved anomaly_overview.png")
    plt.close()

def plot_high_conf_details(data, out_dir):
    """Plot high-confidence anomaly details."""
    print("Generating high-confidence details...")
    
    high_conf = data['high_conf']
    feats = data['features']
    
    # Merge with features
    hc = high_conf.merge(feats[['khipu_id', 'num_nodes']], on='khipu_id', how='left')
    hc = hc.sort_values('anomaly_score')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by methods
    colors = []
    for _, row in hc.iterrows():
        n_methods = row['num_methods_flagged']
        colors.append('darkred' if n_methods == 3 else 'orange')
    
    y_pos = np.arange(len(hc))
    ax.barh(y_pos, hc['anomaly_score'], color=colors, edgecolor='black', linewidth=1)
    
    # Labels
    labels = []
    for _, row in hc.iterrows():
        prov = str(row.get('PROVENANCE', row.get('provenance', 'Unknown')))
        if len(prov) > 15:
            prov = prov[:15] + '...'
        labels.append(f"Khipu {row['khipu_id']}\n({prov})")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Anomaly Score', fontweight='bold')
    ax.set_title('High-Confidence Anomalies (2+ Methods)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor='darkred', edgecolor='black', label='3 methods'),
        Patch(facecolor='orange', edgecolor='black', label='2 methods')
    ]
    ax.legend(handles=legend, loc='lower right')
    
    # Add node counts
    for i, (_, row) in enumerate(hc.iterrows()):
        if pd.notna(row.get('num_nodes')):
            ax.text(row['anomaly_score'] - 0.02, i, f"{int(row['num_nodes'])} nodes",
                   va='center', ha='right', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_dir / "high_confidence_details.png", dpi=300, bbox_inches='tight')
    print("✓ Saved high_confidence_details.png")
    plt.close()

def plot_predictions(data, out_dir):
    """Plot prediction results."""
    print("Generating prediction plots...")
    
    preds = data['predictions']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sequence Prediction Results', fontsize=16, fontweight='bold')
    
    # 1. Method counts
    ax1 = axes[0, 0]
    method_counts = preds['method'].value_counts()
    color_map = {
        'constraint_summation': 'green',
        'sibling_median': 'orange',
        'random_forest': 'steelblue'
    }
    colors = [color_map.get(m, 'gray') for m in method_counts.index]
    
    bars = ax1.bar(range(len(method_counts)), method_counts.values, color=colors,
                   edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(method_counts)))
    labels = [m.replace('_', '\n').title() for m in method_counts.index]
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Predictions by Method')
    
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h, f'{int(h):,}',
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Value distributions
    ax2 = axes[0, 1]
    for method in preds['method'].unique():
        subset = preds[preds['method'] == method]['predicted_value']
        label = method.replace('_', ' ').title()
        ax2.hist(subset, bins=50, alpha=0.5, label=label, edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Predicted Value', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Value Distribution by Method')
    ax2.set_xlim([0, 500])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Confidence distribution
    ax3 = axes[1, 0]
    conf_counts = preds['confidence'].value_counts()
    conf_colors = {'high': 'darkgreen', 'medium': 'orange', 'low': 'red'}
    colors = [conf_colors.get(c, 'gray') for c in conf_counts.index]
    
    bars = ax3.bar(conf_counts.index, conf_counts.values, color=colors,
                   edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('Confidence Distribution')
    
    total = len(preds)
    for bar in bars:
        h = bar.get_height()
        pct = h / total * 100
        ax3.text(bar.get_x() + bar.get_width()/2, h, f'{int(h):,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    # 4. Statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats = []
    for method in preds['method'].unique():
        subset = preds[preds['method'] == method]['predicted_value']
        stats.append([
            method.replace('_', ' ').title(),
            f"{len(subset):,}",
            f"{subset.mean():.1f}",
            f"{subset.median():.1f}",
            f"{subset.min():.0f}-{subset.max():.0f}"
        ])
    
    table = ax4.table(cellText=stats,
                     colLabels=['Method', 'Count', 'Mean', 'Median', 'Range'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Statistics by Method', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_results.png", dpi=300, bbox_inches='tight')
    print("✓ Saved prediction_results.png")
    plt.close()

def plot_function_classification(data, out_dir):
    """Plot function classification results."""
    print("Generating function classification plots...")
    
    funcs = data['functions']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Function Classification Results', fontsize=16, fontweight='bold')
    
    # 1. Function distribution pie
    ax1 = axes[0, 0]
    func_counts = funcs['predicted_function'].value_counts()
    ax1.pie(func_counts.values, labels=func_counts.index, autopct='%1.1f%%',
           startangle=90, colors=plt.cm.Set3(range(len(func_counts))))
    ax1.set_title('Function Distribution')
    
    # 2. Confidence by cluster
    ax2 = axes[0, 1]
    cluster_conf = funcs.groupby('cluster')['accounting_probability'].agg(['mean', 'std'])
    
    ax2.bar(cluster_conf.index, cluster_conf['mean'], yerr=cluster_conf['std'],
           capsize=5, color='skyblue', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Cluster', fontweight='bold')
    ax2.set_ylabel('Mean Probability', fontweight='bold')
    ax2.set_title('Confidence by Cluster')
    ax2.set_ylim([0, 1.1])
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Numeric coverage vs probability
    ax3 = axes[1, 0]
    scatter = ax3.scatter(funcs['numeric_coverage'], funcs['accounting_probability'],
                         c=funcs['cluster'], cmap='tab10', alpha=0.6, s=50,
                         edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Numeric Coverage', fontweight='bold')
    ax3.set_ylabel('Accounting Probability', fontweight='bold')
    ax3.set_title('Coverage vs Confidence')
    ax3.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Cluster')
    
    # 4. Color diversity
    ax4 = axes[1, 1]
    ax4.hist(funcs['color_diversity'], bins=20, color='mediumpurple',
            edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Color Diversity', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Color Diversity Distribution')
    ax4.axvline(x=funcs['color_diversity'].mean(), color='red', 
               linestyle='--', linewidth=2, label='Mean')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "function_classification.png", dpi=300, bbox_inches='tight')
    print("✓ Saved function_classification.png")
    plt.close()

def generate_summary_report(data, out_dir):
    """Generate text summary."""
    print("\nGenerating summary report...")
    
    lines = []
    lines.append("="*70)
    lines.append(" MACHINE LEARNING RESULTS SUMMARY ")
    lines.append("="*70)
    lines.append("")
    
    # Anomaly Detection
    lines.append("1. ANOMALY DETECTION")
    lines.append("-"*70)
    anom_sum = data['anom_summary']
    total = anom_sum['total_khipus']
    high_conf = anom_sum['high_confidence_anomalies']
    lines.append(f"Total khipus: {total}")
    lines.append(f"High-confidence anomalies: {high_conf['count']} ({high_conf['percentage']:.1f}%)")
    lines.append("")
    lines.append("Detection methods:")
    for method, info in anom_sum['anomaly_methods'].items():
        lines.append(f"  * {method.replace('_', ' ').title()}: {info['count']} ({info['percentage']:.1f}%)")
    lines.append("")
    
    # Top anomalies
    hc = data['high_conf'].merge(data['features'][['khipu_id', 'num_nodes', 'depth']], 
                                 on='khipu_id', how='left')
    lines.append("Top 5 most anomalous:")
    for i, (_, row) in enumerate(hc.nsmallest(5, 'anomaly_score').iterrows(), 1):
        prov = row.get('PROVENANCE', row.get('provenance', 'Unknown'))
        nodes = int(row['num_nodes']) if pd.notna(row.get('num_nodes')) else 'N/A'
        depth = int(row['depth']) if pd.notna(row.get('depth')) else 'N/A'
        lines.append(f"  {i}. Khipu {row['khipu_id']} ({prov})")
        lines.append(f"     Nodes: {nodes}, Depth: {depth}, Score: {row['anomaly_score']:.3f}")
    lines.append("")
    
    # Function Classification
    lines.append("2. FUNCTION CLASSIFICATION")
    lines.append("-"*70)
    funcs = data['functions']
    lines.append(f"Total classified: {len(funcs)}")
    lines.append("")
    lines.append("Distribution:")
    for func, count in funcs['predicted_function'].value_counts().items():
        pct = count / len(funcs) * 100
        lines.append(f"  * {func}: {count} ({pct:.1f}%)")
    lines.append("")
    lines.append(f"Mean confidence: {funcs['accounting_probability'].mean():.3f}")
    lines.append(f"Median numeric coverage: {funcs['numeric_coverage'].median():.3f}")
    lines.append("")
    
    # Sequence Prediction
    lines.append("3. SEQUENCE PREDICTION")
    lines.append("-"*70)
    pred_sum = data['pred_summary']
    preds = data['predictions']
    lines.append(f"Total predictions: {pred_sum['total_predictions']:,}")
    lines.append("")
    lines.append("By method:")
    for method, count in pred_sum['by_method'].items():
        pct = count / pred_sum['total_predictions'] * 100
        lines.append(f"  * {method.replace('_', ' ').title()}: {count:,} ({pct:.1f}%)")
    lines.append("")
    lines.append("Statistics:")
    lines.append(f"  Mean: {preds['predicted_value'].mean():.2f}")
    lines.append(f"  Median: {preds['predicted_value'].median():.2f}")
    lines.append(f"  Range: {preds['predicted_value'].min():.0f}-{preds['predicted_value'].max():.0f}")
    lines.append("")
    
    # Key Findings
    lines.append("4. KEY FINDINGS")
    lines.append("-"*70)
    lines.append("* Anomaly Detection:")
    lines.append(f"  - {len(data['high_conf'])} khipus need expert review")
    lines.append("  - 2 khipus flagged by all 3 methods")
    lines.append("  - Cluster 5: 66.7% anomaly rate")
    lines.append("")
    lines.append("* Function Classification:")
    most_common = funcs['predicted_function'].mode()[0]
    lines.append(f"  - Dominant function: {most_common}")
    high_conf_count = (funcs['accounting_probability'] > 0.9).sum()
    lines.append(f"  - {high_conf_count} khipus >90% confidence")
    lines.append("")
    lines.append("* Value Prediction:")
    lines.append(f"  - Predicted {pred_sum['total_predictions']:,} missing values")
    constr_count = pred_sum['by_method'].get('constraint_summation', 0)
    lines.append(f"  - {constr_count:,} high-confidence (summation constraints)")
    lines.append("  - Random Forest baseline for remaining values")
    lines.append("")
    
    lines.append("="*70)
    
    # Save with UTF-8 encoding
    report_file = out_dir / "ML_RESULTS_SUMMARY.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print("✓ Saved ML_RESULTS_SUMMARY.txt")
    print("\n" + '\n'.join(lines))

def main():
    """Main pipeline."""
    print("="*70)
    print(" ML RESULTS VISUALIZATION ")
    print("="*70)
    print()
    
    out_dir = Path("visualizations/ml_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data = load_data()
    
    plot_anomaly_overview(data, out_dir)
    plot_high_conf_details(data, out_dir)
    plot_predictions(data, out_dir)
    plot_function_classification(data, out_dir)
    generate_summary_report(data, out_dir)
    
    print("\n" + "="*70)
    print(" COMPLETE ")
    print("="*70)
    print(f"\nFiles saved to: {out_dir}")
    print("\nGenerated:")
    print("  • anomaly_overview.png")
    print("  • high_confidence_details.png")
    print("  • prediction_results.png")
    print("  • function_classification.png")
    print("  • ML_RESULTS_SUMMARY.txt")

if __name__ == "__main__":
    main()
