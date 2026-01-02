"""
Phase 5 Visualization: Hypothesis Testing

Generates 3 visualizations from actual Phase 5 hypothesis testing results.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path("visualizations/phase5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def plot_color_hypothesis_tests():
    """Plot results from color hypothesis testing."""
    print("Generating color hypothesis test results...")
    
    with open("data/processed/phase5/color_hypothesis_tests.json", "r") as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: White boundary effect
    wb = data['results']['white_boundary']
    axes[0, 0].bar(['With White\nBoundaries', 'Without White\nBoundaries'],
                   [wb['summation_with_white'] * 100, wb['summation_without_white'] * 100],
                   color=['mediumseagreen', 'coral'], edgecolor='black', alpha=0.7)
    axes[0, 0].set_ylabel('Summation Match Rate (%)')
    axes[0, 0].set_title('White Cord Boundary Hypothesis')
    axes[0, 0].set_ylim([0, 35])
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Panel 2: Color-value correlation (top colors)
    cv = data['results']['color_value_correlation']['color_value_stats'][:8]
    colors = [c['color'] for c in cv]
    means = [c['mean_value'] for c in cv]
    axes[0, 1].bar(colors, means, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_ylabel('Mean Numeric Value')
    axes[0, 1].set_xlabel('Color Code')
    axes[0, 1].set_title('Color-Value Correlation (Top 8 Colors)')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Panel 3: Regional color preferences (if exists)
    if 'regional_color_preference' in data['results']:
        rcp = data['results']['regional_color_preference']
        top_regions = list(rcp['region_stats'].items())[:6]
        region_names = [r[0][:15] for r in top_regions]
        color_diversity = [r[1]['color_diversity'] for r in top_regions]
        axes[1, 0].barh(region_names, color_diversity, color='coral', 
                        edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Color Diversity (Unique Colors)')
        axes[1, 0].set_title('Regional Color Diversity')
        axes[1, 0].grid(axis='x', alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Regional color\ndata not available',
                        ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('Regional Color Diversity')
    
    # Panel 4: Hypothesis verdicts
    verdicts = {k: v.get('verdict', 'N/A') for k, v in data['results'].items()}
    verdict_counts = {'SUPPORTED': 0, 'REJECTED': 0, 'MIXED': 0}
    for v in verdicts.values():
        if v in verdict_counts:
            verdict_counts[v] += 1
    
    colors_verdict = {'SUPPORTED': 'mediumseagreen', 'MIXED': 'gold', 'REJECTED': 'coral'}
    axes[1, 1].bar(verdict_counts.keys(), verdict_counts.values(),
                   color=[colors_verdict[k] for k in verdict_counts.keys()],
                   edgecolor='black', alpha=0.7)
    axes[1, 1].set_ylabel('Number of Hypotheses')
    axes[1, 1].set_title(f'Hypothesis Test Verdicts (n={data["hypotheses_tested"]})')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Phase 5: Color Hypothesis Testing Results', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "color_hypothesis_tests.png", bbox_inches='tight')
    plt.close()
    print("  [OK] Saved")


def plot_function_classification():
    """Plot khipu function classification distribution."""
    print("Generating function classification distribution...")
    
    df = pd.read_csv("data/processed/phase5/khipu_function_classification.csv")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel 1: Function distribution
    func_counts = df['predicted_function'].value_counts()
    axes[0].bar(func_counts.index, func_counts.values, 
                color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_ylabel('Number of Khipus')
    axes[0].set_xlabel('Predicted Function')
    axes[0].set_title('Function Classification Distribution')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Panel 2: Accounting probability distribution
    axes[1].hist(df['accounting_probability'], bins=30, 
                 color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(df['accounting_probability'].mean(), color='red',
                    linestyle='--', linewidth=2, 
                    label=f'Mean: {df["accounting_probability"].mean():.3f}')
    axes[1].set_xlabel('Accounting Probability')
    axes[1].set_ylabel('Number of Khipus')
    axes[1].set_title('Accounting Function Confidence')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Panel 3: Function by cluster
    cluster_func = pd.crosstab(df['cluster'], df['predicted_function'])
    cluster_func.plot(kind='bar', stacked=True, ax=axes[2], 
                      color=['steelblue', 'coral', 'mediumseagreen'], alpha=0.7)
    axes[2].set_xlabel('Cluster')
    axes[2].set_ylabel('Number of Khipus')
    axes[2].set_title('Function Distribution Across Clusters')
    axes[2].legend(title='Function')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "function_classification.png", bbox_inches='tight')
    plt.close()
    print("  [OK] Saved")


def plot_geographic_correlations():
    """Plot geographic-cluster correlations."""
    print("Generating geographic correlation analysis...")
    
    with open("data/processed/phase5/geographic_correlation_analysis.json", "r") as f:
        data = json.load(f)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract contingency table
    contingency = data['cluster_provenance_enrichment']['contingency_table']
    
    # Convert to DataFrame
    df_contingency = pd.DataFrame(contingency).T.fillna(0)
    
    # Select top 10 provenances by total count
    top_provenances = df_contingency.sum(axis=1).nlargest(10).index
    df_plot = df_contingency.loc[top_provenances]
    
    # Shorten provenance names
    df_plot.index = [name[:25] + '...' if len(name) > 25 else name 
                     for name in df_plot.index]
    
    # Create heatmap
    sns.heatmap(df_plot, annot=True, fmt='.0f', cmap='YlOrRd',
                cbar_kws={'label': 'Number of Khipus'},
                linewidths=1, linecolor='black', ax=ax)
    
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Provenance')
    ax.set_title('Geographic-Cluster Distribution (Top 10 Provenances)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "geographic_cluster_correlation.png", bbox_inches='tight')
    plt.close()
    print("  [OK] Saved")


def main():
    """Generate all Phase 5 visualizations."""
    print("=" * 80)
    print("PHASE 5 VISUALIZATION GENERATION")
    print("=" * 80)
    print()
    
    plot_color_hypothesis_tests()
    plot_function_classification()
    plot_geographic_correlations()
    
    print()
    print("=" * 80)
    print("PHASE 5 VISUALIZATIONS COMPLETE")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
