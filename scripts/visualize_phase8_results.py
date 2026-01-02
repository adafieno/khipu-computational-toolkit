"""
Phase 8: Visualization Script

Generate comprehensive visualizations for Phase 8 results:
1. Structural cluster analysis
2. Chromatic feature distributions
3. Feature importance comparisons
4. Administrative typology distributions
5. Confidence score analysis
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
import json  # noqa: E402

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class Phase8Visualizer:
    """Generate visualizations for Phase 8 results."""
    
    def __init__(self, data_dir: Path = None,
                 output_dir: Path = None):
        config = get_config()
        self.data_dir = data_dir if data_dir else config.processed_dir / "phase8"
        self.output_dir = output_dir if output_dir else config.root_dir / "visualizations" / "phase8"
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading Phase 8 results from: {self.data_dir}")
        
        # Load data
        self.structural_features = pd.read_csv(self.data_dir / "structural_features.csv")
        self.chromatic_features = pd.read_csv(self.data_dir / "chromatic_features.csv")
        self.cluster_assignments = pd.read_csv(self.data_dir / "structural_cluster_assignments.csv")
        self.cluster_stats = pd.read_csv(self.data_dir / "structural_cluster_statistics.csv")
        self.typology = pd.read_csv(self.data_dir / "administrative_typology.csv")
        
        # Load metadata
        with open(self.data_dir / "phase8_metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        print(f"✓ Loaded {len(self.typology)} khipus")
        print(f"✓ {self.metadata['n_clusters']} structural clusters")
        print(f"✓ Output directory: {self.output_dir}")
    
    def plot_cluster_distribution(self):
        """Plot 1: Structural cluster distribution."""
        print("\nGenerating Plot 1: Cluster Distribution...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Phase 8.1: Structural Typology (Color-Agnostic)', 
                     fontsize=16, fontweight='bold')
        
        # Merge data
        data = self.structural_features.merge(
            self.cluster_assignments, on='khipu_id'
        )
        
        # Plot 1: Cluster sizes
        ax = axes[0, 0]
        cluster_counts = data['structural_cluster'].value_counts().sort_index()
        colors = sns.color_palette("husl", len(cluster_counts))
        ax.bar(cluster_counts.index, cluster_counts.values, color=colors, alpha=0.7)
        ax.set_xlabel('Structural Cluster')
        ax.set_ylabel('Number of Khipus')
        ax.set_title('Cluster Size Distribution')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 2: Cord count by cluster
        ax = axes[0, 1]
        sns.boxplot(data=data, x='structural_cluster', y='cord_count', ax=ax, palette="husl")
        ax.set_xlabel('Structural Cluster')
        ax.set_ylabel('Cord Count')
        ax.set_title('Cord Count by Cluster')
        ax.set_yscale('log')
        
        # Plot 3: Hierarchy depth by cluster
        ax = axes[1, 0]
        sns.violinplot(data=data, x='structural_cluster', y='hierarchy_depth', ax=ax, palette="husl")
        ax.set_xlabel('Structural Cluster')
        ax.set_ylabel('Hierarchy Depth')
        ax.set_title('Hierarchy Depth Distribution by Cluster')
        
        # Plot 4: Summation vs numeric coverage by cluster
        ax = axes[1, 1]
        for cluster_id in sorted(data['structural_cluster'].unique()):
            cluster_data = data[data['structural_cluster'] == cluster_id]
            ax.scatter(cluster_data['numeric_coverage'], 
                      cluster_data['summation_match_rate'],
                      label=f'Cluster {cluster_id}',
                      alpha=0.6, s=50)
        ax.set_xlabel('Numeric Coverage')
        ax.set_ylabel('Summation Match Rate')
        ax.set_title('Summation vs Numeric Coverage by Cluster')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "01_structural_clusters.png", dpi=300, bbox_inches='tight')
        print("  ✓ Saved: 01_structural_clusters.png")
        plt.close()
    
    def plot_chromatic_features(self):
        """Plot 2: Chromatic encoding analysis."""
        print("\nGenerating Plot 2: Chromatic Features...")
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Phase 8.2: Chromatic Encoding as Administrative Affordance',
                     fontsize=16, fontweight='bold')
        
        # Merge with clusters
        data = self.chromatic_features.merge(
            self.cluster_assignments, on='khipu_id'
        )
        
        # Plot 1: Color entropy distribution
        ax = axes[0, 0]
        ax.hist(data['color_entropy'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Color Entropy')
        ax.set_ylabel('Frequency')
        ax.set_title('Color Entropy Distribution')
        ax.axvline(data['color_entropy'].median(), color='red', linestyle='--', 
                   label=f'Median: {data["color_entropy"].median():.2f}')
        ax.legend()
        
        # Plot 2: Unique colors by cluster
        ax = axes[0, 1]
        sns.boxplot(data=data, x='structural_cluster', y='unique_color_count', ax=ax, palette="Set2")
        ax.set_xlabel('Structural Cluster')
        ax.set_ylabel('Unique Color Count')
        ax.set_title('Color Diversity by Cluster')
        
        # Plot 3: Color/cord ratio
        ax = axes[0, 2]
        ax.hist(data['color_cord_ratio'], bins=30, alpha=0.7, color='coral', edgecolor='black')
        ax.set_xlabel('Color/Cord Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Color to Cord Ratio Distribution')
        
        # Plot 4: Multi-color complexity
        ax = axes[1, 0]
        ax.hist(data['multi_color_ratio'], bins=30, alpha=0.7, color='mediumpurple', edgecolor='black')
        ax.set_xlabel('Multi-Color Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Multi-Color Cord Frequency')
        
        # Plot 5: Color transitions by cluster
        ax = axes[1, 1]
        sns.violinplot(data=data, x='structural_cluster', y='color_transitions', ax=ax, palette="Set2")
        ax.set_xlabel('Structural Cluster')
        ax.set_ylabel('Color Transitions')
        ax.set_title('Color Transitions by Cluster')
        
        # Plot 6: Primary vs pendant color diversity
        ax = axes[1, 2]
        ax.scatter(data['primary_color_diversity'], data['pendant_color_diversity'], 
                  alpha=0.5, s=30, c=data['structural_cluster'], cmap='tab10')
        ax.set_xlabel('Primary Cord Color Diversity')
        ax.set_ylabel('Pendant Color Diversity')
        ax.set_title('Color Diversity: Primary vs Pendant')
        ax.plot([0, data['primary_color_diversity'].max()], 
               [0, data['primary_color_diversity'].max()], 
               'r--', alpha=0.5, label='y=x')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "02_chromatic_features.png", dpi=300, bbox_inches='tight')
        print("  ✓ Saved: 02_chromatic_features.png")
        plt.close()
    
    def plot_feature_importance(self):
        """Plot 3: Feature importance comparison across models."""
        print("\nGenerating Plot 3: Feature Importance...")
        
        # Load feature importance files
        importance_files = {
            'Structure Only': 'feature_importance_structure_only.csv',
            'Structure + Numeric': 'feature_importance_structure_numeric.csv',
            'Structure + Numeric + Color': 'feature_importance_structure_numeric_color.csv'
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Phase 8.3: Feature Importance Comparison',
                     fontsize=16, fontweight='bold')
        
        for idx, (model_name, filename) in enumerate(importance_files.items()):
            importance_df = pd.read_csv(self.data_dir / filename)
            top_features = importance_df.head(15)
            
            ax = axes[idx]
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_features['importance'], alpha=0.7, color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features['feature'], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_name}\n({len(importance_df)} features)')
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "03_feature_importance.png", dpi=300, bbox_inches='tight')
        print("  ✓ Saved: 03_feature_importance.png")
        plt.close()
    
    def plot_administrative_typology(self):
        """Plot 4: Administrative typology distribution and characteristics."""
        print("\nGenerating Plot 4: Administrative Typology...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Phase 8: Final Administrative Typology',
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Type distribution
        ax = axes[0, 0]
        type_counts = self.typology['administrative_type'].value_counts()
        colors = sns.color_palette("Set3", len(type_counts))
        wedges, texts, autotexts = ax.pie(type_counts.values, labels=type_counts.index, 
                                            autopct='%1.1f%%', colors=colors, startangle=90)
        for text in texts:
            text.set_fontsize(8)
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_color('black')
        ax.set_title('Administrative Type Distribution')
        
        # Plot 2: Confidence scores by type
        ax = axes[0, 1]
        types = self.typology['administrative_type'].unique()
        type_order = sorted(types)
        sns.boxplot(data=self.typology, x='administrative_type', y='confidence', 
                   ax=ax, order=type_order, palette="Set3")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Confidence Score')
        ax.set_xlabel('')
        ax.set_title('Confidence Scores by Administrative Type')
        ax.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='High Confidence')
        ax.legend()
        
        # Plot 3: Cord count by type
        ax = axes[1, 0]
        sns.violinplot(data=self.typology, x='administrative_type', y='cord_count',
                      ax=ax, order=type_order, palette="Set3")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Cord Count')
        ax.set_xlabel('')
        ax.set_title('Cord Count by Administrative Type')
        ax.set_yscale('log')
        
        # Plot 4: Color usage by type
        ax = axes[1, 1]
        sns.boxplot(data=self.typology, x='administrative_type', y='unique_color_count',
                   ax=ax, order=type_order, palette="Set3")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Unique Color Count')
        ax.set_xlabel('')
        ax.set_title('Color Diversity by Administrative Type')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "04_administrative_typology.png", dpi=300, bbox_inches='tight')
        print("  ✓ Saved: 04_administrative_typology.png")
        plt.close()
    
    def plot_model_comparison(self):
        """Plot 5: Model performance comparison."""
        print("\nGenerating Plot 5: Model Performance...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Phase 8.3: Model Performance Comparison',
                     fontsize=16, fontweight='bold')
        
        # Extract performance metrics
        model_names = []
        cv_means = []
        cv_stds = []
        n_features = []
        
        for model_name, metrics in self.metadata['model_performance'].items():
            model_names.append(model_name.replace('_', ' ').title())
            cv_means.append(metrics['cv_mean'])
            cv_stds.append(metrics['cv_std'])
            n_features.append(metrics['n_features'])
        
        # Plot 1: Cross-validation accuracy
        ax = axes[0]
        x_pos = np.arange(len(model_names))
        ax.bar(x_pos, cv_means, yerr=cv_stds, alpha=0.7, color='steelblue', 
               capsize=10, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.set_ylabel('Cross-Validation Accuracy')
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylim([0.7, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
            ax.text(i, mean + std + 0.01, f'{mean:.3f}', 
                   ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Number of features vs accuracy
        ax = axes[1]
        colors_map = {'Structure Only': 'coral', 'Structure Numeric': 'gold', 
                     'Structure Numeric Color': 'steelblue'}
        for i, (name, features, mean) in enumerate(zip(model_names, n_features, cv_means)):
            color = list(colors_map.values())[i]
            ax.scatter(features, mean, s=200, alpha=0.7, color=color, 
                      edgecolor='black', linewidth=2, label=name)
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Cross-Validation Accuracy')
        ax.set_title('Feature Count vs Accuracy')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "05_model_comparison.png", dpi=300, bbox_inches='tight')
        print("  ✓ Saved: 05_model_comparison.png")
        plt.close()
    
    def plot_structure_color_correlation(self):
        """Plot 6: Structure vs color correlation analysis."""
        print("\nGenerating Plot 6: Structure-Color Correlation...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Phase 8.2: Structure × Color Correlation Analysis',
                     fontsize=16, fontweight='bold')
        
        # Merge structural and chromatic features
        data = self.structural_features.merge(
            self.chromatic_features, on='khipu_id'
        ).merge(
            self.cluster_assignments, on='khipu_id'
        )
        
        # Plot 1: Hierarchy depth vs color entropy
        ax = axes[0, 0]
        scatter = ax.scatter(data['hierarchy_depth'], data['color_entropy'],
                           c=data['structural_cluster'], cmap='tab10', 
                           alpha=0.6, s=30)
        ax.set_xlabel('Hierarchy Depth')
        ax.set_ylabel('Color Entropy')
        ax.set_title('Hierarchy Depth vs Color Entropy')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        # Plot 2: Cord count vs unique colors
        ax = axes[0, 1]
        scatter = ax.scatter(data['cord_count'], data['unique_color_count'],
                           c=data['summation_match_rate'], cmap='RdYlGn',
                           alpha=0.6, s=30)
        ax.set_xlabel('Cord Count')
        ax.set_ylabel('Unique Color Count')
        ax.set_title('Cord Count vs Color Diversity')
        ax.set_xscale('log')
        plt.colorbar(scatter, ax=ax, label='Summation Rate')
        
        # Plot 3: Branching factor vs color transitions
        ax = axes[1, 0]
        scatter = ax.scatter(data['branching_factor'], data['color_transitions'],
                           c=data['structural_cluster'], cmap='tab10',
                           alpha=0.6, s=30)
        ax.set_xlabel('Branching Factor')
        ax.set_ylabel('Color Transitions')
        ax.set_title('Branching vs Color Transitions')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        # Plot 4: Summation rate vs color/cord ratio
        ax = axes[1, 1]
        for cluster_id in sorted(data['structural_cluster'].unique()):
            cluster_data = data[data['structural_cluster'] == cluster_id]
            ax.scatter(cluster_data['summation_match_rate'],
                      cluster_data['color_cord_ratio'],
                      label=f'Cluster {cluster_id}',
                      alpha=0.6, s=30)
        ax.set_xlabel('Summation Match Rate')
        ax.set_ylabel('Color/Cord Ratio')
        ax.set_title('Summation vs Color Usage')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "06_structure_color_correlation.png", 
                   dpi=300, bbox_inches='tight')
        print("  ✓ Saved: 06_structure_color_correlation.png")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all Phase 8 visualizations."""
        print("\n" + "=" * 80)
        print("GENERATING PHASE 8 VISUALIZATIONS")
        print("=" * 80)
        
        self.plot_cluster_distribution()
        self.plot_chromatic_features()
        self.plot_feature_importance()
        self.plot_administrative_typology()
        self.plot_model_comparison()
        self.plot_structure_color_correlation()
        
        print("\n" + "=" * 80)
        print("VISUALIZATION COMPLETE")
        print("=" * 80)
        print(f"\n✓ All plots saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("  1. 01_structural_clusters.png")
        print("  2. 02_chromatic_features.png")
        print("  3. 03_feature_importance.png")
        print("  4. 04_administrative_typology.png")
        print("  5. 05_model_comparison.png")
        print("  6. 06_structure_color_correlation.png")


def main():
    """Generate all Phase 8 visualizations."""
    visualizer = Phase8Visualizer()
    visualizer.generate_all_plots()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
