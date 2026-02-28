"""
Phase 9 Visualization: Meta-Analysis

Generates 4 visualizations from actual Phase 9 meta-analysis results.
"""

import sys
from pathlib import Path

# Add src directory to path for config import
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore

import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import json  # noqa: E402

config = get_config()
OUTPUT_DIR = config.root_dir / "visualizations" / "phase9"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def plot_information_capacity():
    """Plot information capacity metrics."""
    print("Generating information capacity metrics...")
    
    df = pd.read_csv(config.processed_dir / "phase9" / "9.1_information_capacity" / "capacity_metrics.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Numeric entropy distribution
    axes[0].hist(df['numeric_entropy_bits'], bins=30, color='steelblue', 
                 edgecolor='black', alpha=0.7)
    axes[0].axvline(df['numeric_entropy_bits'].mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {df["numeric_entropy_bits"].mean():.2f}')
    axes[0].set_xlabel('Numeric Entropy (bits)')
    axes[0].set_ylabel('Number of Khipus')
    axes[0].set_title('Numeric Information Entropy Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Panel 2: Information per cord vs structural complexity
    axes[1].scatter(df['num_cords_total'], df['info_per_cord'], 
                    alpha=0.6, c='coral', edgecolors='black', s=50)
    axes[1].set_xlabel('Number of Cords')
    axes[1].set_ylabel('Information per Cord (bits)')
    axes[1].set_title('Information Density vs. Khipu Size')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "information_capacity.png", bbox_inches='tight')
    plt.close()
    print("  [OK] Saved")


def plot_robustness_analysis():
    """Plot robustness and error sensitivity."""
    print("Generating robustness analysis...")
    
    df_metrics = pd.read_csv(config.processed_dir / "phase9" / "9.2_robustness" / "robustness_metrics.csv")
    df_error = pd.read_csv(config.processed_dir / "phase9" / "9.2_robustness" / "error_sensitivity.csv")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Robustness score distribution
    axes[0, 0].hist(df_metrics['robustness_score'], bins=30, 
                    color='mediumseagreen', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Robustness Score')
    axes[0, 0].set_ylabel('Number of Khipus')
    axes[0, 0].set_title('Robustness Score Distribution')
    axes[0, 0].grid(alpha=0.3)
    
    # Panel 2: Relative impact
    axes[0, 1].hist(df_metrics['relative_impact'], bins=30, 
                    color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Relative Impact')
    axes[0, 1].set_ylabel('Number of Khipus')
    axes[0, 1].set_title('Error Relative Impact Distribution')
    axes[0, 1].grid(alpha=0.3)
    
    # Panel 3: Error sensitivity
    axes[1, 0].scatter(df_error['num_valued_cords'], df_error['error_magnitude'],
                       alpha=0.6, c='steelblue', edgecolors='black', s=50)
    axes[1, 0].set_xlabel('Number of Valued Cords')
    axes[1, 0].set_ylabel('Error Magnitude')
    axes[1, 0].set_title('Error Sensitivity by Khipu Size')
    axes[1, 0].grid(alpha=0.3)
    
    # Panel 4: Robustness class distribution
    class_counts = df_metrics['robustness_class'].value_counts()
    axes[1, 1].bar(class_counts.index, class_counts.values,
                   color='gold', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Robustness Class')
    axes[1, 1].set_ylabel('Number of Khipus')
    axes[1, 1].set_title('Robustness Classification')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Phase 9: Robustness & Error Sensitivity Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "robustness_analysis.png", bbox_inches='tight')
    plt.close()
    print("  [OK] Saved")


def plot_stability_testing():
    """Plot stability and cross-validation results."""
    print("Generating stability test results...")
    
    with open(config.processed_dir / "phase9" / "9.9_stability" / "stability_summary.json", "r") as f:
        stability = json.load(f)
    
    with open(config.processed_dir / "phase9" / "9.9_stability" / "cross_validation_results.json", "r") as f:
        cv_results = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Stability metrics
    metrics = ['Clustering\nSeed Consensus', 'Cross-Validation\nAccuracy', 
               'Ablation\nMean Stability', 'Masking\nDrift at 10%']
    scores = [
        stability['clustering_stability']['seed_consensus'] * 100,
        cv_results['mean_accuracy'] * 100,
        stability['ablation']['mean_stability'] * 100,
        (1 - stability['masking']['drift_at_10pct']) * 100
    ]
    colors_bar = ['mediumseagreen' if s >= 95 else 'gold' if s >= 90 else 'coral' 
                  for s in scores]
    axes[0, 0].bar(range(len(metrics)), scores, color=colors_bar,
                   edgecolor='black', alpha=0.7)
    axes[0, 0].set_xticks(range(len(metrics)))
    axes[0, 0].set_xticklabels(metrics)
    axes[0, 0].set_ylabel('Stability Score (%)')
    axes[0, 0].set_title('Stability Metrics Across Tests')
    axes[0, 0].set_ylim([30, 102])
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Panel 2: Cross-validation scores
    if 'scores' in cv_results:
        scores_cv = [s * 100 for s in cv_results['scores']]
        axes[0, 1].plot(range(1, len(scores_cv) + 1), scores_cv, marker='o', 
                        linewidth=2, color='steelblue', markersize=8)
        axes[0, 1].axhline(cv_results['mean_accuracy'] * 100, color='red',
                          linestyle='--', linewidth=2, 
                          label=f'Mean: {cv_results["mean_accuracy"]*100:.1f}%')
        axes[0, 1].set_xlabel('Run')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title(f'Cross-Validation Accuracy (n={cv_results["n_runs"]})')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
    
    # Panel 3: Feature ablation results
    df_ablation = pd.read_csv(config.processed_dir / "phase9" / "9.9_stability" / "feature_ablation_results.csv")
    top_features = df_ablation.nsmallest(7, 'stability')  # Lower stability = more important
    axes[1, 0].barh(top_features['ablated_feature'], (1 - top_features['stability']) * 100,
                    color='coral', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Impact When Removed (%)')
    axes[1, 0].set_title('Feature Importance by Ablation')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Panel 4: Data masking sensitivity
    df_masking = pd.read_csv(config.processed_dir / "phase9" / "9.9_stability" / "data_masking_results.csv")
    axes[1, 1].plot(df_masking['masking_level'] * 100, 
                    (1 - df_masking['drift']) * 100,
                    marker='o', linewidth=2, color='mediumseagreen', markersize=8)
    axes[1, 1].set_xlabel('Data Masked (%)')
    axes[1, 1].set_ylabel('Stability Retained (%)')
    axes[1, 1].set_title('Data Masking Sensitivity')
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('Phase 9: Stability & Validation Testing',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stability_testing.png", bbox_inches='tight')
    plt.close()
    print("  [OK] Saved")


def plot_anomaly_taxonomy():
    """Plot anomaly detection taxonomy."""
    print("Generating anomaly taxonomy...")
    
    # Check if anomaly data exists in phase 9
    anomaly_path = config.processed_dir / "phase9" / "9.7_anomaly_taxonomy"
    if not anomaly_path.exists():
        print("  [SKIP] Anomaly taxonomy data not found")
        return
    
    # Use existing anomaly data from phase 7
    df_anomalies = pd.read_csv(config.get_processed_file("anomaly_detection_results.csv", phase=7))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Anomaly type distribution
    if 'anomaly_type' in df_anomalies.columns:
        type_counts = df_anomalies['anomaly_type'].value_counts().head(8)
        axes[0].bar(range(len(type_counts)), type_counts.values,
                    color='coral', edgecolor='black', alpha=0.7)
        axes[0].set_xticks(range(len(type_counts)))
        axes[0].set_xticklabels([t[:15] for t in type_counts.index], 
                                rotation=45, ha='right')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Anomaly Type Distribution')
        axes[0].grid(axis='y', alpha=0.3)
    
    # Panel 2: Anomaly score distribution
    axes[1].hist(df_anomalies['anomaly_score'], bins=30,
                 color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Anomaly Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Anomaly Score Distribution')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "anomaly_taxonomy.png", bbox_inches='tight')
    plt.close()
    print("  [OK] Saved")


def main():
    """Generate all Phase 9 visualizations."""
    print("=" * 80)
    print("PHASE 9 VISUALIZATION GENERATION")
    print("=" * 80)
    print()
    
    plot_information_capacity()
    plot_robustness_analysis()
    plot_stability_testing()
    plot_anomaly_taxonomy()
    
    print()
    print("=" * 80)
    print("PHASE 9 VISUALIZATIONS COMPLETE")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
