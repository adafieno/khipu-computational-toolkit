"""
Phase 1 Visualization: Baseline Validation

Generates 4 core visualizations documenting numeric data extraction quality.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
OUTPUT_DIR = Path("visualizations/phase1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("data/processed/phase1/cord_numeric_values.csv")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_data():
    """Load Phase 1 baseline data."""
    print("Loading Phase 1 data...")
    df = pd.read_csv(DATA_PATH)
    return df


def plot_numeric_value_distribution(df):
    """Plot distribution of extracted numeric values (log scale)."""
    print("Generating numeric value distribution...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Filter numeric values > 0
    numeric_values = df[df['numeric_value'].notna() & (df['numeric_value'] > 0)]['numeric_value']
    
    # Linear scale
    ax1.hist(numeric_values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Numeric Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Numeric Values (Linear Scale)')
    ax1.grid(alpha=0.3)
    
    # Log scale
    ax2.hist(numeric_values, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax2.set_xlabel('Numeric Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Numeric Values (Log Scale)')
    ax2.set_yscale('log')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "numeric_value_distribution.png", bbox_inches='tight')
    plt.close()
    print("  ✓ Saved")


def plot_confidence_scores(df):
    """Plot distribution of numeric confidence scores."""
    print("Generating confidence score distribution...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate confidence per khipu (if confidence column exists)
    # For now, using presence of values as proxy
    khipu_confidence = df.groupby('khipu_id').agg({
        'numeric_value': lambda x: (x.notna().sum() / len(x))
    }).reset_index()
    khipu_confidence.columns = ['khipu_id', 'Coverage']
    
    ax.hist(khipu_confidence['Coverage'], bins=30, edgecolor='black', 
            alpha=0.7, color='mediumseagreen')
    ax.axvline(khipu_confidence['Coverage'].mean(), color='red', 
               linestyle='--', linewidth=2, label=f'Mean: {khipu_confidence["Coverage"].mean():.3f}')
    ax.set_xlabel('Numeric Value Coverage per Khipu')
    ax.set_ylabel('Number of Khipus')
    ax.set_title('Numeric Data Coverage Distribution Across Khipus')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confidence_scores.png", bbox_inches='tight')
    plt.close()
    print("  ✓ Saved")


def plot_data_coverage_heatmap(df):
    """Plot heatmap showing missing data patterns."""
    print("Generating data coverage heatmap...")
    
    # Sample khipus for visualization (all would be too dense)
    sample_khipus = df['khipu_id'].unique()[:50]
    df_sample = df[df['khipu_id'].isin(sample_khipus)]
    
    # Create presence/absence matrix
    features = ['numeric_value', 'confidence', 'num_clusters']
    pivot_data = []
    
    for khipu in sample_khipus:
        khipu_data = df_sample[df_sample['khipu_id'] == khipu]
        row = [
            khipu_data['numeric_value'].notna().mean(),
            khipu_data['confidence'].notna().mean(),
            khipu_data['num_clusters'].notna().mean()
        ]
        pivot_data.append(row)
    
    coverage_matrix = pd.DataFrame(pivot_data, 
                                   index=[f"K{i}" for i in range(len(sample_khipus))],
                                   columns=features)
    
    fig, ax = plt.subplots(figsize=(8, 12))
    sns.heatmap(coverage_matrix, cmap='RdYlGn', vmin=0, vmax=1, 
                cbar_kws={'label': 'Coverage'}, ax=ax)
    ax.set_xlabel('Features')
    ax.set_ylabel('Khipus (Sample of 50)')
    ax.set_title('Data Coverage Heatmap: Features × Khipus')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "data_coverage_heatmap.png", bbox_inches='tight')
    plt.close()
    print("  ✓ Saved")


def plot_validation_summary(df):
    """Plot summary dashboard of validation metrics."""
    print("Generating validation summary dashboard...")
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Metric 1: Total khipus
    ax1 = fig.add_subplot(gs[0, 0])
    total_khipus = df['khipu_id'].nunique()
    ax1.text(0.5, 0.5, str(total_khipus), ha='center', va='center', 
             fontsize=60, fontweight='bold', color='steelblue')
    ax1.text(0.5, 0.2, 'Khipus Analyzed', ha='center', va='center', fontsize=14)
    ax1.axis('off')
    
    # Metric 2: Total cords
    ax2 = fig.add_subplot(gs[0, 1])
    total_cords = len(df)
    ax2.text(0.5, 0.5, f"{total_cords:,}", ha='center', va='center', 
             fontsize=60, fontweight='bold', color='coral')
    ax2.text(0.5, 0.2, 'Cords Extracted', ha='center', va='center', fontsize=14)
    ax2.axis('off')
    
    # Metric 3: Coverage %
    ax3 = fig.add_subplot(gs[0, 2])
    coverage = (df['numeric_value'].notna().sum() / len(df)) * 100
    ax3.text(0.5, 0.5, f"{coverage:.1f}%", ha='center', va='center', 
             fontsize=60, fontweight='bold', color='mediumseagreen')
    ax3.text(0.5, 0.2, 'Numeric Coverage', ha='center', va='center', fontsize=14)
    ax3.axis('off')
    
    # Value range distribution
    ax4 = fig.add_subplot(gs[1, :])
    values = df[df['numeric_value'] > 0]['numeric_value']
    value_ranges = pd.cut(values, bins=[0, 10, 100, 1000, 10000, values.max()],
                          labels=['1-10', '11-100', '101-1K', '1K-10K', '>10K'])
    range_counts = value_ranges.value_counts().sort_index()
    
    ax4.bar(range_counts.index, range_counts.values, color='steelblue', edgecolor='black')
    ax4.set_xlabel('Value Range')
    ax4.set_ylabel('Number of Cords')
    ax4.set_title('Distribution of Numeric Values by Range')
    ax4.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Phase 1: Baseline Validation Summary', fontsize=16, fontweight='bold')
    
    plt.savefig(OUTPUT_DIR / "validation_summary.png", bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {OUTPUT_DIR / 'validation_summary.png'}")


def main():
    """Generate all Phase 1 visualizations."""
    print("=" * 80)
    print("PHASE 1 VISUALIZATION GENERATION")
    print("=" * 80)
    print()
    
    df = load_data()
    
    plot_numeric_value_distribution(df)
    plot_confidence_scores(df)
    plot_data_coverage_heatmap(df)
    plot_validation_summary(df)
    
    print()
    print("=" * 80)
    print("PHASE 1 VISUALIZATIONS COMPLETE")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
