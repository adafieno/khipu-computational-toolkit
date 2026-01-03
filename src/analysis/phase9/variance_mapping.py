"""
Phase 9.5: Variance Mapping and Standardization Analysis

Maps variance across numeric, chromatic, and structural dimensions.
Identifies standardized vs. flexible design features.

Key Analyses:
- Coefficient of variation (CV) for all dimensions
- High-variance (flexible) vs. low-variance (standardized) features
- Constraint vs. flexibility trade-offs
- Signal-to-noise discrimination

References:
- Phase 2: Identified empire-wide color distribution (W: 26.8%, AB: 17.4%, MB: 14.5%, KB: 6.7%)
- Phase 5: Validated empire-wide color uniformity (p=1.00, no provenance variation)
"""
import sys
from pathlib import Path

# Add src directory to path for config import
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
from typing import Dict  # noqa: E402
import json  # noqa: E402

# Empire-standard colors identified in Phase 2 (extraction infrastructure)
# These 4 colors account for 65.4% of all color usage (36,902/56,306 records)
# Cross-khipu prevalence analysis (Phase 9.5):
#   W:  76.9% of khipus (464/603) - also boundary marker (Phase 3, Phase 5)
#   AB: 59.4% of khipus (358/603)
#   MB: 59.4% of khipus (358/603)
#   KB: 45.3% of khipus (273/603)
# K-means clustering (Phase 9.5) validates 4-color standard cluster (mean 60% prevalence)
# Gap to next color (B): 17 percentage points (45.3% → 16.9%)
EMPIRE_STANDARD_COLORS = ['W', 'AB', 'MB', 'KB']


class VarianceAnalyzer:
    """
    Analyze variance patterns across khipu design dimensions.

    NO semantic interpretation - measures only:
    - Statistical variance and CV
    - Standardization vs. flexibility
    - Design constraints
    - Feature variability patterns
    """

    def __init__(self, data_dir: Path = Path("data/processed")):
        """Initialize with data directory."""
        self.config = get_config()
        self.data_dir = self.config.processed_dir
        self.output_dir = self.data_dir / "phase9" / "9.5_variance_mapping"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("PHASE 9.5: VARIANCE MAPPING & STANDARDIZATION ANALYSIS")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}\n")

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load required datasets with correct column names."""
        print("Loading datasets...")

        data = {}

        data['numeric'] = pd.read_csv(self.config.get_processed_file("cord_numeric_values.csv", 1))
        print(f"  ✓ Numeric values: {len(data['numeric'])} records")

        data['color'] = pd.read_csv(self.config.get_processed_file("color_data.csv", 2))
        print(f"  ✓ Color data: {len(data['color'])} cords")

        data['structural'] = pd.read_csv(self.config.get_processed_file("graph_structural_features.csv", 4))
        print(f"  ✓ Structural features: {len(data['structural'])} khipus")

        # UPPERCASE columns!
        data['hierarchy'] = pd.read_csv(self.config.get_processed_file("cord_hierarchy.csv", 2))
        print(f"  ✓ Hierarchy: {len(data['hierarchy'])} cords")

        data['typology'] = pd.read_csv(self.config.get_processed_file("administrative_typology.csv", 8))
        print(f"  ✓ Administrative typology: {len(data['typology'])} khipus")

        print()
        return data

    def calculate_numeric_variance(self, numeric_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate variance metrics for numeric values.

        For each khipu:
        - Mean, std, CV of pendant values
        - Range, IQR
        - Variance class (low/medium/high)
        """
        print("Calculating numeric variance...")

        results = []

        for khipu_id in numeric_data['khipu_id'].unique():
            khipu_values = numeric_data[numeric_data['khipu_id'] == khipu_id]

            # Get valued cords only
            values = khipu_values['numeric_value'].dropna()

            if len(values) < 2:
                continue

            mean_val = values.mean()
            std_val = values.std()
            cv = std_val / mean_val if mean_val > 0 else np.nan

            # Range and IQR
            value_range = values.max() - values.min()
            q25, q75 = values.quantile(0.25), values.quantile(0.75)
            iqr = q75 - q25

            # Variance class
            if pd.isna(cv):
                variance_class = 'undefined'
            elif cv < 0.5:
                variance_class = 'low'
            elif cv < 1.5:
                variance_class = 'medium'
            else:
                variance_class = 'high'

            results.append({
                'khipu_id': khipu_id,
                'num_values': len(values),
                'mean': mean_val,
                'std': std_val,
                'cv': cv,
                'range': value_range,
                'iqr': iqr,
                'variance_class': variance_class
            })

        df_results = pd.DataFrame(results)

        print(f"  ✓ Calculated variance for {len(df_results)} khipus")
        print(f"  Mean CV: {df_results['cv'].mean():.2f}")
        print(f"  Median CV: {df_results['cv'].median():.2f}")
        print("\n  Variance distribution:")
        print(df_results['variance_class'].value_counts())

        return df_results

    def calculate_color_variance(self, color_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate variance in color usage.

        For each khipu:
        - Number of unique colors
        - Color diversity (entropy)
        - Dominant color proportion
        - Color standardization score
        """
        print("\nCalculating color variance...")

        results = []

        for khipu_id in color_data['khipu_id'].unique():
            khipu_colors = color_data[color_data['khipu_id'] == khipu_id]

            # Get color codes (excluding nan) - use color_cd_1 from color_data.csv
            colors = khipu_colors['color_cd_1'].dropna()

            if len(colors) == 0:
                continue

            # Unique colors
            unique_colors = colors.nunique()

            # Color frequency distribution
            color_counts = colors.value_counts()
            dominant_color_pct = color_counts.iloc[0] / len(colors) if len(color_counts) > 0 else 1.0

            # Color diversity (simple entropy)
            proportions = color_counts / len(colors)
            color_entropy = -np.sum(proportions * np.log2(proportions + 1e-10))

            # Standardization score: low entropy = high standardization
            max_entropy = np.log2(unique_colors) if unique_colors > 1 else 1.0
            standardization_score = 1.0 - (color_entropy / max_entropy) if max_entropy > 0 else 1.0

            # Variance class
            if unique_colors <= 2:
                variance_class = 'low'
            elif unique_colors <= 5:
                variance_class = 'medium'
            else:
                variance_class = 'high'

            results.append({
                'khipu_id': khipu_id,
                'num_cords': len(colors),
                'unique_colors': unique_colors,
                'dominant_color_pct': dominant_color_pct,
                'color_entropy': color_entropy,
                'standardization_score': standardization_score,
                'variance_class': variance_class
            })

        df_results = pd.DataFrame(results)

        print(f"  ✓ Calculated color variance for {len(df_results)} khipus")
        print(f"  Mean unique colors: {df_results['unique_colors'].mean():.1f}")
        print(f"  Mean standardization score: {df_results['standardization_score'].mean():.2f}")
        print("\n  Color variance distribution:")
        print(df_results['variance_class'].value_counts())

        return df_results

    def calculate_structural_variance(self,
                                     structural_data: pd.DataFrame,
                                     hierarchy_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate variance in structural features.

        Analyzes variability in:
        - Branching patterns
        - Depth consistency
        - Node degree distribution
        """
        print("\nCalculating structural variance...")

        results = []

        # Calculate per-khipu branching variance using hierarchy (UPPERCASE!)
        for khipu_id in hierarchy_data['KHIPU_ID'].unique():
            khipu_cords = hierarchy_data[hierarchy_data['KHIPU_ID'] == khipu_id]

            # Get structural features (lowercase khipu_id)
            struct = structural_data[structural_data['khipu_id'] == khipu_id]
            if len(struct) == 0:
                continue

            # Calculate branching variance (cords per level)
            level_counts = khipu_cords['CORD_LEVEL'].value_counts()
            branching_variance = level_counts.std() if len(level_counts) > 1 else 0
            branching_cv = branching_variance / level_counts.mean() if level_counts.mean() > 0 else 0

            # Get structural stats
            depth = struct.iloc[0]['depth']
            avg_branching = struct.iloc[0]['avg_branching']
            density = struct.iloc[0]['density']

            # Structural CV proxy: normalized std of branching
            structural_cv = branching_cv

            # Variance class
            if structural_cv < 0.3:
                variance_class = 'low'
            elif structural_cv < 0.7:
                variance_class = 'medium'
            else:
                variance_class = 'high'

            results.append({
                'khipu_id': khipu_id,  # lowercase for output
                'depth': depth,
                'avg_branching': avg_branching,
                'branching_variance': branching_variance,
                'branching_cv': branching_cv,
                'structural_cv': structural_cv,
                'density': density,
                'variance_class': variance_class
            })

        df_results = pd.DataFrame(results)

        print(f"  ✓ Calculated structural variance for {len(df_results)} khipus")
        print(f"  Mean structural CV: {df_results['structural_cv'].mean():.2f}")
        print("\n  Structural variance distribution:")
        print(df_results['variance_class'].value_counts())

        return df_results

    def analyze_empire_wide_color_conventions(self, color_data: pd.DataFrame) -> Dict:
        """
        Analyze cross-khipu color conventions (empire-wide standardization).

        Validates and measures adherence to Phase 2's identified standard colors.

        Phase 2 Finding: W, AB, MB, KB are most common (65.4% of all usage)
        Phase 5 Finding: Color semantics uniform across provenances (p=1.00)
        Phase 9.5 Contribution: Cross-khipu prevalence and conformity metrics

        Measures:
        - Prevalence of standard colors across khipus
        - Color palette overlap between khipus
        - Empire-wide convention strength
        """
        print("\nAnalyzing empire-wide color conventions...")
        print("  Validating Phase 2 standard colors: W, AB, MB, KB")

        # Top colors empire-wide (from Phase 2)
        color_counts = color_data['color_cd_1'].value_counts()
        total_records = len(color_data)
        top5_colors = color_counts.head(5)
        top5_concentration = top5_colors.sum() / total_records

        # Cross-khipu prevalence
        total_khipus = color_data['khipu_id'].nunique()
        color_prevalence = {}

        for color_code in color_counts.head(10).index:
            khipus_with_color = color_data[color_data['color_cd_1'] == color_code]['khipu_id'].nunique()
            prevalence = khipus_with_color / total_khipus
            color_prevalence[color_code] = {
                'khipu_count': int(khipus_with_color),
                'prevalence': float(prevalence),
                'usage_count': int(color_counts[color_code])
            }

        # Standard colors from Phase 2 (data-driven identification)
        standard_color_prevalence = {
            k: v for k, v in color_prevalence.items()
            if k in EMPIRE_STANDARD_COLORS
        }
        strong_conventions = len(EMPIRE_STANDARD_COLORS)

        # Empire-wide standardization score
        # Based on: (1) concentration in top colors, (2) standard color prevalence
        mean_standard_prevalence = np.mean([v['prevalence'] for v in standard_color_prevalence.values()])
        empire_standardization = (top5_concentration + mean_standard_prevalence) / 2

        conventions = {
            'source': 'Phase 2 extraction analysis (W: 26.8%, AB: 17.4%, MB: 14.5%, KB: 6.7%)',
            'validation': 'Phase 5 provenance uniformity test (p=1.00)',
            'standard_colors': EMPIRE_STANDARD_COLORS,
            'standard_colors_method': 'K-means clustering validates 4-color cluster (Phase 9.5)',
            'top5_concentration': float(top5_concentration),
            'top_colors': {k: int(v) for k, v in top5_colors.items()},
            'standard_color_prevalence': standard_color_prevalence,
            'cross_khipu_prevalence': color_prevalence,
            'strong_conventions_count': strong_conventions,
            'mean_standard_prevalence': float(mean_standard_prevalence),
            'empire_standardization_score': float(empire_standardization),
            'interpretation': 'High empire-wide standardization' if empire_standardization > 0.6 else 'Moderate standardization'
        }

        print("  ✓ Empire-wide analysis complete")
        print(f"  Standard colors (Phase 2): {', '.join(EMPIRE_STANDARD_COLORS)}")
        print(f"  Top 5 colors: {top5_concentration:.1%} of all usage")
        print(f"  Mean standard color prevalence: {mean_standard_prevalence:.1%}")
        print(f"  Empire standardization score: {empire_standardization:.2f}")

        return conventions

    def identify_standardized_features(self,
                                      numeric_var: pd.DataFrame,
                                      color_var: pd.DataFrame,
                                      structural_var: pd.DataFrame,
                                      empire_color_conventions: Dict) -> Dict:
        """
        Identify globally standardized vs. flexible features.

        Low population-level variance = standardized constraint
        High population-level variance = flexible parameter
        """
        print("\nIdentifying standardized vs. flexible features...")

        standardization = {}

        # Numeric features
        numeric_cv_mean = numeric_var['cv'].mean()
        standardization['numeric'] = {
            'mean_cv': float(numeric_cv_mean),
            'standardization': 'high' if numeric_cv_mean < 0.5 else 'medium' if numeric_cv_mean < 1.5 else 'low',
            'interpretation': 'Values are highly variable' if numeric_cv_mean > 1.5 else 'Values show moderate consistency'
        }

        # Color features - TWO dimensions
        within_khipu_std = color_var['standardization_score'].mean()
        empire_std = empire_color_conventions['empire_standardization_score']

        standardization['color'] = {
            'within_khipu_standardization': float(within_khipu_std),
            'within_khipu_interpretation': 'Colors evenly distributed within khipus' if within_khipu_std < 0.4 else 'One color dominates within khipus',
            'empire_wide_standardization': float(empire_std),
            'empire_wide_interpretation': empire_color_conventions['interpretation'],
            'mean_unique_colors': float(color_var['unique_colors'].mean()),
            'standardization': 'high' if empire_std > 0.6 else 'medium' if empire_std > 0.4 else 'low',
            'note': 'Empire-wide conventions are STRONG despite diverse within-khipu usage'
        }

        # Structural features
        structural_cv_mean = structural_var['structural_cv'].mean()
        standardization['structural'] = {
            'mean_cv': float(structural_cv_mean),
            'standardization': 'high' if structural_cv_mean < 0.3 else 'medium' if structural_cv_mean < 0.7 else 'low',
            'interpretation': 'Branching is regular' if structural_cv_mean < 0.3 else 'Branching is irregular'
        }

        print("  ✓ Standardization analysis complete")
        print(f"\n  Numeric standardization: {standardization['numeric']['standardization']}")
        print(f"  Color standardization (empire-wide): {standardization['color']['standardization']}")
        print(f"  Color within-khipu: {within_khipu_std:.2f} (evenness, not convention)")
        print(f"  Structural standardization: {standardization['structural']['standardization']}")

        return standardization

    def calculate_constraint_flexibility_score(self,
                                               numeric_var: pd.DataFrame,
                                               color_var: pd.DataFrame,
                                               structural_var: pd.DataFrame) -> pd.DataFrame:
        """
        Composite score: constraint vs. flexibility.

        High constraint = low variance across all dimensions (rigid design)
        High flexibility = high variance (adaptable design)
        """
        print("\nCalculating constraint vs. flexibility scores...")

        # Merge all variance metrics (lowercase khipu_id)
        combined = numeric_var[['khipu_id', 'cv']].rename(columns={'cv': 'numeric_cv'}).merge(
            color_var[['khipu_id', 'standardization_score']],
            on='khipu_id',
            how='outer'
        ).merge(
            structural_var[['khipu_id', 'structural_cv']],
            on='khipu_id',
            how='outer'
        )

        # Fill NaN
        combined = combined.fillna(0)

        # Normalize numeric CV (cap at 3.0)
        combined['numeric_cv_norm'] = np.clip(combined['numeric_cv'], 0, 3.0) / 3.0

        # Invert standardization score (high std = low flexibility)
        combined['color_flexibility'] = 1.0 - combined['standardization_score']

        # Normalize structural CV (cap at 2.0)
        combined['structural_cv_norm'] = np.clip(combined['structural_cv'], 0, 2.0) / 2.0

        # Composite flexibility score (0 = constrained, 1 = flexible)
        combined['flexibility_score'] = (
            0.40 * combined['numeric_cv_norm'] +
            0.30 * combined['color_flexibility'] +
            0.30 * combined['structural_cv_norm']
        )

        # Constraint score (inverse of flexibility)
        combined['constraint_score'] = 1.0 - combined['flexibility_score']

        # Classify
        combined['design_class'] = pd.cut(
            combined['flexibility_score'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Constrained', 'Balanced', 'Flexible']
        )

        print(f"  ✓ Calculated scores for {len(combined)} khipus")
        print(f"  Mean flexibility: {combined['flexibility_score'].mean():.3f}")
        print(f"  Mean constraint: {combined['constraint_score'].mean():.3f}")
        print("\n  Design class distribution:")
        print(combined['design_class'].value_counts())

        return combined

    def analyze_variance_by_type(self,
                                variance_data: pd.DataFrame,
                                typology: pd.DataFrame) -> pd.DataFrame:
        """
        Compare variance patterns across administrative types.
        """
        print("\nAnalyzing variance by administrative type...")

        # Merge with typology (lowercase khipu_id)
        by_type = variance_data.merge(
            typology[['khipu_id', 'administrative_type']],
            on='khipu_id',
            how='left'
        )

        # Aggregate by type
        type_stats = by_type.groupby('administrative_type').agg({
            'flexibility_score': ['mean', 'std'],
            'constraint_score': ['mean', 'std'],
            'numeric_cv': ['mean', 'std'],
            'standardization_score': ['mean', 'std'],
            'structural_cv': ['mean', 'std'],
            'khipu_id': 'count'
        }).round(3)

        print(f"  ✓ Analyzed {len(type_stats)} administrative types")
        print("\nTop 3 most flexible types:")
        top_flexible = type_stats.nlargest(3, ('flexibility_score', 'mean'))
        print(top_flexible[[('flexibility_score', 'mean'), ('khipu_id', 'count')]])

        print("\nTop 3 most constrained types:")
        top_constrained = type_stats.nlargest(3, ('constraint_score', 'mean'))
        print(top_constrained[[('constraint_score', 'mean'), ('khipu_id', 'count')]])

        return type_stats

    def identify_signal_vs_noise(self,
                                 numeric_var: pd.DataFrame,
                                 color_var: pd.DataFrame) -> Dict:
        """
        Distinguish meaningful variance (signal) from random noise.

        Uses consistency across khipus to identify signal.
        """
        print("\nIdentifying signal vs. noise patterns...")

        # Numeric: low within-khipu variance = signal
        # High within-khipu variance = noise or flexible encoding
        low_variance_count = int((numeric_var['cv'] < 0.5).sum())
        high_variance_count = int((numeric_var['cv'] > 1.5).sum())

        # Color: consistent palette = signal
        high_standardization_count = int((color_var['standardization_score'] > 0.7).sum())
        low_standardization_count = int((color_var['standardization_score'] < 0.3).sum())

        signal_analysis = {
            'numeric': {
                'low_variance_khipus': low_variance_count,
                'high_variance_khipus': high_variance_count,
                'signal_interpretation': f'{low_variance_count} khipus show structured numeric patterns (signal)',
                'noise_interpretation': f'{high_variance_count} khipus show high numeric variability (noise or flexibility)'
            },
            'color': {
                'high_standardization_khipus': high_standardization_count,
                'low_standardization_khipus': low_standardization_count,
                'signal_interpretation': f'{high_standardization_count} khipus use restricted color palettes (signal)',
                'noise_interpretation': f'{low_standardization_count} khipus use diverse colors (flexible encoding)'
            }
        }

        print("  ✓ Signal vs. noise analysis complete")
        print(f"\n  Numeric signal: {low_variance_count} khipus with low CV")
        print(f"  Color signal: {high_standardization_count} khipus with restricted palettes")

        return signal_analysis

    def save_results(self,
                    variance_data: pd.DataFrame,
                    numeric_var: pd.DataFrame,
                    color_var: pd.DataFrame,
                    structural_var: pd.DataFrame,
                    type_stats: pd.DataFrame,
                    standardization: Dict,
                    signal_analysis: Dict,
                    empire_color_conventions: Dict):
        """Save all Phase 9.5 results."""
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        # Composite variance metrics
        variance_data.to_csv(self.output_dir / "variance_metrics.csv", index=False)
        print(f"  ✓ variance_metrics.csv ({len(variance_data)} khipus)")

        # Individual dimension variance
        numeric_var.to_csv(self.output_dir / "numeric_variance.csv", index=False)
        print(f"  ✓ numeric_variance.csv ({len(numeric_var)} khipus)")

        color_var.to_csv(self.output_dir / "color_variance.csv", index=False)
        print(f"  ✓ color_variance.csv ({len(color_var)} khipus)")

        structural_var.to_csv(self.output_dir / "structural_variance.csv", index=False)
        print(f"  ✓ structural_variance.csv ({len(structural_var)} khipus)")

        # By administrative type
        type_stats.to_csv(self.output_dir / "variance_by_type.csv")
        print(f"  ✓ variance_by_type.csv ({len(type_stats)} types)")

        # Standardization analysis (now includes empire-wide)
        with open(self.output_dir / "standardization_analysis.json", 'w') as f:
            json.dump(standardization, f, indent=2)
        print("  ✓ standardization_analysis.json")

        # Empire-wide color conventions (NEW)
        with open(self.output_dir / "empire_color_conventions.json", 'w') as f:
            json.dump(empire_color_conventions, f, indent=2)
        print("  ✓ empire_color_conventions.json")

        # Signal vs. noise
        with open(self.output_dir / "signal_noise_analysis.json", 'w') as f:
            json.dump(signal_analysis, f, indent=2)
        print("  ✓ signal_noise_analysis.json")

        print(f"\n✓ All results saved to: {self.output_dir}")

    def run_analysis(self):
        """Execute complete 9.5 analysis pipeline."""
        # Load data
        data = self.load_data()

        # Calculate variance by dimension
        numeric_var = self.calculate_numeric_variance(data['numeric'])
        color_var = self.calculate_color_variance(data['color'])
        structural_var = self.calculate_structural_variance(
            data['structural'], data['hierarchy']
        )

        # Analyze empire-wide color conventions (NEW)
        empire_color_conventions = self.analyze_empire_wide_color_conventions(data['color'])

        # Identify standardized features (now includes empire-wide)
        standardization = self.identify_standardized_features(
            numeric_var, color_var, structural_var, empire_color_conventions
        )

        # Constraint vs. flexibility scores
        variance_data = self.calculate_constraint_flexibility_score(
            numeric_var, color_var, structural_var
        )

        # Analyze by administrative type
        type_stats = self.analyze_variance_by_type(
            variance_data, data['typology']
        )

        # Signal vs. noise
        signal_analysis = self.identify_signal_vs_noise(
            numeric_var, color_var
        )

        # Save all results
        self.save_results(
            variance_data, numeric_var, color_var, structural_var,
            type_stats, standardization, signal_analysis, empire_color_conventions
        )

        print("\n" + "=" * 80)
        print("PHASE 9.5 COMPLETE")
        print("=" * 80)

        return variance_data, numeric_var, color_var, structural_var, type_stats


def run_variance_analysis():
    """Main entry point for Phase 9.5."""
    analyzer = VarianceAnalyzer()
    variance_data, numeric_var, color_var, structural_var, type_stats = analyzer.run_analysis()
    return analyzer, variance_data, numeric_var, color_var, structural_var, type_stats


if __name__ == "__main__":
    analyzer, variance, numeric, color, structural, stats = run_variance_analysis()
