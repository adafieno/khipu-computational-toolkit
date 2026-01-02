"""
Phase 9.1: Information Capacity & Efficiency Analysis

Quantifies how much information khipus encode and how efficiently.
Uses information-theoretic metrics WITHOUT semantic interpretation.

Key Metrics:
- Shannon entropy (bits per cord, per knot, per level)
- Compression efficiency
- Redundancy ratios
- Information density vs. structural depth

References:
- Phase 2: Identified 66 unique color codes (W, AB, MB, KB most common)
- Phase 5: Color diversity = 26.8% of feature importance in functional classification
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from scipy.stats import entropy as scipy_entropy
import json


class InformationCapacityAnalyzer:
    """
    Analyze information capacity and efficiency of khipus as designed artifacts.

    NO semantic interpretation - measures only:
    - How much distinguishable information exists
    - How efficiently structure encodes it
    - Compression and redundancy properties
    """

    def __init__(self, data_dir: Path = Path("data/processed")):
        """Initialize with data directory."""
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "phase9" / "9.1_information_capacity"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("PHASE 9.1: INFORMATION CAPACITY & EFFICIENCY")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}\n")

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all required datasets with correct column names."""
        print("Loading datasets...")

        data = {}

        # Load with explicit column name checking
        data['structural'] = pd.read_csv(self.data_dir / "graph_structural_features.csv")
        print(f"  ✓ Structural features: {len(data['structural'])} khipus")

        data['numeric'] = pd.read_csv(self.data_dir / "cord_numeric_values.csv")
        print(f"  ✓ Numeric values: {len(data['numeric'])} records")

        data['color'] = pd.read_csv(self.data_dir / "color_data.csv")
        print(f"  ✓ Color data: {len(data['color'])} records")

        # UPPERCASE columns!
        data['hierarchy'] = pd.read_csv(self.data_dir / "cord_hierarchy.csv")
        print(f"  ✓ Hierarchy: {len(data['hierarchy'])} cords")

        data['typology'] = pd.read_csv(self.data_dir / "phase8" / "administrative_typology.csv")
        print(f"  ✓ Administrative typology: {len(data['typology'])} khipus")

        print()
        return data

    def calculate_numeric_entropy(self, numeric_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Shannon entropy of numeric value distributions per khipu.

        Entropy measures: How many bits needed to represent the values?
        High entropy = many different values (high information)
        Low entropy = few repeated values (low information, high redundancy)
        """
        print("Calculating numeric entropy...")

        results = []

        for khipu_id in numeric_data['khipu_id'].unique():
            khipu_values = numeric_data[
                numeric_data['khipu_id'] == khipu_id
            ]['numeric_value'].dropna()

            if len(khipu_values) == 0:
                continue

            # Shannon entropy in bits
            value_counts = khipu_values.value_counts()
            probs = value_counts / value_counts.sum()
            h_numeric = scipy_entropy(probs, base=2)

            # Entropy per cord
            num_cords = len(khipu_values)
            h_per_cord = h_numeric / num_cords if num_cords > 0 else 0

            # Maximum possible entropy (uniform distribution)
            n_unique = len(value_counts)
            h_max = np.log2(n_unique) if n_unique > 1 else 0

            # Normalized entropy (0 to 1)
            h_normalized = h_numeric / h_max if h_max > 0 else 0

            results.append({
                'khipu_id': khipu_id,
                'numeric_entropy_bits': h_numeric,
                'numeric_entropy_per_cord': h_per_cord,
                'max_numeric_entropy': h_max,
                'normalized_numeric_entropy': h_normalized,
                'num_unique_values': n_unique,
                'num_cords_with_values': num_cords
            })

        df_results = pd.DataFrame(results)
        print(f"  ✓ Calculated entropy for {len(df_results)} khipus")
        print(f"  Mean entropy: {df_results['numeric_entropy_bits'].mean():.2f} bits")
        print(f"  Mean entropy per cord: {df_results['numeric_entropy_per_cord'].mean():.3f} bits/cord")

        return df_results

    def calculate_color_entropy(self, color_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Shannon entropy of color distributions per khipu.

        Measures chromatic information content WITHOUT semantic meaning.

        Phase 2 Context: 66 unique color codes identified, 4 empire-standard (W/AB/MB/KB)
        Phase 5 Finding: Color diversity correlates with functional complexity (26.8% feature importance)

        High entropy = many different colors used (high information capacity)
        Low entropy = few colors repeated (low capacity, possibly conforming to empire standards)
        """
        print("\nCalculating color entropy...")
        print("  Note: Phase 2 identified 4 empire-standard colors (W, AB, MB, KB)")

        results = []

        for khipu_id in color_data['khipu_id'].unique():
            khipu_colors = color_data[
                color_data['khipu_id'] == khipu_id
            ]['color_cd_1'].dropna()

            if len(khipu_colors) == 0:
                continue

            # Shannon entropy of color distribution
            color_counts = khipu_colors.value_counts()
            probs = color_counts / color_counts.sum()
            h_color = scipy_entropy(probs, base=2)

            # Maximum possible
            n_unique_colors = len(color_counts)
            h_max = np.log2(n_unique_colors) if n_unique_colors > 1 else 0
            h_normalized = h_color / h_max if h_max > 0 else 0

            results.append({
                'khipu_id': khipu_id,
                'color_entropy_bits': h_color,
                'max_color_entropy': h_max,
                'normalized_color_entropy': h_normalized,
                'num_unique_colors': n_unique_colors,
                'num_color_records': len(khipu_colors)
            })

        df_results = pd.DataFrame(results)
        print(f"  ✓ Calculated color entropy for {len(df_results)} khipus")
        print(f"  Mean color entropy: {df_results['color_entropy_bits'].mean():.2f} bits")

        return df_results

    def calculate_structural_entropy(self, hierarchy_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate entropy of structural patterns (branching, levels).

        Note: hierarchy_data uses UPPERCASE columns (KHIPU_ID, CORD_LEVEL, etc.)
        """
        print("\nCalculating structural entropy...")

        results = []

        # Group by KHIPU_ID (UPPERCASE!)
        for khipu_id in hierarchy_data['KHIPU_ID'].unique():
            khipu_cords = hierarchy_data[hierarchy_data['KHIPU_ID'] == khipu_id]

            # Level distribution entropy
            level_counts = khipu_cords['CORD_LEVEL'].value_counts()
            level_probs = level_counts / level_counts.sum()
            h_levels = scipy_entropy(level_probs, base=2)

            # Branching pattern entropy (children per parent)
            branching_counts = khipu_cords['ATTACHED_TO'].value_counts()
            if len(branching_counts) > 1:
                branch_probs = branching_counts / branching_counts.sum()
                h_branching = scipy_entropy(branch_probs, base=2)
            else:
                h_branching = 0

            results.append({
                'khipu_id': khipu_id,  # lowercase for output consistency
                'structural_level_entropy': h_levels,
                'structural_branching_entropy': h_branching,
                'num_levels': khipu_cords['CORD_LEVEL'].nunique(),
                'num_cords_total': len(khipu_cords)
            })

        df_results = pd.DataFrame(results)
        print(f"  ✓ Calculated structural entropy for {len(df_results)} khipus")
        print(f"  Mean level entropy: {df_results['structural_level_entropy'].mean():.2f} bits")

        return df_results

    def calculate_total_information(self,
                                   numeric_entropy: pd.DataFrame,
                                   color_entropy: pd.DataFrame,
                                   structural_entropy: pd.DataFrame) -> pd.DataFrame:
        """
        Combine all entropy sources to estimate total information capacity.

        Total information = numeric + color + structural (assuming independence)
        """
        print("\nCalculating total information capacity...")

        # Merge all entropy measurements (lowercase khipu_id)
        total = numeric_entropy.merge(color_entropy, on='khipu_id', how='outer')
        total = total.merge(structural_entropy, on='khipu_id', how='outer')

        # Fill NaN with 0 (khipus without certain data types)
        total = total.fillna(0)

        # Total information (sum of independent sources)
        total['total_information_bits'] = (
            total['numeric_entropy_bits'] +
            total['color_entropy_bits'] +
            total['structural_level_entropy'] +
            total['structural_branching_entropy']
        )

        # Information per cord (normalized by size)
        total['info_per_cord'] = total['total_information_bits'] / (
            total['num_cords_with_values'] + total['num_cords_total']
        ).replace(0, 1)

        print(f"  ✓ Total information calculated for {len(total)} khipus")
        print(f"  Mean total information: {total['total_information_bits'].mean():.2f} bits")
        print(f"  Mean info per cord: {total['info_per_cord'].mean():.3f} bits/cord")

        return total

    def calculate_redundancy(self, total_info: pd.DataFrame,
                            structural_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate redundancy ratios: actual bits vs. minimum required.

        Redundancy indicates over-engineering or error-correction capacity.
        """
        print("\nCalculating redundancy ratios...")

        # Merge with structural data (lowercase khipu_id)
        redundancy = total_info.merge(
            structural_data[['khipu_id', 'num_nodes', 'depth', 'avg_branching']],
            on='khipu_id',
            how='left'
        )

        # Theoretical minimum bits (just to represent unique states)
        redundancy['min_bits_numeric'] = np.log2(
            redundancy['num_unique_values'].replace(0, 1)
        )
        redundancy['min_bits_color'] = np.log2(
            redundancy['num_unique_colors'].replace(0, 1)
        )
        redundancy['min_bits_structure'] = np.log2(
            redundancy['num_levels'].replace(0, 1)
        )

        redundancy['total_min_bits'] = (
            redundancy['min_bits_numeric'] +
            redundancy['min_bits_color'] +
            redundancy['min_bits_structure']
        )

        # Redundancy ratio (>1 means over-engineered, <1 impossible)
        redundancy['redundancy_ratio'] = (
            redundancy['total_information_bits'] /
            redundancy['total_min_bits'].replace(0, 1)
        )

        # Compression efficiency (lower is more efficient)
        redundancy['compression_efficiency'] = (
            redundancy['total_min_bits'] /
            redundancy['total_information_bits'].replace(0, 1)
        )

        print(f"  ✓ Redundancy calculated for {len(redundancy)} khipus")
        print(f"  Mean redundancy ratio: {redundancy['redundancy_ratio'].mean():.2f}x")
        print(f"  Mean compression efficiency: {redundancy['compression_efficiency'].mean():.2f}")

        return redundancy

    def calculate_capacity_bounds(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Calculate system-wide capacity bounds.

        Lower bound: Minimum bits actually observed
        Upper bound: Maximum theoretically distinguishable states
        """
        print("\nCalculating system-wide capacity bounds...")

        # Numeric capacity
        all_numeric_values = data['numeric']['numeric_value'].dropna().unique()
        numeric_states = len(all_numeric_values)
        numeric_capacity_bits = np.log2(numeric_states) if numeric_states > 0 else 0

        # Color capacity
        all_colors = data['color']['color_cd_1'].dropna().unique()
        color_states = len(all_colors)
        color_capacity_bits = np.log2(color_states) if color_states > 0 else 0

        # Structural capacity (from Phase 4)
        max_depth = data['structural']['depth'].max()
        max_branching = data['structural']['avg_branching'].max()
        structural_capacity_bits = np.log2(max_depth * max_branching)

        # System-wide bounds
        lower_bound = numeric_capacity_bits  # Just numeric
        upper_bound = numeric_capacity_bits + color_capacity_bits + structural_capacity_bits

        bounds = {
            'numeric_states': int(numeric_states),
            'numeric_capacity_bits': float(numeric_capacity_bits),
            'color_states': int(color_states),
            'color_capacity_bits': float(color_capacity_bits),
            'max_depth': int(max_depth),
            'max_branching': float(max_branching),
            'structural_capacity_bits': float(structural_capacity_bits),
            'system_lower_bound_bits': float(lower_bound),
            'system_upper_bound_bits': float(upper_bound),
            'capacity_range': f"{lower_bound:.1f} - {upper_bound:.1f} bits"
        }

        print("  ✓ System capacity bounds:")
        print(f"    Numeric states: {bounds['numeric_states']} ({bounds['numeric_capacity_bits']:.1f} bits)")
        print(f"    Color states: {bounds['color_states']} ({bounds['color_capacity_bits']:.1f} bits)")
        print(f"    Structural capacity: {bounds['structural_capacity_bits']:.1f} bits")
        print(f"    Total range: {bounds['capacity_range']}")

        return bounds

    def analyze_by_administrative_type(self,
                                       capacity_data: pd.DataFrame,
                                       typology: pd.DataFrame) -> pd.DataFrame:
        """
        Compare information capacity across Phase 8 administrative types.
        """
        print("\nAnalyzing capacity by administrative type...")

        # Merge with typology (lowercase khipu_id)
        by_type = capacity_data.merge(
            typology[['khipu_id', 'administrative_type', 'structural_cluster']],
            on='khipu_id',
            how='left'
        )

        # Aggregate by type
        type_stats = by_type.groupby('administrative_type').agg({
            'total_information_bits': ['mean', 'std', 'min', 'max'],
            'info_per_cord': ['mean', 'std'],
            'redundancy_ratio': ['mean', 'std'],
            'compression_efficiency': ['mean', 'std'],
            'khipu_id': 'count'
        }).round(3)

        print(f"  ✓ Analyzed {len(type_stats)} administrative types")
        print("\nTop 3 types by mean information:")
        top_types = type_stats.nlargest(3, ('total_information_bits', 'mean'))
        print(top_types[[('total_information_bits', 'mean'), ('khipu_id', 'count')]])

        return type_stats

    def save_results(self,
                    capacity_data: pd.DataFrame,
                    type_stats: pd.DataFrame,
                    bounds: Dict):
        """Save all Phase 9.1 results."""
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        # Per-khipu capacity metrics
        capacity_data.to_csv(self.output_dir / "capacity_metrics.csv", index=False)
        print(f"  ✓ capacity_metrics.csv ({len(capacity_data)} khipus)")

        # By administrative type
        type_stats.to_csv(self.output_dir / "efficiency_by_type.csv")
        print(f"  ✓ efficiency_by_type.csv ({len(type_stats)} types)")

        # System-wide bounds
        with open(self.output_dir / "capacity_bounds.json", 'w') as f:
            json.dump(bounds, f, indent=2)
        print("  ✓ capacity_bounds.json")

        print(f"\n✓ All results saved to: {self.output_dir}")

    def run_analysis(self):
        """Execute complete 9.1 analysis pipeline."""
        # Load data
        data = self.load_data()

        # Calculate entropy by modality
        numeric_entropy = self.calculate_numeric_entropy(data['numeric'])
        color_entropy = self.calculate_color_entropy(data['color'])
        structural_entropy = self.calculate_structural_entropy(data['hierarchy'])

        # Combine into total information
        total_info = self.calculate_total_information(
            numeric_entropy, color_entropy, structural_entropy
        )

        # Calculate redundancy
        redundancy = self.calculate_redundancy(total_info, data['structural'])

        # System-wide bounds
        bounds = self.calculate_capacity_bounds(data)

        # Analyze by administrative type
        type_stats = self.analyze_by_administrative_type(redundancy, data['typology'])

        # Save all results
        self.save_results(redundancy, type_stats, bounds)

        print("\n" + "=" * 80)
        print("PHASE 9.1 COMPLETE")
        print("=" * 80)

        return redundancy, type_stats, bounds


def run_information_capacity_analysis():
    """Main entry point for Phase 9.1."""
    analyzer = InformationCapacityAnalyzer()
    capacity_data, type_stats, bounds = analyzer.run_analysis()
    return analyzer, capacity_data, type_stats, bounds


if __name__ == "__main__":
    analyzer, capacity_data, type_stats, bounds = run_information_capacity_analysis()
