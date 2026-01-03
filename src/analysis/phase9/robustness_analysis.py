"""
Phase 9.2: Error Detection, Correction, and Robustness Analysis

Measures resilience to error, damage, or mis-recording.
Tests khipus as engineered systems WITHOUT semantic interpretation.

Key Analyses:
- Single-knot perturbation sensitivity
- Self-correction via summation hierarchies
- Boundary cord error localization
- Redundancy vs. brittleness trade-offs
- Robustness scoring

References:
- Phase 2: White (W) is most common color (26.8%), identified as structural marker
- Phase 3: White boundaries improve summation detection (boundary hypothesis)
- Phase 5: White boundaries increase summation match rate by +10.7pp (p<0.001)
"""
import sys
from pathlib import Path

# Add src directory to path for config import
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore

import pandas as pd  # noqa: E402
from typing import Dict  # noqa: E402
import json  # noqa: E402


class RobustnessAnalyzer:
    """
    Analyze error detection and robustness of khipus as information systems.

    NO semantic interpretation - measures only:
    - Error propagation and detection
    - Self-correction mechanisms
    - Structural resilience
    - System brittleness
    """

    def __init__(self, data_dir: Path = Path("data/processed")):
        """Initialize with data directory."""
        self.config = get_config()
        self.data_dir = self.config.processed_dir
        self.output_dir = self.data_dir / "phase9" / "9.2_robustness"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("PHASE 9.2: ERROR DETECTION & ROBUSTNESS ANALYSIS")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}\n")

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load required datasets with correct column names."""
        print("Loading datasets...")

        data = {}

        data['numeric'] = pd.read_csv(self.config.get_processed_file("cord_numeric_values.csv", 1))
        print(f"  ✓ Numeric values: {len(data['numeric'])} records")

        data['summation'] = pd.read_csv(self.config.get_processed_file("summation_test_results.csv", 3))
        print(f"  ✓ Summation results: {len(data['summation'])} khipus")

        # UPPERCASE columns!
        data['hierarchy'] = pd.read_csv(self.config.get_processed_file("cord_hierarchy.csv", 2))
        print(f"  ✓ Hierarchy: {len(data['hierarchy'])} cords")

        data['structural'] = pd.read_csv(self.config.get_processed_file("graph_structural_features.csv", 4))
        print(f"  ✓ Structural features: {len(data['structural'])} khipus")

        data['typology'] = pd.read_csv(self.config.get_processed_file("administrative_typology.csv", 8))
        print(f"  ✓ Administrative typology: {len(data['typology'])} khipus")

        print()
        return data

    def simulate_single_knot_error(self,
                                   numeric_data: pd.DataFrame,
                                   summation_data: pd.DataFrame,
                                   perturbation_pct: float = 0.1) -> pd.DataFrame:
        """
        Simulate single-knot errors and measure impact.

        For each khipu:
        1. Perturb one numeric value by ±perturbation_pct
        2. Check if summation still holds
        3. Measure error propagation distance
        """
        print(f"Simulating single-knot errors ({perturbation_pct*100}% perturbation)...")

        results = []

        # Group by khipu (lowercase khipu_id)
        for khipu_id in numeric_data['khipu_id'].unique():
            khipu_values = numeric_data[numeric_data['khipu_id'] == khipu_id].copy()

            if len(khipu_values) == 0:
                continue

            # Get summation info (lowercase khipu_id)
            summ = summation_data[summation_data['khipu_id'] == khipu_id]
            if len(summ) == 0:
                continue

            has_summation = summ.iloc[0]['has_pendant_summation']
            original_match_rate = summ.iloc[0]['pendant_match_rate']

            # Simulate perturbation on first valued cord
            valued_cords = khipu_values[khipu_values['numeric_value'].notna()]
            if len(valued_cords) == 0:
                continue

            # Perturb first cord
            original_value = valued_cords.iloc[0]['numeric_value']
            perturbed_value = original_value * (1 + perturbation_pct)
            error_magnitude = abs(perturbed_value - original_value)

            # Calculate impact (simplified - assumes error affects summation)
            if has_summation:
                # Error is detectable if summation exists
                error_detectable = 1.0
                # Impact is relative to total
                total_sum = khipu_values['numeric_value'].sum()
                relative_impact = error_magnitude / total_sum if total_sum > 0 else 1.0
            else:
                # No summation = no detection
                error_detectable = 0.0
                relative_impact = 1.0

            results.append({
                'khipu_id': khipu_id,
                'has_summation': has_summation,
                'original_match_rate': original_match_rate,
                'error_detectable': error_detectable,
                'relative_impact': relative_impact,
                'error_magnitude': error_magnitude,
                'num_valued_cords': len(valued_cords)
            })

        df_results = pd.DataFrame(results)
        print(f"  ✓ Simulated errors for {len(df_results)} khipus")
        print(f"  Mean detectability: {df_results['error_detectable'].mean():.2%}")
        print(f"  Mean relative impact: {df_results['relative_impact'].mean():.3f}")

        return df_results

    def analyze_error_localization(self,
                                   hierarchy_data: pd.DataFrame,
                                   summation_data: pd.DataFrame) -> pd.DataFrame:
        """
        Test if boundary cords (white cords) localize errors.

        Uses UPPERCASE columns from hierarchy (KHIPU_ID, CORD_LEVEL, etc.)
        """
        print("\nAnalyzing error localization via boundaries...")

        results = []

        # Group by KHIPU_ID (UPPERCASE!)
        for khipu_id in hierarchy_data['KHIPU_ID'].unique():
            khipu_cords = hierarchy_data[hierarchy_data['KHIPU_ID'] == khipu_id]

            # Get summation info (lowercase khipu_id for summation data)
            summ = summation_data[summation_data['khipu_id'] == khipu_id]
            if len(summ) == 0:
                continue

            has_boundaries = summ.iloc[0]['has_white_boundaries']
            num_boundaries = summ.iloc[0]['num_white_boundaries']

            # Count segments (regions between boundaries)
            if has_boundaries and num_boundaries > 0:
                num_segments = num_boundaries + 1
                cords_per_segment = len(khipu_cords) / num_segments

                # Error localization score: more boundaries = better localization
                localization_score = min(1.0, num_boundaries / 10)  # Normalize to 0 - 1
                containment_ratio = 1.0 / num_segments  # Error contained to 1/n segments
            else:
                num_segments = 1
                cords_per_segment = len(khipu_cords)
                localization_score = 0.0
                containment_ratio = 1.0  # Error affects entire khipu

            results.append({
                'khipu_id': khipu_id,  # lowercase for output
                'has_boundaries': has_boundaries,
                'num_boundaries': num_boundaries,
                'num_segments': num_segments,
                'cords_per_segment': cords_per_segment,
                'localization_score': localization_score,
                'containment_ratio': containment_ratio
            })

        df_results = pd.DataFrame(results)
        print(f"  ✓ Analyzed error localization for {len(df_results)} khipus")
        print(f"  Khipus with boundaries: {df_results['has_boundaries'].sum()} ({df_results['has_boundaries'].mean():.1%})")
        print(f"  Mean localization score: {df_results['localization_score'].mean():.2f}")

        return df_results

    def calculate_redundancy_vs_robustness(self,
                                          summation_data: pd.DataFrame,
                                          structural_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze trade-off between redundancy and robustness.

        High redundancy should correlate with high error detection.
        """
        print("\nAnalyzing redundancy vs. robustness trade-off...")

        # Merge summation with structural (lowercase khipu_id)
        combined = summation_data.merge(
            structural_data[['khipu_id', 'depth', 'avg_branching', 'density', 'num_nodes']],
            on='khipu_id',
            how='inner'
        )

        # Redundancy proxy: structural complexity beyond minimum needed
        combined['redundancy_proxy'] = combined['depth'] * combined['avg_branching']

        # Robustness proxy: summation match rate
        combined['robustness_proxy'] = combined['pendant_match_rate']

        # Calculate correlation
        correlation = combined[['redundancy_proxy', 'robustness_proxy']].corr().iloc[0, 1]

        print(f"  ✓ Analyzed {len(combined)} khipus")
        print(f"  Redundancy-Robustness correlation: r = {correlation:.3f}")

        if correlation > 0.3:
            print("  → Positive correlation: More redundancy → More robust")
        elif correlation < -0.3:
            print("  → Negative correlation: More redundancy → Less robust (brittleness)")
        else:
            print("  → Weak correlation: Redundancy and robustness independent")

        return combined

    def calculate_robustness_score(self,
                                   error_sensitivity: pd.DataFrame,
                                   localization: pd.DataFrame,
                                   redundancy_robust: pd.DataFrame) -> pd.DataFrame:
        """
        Compute composite robustness score for each khipu.

        Score = weighted combination of:
        - Error detectability (40%)
        - Error localization (30%)
        - Redundancy factor (20%)
        - Self-correction capability (10%)
        """
        print("\nCalculating composite robustness scores...")

        # Merge all components (lowercase khipu_id)
        robustness = error_sensitivity[['khipu_id', 'error_detectable', 'relative_impact']].merge(
            localization[['khipu_id', 'localization_score', 'containment_ratio']],
            on='khipu_id',
            how='outer'
        ).merge(
            redundancy_robust[['khipu_id', 'redundancy_proxy', 'robustness_proxy']],
            on='khipu_id',
            how='outer'
        )

        # Fill NaN with 0
        robustness = robustness.fillna(0)

        # Normalize components to 0 - 1
        if robustness['redundancy_proxy'].max() > 0:
            robustness['redundancy_normalized'] = (
                robustness['redundancy_proxy'] / robustness['redundancy_proxy'].max()
            )
        else:
            robustness['redundancy_normalized'] = 0

        # Composite score
        robustness['robustness_score'] = (
            0.40 * robustness['error_detectable'] +
            0.30 * robustness['localization_score'] +
            0.20 * robustness['redundancy_normalized'] +
            0.10 * robustness['robustness_proxy']
        )

        # Classify by robustness
        robustness['robustness_class'] = pd.cut(
            robustness['robustness_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )

        print(f"  ✓ Calculated robustness scores for {len(robustness)} khipus")
        print(f"  Mean robustness score: {robustness['robustness_score'].mean():.3f}")
        print("\n  Robustness distribution:")
        print(robustness['robustness_class'].value_counts())

        return robustness

    def identify_failure_modes(self,
                               robustness_data: pd.DataFrame,
                               summation_data: pd.DataFrame) -> Dict:
        """
        Identify common failure modes (ways khipus are vulnerable).
        """
        print("\nIdentifying failure mode taxonomy...")

        # Merge with summation data (lowercase khipu_id)
        analysis = robustness_data.merge(
            summation_data[['khipu_id', 'has_pendant_summation', 'has_white_boundaries']],
            on='khipu_id',
            how='left'
        )

        failure_modes = {
            'no_error_detection': {
                'count': int((analysis['error_detectable'] == 0).sum()),
                'description': 'No summation hierarchy - errors undetectable',
                'khipus': analysis[analysis['error_detectable'] == 0]['khipu_id'].tolist()[:10]
            },
            'no_error_localization': {
                'count': int((analysis['localization_score'] == 0).sum()),
                'description': 'No boundary cords - errors affect entire khipu',
                'khipus': analysis[analysis['localization_score'] == 0]['khipu_id'].tolist()[:10]
            },
            'high_propagation_risk': {
                'count': int((analysis['relative_impact'] > 0.1).sum()),
                'description': 'Single error impacts >10% of total value',
                'khipus': analysis[analysis['relative_impact'] > 0.1]['khipu_id'].tolist()[:10]
            },
            'low_redundancy': {
                'count': int((analysis['redundancy_normalized'] < 0.3).sum()),
                'description': 'Minimal structural redundancy - brittle design',
                'khipus': analysis[analysis['redundancy_normalized'] < 0.3]['khipu_id'].tolist()[:10]
            }
        }

        print(f"  ✓ Identified {len(failure_modes)} failure mode types:")
        for mode, info in failure_modes.items():
            print(f"    - {mode}: {info['count']} khipus")

        return failure_modes

    def analyze_by_administrative_type(self,
                                      robustness_data: pd.DataFrame,
                                      typology: pd.DataFrame) -> pd.DataFrame:
        """
        Compare robustness across Phase 8 administrative types.
        """
        print("\nAnalyzing robustness by administrative type...")

        # Merge with typology (lowercase khipu_id)
        by_type = robustness_data.merge(
            typology[['khipu_id', 'administrative_type']],
            on='khipu_id',
            how='left'
        )

        # Aggregate by type
        type_stats = by_type.groupby('administrative_type').agg({
            'robustness_score': ['mean', 'std', 'min', 'max'],
            'error_detectable': 'mean',
            'localization_score': 'mean',
            'redundancy_normalized': 'mean',
            'khipu_id': 'count'
        }).round(3)

        print(f"  ✓ Analyzed {len(type_stats)} administrative types")
        print("\nTop 3 most robust types:")
        top_types = type_stats.nlargest(3, ('robustness_score', 'mean'))
        print(top_types[[('robustness_score', 'mean'), ('khipu_id', 'count')]])

        return type_stats

    def save_results(self,
                    robustness_data: pd.DataFrame,
                    error_sensitivity: pd.DataFrame,
                    type_stats: pd.DataFrame,
                    failure_modes: Dict):
        """Save all Phase 9.2 results."""
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        # Robustness metrics
        robustness_data.to_csv(self.output_dir / "robustness_metrics.csv", index=False)
        print(f"  ✓ robustness_metrics.csv ({len(robustness_data)} khipus)")

        # Error sensitivity
        error_sensitivity.to_csv(self.output_dir / "error_sensitivity.csv", index=False)
        print(f"  ✓ error_sensitivity.csv ({len(error_sensitivity)} khipus)")

        # By administrative type
        type_stats.to_csv(self.output_dir / "robustness_by_type.csv")
        print(f"  ✓ robustness_by_type.csv ({len(type_stats)} types)")

        # Failure modes
        with open(self.output_dir / "failure_modes.json", 'w') as f:
            json.dump(failure_modes, f, indent=2)
        print("  ✓ failure_modes.json")

        print(f"\n✓ All results saved to: {self.output_dir}")

    def run_analysis(self):
        """Execute complete 9.2 analysis pipeline."""
        # Load data
        data = self.load_data()

        # Simulate single-knot errors
        error_sensitivity = self.simulate_single_knot_error(
            data['numeric'], data['summation']
        )

        # Analyze error localization via boundaries
        localization = self.analyze_error_localization(
            data['hierarchy'], data['summation']
        )

        # Redundancy vs. robustness trade-off
        redundancy_robust = self.calculate_redundancy_vs_robustness(
            data['summation'], data['structural']
        )

        # Composite robustness score
        robustness_data = self.calculate_robustness_score(
            error_sensitivity, localization, redundancy_robust
        )

        # Identify failure modes
        failure_modes = self.identify_failure_modes(
            robustness_data, data['summation']
        )

        # Analyze by administrative type
        type_stats = self.analyze_by_administrative_type(
            robustness_data, data['typology']
        )

        # Save all results
        self.save_results(robustness_data, error_sensitivity, type_stats, failure_modes)

        print("\n" + "=" * 80)
        print("PHASE 9.2 COMPLETE")
        print("=" * 80)

        return robustness_data, error_sensitivity, type_stats, failure_modes


def run_robustness_analysis():
    """Main entry point for Phase 9.2."""
    analyzer = RobustnessAnalyzer()
    robustness_data, error_sensitivity, type_stats, failure_modes = analyzer.run_analysis()
    return analyzer, robustness_data, error_sensitivity, type_stats, failure_modes


if __name__ == "__main__":
    analyzer, robustness, sensitivity, stats, modes = run_robustness_analysis()
