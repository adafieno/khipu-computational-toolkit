"""
Phase 9.6: Boundary Phenomena Analysis

Characterizes what happens at structural boundaries within khipus:
- White cord boundaries (from Phase 2/3/5)
- Level transitions
- Summation points
- Numeric discontinuities

Source References:
- Phase 2: White cord identification (26.8% prevalence)
- Phase 3: White cords improve summation +10.7pp (p<0.001, d=0.43)
- Phase 5: White boundaries uniform across provenances (p=1.00)
"""
import sys
from pathlib import Path

# Add src directory to path for config import
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore

import json  # noqa: E402
from typing import Dict  # noqa: E402

import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


class BoundaryPhenomenaAnalyzer:
    """Analyzes boundary phenomena in khipu structures."""

    def __init__(self, data_dir: str = "data/processed"):
        self.config = get_config()
        self.data_dir = self.config.processed_dir
        self.db_path = self.config.get_database_path()
        self.output_dir = self.data_dir / "phase9" / "9.6_boundary_phenomena"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 2/3/5 validated colors
        self.BOUNDARY_COLORS = ['W', 'AB', 'MB', 'KB']  # Empire-standard colors
        self.PRIMARY_BOUNDARY = 'W'  # White cords are primary boundaries

    def load_data(self) -> None:
        """Load required datasets from CSV files."""
        print("Loading datasets...")

        # Load hierarchy with positions
        hierarchy_file = self.config.get_processed_file("cord_hierarchy.csv", 2)
        self.hierarchy = pd.read_csv(hierarchy_file)
        # Normalize column names
        self.hierarchy = self.hierarchy.rename(columns={
            'KHIPU_ID': 'khipu_name',
            'CORD_ID': 'cord_name',
            'ATTACHED_TO': 'parent_cord',
            'CORD_ORDINAL': 'position',
            'CORD_LEVEL': 'depth'
        })
        # Keep num_knots from hierarchy for analysis
        if 'num_knots' not in self.hierarchy.columns:
            self.hierarchy['num_knots'] = 0
        print(f"  ✓ Hierarchy: {len(self.hierarchy)} cords")

        # Load color data
        color_file = self.config.get_processed_file("color_data.csv", 2)
        self.colors = pd.read_csv(color_file)
        # Normalize column names
        self.colors = self.colors.rename(columns={
            'khipu_id': 'khipu_name',
            'cord_id': 'cord_name',
            'color_cd_1': 'main_color'
        })
        print(f"  ✓ Colors: {len(self.colors)} records")

        # Load numeric values
        numeric_file = self.config.get_processed_file("cord_numeric_values.csv", 1)
        self.numeric = pd.read_csv(numeric_file)
        # Normalize column names
        if 'khipu_id' in self.numeric.columns:
            self.numeric = self.numeric.rename(columns={
                'khipu_id': 'khipu_name',
                'cord_id': 'cord_name',
                'numeric_value': 'pendant_value'
            })
        # Get num_knots from hierarchy since it's not in numeric file
        print(f"  ✓ Numeric: {len(self.numeric)} records")

        # Load summation results (Phase 3)
        summation_file = self.config.get_processed_file("summation_test_results.csv", 3)
        if summation_file.exists():
            self.summation = pd.read_csv(summation_file)
            # Normalize column names
            if 'khipu_id' in self.summation.columns:
                self.summation = self.summation.rename(columns={'khipu_id': 'khipu_name'})
            print(f"  ✓ Summation: {len(self.summation)} tests")
        else:
            self.summation = pd.DataFrame()
            print("  ⚠ Summation data not found")

        # Merge datasets (colors have more records than hierarchy, so use left join from hierarchy)
        self.data = self.hierarchy.merge(
            self.colors[['khipu_name', 'cord_name', 'main_color']].drop_duplicates(subset=['khipu_name', 'cord_name']),
            on=['khipu_name', 'cord_name'], how='left'
        ).merge(
            self.numeric[['khipu_name', 'cord_name', 'pendant_value']],
            on=['khipu_name', 'cord_name'], how='left'
        )

    def identify_boundaries(self) -> pd.DataFrame:
        """Identify all boundary types in the dataset."""
        print("\nIdentifying boundaries...")

        boundaries = []

        for khipu_name in self.data['khipu_name'].unique():
            khipu_data = self.data[self.data['khipu_name'] == khipu_name].copy()
            khipu_data = khipu_data.sort_values('position')

            for idx, row in khipu_data.iterrows():
                boundary_record = {
                    'khipu_name': khipu_name,
                    'cord_name': row['cord_name'],
                    'position': row['position'],
                    'depth': row['depth'],
                    'num_knots': row['num_knots'],
                    'is_white_boundary': row['main_color'] == self.PRIMARY_BOUNDARY,
                    'is_standard_color': row['main_color'] in self.BOUNDARY_COLORS,
                    'is_level_transition': False,
                    'is_summation_point': False,
                    'color': row['main_color']
                }

                # Check for level transitions (depth changes)
                prev_cords = khipu_data[khipu_data['position'] < row['position']]
                if len(prev_cords) > 0:
                    prev_depth = prev_cords.iloc[-1]['depth']
                    if row['depth'] != prev_depth:
                        boundary_record['is_level_transition'] = True

                # Check for summation points
                if not self.summation.empty:
                    # Summation file is at khipu level, just check if khipu has summation
                    is_summation = (
                        (self.summation['khipu_name'] == khipu_name) &
                        (self.summation['has_pendant_summation'] )
                    ).any()
                    # Mark white cords in summation khipus as summation points
                    boundary_record['is_summation_point'] = is_summation and boundary_record['is_white_boundary']

                boundaries.append(boundary_record)

        boundaries_df = pd.DataFrame(boundaries)

        # Calculate boundary scores
        boundaries_df['boundary_score'] = (
            0.4 * boundaries_df['is_white_boundary'].astype(float) +
            0.3 * boundaries_df['is_summation_point'].astype(float) +
            0.2 * boundaries_df['is_level_transition'].astype(float) +
            0.1 * boundaries_df['is_standard_color'].astype(float)
        )

        # Classify as boundary if score >= 0.3
        boundaries_df['is_boundary'] = boundaries_df['boundary_score'] >= 0.3

        n_boundaries = boundaries_df['is_boundary'].sum()
        n_white = boundaries_df['is_white_boundary'].sum()
        n_level = boundaries_df['is_level_transition'].sum()
        n_summation = boundaries_df['is_summation_point'].sum()

        print(f"  ✓ Total boundaries identified: {n_boundaries}/{len(boundaries_df)} ({100*n_boundaries/len(boundaries_df):.1f}%)")
        print(f"    - White boundaries: {n_white}")
        print(f"    - Level transitions: {n_level}")
        print(f"    - Summation points: {n_summation}")

        return boundaries_df

    def analyze_knot_density(self, boundaries_df: pd.DataFrame) -> Dict:
        """Compare knot density at boundaries vs interior."""
        print("\nAnalyzing knot density at boundaries...")

        # Use num_knots from hierarchy (already in boundaries_df after merge)
        boundary_knots = boundaries_df[boundaries_df['is_boundary']]['num_knots'].dropna()
        interior_knots = boundaries_df[~boundaries_df['is_boundary']]['num_knots'].dropna()

        if len(boundary_knots) == 0 or len(interior_knots) == 0:
            print("  ⚠ Insufficient data for density analysis")
            return {}

        # Statistical test
        t_stat, p_val = stats.ttest_ind(boundary_knots, interior_knots)

        results = {
            'boundary_mean_knots': float(boundary_knots.mean()),
            'boundary_std_knots': float(boundary_knots.std()),
            'interior_mean_knots': float(interior_knots.mean()),
            'interior_std_knots': float(interior_knots.std()),
            'density_ratio': float(boundary_knots.mean() / interior_knots.mean()) if interior_knots.mean() > 0 else 0,
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'significant': bool(p_val < 0.05),
            'n_boundary': int(len(boundary_knots)),
            'n_interior': int(len(interior_knots))
        }

        print(f"  ✓ Boundary mean knots: {results['boundary_mean_knots']:.2f} ± {results['boundary_std_knots']:.2f}")
        print(f"  ✓ Interior mean knots: {results['interior_mean_knots']:.2f} ± {results['interior_std_knots']:.2f}")
        print(f"  ✓ Density ratio: {results['density_ratio']:.2f}x")
        print(f"  ✓ Statistical test: t={results['t_statistic']:.3f}, p={results['p_value']:.4f}")

        return results

    def analyze_color_transitions(self, boundaries_df: pd.DataFrame) -> Dict:
        """Analyze color transitions at boundaries vs interior."""
        print("\nAnalyzing color transitions...")

        transitions = []

        for khipu_name in boundaries_df['khipu_name'].unique():
            khipu_data = boundaries_df[boundaries_df['khipu_name'] == khipu_name].copy()
            khipu_data = khipu_data.sort_values('position')

            colors = khipu_data['color'].values
            is_boundary = khipu_data['is_boundary'].values

            for i in range(len(colors) - 1):
                if pd.notna(colors[i]) and pd.notna(colors[i + 1]):
                    has_transition = colors[i] != colors[i + 1]
                    transitions.append({
                        'khipu_name': khipu_name,
                        'position': i,
                        'has_transition': has_transition,
                        'at_boundary': is_boundary[i] or is_boundary[i + 1],
                        'from_color': colors[i],
                        'to_color': colors[i + 1]
                    })

        transitions_df = pd.DataFrame(transitions)

        if len(transitions_df) == 0:
            print("  ⚠ No transitions found")
            return {}

        # Calculate alignment
        boundary_transitions = transitions_df[transitions_df['at_boundary']]['has_transition'].sum()
        total_transitions = transitions_df['has_transition'].sum()

        boundary_transition_rate = transitions_df[transitions_df['at_boundary']]['has_transition'].mean()
        interior_transition_rate = transitions_df[~transitions_df['at_boundary']]['has_transition'].mean()

        # Chi-square test
        contingency = pd.crosstab(
            transitions_df['at_boundary'],
            transitions_df['has_transition']
        )
        chi2, p_val, _, _ = stats.chi2_contingency(contingency)

        results = {
            'total_transitions': int(total_transitions),
            'boundary_transitions': int(boundary_transitions),
            'alignment_ratio': float(boundary_transitions / total_transitions) if total_transitions > 0 else 0,
            'boundary_transition_rate': float(boundary_transition_rate),
            'interior_transition_rate': float(interior_transition_rate),
            'chi2_statistic': float(chi2),
            'p_value': float(p_val),
            'significant': bool(p_val < 0.05)
        }

        print(f"  ✓ Total color transitions: {results['total_transitions']}")
        print(f"  ✓ Transitions at boundaries: {results['boundary_transitions']} ({100*results['alignment_ratio']:.1f}%)")
        print(f"  ✓ Boundary transition rate: {100*results['boundary_transition_rate']:.1f}%")
        print(f"  ✓ Interior transition rate: {100*results['interior_transition_rate']:.1f}%")
        print(f"  ✓ Chi-square test: χ²={results['chi2_statistic']:.3f}, p={results['p_value']:.4f}")

        return results

    def analyze_numeric_discontinuities(self, boundaries_df: pd.DataFrame) -> Dict:
        """Detect numeric value resets at boundaries."""
        print("\nAnalyzing numeric discontinuities...")

        # Merge with numeric values
        analysis_data = boundaries_df.merge(
            self.numeric[['khipu_name', 'cord_name', 'pendant_value']],
            on=['khipu_name', 'cord_name'],
            how='left'
        )

        discontinuities = []

        for khipu_name in analysis_data['khipu_name'].unique():
            khipu_data = analysis_data[analysis_data['khipu_name'] == khipu_name].copy()
            khipu_data = khipu_data.sort_values('position')

            values = khipu_data['pendant_value'].values
            is_boundary = khipu_data['is_boundary'].values

            for i in range(1, len(values)):
                if pd.notna(values[i]) and pd.notna(values[i - 1]):
                    diff = abs(values[i] - values[i - 1])
                    discontinuities.append({
                        'khipu_name': khipu_name,
                        'position': i,
                        'discontinuity': diff,
                        'at_boundary': is_boundary[i],
                        'value_before': values[i - 1],
                        'value_after': values[i]
                    })

        disc_df = pd.DataFrame(discontinuities)

        if len(disc_df) == 0:
            print("  ⚠ No discontinuities calculated")
            return {}

        # Compare boundary vs interior discontinuities
        boundary_disc = disc_df[disc_df['at_boundary']]['discontinuity']
        interior_disc = disc_df[~disc_df['at_boundary']]['discontinuity']

        if len(boundary_disc) == 0 or len(interior_disc) == 0:
            print("  ⚠ Insufficient data for discontinuity analysis")
            return {}

        # Statistical test
        u_stat, p_val = stats.mannwhitneyu(boundary_disc, interior_disc, alternative='two-sided')

        results = {
            'boundary_mean_disc': float(boundary_disc.mean()),
            'boundary_median_disc': float(boundary_disc.median()),
            'interior_mean_disc': float(interior_disc.mean()),
            'interior_median_disc': float(interior_disc.median()),
            'discontinuity_ratio': float(boundary_disc.mean() / interior_disc.mean()) if interior_disc.mean() > 0 else 0,
            'u_statistic': float(u_stat),
            'p_value': float(p_val),
            'significant': bool(p_val < 0.05),
            'n_boundary': int(len(boundary_disc)),
            'n_interior': int(len(interior_disc))
        }

        print(f"  ✓ Boundary mean discontinuity: {results['boundary_mean_disc']:.2f} (median: {results['boundary_median_disc']:.2f})")
        print(f"  ✓ Interior mean discontinuity: {results['interior_mean_disc']:.2f} (median: {results['interior_median_disc']:.2f})")
        print(f"  ✓ Discontinuity ratio: {results['discontinuity_ratio']:.2f}x")
        print(f"  ✓ Mann-Whitney U test: U={results['u_statistic']:.0f}, p={results['p_value']:.4f}")

        return results

    def classify_boundary_types(self, boundaries_df: pd.DataFrame) -> Dict:
        """Classify boundaries into functional types."""
        print("\nClassifying boundary types...")

        boundaries = boundaries_df[boundaries_df['is_boundary']].copy()

        # Define typology
        def classify_boundary(row):
            if row['is_white_boundary'] and row['is_summation_point']:
                return 'summarizing'
            elif row['is_white_boundary'] and not row['is_summation_point']:
                return 'separating'
            elif row['is_level_transition']:
                return 'structural'
            elif row['is_standard_color']:
                return 'marking'
            else:
                return 'other'

        boundaries['boundary_type'] = boundaries.apply(classify_boundary, axis=1)

        # Count types
        type_counts = boundaries['boundary_type'].value_counts()

        typology = {
            'total_boundaries': int(len(boundaries)),
            'type_distribution': {
                'summarizing': int(type_counts.get('summarizing', 0)),
                'separating': int(type_counts.get('separating', 0)),
                'structural': int(type_counts.get('structural', 0)),
                'marking': int(type_counts.get('marking', 0)),
                'other': int(type_counts.get('other', 0))
            },
            'type_proportions': {
                'summarizing': float(type_counts.get('summarizing', 0) / len(boundaries)),
                'separating': float(type_counts.get('separating', 0) / len(boundaries)),
                'structural': float(type_counts.get('structural', 0) / len(boundaries)),
                'marking': float(type_counts.get('marking', 0) / len(boundaries)),
                'other': float(type_counts.get('other', 0) / len(boundaries))
            }
        }

        print(f"  ✓ Total boundaries: {typology['total_boundaries']}")
        print("  ✓ Type distribution:")
        for btype, count in typology['type_distribution'].items():
            pct = 100 * typology['type_proportions'][btype]
            print(f"    - {btype}: {count} ({pct:.1f}%)")

        return typology, boundaries

    def save_results(self, boundaries_df: pd.DataFrame, knot_density: Dict,
                    color_transitions: Dict, discontinuities: Dict,
                    typology: Dict) -> None:
        """Save all analysis results."""
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        # Save boundary catalog
        boundaries_df.to_csv(self.output_dir / "boundary_catalog.csv", index=False)
        print(f"  ✓ boundary_catalog.csv ({len(boundaries_df)} cords)")

        # Save knot density results
        with open(self.output_dir / "knot_density_analysis.json", 'w') as f:
            json.dump(knot_density, f, indent=2)
        print("  ✓ knot_density_analysis.json")

        # Save color transition results
        with open(self.output_dir / "color_transition_analysis.json", 'w') as f:
            json.dump(color_transitions, f, indent=2)
        print("  ✓ color_transition_analysis.json")

        # Save discontinuity results
        with open(self.output_dir / "numeric_discontinuities.json", 'w') as f:
            json.dump(discontinuities, f, indent=2)
        print("  ✓ numeric_discontinuities.json")

        # Save boundary typology
        with open(self.output_dir / "boundary_typology.json", 'w') as f:
            json.dump(typology, f, indent=2)
        print("  ✓ boundary_typology.json")

        # Summary report
        summary = {
            'analysis': 'Phase 9.6: Boundary Phenomena Analysis',
            'date': '2026 - 01 - 01',
            'source_references': {
                'phase2': 'White cord identification (26.8% prevalence)',
                'phase3': 'White cords improve summation +10.7pp (p<0.001, d=0.43)',
                'phase5': 'White boundaries uniform across provenances (p=1.00)'
            },
            'total_cords': int(len(boundaries_df)),
            'total_boundaries': int(boundaries_df['is_boundary'].sum()),
            'boundary_prevalence': float(boundaries_df['is_boundary'].mean()),
            'knot_density_analysis': knot_density,
            'color_transition_analysis': color_transitions,
            'numeric_discontinuity_analysis': discontinuities,
            'boundary_typology': typology
        }

        with open(self.output_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print("  ✓ analysis_summary.json")

        print(f"\n✓ All results saved to: {self.output_dir}")

    def run_analysis(self) -> None:
        """Execute complete boundary phenomena analysis."""
        print("=" * 80)
        print("PHASE 9.6: BOUNDARY PHENOMENA ANALYSIS")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")

        # Load data
        self.load_data()

        # Identify boundaries
        boundaries_df = self.identify_boundaries()

        # Analyze knot density
        knot_density = self.analyze_knot_density(boundaries_df)

        # Analyze color transitions
        color_transitions = self.analyze_color_transitions(boundaries_df)

        # Analyze numeric discontinuities
        discontinuities = self.analyze_numeric_discontinuities(boundaries_df)

        # Classify boundary types
        typology, _ = self.classify_boundary_types(boundaries_df)

        # Save results
        self.save_results(boundaries_df, knot_density, color_transitions,
                         discontinuities, typology)

        print("\n" + "=" * 80)
        print("PHASE 9.6 COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    analyzer = BoundaryPhenomenaAnalyzer()
    analyzer.run_analysis()
