"""
Phase 9.3: Cognitive Load & Usability Modeling

Estimates cognitive demands imposed on khipu users:
- Working memory load (Miller's Law: 7±2 items)
- Hierarchy traversal costs
- Visual parsing complexity
- Sequential vs parallel interpretability
- Specialist operation thresholds

Integrates with:
- Phase 9.1: Information capacity
- Phase 9.6: Boundary phenomena (color transitions)
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


class CognitiveLoadAnalyzer:
    """Analyzes cognitive load and usability of khipu structures."""

    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "phase9" / "9.3_cognitive_load"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Miller's Law: 7±2 items in working memory
        self.WORKING_MEMORY_CAPACITY = 7

    def load_data(self) -> None:
        """Load required datasets."""
        print("Loading datasets...")

        # Load structural features (khipu_id, num_nodes, depth, avg_branching, width)
        struct_file = self.data_dir / "graph_structural_features.csv"
        self.structural = pd.read_csv(struct_file)
        print(f"  ✓ Structural features: {len(self.structural)} khipus")

        # Load hierarchy (KHIPU_ID, CORD_ID, CORD_LEVEL, ATTACHED_TO)
        hierarchy_file = self.data_dir / "cord_hierarchy.csv"
        self.hierarchy = pd.read_csv(hierarchy_file)
        print(f"  ✓ Hierarchy: {len(self.hierarchy)} cords")

        # Load color data (khipu_id, cord_id, color_cd_1)
        color_file = self.data_dir / "color_data.csv"
        self.colors = pd.read_csv(color_file)
        print(f"  ✓ Color data: {len(self.colors)} records")

        # Load Phase 9.1 capacity metrics (khipu_id, color_entropy_bits, structural_level_entropy)
        capacity_file = self.data_dir / "phase9" / "9.1_information_capacity" / "capacity_metrics.csv"
        if capacity_file.exists():
            self.capacity = pd.read_csv(capacity_file)
            print(f"  ✓ Information capacity: {len(self.capacity)} khipus")
        else:
            self.capacity = pd.DataFrame()
            print("  ⚠ Information capacity data not found")

        # Load Phase 9.6 boundary data (khipu_id implied from boundary_catalog)
        boundary_file = self.data_dir / "phase9" / "9.6_boundary_phenomena" / "boundary_catalog.csv"
        if boundary_file.exists():
            self.boundaries = pd.read_csv(boundary_file)
            print(f"  ✓ Boundary data: {len(self.boundaries)} cords")
        else:
            self.boundaries = pd.DataFrame()
            print("  ⚠ Boundary data not found")

    def calculate_working_memory_load(self) -> pd.DataFrame:
        """Calculate working memory load based on hierarchy structure."""
        print("\nCalculating working memory load...")

        wm_loads = []

        for khipu_id in self.structural['khipu_id'].unique():
            khipu_struct = self.structural[self.structural['khipu_id'] == khipu_id].iloc[0]
            khipu_hierarchy = self.hierarchy[self.hierarchy['KHIPU_ID'] == khipu_id]

            # Base load: depth * log(branching) + cords/chunking_factor
            depth = khipu_struct['depth']
            branching = khipu_struct['avg_branching']
            num_cords = len(khipu_hierarchy)

            # Miller's Law chunking (groups of 7)
            chunks_required = num_cords / self.WORKING_MEMORY_CAPACITY

            # Working memory load model
            wm_load = (
                depth * np.log2(branching + 1) +  # Hierarchy navigation cost
                chunks_required  # Number of chunks to maintain
            )

            # Account for color boundaries (reduce load through segmentation)
            if not self.boundaries.empty:
                khipu_boundaries = self.boundaries[
                    (self.boundaries['khipu_name'] == khipu_id) &
                    (self.boundaries['is_boundary'] )
                ]
                n_boundaries = len(khipu_boundaries)

                # Boundaries reduce cognitive load by enabling chunking
                if n_boundaries > 0:
                    boundary_reduction = min(0.3 * np.log2(n_boundaries + 1), wm_load * 0.5)
                    wm_load = wm_load - boundary_reduction
                else:
                    boundary_reduction = 0
            else:
                n_boundaries = 0
                boundary_reduction = 0

            # Exceeds working memory capacity?
            exceeds_capacity = chunks_required > self.WORKING_MEMORY_CAPACITY

            wm_loads.append({
                'khipu_id': khipu_id,
                'wm_load': wm_load,
                'chunks_required': chunks_required,
                'n_boundaries': n_boundaries,
                'boundary_reduction': boundary_reduction,
                'exceeds_capacity': exceeds_capacity,
                'depth': depth,
                'num_cords': num_cords,
                'avg_branching': branching
            })

        wm_df = pd.DataFrame(wm_loads)

        # Statistics
        mean_load = wm_df['wm_load'].mean()
        exceeds_rate = wm_df['exceeds_capacity'].mean()

        print(f"  ✓ Analyzed {len(wm_df)} khipus")
        print(f"  ✓ Mean working memory load: {mean_load:.2f} chunks")
        print(f"  ✓ Exceeds capacity: {100*exceeds_rate:.1f}% of khipus")
        print(f"  ✓ Load range: {wm_df['wm_load'].min():.2f} - {wm_df['wm_load'].max():.2f}")

        return wm_df

    def calculate_traversal_costs(self) -> pd.DataFrame:
        """Calculate hierarchy traversal and navigation costs."""
        print("\nCalculating traversal costs...")

        traversal_costs = []

        for khipu_id in self.hierarchy['KHIPU_ID'].unique():
            khipu_hierarchy = self.hierarchy[self.hierarchy['KHIPU_ID'] == khipu_id]

            # Calculate total path length (sum of depths)
            total_path_length = khipu_hierarchy['CORD_LEVEL'].sum()

            # Count context switches (level changes)
            khipu_sorted = khipu_hierarchy.sort_values('CORD_ORDINAL')
            level_diffs = khipu_sorted['CORD_LEVEL'].diff().abs()
            context_switches = level_diffs[level_diffs > 0].sum()

            # Calculate average path to leaf
            leaf_depths = khipu_hierarchy[khipu_hierarchy['CORD_LEVEL'] == khipu_hierarchy['CORD_LEVEL'].max()]['CORD_LEVEL']
            avg_leaf_depth = leaf_depths.mean() if len(leaf_depths) > 0 else 0

            # Traversal cost: path length + context switches
            traversal_cost = total_path_length + context_switches

            # Normalize by number of cords
            normalized_traversal = traversal_cost / len(khipu_hierarchy) if len(khipu_hierarchy) > 0 else 0

            traversal_costs.append({
                'khipu_id': khipu_id,
                'total_path_length': total_path_length,
                'context_switches': context_switches,
                'avg_leaf_depth': avg_leaf_depth,
                'traversal_cost': traversal_cost,
                'normalized_traversal': normalized_traversal,
                'n_cords': len(khipu_hierarchy)
            })

        traversal_df = pd.DataFrame(traversal_costs)

        # Statistics
        mean_cost = traversal_df['normalized_traversal'].mean()
        mean_switches = traversal_df['context_switches'].mean()

        print(f"  ✓ Analyzed {len(traversal_df)} khipus")
        print(f"  ✓ Mean normalized traversal cost: {mean_cost:.2f}")
        print(f"  ✓ Mean context switches: {mean_switches:.1f}")

        return traversal_df

    def calculate_visual_parsing_complexity(self) -> pd.DataFrame:
        """Calculate visual parsing load from color patterns."""
        print("\nCalculating visual parsing complexity...")

        visual_complexity = []

        for khipu_id in self.colors['khipu_id'].unique():
            khipu_colors = self.colors[self.colors['khipu_id'] == khipu_id]

            # Color entropy (from Phase 9.1 if available)
            if not self.capacity.empty:
                capacity_row = self.capacity[self.capacity['khipu_id'] == khipu_id]
                if len(capacity_row) > 0:
                    color_entropy = capacity_row.iloc[0]['color_entropy_bits']
                else:
                    color_entropy = 0
            else:
                # Calculate on the fly
                color_counts = khipu_colors['color_cd_1'].value_counts()
                if len(color_counts) > 0:
                    probs = color_counts / color_counts.sum()
                    color_entropy = -sum(probs * np.log2(probs + 1e-10))
                else:
                    color_entropy = 0

            # Count color transitions (attention-switching cost)
            khipu_colors_sorted = khipu_colors.sort_values('cord_id')
            colors = khipu_colors_sorted['color_cd_1'].values
            transitions = sum(1 for i in range(len(colors) - 1) if colors[i] != colors[i + 1])
            transition_rate = transitions / len(colors) if len(colors) > 0 else 0

            # Visual load: entropy + transition frequency
            visual_load = color_entropy + transition_rate

            # Simulated time-to-locate (random target search)
            # Proportional to number of colors and transitions
            n_colors = khipu_colors['color_cd_1'].nunique()
            time_to_locate = np.log2(len(khipu_colors) + 1) * (1 + transition_rate)

            visual_complexity.append({
                'khipu_id': khipu_id,
                'color_entropy': color_entropy,
                'n_colors': n_colors,
                'transitions': transitions,
                'transition_rate': transition_rate,
                'visual_load': visual_load,
                'time_to_locate': time_to_locate,
                'n_color_records': len(khipu_colors)
            })

        visual_df = pd.DataFrame(visual_complexity)

        # Statistics
        mean_load = visual_df['visual_load'].mean()
        mean_transitions = visual_df['transition_rate'].mean()

        print(f"  ✓ Analyzed {len(visual_df)} khipus")
        print(f"  ✓ Mean visual load: {mean_load:.2f}")
        print(f"  ✓ Mean transition rate: {100*mean_transitions:.1f}%")

        return visual_df

    def calculate_parallelization_potential(self) -> pd.DataFrame:
        """Measure sequential vs parallel interpretability."""
        print("\nCalculating parallelization potential...")

        parallelization = []

        for khipu_id in self.hierarchy['KHIPU_ID'].unique():
            khipu_hierarchy = self.hierarchy[self.hierarchy['KHIPU_ID'] == khipu_id]

            # Count parallel-readable sections (same level)
            level_counts = khipu_hierarchy.groupby('CORD_LEVEL').size()
            parallel_sections = (level_counts > 1).sum()
            max_parallel_width = level_counts.max() if len(level_counts) > 0 else 0

            # Count serial dependencies (parent-child chains)
            max_depth = khipu_hierarchy['CORD_LEVEL'].max()
            serial_chain_length = max_depth + 1 if not pd.isna(max_depth) else 1

            # Parallelization potential: parallel width / serial length
            parallelization_ratio = max_parallel_width / serial_chain_length if serial_chain_length > 0 else 0

            # Fraction of cords that can be read in parallel
            total_parallel_cords = level_counts[level_counts > 1].sum() if len(level_counts[level_counts > 1]) > 0 else 0
            parallel_fraction = total_parallel_cords / len(khipu_hierarchy) if len(khipu_hierarchy) > 0 else 0

            parallelization.append({
                'khipu_id': khipu_id,
                'parallel_sections': parallel_sections,
                'max_parallel_width': max_parallel_width,
                'serial_chain_length': serial_chain_length,
                'parallelization_ratio': parallelization_ratio,
                'parallel_fraction': parallel_fraction,
                'n_cords': len(khipu_hierarchy)
            })

        parallel_df = pd.DataFrame(parallelization)

        # Statistics
        mean_ratio = parallel_df['parallelization_ratio'].mean()
        mean_fraction = parallel_df['parallel_fraction'].mean()

        print(f"  ✓ Analyzed {len(parallel_df)} khipus")
        print(f"  ✓ Mean parallelization ratio: {mean_ratio:.2f}")
        print(f"  ✓ Mean parallel fraction: {100*mean_fraction:.1f}%")

        return parallel_df

    def identify_specialist_khipus(self, wm_df: pd.DataFrame, traversal_df: pd.DataFrame,
                                   visual_df: pd.DataFrame) -> Dict:
        """Identify khipus requiring specialist knowledge."""
        print("\nIdentifying specialist-level khipus...")

        # Merge all complexity metrics
        complexity_data = wm_df[['khipu_id', 'wm_load']].merge(
            traversal_df[['khipu_id', 'normalized_traversal']],
            on='khipu_id'
        ).merge(
            visual_df[['khipu_id', 'visual_load']],
            on='khipu_id'
        )

        # Calculate overall cognitive complexity
        # Normalize each component to 0 - 1 scale
        complexity_data['wm_normalized'] = (
            (complexity_data['wm_load'] - complexity_data['wm_load'].min()) /
            (complexity_data['wm_load'].max() - complexity_data['wm_load'].min())
        )
        complexity_data['traversal_normalized'] = (
            (complexity_data['normalized_traversal'] - complexity_data['normalized_traversal'].min()) /
            (complexity_data['normalized_traversal'].max() - complexity_data['normalized_traversal'].min())
        )
        complexity_data['visual_normalized'] = (
            (complexity_data['visual_load'] - complexity_data['visual_load'].min()) /
            (complexity_data['visual_load'].max() - complexity_data['visual_load'].min())
        )

        # Overall cognitive complexity (weighted average)
        complexity_data['cognitive_complexity'] = (
            0.4 * complexity_data['wm_normalized'] +
            0.3 * complexity_data['traversal_normalized'] +
            0.3 * complexity_data['visual_normalized']
        )

        # Specialist threshold: 90th percentile
        specialist_threshold = complexity_data['cognitive_complexity'].quantile(0.9)
        complexity_data['is_specialist'] = complexity_data['cognitive_complexity'] >= specialist_threshold

        n_specialist = complexity_data['is_specialist'].sum()
        specialist_rate = n_specialist / len(complexity_data)

        print(f"  ✓ Specialist threshold: {specialist_threshold:.3f}")
        print(f"  ✓ Specialist khipus: {n_specialist} ({100*specialist_rate:.1f}%)")

        # Get specialist khipus
        specialist_khipus = complexity_data[complexity_data['is_specialist']].sort_values(
            'cognitive_complexity', ascending=False
        )

        results = {
            'specialist_threshold': float(specialist_threshold),
            'n_specialist': int(n_specialist),
            'specialist_rate': float(specialist_rate),
            'mean_complexity': float(complexity_data['cognitive_complexity'].mean()),
            'std_complexity': float(complexity_data['cognitive_complexity'].std()),
            'specialist_khipus': specialist_khipus[['khipu_id', 'cognitive_complexity',
                                                     'wm_load', 'normalized_traversal',
                                                     'visual_load']].to_dict('records')
        }

        return results, complexity_data

    def analyze_correlations(self, wm_df: pd.DataFrame, complexity_data: pd.DataFrame) -> Dict:
        """Analyze correlations between cognitive load and khipu properties."""
        print("\nAnalyzing correlations...")

        # Merge with structural features
        analysis_data = complexity_data.merge(
            self.structural[['khipu_id', 'num_nodes', 'depth', 'avg_branching']],
            on='khipu_id'
        )

        # Correlation: cognitive load vs size
        corr_size, p_size = stats.pearsonr(
            analysis_data['cognitive_complexity'],
            analysis_data['num_nodes']
        )

        # Correlation: cognitive load vs depth
        corr_depth, p_depth = stats.pearsonr(
            analysis_data['cognitive_complexity'],
            analysis_data['depth']
        )

        # Correlation: working memory vs chunking
        corr_chunks, p_chunks = stats.pearsonr(
            wm_df['wm_load'],
            wm_df['chunks_required']
        )

        results = {
            'complexity_vs_size': {
                'correlation': float(corr_size),
                'p_value': float(p_size),
                'significant': bool(p_size < 0.05)
            },
            'complexity_vs_depth': {
                'correlation': float(corr_depth),
                'p_value': float(p_depth),
                'significant': bool(p_depth < 0.05)
            },
            'wm_load_vs_chunks': {
                'correlation': float(corr_chunks),
                'p_value': float(p_chunks),
                'significant': bool(p_chunks < 0.05)
            }
        }

        print(f"  ✓ Complexity vs size: r={corr_size:.3f}, p={p_size:.4f}")
        print(f"  ✓ Complexity vs depth: r={corr_depth:.3f}, p={p_depth:.4f}")
        print(f"  ✓ WM load vs chunks: r={corr_chunks:.3f}, p={p_chunks:.4f}")

        return results

    def save_results(self, wm_df: pd.DataFrame, traversal_df: pd.DataFrame,
                    visual_df: pd.DataFrame, parallel_df: pd.DataFrame,
                    specialist_results: Dict, complexity_data: pd.DataFrame,
                    correlations: Dict) -> None:
        """Save all analysis results."""
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        # Save working memory analysis
        wm_df.to_csv(self.output_dir / "working_memory_load.csv", index=False)
        print(f"  ✓ working_memory_load.csv ({len(wm_df)} khipus)")

        # Save traversal costs
        traversal_df.to_csv(self.output_dir / "traversal_costs.csv", index=False)
        print(f"  ✓ traversal_costs.csv ({len(traversal_df)} khipus)")

        # Save visual complexity
        visual_df.to_csv(self.output_dir / "visual_complexity.csv", index=False)
        print(f"  ✓ visual_complexity.csv ({len(visual_df)} khipus)")

        # Save parallelization analysis
        parallel_df.to_csv(self.output_dir / "parallelization_potential.csv", index=False)
        print(f"  ✓ parallelization_potential.csv ({len(parallel_df)} khipus)")

        # Save cognitive complexity
        complexity_data.to_csv(self.output_dir / "cognitive_complexity.csv", index=False)
        print(f"  ✓ cognitive_complexity.csv ({len(complexity_data)} khipus)")

        # Save specialist indicators
        with open(self.output_dir / "specialist_indicators.json", 'w') as f:
            json.dump(specialist_results, f, indent=2)
        print("  ✓ specialist_indicators.json")

        # Save correlations
        with open(self.output_dir / "correlations.json", 'w') as f:
            json.dump(correlations, f, indent=2)
        print("  ✓ correlations.json")

        # Summary report
        summary = {
            'analysis': 'Phase 9.3: Cognitive Load & Usability Modeling',
            'date': '2026 - 01 - 01',
            'working_memory_capacity': self.WORKING_MEMORY_CAPACITY,
            'working_memory': {
                'mean_load': float(wm_df['wm_load'].mean()),
                'exceeds_capacity_rate': float(wm_df['exceeds_capacity'].mean()),
                'n_khipus': int(len(wm_df))
            },
            'traversal': {
                'mean_normalized_cost': float(traversal_df['normalized_traversal'].mean()),
                'mean_context_switches': float(traversal_df['context_switches'].mean()),
                'n_khipus': int(len(traversal_df))
            },
            'visual_parsing': {
                'mean_visual_load': float(visual_df['visual_load'].mean()),
                'mean_transition_rate': float(visual_df['transition_rate'].mean()),
                'n_khipus': int(len(visual_df))
            },
            'parallelization': {
                'mean_ratio': float(parallel_df['parallelization_ratio'].mean()),
                'mean_parallel_fraction': float(parallel_df['parallel_fraction'].mean()),
                'n_khipus': int(len(parallel_df))
            },
            'specialist_indicators': {k: v for k, v in specialist_results.items() if k != 'specialist_khipus'},
            'correlations': correlations
        }

        with open(self.output_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print("  ✓ analysis_summary.json")

        print(f"\n✓ All results saved to: {self.output_dir}")

    def run_analysis(self) -> None:
        """Execute complete cognitive load analysis."""
        print("=" * 80)
        print("PHASE 9.3: COGNITIVE LOAD & USABILITY MODELING")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Working memory capacity (Miller's Law): {self.WORKING_MEMORY_CAPACITY} items")

        # Load data
        self.load_data()

        # Calculate working memory load
        wm_df = self.calculate_working_memory_load()

        # Calculate traversal costs
        traversal_df = self.calculate_traversal_costs()

        # Calculate visual parsing complexity
        visual_df = self.calculate_visual_parsing_complexity()

        # Calculate parallelization potential
        parallel_df = self.calculate_parallelization_potential()

        # Identify specialist khipus
        specialist_results, complexity_data = self.identify_specialist_khipus(
            wm_df, traversal_df, visual_df
        )

        # Analyze correlations
        correlations = self.analyze_correlations(wm_df, complexity_data)

        # Save results
        self.save_results(wm_df, traversal_df, visual_df, parallel_df,
                         specialist_results, complexity_data, correlations)

        print("\n" + "=" * 80)
        print("PHASE 9.3 COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    analyzer = CognitiveLoadAnalyzer()
    analyzer.run_analysis()
