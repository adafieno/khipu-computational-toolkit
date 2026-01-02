"""
Phase 9.4: Structural Minimalism vs. Expressiveness

Determines if khipus are minimally sufficient or intentionally redundant:
- Marginal expressiveness: Information gain per structural layer
- Complexity vs. scale: Growth patterns
- Template matching: Standard vs. bespoke designs
- Efficiency curves: Information capacity vs. structural complexity

Integrates with:
- Phase 9.1: Information capacity metrics
- Phase 4: Structural clustering
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN


class MinimalismExpressivenessAnalyzer:
    """Analyzes structural minimalism and expressiveness in khipus."""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "phase9" / "9.4_minimalism_expressiveness"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> None:
        """Load required datasets."""
        print("Loading datasets...")
        
        # Load structural features (khipu_id, num_nodes, depth, avg_branching, etc.)
        struct_file = self.data_dir / "graph_structural_features.csv"
        self.structural = pd.read_csv(struct_file)
        print(f"  ✓ Structural features: {len(self.structural)} khipus")
        
        # Load Phase 9.1 information capacity results (khipu_id, total_information_bits, etc.)
        capacity_file = self.data_dir / "phase9" / "9.1_information_capacity" / "capacity_metrics.csv"
        if capacity_file.exists():
            self.capacity = pd.read_csv(capacity_file)
            print(f"  ✓ Information capacity: {len(self.capacity)} khipus")
        else:
            self.capacity = pd.DataFrame()
            print("  ⚠ Information capacity data not found")
        
        # Load hierarchy for depth analysis (KHIPU_ID, CORD_ID, CORD_LEVEL)
        hierarchy_file = self.data_dir / "cord_hierarchy.csv"
        self.hierarchy = pd.read_csv(hierarchy_file)
        print(f"  ✓ Hierarchy: {len(self.hierarchy)} cords")
        
        # Load color data for entropy calculations (khipu_id, cord_id, color_cd_1)
        color_file = self.data_dir / "color_data.csv"
        self.colors = pd.read_csv(color_file)
        print(f"  ✓ Color data: {len(self.colors)} records")
        
    def calculate_marginal_expressiveness(self) -> pd.DataFrame:
        """Calculate information gain per structural depth level."""
        print("\nCalculating marginal expressiveness...")
        
        expressiveness = []
        
        for khipu_id in self.hierarchy['KHIPU_ID'].unique():
            khipu_hierarchy = self.hierarchy[self.hierarchy['KHIPU_ID'] == khipu_id]
            khipu_colors = self.colors[self.colors['khipu_id'] == khipu_id]
            
            # Get max depth (CORD_LEVEL)
            max_depth = khipu_hierarchy['CORD_LEVEL'].max()
            if pd.isna(max_depth) or max_depth < 1:
                continue
            
            # Calculate entropy at each depth level
            depth_entropies = []
            for d in range(int(max_depth) + 1):
                # Get cords at this depth
                cords_at_depth = khipu_hierarchy[khipu_hierarchy['CORD_LEVEL'] == d]['CORD_ID'].values
                
                if len(cords_at_depth) == 0:
                    depth_entropies.append(0.0)
                    continue
                
                # Get colors for these cords
                colors_at_depth = khipu_colors[khipu_colors['cord_id'].isin(cords_at_depth)]
                
                if len(colors_at_depth) == 0:
                    depth_entropies.append(0.0)
                    continue
                
                # Calculate color entropy
                color_counts = colors_at_depth['color_cd_1'].value_counts()
                probabilities = color_counts / color_counts.sum()
                entropy = -sum(probabilities * np.log2(probabilities + 1e-10))
                depth_entropies.append(entropy)
            
            # Calculate marginal expressiveness (delta entropy)
            marginal_info = [depth_entropies[0]]  # Base level
            for i in range(1, len(depth_entropies)):
                delta = depth_entropies[i] - depth_entropies[i-1]
                marginal_info.append(delta)
            
            # Store results
            for d, (entropy, marginal) in enumerate(zip(depth_entropies, marginal_info)):
                expressiveness.append({
                    'khipu_id': khipu_id,
                    'depth': d,
                    'entropy': entropy,
                    'marginal_expressiveness': marginal,
                    'cumulative_entropy': sum(depth_entropies[:d+1])
                })
        
        expr_df = pd.DataFrame(expressiveness)
        
        # Calculate summary statistics
        avg_marginal = expr_df.groupby('depth')['marginal_expressiveness'].mean()
        
        print(f"  ✓ Analyzed {len(expr_df['khipu_id'].unique())} khipus")
        print(f"  ✓ Average marginal expressiveness by depth:")
        for depth, value in avg_marginal.head(5).items():
            print(f"    - Depth {depth}: {value:.3f} bits")
        
        # Identify diminishing returns point
        if len(avg_marginal) > 2:
            diminishing_point = None
            for i in range(1, len(avg_marginal)):
                if avg_marginal.iloc[i] < 0.1:  # Threshold for diminishing returns
                    diminishing_point = i
                    break
            
            if diminishing_point:
                print(f"  ✓ Diminishing returns after depth: {diminishing_point}")
            else:
                print("  ✓ No clear diminishing returns point found")
        
        return expr_df
    
    def analyze_complexity_vs_scale(self) -> Dict:
        """Analyze how complexity grows with administrative scale."""
        print("\nAnalyzing complexity vs. scale...")
        
        # Merge structural features with capacity
        if self.capacity.empty:
            print("  ⚠ Cannot analyze without capacity data")
            return {}
        
        analysis_data = self.structural.merge(
            self.capacity[['khipu_id', 'total_information_bits', 'compression_efficiency']],
            on='khipu_id',
            how='inner'
        )
        
        # Define complexity as combination of depth, nodes, and branching
        analysis_data['structural_complexity'] = (
            (analysis_data['depth'] / analysis_data['depth'].max()) * 0.4 +
            (analysis_data['num_nodes'] / analysis_data['num_nodes'].max()) * 0.4 +
            (analysis_data['avg_branching'] / analysis_data['avg_branching'].max()) * 0.2
        )
        
        # Define scale as number of nodes
        scale = analysis_data['num_nodes'].values
        complexity = analysis_data['structural_complexity'].values
        
        # Fit different growth models
        # Linear: y = a + bx
        linear_params = np.polyfit(scale, complexity, 1)
        linear_pred = np.polyval(linear_params, scale)
        linear_r2 = 1 - (np.sum((complexity - linear_pred)**2) / np.sum((complexity - complexity.mean())**2))
        
        # Logarithmic: y = a + b*log(x)
        log_scale = np.log(scale + 1)
        log_params = np.polyfit(log_scale, complexity, 1)
        log_pred = np.polyval(log_params, log_scale)
        log_r2 = 1 - (np.sum((complexity - log_pred)**2) / np.sum((complexity - complexity.mean())**2))
        
        # Power law: y = a * x^b (using log-log regression)
        log_scale_pl = np.log(scale + 1)
        log_complexity = np.log(complexity + 1e-10)
        power_params = np.polyfit(log_scale_pl, log_complexity, 1)
        power_pred = np.exp(np.polyval(power_params, log_scale_pl))
        power_r2 = 1 - (np.sum((complexity - power_pred)**2) / np.sum((complexity - complexity.mean())**2))
        
        # Determine best fit
        best_model = max(
            [('linear', linear_r2), ('logarithmic', log_r2), ('power', power_r2)],
            key=lambda x: x[1]
        )
        
        results = {
            'models': {
                'linear': {
                    'r_squared': float(linear_r2),
                    'slope': float(linear_params[0]),
                    'intercept': float(linear_params[1])
                },
                'logarithmic': {
                    'r_squared': float(log_r2),
                    'slope': float(log_params[0]),
                    'intercept': float(log_params[1])
                },
                'power': {
                    'r_squared': float(power_r2),
                    'exponent': float(power_params[0]),
                    'coefficient': float(np.exp(power_params[1]))
                }
            },
            'best_fit': best_model[0],
            'best_r_squared': float(best_model[1]),
            'n_khipus': int(len(analysis_data))
        }
        
        print(f"  ✓ Analyzed {results['n_khipus']} khipus")
        print(f"  ✓ Best fit model: {best_model[0]} (R²={best_model[1]:.3f})")
        print(f"    - Linear R²: {linear_r2:.3f}")
        print(f"    - Logarithmic R²: {log_r2:.3f}")
        print(f"    - Power R²: {power_r2:.3f}")
        
        # Identify over-engineered outliers (high complexity, low scale)
        analysis_data['complexity_residual'] = complexity - linear_pred
        outliers = analysis_data[analysis_data['complexity_residual'] > 2 * complexity.std()]
        
        print(f"  ✓ Identified {len(outliers)} over-engineered outliers")
        
        results['outliers'] = outliers[['khipu_id', 'num_nodes', 'structural_complexity']].to_dict('records')
        
        return results
    
    def identify_structural_templates(self) -> Tuple[pd.DataFrame, Dict]:
        """Identify standard structural templates using clustering."""
        print("\nIdentifying structural templates...")
        
        # Create feature vectors for clustering
        features = self.structural[['num_nodes', 'depth', 'avg_branching', 'width']].copy()
        features = features.fillna(0)
        
        # Normalize features
        features_normalized = (features - features.mean()) / features.std()
        
        # Use DBSCAN to find dense regions (templates)
        clustering = DBSCAN(eps=0.5, min_samples=10)
        self.structural['template_cluster'] = clustering.fit_predict(features_normalized)
        
        # Identify templates (clusters with >= 10 members)
        template_counts = self.structural['template_cluster'].value_counts()
        templates = template_counts[template_counts >= 10]
        templates = templates[templates.index != -1]  # Exclude noise
        
        print(f"  ✓ Identified {len(templates)} structural templates")
        
        # Calculate template statistics
        template_stats = []
        for template_id, count in templates.items():
            template_khipus = self.structural[self.structural['template_cluster'] == template_id]
            
            template_stats.append({
                'template_id': int(template_id),
                'count': int(count),
                'proportion': float(count / len(self.structural)),
                'mean_nodes': float(template_khipus['num_nodes'].mean()),
                'mean_depth': float(template_khipus['depth'].mean()),
                'mean_branching': float(template_khipus['avg_branching'].mean()),
                'std_nodes': float(template_khipus['num_nodes'].std()),
                'std_depth': float(template_khipus['depth'].std()),
                'std_branching': float(template_khipus['avg_branching'].std())
            })
        
        template_df = pd.DataFrame(template_stats).sort_values('count', ascending=False)
        
        # Calculate template adherence
        n_templated = len(self.structural[self.structural['template_cluster'] != -1])
        adherence_rate = n_templated / len(self.structural)
        
        print(f"  ✓ Template adherence: {100*adherence_rate:.1f}% of khipus")
        print(f"  ✓ Top 3 templates:")
        for _, row in template_df.head(3).iterrows():
            print(f"    - Template {row['template_id']}: {row['count']} khipus ({100*row['proportion']:.1f}%)")
            print(f"      Nodes: {row['mean_nodes']:.1f}±{row['std_nodes']:.1f}, Depth: {row['mean_depth']:.1f}±{row['std_depth']:.1f}")
        
        summary = {
            'n_templates': int(len(templates)),
            'adherence_rate': float(adherence_rate),
            'n_templated': int(n_templated),
            'n_bespoke': int(len(self.structural) - n_templated),
            'largest_template_size': int(template_df.iloc[0]['count']) if len(template_df) > 0 else 0
        }
        
        return template_df, summary
    
    def calculate_efficiency_frontier(self) -> Dict:
        """Calculate design efficiency frontier (Pareto optimal)."""
        print("\nCalculating efficiency frontier...")
        
        if self.capacity.empty:
            print("  ⚠ Cannot calculate efficiency without capacity data")
            return {}
        
        # Merge data
        analysis_data = self.structural.merge(
            self.capacity[['khipu_id', 'total_information_bits', 'compression_efficiency']],
            on='khipu_id',
            how='inner'
        )
        
        # Define complexity and capacity
        analysis_data['complexity'] = (
            analysis_data['depth'] * np.log2(analysis_data['avg_branching'] + 1) +
            analysis_data['num_nodes'] / 10
        )
        analysis_data['capacity'] = analysis_data['total_information_bits']
        
        # Identify Pareto frontier
        # A point is Pareto optimal if no other point has both higher capacity and lower complexity
        pareto_optimal = []
        
        for idx, row in analysis_data.iterrows():
            is_dominated = False
            for _, other_row in analysis_data.iterrows():
                if (other_row['capacity'] >= row['capacity'] and 
                    other_row['complexity'] <= row['complexity'] and
                    (other_row['capacity'] > row['capacity'] or other_row['complexity'] < row['complexity'])):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(idx)
        
        analysis_data['is_pareto_optimal'] = False
        analysis_data.loc[pareto_optimal, 'is_pareto_optimal'] = True
        
        n_optimal = len(pareto_optimal)
        optimal_rate = n_optimal / len(analysis_data)
        
        print(f"  ✓ Identified {n_optimal} Pareto-optimal designs ({100*optimal_rate:.1f}%)")
        
        # Calculate efficiency ratio for all khipus
        analysis_data['efficiency_ratio'] = analysis_data['capacity'] / (analysis_data['complexity'] + 1)
        
        # Statistics
        optimal_khipus = analysis_data[analysis_data['is_pareto_optimal']]
        non_optimal_khipus = analysis_data[~analysis_data['is_pareto_optimal']]
        
        results = {
            'n_pareto_optimal': int(n_optimal),
            'pareto_rate': float(optimal_rate),
            'optimal_mean_efficiency': float(optimal_khipus['efficiency_ratio'].mean()),
            'non_optimal_mean_efficiency': float(non_optimal_khipus['efficiency_ratio'].mean()),
            'efficiency_improvement': float(
                (optimal_khipus['efficiency_ratio'].mean() / non_optimal_khipus['efficiency_ratio'].mean() - 1) * 100
            ),
            'pareto_khipus': optimal_khipus[['khipu_id', 'complexity', 'capacity', 'efficiency_ratio']].to_dict('records')
        }
        
        print(f"  ✓ Optimal mean efficiency: {results['optimal_mean_efficiency']:.3f}")
        print(f"  ✓ Non-optimal mean efficiency: {results['non_optimal_mean_efficiency']:.3f}")
        print(f"  ✓ Efficiency improvement: {results['efficiency_improvement']:.1f}%")
        
        return results
    
    def save_results(self, expressiveness_df: pd.DataFrame, complexity_analysis: Dict,
                    template_df: pd.DataFrame, template_summary: Dict,
                    efficiency_frontier: Dict) -> None:
        """Save all analysis results."""
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        
        # Save expressiveness curves
        expressiveness_df.to_csv(self.output_dir / "expressiveness_curves.csv", index=False)
        print(f"  ✓ expressiveness_curves.csv ({len(expressiveness_df)} records)")
        
        # Save template analysis
        template_df.to_csv(self.output_dir / "template_analysis.csv", index=False)
        print(f"  ✓ template_analysis.csv ({len(template_df)} templates)")
        
        # Save template assignments
        template_assignments = self.structural[['khipu_id', 'template_cluster']].copy()
        template_assignments.to_csv(self.output_dir / "template_assignments.csv", index=False)
        print(f"  ✓ template_assignments.csv ({len(template_assignments)} khipus)")
        
        # Save complexity analysis
        with open(self.output_dir / "complexity_vs_scale.json", 'w') as f:
            json.dump(complexity_analysis, f, indent=2)
        print("  ✓ complexity_vs_scale.json")
        
        # Save efficiency frontier
        with open(self.output_dir / "efficiency_frontier.json", 'w') as f:
            json.dump(efficiency_frontier, f, indent=2)
        print("  ✓ efficiency_frontier.json")
        
        # Summary report
        summary = {
            'analysis': 'Phase 9.4: Structural Minimalism vs. Expressiveness',
            'date': '2026-01-01',
            'expressiveness': {
                'n_khipus': int(expressiveness_df['khipu_id'].nunique()),
                'max_depth_analyzed': int(expressiveness_df['depth'].max()),
                'mean_marginal_expressiveness': float(expressiveness_df['marginal_expressiveness'].mean())
            },
            'complexity_vs_scale': complexity_analysis,
            'templates': template_summary,
            'efficiency_frontier': {k: v for k, v in efficiency_frontier.items() if k != 'pareto_khipus'}
        }
        
        with open(self.output_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print("  ✓ analysis_summary.json")
        
        print(f"\n✓ All results saved to: {self.output_dir}")
    
    def run_analysis(self) -> None:
        """Execute complete minimalism vs. expressiveness analysis."""
        print("=" * 80)
        print("PHASE 9.4: STRUCTURAL MINIMALISM VS. EXPRESSIVENESS")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Load data
        self.load_data()
        
        # Calculate marginal expressiveness
        expressiveness_df = self.calculate_marginal_expressiveness()
        
        # Analyze complexity vs scale
        complexity_analysis = self.analyze_complexity_vs_scale()
        
        # Identify structural templates
        template_df, template_summary = self.identify_structural_templates()
        
        # Calculate efficiency frontier
        efficiency_frontier = self.calculate_efficiency_frontier()
        
        # Save results
        self.save_results(expressiveness_df, complexity_analysis, template_df,
                         template_summary, efficiency_frontier)
        
        print("\n" + "=" * 80)
        print("PHASE 9.4 COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    analyzer = MinimalismExpressivenessAnalyzer()
    analyzer.run_analysis()
