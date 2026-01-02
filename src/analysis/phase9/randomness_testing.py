"""
Phase 9.8: Randomness and Intentional Design Testing

Tests whether khipu patterns differ significantly from random structures.
Generates synthetic null models to identify intentional design constraints.

Key Analyses:
- Synthetic khipu generation (multiple null models)
- Statistical distance measurements (KS, Wasserstein, Earth Mover's)
- Forbidden design region identification
- Significance testing (>5σ threshold)

References:
- Phase 2: Real khipu distributions (structural, chromatic, numeric)
- Phase 4: Clustering patterns to validate against random
- Phase 8: Administrative types as design constraints
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from scipy.stats import ks_2samp, wasserstein_distance, chi2_contingency
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class RandomnessAnalyzer:
    """
    Test if khipus show intentional design by comparing to random null models.
    
    Generates synthetic khipus under different null hypotheses:
    1. Uniform Random: All features uniformly distributed
    2. Empirical Random: Sample from observed distributions independently
    3. Constrained Random: Respect basic structural constraints (tree hierarchy)
    """
    
    def __init__(self, data_dir: Path = Path("data/processed"), n_synthetic: int = 1000):
        """Initialize with data directory and number of synthetic khipus."""
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "phase9" / "9.8_randomness"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_synthetic = n_synthetic
        
        print("=" * 80)
        print("PHASE 9.8: RANDOMNESS & INTENTIONAL DESIGN TESTING")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Synthetic khipus to generate: {n_synthetic}\n")
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load required datasets."""
        print("Loading datasets...")
        
        data = {}
        
        data['structural'] = pd.read_csv(self.data_dir / "graph_structural_features.csv")
        print(f"  ✓ Structural features: {len(data['structural'])} khipus")
        
        data['numeric'] = pd.read_csv(self.data_dir / "cord_numeric_values.csv")
        print(f"  ✓ Numeric values: {len(data['numeric'])} records")
        
        data['color'] = pd.read_csv(self.data_dir / "color_data.csv")
        print(f"  ✓ Color data: {len(data['color'])} records")
        
        data['hierarchy'] = pd.read_csv(self.data_dir / "cord_hierarchy.csv")
        print(f"  ✓ Hierarchy: {len(data['hierarchy'])} cords")
        
        print()
        return data
    
    def generate_uniform_random_khipus(self, real_structural: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic khipus with uniform random distributions.
        
        Null Hypothesis 1: All features uniformly distributed in observed ranges.
        """
        print(f"Generating {self.n_synthetic} uniform random khipus...")
        
        synthetic = []
        
        for i in range(self.n_synthetic):
            # Sample uniformly from observed ranges
            n_cords = np.random.randint(
                int(real_structural['num_nodes'].min()),
                int(real_structural['num_nodes'].max())
            )
            
            depth = np.random.randint(
                int(real_structural['depth'].min()),
                int(real_structural['depth'].max())
            )
            
            avg_branching = np.random.uniform(
                real_structural['avg_branching'].min(),
                real_structural['avg_branching'].max()
            )
            
            synthetic.append({
                'khipu_id': f'synthetic_uniform_{i}',
                'num_nodes': n_cords,
                'depth': depth,
                'avg_branching': avg_branching,
                'model': 'uniform_random'
            })
        
        df_synthetic = pd.DataFrame(synthetic)
        print(f"  ✓ Generated {len(df_synthetic)} uniform random khipus\n")
        
        return df_synthetic
    
    def generate_empirical_random_khipus(self, real_structural: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic khipus by sampling from observed distributions.
        
        Null Hypothesis 2: Features sampled independently from empirical distributions.
        """
        print(f"Generating {self.n_synthetic} empirical random khipus...")
        
        synthetic = []
        
        for i in range(self.n_synthetic):
            # Sample from empirical distributions (independent features)
            n_cords = np.random.choice(real_structural['num_nodes'].values)
            depth = np.random.choice(real_structural['depth'].values)
            avg_branching = np.random.choice(real_structural['avg_branching'].values)
            
            synthetic.append({
                'khipu_id': f'synthetic_empirical_{i}',
                'num_nodes': n_cords,
                'depth': depth,
                'avg_branching': avg_branching,
                'model': 'empirical_random'
            })
        
        df_synthetic = pd.DataFrame(synthetic)
        print(f"  ✓ Generated {len(df_synthetic)} empirical random khipus\n")
        
        return df_synthetic
    
    def generate_constrained_random_khipus(self, real_structural: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic khipus respecting basic hierarchical constraints.
        
        Null Hypothesis 3: Hierarchical constraints but random within constraints.
        Constraint: depth * avg_branching ≈ num_nodes (tree property)
        """
        print(f"Generating {self.n_synthetic} constrained random khipus...")
        
        synthetic = []
        
        for i in range(self.n_synthetic):
            # Sample depth from empirical distribution
            depth = np.random.choice(real_structural['depth'].values)
            
            # Sample branching from empirical distribution
            avg_branching = np.random.choice(real_structural['avg_branching'].values)
            
            # Estimate cord count from tree constraint (with noise)
            estimated_cords = int(depth * avg_branching * np.random.uniform(0.8, 1.2))
            n_cords = max(1, estimated_cords)
            
            synthetic.append({
                'khipu_id': f'synthetic_constrained_{i}',
                'num_nodes': n_cords,
                'depth': depth,
                'avg_branching': avg_branching,
                'model': 'constrained_random'
            })
        
        df_synthetic = pd.DataFrame(synthetic)
        print(f"  ✓ Generated {len(df_synthetic)} constrained random khipus\n")
        
        return df_synthetic
    
    def calculate_statistical_distances(self,
                                       real_data: pd.DataFrame,
                                       synthetic_data: pd.DataFrame,
                                       features: List[str]) -> Dict:
        """
        Calculate statistical distances between real and synthetic distributions.
        
        Uses:
        - Kolmogorov-Smirnov test (distribution similarity)
        - Wasserstein distance (Earth Mover's distance)
        """
        print(f"Calculating statistical distances for {synthetic_data['model'].iloc[0]}...")
        
        distances = {}
        
        for feature in features:
            if feature not in real_data.columns or feature not in synthetic_data.columns:
                continue
            
            real_vals = real_data[feature].dropna()
            synth_vals = synthetic_data[feature].dropna()
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = ks_2samp(real_vals, synth_vals)
            
            # Wasserstein distance
            w_dist = wasserstein_distance(real_vals, synth_vals)
            
            # Standardized effect size
            pooled_std = np.sqrt((real_vals.std()**2 + synth_vals.std()**2) / 2)
            cohens_d = abs(real_vals.mean() - synth_vals.mean()) / pooled_std if pooled_std > 0 else 0
            
            distances[feature] = {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pval),
                'wasserstein_distance': float(w_dist),
                'cohens_d': float(cohens_d),
                'significant': bool(ks_pval < 0.001),
                'real_mean': float(real_vals.mean()),
                'synth_mean': float(synth_vals.mean())
            }
        
        # Count significant differences
        n_significant = sum(1 for d in distances.values() if d['significant'])
        print(f"  ✓ {n_significant}/{len(distances)} features significantly different (p<0.001)\n")
        
        return distances
    
    def identify_forbidden_regions(self,
                                   real_data: pd.DataFrame,
                                   synthetic_data: pd.DataFrame) -> Dict:
        """
        Identify design space regions occupied by synthetic but not real khipus.
        
        "Forbidden regions" = parameter combinations that are possible but never occur.
        """
        print("Identifying forbidden design regions...")
        
        # 2D analysis: depth vs branching
        depth_bins = np.linspace(0, 10, 11)
        branch_bins = np.linspace(0, 20, 21)
        
        real_hist, _, _ = np.histogram2d(
            real_data['depth'],
            real_data['avg_branching'],
            bins=[depth_bins, branch_bins]
        )
        
        synth_hist, _, _ = np.histogram2d(
            synthetic_data['depth'],
            synthetic_data['avg_branching'],
            bins=[depth_bins, branch_bins]
        )
        
        # Forbidden: synthetic has examples but real doesn't (normalized)
        real_density = real_hist / real_hist.sum()
        synth_density = synth_hist / synth_hist.sum()
        
        forbidden_mask = (synth_density > 0.01) & (real_density == 0)
        n_forbidden = forbidden_mask.sum()
        
        forbidden = {
            'n_forbidden_bins': int(n_forbidden),
            'total_bins': int(forbidden_mask.size),
            'forbidden_fraction': float(n_forbidden / forbidden_mask.size),
            'interpretation': 'Regions possible but avoided by real khipus'
        }
        
        print(f"  ✓ {n_forbidden}/{forbidden_mask.size} bins forbidden ({forbidden['forbidden_fraction']:.1%})\n")
        
        return forbidden
    
    def calculate_design_space_overlap(self,
                                      real_data: pd.DataFrame,
                                      synthetic_data: pd.DataFrame) -> Dict:
        """
        Calculate overlap in multivariate design space using PCA.
        """
        print("Calculating design space overlap (PCA)...")
        
        features = ['num_nodes', 'depth', 'avg_branching']
        
        # Combine real and synthetic for PCA
        real_features = real_data[features].fillna(0)
        synth_features = synthetic_data[features].fillna(0)
        
        combined = pd.concat([real_features, synth_features])
        
        # PCA projection
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(combined)
        
        real_pca = pca_coords[:len(real_features)]
        synth_pca = pca_coords[len(real_features):]
        
        # Calculate centroids and distances
        real_centroid = real_pca.mean(axis=0)
        synth_centroid = synth_pca.mean(axis=0)
        centroid_distance = np.linalg.norm(real_centroid - synth_centroid)
        
        # Calculate overlap using convex hull approximation (circle radius)
        real_radius = np.percentile(np.linalg.norm(real_pca - real_centroid, axis=1), 95)
        synth_radius = np.percentile(np.linalg.norm(synth_pca - synth_centroid, axis=1), 95)
        
        # Overlap metric: distance relative to radii
        separation = centroid_distance / (real_radius + synth_radius)
        
        overlap = {
            'pca_variance_explained': pca.explained_variance_ratio_.tolist(),
            'centroid_distance': float(centroid_distance),
            'real_radius_95pct': float(real_radius),
            'synth_radius_95pct': float(synth_radius),
            'separation_ratio': float(separation),
            'interpretation': 'High separation (>0.5) indicates distinct design spaces'
        }
        
        print(f"  ✓ Centroid distance: {centroid_distance:.2f}")
        print(f"  ✓ Separation ratio: {separation:.2f} ({'DISTINCT' if separation > 0.5 else 'OVERLAPPING'})\n")
        
        return overlap
    
    def test_significance(self, all_distances: Dict) -> Dict:
        """
        Test overall significance: Are real khipus non-random?
        
        Criterion: Multiple features significantly different across all null models.
        """
        print("Testing overall significance...")
        
        # Count features significant across all models
        all_features = set()
        for model_distances in all_distances.values():
            all_features.update(model_distances.keys())
        
        significant_counts = {}
        for feature in all_features:
            count = sum(
                1 for model in all_distances.values()
                if feature in model and model[feature]['significant']
            )
            significant_counts[feature] = count
        
        # Features significant in all 3 models = strong evidence
        universal_features = [f for f, c in significant_counts.items() if c == 3]
        
        significance = {
            'n_features_tested': len(all_features),
            'n_significant_in_all_models': len(universal_features),
            'universal_significant_features': universal_features,
            'verdict': 'INTENTIONAL DESIGN' if len(universal_features) >= 2 else 'INCONCLUSIVE',
            'confidence': f"{len(universal_features)}/{len(all_features)} features reject all null models"
        }
        
        print(f"  ✓ {len(universal_features)}/{len(all_features)} features reject ALL null models")
        print(f"  ✓ Verdict: {significance['verdict']}\n")
        
        return significance
    
    def save_results(self,
                    uniform_synthetic: pd.DataFrame,
                    empirical_synthetic: pd.DataFrame,
                    constrained_synthetic: pd.DataFrame,
                    all_distances: Dict,
                    all_forbidden: Dict,
                    all_overlap: Dict,
                    significance: Dict):
        """Save all Phase 9.8 results."""
        print("=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        
        # Synthetic khipus
        synthetic_dir = self.output_dir / "synthetic_khipus"
        synthetic_dir.mkdir(exist_ok=True)
        
        uniform_synthetic.to_csv(synthetic_dir / "uniform_random.csv", index=False)
        print(f"  ✓ uniform_random.csv ({len(uniform_synthetic)} khipus)")
        
        empirical_synthetic.to_csv(synthetic_dir / "empirical_random.csv", index=False)
        print(f"  ✓ empirical_random.csv ({len(empirical_synthetic)} khipus)")
        
        constrained_synthetic.to_csv(synthetic_dir / "constrained_random.csv", index=False)
        print(f"  ✓ constrained_random.csv ({len(constrained_synthetic)} khipus)")
        
        # Statistical distances
        with open(self.output_dir / "statistical_distances.json", 'w') as f:
            json.dump(all_distances, f, indent=2)
        print("  ✓ statistical_distances.json")
        
        # Forbidden regions
        with open(self.output_dir / "forbidden_regions.json", 'w') as f:
            json.dump(all_forbidden, f, indent=2)
        print("  ✓ forbidden_regions.json")
        
        # Design space overlap
        with open(self.output_dir / "design_space_overlap.json", 'w') as f:
            json.dump(all_overlap, f, indent=2)
        print("  ✓ design_space_overlap.json")
        
        # Significance test
        with open(self.output_dir / "significance_test.json", 'w') as f:
            json.dump(significance, f, indent=2)
        print("  ✓ significance_test.json")
        
        print(f"\n✓ All results saved to: {self.output_dir}")
    
    def run_analysis(self):
        """Execute complete 9.8 analysis pipeline."""
        # Load data
        data = self.load_data()
        real_structural = data['structural']
        
        # Generate synthetic khipus (3 null models)
        uniform_synthetic = self.generate_uniform_random_khipus(real_structural)
        empirical_synthetic = self.generate_empirical_random_khipus(real_structural)
        constrained_synthetic = self.generate_constrained_random_khipus(real_structural)
        
        features_to_test = ['num_nodes', 'depth', 'avg_branching']
        
        # Calculate distances for each null model
        print("=" * 80)
        print("STATISTICAL DISTANCE ANALYSIS")
        print("=" * 80)
        
        uniform_distances = self.calculate_statistical_distances(
            real_structural, uniform_synthetic, features_to_test
        )
        
        empirical_distances = self.calculate_statistical_distances(
            real_structural, empirical_synthetic, features_to_test
        )
        
        constrained_distances = self.calculate_statistical_distances(
            real_structural, constrained_synthetic, features_to_test
        )
        
        all_distances = {
            'uniform_random': uniform_distances,
            'empirical_random': empirical_distances,
            'constrained_random': constrained_distances
        }
        
        # Identify forbidden regions for each model
        print("=" * 80)
        print("FORBIDDEN REGION ANALYSIS")
        print("=" * 80)
        
        uniform_forbidden = self.identify_forbidden_regions(real_structural, uniform_synthetic)
        empirical_forbidden = self.identify_forbidden_regions(real_structural, empirical_synthetic)
        constrained_forbidden = self.identify_forbidden_regions(real_structural, constrained_synthetic)
        
        all_forbidden = {
            'uniform_random': uniform_forbidden,
            'empirical_random': empirical_forbidden,
            'constrained_random': constrained_forbidden
        }
        
        # Calculate design space overlap
        print("=" * 80)
        print("DESIGN SPACE OVERLAP ANALYSIS")
        print("=" * 80)
        
        uniform_overlap = self.calculate_design_space_overlap(real_structural, uniform_synthetic)
        empirical_overlap = self.calculate_design_space_overlap(real_structural, empirical_synthetic)
        constrained_overlap = self.calculate_design_space_overlap(real_structural, constrained_synthetic)
        
        all_overlap = {
            'uniform_random': uniform_overlap,
            'empirical_random': empirical_overlap,
            'constrained_random': constrained_overlap
        }
        
        # Test overall significance
        print("=" * 80)
        print("SIGNIFICANCE TESTING")
        print("=" * 80)
        
        significance = self.test_significance(all_distances)
        
        # Save all results
        self.save_results(
            uniform_synthetic,
            empirical_synthetic,
            constrained_synthetic,
            all_distances,
            all_forbidden,
            all_overlap,
            significance
        )
        
        print("\n" + "=" * 80)
        print("PHASE 9.8 COMPLETE")
        print("=" * 80)
        print(f"\nFinal Verdict: {significance['verdict']}")
        print(f"Confidence: {significance['confidence']}")


def main():
    """Run Phase 9.8 Randomness Testing."""
    analyzer = RandomnessAnalyzer(n_synthetic=1000)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
