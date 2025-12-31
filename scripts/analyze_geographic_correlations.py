"""
Analyze Geographic Correlations

This script analyzes whether khipu structural patterns correlate with
geographic provenance. Tests:
1. Structural feature differences across provenances
2. Clustering enrichment by provenance
3. Summation patterns by region
4. Statistical significance of geographic patterns
"""

import pandas as pd
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from scipy.stats import chi2_contingency, kruskal
from typing import Dict


class GeographicAnalyzer:
    """Analyze geographic correlations in khipu patterns."""
    
    def __init__(self, db_path: str = "khipu.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all necessary data files."""
        print("Loading data files...")
        
        data = {
            'features': pd.read_csv("data/processed/graph_structural_features.csv"),
            'clusters': pd.read_csv("data/processed/cluster_assignments_kmeans.csv"),
            'summation': pd.read_csv("data/processed/summation_test_results.csv"),
            'high_match': pd.read_csv("data/processed/high_match_khipus.csv")
        }
        
        # Get provenance from database
        query = """
        SELECT KHIPU_ID, PROVENANCE, REGION, MUSEUM_NAME
        FROM khipu_main
        """
        data['provenance'] = pd.read_sql_query(query, self.conn)
        
        print(f"✓ Loaded {len(data['provenance'])} khipu metadata records")
        return data
    
    def clean_provenance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and categorize provenance data."""
        # Replace empty strings and whitespace with 'Unknown'
        df['PROVENANCE'] = df['PROVENANCE'].fillna('Unknown')
        df['PROVENANCE'] = df['PROVENANCE'].replace(['', ' ', '  '], 'Unknown')
        df['PROVENANCE'] = df['PROVENANCE'].str.strip()
        
        # Categorize by frequency
        prov_counts = df['PROVENANCE'].value_counts()
        
        # Keep top provenances, group rest as 'Other'
        top_provenances = prov_counts[prov_counts >= 10].index.tolist()
        df['provenance_category'] = df['PROVENANCE'].apply(
            lambda x: x if x in top_provenances else 'Other'
        )
        
        return df
    
    def analyze_cluster_provenance_enrichment(self, data: Dict) -> Dict:
        """Test if clusters are enriched for specific provenances."""
        print("\n" + "="*80)
        print("CLUSTER-PROVENANCE ENRICHMENT ANALYSIS")
        print("="*80)
        
        # Merge clusters with provenance
        merged = data['clusters'].merge(
            data['provenance'], 
            left_on='khipu_id', 
            right_on='KHIPU_ID',
            how='left'
        )
        merged = self.clean_provenance_data(merged)
        
        # Remove noise cluster if present
        merged = merged[merged['cluster'] != -1]
        
        # Create contingency table
        contingency = pd.crosstab(
            merged['cluster'],
            merged['provenance_category']
        )
        
        print("\nContingency Table (Cluster × Provenance):")
        print(contingency)
        
        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        print("\nChi-Square Test:")
        print(f"  χ² = {chi2:.2f}")
        print(f"  p-value = {p_value:.6f}")
        print(f"  degrees of freedom = {dof}")
        print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")
        
        # Compute enrichment scores (observed/expected)
        enrichment = contingency / expected
        
        print("\nEnrichment Scores (Observed/Expected > 1.5):")
        for cluster in enrichment.index:
            for prov in enrichment.columns:
                score = enrichment.loc[cluster, prov]
                if score > 1.5 and contingency.loc[cluster, prov] >= 5:
                    print(f"  Cluster {cluster} × {prov}: {score:.2f}x "
                          f"(n={contingency.loc[cluster, prov]})")
        
        return {
            'contingency_table': contingency.to_dict(),
            'chi2': float(chi2),
            'p_value': float(p_value),
            'dof': int(dof),
            'significant': bool(p_value < 0.05),
            'enrichment': enrichment.to_dict()
        }
    
    def analyze_structural_features_by_provenance(self, data: Dict) -> Dict:
        """Test if structural features differ by provenance."""
        print("\n" + "="*80)
        print("STRUCTURAL FEATURES BY PROVENANCE")
        print("="*80)
        
        # Merge features with provenance
        merged = data['features'].merge(
            data['provenance'],
            left_on='khipu_id',
            right_on='KHIPU_ID',
            how='left'
        )
        merged = self.clean_provenance_data(merged)
        
        # Filter to top provenances
        top_provs = merged['provenance_category'].value_counts().head(6).index.tolist()
        merged_top = merged[merged['provenance_category'].isin(top_provs)]
        
        # Test key features
        features_to_test = ['num_nodes', 'depth', 'avg_branching', 'has_numeric']
        
        results = {}
        
        for feature in features_to_test:
            # Group by provenance
            groups = [
                merged_top[merged_top['provenance_category'] == prov][feature].dropna()
                for prov in top_provs
            ]
            
            # Skip if any group is too small
            if any(len(g) < 3 for g in groups):
                continue
            
            # Perform Kruskal-Wallis test (non-parametric ANOVA)
            h_stat, p_value = kruskal(*groups)
            
            # Compute means per provenance
            means = {
                prov: float(merged_top[merged_top['provenance_category'] == prov][feature].mean())
                for prov in top_provs
            }
            
            results[feature] = {
                'h_statistic': float(h_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05),
                'means_by_provenance': means
            }
            
            print(f"\n{feature}:")
            print(f"  Kruskal-Wallis H = {h_stat:.2f}, p = {p_value:.6f}")
            print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")
            if p_value < 0.05:
                print("  Means by provenance:")
                for prov, mean in sorted(means.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {prov}: {mean:.2f}")
        
        return results
    
    def analyze_summation_by_provenance(self, data: Dict) -> Dict:
        """Test if summation patterns differ by provenance."""
        print("\n" + "="*80)
        print("SUMMATION PATTERNS BY PROVENANCE")
        print("="*80)
        
        # Merge summation results with provenance
        merged = data['summation'].merge(
            data['provenance'],
            left_on='khipu_id',
            right_on='KHIPU_ID',
            how='left'
        )
        merged = self.clean_provenance_data(merged)
        
        # Filter to top provenances
        top_provs = merged['provenance_category'].value_counts().head(6).index.tolist()
        merged_top = merged[merged['provenance_category'].isin(top_provs)]
        
        # Compute summation statistics by provenance
        summation_stats = []
        
        for prov in top_provs:
            prov_data = merged_top[merged_top['provenance_category'] == prov]
            
            stats = {
                'provenance': prov,
                'count': len(prov_data),
                'pct_with_summation': float(prov_data['has_pendant_summation'].mean() * 100),
                'avg_match_rate': float(prov_data['pendant_match_rate'].mean()),
                'pct_with_white': float(prov_data['has_white_boundaries'].mean() * 100)
            }
            summation_stats.append(stats)
            
            print(f"\n{prov} (n={stats['count']}):")
            print(f"  With summation: {stats['pct_with_summation']:.1f}%")
            print(f"  Avg match rate: {stats['avg_match_rate']:.3f}")
            print(f"  With white boundaries: {stats['pct_with_white']:.1f}%")
        
        # Test for significant differences
        # Chi-square for has_pendant_summation (binary)
        contingency_summation = pd.crosstab(
            merged_top['provenance_category'],
            merged_top['has_pendant_summation']
        )
        chi2_sum, p_sum, _, _ = chi2_contingency(contingency_summation)
        
        print("\nChi-Square Test (Summation Presence):")
        print(f"  χ² = {chi2_sum:.2f}, p = {p_sum:.6f}")
        print(f"  Significant: {'YES' if p_sum < 0.05 else 'NO'}")
        
        # Kruskal-Wallis for match rates
        groups_match = [
            merged_top[merged_top['provenance_category'] == prov]['pendant_match_rate'].dropna()
            for prov in top_provs
        ]
        h_match, p_match = kruskal(*groups_match)
        
        print("\nKruskal-Wallis Test (Match Rate):")
        print(f"  H = {h_match:.2f}, p = {p_match:.6f}")
        print(f"  Significant: {'YES' if p_match < 0.05 else 'NO'}")
        
        return {
            'summation_stats': summation_stats,
            'chi2_summation': float(chi2_sum),
            'p_value_summation': float(p_sum),
            'h_match_rate': float(h_match),
            'p_value_match_rate': float(p_match)
        }
    
    def analyze_high_match_provenances(self, data: Dict) -> Dict:
        """Analyze provenance distribution of high-match khipus."""
        print("\n" + "="*80)
        print("HIGH-MATCH KHIPU PROVENANCES")
        print("="*80)
        
        # High-match data already has provenance from previous merge
        high_match = data['high_match']
        
        prov_counts = high_match['PROVENANCE'].value_counts()
        
        print(f"\nHigh-match khipus (n={len(high_match)}) by provenance:")
        for prov, count in prov_counts.items():
            print(f"  {prov}: {count}")
        
        # Compare to overall distribution
        overall = data['provenance']['PROVENANCE'].value_counts()
        
        print("\nComparison to overall distribution:")
        for prov in prov_counts.index:
            high_match_pct = prov_counts[prov] / len(high_match) * 100
            overall_pct = overall.get(prov, 0) / len(data['provenance']) * 100
            enrichment = high_match_pct / overall_pct if overall_pct > 0 else 0
            print(f"  {prov}: {high_match_pct:.1f}% vs {overall_pct:.1f}% "
                  f"(enrichment: {enrichment:.2f}x)")
        
        return {
            'high_match_provenance_counts': prov_counts.to_dict(),
            'total_high_match': len(high_match)
        }
    
    def create_provenance_summary(self, data: Dict) -> Dict:
        """Create summary statistics by provenance."""
        print("\n" + "="*80)
        print("PROVENANCE SUMMARY STATISTICS")
        print("="*80)
        
        # Merge all data
        merged = data['features'].merge(
            data['provenance'], left_on='khipu_id', right_on='KHIPU_ID', how='left'
        ).merge(
            data['summation'], left_on='khipu_id', right_on='khipu_id', how='left'
        ).merge(
            data['clusters'][['khipu_id', 'cluster']], on='khipu_id', how='left'
        )
        
        merged = self.clean_provenance_data(merged)
        
        # Get top provenances
        top_provs = merged['provenance_category'].value_counts().head(8).index.tolist()
        
        summary = []
        
        for prov in top_provs:
            prov_data = merged[merged['provenance_category'] == prov]
            
            summary.append({
                'provenance': prov,
                'count': len(prov_data),
                'avg_nodes': float(prov_data['num_nodes'].mean()),
                'avg_depth': float(prov_data['depth'].mean()),
                'avg_branching': float(prov_data['avg_branching'].mean()),
                'pct_numeric': float(prov_data['has_numeric'].mean() * 100),
                'pct_summation': float(prov_data['has_pendant_summation'].mean() * 100),
                'avg_match_rate': float(prov_data['pendant_match_rate'].mean()),
                'most_common_cluster': int(prov_data['cluster'].mode()[0]) if len(prov_data['cluster'].mode()) > 0 else -1
            })
        
        print("\nTop Provenances Summary:")
        print("-"*80)
        for s in summary:
            print(f"\n{s['provenance']} (n={s['count']}):")
            print(f"  Avg size: {s['avg_nodes']:.1f} nodes")
            print(f"  Avg depth: {s['avg_depth']:.2f}")
            print(f"  Numeric coverage: {s['pct_numeric']:.1f}%")
            print(f"  Summation rate: {s['pct_summation']:.1f}%")
            print(f"  Most common cluster: {s['most_common_cluster']}")
        
        return summary
    
    def export_results(self, all_results: Dict, output_dir: str = "data/processed"):
        """Export geographic analysis results."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_json = Path(output_dir) / "geographic_correlation_analysis.json"
        
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'cluster_provenance_enrichment': all_results['cluster_enrichment'],
            'structural_features_by_provenance': all_results['features_by_prov'],
            'summation_by_provenance': all_results['summation_by_prov'],
            'high_match_provenances': all_results['high_match_prov'],
            'provenance_summary': all_results['prov_summary']
        }
        
        with open(output_json, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✓ Exported analysis to {output_json}")
    
    def run_analysis(self):
        """Run complete geographic correlation analysis."""
        print("="*80)
        print("GEOGRAPHIC CORRELATION ANALYSIS")
        print("="*80)
        
        # Load data
        data = self.load_data()
        
        # Run analyses
        cluster_enrichment = self.analyze_cluster_provenance_enrichment(data)
        features_by_prov = self.analyze_structural_features_by_provenance(data)
        summation_by_prov = self.analyze_summation_by_provenance(data)
        high_match_prov = self.analyze_high_match_provenances(data)
        prov_summary = self.create_provenance_summary(data)
        
        # Compile results
        all_results = {
            'cluster_enrichment': cluster_enrichment,
            'features_by_prov': features_by_prov,
            'summation_by_prov': summation_by_prov,
            'high_match_prov': high_match_prov,
            'prov_summary': prov_summary
        }
        
        # Export
        self.export_results(all_results)
        
        print("\n" + "="*80)
        print("GEOGRAPHIC CORRELATION ANALYSIS COMPLETE")
        print("="*80)
        
        return all_results
    
    def __del__(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()


if __name__ == "__main__":
    analyzer = GeographicAnalyzer()
    analyzer.run_analysis()
