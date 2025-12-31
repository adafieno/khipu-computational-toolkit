"""
Analyze High-Match Summation Khipus

This script identifies and analyzes khipus with high summation match rates
to discover structural patterns, color usage, and hierarchical characteristics
that correlate with successful summation encoding.
"""

import pandas as pd
import sqlite3
import json
from pathlib import Path
from datetime import datetime


class HighMatchAnalyzer:
    """Analyze khipus with high summation match rates for pattern discovery."""
    
    def __init__(self, db_path: str = "khipu.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
    def load_summation_results(self, results_path: str = "data/processed/summation_test_results.csv") -> pd.DataFrame:
        """Load summation test results and enrich with additional khipu data."""
        df = pd.read_csv(results_path)
        
        # Load cord hierarchy to get additional statistics
        cord_df = pd.read_csv("data/processed/cord_hierarchy.csv")
        
        # Compute per-khipu statistics
        khipu_stats = cord_df.groupby('KHIPU_ID').agg({
            'CORD_ID': 'count',
            'has_numeric_value': 'sum',
            'CORD_LEVEL': 'max'
        }).reset_index()
        khipu_stats.columns = ['KHIPU_ID', 'total_cords', 'cords_with_numeric', 'max_hierarchy_depth']
        khipu_stats['numeric_coverage'] = khipu_stats['cords_with_numeric'] / khipu_stats['total_cords']
        
        # Load color data for white cord counts
        color_df = pd.read_csv("data/processed/color_data.csv")
        white_counts = color_df[color_df['color_cd_1'] == 'W'].groupby('khipu_id').size().reset_index(name='white_cord_count')
        
        # Merge all data
        df = df.merge(khipu_stats, left_on='khipu_id', right_on='KHIPU_ID', how='left')
        df = df.merge(white_counts, left_on='khipu_id', right_on='khipu_id', how='left')
        
        # Fill NaN values
        df['white_cord_count'] = df['white_cord_count'].fillna(0)
        df['has_white_cords'] = df['white_cord_count'] > 0
        df['total_cords'] = df['total_cords'].fillna(0)
        df['numeric_coverage'] = df['numeric_coverage'].fillna(0)
        df['max_hierarchy_depth'] = df['max_hierarchy_depth'].fillna(0)
        
        # Rename khipu_id to KHIPU_ID for consistency
        df['KHIPU_ID'] = df['khipu_id']
        
        print(f"Loaded summation results for {len(df)} khipus")
        return df
    
    def identify_high_match_khipus(self, df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """Identify khipus with match rate above threshold."""
        high_match = df[df['pendant_match_rate'] >= threshold].copy()
        high_match = high_match.sort_values('pendant_match_rate', ascending=False)
        
        print(f"\nIdentified {len(high_match)} khipus with match rate >= {threshold}")
        print(f"  Perfect matches (1.0): {len(high_match[high_match['pendant_match_rate'] == 1.0])}")
        print(f"  High matches (0.9-0.99): {len(high_match[(high_match['pendant_match_rate'] >= 0.9) & (high_match['pendant_match_rate'] < 1.0)])}")
        print(f"  Good matches (0.8-0.89): {len(high_match[(high_match['pendant_match_rate'] >= 0.8) & (high_match['pendant_match_rate'] < 0.9)])}")
        
        return high_match
    
    def get_khipu_metadata(self, khipu_ids: list) -> pd.DataFrame:
        """Get metadata for specified khipus."""
        placeholders = ','.join(['?'] * len(khipu_ids))
        query = f"""
        SELECT 
            KHIPU_ID,
            PROVENANCE,
            REGION,
            MUSEUM_NAME,
            OKR_NUM,
            INVESTIGATOR_NUM,
            NOTES
        FROM khipu_main
        WHERE KHIPU_ID IN ({placeholders})
        """
        
        df = pd.read_sql_query(query, self.conn, params=khipu_ids)
        return df
    
    def analyze_color_patterns(self, khipu_ids: list) -> pd.DataFrame:
        """Analyze color usage patterns in high-match khipus."""
        placeholders = ','.join(['?'] * len(khipu_ids))
        query = f"""
        SELECT 
            c.KHIPU_ID,
            acc.COLOR_CD_1,
            acc.FULL_COLOR,
            COUNT(*) as color_count,
            AVG(c.CORD_LENGTH) as avg_cord_length
        FROM cord c
        LEFT JOIN ascher_cord_color acc ON c.CORD_ID = acc.CORD_ID
        WHERE c.KHIPU_ID IN ({placeholders})
        GROUP BY c.KHIPU_ID, acc.COLOR_CD_1, acc.FULL_COLOR
        """
        
        df = pd.read_sql_query(query, self.conn, params=khipu_ids)
        return df
    
    def analyze_hierarchical_structure(self, khipu_ids: list) -> pd.DataFrame:
        """Analyze hierarchical structure patterns."""
        results = []
        
        for khipu_id in khipu_ids:
            query = """
            SELECT 
                CORD_LEVEL,
                COUNT(*) as cord_count,
                AVG(CORD_LENGTH) as avg_length,
                COUNT(DISTINCT PENDANT_FROM) as unique_parents
            FROM cord
            WHERE KHIPU_ID = ?
            GROUP BY CORD_LEVEL
            ORDER BY CORD_LEVEL
            """
            
            level_df = pd.read_sql_query(query, self.conn, params=[khipu_id])
            
            results.append({
                'KHIPU_ID': khipu_id,
                'max_depth': level_df['CORD_LEVEL'].max() if len(level_df) > 0 else 0,
                'total_levels': len(level_df),
                'cords_at_level_1': level_df[level_df['CORD_LEVEL'] == 1]['cord_count'].iloc[0] if len(level_df[level_df['CORD_LEVEL'] == 1]) > 0 else 0,
                'avg_branching': level_df['unique_parents'].mean() if len(level_df) > 0 else 0
            })
        
        return pd.DataFrame(results)
    
    def analyze_white_cord_usage(self, khipu_ids: list) -> pd.DataFrame:
        """Analyze white cord usage patterns."""
        placeholders = ','.join(['?'] * len(khipu_ids))
        query = f"""
        SELECT 
            c.KHIPU_ID,
            COUNT(CASE WHEN acc.COLOR_CD_1 = 'W' THEN 1 END) as white_cord_count,
            COUNT(*) as total_cords,
            CAST(COUNT(CASE WHEN acc.COLOR_CD_1 = 'W' THEN 1 END) AS FLOAT) / COUNT(*) as white_ratio,
            AVG(CASE WHEN acc.COLOR_CD_1 = 'W' THEN c.CORD_LEVEL END) as avg_white_level
        FROM cord c
        LEFT JOIN ascher_cord_color acc ON c.CORD_ID = acc.CORD_ID
        WHERE c.KHIPU_ID IN ({placeholders})
        GROUP BY c.KHIPU_ID
        """
        
        df = pd.read_sql_query(query, self.conn, params=khipu_ids)
        return df
    
    def compute_statistics(self, high_match_df: pd.DataFrame) -> dict:
        """Compute summary statistics for high-match khipus."""
        stats = {
            'total_high_match': len(high_match_df),
            'avg_match_rate': high_match_df['pendant_match_rate'].mean(),
            'avg_numeric_coverage': high_match_df['numeric_coverage'].mean(),
            'avg_white_cord_count': high_match_df['white_cord_count'].mean(),
            'avg_hierarchy_depth': high_match_df['max_hierarchy_depth'].mean(),
            'avg_total_cords': high_match_df['total_cords'].mean(),
            'pct_with_white_cords': (high_match_df['has_white_cords'].sum() / len(high_match_df)) * 100
        }
        
        return stats
    
    def compare_high_vs_low_match(self, all_results: pd.DataFrame, threshold: float = 0.8) -> dict:
        """Compare characteristics of high vs low match khipus."""
        high_match = all_results[all_results['pendant_match_rate'] >= threshold]
        low_match = all_results[all_results['pendant_match_rate'] < threshold]
        
        comparison = {
            'high_match': {
                'count': len(high_match),
                'avg_match_rate': high_match['pendant_match_rate'].mean(),
                'avg_white_cord_count': high_match['white_cord_count'].mean(),
                'pct_with_white': (high_match['has_white_cords'].sum() / len(high_match)) * 100 if len(high_match) > 0 else 0,
                'avg_depth': high_match['max_hierarchy_depth'].mean(),
                'avg_numeric_coverage': high_match['numeric_coverage'].mean()
            },
            'low_match': {
                'count': len(low_match),
                'avg_match_rate': low_match['pendant_match_rate'].mean(),
                'avg_white_cord_count': low_match['white_cord_count'].mean(),
                'pct_with_white': (low_match['has_white_cords'].sum() / len(low_match)) * 100 if len(low_match) > 0 else 0,
                'avg_depth': low_match['max_hierarchy_depth'].mean(),
                'avg_numeric_coverage': low_match['numeric_coverage'].mean()
            }
        }
        
        # Compute differences
        comparison['differences'] = {
            'match_rate_diff': comparison['high_match']['avg_match_rate'] - comparison['low_match']['avg_match_rate'],
            'white_cord_diff': comparison['high_match']['avg_white_cord_count'] - comparison['low_match']['avg_white_cord_count'],
            'white_pct_diff': comparison['high_match']['pct_with_white'] - comparison['low_match']['pct_with_white'],
            'depth_diff': comparison['high_match']['avg_depth'] - comparison['low_match']['avg_depth'],
            'coverage_diff': comparison['high_match']['avg_numeric_coverage'] - comparison['low_match']['avg_numeric_coverage']
        }
        
        return comparison
    
    def identify_templates(self, high_match_df: pd.DataFrame, metadata_df: pd.DataFrame) -> list:
        """Identify potential template khipus (perfect matches with good documentation)."""
        perfect = high_match_df[high_match_df['pendant_match_rate'] == 1.0].copy()
        perfect = perfect.merge(metadata_df, on='KHIPU_ID', how='left')
        
        # Prioritize well-documented khipus with good numeric coverage
        perfect = perfect[perfect['numeric_coverage'] > 0.8]
        perfect = perfect.sort_values(['total_cords', 'numeric_coverage'], ascending=[False, False])
        
        templates = []
        for _, row in perfect.head(10).iterrows():
            templates.append({
                'KHIPU_ID': row['KHIPU_ID'],
                'OKR_NUM': row.get('OKR_NUM', 'N/A'),
                'PROVENANCE': row.get('PROVENANCE', 'Unknown'),
                'total_cords': row['total_cords'],
                'numeric_coverage': row['numeric_coverage'],
                'white_cord_count': row['white_cord_count'],
                'max_depth': row['max_hierarchy_depth']
            })
        
        return templates
    
    def export_results(self, high_match_df: pd.DataFrame, stats: dict, 
                      comparison: dict, templates: list, output_dir: str = "data/processed"):
        """Export analysis results."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Export high-match khipus CSV
        output_csv = Path(output_dir) / "high_match_khipus.csv"
        high_match_df.to_csv(output_csv, index=False)
        print(f"\n✓ Exported high-match khipus to {output_csv}")
        
        # Export analysis results JSON
        results = {
            'generated_at': datetime.now().isoformat(),
            'analysis_threshold': 0.8,
            'statistics': stats,
            'high_vs_low_comparison': comparison,
            'template_khipus': templates
        }
        
        output_json = Path(output_dir) / "high_match_analysis.json"
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Exported analysis results to {output_json}")
    
    def run_analysis(self):
        """Run complete high-match analysis."""
        print("="*80)
        print("HIGH-MATCH SUMMATION KHIPU ANALYSIS")
        print("="*80)
        
        # Load summation results
        results_df = self.load_summation_results()
        
        # Identify high-match khipus (≥80% match rate)
        high_match = self.identify_high_match_khipus(results_df, threshold=0.8)
        
        # Get metadata for high-match khipus
        khipu_ids = high_match['KHIPU_ID'].tolist()
        metadata = self.get_khipu_metadata(khipu_ids)
        
        # Merge with metadata
        high_match = high_match.merge(metadata, on='KHIPU_ID', how='left')
        
        # Compute statistics
        print("\n" + "-"*80)
        print("STATISTICS FOR HIGH-MATCH KHIPUS (≥80%)")
        print("-"*80)
        stats = self.compute_statistics(high_match)
        for key, value in stats.items():
            print(f"  {key}: {value:.3f}")
        
        # Compare high vs low match
        print("\n" + "-"*80)
        print("HIGH-MATCH vs LOW-MATCH COMPARISON")
        print("-"*80)
        comparison = self.compare_high_vs_low_match(results_df, threshold=0.8)
        
        print("\nHigh-Match Khipus (≥80%):")
        for key, value in comparison['high_match'].items():
            print(f"  {key}: {value:.3f}")
        
        print("\nLow-Match Khipus (<80%):")
        for key, value in comparison['low_match'].items():
            print(f"  {key}: {value:.3f}")
        
        print("\nDifferences (High - Low):")
        for key, value in comparison['differences'].items():
            sign = "+" if value > 0 else ""
            print(f"  {key}: {sign}{value:.3f}")
        
        # Identify template khipus
        print("\n" + "-"*80)
        print("TEMPLATE KHIPUS (Perfect Matches)")
        print("-"*80)
        templates = self.identify_templates(high_match, metadata)
        print(f"Identified {len(templates)} template khipus:\n")
        for i, template in enumerate(templates, 1):
            print(f"{i}. Khipu {template['KHIPU_ID']} (OKR: {template['OKR_NUM']})")
            print(f"   Provenance: {template['PROVENANCE']}")
            print(f"   Cords: {template['total_cords']}, Coverage: {template['numeric_coverage']:.1%}")
            print(f"   White cords: {template['white_cord_count']}, Depth: {template['max_depth']}")
            print()
        
        # Export results
        self.export_results(high_match, stats, comparison, templates)
        
        print("="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        return high_match, stats, comparison, templates
    
    def __del__(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()


if __name__ == "__main__":
    analyzer = HighMatchAnalyzer()
    analyzer.run_analysis()
