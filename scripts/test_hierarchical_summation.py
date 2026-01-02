"""
Test Hierarchical Summation Hypotheses

This script tests multi-level recursive summation patterns where:
1. Level 1 pendants sum to primary cord
2. Level 2 subsidiaries sum to their level 1 parent
3. Level 3+ continue the recursive pattern

This extends the basic pendant-to-parent summation to test whether
hierarchical accounting structures follow consistent summation rules
at multiple levels of the cord tree.
"""

import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add src directory to path for config import
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore

import pandas as pd  # noqa: E402
import sqlite3  # noqa: E402
import json  # noqa: E402


class HierarchicalSummationTester:
    """Test multi-level recursive summation patterns in khipus."""
    
    def __init__(self, db_path: str = None):
        config = get_config()
        self.config = config
        self.db_path = db_path if db_path else config.get_database_path()
        self.conn = sqlite3.connect(self.db_path)
        
    def get_khipu_hierarchy(self, khipu_id: int) -> pd.DataFrame:
        """Get complete cord hierarchy with numeric values for a khipu."""
        query = """
        SELECT 
            c.CORD_ID,
            c.KHIPU_ID,
            c.CORD_LEVEL,
            c.PENDANT_FROM,
            c.ATTACHED_TO,
            c.CORD_ORDINAL,
            cn.numeric_value,
            cn.confidence
        FROM cord c
        LEFT JOIN (
            SELECT CORD_ID, numeric_value, confidence
            FROM cord_numeric_values
        ) cn ON c.CORD_ID = cn.CORD_ID
        WHERE c.KHIPU_ID = ?
        ORDER BY c.CORD_LEVEL, c.CORD_ORDINAL
        """
        
        df = pd.read_sql_query(query, self.conn, params=[khipu_id])
        return df
    
    def build_hierarchy_tree(self, df: pd.DataFrame) -> dict:
        """Build hierarchical tree structure."""
        tree = defaultdict(list)
        
        for _, row in df.iterrows():
            parent = row['PENDANT_FROM'] if pd.notna(row['PENDANT_FROM']) else row['ATTACHED_TO']
            if pd.notna(parent):
                tree[int(parent)].append({
                    'cord_id': row['CORD_ID'],
                    'level': row['CORD_LEVEL'],
                    'value': row['numeric_value'] if pd.notna(row['numeric_value']) else None,
                    'confidence': row['confidence'] if pd.notna(row['confidence']) else 0.0
                })
        
        return tree
    
    def test_level_summation(self, parent_value: float, children: list, tolerance: float = 1.0) -> dict:
        """Test if children sum to parent value."""
        # Filter children with numeric values
        valid_children = [c for c in children if c['value'] is not None]
        
        if not valid_children or parent_value is None:
            return {
                'tested': False,
                'match': False,
                'parent_value': parent_value,
                'child_count': len(valid_children),
                'child_sum': None,
                'difference': None
            }
        
        child_sum = sum(c['value'] for c in valid_children)
        difference = abs(parent_value - child_sum)
        match = difference <= tolerance
        
        return {
            'tested': True,
            'match': match,
            'parent_value': parent_value,
            'child_count': len(valid_children),
            'child_sum': child_sum,
            'difference': difference,
            'avg_child_confidence': sum(c['confidence'] for c in valid_children) / len(valid_children)
        }
    
    def test_hierarchical_summation(self, khipu_id: int) -> dict:
        """Test multi-level summation for a single khipu."""
        df = self.get_khipu_hierarchy(khipu_id)
        
        if len(df) == 0:
            return {'khipu_id': khipu_id, 'error': 'No cords found'}
        
        tree = self.build_hierarchy_tree(df)
        
        # Get cord values lookup
        cord_values = df.set_index('CORD_ID')['numeric_value'].to_dict()
        
        # Test summation at each level
        level_results = defaultdict(list)
        
        for parent_id, children in tree.items():
            parent_value = cord_values.get(parent_id)
            
            if parent_value is not None and len(children) > 0:
                # Determine level from first child (all children same level)
                level = children[0]['level']
                
                result = self.test_level_summation(parent_value, children)
                if result['tested']:
                    level_results[level].append(result)
        
        # Aggregate results by level
        level_stats = {}
        for level, results in level_results.items():
            matches = sum(1 for r in results if r['match'])
            tested = len(results)
            
            level_stats[f'level_{level}'] = {
                'tests_performed': tested,
                'matches': matches,
                'match_rate': matches / tested if tested > 0 else 0.0,
                'avg_difference': sum(r['difference'] for r in results) / tested if tested > 0 else 0.0,
                'avg_confidence': sum(r['avg_child_confidence'] for r in results) / tested if tested > 0 else 0.0
            }
        
        # Overall statistics
        all_tests = sum(s['tests_performed'] for s in level_stats.values())
        all_matches = sum(s['matches'] for s in level_stats.values())
        
        return {
            'khipu_id': khipu_id,
            'total_cords': len(df),
            'max_level': int(df['CORD_LEVEL'].max()) if len(df) > 0 else 0,
            'total_tests': all_tests,
            'total_matches': all_matches,
            'overall_match_rate': all_matches / all_tests if all_tests > 0 else 0.0,
            'has_multi_level_summation': len(level_stats) > 1,
            'levels_tested': len(level_stats),
            'level_statistics': level_stats
        }
    
    def test_all_khipus(self, khipu_ids: list = None) -> list:
        """Test hierarchical summation for all khipus or specified subset."""
        if khipu_ids is None:
            # Get all khipu IDs
            query = "SELECT DISTINCT KHIPU_ID FROM khipu_main ORDER BY KHIPU_ID"
            khipu_ids = pd.read_sql_query(query, self.conn)['KHIPU_ID'].tolist()
        
        print(f"Testing hierarchical summation for {len(khipu_ids)} khipus...")
        
        results = []
        for i, khipu_id in enumerate(khipu_ids, 1):
            if i % 50 == 0:
                print(f"  Tested {i}/{len(khipu_ids)} khipus...")
            
            result = self.test_hierarchical_summation(khipu_id)
            results.append(result)
        
        print(f"✓ Completed testing {len(results)} khipus")
        return results
    
    def analyze_results(self, results: list) -> dict:
        """Analyze hierarchical summation test results."""
        df = pd.DataFrame(results)
        
        # Filter khipus that had tests performed
        tested_df = df[df['total_tests'] > 0]
        
        # Identify multi-level summation khipus
        multi_level = tested_df[tested_df['has_multi_level_summation']]
        
        # Identify khipus with high match rates
        high_match_threshold = 0.8
        high_match = tested_df[tested_df['overall_match_rate'] >= high_match_threshold]
        
        # Compute statistics
        stats = {
            'total_khipus': len(df),
            'khipus_tested': len(tested_df),
            'khipus_with_multi_level': len(multi_level),
            'pct_multi_level': (len(multi_level) / len(tested_df)) * 100 if len(tested_df) > 0 else 0,
            'khipus_high_match': len(high_match),
            'pct_high_match': (len(high_match) / len(tested_df)) * 100 if len(tested_df) > 0 else 0,
            'avg_overall_match_rate': tested_df['overall_match_rate'].mean() if len(tested_df) > 0 else 0,
            'avg_levels_tested': tested_df['levels_tested'].mean() if len(tested_df) > 0 else 0,
            'max_levels_tested': tested_df['levels_tested'].max() if len(tested_df) > 0 else 0
        }
        
        # Analyze match rates by level
        level_analysis = self._analyze_by_level(results)
        
        return {
            'summary_statistics': stats,
            'level_analysis': level_analysis,
            'multi_level_khipus': multi_level['khipu_id'].tolist(),
            'high_match_khipus': high_match['khipu_id'].tolist()
        }
    
    def _analyze_by_level(self, results: list) -> dict:
        """Analyze match rates by hierarchy level."""
        level_stats = defaultdict(lambda: {'tests': 0, 'matches': 0, 'khipu_count': 0})
        
        for result in results:
            if 'level_statistics' in result:
                for level_key, stats in result['level_statistics'].items():
                    level_stats[level_key]['tests'] += stats['tests_performed']
                    level_stats[level_key]['matches'] += stats['matches']
                    level_stats[level_key]['khipu_count'] += 1
        
        # Compute match rates
        level_match_rates = {}
        for level, stats in level_stats.items():
            if stats['tests'] > 0:
                level_match_rates[level] = {
                    'total_tests': stats['tests'],
                    'total_matches': stats['matches'],
                    'match_rate': stats['matches'] / stats['tests'],
                    'khipu_count': stats['khipu_count']
                }
        
        return level_match_rates
    
    def export_results(self, results: list, analysis: dict, output_dir: str = None):
        """Export hierarchical summation test results."""
        if output_dir is None:
            output_dir = self.config.processed_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Export detailed results CSV
        df = pd.DataFrame(results)
        output_csv = Path(output_dir) / "hierarchical_summation_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Exported results to {output_csv}")
        
        # Export analysis JSON
        output_json = Path(output_dir) / "hierarchical_summation_analysis.json"
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'analysis': analysis,
            'detailed_results': results
        }
        
        with open(output_json, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"✓ Exported analysis to {output_json}")
    
    def run_analysis(self):
        """Run complete hierarchical summation analysis."""
        print("="*80)
        print("HIERARCHICAL SUMMATION HYPOTHESIS TESTING")
        print("="*80)
        
        # Test all khipus
        results = self.test_all_khipus()
        
        # Analyze results
        print("\n" + "-"*80)
        print("ANALYSIS RESULTS")
        print("-"*80)
        analysis = self.analyze_results(results)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        for key, value in analysis['summary_statistics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        # Print level analysis
        print("\nMatch Rates by Hierarchy Level:")
        for level, stats in sorted(analysis['level_analysis'].items()):
            print(f"  {level}:")
            print(f"    Tests: {stats['total_tests']}")
            print(f"    Matches: {stats['total_matches']}")
            print(f"    Match Rate: {stats['match_rate']:.3f}")
            print(f"    Khipus: {stats['khipu_count']}")
        
        # Export results
        self.export_results(results, analysis)
        
        print("\n" + "="*80)
        print("HIERARCHICAL SUMMATION TESTING COMPLETE")
        print("="*80)
        
        return results, analysis
    
    def __del__(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()


if __name__ == "__main__":
    # Need to create a view or load cord_numeric_values into database
    # For now, load from CSV
    
    config = get_config()
    
    # Load cord numeric values
    print("Loading cord numeric values...")
    cord_values = pd.read_csv(config.get_processed_file("cord_numeric_values.csv"))
    
    # Connect to database and create temporary table
    conn = sqlite3.connect(config.get_database_path())
    cord_values.to_sql('cord_numeric_values', conn, if_exists='replace', index=False)
    conn.close()
    print("✓ Loaded numeric values into temporary table\n")
    
    # Run analysis
    tester = HierarchicalSummationTester()
    tester.run_analysis()
