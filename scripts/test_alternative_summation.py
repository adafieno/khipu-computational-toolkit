"""
Alternative Summation Models Tester

Test multiple arithmetic encoding schemes beyond standard pendant-to-parent summation:
1. Modulo-10 summation (Inka decimal system)
2. Base-10 positional with carry
3. Variable tolerance levels (±2, ±5)
4. Cross-level summation (grandparent = all descendants)
5. Partial summation (some groups sum, others don't)

These alternative models may explain the 74% of khipus without detected summation patterns.
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add src directory to path for config import
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore

import pandas as pd  # noqa: E402


class AlternativeSummationTester:
    """Test alternative arithmetic encoding schemes."""
    
    def __init__(self):
        pass
        
    def load_khipu_values(self, khipu_id: str) -> pd.DataFrame:
        """Load all cord values and hierarchy for a khipu."""
        # Load from CSV files
        config = get_config()
        hierarchy = pd.read_csv(config.get_processed_file('cord_hierarchy.csv', 2))
        values = pd.read_csv(config.get_processed_file('cord_numeric_values.csv', 1))
        
        # Convert khipu_id to int for filtering
        khipu_id_int = int(khipu_id)
        
        # Filter for this khipu
        khipu_cords = hierarchy[hierarchy['KHIPU_ID'] == khipu_id_int].copy()
        
        # Merge with numeric values (use lowercase column names)
        df = khipu_cords.merge(
            values[['cord_id', 'numeric_value']], 
            left_on='CORD_ID',
            right_on='cord_id',
            how='left'
        )
        
        # Use PENDANT_FROM as parent (shows which cord this hangs from)
        df['PARENT_ID'] = df['PENDANT_FROM']
        df['POSITION'] = df['CORD_ORDINAL']
        
        # Sort by position
        df = df.sort_values('POSITION')
        
        return df[['CORD_ID', 'PARENT_ID', 'POSITION', 'numeric_value']]
    
    def test_modulo10_summation(self, khipu_id: str) -> Dict:
        """Test if children sum to parent modulo 10."""
        df = self.load_khipu_values(khipu_id)
        
        matches = 0
        total = 0
        
        # Only look at cord-to-cord relationships (not main cord to level 1)
        # Filter for parent IDs that are actually cord IDs
        cord_parents = df[df['PARENT_ID'].isin(df['CORD_ID'])]['PARENT_ID'].unique()
        
        for parent_id in cord_parents:
            children = df[df['PARENT_ID'] == parent_id]
            parent_row = df[df['CORD_ID'] == parent_id]
            
            if len(parent_row) == 0:
                continue
                
            parent_value = parent_row.iloc[0]['numeric_value']
            child_values = children['numeric_value'].dropna()
            
            if len(child_values) == 0 or pd.isna(parent_value):
                continue
                
            total += 1
            child_sum_mod10 = int(child_values.sum()) % 10
            parent_mod10 = int(parent_value) % 10
            
            if child_sum_mod10 == parent_mod10:
                matches += 1
        
        match_rate = matches / total if total > 0 else 0
        
        return {
            'khipu_id': khipu_id,
            'model': 'modulo_10',
            'matches': matches,
            'total': total,
            'match_rate': match_rate,
            'has_pattern': match_rate > 0.5
        }
    
    def test_positional_summation(self, khipu_id: str) -> Dict:
        """Test base-10 positional encoding with carry."""
        df = self.load_khipu_values(khipu_id)
        
        matches = 0
        total = 0
        
        # Only look at cord-to-cord relationships
        cord_parents = df[df['PARENT_ID'].isin(df['CORD_ID'])]['PARENT_ID'].unique()
        
        for parent_id in cord_parents:
            children = df[df['PARENT_ID'] == parent_id]
            parent_row = df[df['CORD_ID'] == parent_id]
            
            if len(parent_row) == 0:
                continue
                
            parent_value = parent_row.iloc[0]['numeric_value']
            child_values = children['numeric_value'].dropna().values
            
            if len(child_values) == 0 or pd.isna(parent_value):
                continue
                
            total += 1
            
            # Test positional: each child position represents power of 10
            positional_sum = sum(val * (10 ** i) for i, val in enumerate(child_values))
            
            if abs(positional_sum - parent_value) <= 1:  # Allow ±1 tolerance
                matches += 1
        
        match_rate = matches / total if total > 0 else 0
        
        return {
            'khipu_id': khipu_id,
            'model': 'positional_base10',
            'matches': matches,
            'total': total,
            'match_rate': match_rate,
            'has_pattern': match_rate > 0.5
        }
    
    def test_variable_tolerance(self, khipu_id: str, tolerance: int = 2) -> Dict:
        """Test standard summation with variable tolerance."""
        df = self.load_khipu_values(khipu_id)
        
        matches = 0
        total = 0
        
        # Only look at cord-to-cord relationships
        cord_parents = df[df['PARENT_ID'].isin(df['CORD_ID'])]['PARENT_ID'].unique()
        
        for parent_id in cord_parents:
            children = df[df['PARENT_ID'] == parent_id]
            parent_row = df[df['CORD_ID'] == parent_id]
            
            if len(parent_row) == 0:
                continue
                
            parent_value = parent_row.iloc[0]['numeric_value']
            child_values = children['numeric_value'].dropna()
            
            if len(child_values) == 0 or pd.isna(parent_value):
                continue
                
            total += 1
            child_sum = child_values.sum()
            
            if abs(child_sum - parent_value) <= tolerance:
                matches += 1
        
        match_rate = matches / total if total > 0 else 0
        
        return {
            'khipu_id': khipu_id,
            'model': f'standard_tolerance_{tolerance}',
            'tolerance': tolerance,
            'matches': matches,
            'total': total,
            'match_rate': match_rate,
            'has_pattern': match_rate > 0.5
        }
    
    def test_cross_level_summation(self, khipu_id: str) -> Dict:
        """Test if grandparents sum all descendants (not just children)."""
        df = self.load_khipu_values(khipu_id)
        
        # Build hierarchy tree
        def get_all_descendants(cord_id):
            """Recursively get all descendant values."""
            children = df[df['PARENT_ID'] == cord_id]
            values = children['numeric_value'].dropna().tolist()
            
            for child_id in children['CORD_ID']:
                values.extend(get_all_descendants(child_id))
            
            return values
        
        matches = 0
        total = 0
        
        # Test each cord that has descendants (not just children)
        for cord_id in df['CORD_ID']:
            cord_row = df[df['CORD_ID'] == cord_id].iloc[0]
            cord_value = cord_row['numeric_value']
            
            if pd.isna(cord_value):
                continue
            
            descendants = get_all_descendants(cord_id)
            
            if len(descendants) < 2:  # Need at least grandchildren
                continue
                
            total += 1
            descendant_sum = sum(descendants)
            
            if abs(descendant_sum - cord_value) <= 1:
                matches += 1
        
        match_rate = matches / total if total > 0 else 0
        
        return {
            'khipu_id': khipu_id,
            'model': 'cross_level_all_descendants',
            'matches': matches,
            'total': total,
            'match_rate': match_rate,
            'has_pattern': match_rate > 0.5
        }
    
    def test_partial_summation(self, khipu_id: str) -> Dict:
        """Detect if only SOME groups show summation patterns."""
        df = self.load_khipu_values(khipu_id)
        
        group_results = []
        
        # Only look at cord-to-cord relationships
        cord_parents = df[df['PARENT_ID'].isin(df['CORD_ID'])]['PARENT_ID'].unique()
        
        for parent_id in cord_parents:
            children = df[df['PARENT_ID'] == parent_id]
            parent_row = df[df['CORD_ID'] == parent_id]
            
            if len(parent_row) == 0:
                continue
                
            parent_value = parent_row.iloc[0]['numeric_value']
            child_values = children['numeric_value'].dropna()
            
            if len(child_values) == 0 or pd.isna(parent_value):
                continue
            
            child_sum = child_values.sum()
            matches = abs(child_sum - parent_value) <= 1
            
            group_results.append({
                'parent_id': parent_id,
                'matches': matches,
                'children_count': len(child_values)
            })
        
        if len(group_results) == 0:
            return {
                'khipu_id': khipu_id,
                'model': 'partial_summation',
                'total_groups': 0,
                'summation_groups': 0,
                'non_summation_groups': 0,
                'is_mixed': False
            }
        
        summation_groups = sum(1 for g in group_results if g['matches'])
        non_summation_groups = len(group_results) - summation_groups
        
        # Mixed if 20-80% of groups show summation
        is_mixed = 0.2 < (summation_groups / len(group_results)) < 0.8
        
        return {
            'khipu_id': khipu_id,
            'model': 'partial_summation',
            'total_groups': len(group_results),
            'summation_groups': summation_groups,
            'non_summation_groups': non_summation_groups,
            'summation_rate': summation_groups / len(group_results) if len(group_results) > 0 else 0,
            'is_mixed': is_mixed
        }
    
    def test_all_models(self, khipu_ids: List[str]) -> pd.DataFrame:
        """Test all alternative models on a list of khipus."""
        print(f"Testing {len(khipu_ids)} khipus with 5 alternative models...")
        
        results = []
        
        for i, khipu_id in enumerate(khipu_ids):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(khipu_ids)}...")
            
            try:
                # Test all models
                results.append(self.test_modulo10_summation(khipu_id))
                results.append(self.test_positional_summation(khipu_id))
                results.append(self.test_variable_tolerance(khipu_id, tolerance=2))
                results.append(self.test_variable_tolerance(khipu_id, tolerance=5))
                results.append(self.test_cross_level_summation(khipu_id))
                
                # Partial summation returns different schema
                partial = self.test_partial_summation(khipu_id)
                results.append({
                    'khipu_id': khipu_id,
                    'model': 'partial_summation',
                    'matches': partial['summation_groups'],
                    'total': partial['total_groups'],
                    'match_rate': partial.get('summation_rate', 0),
                    'has_pattern': partial.get('is_mixed', False)
                })
                
            except Exception as e:
                print(f"  Error processing {khipu_id}: {e}")
                continue
        
        df = pd.DataFrame(results)
        print(f"✓ Tested {len(df)} model × khipu combinations")
        
        return df
    
    def export_results(self, results_df: pd.DataFrame, output_path: str = None):
        """Export results to CSV."""
        if output_path is None:
            config = get_config()
            output_path = config.get_processed_file("alternative_summation_results.csv", 3)
        output_path = Path(output_path)
        results_df.to_csv(output_path, index=False)
        print(f"✓ Exported to {output_path}")
        
        # Generate summary statistics
        summary = results_df.groupby('model').agg({
            'khipu_id': 'count',
            'has_pattern': 'sum',
            'match_rate': 'mean'
        }).round(3)
        
        summary.columns = ['Khipus Tested', 'Khipus with Pattern', 'Avg Match Rate']
        summary['Detection Rate (%)'] = (summary['Khipus with Pattern'] / summary['Khipus Tested'] * 100).round(1)
        
        print("\n" + "="*70)
        print("ALTERNATIVE SUMMATION MODELS SUMMARY")
        print("="*70)
        print(summary)
        
        return summary


def main():
    print("="*80)
    print("ALTERNATIVE SUMMATION MODELS TESTING")
    print("="*80)
    print()
    
    tester = AlternativeSummationTester()
    
    # Get all khipu IDs from cord_hierarchy.csv
    config = get_config()
    hierarchy = pd.read_csv(config.get_processed_file("cord_hierarchy.csv", 2))
    khipu_ids = hierarchy['KHIPU_ID'].unique().tolist()
    
    print(f"Testing {len(khipu_ids)} khipus")
    print()
    
    # Test all models
    results = tester.test_all_models(khipu_ids)
    
    # Export
    summary = tester.export_results(results)
    
    print()
    print("="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Compare to standard model
    standard_results = pd.read_csv(config.get_processed_file("summation_test_results.csv", 3))
    standard_rate = standard_results['has_pendant_summation'].mean() * 100
    
    print(f"\nStandard summation (±1): {standard_rate:.1f}% detection rate")
    print("\nAlternative models:")
    for model in summary.index:
        rate = summary.loc[model, 'Detection Rate (%)']
        improvement = rate - standard_rate
        print(f"  {model}: {rate:.1f}% ({improvement:+.1f}% vs standard)")
    
    print()
    print("="*80)


if __name__ == "__main__":
    main()
