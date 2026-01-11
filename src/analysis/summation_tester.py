"""
Summation Hypothesis Tester - Test arithmetic summation patterns in khipus.

Based on Medrano & Khosla 2024 findings:
- ~74% of khipus show summation consistency
- White cords act as boundary markers between sum groups
- Pendant cords often sum to values on parent/sibling cords

This module tests these hypotheses systematically across all khipus.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime


class SummationTester:
    """Test arithmetic summation hypotheses on khipu data."""
    
    def __init__(self, db_path: Path):
        """Initialize tester with database path."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
    
    def get_cord_numeric_value(self, cord_id: int) -> Optional[int]:
        """
        Get the numeric value encoded on a cord using Ascher notation.
        Returns None if cord has no numeric encoding.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all knots on this cord
        cursor.execute("""
            SELECT TYPE_CODE, NUM_TURNS
            FROM knot
            WHERE CORD_ID = ?
            ORDER BY KNOT_ORDINAL
        """, (cord_id,))
        
        knots = cursor.fetchall()
        conn.close()
        
        if not knots:
            return None
        
        total = 0
        has_value = False
        
        for knot_type, num_turns in knots:
            has_value = True
            
            # Ascher & Ascher positional notation
            if knot_type == 'L':  # Long knot - tens position
                if num_turns is not None and num_turns > 0:
                    total += int(num_turns) * 10
            elif knot_type == 'S':  # Single knot - hundreds position
                total += 1 * 100
            elif knot_type == 'E':  # Figure-eight - units position
                total += 1 * 1
        
        return total if has_value else None
    
    def get_khipu_cord_values(self, khipu_id: int) -> pd.DataFrame:
        """
        Get all cord values for a khipu with hierarchy information.
        
        Returns DataFrame with:
        - cord_id
        - numeric_value
        - cord_classification
        - cord_level
        - pendant_from
        - attached_to
        - cord_ordinal
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get all cords with structure
        query = """
        SELECT 
            CORD_ID,
            CORD_CLASSIFICATION,
            CORD_LEVEL,
            PENDANT_FROM,
            ATTACHED_TO,
            CORD_ORDINAL
        FROM cord
        WHERE KHIPU_ID = ?
        ORDER BY CORD_LEVEL, CORD_ORDINAL
        """
        
        df = pd.read_sql_query(query, conn, params=(khipu_id,))
        conn.close()
        
        print(f"  Processing {len(df)} cords...")
        
        # Get numeric value for each cord
        df['numeric_value'] = df['CORD_ID'].apply(self.get_cord_numeric_value)
        
        return df
    
    def test_pendant_summation(self, khipu_id: int) -> Dict:
        """
        Test if pendant cords sum to their parent/sibling cord values.
        
        Common pattern: pendants hanging from a cord sum to that cord's value.
        """
        df = self.get_khipu_cord_values(khipu_id)
        
        # Filter to cords with numeric values
        df = df[df['numeric_value'].notna()].copy()
        
        if len(df) == 0:
            return {
                'khipu_id': khipu_id,
                'has_summation': False,
                'reason': 'no_numeric_data'
            }
        
        summation_groups = []
        
        # For each cord that has pendants
        for _, parent in df.iterrows():
            parent_id = parent['CORD_ID']
            
            # Find all pendants hanging from this cord
            pendants = df[df['PENDANT_FROM'] == parent_id]
            
            if len(pendants) == 0:
                continue
            
            # Calculate sum of pendant values
            pendant_sum = pendants['numeric_value'].sum()
            parent_value = parent['numeric_value']
            
            if parent_value is None:
                continue
            
            # Check if sum matches
            matches = (pendant_sum == parent_value)
            
            summation_groups.append({
                'parent_cord': int(parent_id),
                'parent_value': int(parent_value),
                'num_pendants': int(len(pendants)),
                'pendant_sum': int(pendant_sum),
                'matches': bool(matches),
                'difference': int(abs(pendant_sum - parent_value))
            })
        
        if len(summation_groups) == 0:
            return {
                'khipu_id': khipu_id,
                'has_summation': False,
                'reason': 'no_pendant_groups'
            }
        
        # Calculate statistics
        matching_groups = sum(1 for g in summation_groups if g['matches'])
        
        return {
            'khipu_id': khipu_id,
            'has_summation': bool(matching_groups > 0),
            'total_groups': int(len(summation_groups)),
            'matching_groups': int(matching_groups),
            'match_rate': float(matching_groups / len(summation_groups)),
            'groups': summation_groups
        }
    
    def find_white_cord_boundaries(self, khipu_id: int) -> List[int]:
        """
        Identify white cords that may act as boundary markers.
        
        Per Medrano & Khosla 2024: white cords often separate arithmetic groups.
        """
        conn = sqlite3.connect(self.db_path)
        
        # Find cords that are primarily white
        # Color code 'W' in the database represents white
        query = """
        SELECT DISTINCT c.CORD_ID, c.CORD_ORDINAL
        FROM cord c
        JOIN ascher_cord_color cc ON c.CORD_ID = cc.CORD_ID
        WHERE c.KHIPU_ID = ?
        AND cc.COLOR_CD_1 = 'W'
        ORDER BY c.CORD_ORDINAL
        """
        
        cursor = conn.cursor()
        cursor.execute(query, (khipu_id,))
        white_cords = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return white_cords
    
    def test_boundary_summation(self, khipu_id: int) -> Dict:
        """
        Test if cords between white boundaries form summation groups.
        
        Hypothesis: White cords separate groups where cords within each group
        have arithmetic relationships.
        """
        white_boundaries = self.find_white_cord_boundaries(khipu_id)
        df = self.get_khipu_cord_values(khipu_id)
        
        if len(white_boundaries) == 0:
            return {
                'khipu_id': khipu_id,
                'has_boundaries': False,
                'reason': 'no_white_cords'
            }
        
        # Filter to cords with numeric values
        df = df[df['numeric_value'].notna()].copy()
        
        if len(df) == 0:
            return {
                'khipu_id': khipu_id,
                'has_boundaries': True,
                'num_boundaries': len(white_boundaries),
                'has_summation': False,
                'reason': 'no_numeric_data'
            }
        
        # Split cords into groups based on white boundaries
        # This is a simplified version - more sophisticated grouping may be needed
        boundary_positions = sorted([
            df[df['CORD_ID'] == bid]['CORD_ORDINAL'].iloc[0] 
            for bid in white_boundaries 
            if bid in df['CORD_ID'].values
        ])
        
        # For now, just return boundary info
        # Full implementation would analyze summation within groups
        return {
            'khipu_id': khipu_id,
            'has_boundaries': True,
            'num_boundaries': len(white_boundaries),
            'boundary_cords': white_boundaries,
            'num_groups': len(boundary_positions) + 1,
            'requires_detailed_analysis': True
        }
    
    def test_all_khipus(self, output_path: Path) -> pd.DataFrame:
        """
        Test summation hypotheses across all khipus.
        
        Returns summary DataFrame with results per khipu.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT KHIPU_ID FROM khipu_main ORDER BY KHIPU_ID")
        khipu_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        print(f"Testing {len(khipu_ids)} khipus for summation patterns...")
        print()
        
        results = []
        
        for i, khipu_id in enumerate(khipu_ids, 1):
            if i % 50 == 0:
                print(f"  Tested {i}/{len(khipu_ids)} khipus...")
            
            # Test pendant summation
            pendant_result = self.test_pendant_summation(khipu_id)
            
            # Test white boundary summation
            boundary_result = self.test_boundary_summation(khipu_id)
            
            results.append({
                'khipu_id': khipu_id,
                'has_pendant_summation': pendant_result.get('has_summation', False),
                'pendant_match_rate': pendant_result.get('match_rate', 0.0),
                'num_pendant_groups': pendant_result.get('total_groups', 0),
                'has_white_boundaries': boundary_result.get('has_boundaries', False),
                'num_white_boundaries': boundary_result.get('num_boundaries', 0),
                'pendant_details': pendant_result,
                'boundary_details': boundary_result
            })
        
        print(f"  ✓ Tested {len(khipu_ids)} khipus")
        print()
        
        df = pd.DataFrame(results)
        
        # Export results
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("  Writing results...")
        df[['khipu_id', 'has_pendant_summation', 'pendant_match_rate', 
            'num_pendant_groups', 'has_white_boundaries', 'num_white_boundaries']].to_csv(
            output_path, index=False
        )
        
        # Export detailed JSON
        detailed_results = {
            'generated_at': datetime.now().isoformat(),
            'source_database': str(self.db_path),
            'total_khipus': len(df),
            'khipus_with_pendant_summation': int(df['has_pendant_summation'].sum()),
            'khipus_with_white_boundaries': int(df['has_white_boundaries'].sum()),
            'average_pendant_match_rate': float(df['pendant_match_rate'].mean()),
            'results': results
        }
        
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print("  ✓ Results written")
        
        return df
