"""
Arithmetic validation for khipu numeric encoding.

Tests summation consistency following established conventions:
- Pendant cords should sum to expected totals
- Knot clusters represent positional decimal numbers
- Internal arithmetic relationships should be consistent

Based on findings from Medrano & Khosla (2024) and Ascher & Ascher.

IMPORTANT: Uses ValueComputer for correct value computation with
CLUSTER_ORDINAL clustering (not fixed position decoding).
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json
import pandas as pd
from datetime import datetime
import sys

# Import value computation
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.value_computation import ValueComputer


class SummationType(Enum):
    """Types of summation relationships found in khipus."""
    SIMPLE_SUM = "simple_sum"  # Sum of pendants = total
    NESTED_SUM = "nested_sum"  # Hierarchical summation
    DIFFERENCE = "difference"   # Subtraction relationship
    UNKNOWN = "unknown"


@dataclass
class KnotValue:
    """Represents a decoded knot value with metadata."""
    cord_id: int
    knot_id: int
    value: Optional[int]
    knot_type: str
    position: Optional[float]
    cluster_id: int
    confidence: float = 1.0  # 0.0 to 1.0


@dataclass
class CordValue:
    """Represents the total numeric value of a cord."""
    cord_id: int
    total_value: Optional[int]
    knot_clusters: List[int]
    confidence: float = 1.0
    validation_notes: str = ""


@dataclass
class SummationTest:
    """Result of testing a summation relationship."""
    khipu_id: int
    summation_type: SummationType
    expected_sum: Optional[int]
    actual_sum: Optional[int]
    matches: bool
    tolerance: int = 0
    cord_ids: List[int] = None
    confidence: float = 1.0
    notes: str = ""


class ArithmeticValidator:
    """
    Validate arithmetic consistency in khipus.
    
    Tests whether knot encodings follow established numeric conventions
    and whether summation relationships hold.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize validator.
        
        Args:
            db_path: Path to khipu.db
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "khipu.db"
        
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        # Initialize value computer for correct value calculations
        self.value_computer = ValueComputer(str(self.db_path))
    
    def _connect(self) -> sqlite3.Connection:
        """Create database connection."""
        return sqlite3.connect(self.db_path)
    
    def get_cord_numeric_value(self, cord_id: int) -> CordValue:
        """
        Extract the numeric value encoded on a cord using correct CLUSTER_ORDINAL clustering.
        
        Delegates to ValueComputer for proper computation.
        
        Args:
            cord_id: The cord to analyze
            
        Returns:
            CordValue with extracted numeric data
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            
            # Get cluster IDs for this cord
            cursor.execute("""
                SELECT DISTINCT CLUSTER_ORDINAL
                FROM knot
                WHERE CORD_ID = ? AND CLUSTER_ORDINAL IS NOT NULL
                ORDER BY CLUSTER_ORDINAL
            """, (cord_id,))
            
            cluster_ordinals = [row[0] for row in cursor.fetchall()]
            
            # Use ValueComputer for correct value calculation
            computed = self.value_computer.get_best_value(str(cord_id))
            
            if not cluster_ordinals:
                return CordValue(
                    cord_id=cord_id,
                    total_value=None,
                    knot_clusters=[],
                    confidence=0.0,
                    validation_notes="No knots found"
                )
            
            return CordValue(
                cord_id=cord_id,
                total_value=int(computed.value) if computed.value is not None else None,
                knot_clusters=cluster_ordinals,
                confidence=computed.confidence,
                validation_notes=f"Computed using {computed.method.value}"
            )
    
    def detect_white_cord_groups(self, khipu_id: int) -> List[Dict]:
        """
        Detect groups of cords separated by white cord boundaries.
        
        MIT-validated approach (per feedback from Ashok Khosla):
        "We don't look at all white cords in our summation analysis. 
        Generally, only white cords that are the first cord in a group 
        are summation cords."
        
        Algorithm:
        1. Get all Level 1 pendants in ordinal order
        2. Identify white cords (using color data)
        3. Group cords between white boundaries
        4. Each group: [white_boundary_cord, pendant1, ..., pendantN]
        
        Args:
            khipu_id: The khipu to analyze
            
        Returns:
            List of group dictionaries:
            {
                'group_id': int,
                'boundary_cord_id': int,  # The white cord
                'boundary_value': Optional[int],
                'member_cord_ids': List[int],  # Cords in this group
                'member_values': List[int],
                'expected_sum': Optional[int],  # Boundary cord value
                'actual_sum': int,  # Sum of member values
                'confidence': float
            }
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            
            # Get all Level 1 pendants with colors
            # Note: Colors can be in COLOR_CD_1 through COLOR_CD_5
            cursor.execute("""
                SELECT c.CORD_ID, c.CORD_ORDINAL, 
                       ac.COLOR_CD_1, ac.COLOR_CD_2, ac.COLOR_CD_3, 
                       ac.COLOR_CD_4, ac.COLOR_CD_5
                FROM cord c
                LEFT JOIN ascher_cord_color ac ON c.CORD_ID = ac.CORD_ID
                WHERE c.KHIPU_ID = ? AND c.CORD_LEVEL = 1
                ORDER BY c.CORD_ORDINAL
            """, (khipu_id,))
            
            cord_data = cursor.fetchall()
            
            if not cord_data:
                return []
            
            # Identify white cords (W appears in any color column)
            white_indices = []
            for i, row in enumerate(cord_data):
                cord_id, ordinal = row[0], row[1]
                colors = [c for c in row[2:7] if c]  # Get non-null color columns
                
                # Check if 'W' appears in any color
                is_white = any('W' in str(c).upper() for c in colors if c)
                if is_white:
                    white_indices.append(i)
            
            if not white_indices:
                return []  # No white boundaries
            
            # Build groups between white boundaries
            groups = []
            for group_idx, white_idx in enumerate(white_indices):
                boundary_cord_id = cord_data[white_idx][0]
                
                # Find end of this group (next white cord or end of list)
                if group_idx + 1 < len(white_indices):
                    end_idx = white_indices[group_idx + 1]
                else:
                    end_idx = len(cord_data)
                
                # Members are cords AFTER the white boundary, before next white
                member_indices = range(white_idx + 1, end_idx)
                member_cord_ids = [cord_data[i][0] for i in member_indices]
                
                if not member_cord_ids:
                    continue  # Empty group
                
                # Get numeric values
                boundary_val = self.get_cord_numeric_value(boundary_cord_id)
                member_vals = []
                member_confidences = []
                
                for member_id in member_cord_ids:
                    val = self.get_cord_numeric_value(member_id)
                    if val.total_value is not None:
                        member_vals.append(val.total_value)
                        member_confidences.append(val.confidence)
                
                if not member_vals:
                    continue  # No numeric data in group
                
                groups.append({
                    'group_id': group_idx,
                    'boundary_cord_id': boundary_cord_id,
                    'boundary_value': boundary_val.total_value,
                    'member_cord_ids': member_cord_ids,
                    'member_values': member_vals,
                    'expected_sum': boundary_val.total_value,
                    'actual_sum': sum(member_vals),
                    'confidence': min([boundary_val.confidence] + member_confidences)
                })
            
            return groups
    
    def test_pendant_summation(
        self,
        khipu_id: int,
        tolerance: int = 1
    ) -> Dict:
        """
        Test Ascher summation hypothesis using MIT-validated methodology.
        
        CORRECTED ALGORITHM (based on MIT feedback, Feb 2026):
        "We don't look at all white cords in our summation analysis. 
        Generally, only white cords that are the first cord in a group 
        are summation cords."
        
        Previous (INCORRECT) approach tested ALL white cords.
        New (CORRECT) approach tests white cord GROUPS.
        
        Algorithm:
        1. Detect cord groups separated by white boundaries
        2. For each group where first cord is white:
           - Sum all pendants in the group
           - Compare to white boundary cord value
           - Mark as match if within tolerance
        3. Compute match rate across all groups
        4. Use 30% threshold per MIT guidance (not 80%)
        
        Args:
            khipu_id: The khipu to test
            tolerance: Allowable difference for "match" (default 1)
            
        Returns:
            Dictionary with:
            {
                'khipu_id': int,
                'has_pendant_summation': bool,  # >30% match rate
                'pendant_match_rate': float,  # 0.0 to 1.0
                'num_pendant_groups': int,
                'num_matches': int,
                'has_white_boundaries': bool,
                'num_white_boundaries': int,
                'groups': List[Dict],  # Detailed group data
                'confidence': float
            }
        """
        # Detect white cord groups
        groups = self.detect_white_cord_groups(khipu_id)
        
        if not groups:
            # No white boundaries found
            return {
                'khipu_id': khipu_id,
                'has_pendant_summation': False,
                'pendant_match_rate': 0.0,
                'num_pendant_groups': 0,
                'num_matches': 0,
                'has_white_boundaries': False,
                'num_white_boundaries': 0,
                'groups': [],
                'confidence': 0.0
            }
        
        # Test each group for summation
        num_matches = 0
        testable_groups = 0
        
        for group in groups:
            if group['expected_sum'] is not None:
                testable_groups += 1
                diff = abs(group['expected_sum'] - group['actual_sum'])
                if diff <= tolerance:
                    num_matches += 1
                    group['matches'] = True
                else:
                    group['matches'] = False
            else:
                group['matches'] = None  # Untestable (no boundary value)
        
        # Compute match rate
        if testable_groups > 0:
            match_rate = num_matches / testable_groups
        else:
            match_rate = 0.0
        
        # MIT guidance: 30% occurrence = "interesting sign"
        has_summation = match_rate >= 0.30
        
        # Overall confidence (minimum across all groups)
        confidence = min([g['confidence'] for g in groups]) if groups else 0.0
        
        return {
            'khipu_id': khipu_id,
            'has_pendant_summation': has_summation,
            'pendant_match_rate': match_rate,
            'num_pendant_groups': testable_groups,
            'num_matches': num_matches,
            'has_white_boundaries': True,
            'num_white_boundaries': len(groups),
            'groups': groups,
            'confidence': confidence
        }
    
    def validate_khipu_arithmetic(self, khipu_id: int) -> Dict:
        """
        Comprehensive arithmetic validation for a khipu.
        
        Tests multiple hypotheses:
        - Pendant summation patterns
        - Cluster consistency
        - Hierarchical relationships
        
        Args:
            khipu_id: The khipu to validate
            
        Returns:
            Dictionary with validation results and confidence scores
        """
        results = {
            'khipu_id': khipu_id,
            'has_numeric_data': False,
            'summation_tests': [],
            'cord_values': {},
            'overall_confidence': 0.0,
            'validation_notes': []
        }
        
        with self._connect() as conn:
            cursor = conn.cursor()
            
            # Get cord count
            cursor.execute(
                "SELECT COUNT(*) FROM cord WHERE KHIPU_ID = ?",
                (khipu_id,)
            )
            cord_count = cursor.fetchone()[0]
            
            if cord_count == 0:
                results['validation_notes'].append("No cords found")
                return results
            
            # Get all cords and their values
            cursor.execute("""
                SELECT CORD_ID FROM cord 
                WHERE KHIPU_ID = ?
                ORDER BY CORD_LEVEL, CORD_ORDINAL
            """, (khipu_id,))
            
            cord_ids = [row[0] for row in cursor.fetchall()]
            
            values_found = 0
            total_confidence = 0.0
            
            for cord_id in cord_ids:
                cord_val = self.get_cord_numeric_value(cord_id)
                results['cord_values'][cord_id] = {
                    'value': cord_val.total_value,
                    'confidence': cord_val.confidence,
                    'notes': cord_val.validation_notes
                }
                
                if cord_val.total_value is not None:
                    values_found += 1
                    total_confidence += cord_val.confidence
            
            results['has_numeric_data'] = values_found > 0
            
            if values_found > 0:
                results['overall_confidence'] = total_confidence / values_found
            
            # Test summation patterns
            summation_tests = self.test_pendant_summation(khipu_id)
            results['summation_tests'] = [
                {
                    'type': t.summation_type.value,
                    'expected': t.expected_sum,
                    'actual': t.actual_sum,
                    'matches': t.matches,
                    'confidence': t.confidence,
                    'notes': t.notes
                }
                for t in summation_tests
            ]
            
            results['validation_notes'].append(
                f"Found {values_found}/{len(cord_ids)} cords with numeric values"
            )
            
            return results
    
    def identify_validated_khipus(
        self,
        min_confidence: float = 0.7,
        require_summation: bool = False
    ) -> List[int]:
        """
        Identify khipus suitable for use as validated test set.
        
        Args:
            min_confidence: Minimum overall confidence score (0-1)
            require_summation: Whether to require summation patterns
            
        Returns:
            List of KHIPU_IDs that pass validation criteria
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT KHIPU_ID FROM khipu_main ORDER BY KHIPU_ID")
            all_khipu_ids = [row[0] for row in cursor.fetchall()]
        
        validated = []
        
        for khipu_id in all_khipu_ids:
            result = self.validate_khipu_arithmetic(khipu_id)
            
            if not result['has_numeric_data']:
                continue
            
            if result['overall_confidence'] < min_confidence:
                continue
            
            if require_summation and not result['summation_tests']:
                continue
            
    
    def export_cord_values(self, output_path: Path, khipu_ids: Optional[List[int]] = None):
        """
        Export decoded cord numeric values to CSV.
        
        Args:
            output_path: Where to save the CSV file
            khipu_ids: Optional list of specific khipus (default: all)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._connect() as conn:
            cursor = conn.cursor()
            
            if khipu_ids:
                placeholders = ','.join('?' * len(khipu_ids))
                query = f"SELECT CORD_ID, KHIPU_ID FROM cord WHERE KHIPU_ID IN ({placeholders})"
                cursor.execute(query, khipu_ids)
            else:
                cursor.execute("SELECT CORD_ID, KHIPU_ID FROM cord")
            
            cords = cursor.fetchall()
        
        records = []
        total_cords = len(cords)
        print(f"Processing {total_cords:,} cords...")
        
        for i, (cord_id, khipu_id) in enumerate(cords, 1):
            if i % 1000 == 0:
                print(f"  Processed {i:,}/{total_cords:,} cords...")
            
            cord_val = self.get_cord_numeric_value(cord_id)
            records.append({
                'khipu_id': khipu_id,
                'cord_id': cord_id,
                'numeric_value': cord_val.total_value,
                'confidence': cord_val.confidence,
                'num_clusters': len(cord_val.knot_clusters),
                'validation_notes': cord_val.validation_notes
            })
        
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        
        # Create metadata file
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'source_database': str(self.db_path),
            'total_cords': len(records),
            'cords_with_values': len([r for r in records if r['numeric_value'] is not None]),
            'khipu_count': len(df['khipu_id'].unique()),
            'decoding_method': 'Ascher_Ascher_positional_notation',
            'formula': 'S=100, L=(NUM_TURNS×10), E=1',
            'notes': 'S (single) = hundreds, L (long) = tens, E (figure-eight) = units'
        }
        
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return df
    
    def export_validation_results(self, output_path: Path, khipu_ids: Optional[List[int]] = None):
        """
        Export comprehensive validation results to JSON.
        
        Args:
            output_path: Where to save the JSON file
            khipu_ids: Optional list of specific khipus (default: all)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._connect() as conn:
            cursor = conn.cursor()
            
            if khipu_ids:
                placeholders = ','.join('?' * len(khipu_ids))
                query = f"SELECT KHIPU_ID FROM khipu_main WHERE KHIPU_ID IN ({placeholders})"
                cursor.execute(query, khipu_ids)
            else:
                cursor.execute("SELECT KHIPU_ID FROM khipu_main")
            
            all_khipu_ids = [row[0] for row in cursor.fetchall()]
        
        results = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'source_database': str(self.db_path),
                'khipu_count': len(all_khipu_ids),
                'validation_method': 'arithmetic_consistency'
            },
            'khipus': {}
        }
        
        total_khipus = len(all_khipu_ids)
        print(f"Validating {total_khipus} khipus...")
        
        for i, khipu_id in enumerate(all_khipu_ids, 1):
            if i % 10 == 0:
                print(f"  Validated {i}/{total_khipus} khipus...")
            
            validation = self.validate_khipu_arithmetic(khipu_id)
            results['khipus'][str(khipu_id)] = validation
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


if __name__ == "__main__":
    # Example usage
    validator = ArithmeticValidator()
    
    print("Testing arithmetic validation on first 5 khipus...")
    print("=" * 80)
    
    with sqlite3.connect(validator.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT KHIPU_ID FROM khipu_main LIMIT 5")
        test_khipus = [row[0] for row in cursor.fetchall()]
    
    for khipu_id in test_khipus:
        print(f"\nKhipu {khipu_id}:")
        print("-" * 80)
        
        result = validator.validate_khipu_arithmetic(khipu_id)
        print(f"  Has numeric data: {result['has_numeric_data']}")
        print(f"  Overall confidence: {result['overall_confidence']:.2f}")
        print(f"  Cords with values: {len([v for v in result['cord_values'].values() if v['value'] is not None])}/{len(result['cord_values'])}")
        
        if result['summation_tests']:
            print(f"  Summation tests: {len(result['summation_tests'])}")
            for test in result['summation_tests']:
                print(f"    - {test['type']}: sum={test['actual']}, confidence={test['confidence']:.2f}")
