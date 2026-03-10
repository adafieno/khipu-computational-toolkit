"""
Knot Extractor - Extract knot details with positional and numeric validation.

Handles:
- Knot positions and ordering (CLUSTER_ORDINAL - correct schema!)
- Knot types (L=long, E=figure-eight, S=single)
- Numeric decoding via value_computation module (correct clustering logic)
- Data quality validation

IMPORTANT: This module now delegates value computation to value_computation.py
which implements correct Ascher decimal clustering by CLUSTER_ORDINAL.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import json
from datetime import datetime
import sys

# Import value computation module
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.value_computation import ValueComputer


class KnotExtractor:
    """Extract and validate knot data with positional information."""
    
    def __init__(self, db_path: Path):
        """Initialize extractor with database path."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        # Initialize value computer for correct value calculations
        self.value_computer = ValueComputer(str(db_path))
    
    def get_cord_knots(self, cord_id: int) -> pd.DataFrame:
        """
        Extract all knots for a specific cord with decoded values.
        
        Returns DataFrame with columns:
        - knot_id: Unique knot identifier
        - cord_id: Parent cord
        - cluster_ordinal: Decimal position (1.0=units, 2.0=tens, 3.0=hundreds)
        - knot_type: L (long), E (figure-eight), S (single)
        - num_turns: Digit value (for long knots)
        - numeric_value: Decoded cord value (uses ValueComputer for correct clustering)
        - confidence: Data completeness score
        """
        conn = sqlite3.connect(self.db_path)
        
        # CORRECTED: Use CLUSTER_ORDINAL (not KNOT_ORDINAL which doesn't exist!)
        query = """
        SELECT 
            KNOT_ID,
            CORD_ID,
            CLUSTER_ORDINAL,
            TYPE_CODE as knot_type,
            NUM_TURNS,
            DIRECTION
        FROM knot
        WHERE CORD_ID = ?
        ORDER BY CLUSTER_ORDINAL DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(cord_id,))
        conn.close()
        
        # Get correct numeric value using ValueComputer
        # (handles clustering by CLUSTER_ORDINAL properly)
        computed_value = self.value_computer.get_best_value(str(cord_id))
        df['numeric_value'] = computed_value.value if not df.empty else None
        df['confidence'] = computed_value.confidence if not df.empty else 0.0
        
        return df
    
    def get_all_knots(self) -> pd.DataFrame:
        """
        Extract all knots across all khipus with decoded values.
        
        Returns comprehensive dataset for analysis.
        
        NOTE: For cord-level values, use get_cord_values() which properly
        clusters knots by CLUSTER_ORDINAL. This method returns individual
        knot records for inspection.
        """
        conn = sqlite3.connect(self.db_path)
        
        print("Extracting all knots from database...")
        
        # CORRECTED: Use CLUSTER_ORDINAL (not KNOT_ORDINAL)
        query = """
        SELECT 
            k.KNOT_ID,
            k.CORD_ID,
            k.CLUSTER_ORDINAL,
            k.TYPE_CODE as knot_type,
            k.NUM_TURNS,
            k.DIRECTION,
            c.KHIPU_ID,
            c.CORD_LEVEL
        FROM knot k
        JOIN cord c ON k.CORD_ID = c.CORD_ID
        ORDER BY c.KHIPU_ID, k.CORD_ID, k.CLUSTER_ORDINAL DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"  ✓ SQL query complete: {len(df):,} knots extracted")
        print("  Note: For cord values, use get_cord_values() for proper clustering")
        
        return df
    
    def get_cord_values(self, khipu_id: Optional[int] = None, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Get computed values for all cords using correct clustering logic.
        
        This delegates to ValueComputer which properly:
        - Groups knots by CLUSTER_ORDINAL (1.0, 2.0, 3.0...)
        - Handles multiple knots in same cluster correctly
        - Counts single knots instead of assuming they're always 100
        - Uses NUM_TURNS for long string knots
        - Gracefully handles missing CLUSTER_ORDINAL values
        
        Args:
            khipu_id: Optional khipu ID to filter (None = all cords)
            sample_size: Optional limit on number of cords (for testing/stats)
            
        Returns:
            DataFrame with columns:
            - cord_id: Cord identifier
            - khipu_id: Khipu identifier
            - numeric_value: Correctly computed value
            - confidence: Computation confidence (0.0-1.0)
            - method: Computation method used
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get all cords (optionally filtered by khipu)
        if khipu_id is not None:
            query = "SELECT CORD_ID, KHIPU_ID FROM cord WHERE KHIPU_ID = ? ORDER BY CORD_ORDINAL"
            cords_df = pd.read_sql_query(query, conn, params=(khipu_id,))
        else:
            query = "SELECT CORD_ID, KHIPU_ID FROM cord ORDER BY KHIPU_ID, CORD_ORDINAL"
            cords_df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        # Apply sample limit if specified (for testing large datasets)
        if sample_size is not None and len(cords_df) > sample_size:
            print(f"Sampling {sample_size:,} of {len(cords_df):,} cords for statistics...")
            cords_df = cords_df.head(sample_size)
        else:
            print(f"Computing values for {len(cords_df):,} cords...")
        
        # Compute value for each cord using ValueComputer
        results = []
        for i, row in enumerate(cords_df.itertuples(), 1):
            if i % 1000 == 0:
                print(f"  Progress: {i:,}/{len(cords_df):,} cords ({100*i/len(cords_df):.1f}%)")
            
            cord_id = str(row.CORD_ID)
            try:
                computed = self.value_computer.get_best_value(cord_id)
                
                results.append({
                    'cord_id': cord_id,
                    'khipu_id': row.KHIPU_ID,
                    'numeric_value': computed.value,
                    'confidence': computed.confidence,
                    'method': computed.method.value
                })
            except Exception as e:
                # Log error but continue processing
                results.append({
                    'cord_id': cord_id,
                    'khipu_id': row.KHIPU_ID,
                    'numeric_value': 0.0,
                    'confidence': 0.0,
                    'method': 'error'
                })
        
        print(f"  ✓ Computed {len(results):,} cord values")
        
        return pd.DataFrame(results)
    
    def get_knot_clusters(self, khipu_id: int) -> pd.DataFrame:
        """
        Get knot clusters (groups of knots) for a khipu.
        
        Clusters represent groups of knots at the same position level.
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            kc.CLUSTER_ID,
            kc.KHIPU_ID,
            kc.CORD_ID,
            kc.TOTAL_VALUE,
            kc.CLUSTER_ORDINAL,
            COUNT(k.KNOT_ID) as num_knots
        FROM knot_cluster kc
        LEFT JOIN knot k ON kc.CLUSTER_ID = k.CLUSTER_ID
        WHERE kc.KHIPU_ID = ?
        GROUP BY kc.CLUSTER_ID
        ORDER BY kc.CORD_ID, kc.CLUSTER_ORDINAL
        """
        
        df = pd.read_sql_query(query, conn, params=(khipu_id,))
        conn.close()
        
        return df
    
    def export_knot_data(self, output_path: Path, khipu_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Export knot data to CSV with metadata JSON.
        
        Args:
            output_path: Path for CSV output
            khipu_ids: Optional list of khipu IDs to export (None = all)
        
        Returns:
            DataFrame of exported knots
        """
        # Get raw knot data
        if khipu_ids is None:
            df = self.get_all_knots()
        else:
            conn = sqlite3.connect(self.db_path)
            
            placeholders = ','.join(['?'] * len(khipu_ids))
            # CORRECTED: Use CLUSTER_ORDINAL
            query = f"""
            SELECT 
                k.KNOT_ID,
                k.CORD_ID,
                k.CLUSTER_ORDINAL,
                k.TYPE_CODE as knot_type,
                k.NUM_TURNS,
                k.DIRECTION,
                c.KHIPU_ID,
                c.CORD_LEVEL
            FROM knot k
            JOIN cord c ON k.CORD_ID = c.CORD_ID
            WHERE c.KHIPU_ID IN ({placeholders})
            ORDER BY c.KHIPU_ID, k.CORD_ID, k.CLUSTER_ORDINAL DESC
            """
            
            df = pd.read_sql_query(query, conn, params=khipu_ids)
            conn.close()
        
        # Export CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Writing CSV ({len(df):,} rows)...")
        df.to_csv(output_path, index=False)
        print("  ✓ CSV written")
        
        # Export metadata JSON
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'source_database': str(self.db_path),
            'note': 'For cord values, use get_cord_values() which properly clusters by CLUSTER_ORDINAL',
            'total_knots': len(df),
            'unique_cords': int(df['CORD_ID'].nunique()),
            'unique_khipus': int(df['KHIPU_ID'].nunique()),
            'missing_cluster_ordinal_count': int(df['CLUSTER_ORDINAL'].isna().sum()),
            'missing_num_turns_count': int(df['NUM_TURNS'].isna().sum()),
            'knot_type_distribution': df['knot_type'].value_counts().to_dict()
        }
        
        print("  Writing metadata JSON...")
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print("  ✓ Metadata written")
        
        return df
    
    def get_summary_stats(self, sample_size: Optional[int] = None) -> Dict:
        """
        Get summary statistics for knot data quality.
        
        Args:
            sample_size: Optional number of cords to sample for value computation statistics.
                        None = compute all cords (default, takes 3-5 minutes)
        """
        df = self.get_all_knots()
        
        # Get cord values using proper computation (full dataset by default)
        print("Computing cord values for statistics...")
        cord_values = self.get_cord_values(sample_size=sample_size)
        
        return {
            'total_knots': len(df),
            'unique_cords': int(df['CORD_ID'].nunique()),
            'unique_khipus': int(df['KHIPU_ID'].nunique()),
            'cords_with_numeric_values': int(cord_values['numeric_value'].notna().sum()),
            'cords_with_numeric_pct': float(cord_values['numeric_value'].notna().mean() * 100),
            'missing_cluster_ordinal_count': int(df['CLUSTER_ORDINAL'].isna().sum()),
            'missing_cluster_ordinal_pct': float(df['CLUSTER_ORDINAL'].isna().mean() * 100),
            'missing_num_turns_count': int(df['NUM_TURNS'].isna().sum()),
            'missing_num_turns_pct': float(df['NUM_TURNS'].isna().mean() * 100),
            'average_confidence': float(cord_values['confidence'].mean()),
            'knot_types': df['knot_type'].value_counts().to_dict()
        }