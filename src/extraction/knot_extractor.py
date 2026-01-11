"""
Knot Extractor - Extract knot details with positional and numeric validation.

Handles:
- Knot positions and ordering (KNOT_ORDINAL)
- Knot types (L=long, E=figure-eight, S=single)
- Numeric decoding (knot_value_type × NUM_TURNS)
- Data quality validation
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import json
from datetime import datetime


class KnotExtractor:
    """Extract and validate knot data with positional information."""
    
    def __init__(self, db_path: Path):
        """Initialize extractor with database path."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
    
    def get_cord_knots(self, cord_id: int) -> pd.DataFrame:
        """
        Extract all knots for a specific cord with decoded values.
        
        Returns DataFrame with columns:
        - knot_id: Unique knot identifier
        - cord_id: Parent cord
        - knot_ordinal: Position on cord
        - knot_type: L (long), E (figure-eight), S (single)
        - knot_value_type: Place value (1, 10, 100, 1000)
        - num_turns: Digit value (for long knots)
        - numeric_value: Decoded value (knot_value_type × num_turns)
        - position_cm: Distance from cord attachment
        - confidence: Data completeness score
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            KNOT_ID,
            CORD_ID,
            KNOT_ORDINAL,
            TYPE_CODE as knot_type,
            knot_value_type,
            NUM_TURNS,
            DIRECTION
        FROM knot
        WHERE CORD_ID = ?
        ORDER BY KNOT_ORDINAL
        """
        
        df = pd.read_sql_query(query, conn, params=(cord_id,))
        conn.close()
        
        # Decode numeric values
        df['numeric_value'] = self._decode_knot_value(df)
        
        # Calculate confidence
        df['confidence'] = self._calculate_knot_confidence(df)
        
        return df
    
    def get_all_knots(self) -> pd.DataFrame:
        """
        Extract all knots across all khipus with decoded values.
        
        Returns comprehensive dataset for analysis.
        """
        conn = sqlite3.connect(self.db_path)
        
        print("Extracting all knots from database...")
        
        query = """
        SELECT 
            k.KNOT_ID,
            k.CORD_ID,
            k.KNOT_ORDINAL,
            k.TYPE_CODE as knot_type,
            k.knot_value_type,
            k.NUM_TURNS,
            k.DIRECTION,
            c.KHIPU_ID,
            c.CORD_LEVEL
        FROM knot k
        JOIN cord c ON k.CORD_ID = c.CORD_ID
        ORDER BY c.KHIPU_ID, k.CORD_ID, k.KNOT_ORDINAL
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"  ✓ SQL query complete: {len(df):,} knots extracted")
        print("  Decoding numeric values...")
        
        # Decode numeric values
        df['numeric_value'] = self._decode_knot_value(df)
        
        # Calculate confidence
        print("  Calculating confidence scores...")
        df['confidence'] = self._calculate_knot_confidence(df)
        
        print("  ✓ Processing complete")
        
        return df
    
    def _decode_knot_value(self, df: pd.DataFrame) -> pd.Series:
        """
        Decode numeric value from knot encoding using Ascher notation.
        
        Formula (Ascher & Ascher):
        - S (single) = hundreds position = 100
        - L (long) = tens position = NUM_TURNS × 10
        - E (figure-eight) = units position = 1
        """
        values = pd.Series(None, index=df.index, dtype='Int64')
        
        # Long knots: NUM_TURNS × 10 (tens position)
        long_mask = (df['knot_type'] == 'L') & df['NUM_TURNS'].notna()
        values[long_mask] = (df.loc[long_mask, 'NUM_TURNS'] * 10).astype('Int64')
        
        # Single knots: 100 (hundreds position)
        single_mask = (df['knot_type'] == 'S')
        values[single_mask] = 100
        
        # Figure-eight knots: 1 (units position)
        eight_mask = (df['knot_type'] == 'E')
        values[eight_mask] = 1
        
        return values
    
    def _calculate_knot_confidence(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate confidence score for knot data quality.
        
        Penalizes:
        - Missing KNOT_ORDINAL (affects ordering)
        - Missing knot_value_type (can't decode)
        - Missing NUM_TURNS for long knots
        - Invalid knot types
        """
        confidence = pd.Series(1.0, index=df.index)
        
        # Missing ordinal (critical for position)
        confidence -= df['KNOT_ORDINAL'].isna() * 0.4
        
        # Missing value type (can't decode)
        confidence -= df['knot_value_type'].isna() * 0.3
        
        # Long knots missing NUM_TURNS
        long_missing_turns = (df['knot_type'] == 'L') & df['NUM_TURNS'].isna()
        confidence -= long_missing_turns * 0.3
        
        return confidence.clip(lower=0.0)
    
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
        if khipu_ids is None:
            # Export all
            df = self.get_all_knots()
        else:
            # Export specific khipus
            conn = sqlite3.connect(self.db_path)
            
            placeholders = ','.join(['?'] * len(khipu_ids))
            query = f"""
            SELECT 
                k.KNOT_ID,
                k.CORD_ID,
                k.KNOT_ORDINAL,
                k.TYPE_CODE as knot_type,
                k.knot_value_type,
                k.NUM_TURNS,
                k.DIRECTION,
                c.KHIPU_ID,
                c.CORD_LEVEL
            FROM knot k
            JOIN cord c ON k.CORD_ID = c.CORD_ID
            WHERE c.KHIPU_ID IN ({placeholders})
            ORDER BY c.KHIPU_ID, k.CORD_ID, k.KNOT_ORDINAL
            """
            
            df = pd.read_sql_query(query, conn, params=khipu_ids)
            conn.close()
            
            df['numeric_value'] = self._decode_knot_value(df)
            df['confidence'] = self._calculate_knot_confidence(df)
        
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
            'total_knots': len(df),
            'unique_cords': df['CORD_ID'].nunique(),
            'unique_khipus': df['KHIPU_ID'].nunique(),
            'knots_with_numeric_values': int(df['numeric_value'].notna().sum()),
            'missing_ordinal_count': int(df['KNOT_ORDINAL'].isna().sum()),
            'missing_value_type_count': int(df['knot_value_type'].isna().sum()),
            'average_confidence': float(df['confidence'].mean()),
            'knot_type_distribution': df['knot_type'].value_counts().to_dict(),
            'value_type_distribution': df['knot_value_type'].value_counts().to_dict()
        }
        
        print("  Writing metadata JSON...")
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print("  ✓ Metadata written")
        
        return df
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for knot data quality."""
        df = self.get_all_knots()
        
        return {
            'total_knots': len(df),
            'unique_cords': df['CORD_ID'].nunique(),
            'unique_khipus': df['KHIPU_ID'].nunique(),
            'knots_with_numeric_values': int(df['numeric_value'].notna().sum()),
            'knots_with_numeric_pct': float(df['numeric_value'].notna().mean() * 100),
            'missing_ordinal_count': int(df['KNOT_ORDINAL'].isna().sum()),
            'missing_ordinal_pct': float(df['KNOT_ORDINAL'].isna().mean() * 100),
            'missing_value_type_count': int(df['knot_value_type'].isna().sum()),
            'missing_value_type_pct': float(df['knot_value_type'].isna().mean() * 100),
            'average_confidence': float(df['confidence'].mean()),
            'knot_types': df['knot_type'].value_counts().to_dict(),
            'value_types': df['knot_value_type'].value_counts().to_dict()
        }
