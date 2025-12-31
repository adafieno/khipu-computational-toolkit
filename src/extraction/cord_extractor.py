"""
Cord Extractor - Extract hierarchical cord structure from khipu database.

Handles:
- Primary cord identification
- Pendant relationships (PENDANT_FROM, ATTACHED_TO)
- Cord level hierarchy (CORD_LEVEL)
- Missing relationship inference
- Integration with numeric validation
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import json
from datetime import datetime


class CordExtractor:
    """Extract and validate cord hierarchical structure."""
    
    def __init__(self, db_path: Path):
        """Initialize extractor with database path."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
    
    def get_cord_hierarchy(self, khipu_id: int) -> pd.DataFrame:
        """
        Extract complete cord hierarchy for a single khipu.
        
        Returns DataFrame with columns:
        - cord_id: Unique cord identifier
        - khipu_id: Parent khipu
        - cord_type: PRIMARY, PENDANT, SUBSIDIARY, TOP, etc.
        - cord_level: Hierarchy depth (0=primary, 1=pendant, etc.)
        - pendant_from: Parent cord (if pendant)
        - attached_to: Primary attachment point
        - position: Position along parent
        - has_numeric_value: Whether cord has knots with numeric data
        - cord_length: Physical length in cm
        - num_knots: Count of knots on this cord
        - confidence: Data completeness score
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            c.CORD_ID,
            c.KHIPU_ID,
            c.CORD_CLASSIFICATION,
            c.CORD_LEVEL,
            c.PENDANT_FROM,
            c.ATTACHED_TO,
            c.CORD_ORDINAL,
            c.CORD_LENGTH,
            COUNT(k.KNOT_ID) as num_knots,
            COUNT(CASE WHEN k.knot_value_type IS NOT NULL THEN 1 END) as num_valued_knots
        FROM cord c
        LEFT JOIN knot k ON c.CORD_ID = k.CORD_ID
        WHERE c.KHIPU_ID = ?
        GROUP BY c.CORD_ID
        ORDER BY c.CORD_LEVEL, c.CORD_ORDINAL
        """
        
        df = pd.read_sql_query(query, conn, params=(khipu_id,))
        conn.close()
        
        # Add derived fields
        df['has_numeric_value'] = df['num_valued_knots'] > 0
        
        # Calculate confidence score
        df['confidence'] = self._calculate_cord_confidence(df)
        
        return df
    
    def get_all_cords(self) -> pd.DataFrame:
        """
        Extract all cords across all khipus with hierarchy metadata.
        
        Returns comprehensive dataset for graph construction.
        """
        conn = sqlite3.connect(self.db_path)
        
        print("Extracting all cords from database...")
        
        query = """
        SELECT 
            c.CORD_ID,
            c.KHIPU_ID,
            c.CORD_CLASSIFICATION,
            c.CORD_LEVEL,
            c.PENDANT_FROM,
            c.ATTACHED_TO,
            c.CORD_ORDINAL,
            c.CORD_LENGTH,
            c.TWIST,
            c.FIBER,
            COUNT(k.KNOT_ID) as num_knots,
            COUNT(CASE WHEN k.knot_value_type IS NOT NULL THEN 1 END) as num_valued_knots
        FROM cord c
        LEFT JOIN knot k ON c.CORD_ID = k.CORD_ID
        GROUP BY c.CORD_ID
        ORDER BY c.KHIPU_ID, c.CORD_LEVEL, c.CORD_ORDINAL
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"  ✓ SQL query complete: {len(df):,} cords extracted")
        print("  Processing derived fields...")
        
        # Add derived fields
        df['has_numeric_value'] = df['num_valued_knots'] > 0
        df['has_missing_attachment'] = df['ATTACHED_TO'].isna()
        df['has_missing_ordinal'] = df['CORD_ORDINAL'].isna()
        
        # Calculate confidence score
        print("  Calculating confidence scores...")
        df['confidence'] = self._calculate_cord_confidence(df)
        print("  ✓ Processing complete")
        
        return df
    
    def _calculate_cord_confidence(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate confidence score for cord data quality.
        
        Penalizes:
        - Missing ATTACHED_TO (critical for hierarchy)
        - Missing POSITION (affects ordering)
        - Missing CORD_LEVEL (affects depth)
        - Missing physical measurements
        """
        confidence = pd.Series(1.0, index=df.index)
        
        # Critical structural fields
        confidence -= df['ATTACHED_TO'].isna() * 0.3
        confidence -= df['CORD_ORDINAL'].isna() * 0.2
        confidence -= df['CORD_LEVEL'].isna() * 0.3
        
        # Physical measurements (less critical)
        confidence -= df['CORD_LENGTH'].isna() * 0.1
        
        return confidence.clip(lower=0.0)
    
    def identify_primary_cords(self, khipu_id: int) -> List[int]:
        """Identify all primary cords for a khipu."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT CORD_ID 
            FROM cord 
            WHERE KHIPU_ID = ? AND CORD_CLASSIFICATION = 'M'
            ORDER BY CORD_ORDINAL
        """, (khipu_id,))
        
        primary_cords = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return primary_cords
    
    def get_cord_children(self, cord_id: int) -> List[int]:
        """Get all cords that hang from (are pendant to) this cord."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT CORD_ID 
            FROM cord 
            WHERE PENDANT_FROM = ?
            ORDER BY CORD_ORDINAL
        """, (cord_id,))
        
        children = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return children
    
    def build_cord_tree(self, khipu_id: int) -> Dict:
        """
        Build recursive tree structure for a khipu's cord hierarchy.
        
        Returns nested dictionary with structure:
        {
            'cord_id': int,
            'type': str,
            'level': int,
            'position': int,
            'children': [...]
        }
        """
        df = self.get_cord_hierarchy(khipu_id)
        
        # Build adjacency mapping
        tree_map = {}
        for _, row in df.iterrows():
            cord_id = row['CORD_ID']
            tree_map[cord_id] = {
                'cord_id': cord_id,
                'classification': row['CORD_CLASSIFICATION'],
                'level': row['CORD_LEVEL'],
                'ordinal': row['CORD_ORDINAL'],
                'has_numeric': row['has_numeric_value'],
                'num_knots': row['num_knots'],
                'children': []
            }
        
        # Build parent-child relationships
        for _, row in df.iterrows():
            if pd.notna(row['PENDANT_FROM']):
                parent_id = int(row['PENDANT_FROM'])
                if parent_id in tree_map:
                    tree_map[parent_id]['children'].append(tree_map[row['CORD_ID']])
        
        # Find root (primary/main cord - classification 'M')
        primary = df[df['CORD_CLASSIFICATION'] == 'M'].iloc[0] if len(df[df['CORD_CLASSIFICATION'] == 'M']) > 0 else df.iloc[0]
        
        return tree_map[primary['CORD_ID']]
    
    def export_cord_hierarchy(self, output_path: Path, khipu_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Export cord hierarchy data to CSV with metadata JSON.
        
        Args:
            output_path: Path for CSV output
            khipu_ids: Optional list of khipu IDs to export (None = all)
        
        Returns:
            DataFrame of exported cords
        """
        if khipu_ids is None:
            # Export all
            df = self.get_all_cords()
        else:
            # Export specific khipus
            conn = sqlite3.connect(self.db_path)
            
            placeholders = ','.join(['?'] * len(khipu_ids))
            query = f"""
            SELECT 
                c.CORD_ID,
                c.KHIPU_ID,
                c.CORD_CLASSIFICATION,
                c.CORD_LEVEL,
                c.PENDANT_FROM,
                c.ATTACHED_TO,
                c.CORD_ORDINAL,
                c.CORD_LENGTH,
                COUNT(k.KNOT_ID) as num_knots,
                COUNT(CASE WHEN k.knot_value_type IS NOT NULL THEN 1 END) as num_valued_knots
            FROM cord c
            LEFT JOIN knot k ON c.CORD_ID = k.CORD_ID
            WHERE c.KHIPU_ID IN ({placeholders})
            GROUP BY c.CORD_ID
            ORDER BY c.KHIPU_ID, c.CORD_LEVEL, c.CORD_ORDINAL
            """
            
            df = pd.read_sql_query(query, conn, params=khipu_ids)
            conn.close()
            
            df['has_numeric_value'] = df['num_valued_knots'] > 0
            df['confidence'] = self._calculate_cord_confidence(df)
        
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
            'total_cords': len(df),
            'unique_khipus': df['KHIPU_ID'].nunique(),
            'cords_with_numeric_values': int(df['has_numeric_value'].sum()),
            'missing_attachment_count': int(df['ATTACHED_TO'].isna().sum()),
            'missing_ordinal_count': int(df['CORD_ORDINAL'].isna().sum()),
            'average_confidence': float(df['confidence'].mean()),
            'cord_classification_distribution': df['CORD_CLASSIFICATION'].value_counts().to_dict(),
            'level_distribution': df['CORD_LEVEL'].value_counts().to_dict()
        }
        
        print("  Writing metadata JSON...")
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print("  ✓ Metadata written")
        
        return df
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for cord data quality."""
        df = self.get_all_cords()
        
        return {
            'total_cords': len(df),
            'unique_khipus': df['KHIPU_ID'].nunique(),
            'cords_with_numeric_values': int(df['has_numeric_value'].sum()),
            'cords_with_numeric_pct': float(df['has_numeric_value'].mean() * 100),
            'missing_attachment_count': int(df['ATTACHED_TO'].isna().sum()),
            'missing_attachment_pct': float(df['ATTACHED_TO'].isna().mean() * 100),
            'missing_ordinal_count': int(df['CORD_ORDINAL'].isna().sum()),
            'missing_ordinal_pct': float(df['CORD_ORDINAL'].isna().mean() * 100),
            'average_confidence': float(df['confidence'].mean()),
            'cord_classifications': df['CORD_CLASSIFICATION'].value_counts().to_dict(),
            'level_range': (int(df['CORD_LEVEL'].min()), int(df['CORD_LEVEL'].max()))
        }
