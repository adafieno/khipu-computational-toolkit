"""
Color Extractor - Extract and decode cord color information from khipu database.

Handles:
- Ascher color codes (64 standardized codes)
- Multi-color cords (operators: -, :, *, etc.)
- Color ranges along cord length
- RGB mappings from ISCC-NBS standard
- White cord identification for boundary analysis
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime


class ColorExtractor:
    """Extract and validate color data with standardized mappings."""
    
    def __init__(self, db_path: Path):
        """Initialize extractor with database path."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        # Load color dictionary on initialization
        self.color_dict = self._load_color_dictionary()
    
    def _load_color_dictionary(self) -> pd.DataFrame:
        """Load standardized color codes with RGB values."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            AS_COLOR_CD as color_code,
            COLOR_DESCR as description,
            R_DEC as red,
            G_DEC as green,
            B_DEC as blue,
            COLOR as color_category,
            INTENSITY
        FROM ascher_color_dc
        ORDER BY AS_COLOR_CD
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_cord_colors(self, cord_id: int) -> pd.DataFrame:
        """
        Extract all color information for a specific cord.
        
        Returns DataFrame with:
        - color_id
        - cord_id
        - khipu_id
        - color_cd_1 through color_cd_5 (multi-color support)
        - operator_1 through operator_5 (how colors combine)
        - full_color (combined representation)
        - range_beg, range_end (position along cord in cm)
        - RGB values from color dictionary
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            color_id,
            KHIPU_ID as khipu_id,
            CORD_ID as cord_id,
            COLOR_CD_1 as color_cd_1,
            OPERATOR_1 as operator_1,
            COLOR_CD_2 as color_cd_2,
            OPERATOR_2 as operator_2,
            COLOR_CD_3 as color_cd_3,
            OPERATOR_3 as operator_3,
            COLOR_CD_4 as color_cd_4,
            OPERATOR_4 as operator_4,
            COLOR_CD_5 as color_cd_5,
            OPERATOR_5 as operator_5,
            FULL_COLOR as full_color,
            COLOR_RANGE as color_range,
            RANGE_BEG as range_beg,
            RANGE_END as range_end,
            PCORD_FLAG as pcord_flag
        FROM ascher_cord_color
        WHERE CORD_ID = ?
        ORDER BY COLOR_RANGE
        """
        
        df = pd.read_sql_query(query, conn, params=(cord_id,))
        conn.close()
        
        # Add RGB values for primary color
        df = self._enrich_with_rgb(df)
        
        return df
    
    def _enrich_with_rgb(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RGB values from color dictionary."""
        if len(df) == 0:
            return df
        
        # Merge with color dictionary for primary color
        df = df.merge(
            self.color_dict[['color_code', 'description', 'red', 'green', 'blue', 'color_category']],
            left_on='color_cd_1',
            right_on='color_code',
            how='left',
            suffixes=('', '_dict')
        )
        
        # Clean up
        if 'color_code' in df.columns:
            df = df.drop('color_code', axis=1)
        
        return df
    
    def get_all_cord_colors(self) -> pd.DataFrame:
        """
        Extract all cord colors across all khipus.
        
        Returns comprehensive color dataset.
        """
        conn = sqlite3.connect(self.db_path)
        
        print("Extracting all cord colors from database...")
        
        query = """
        SELECT 
            color_id,
            KHIPU_ID as khipu_id,
            CORD_ID as cord_id,
            COLOR_CD_1 as color_cd_1,
            OPERATOR_1 as operator_1,
            COLOR_CD_2 as color_cd_2,
            OPERATOR_2 as operator_2,
            COLOR_CD_3 as color_cd_3,
            FULL_COLOR as full_color,
            COLOR_RANGE as color_range,
            RANGE_BEG as range_beg,
            RANGE_END as range_end,
            PCORD_FLAG as pcord_flag
        FROM ascher_cord_color
        ORDER BY khipu_id, cord_id, color_range
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"  ✓ SQL query complete: {len(df):,} color records extracted")
        print("  Enriching with RGB values...")
        
        # Add RGB values
        df = self._enrich_with_rgb(df)
        
        print("  ✓ Processing complete")
        
        return df
    
    def identify_white_cords(self, khipu_id: Optional[int] = None) -> pd.DataFrame:
        """
        Identify cords that are white or predominantly white.
        
        White cords ('W' color code) are significant for:
        - Boundary markers between sum groups (Medrano & Khosla 2024)
        - Structural organization
        
        Args:
            khipu_id: Optional specific khipu to analyze (None = all khipus)
        
        Returns:
            DataFrame with white cords and their properties
        """
        conn = sqlite3.connect(self.db_path)
        
        if khipu_id is not None:
            query = """
            SELECT 
                c.KHIPU_ID,
                c.CORD_ID,
                c.CORD_CLASSIFICATION,
                c.CORD_LEVEL,
                c.CORD_ORDINAL,
                cc.COLOR_CD_1,
                cc.FULL_COLOR,
                cc.RANGE_BEG,
                cc.RANGE_END
            FROM cord c
            JOIN ascher_cord_color cc ON c.CORD_ID = cc.CORD_ID
            WHERE c.KHIPU_ID = ?
            AND cc.COLOR_CD_1 = 'W'
            ORDER BY c.CORD_ORDINAL
            """
            df = pd.read_sql_query(query, conn, params=(khipu_id,))
        else:
            query = """
            SELECT 
                c.KHIPU_ID,
                c.CORD_ID,
                c.CORD_CLASSIFICATION,
                c.CORD_LEVEL,
                c.CORD_ORDINAL,
                cc.COLOR_CD_1,
                cc.FULL_COLOR,
                cc.RANGE_BEG,
                cc.RANGE_END
            FROM cord c
            JOIN ascher_cord_color cc ON c.CORD_ID = cc.CORD_ID
            WHERE cc.COLOR_CD_1 = 'W'
            ORDER BY c.KHIPU_ID, c.CORD_ORDINAL
            """
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        return df
    
    def get_color_distribution(self, khipu_id: Optional[int] = None) -> Dict:
        """
        Get distribution of colors across khipu(s).
        
        Returns statistics on:
        - Most common colors
        - Color diversity
        - Multi-color cord frequency
        - White cord count
        """
        if khipu_id is not None:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT COLOR_CD_1, FULL_COLOR, OPERATOR_1
            FROM ascher_cord_color
            WHERE KHIPU_ID = ?
            """
            df = pd.read_sql_query(query, conn, params=(khipu_id,))
            conn.close()
        else:
            df = self.get_all_cord_colors()
        
        # Count primary colors
        color_counts = df['color_cd_1'].value_counts().to_dict()
        
        # Count multi-color cords
        multi_color = df['operator_1'].notna().sum()
        
        # White cords
        white_count = (df['color_cd_1'] == 'W').sum()
        
        # Unique colors
        unique_colors = df['color_cd_1'].nunique()
        
        return {
            'total_color_records': len(df),
            'unique_cords': df['cord_id'].nunique() if 'cord_id' in df.columns else None,
            'unique_khipus': df['khipu_id'].nunique() if 'khipu_id' in df.columns else None,
            'unique_colors': unique_colors,
            'white_cord_count': int(white_count),
            'multi_color_cord_count': int(multi_color),
            'color_distribution': color_counts,
            'most_common_color': max(color_counts, key=color_counts.get) if color_counts else None
        }
    
    def export_color_data(self, output_path: Path, khipu_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Export color data to CSV with metadata JSON.
        
        Args:
            output_path: Path for CSV output
            khipu_ids: Optional list of khipu IDs to export (None = all)
        
        Returns:
            DataFrame of exported colors
        """
        if khipu_ids is None:
            # Export all
            df = self.get_all_cord_colors()
        else:
            # Export specific khipus
            conn = sqlite3.connect(self.db_path)
            
            placeholders = ','.join(['?'] * len(khipu_ids))
            query = f"""
            SELECT 
                color_id,
                KHIPU_ID as khipu_id,
                CORD_ID as cord_id,
                COLOR_CD_1 as color_cd_1,
                OPERATOR_1 as operator_1,
                COLOR_CD_2 as color_cd_2,
                FULL_COLOR as full_color,
                COLOR_RANGE as color_range,
                RANGE_BEG as range_beg,
                RANGE_END as range_end
            FROM ascher_cord_color
            WHERE KHIPU_ID IN ({placeholders})
            ORDER BY khipu_id, cord_id, color_range
            """
            
            df = pd.read_sql_query(query, conn, params=khipu_ids)
            conn.close()
            
            df = self._enrich_with_rgb(df)
        
        # Export CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Writing CSV ({len(df):,} rows)...")
        df.to_csv(output_path, index=False)
        print("  ✓ CSV written")
        
        # Get distribution stats
        stats = self.get_color_distribution()
        
        # Export metadata JSON
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'source_database': str(self.db_path),
            'total_color_records': len(df),
            'unique_cords': int(df['cord_id'].nunique()),
            'unique_khipus': int(df['khipu_id'].nunique()),
            'unique_colors': stats['unique_colors'],
            'white_cord_count': stats['white_cord_count'],
            'multi_color_cord_count': stats['multi_color_cord_count'],
            'color_distribution': stats['color_distribution'],
            'most_common_color': stats['most_common_color'],
            'color_codes_available': len(self.color_dict)
        }
        
        print("  Writing metadata JSON...")
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print("  ✓ Metadata written")
        
        return df
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for color data."""
        stats = self.get_color_distribution()
        
        # Add color dictionary info
        stats['color_codes_in_dictionary'] = len(self.color_dict)
        stats['color_categories'] = self.color_dict['color_category'].value_counts().to_dict()
        
        return stats
