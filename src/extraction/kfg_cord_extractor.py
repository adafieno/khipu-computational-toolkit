"""
KFG Cord Extractor

Extraction utilities for KFG database with native KFG schema.
Parallel to cord_extractor.py but adapted for KFG's structure.
"""

import sqlite3
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Any


class KFGCordExtractor:
    """Extract and analyze cord data from KFG database."""
    
    def __init__(self, db_path: Path):
        """
        Initialize extractor.
        
        Args:
            db_path: Path to KFG SQLite database
        """
        self.db_path = Path(db_path)
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
    
    def get_all_cords(self) -> pd.DataFrame:
        """
        Extract all cords with enriched information.
        
        Returns:
            DataFrame with cord information, analogous to OKR format
            but using KFG schema columns.
        """
        conn = sqlite3.connect(self.db_path)
        
        print("  Extracting all cords from KFG database...")
        
        query = """
        SELECT
            c.cord_id,
            c.kfg_id as khipu_id,
            c.cord_name,
            c.pendant_num,
            c.hierarchy_level as cord_level,
            c.parent_cord,
            c.twist,
            c.attachment,
            c.knots as knots_string,
            c.length as cord_length,
            c.thickness,
            c.color,
            c.value as numeric_value,
            c.alt_value,
            c.notes,
            COUNT(k.cluster_id) as num_knot_clusters,
            SUM(k.num_knots) as total_knots
        FROM cords c
        LEFT JOIN knot_clusters k ON c.cord_id = k.cord_id
        GROUP BY c.cord_id
        ORDER BY c.kfg_id, c.hierarchy_level, c.pendant_num
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"  ✓ SQL query complete: {len(df):,} cords extracted")
        
        # Add derived fields for compatibility
        df['has_numeric_value'] = df['numeric_value'].notna() & (df['numeric_value'] > 0)
        df['has_knots'] = df['num_knot_clusters'] > 0
        
        return df
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about the KFG cord dataset.
        
        Returns:
            Dictionary with statistics similar to OKR extractor
        """
        df = self.get_all_cords()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {
            'total_cords': len(df),
            'unique_khipus': df['khipu_id'].nunique(),
            'cords_with_numeric_values': df['has_numeric_value'].sum(),
            'cords_with_numeric_pct': (df['has_numeric_value'].sum() / len(df) * 100) if len(df) > 0 else 0,
            'cords_with_knots': df['has_knots'].sum(),
            'cords_with_knots_pct': (df['has_knots'].sum() / len(df) * 100) if len(df) > 0 else 0,
            'total_knot_clusters': df['num_knot_clusters'].sum(),
            'avg_clusters_per_cord': df['num_knot_clusters'].mean(),
        }
        
        # Hierarchy level distribution
        level_dist = df['cord_level'].value_counts().to_dict()
        stats['hierarchy_levels'] = level_dist
        stats['level_range'] = (df['cord_level'].min(), df['cord_level'].max())
        
        # Color distribution (top 10)
        cursor.execute("""
            SELECT color_code, COUNT(*) as count
            FROM cord_colors
            GROUP BY color_code
            ORDER BY count DESC
            LIMIT 10
        """)
        stats['top_colors'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Provenance distribution
        cursor.execute("""
            SELECT provenance, COUNT(*) as count
            FROM khipu_metadata
            WHERE provenance IS NOT NULL
            GROUP BY provenance
            ORDER BY count DESC
            LIMIT 10
        """)
        stats['top_provenances'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return stats
    
    def export_cord_hierarchy(self, output_path: Path) -> pd.DataFrame:
        """
        Export cord hierarchy to CSV.
        
        Args:
            output_path: Where to save CSV file
            
        Returns:
            DataFrame that was exported
        """
        df = self.get_all_cords()
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        df.to_csv(output_path, index=False)
        
        # Save metadata JSON
        import json
        metadata = {
            'source': 'KFG Database',
            'database_path': str(self.db_path),
            'total_cords': len(df),
            'unique_khipus': int(df['khipu_id'].nunique()),
            'date_exported': pd.Timestamp.now().isoformat()
        }
        
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return df
    
    def build_cord_tree(self, kfg_id: str) -> Dict[str, Any]:
        """
        Build hierarchical tree structure for a khipu.
        
        Args:
            kfg_id: KFG ID (e.g., "KH0001")
            
        Returns:
            Tree structure with root and children
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT cord_id, cord_name, hierarchy_level, parent_cord, 
               pendant_num, numeric_value, num_knot_clusters
        FROM (
            SELECT c.cord_id, c.cord_name, c.hierarchy_level, c.parent_cord,
                   c.pendant_num, c.value as numeric_value,
                   COUNT(k.cluster_id) as num_knot_clusters
            FROM cords c
            LEFT JOIN knot_clusters k ON c.cord_id = k.cord_id
            WHERE c.kfg_id = ?
            GROUP BY c.cord_id
        )
        ORDER BY hierarchy_level, pendant_num
        """
        
        df = pd.read_sql_query(query, params=[kfg_id], con=conn)
        conn.close()
        
        if len(df) == 0:
            return {}
        
        # Build tree (KFG uses string-based hierarchy, simpler than OKR)
        nodes_by_name = {}
        
        for _, row in df.iterrows():
            node = {
                'cord_id': row['cord_id'],
                'cord_name': row['cord_name'],
                'level': row['hierarchy_level'],
                'pendant_num': row['pendant_num'],
                'numeric_value': row['numeric_value'],
                'num_clusters': row['num_knot_clusters'],
                'children': []
            }
            nodes_by_name[row['cord_name']] = node
        
        # Link parent-child relationships
        root = {'cord_name': 'PRIMARY', 'level': -1, 'children': []}
        
        for cord_name, node in nodes_by_name.items():
            parent_name = node.get('parent_cord') if pd.notna(df[df['cord_name'] == cord_name]['parent_cord'].iloc[0]) else None
            
            if parent_name and parent_name in nodes_by_name:
                nodes_by_name[parent_name]['children'].append(node)
            else:
                # Level 0 cords (pendants) attach to root
                root['children'].append(node)
        
        return root


def main():
    """Test KFG extractor."""
    from config_kfg import get_kfg_config
    
    config = get_kfg_config()
    db_path = config.get_database_path()
    
    print("Testing KFG Cord Extractor")
    print("=" * 80)
    print(f"Database: {db_path}")
    print()
    
    extractor = KFGCordExtractor(db_path)
    
    # Get stats
    stats = extractor.get_summary_stats()
    
    print("Summary Statistics:")
    print("-" * 80)
    print(f"Total cords: {stats['total_cords']:,}")
    print(f"Unique khipus: {stats['unique_khipus']}")
    print(f"Cords with values: {stats['cords_with_numeric_values']:,} ({stats['cords_with_numeric_pct']:.1f}%)")
    print(f"Total knot clusters: {stats['total_knot_clusters']:,}")
    print(f"Avg clusters per cord: {stats['avg_clusters_per_cord']:.1f}")
    print()
    
    print("Hierarchy levels:")
    for level, count in sorted(stats['hierarchy_levels'].items()):
        print(f"  Level {level}: {count:,} cords")
    print()
    
    print("Top 5 colors:")
    for color, count in list(stats['top_colors'].items())[:5]:
        print(f"  {color}: {count:,}")
    print()
    
    print("Top 5 provenances:")
    for prov, count in list(stats['top_provenances'].items())[:5]:
        print(f"  {prov}: {count}")
    print()
    
    # Test tree building
    print("Testing tree construction for KH0001...")
    tree = extractor.build_cord_tree("KH0001")
    print(f"✓ Tree built with {len(tree['children'])} pendant cords")
    print()


if __name__ == "__main__":
    main()
