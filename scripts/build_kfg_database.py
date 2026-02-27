"""
Build KFG Database from Excel Files

This script creates a SQLite database from the 650+ KFG Excel files,
using a native KFG schema optimized for the Excel format structure.

Schema documentation: docs/KFG_DATABASE_SCHEMA.md
"""

from pathlib import Path
import sys
import sqlite3
import pandas as pd
from typing import Dict, Any, List
import argparse

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from extraction.kfg_parsers import (
    parse_kfg_metadata,
    parse_primary_cord,
    parse_cord_hierarchy,
    parse_kfg_knots,
    parse_kfg_color,
    compute_cord_value
)


def create_schema(conn: sqlite3.Connection):
    """Create KFG-native database schema."""
    cursor = conn.cursor()
    
    print("Creating database schema...")
    print("-" * 80)
    
    # Table 1: Khipu Metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS khipu_metadata (
            kfg_id TEXT PRIMARY KEY,
            kfg_name TEXT,
            aliases TEXT,
            contributors TEXT,
            kfg_url TEXT,
            museum_name TEXT,
            museum_number TEXT,
            museum_city_state TEXT,
            museum_country TEXT,
            museum_url TEXT,
            provenance TEXT,
            region TEXT,
            creation_date TEXT,
            excel_write_date TEXT,
            excel_creator TEXT
        )
    """)
    
    # Table 2: Primary Cord
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS primary_cord (
            kfg_id TEXT PRIMARY KEY,
            structure TEXT,
            thickness REAL,
            length REAL,
            color TEXT,
            fiber TEXT,
            FOREIGN KEY (kfg_id) REFERENCES khipu_metadata(kfg_id)
        )
    """)
    
    # Table 3: Cords
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cords (
            cord_id INTEGER PRIMARY KEY AUTOINCREMENT,
            kfg_id TEXT NOT NULL,
            cord_name TEXT NOT NULL,
            pendant_num INTEGER,
            hierarchy_level INTEGER,
            parent_cord TEXT,
            twist TEXT,
            attachment TEXT,
            knots TEXT,
            length REAL,
            termination TEXT,
            thickness REAL,
            color TEXT,
            value INTEGER,
            alt_value INTEGER,
            position REAL,
            notes TEXT,
            FOREIGN KEY (kfg_id) REFERENCES khipu_metadata(kfg_id)
        )
    """)
    
    # Table 4: Knot Clusters
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knot_clusters (
            cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
            cord_id INTEGER NOT NULL,
            cluster_ordinal INTEGER,
            knot_type TEXT,
            num_knots INTEGER,
            position_cm REAL,
            direction TEXT,
            cluster_value INTEGER,
            axis_orientation TEXT,
            FOREIGN KEY (cord_id) REFERENCES cords(cord_id)
        )
    """)
    
    # Table 5: Cord Colors
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cord_colors (
            color_id INTEGER PRIMARY KEY AUTOINCREMENT,
            cord_id INTEGER NOT NULL,
            color_code TEXT,
            sequence_ord INTEGER,
            FOREIGN KEY (cord_id) REFERENCES cords(cord_id)
        )
    """)
    
    # Create indexes for common queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cords_kfg_id ON cords(kfg_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cords_cord_name ON cords(cord_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cords_hierarchy ON cords(hierarchy_level)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_knot_clusters_cord_id ON knot_clusters(cord_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cord_colors_cord_id ON cord_colors(cord_id)")
    
    conn.commit()
    print("✓ Schema created successfully")
    print()


def import_khipu(excel_file: Path, conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Import a single khipu Excel file into the database.
    
    Returns:
        Dictionary with import statistics
    """
    kfg_id = excel_file.stem  # e.g., "KH0001"
    cursor = conn.cursor()
    
    stats = {
        'kfg_id': kfg_id,
        'cords': 0,
        'knot_clusters': 0,
        'colors': 0,
        'errors': []
    }
    
    try:
        # Read all sheets
        khipu_df = pd.read_excel(excel_file, sheet_name='Khipu')
        primary_cord_df = pd.read_excel(excel_file, sheet_name='PrimaryCord')
        cords_df = pd.read_excel(excel_file, sheet_name='Cords')
        
        # 1. Parse and insert metadata
        metadata = parse_kfg_metadata(khipu_df)
        cursor.execute("""
            INSERT INTO khipu_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            kfg_id,
            metadata.get('kfg_name'),
            ','.join(metadata.get('aliases', [])) if metadata.get('aliases') else None,
            metadata.get('contributors'),
            metadata.get('kfg_url'),
            metadata.get('museum_name'),
            metadata.get('museum_number'),
            metadata.get('museum_city_state'),
            metadata.get('museum_country'),
            metadata.get('museum_url'),
            metadata.get('provenance'),
            metadata.get('region'),
            metadata.get('creation_date'),
            metadata.get('excel_write_date'),
            metadata.get('excel_creator')
        ))
        
        # 2. Parse and insert primary cord
        primary = parse_primary_cord(primary_cord_df)
        cursor.execute("""
            INSERT INTO primary_cord VALUES (?, ?, ?, ?, ?, ?)
        """, (
            kfg_id,
            primary.get('structure'),
            primary.get('thickness'),
            primary.get('length'),
            primary.get('color'),
            primary.get('fiber')
        ))
        
        # 3. Parse and insert cords
        for _, cord_row in cords_df.iterrows():
            cord_name = str(cord_row['Cord_Name'])
            
            # Parse hierarchy
            hierarchy = parse_cord_hierarchy(cord_name)
            pendant_num = hierarchy['pendant_num'] if hierarchy else None
            level = hierarchy['level'] if hierarchy else None
            parent = hierarchy['parent'] if hierarchy else None
            
            # Insert cord
            cursor.execute("""
                INSERT INTO cords (
                    kfg_id, cord_name, pendant_num, hierarchy_level, parent_cord,
                    twist, attachment, knots, length, termination, thickness,
                    color, value, alt_value, position, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                kfg_id,
                cord_name,
                pendant_num,
                level,
                parent,
                cord_row.get('Twist'),
                cord_row.get('Attachment'),
                cord_row.get('Knots'),
                cord_row.get('Length'),
                cord_row.get('Termination'),
                cord_row.get('Thickness'),
                cord_row.get('Color'),
                cord_row.get('Value'),
                cord_row.get('Alt_Value'),
                cord_row.get('Position'),
                cord_row.get('Notes')
            ))
            
            cord_id = cursor.lastrowid
            stats['cords'] += 1
            
            # 4. Parse and insert knot clusters
            knots_str = cord_row.get('Knots')
            if pd.notna(knots_str):
                knot_clusters = parse_kfg_knots(str(knots_str))
                for cluster in knot_clusters:
                    cursor.execute("""
                        INSERT INTO knot_clusters (
                            cord_id, cluster_ordinal, knot_type, num_knots,
                            position_cm, direction, cluster_value, axis_orientation
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        cord_id,
                        cluster['cluster_ordinal'],
                        cluster['knot_type'],
                        cluster['num_knots'],
                        cluster['position_cm'],
                        cluster['direction'],
                        cluster.get('cluster_value'),
                        cluster.get('axis_orientation')
                    ))
                    stats['knot_clusters'] += 1
            
            # 5. Parse and insert colors
            color_str = cord_row.get('Color')
            if pd.notna(color_str):
                colors = parse_kfg_color(str(color_str))
                for color in colors:
                    cursor.execute("""
                        INSERT INTO cord_colors (cord_id, color_code, sequence_ord)
                        VALUES (?, ?, ?)
                    """, (
                        cord_id,
                        color['color_code'],
                        color['sequence_ord']
                    ))
                    stats['colors'] += 1
        
        conn.commit()
        
    except Exception as e:
        stats['errors'].append(str(e))
        conn.rollback()
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Build KFG SQLite database from Excel files'
    )
    parser.add_argument(
        '--kfg-dir',
        type=Path,
        default=Path('data/kfg/KFG/KFG'),
        help='Directory containing KFG Excel files (default: data/kfg/KFG/KFG)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/kfg/khipu_database.db'),
        help='Output database path (default: data/kfg/khipu_database.db)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of files to process (for testing)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing database'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("KFG DATABASE BUILDER")
    print("=" * 80)
    print()
    print(f"Source directory: {args.kfg_dir}")
    print(f"Output database: {args.output}")
    print()
    
    # Check if database exists
    if args.output.exists():
        if args.force:
            print(f"⚠️  Removing existing database: {args.output}")
            args.output.unlink()
        else:
            print(f"❌ Database already exists: {args.output}")
            print("   Use --force to overwrite")
            sys.exit(1)
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all KFG Excel files
    excel_files = sorted(args.kfg_dir.glob("*.xlsx"))
    
    if not excel_files:
        print(f"❌ No Excel files found in {args.kfg_dir}")
        sys.exit(1)
    
    if args.limit:
        excel_files = excel_files[:args.limit]
        print(f"ℹ️  Limited to {args.limit} files for testing")
        print()
    
    print(f"Found {len(excel_files)} Excel files to import")
    print()
    
    # Create database and schema
    conn = sqlite3.connect(args.output)
    create_schema(conn)
    
    # Import all khipus
    print("Importing khipus...")
    print("-" * 80)
    
    total_stats = {
        'imported': 0,
        'failed': 0,
        'total_cords': 0,
        'total_knot_clusters': 0,
        'total_colors': 0
    }
    
    for i, excel_file in enumerate(excel_files, 1):
        stats = import_khipu(excel_file, conn)
        
        if stats['errors']:
            print(f"  ❌ {stats['kfg_id']}: {stats['errors'][0]}")
            total_stats['failed'] += 1
        else:
            if i % 50 == 0:
                print(f"  ✓ {stats['kfg_id']}: {stats['cords']} cords, {stats['knot_clusters']} clusters")
            total_stats['imported'] += 1
            total_stats['total_cords'] += stats['cords']
            total_stats['total_knot_clusters'] += stats['knot_clusters']
            total_stats['total_colors'] += stats['colors']
    
    conn.close()
    
    # Summary
    print()
    print("=" * 80)
    print("IMPORT COMPLETE")
    print("=" * 80)
    print()
    print(f"✓ Khipus imported: {total_stats['imported']}")
    print(f"❌ Failed: {total_stats['failed']}")
    print(f"  Total cords: {total_stats['total_cords']:,}")
    print(f"  Total knot clusters: {total_stats['total_knot_clusters']:,}")
    print(f"  Total colors: {total_stats['total_colors']:,}")
    print()
    print(f"Database created: {args.output}")
    print(f"Size: {args.output.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    
    # Verify with sample query
    print("Sample queries:")
    print("-" * 80)
    conn = sqlite3.connect(args.output)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM khipu_metadata")
    print(f"  Khipus: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM cords")
    print(f"  Cords: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM knot_clusters")
    print(f"  Knot clusters: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT provenance, COUNT(*) as n FROM khipu_metadata WHERE provenance IS NOT NULL GROUP BY provenance ORDER BY n DESC LIMIT 5")
    print()
    print("  Top 5 provenances:")
    for row in cursor.fetchall():
        print(f"    {row[0]}: {row[1]}")
    
    conn.close()
    print()
    print("✓ Database ready for analysis!")
    print()


if __name__ == "__main__":
    main()
