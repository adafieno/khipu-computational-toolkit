"""
Import KFG Summation Check Data

Imports the KFG team's validated summation patterns into the database.
This provides ground truth for validating and enhancing our algorithms.

Source: data/kfg/KFG/KFG/checks/*.csv (9 summation pattern types)
"""

from pathlib import Path
import sys
import sqlite3
import pandas as pd
import argparse
import json

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from config_kfg import get_kfg_config


def extend_schema(conn: sqlite3.Connection):
    """Add tables for KFG summation check data."""
    cursor = conn.cursor()
    
    print("Extending schema with summation check tables...")
    print("-" * 80)
    
    # Main summation patterns table (summary per khipu)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summation_patterns_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kfg_id TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            num_sum_cords INTEGER,
            num_left_sums INTEGER,
            num_right_sums INTEGER,
            handedness TEXT,
            min_sum REAL,
            mean_sum REAL,
            max_sum REAL,
            max_sum_length INTEGER,
            num_dual_sums INTEGER,
            num_multisummands INTEGER,
            FOREIGN KEY (kfg_id) REFERENCES khipu_metadata(kfg_id),
            UNIQUE(kfg_id, pattern_type)
        )
    """)
    
    # Detailed summation relationships table (individual sum cords)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summation_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kfg_id TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            cord_name TEXT,
            cord_index TEXT,
            cord_value INTEGER,
            cord_color TEXT,
            num_summands INTEGER,
            handedness INTEGER,
            has_figure8knot_indicator BOOLEAN,
            has_left_exact_8knot_cord BOOLEAN,
            has_right_exact_8knot_cord BOOLEAN,
            has_parity_bit BOOLEAN,
            summand_string TEXT,
            pattern_specific_data TEXT,
            FOREIGN KEY (kfg_id) REFERENCES khipu_metadata(kfg_id)
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_summation_patterns_kfg ON summation_patterns_summary(kfg_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_summation_patterns_type ON summation_patterns_summary(pattern_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_summation_relationships_kfg ON summation_relationships(kfg_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_summation_relationships_type ON summation_relationships(pattern_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_summation_relationships_cord ON summation_relationships(cord_name)")
    
    conn.commit()
    print("✓ Schema extended successfully")
    print()


def import_summation_pattern(checks_dir: Path, pattern_type: str, conn: sqlite3.Connection) -> dict:
    """
    Import one summation pattern type.
    
    Returns:
        Dictionary with import statistics
    """
    summary_file = checks_dir / f"{pattern_type}.csv"
    relation_file = checks_dir / f"{pattern_type}_relation.csv"
    
    stats = {
        'pattern_type': pattern_type,
        'summary_rows': 0,
        'relation_rows': 0,
        'errors': []
    }
    
    cursor = conn.cursor()
    
    # Import summary data
    if summary_file.exists():
        try:
            df = pd.read_csv(summary_file)
            
            for _, row in df.iterrows():
                kfg_id = row['kfg_name']
                
                # Map columns (some patterns have different column names)
                cursor.execute("""
                    INSERT OR REPLACE INTO summation_patterns_summary (
                        kfg_id, pattern_type, num_sum_cords, num_left_sums, num_right_sums,
                        handedness, min_sum, mean_sum, max_sum, max_sum_length,
                        num_dual_sums, num_multisummands
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    kfg_id,
                    pattern_type,
                    row.get('num_sum_cords'),
                    row.get('num_left_sums'),
                    row.get('num_right_sums'),
                    row.get('handedness'),
                    row.get('min_sum'),
                    row.get('mean_sum'),
                    row.get('max_sum'),
                    row.get('max_sum_length'),
                    row.get('num_dual_sums'),
                    row.get('num_multisummands')
                ))
                stats['summary_rows'] += 1
            
        except Exception as e:
            stats['errors'].append(f"Summary import error: {e}")
    
    # Import relation data - handle different schemas for specialized patterns
    if relation_file.exists():
        try:
            df = pd.read_csv(relation_file)
            
            # Specialized patterns with different schemas
            specialized_patterns = {
                'group_group_sum': ['group_sum', 'is_top_group', 'left_group_index', 
                                   'right_group_index', 'group_distance', 
                                   'left_group_summand_string', 'right_group_summand_string'],
                'group_sum_bands': ['group_sum', 'is_top_group', 'left_group_index',
                                   'right_group_index', 'group_distance',
                                   'left_group_summand_string', 'right_group_summand_string'],
                'ascher_decreasing_group': ['group_sum', 'is_top_group', 'left_group_index',
                                           'right_group_index', 'group_distance',
                                           'left_group_summand_string', 'right_group_summand_string'],
                'pendant_sub_neighbor': ['pendant_sub_name', 'neighbor_name', 'pendant_value',
                                        'pendant_sub_sum', 'neighbor_value']
            }
            
            if pattern_type in specialized_patterns:
                # Handle specialized patterns - store as JSON
                for _, row in df.iterrows():
                    kfg_id = row['kfg_name']
                    
                    # Extract pattern-specific fields as JSON
                    specific_data = {}
                    for col in specialized_patterns[pattern_type]:
                        if col in row:
                            specific_data[col] = row[col] if pd.notna(row[col]) else None
                    
                    cursor.execute("""
                        INSERT INTO summation_relationships (
                            kfg_id, pattern_type, handedness, pattern_specific_data
                        ) VALUES (?, ?, ?, ?)
                    """, (
                        kfg_id,
                        pattern_type,
                        row.get('handedness'),
                        json.dumps(specific_data)
                    ))
                    stats['relation_rows'] += 1
            else:
                # Handle standard cord-based patterns
                for _, row in df.iterrows():
                    kfg_id = row['kfg_name']
                    
                    cursor.execute("""
                        INSERT INTO summation_relationships (
                            kfg_id, pattern_type, cord_name, cord_index, cord_value,
                            cord_color, num_summands, handedness, has_figure8knot_indicator,
                            has_left_exact_8knot_cord, has_right_exact_8knot_cord,
                            has_parity_bit, summand_string
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        kfg_id,
                        pattern_type,
                        row.get('cord_name'),
                        str(row.get('cord_index')) if pd.notna(row.get('cord_index')) else None,
                        row.get('cord_value'),
                        row.get('cord_color'),
                        row.get('num_summands'),
                        row.get('handedness'),
                        row.get('has_figure8knot_indicator'),
                        row.get('has_left_exact_8knot_cord'),
                        row.get('has_right_exact_8knot_cord'),
                        row.get('has_parity_bit'),
                        row.get('summand_string')
                    ))
                    stats['relation_rows'] += 1
            
        except Exception as e:
            stats['errors'].append(f"Relation import error: {e}")
    
    conn.commit()
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Import KFG summation check data into database'
    )
    parser.add_argument(
        '--checks-dir',
        type=Path,
        default=Path('data/kfg/KFG/KFG/checks'),
        help='Directory containing check CSV files'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("KFG SUMMATION CHECK DATA IMPORTER")
    print("=" * 80)
    print()
    
    # Get database path
    config = get_kfg_config()
    db_path = config.get_database_path()
    
    print(f"Database: {db_path}")
    print(f"Checks directory: {args.checks_dir}")
    print()
    
    if not args.checks_dir.exists():
        print(f"❌ Checks directory not found: {args.checks_dir}")
        sys.exit(1)
    
    # Connect and extend schema
    conn = sqlite3.connect(db_path)
    extend_schema(conn)
    
    # Import all summation pattern types
    pattern_types = [
        'pendant_pendant_sum',
        'indexed_pendant_sum',
        'subsidiary_pendant_sum',
        'colored_pendant_sum',
        'indexed_subsidiary_sum',
        'group_group_sum',
        'group_sum_bands',
        'ascher_decreasing_group',
        'pendant_sub_neighbor'
    ]
    
    print("Importing summation patterns...")
    print("-" * 80)
    
    total_stats = {
        'patterns_imported': 0,
        'total_summary_rows': 0,
        'total_relation_rows': 0,
        'errors': []
    }
    
    for pattern_type in pattern_types:
        stats = import_summation_pattern(args.checks_dir, pattern_type, conn)
        
        if stats['errors']:
            print(f"  ⚠️  {pattern_type}: {stats['errors'][0]}")
            total_stats['errors'].extend(stats['errors'])
        else:
            print(f"  ✓ {pattern_type}: {stats['summary_rows']} summaries, {stats['relation_rows']} relationships")
            total_stats['patterns_imported'] += 1
            total_stats['total_summary_rows'] += stats['summary_rows']
            total_stats['total_relation_rows'] += stats['relation_rows']
    
    conn.close()
    
    # Summary
    print()
    print("=" * 80)
    print("IMPORT COMPLETE")
    print("=" * 80)
    print()
    print(f"✓ Pattern types imported: {total_stats['patterns_imported']}/{len(pattern_types)}")
    print(f"  Total summary records: {total_stats['total_summary_rows']:,}")
    print(f"  Total relationships: {total_stats['total_relation_rows']:,}")
    
    if total_stats['errors']:
        print(f"⚠️  Errors: {len(total_stats['errors'])}")
    
    print()
    
    # Verification queries
    print("Verification:")
    print("-" * 80)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(DISTINCT kfg_id) FROM summation_patterns_summary")
    print(f"  Khipus with patterns: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM summation_relationships")
    print(f"  Total sum relationships: {cursor.fetchone()[0]:,}")
    
    cursor.execute("""
        SELECT pattern_type, COUNT(*) as n 
        FROM summation_relationships 
        GROUP BY pattern_type 
        ORDER BY n DESC
    """)
    print()
    print("  Relationships by pattern:")
    for row in cursor.fetchall():
        print(f"    {row[0]:30} {row[1]:6,}")
    
    conn.close()
    print()
    print("✓ KFG summation ground truth ready for validation!")
    print()


if __name__ == "__main__":
    main()
