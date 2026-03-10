"""
Extract cord hierarchy data and export to processed datasets.
"""

from pathlib import Path
import sys
import argparse

# Add src to path for runtime
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from extraction.cord_extractor import CordExtractor  # noqa: E402
from extraction.kfg_cord_extractor import KFGCordExtractor  # noqa: E402
from config import get_config  # noqa: E402
from config_kfg import get_kfg_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Extract cord hierarchy data from khipu database')
    parser.add_argument('--kfg', action='store_true', 
                       help='Use KFG database instead of OKR database')
    args = parser.parse_args()

    print("=" * 80)
    print("CORD HIERARCHY EXTRACTION")
    print("=" * 80)
    print()

    # Get configuration and appropriate extractor
    if args.kfg:
        config = get_kfg_config()
        data_source = "KFG"
        ExtractorClass = KFGCordExtractor
    else:
        config = get_config()
        data_source = "OKR"
        ExtractorClass = CordExtractor
    
    print(f"Data Source: {data_source}")
    print()

    # Validate setup
    validation = config.validate_setup()
    if not validation['valid']:
        print("\nConfiguration errors:")
        for error in validation['errors']:
            print(f"  • {error}")
        sys.exit(1)

    print(f"Database: {config.get_database_path()}")
    print()

    # Initialize extractor (KFG or OKR)
    db_path = config.get_database_path()
    extractor = ExtractorClass(db_path)

    # Get summary stats first
    print("Analyzing cord structure...")
    print("-" * 80)
    stats = extractor.get_summary_stats()

    print(f"Total cords: {stats['total_cords']:,}")
    print(f"Unique khipus: {stats['unique_khipus']}")
    print(f"Cords with numeric values: {stats['cords_with_numeric_values']:,} ({stats['cords_with_numeric_pct']:.1f}%)")
    
    # OKR-specific stats
    if 'missing_attachment_count' in stats:
        print(f"Missing ATTACHED_TO: {stats['missing_attachment_count']:,} ({stats['missing_attachment_pct']:.1f}%)")
    if 'missing_ordinal_count' in stats:
        print(f"Missing CORD_ORDINAL: {stats['missing_ordinal_count']:,} ({stats['missing_ordinal_pct']:.1f}%)")
    if 'average_confidence' in stats:
        print(f"Average confidence: {stats['average_confidence']:.3f}")
    
    # KFG-specific stats
    if 'cords_with_knots' in stats:
        print(f"Cords with knots: {stats['cords_with_knots']:,} ({stats['cords_with_knots_pct']:.1f}%)")
    if 'total_knot_clusters' in stats:
        print(f"Total knot clusters: {stats['total_knot_clusters']:,}")
    if 'avg_clusters_per_cord' in stats:
        print(f"Avg clusters per cord: {stats['avg_clusters_per_cord']:.1f}")
    
    print()

    # Cord classification or hierarchy levels
    if 'cord_classifications' in stats:
        print("Cord classifications:")
        for classification, count in sorted(stats['cord_classifications'].items(), key=lambda x: -x[1]):
            print(f"  {classification}: {count:,}")
    elif 'hierarchy_levels' in stats:
        print("Hierarchy levels:")
        for level, count in sorted(stats['hierarchy_levels'].items()):
            print(f"  Level {level}: {count:,} cords")
    print()

    print(f"Level range: {stats['level_range'][0]} to {stats['level_range'][1]}")
    print()

    # Export full hierarchy
    print("Exporting cord hierarchy...")
    print("-" * 80)

    # Ensure directories exist
    config.ensure_directories()

    # Save to phase2 directory
    output_path = config.get_processed_file('cord_hierarchy.csv', phase=2)

    df = extractor.export_cord_hierarchy(output_path)

    print(f"✓ Exported {len(df):,} cords to:")
    print(f"  {output_path}")
    print(f"  {output_path.with_suffix('.json')} (metadata)")
    print()

    # Test: Build tree for first khipu
    print("Testing tree construction for first khipu...")
    print("-" * 80)

    if args.kfg:
        # KFG: use first KFG ID
        test_khipu = "KH0001"
    else:
        # OKR: query database for first khipu
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT KHIPU_ID FROM khipu_main LIMIT 1")
        test_khipu = cursor.fetchone()[0]
        conn.close()

    tree = extractor.build_cord_tree(test_khipu)

    def count_nodes(node):
        return 1 + sum(count_nodes(child) for child in node.get('children', []))

    total_nodes = count_nodes(tree) if tree else 0
    print(f"✓ Built tree for khipu {test_khipu}")
    
    if tree and 'cord_name' in tree:
        root_type = tree.get('cord_name', 'unknown')
        root_level = tree.get('level', -1)
        print(f"  Root: {root_type} (level {root_level})")
    
    print(f"  Total nodes: {total_nodes}")
    print(f"  Direct children: {len(tree.get('children', []))}")
    print()

    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print()
    print(f"Generated: {output_path}")
    print()
    print("Next steps:")
    print("  1. Build knot extractor")
    print("  2. Test summation hypotheses with validated data")
    print("  3. Construct graph representations")


if __name__ == "__main__":
    main()
