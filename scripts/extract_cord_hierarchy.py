"""
Extract cord hierarchy data and export to processed datasets.
"""

from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from extraction.cord_extractor import CordExtractor  # noqa: E402 # type: ignore


def main():
    print("=" * 80)
    print("CORD HIERARCHY EXTRACTION")
    print("=" * 80)
    print()
    
    # Initialize extractor
    db_path = Path(__file__).parent.parent / "khipu.db"
    extractor = CordExtractor(db_path)
    
    # Get summary stats first
    print("Analyzing cord structure...")
    print("-" * 80)
    stats = extractor.get_summary_stats()
    
    print(f"Total cords: {stats['total_cords']:,}")
    print(f"Unique khipus: {stats['unique_khipus']}")
    print(f"Cords with numeric values: {stats['cords_with_numeric_values']:,} ({stats['cords_with_numeric_pct']:.1f}%)")
    print(f"Missing ATTACHED_TO: {stats['missing_attachment_count']:,} ({stats['missing_attachment_pct']:.1f}%)")
    print(f"Missing CORD_ORDINAL: {stats['missing_ordinal_count']:,} ({stats['missing_ordinal_pct']:.1f}%)")
    print(f"Average confidence: {stats['average_confidence']:.3f}")
    print()
    
    print("Cord classifications:")
    for classification, count in sorted(stats['cord_classifications'].items(), key=lambda x: -x[1]):
        print(f"  {classification}: {count:,}")
    print()
    
    print(f"Level range: {stats['level_range'][0]} to {stats['level_range'][1]}")
    print()
    
    # Export full hierarchy
    print("Exporting cord hierarchy...")
    print("-" * 80)
    
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_path = output_dir / "cord_hierarchy.csv"
    
    df = extractor.export_cord_hierarchy(output_path)
    
    print(f"✓ Exported {len(df):,} cords to:")
    print(f"  {output_path}")
    print(f"  {output_path.with_suffix('.json')} (metadata)")
    print()
    
    # Test: Build tree for first khipu
    print("Testing tree construction for first khipu...")
    print("-" * 80)
    
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT KHIPU_ID FROM khipu_main LIMIT 1")
    test_khipu = cursor.fetchone()[0]
    conn.close()
    
    tree = extractor.build_cord_tree(test_khipu)
    
    def count_nodes(node):
        return 1 + sum(count_nodes(child) for child in node['children'])
    
    total_nodes = count_nodes(tree)
    print(f"✓ Built tree for khipu {test_khipu}")
    print(f"  Root: {tree['classification']} cord (level {tree['level']})")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Direct children: {len(tree['children'])}")
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
