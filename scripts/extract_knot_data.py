"""
Extract knot data and export to processed datasets.
"""

from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from extraction.knot_extractor import KnotExtractor  # noqa: E402 # type: ignore


def main():
    print("=" * 80)
    print("KNOT DATA EXTRACTION")
    print("=" * 80)
    print()
    
    # Initialize extractor
    db_path = Path(__file__).parent.parent / "khipu.db"
    extractor = KnotExtractor(db_path)
    
    # Get summary stats first
    print("Analyzing knot structure...")
    print("-" * 80)
    stats = extractor.get_summary_stats()
    
    print(f"Total knots: {stats['total_knots']:,}")
    print(f"Unique cords: {stats['unique_cords']:,}")
    print(f"Unique khipus: {stats['unique_khipus']}")
    print(f"Knots with numeric values: {stats['knots_with_numeric_values']:,} ({stats['knots_with_numeric_pct']:.1f}%)")
    print(f"Missing KNOT_ORDINAL: {stats['missing_ordinal_count']:,} ({stats['missing_ordinal_pct']:.1f}%)")
    print(f"Missing knot_value_type: {stats['missing_value_type_count']:,} ({stats['missing_value_type_pct']:.1f}%)")
    print(f"Average confidence: {stats['average_confidence']:.3f}")
    print()
    
    print("Knot types:")
    for knot_type, count in sorted(stats['knot_types'].items(), key=lambda x: -x[1]):
        print(f"  {knot_type}: {count:,}")
    print()
    
    print("Value types (place values):")
    for value_type, count in sorted(stats['value_types'].items(), key=lambda x: -x[1] if x[0] else 0):
        if value_type:
            print(f"  {int(value_type)}: {count:,}")
        else:
            print(f"  NULL: {count:,}")
    print()
    
    # Export full knot dataset
    print("Exporting knot data...")
    print("-" * 80)
    
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_path = output_dir / "knot_data.csv"
    
    df = extractor.export_knot_data(output_path)
    
    print(f"✓ Exported {len(df):,} knots to:")
    print(f"  {output_path}")
    print(f"  {output_path.with_suffix('.json')} (metadata)")
    print()
    
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print()
    print(f"Generated: {output_path}")
    print()
    print("Next steps:")
    print("  1. Test summation hypotheses with validated data")
    print("  2. Build color extractor")
    print("  3. Construct graph representations")


if __name__ == "__main__":
    main()
