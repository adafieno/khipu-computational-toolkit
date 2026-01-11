"""
Extract knot data and export to processed datasets.
"""

from pathlib import Path
import sys

# Add src to path for runtime
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from extraction.knot_extractor import KnotExtractor  # noqa: E402
from config import get_config  # noqa: E402


def main():
    print("=" * 80)
    print("KNOT DATA EXTRACTION")
    print("=" * 80)
    print()

    # Get configuration
    config = get_config()

    # Validate setup
    validation = config.validate_setup()
    if not validation['valid']:
        print("\nConfiguration errors:")
        for error in validation['errors']:
            print(f"  • {error}")
        sys.exit(1)

    print(f"Database: {config.get_database_path()}")
    print()

    # Initialize extractor
    db_path = config.get_database_path()
    extractor = KnotExtractor(db_path)

    # Get summary stats first
    print("Analyzing knot structure...")
    print("-" * 80)
    stats = extractor.get_summary_stats()

    print(f"Total knots: {stats['total_knots']:,}")
    print(f"Unique cords: {stats['unique_cords']:,}")
    print(f"Unique khipus: {stats['unique_khipus']}")
    print(f"Cords with numeric values: {stats['cords_with_numeric_values']:,} ({stats['cords_with_numeric_pct']:.1f}%)")
    print(f"Missing CLUSTER_ORDINAL: {stats['missing_cluster_ordinal_count']:,} ({stats['missing_cluster_ordinal_pct']:.1f}%)")
    print(f"Missing NUM_TURNS: {stats['missing_num_turns_count']:,} ({stats['missing_num_turns_pct']:.1f}%)")
    print(f"Average confidence: {stats['average_confidence']:.1%}")
    print()

    print("Knot types:")
    for knot_type, count in sorted(stats['knot_types'].items(), key=lambda x: -x[1]):
        print(f"  {knot_type or 'NULL'}: {count:,}")
    print()

    # Export full knot dataset
    print("Exporting knot data...")
    print("-" * 80)

    # Ensure directories exist
    config.ensure_directories()

    # Save to phase2 directory
    output_path = config.get_processed_file('knot_data.csv', phase=2)

    df = extractor.export_knot_data(output_path)

    print(f"✓ Exported {len(df):,} knots to:")
    print(f"  {output_path}")
    print(f"  {output_path.with_suffix('.json')} (metadata)")
    print()

    # Export cord values (FULL DATASET)
    print("Computing cord values for all cords...")
    print("-" * 80)
    print("Note: This computes values for all 54K+ cords (may take 3-5 minutes)")
    print()
    
    import time
    start_time = time.time()
    
    cord_values_path = config.get_processed_file('cord_values.csv', phase=2)
    df_cord_values = extractor.get_cord_values()  # No sample_size = full dataset
    df_cord_values.to_csv(cord_values_path, index=False)
    
    elapsed = time.time() - start_time
    
    print(f"✓ Computed and exported {len(df_cord_values):,} cord values")
    print(f"  {cord_values_path}")
    print(f"  Completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print()
    
    print("Cord value statistics:")
    print(f"  Non-zero values: {(df_cord_values['numeric_value'] > 0).sum():,}")
    print(f"  Average confidence: {df_cord_values['confidence'].mean():.1%}")
    print()

    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print()
    print(f"Generated files:")
    print(f"  1. {output_path} ({len(df):,} knots)")
    print(f"  2. {cord_values_path} ({len(df_cord_values):,} cord values)")
    print()
    print("Next steps:")
    print("  1. Run Phase 3 summation testing with corrected values")
    print("  2. Build color extractor")
    print("  3. Construct graph representations")


if __name__ == "__main__":
    main()