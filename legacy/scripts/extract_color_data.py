"""
Extract color data and validate white cord boundary hypothesis.
"""

from pathlib import Path
import sys

# Add src to path for runtime
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from extraction.color_extractor import ColorExtractor  # noqa: E402
from config import get_config  # noqa: E402


def main():
    print("=" * 80)
    print("COLOR DATA EXTRACTION")
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
    extractor = ColorExtractor(db_path)

    # Get summary stats first
    print("Analyzing color data...")
    print("-" * 80)
    stats = extractor.get_summary_stats()

    print(f"Total color records: {stats['total_color_records']:,}")
    print(f"Unique cords with color data: {stats['unique_cords']:,}")
    print(f"Unique khipus: {stats['unique_khipus']}")
    print(f"Unique color codes: {stats['unique_colors']}")
    print(f"White cords: {stats['white_cord_count']:,}")
    print(f"Multi-color cords: {stats['multi_color_cord_count']:,}")
    print(f"Most common color: {stats['most_common_color']}")
    print()

    print("Top 10 colors:")
    for color, count in sorted(stats['color_distribution'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {color}: {count:,}")
    print()

    print("Color categories:")
    for category, count in sorted(stats['color_categories'].items(), key=lambda x: -x[1]):
        print(f"  {category}: {count}")
    print()

    # Analyze white cords specifically
    print("White Cord Analysis (for boundary validation)...")
    print("-" * 80)
    white_cords = extractor.identify_white_cords()

    print(f"Total white cord segments: {len(white_cords):,}")
    print(f"Unique white cords: {white_cords['CORD_ID'].nunique():,}")
    print(f"Khipus with white cords: {white_cords['KHIPU_ID'].nunique()}")
    print(f"Average white cords per khipu: {white_cords.groupby('KHIPU_ID')['CORD_ID'].nunique().mean():.1f}")
    print()

    # Export full color dataset
    print("Exporting color data...")
    print("-" * 80)

    # Ensure directories exist
    config.ensure_directories()

    # Save to phase2 directory
    output_path = config.get_processed_file('color_data.csv', phase=2)

    df = extractor.export_color_data(output_path)

    print(f"✓ Exported {len(df):,} color records to:")
    print(f"  {output_path}")
    print(f"  {output_path.with_suffix('.json')} (metadata)")
    print()

    # Export white cords specifically for boundary analysis
    white_output_path = config.get_processed_file('white_cords.csv', phase=2)
    print("Exporting white cord data for boundary analysis...")
    white_cords.to_csv(white_output_path, index=False)
    print(f"✓ Exported {len(white_cords):,} white cord records to:")
    print(f"  {white_output_path}")
    print()

    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print()
    print("Generated files:")
    print(f"  {output_path}")
    print(f"  {white_output_path}")
    print()
    print("Next steps:")
    print("  1. Analyze color patterns in high-match summation khipus")
    print("  2. Validate white cord boundary hypothesis")
    print("  3. Test color-numeric correlations")
    print("  4. Construct graph representations with color attributes")


if __name__ == "__main__":
    main()
