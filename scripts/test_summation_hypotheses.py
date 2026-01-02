"""
Test summation hypotheses across all khipus.
Based on Medrano & Khosla 2024 findings about arithmetic consistency.
"""

from pathlib import Path
import sys

# Add src to path for runtime
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from analysis.summation_tester import SummationTester  # noqa: E402 # type: ignore
from config import get_config  # noqa: E402 # type: ignore


def main():
    print("=" * 80)
    print("SUMMATION HYPOTHESIS TESTING")
    print("=" * 80)
    print()
    print("Testing arithmetic summation patterns per Medrano & Khosla 2024:")
    print("  1. Pendant cords summing to parent cord values")
    print("  2. White cords as boundary markers between sum groups")
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

    # Initialize tester
    db_path = config.get_database_path()
    tester = SummationTester(db_path)

    # Test all khipus
    output_path = config.get_processed_file(
        'summation_test_results.csv', phase=3)

    df = tester.test_all_khipus(output_path)

    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    print(f"Total khipus tested: {len(df)}")
    print()

    print("Pendant Summation:")
    pendant_sum = df['has_pendant_summation'].sum()
    pendant_pct = df['has_pendant_summation'].mean() * 100
    print(f"  Khipus with pendant summation: {pendant_sum} ({pendant_pct:.1f}%)")
    print(f"  Average match rate: {df['pendant_match_rate'].mean():.3f}")
    print(
        f"  Khipus with >0.5 match rate: {(df['pendant_match_rate'] > 0.5).sum()}")
    print(
        f"  Khipus with perfect match (1.0): {(df['pendant_match_rate'] == 1.0).sum()}")
    print()

    print("White Cord Boundaries:")
    white_sum = df['has_white_boundaries'].sum()
    white_pct = df['has_white_boundaries'].mean() * 100
    print(f"  Khipus with white cords: {white_sum} ({white_pct:.1f}%)")
    print(
        f"  Average boundaries per khipu: {df['num_white_boundaries'].mean():.1f}")
    print()

    print("Combined Patterns:")
    both = df['has_pendant_summation'] & df['has_white_boundaries']
    print(
        f"  Khipus with both patterns: {both.sum()} ({both.mean()*100:.1f}%)")
    print()

    print("=" * 80)
    print("FILES GENERATED")
    print("=" * 80)
    print()
    print(f"  {output_path}")
    print(f"  {output_path.with_suffix('.json')} (detailed results)")
    print()

    print("Next steps:")
    print("  1. Analyze specific high-match khipus for patterns")
    print("  2. Build color extractor for white cord validation")
    print("  3. Test hierarchical summation hypotheses")
    print("  4. Construct graph representations")


if __name__ == "__main__":
    main()
