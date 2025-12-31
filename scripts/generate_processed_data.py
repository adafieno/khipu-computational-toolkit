"""
Generate processed data pipeline outputs.
Reads from khipu.db (immutable) and exports decoded/validated data.
"""

from pathlib import Path
import sys
import sqlite3

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from utils.arithmetic_validator import ArithmeticValidator  # noqa: E402 # type: ignore

def main():
    print("=" * 80)
    print("KHIPU DATA PROCESSING PIPELINE")
    print("=" * 80)
    print()
    
    # Initialize validator
    db_path = Path(__file__).parent.parent / "khipu.db"
    validator = ArithmeticValidator(db_path)
    
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Step 1: Exporting decoded cord numeric values...")
    print("-" * 80)
    
    cord_values_path = output_dir / "cord_numeric_values.csv"
    df = validator.export_cord_values(cord_values_path)
    
    print(f"✓ Exported {len(df)} cord values to:")
    print(f"  {cord_values_path}")
    print(f"  {cord_values_path.with_suffix('.json')} (metadata)")
    print()
    print("  Summary:")
    print(f"    Total cords: {len(df):,}")
    print(f"    Cords with numeric values: {len(df[df['numeric_value'].notna()]):,}")
    print(f"    Unique khipus: {df['khipu_id'].nunique()}")
    print(f"    Average confidence: {df['confidence'].mean():.3f}")
    print()
    
    print("Step 2: Running validation tests on ALL khipus...")
    print("-" * 80)
    
    # Run full validation on all khipus
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT KHIPU_ID FROM khipu_main")
    test_khipus = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"  Found {len(test_khipus)} khipus to validate...")
    print()
    
    validation_path = output_dir / "validation_results_full.json"
    results = validator.export_validation_results(validation_path, test_khipus)
    
    print("✓ Exported validation results to:")
    print(f"  {validation_path}")
    print()
    
    # Summary stats
    khipus_with_data = sum(1 for k in results['khipus'].values() if k['has_numeric_data'])
    avg_confidence = sum(k['overall_confidence'] for k in results['khipus'].values()) / len(results['khipus'])
    
    print("  Summary:")
    print(f"    Khipus tested: {len(results['khipus'])}")
    print(f"    Khipus with numeric data: {khipus_with_data}")
    print(f"    Average confidence: {avg_confidence:.3f}")
    print()
    
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print()
    print("Generated files:")
    print(f"  {cord_values_path}")
    print(f"  {cord_values_path.with_suffix('.json')}")
    print(f"  {validation_path}")
    print()
    print("Next steps:")
    print("  1. Review validation results")
    print("  2. Identify gold-standard subset (high confidence khipus)")
    print("  3. Test summation hypotheses")
    print("  4. Build cord/knot extractors with validation hooks")

if __name__ == "__main__":
    main()
