"""
Test KFG Summation Detector

Compare our detection algorithms against KFG ground truth.
This validates our implementation of the 9 summation pattern types.
"""

from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analysis.kfg_summation_detector import KFGSummationDetector
from config_kfg import get_kfg_config


def load_ground_truth(checks_dir: Path,pattern_type: str) -> pd.DataFrame:
    """Load KFG ground truth for one pattern type."""
    relation_file = checks_dir / f"{pattern_type}_relation.csv"
    if relation_file.exists():
        return pd.read_csv(relation_file)
    return pd.DataFrame()


def test_khipu(detector: KFGSummationDetector, kfg_id: str, checks_dir: Path):
    """Test detection for one khipu and compare to ground truth."""
    
    print(f"\n{'='*80}")
    print(f"Testing: {kfg_id}")
    print(f"{'='*80}")
    
    # Run our detector
    summary = detector.summarize_khipu(kfg_id, tolerance=1)
    all_patterns = detector.detect_all_patterns(kfg_id, tolerance=1)
    
    print(f"\nOur Detection:")
    print(f"  Has summation: {summary['has_summation']}")
    print(f"  Total relationships: {summary['total_relationships']}")
    print(f"  Total matches: {summary['total_matches']}")
    print(f"  Overall match rate: {summary['overall_match_rate']:.1%}")
    print(f"  Pattern types detected: {summary['num_pattern_types']}")
    
    if summary['pattern_stats']:
        print(f"\n  By pattern type:")
        for pattern_type, stats in summary['pattern_stats'].items():
            print(f"    {pattern_type:30} {stats['matches']:3}/{stats['total']:3} ({stats['match_rate']:.1%})")
    
    # Load ground truth
    print(f"\nGround Truth:")
    
    cord_based_patterns = [
        'pendant_pendant_sum',
        'colored_pendant_sum',
        'indexed_pendant_sum',
        'subsidiary_pendant_sum',
        'indexed_subsidiary_sum'
    ]
    
    ground_truth_totals = {}
    for pattern_type in cord_based_patterns:
        gt_df = load_ground_truth(checks_dir, pattern_type)
        if not gt_df.empty:
            khipu_rels = gt_df[gt_df['kfg_name'] == kfg_id]
            if not khipu_rels.empty:
                ground_truth_totals[pattern_type] = len(khipu_rels)
                print(f"  {pattern_type:30} {len(khipu_rels):3} relationships")
    
    # Comparison
    print(f"\nComparison:")
    for pattern_type in cord_based_patterns:
        our_count = len(all_patterns.get(pattern_type, []))
        gt_count = ground_truth_totals.get(pattern_type, 0)
        
        if gt_count > 0 or our_count > 0:
            if gt_count == 0:
                status = "❌ False positives" if our_count > 0 else ""
            elif our_count == 0:
                status = "❌ Missed all"
            elif our_count < gt_count:
                status = f"⚠️  Detected {our_count}/{gt_count} ({our_count/gt_count:.1%})"
            elif our_count == gt_count:
                status = "✓ Perfect match"
            else:
                status = f"⚠️  Over-detected: {our_count} vs {gt_count}"
            
            if status:
                print(f"  {pattern_type:30} {status}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test KFG summation detector')
    parser.add_argument('--khipus', nargs='+', help='Specific khipus to test')
    parser.add_argument('--sample', type=int, default=5, help='Number of random khipus to test')
    
    args = parser.parse_args()
    
    # Setup
    config = get_kfg_config()
    db_path = config.get_database_path()
    checks_dir = Path('data/kfg/KFG/KFG/checks')
    
    print("=" * 80)
    print("KFG SUMMATION DETECTOR TEST")
    print("=" * 80)
    print()
    print(f"Database: {db_path}")
    print(f"Checks directory: {checks_dir}")
    print()
    
    detector = KFGSummationDetector(db_path)
    
    # Select test khipus
    if args.khipus:
        test_khipus = args.khipus
    else:
        # Load sample from ground truth
        gt_df = load_ground_truth(checks_dir, 'pendant_pendant_sum')
        if not gt_df.empty:
            # Get khipus with varying numbers of relationships
            khipu_counts = gt_df['kfg_name'].value_counts()
            
            # Sample: 1 with few, 1 with many, rest random
            test_khipus = []
            
            # Khipu with few relationships
            few_rels = khipu_counts[khipu_counts < 5]
            if not few_rels.empty:
                test_khipus.append(few_rels.index[0])
            
            # Khipu with many relationships
            many_rels = khipu_counts[khipu_counts > 20]
            if not many_rels.empty:
                test_khipus.append(many_rels.index[0])
            
            # Random samples
            remaining = khipu_counts.index.drop(test_khipus, errors='ignore')
            if len(remaining) > 0:
                sample_size = min(args.sample - len(test_khipus), len(remaining))
                test_khipus.extend(remaining[:sample_size].tolist())
        else:
            test_khipus = ['KH0001', 'KH0002', 'CM009']
    
    print(f"Testing {len(test_khipus)} khipus:")
    print(f"  {', '.join(test_khipus)}")
    
    # Test each khipu
    for kfg_id in test_khipus:
        try:
            test_khipu(detector, kfg_id, checks_dir)
        except Exception as e:
            print(f"\n❌ Error testing {kfg_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
