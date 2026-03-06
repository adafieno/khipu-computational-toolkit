"""
Phase 2 Summation Pattern Analysis Script

Runs the KFG Summation Detector with enhanced features:
- Performance timing for each pattern type
- Handedness analysis (left/right directionality)
- Dual sum detection (cords with multiple summand windows)
- Figure-8 knot proximity analysis

Generates statistics for the Phase 2 report.

Usage:
    python scripts/run_phase2_analysis.py
"""

from pathlib import Path
import sys
from collections import defaultdict
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.kfg_summation_detector import KFGSummationDetector

ROOT = Path(__file__).resolve().parent.parent
DB_DEFAULT = ROOT / "data" / "kfg" / "khipu_database.db"

def main():
    db_path = DB_DEFAULT
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    print("=" * 80)
    print("PHASE 2: KFG SUMMATION PATTERN ANALYSIS")
    print("=" * 80)
    print()
    
    # Initialize detector with timing enabled
    print("Initializing detector with timing enabled...")
    detector = KFGSummationDetector(str(db_path), enable_timing=True)
    print()
    
    # Get all khipus from the database
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT kfg_id FROM cords ORDER BY kfg_id")
        all_khipus = [row[0] for row in cur.fetchall()]
    
    print(f"Found {len(all_khipus)} khipus in database")
    print()
    
    # Run detector on all khipus
    print("Running detector on all khipus...")
    all_results = {}  # pattern_type -> list of matches across all khipus
    khipu_summary = []
    per_khipu_dual_sums = defaultdict(int)  # kfg_id -> dual sum cord count
    
    for i, kfg_id in enumerate(all_khipus, 1):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(all_khipus)} khipus...")
        
        results_dict = detector.detect_all_patterns(kfg_id)
        
       # Aggregate matches by pattern type
        for pattern_type, matches in results_dict.items():
            if pattern_type not in all_results:
                all_results[pattern_type] = []
            all_results[pattern_type].extend(matches)
            # Track per-khipu dual sum cords
            for m in matches:
                if m.is_dual_sum:
                    per_khipu_dual_sums[kfg_id] += 1
        
        # Summarize per khipu
        has_any_match = any(len(matches) > 0 for matches in results_dict.values())
        total_matches = sum(len(matches) for matches in results_dict.values())
        
        summary = {
            'kfg_id': kfg_id,
            'has_summation': has_any_match,
            'total_matches': total_matches,
        }
        for pt in detector.PATTERN_TYPES:
            summary[f'{pt}_count'] = len(results_dict.get(pt, []))
        
        khipu_summary.append(summary)
    
    print(f"✓ Processed all {len(all_khipus)} khipus")
    total_matches = sum(len(matches) for matches in all_results.values())
    print(f"✓ Found {total_matches} total summation relationships")
    print()
    
    # ========================================================================
    # CORPUS-WIDE STATISTICS
    # ========================================================================
    
    print("=" * 80)
    print("CORPUS-WIDE COVERAGE")
    print("=" * 80)
    print()
    
    khipus_with_summation = sum(1 for s in khipu_summary if s['has_summation'])
    khipus_without = len(khipu_summary) - khipus_with_summation
    
    print(f"Khipus tested:              {len(khipu_summary)}")
    print(f"With any summation pattern: {khipus_with_summation} ({100*khipus_with_summation/len(khipu_summary):.1f}%)")
    print(f"Without any pattern:        {khipus_without} ({100*khipus_without/len(khipu_summary):.1f}%)")
    print()
    
    # Pattern type distribution
    print("=" * 80)
    print("BY PATTERN TYPE")
    print("=" * 80)
    print()
    
    pattern_khipu_counts = defaultdict(int)
    for s in khipu_summary:
        for pt in detector.PATTERN_TYPES:
            if s.get(f'{pt}_count', 0) > 0:
                pattern_khipu_counts[pt] += 1
    
    print(f"{'Pattern Type':<30} {'Khipus':<10} {'Rate':<10}")
    print("-" * 80)
    for pt in sorted(pattern_khipu_counts.keys(), key=lambda x: pattern_khipu_counts[x], reverse=True):
        count = pattern_khipu_counts[pt]
        rate = 100 * count / len(khipu_summary)
        print(f"{pt:<30} {count:<10} {rate:.1f}%")
    print()
    
    # ========================================================================
    # HANDEDNESS ANALYSIS
    # ========================================================================
    
    print("=" * 80)
    print("HANDEDNESS ANALYSIS")
    print("=" * 80)
    print()
    
    handedness_stats = detector.analyze_handedness(all_results)
    
    # Focus on pendant_pendant_sum
    if 'pendant_pendant_sum' in handedness_stats:
        pp_stats = handedness_stats['pendant_pendant_sum']
        left_count = pp_stats['num_left']
        right_count = pp_stats['num_right']
        total = pp_stats['total']
        ratio = pp_stats['handedness_ratio']
        
        # Count khipus with pendant_pendant_sum
        pp_khipu_count = sum(1 for s in khipu_summary if s.get('pendant_pendant_sum_count', 0) > 0)
        
        print(f"Pendant-pendant sum handedness ({pp_khipu_count} khipus with PPS patterns):")
        print()
        print(f"{'Direction':<45} {'Count':<10} {'Rate':<10}")
        print("-" * 80)
        print(f"{'Left-handed (sum cord right of summands)':<45} {left_count:<10} {100*left_count/total:.1f}%")
        print(f"{'Right-handed (sum cord left of summands)':<45} {right_count:<10} {100*right_count/total:.1f}%")
        print(f"{'Total relationships':<45} {total:<10}")
        print()
        
        print(f"Corpus-wide handedness ratio: {ratio:+.2f} (positive = right-biased)")
        print()
        
        # Per-khipu asymmetry analysis (would need per-khipu data)
        print("Note: Per-khipu asymmetry distribution requires individual khipu analysis")
        print()
    
    # Other patterns
    for pt in ['indexed_pendant_sum', 'subsidiary_pendant_sum']:
        if pt in handedness_stats:
            stats = handedness_stats[pt]
            print(f"{pt} handedness:")
            print(f"  Left-handed:  {stats['num_left']} ({100*stats['num_left']/stats['total']:.1f}%)")
            print(f"  Right-handed: {stats['num_right']} ({100*stats['num_right']/stats['total']:.1f}%)")
            print(f"  Total:        {stats['total']}")
            print()
    
    # ========================================================================
    # DUAL SUM ANALYSIS
    # ========================================================================
    
    print("=" * 80)
    print("DUAL SUM DETECTION")
    print("=" * 80)
    print()
    
    dual_sum_stats = detector.analyze_dual_sums(all_results)
    
    print("Dual sum prevalence:")
    print()
    print(f"{'Pattern Type':<30} {'Khipus w/ Dual Sums':<25} {'Dual Sum Rate':<20}")
    print("-" * 80)
    
    for pt in ['pendant_pendant_sum', 'indexed_pendant_sum', 'colored_pendant_sum']:
        if pt in dual_sum_stats:
            stats = dual_sum_stats[pt]
            num_dual = stats['num_dual_sums']
            rate = stats['dual_sum_rate']
            
            # Count khipus with this pattern
            khipu_count = sum(1 for s in khipu_summary if s.get(f'{pt}_count', 0) > 0)
            
            print(f"{pt:<30} {num_dual:<25} {100*rate:.1f}%")
    print()
    
    # Top khipus with extensive dual sums
    top_dual = sorted(per_khipu_dual_sums.items(), key=lambda x: x[1], reverse=True)[:10]
    if top_dual:
        print("Khipus with extensive dual sums:")
        for kfg_id, count in top_dual:
            print(f"  {kfg_id}: {count} cords with dual summation paths")
        print()
    
    # ========================================================================
    # FIGURE-8 KNOT PROXIMITY ANALYSIS
    # ========================================================================
    
    print("=" * 80)
    print("FIGURE-8 KNOT PROXIMITY ANALYSIS")
    print("=" * 80)
    print()
    
    fig8_stats = detector.analyze_figure8_markers(all_results)
    
    print("Figure-8 proximity results:")
    print()
    print(f"{'Pattern Type':<30} {'Matches w/ Figure-8s':<25} {'Proximity Rate':<20}")
    print("-" * 80)
    
    for pt in ['pendant_pendant_sum', 'colored_pendant_sum', 'indexed_pendant_sum', 'subsidiary_pendant_sum']:
        if pt in fig8_stats:
            stats = fig8_stats[pt]
            count = stats['num_with_figure8']
            rate = stats['figure8_rate']
            print(f"{pt:<30} {count:<25} {100*rate:.1f}%")
    print()
    
    # Location distribution
    if 'pendant_pendant_sum' in fig8_stats:
        pp_stats = fig8_stats['pendant_pendant_sum']
        if 'locations' in pp_stats:
            locations = pp_stats['locations']
            total_with_fig8 = pp_stats['num_with_figure8']
            
            print("Figure-8 location distribution (pendant_pendant_sum only):")
            for loc, count in locations.items():
                print(f"  {loc}: {count} ({100*count/total_with_fig8:.1f}%)")
            print()
    
    # ========================================================================
    # PERFORMANCE TIMING
    # ========================================================================
    
    print("=" * 80)
    print("PERFORMANCE TIMING")
    print("=" * 80)
    print()
    
    timing_stats = detector.get_timing_stats()
    
    if timing_stats:
        print(f"{'Pattern Type':<35} {'Total Time (s)':<20} {'Avg per Khipu (ms)':<20}")
        print("-" * 80)
        for pt in detector.PATTERN_TYPES:
            if pt in timing_stats:
                total_time = timing_stats[pt]
                avg_ms = (total_time / len(all_khipus)) * 1000
                print(f"{pt:<35} {total_time:<20.3f} {avg_ms:<20.2f}")
        print()
    else:
        print("No timing data available (timing may not be enabled)")
        print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Update reports/phase2_summation_patterns.md with these statistics.")
    print()

if __name__ == '__main__':
    main()
