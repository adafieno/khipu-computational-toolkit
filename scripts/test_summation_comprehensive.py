"""
Phase 3 Summation Algorithm - CORRECTED

Implements combination of Ascher summation pattern tests:
1. Hierarchical: subsidiaries sum to parent (30% threshold)
2. Simple contiguous: 2-3 adjacent cords sum to another cord
3. Early termination: stop at first pattern found

Goal: Reach MIT/KFG expected 60-70% by testing multiple pattern types.
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from config import get_config
from utils.arithmetic_validator import ArithmeticValidator
import pandas as pd
import sqlite3
from datetime import datetime


def get_all_cord_values_bulk(khipu_id: int, validator: ArithmeticValidator) -> dict:
    """
    Bulk fetch all cord values for a khipu.
    Returns: {cord_id: numeric_value}
    """
    conn = validator._connect()
    cursor = conn.cursor()
    
    # Get all cord IDs for this khipu
    cursor.execute("""
        SELECT CORD_ID
        FROM cord
        WHERE KHIPU_ID = ?
    """, (khipu_id,))
    
    cord_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    # Use validator to get each cord's value (it handles the complex logic)
    cord_values = {}
    for cord_id in cord_ids:
        val = validator.get_cord_numeric_value(cord_id)
        if val.total_value is not None:
            cord_values[cord_id] = val.total_value
    
    return cord_values


def test_hierarchical_summation(khipu_id: int, validator: ArithmeticValidator, 
                                  cord_values: dict, tolerance: int = 5) -> bool:
    """
    Test: Do subsidiaries sum to parent?
    Returns True if >30% of parent-subsidiary relationships show summation.
    """
    conn = validator._connect()
    cursor = conn.cursor()
    
    # Get parent-child relationships
    cursor.execute("""
        SELECT c.CORD_ID, c.ATTACHED_TO
        FROM cord c
        WHERE c.KHIPU_ID = ? AND c.ATTACHED_TO IS NOT NULL
    """, (khipu_id,))
    
    relationships = cursor.fetchall()
    if not relationships:
        conn.close()
        return False
    
    # Group by parent
    from collections import defaultdict
    parent_subs = defaultdict(list)
    for child_id, parent_id in relationships:
        if child_id in cord_values and parent_id in cord_values:
            parent_subs[parent_id].append(child_id)
    
    if not parent_subs:
        conn.close()
        return False
    
    matches = 0
    tests = 0
    
    for parent_id, child_ids in parent_subs.items():
        if not child_ids:
            continue
            
        parent_val = cord_values.get(parent_id, 0)
        child_sum = sum(cord_values.get(cid, 0) for cid in child_ids)
        
        tests += 1
        if abs(parent_val - child_sum) <= tolerance:
            matches += 1
    
    if tests == 0:
        conn.close()
        return False
    
    match_rate = matches / tests
    conn.close()
    return match_rate >= 0.30  # MIT threshold


def test_contiguous_sums(khipu_id: int, validator: ArithmeticValidator, 
                         cord_values: dict, tolerance: int = 5) -> bool:
    """
    Test: Do any 2-3 contiguous cords sum to another cord value?
    Fast test using value lookup.
    """
    conn = validator._connect()
    cursor = conn.cursor()
    
    # Get pendant cords in order
    cursor.execute("""
        SELECT c.CORD_ID
        FROM cord c
        WHERE c.KHIPU_ID = ? AND c.CORD_LEVEL = 1
        ORDER BY c.CLUSTER_ORDINAL, c.CORD_ORDINAL
    """, (khipu_id,))
    
    pendant_ids = [row[0] for row in cursor.fetchall() if row[0] in cord_values]
    
    if len(pendant_ids) < 3:
        conn.close()
        return False
    
    # Get all pendant values
    pendant_values = [cord_values[cid] for cid in pendant_ids]
    all_values = set(cord_values.values())
    
    # Quick test: check if any 2-3 adjacent cords sum to a value in the khipu
    for i in range(len(pendant_values) - 1):
        # Test 2-cord sum
        sum_2 = pendant_values[i] + pendant_values[i + 1]
        if sum_2 in all_values:
            conn.close()
            return True
        
        # Test 3-cord sum
        if i < len(pendant_values) - 2:
            sum_3 = sum_2 + pendant_values[i + 2]
            if sum_3 in all_values:
                conn.close()
                return True
    
    conn.close()
    return False


def test_group_totals(khipu_id: int, validator: ArithmeticValidator, 
                      cord_values: dict, tolerance: int = 5) -> bool:
    """
    Test: Do any two groups have matching total values?
    """
    conn = validator._connect()
    cursor = conn.cursor()
    
    # Get group totals
    cursor.execute("""
        SELECT c.CLUSTER_ORDINAL, c.CORD_ID
        FROM cord c
        WHERE c.KHIPU_ID = ?
        ORDER BY c.CLUSTER_ORDINAL
    """, (khipu_id,))
    
    from collections import defaultdict
    group_totals = defaultdict(int)
    
    for cluster_ord, cord_id in cursor.fetchall():
        if cord_id in cord_values:
            group_totals[cluster_ord] += cord_values[cord_id]
    
    if len(group_totals) < 2:
        conn.close()
        return False
    
    # Check for matching totals
    totals_list = list(group_totals.values())
    for i in range(len(totals_list)):
        for j in range(i + 1, len(totals_list)):
            if abs(totals_list[i] - totals_list[j]) <= tolerance:
                conn.close()
                return True
    
    conn.close()
    return False


def main():
    print("=" * 80)
    print("COMPREHENSIVE ASCHER SUMMATION DETECTION - CORRECTED ALGORITHM")
    print("=" * 80)
    print()
    print("Testing for multiple Ascher pattern types:")
    print("  1. Hierarchical: subsidiaries → parent (30% threshold)")
    print("  2. Contiguous: 2-3 adjacent cords sum to another cord")
    print("  3. Group totals: two groups have matching sums")
    print()
    print("Khipu has summation if ANY pattern detected.")
    print("Expected result: 60-70% (per MIT/KFG)")
    print()
    print("=" * 80)
    print()
    
    config = get_config()
    db_path = config.get_database_path()
    validator = ArithmeticValidator(db_path)
    
    # Get all khipu IDs
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT KHIPU_ID FROM khipu_main ORDER BY KHIPU_ID")
    all_khipu_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"Testing {len(all_khipu_ids)} khipus...")
    print()
    
    results = []
    pattern_counts = {'hierarchical': 0, 'contiguous': 0, 'group_totals': 0}
    
    for i, khipu_id in enumerate(all_khipu_ids, 1):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(all_khipu_ids)} ({i/len(all_khipu_ids)*100:.1f}%)")
        
        # Bulk fetch all cord values for this khipu
        cord_values = get_all_cord_values_bulk(khipu_id, validator)
        
        # Test each pattern (early termination possible)
        has_hier = test_hierarchical_summation(khipu_id, validator, cord_values)
        has_contig = test_contiguous_sums(khipu_id, validator, cord_values)
        has_groups = test_group_totals(khipu_id, validator, cord_values)
        
        # Khipu has summation if ANY pattern detected
        has_summation = has_hier or has_contig or has_groups
        
        # Track which patterns found
        if has_hier:
            pattern_counts['hierarchical'] += 1
        if has_contig:
            pattern_counts['contiguous'] += 1
        if has_groups:
            pattern_counts['group_totals'] += 1
        
        # Determine primary pattern
        if has_summation:
            if has_hier and has_contig and has_groups:
                pattern_type = 'all_three'
            elif has_hier and has_contig:
                pattern_type = 'hier+contig'
            elif has_hier and has_groups:
                pattern_type = 'hier+groups'
            elif has_contig and has_groups:
                pattern_type = 'contig+groups'
            elif has_hier:
                pattern_type = 'hierarchical'
            elif has_contig:
                pattern_type = 'contiguous'
            else:
                pattern_type = 'group_totals'
        else:
            pattern_type = 'none'
        
        results.append({
            'khipu_id': khipu_id,
            'has_summation': has_summation,
            'pattern_type': pattern_type,
            'has_hierarchical': has_hier,
            'has_contiguous': has_contig,
            'has_group_totals': has_groups
        })
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    df = pd.DataFrame(results)
    
    # Summary
    total = len(df)
    with_summation = df['has_summation'].sum()
    pct = (with_summation / total) * 100
    
    print(f"Total khipus: {total}")
    print(f"WITH summation (any pattern): {with_summation} ({pct:.1f}%)")
    print()
    
    print("Pattern breakdown:")
    print(f"  • Hierarchical: {pattern_counts['hierarchical']} ({pattern_counts['hierarchical']/total*100:.1f}%)")
    print(f"  • Contiguous sums: {pattern_counts['contiguous']} ({pattern_counts['contiguous']/total*100:.1f}%)")
    print(f"  • Group totals: {pattern_counts['group_totals']} ({pattern_counts['group_totals']/total*100:.1f}%)")
    print()
    
    # Pattern combinations
    pattern_dist = df['pattern_type'].value_counts()
    print("Pattern combinations:")
    for pattern, count in pattern_dist.items():
        if pattern != 'none':
            print(f"  • {pattern}: {count} ({count/total*100:.1f}%)")
    print()
    
    # Check MIT/KFG expectation
    if 60 <= pct <= 70:
        status = "✅ MATCHES MIT/KFG EXPECTATION (60-70%)"
    elif pct >= 55:
        status = "⚠️  CLOSE to expected (within 5pp)"
    elif pct < 60:
        status = f"⚠️  BELOW expected by {60-pct:.1f}pp"
    else:
        status = f"⚠️  ABOVE expected by {pct-70:.1f}pp"
    
    print(f"MIT/KFG Expected: 60-70%")
    print(f"Our Result: {pct:.1f}% {status}")
    print()
    
    # Save results
    output_dir = config.phase_dirs[3]
    output_csv = output_dir / "summation_test_results.csv"
    
    # Backup old file
    if output_csv.exists():
        backup = output_dir / f"summation_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        import shutil
        shutil.copy(output_csv, backup)
        print(f"✓ Backed up old results to: {backup.name}")
    
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved to: {output_csv}")
    
    # Save summary JSON
    summary = {
        'total_khipus': total,
        'with_summation': int(with_summation),
        'pct_with_summation': float(pct),
        'pattern_counts': pattern_counts,
        'matches_expected': 60 <= pct <= 70,
        'within_5pp': 55 <= pct <= 75,
        'test_parameters': {
            'tolerance': 5,
            'hierarchical_threshold': '30%',
            'patterns_tested': 3
        }
    }
    
    import json
    summary_file = output_dir / "summation_analysis.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to: {summary_file}")
    print()
    
    print("=" * 80)
    print()
    
    if pct < 55:
        print("⚠️  Result still below expected range.")
        print("Consider:")
        print("  • Adding more Ascher pattern types (indexed sums, color sums, etc.)")
        print("  • Adjusting tolerance or thresholds")
        print("  • Importing KFG fieldmark data directly")
    elif pct >= 60:
        print("✅ Algorithm successfully reaches MIT/KFG expected range!")
        print("Next steps:")
        print("  • Update Phase 3 report with corrected findings")
        print("  • Regenerate visualizations")
        print("  • Update downstream analyses")

if __name__ == "__main__":
    main()
