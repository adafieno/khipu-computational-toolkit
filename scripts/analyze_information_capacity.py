"""
Phase 9.1 Analysis Script: Information Capacity & Efficiency

Executes information-theoretic analysis of khipus.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.phase9.information_capacity import run_information_capacity_analysis

if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 9.1: INFORMATION CAPACITY & EFFICIENCY ANALYSIS")
    print("=" * 80)
    print()
    
    analyzer, capacity_data, type_stats, bounds = run_information_capacity_analysis()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {analyzer.output_dir}")
    print("\nNext steps:")
    print("  1. Review: data/processed/phase9/9.1_information_capacity/")
    print("  2. Visualize: python scripts/visualize_phase9_capacity.py")
    print("  3. Continue with Phase 9.2 (Robustness Analysis)")
