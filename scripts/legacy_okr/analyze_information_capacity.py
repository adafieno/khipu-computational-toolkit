"""
Phase 9.1 Analysis Script: Information Capacity & Efficiency

Executes information-theoretic analysis of khipus.
"""

import sys
from pathlib import Path

# Add src directory to path for config import
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore
from analysis.phase9.information_capacity import run_information_capacity_analysis  # noqa: E402

if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 9.1: INFORMATION CAPACITY & EFFICIENCY ANALYSIS")
    print("=" * 80)
    print()

    analyzer, capacity_data, type_stats, bounds = run_information_capacity_analysis()

    config = get_config()
    output_dir = config.processed_dir / "phase9" / "9.1_information_capacity"

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print(f"  1. Review: {output_dir}")
    print("  2. Visualize: python scripts/visualize_phase9_capacity.py")
    print("  3. Continue with Phase 9.2 (Robustness Analysis)")
