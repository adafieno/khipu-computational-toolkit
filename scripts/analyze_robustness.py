"""
Execute Phase 9.2: Error Detection, Correction, and Robustness Analysis
"""

import sys
from pathlib import Path
import importlib.util

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Dynamic import to avoid module issues
spec = importlib.util.spec_from_file_location(
    "robustness_analysis",
    project_root / "src" / "analysis" / "phase9" / "robustness_analysis.py"
)
robustness_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(robustness_module)

if __name__ == "__main__":
    analyzer, robustness, sensitivity, stats, modes = robustness_module.run_robustness_analysis()
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"\nRobustness Statistics:")
    print(f"  Mean robustness score: {robustness['robustness_score'].mean():.3f}")
    print(f"  Median robustness: {robustness['robustness_score'].median():.3f}")
    print(f"  Std deviation: {robustness['robustness_score'].std():.3f}")
    
    print(f"\nError Detection:")
    print(f"  Khipus with summation (error detection): {sensitivity['has_summation'].sum()}")
    print(f"  Mean detectability rate: {sensitivity['error_detectable'].mean():.1%}")
    
    print(f"\nTop 5 Most Robust Khipus:")
    top_robust = robustness.nlargest(5, 'robustness_score')[['khipu_id', 'robustness_score', 'robustness_class']]
    print(top_robust.to_string(index=False))
    
    print("\n✓ Phase 9.2 complete!")
