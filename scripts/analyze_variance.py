"""
Execute Phase 9.5: Variance Mapping and Standardization Analysis
"""

import sys
from pathlib import Path
import importlib.util

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Dynamic import to avoid module issues
spec = importlib.util.spec_from_file_location(
    "variance_mapping",
    project_root / "src" / "analysis" / "phase9" / "variance_mapping.py"
)
variance_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(variance_module)

if __name__ == "__main__":
    analyzer, variance, numeric, color, structural, stats = variance_module.run_variance_analysis()

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    print("\nVariance Statistics:")
    print(
        f"  Mean flexibility score: {variance['flexibility_score'].mean():.3f}")
    print(
        f"  Mean constraint score: {variance['constraint_score'].mean():.3f}")

    print("\nDimensional Variance:")
    print(f"  Numeric CV (mean): {numeric['cv'].mean():.2f}")
    print(
        f"  Color within-khipu evenness: {color['standardization_score'].mean():.2f}")
    print(f"  Structural CV (mean): {structural['structural_cv'].mean():.2f}")

    print("\nDesign Classes:")
    print(variance['design_class'].value_counts())

    print("\nTop 5 Most Flexible Khipus:")
    top_flexible = variance.nlargest(5, 'flexibility_score')[
        ['khipu_id', 'flexibility_score', 'design_class']]
    print(top_flexible.to_string(index=False))

    print("\nTop 5 Most Constrained Khipus:")
    top_constrained = variance.nlargest(5, 'constraint_score')[
        ['khipu_id', 'constraint_score', 'design_class']]
    print(top_constrained.to_string(index=False))

    print("\nPhase 9.5 complete!")
    print("\nNOTE: Empire-wide color conventions saved to empire_color_conventions.json")
