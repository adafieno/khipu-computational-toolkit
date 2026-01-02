"""
Phase 8: Administrative Function Analysis - Main Script

Executes complete Phase 8 analysis pipeline:
1. Structural typology (color-agnostic baseline)
2. Chromatic encoding analysis (administrative affordances)
3. Integrated classification (structure + color + numeric)

Generates comprehensive results and visualizations.
"""

import sys
from pathlib import Path

# Add src directory to path for config import
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore

import importlib.util  # noqa: E402


def main():
    """Run Phase 8 analysis."""
    print("\n" + "=" * 80)
    print(" " * 15 + "PHASE 8: ADMINISTRATIVE FUNCTION & ENCODING STRATEGIES")
    print("=" * 80)
    print()
    print("Framing Principles:")
    print("  1. No semantic decoding - operational features only")
    print("  2. Function before interpretation - how used, not what said")
    print("  3. Expert-in-the-loop validation - probabilistic assignments")
    print()
    print("=" * 80)
    print()

    try:
        # Dynamically load the module
        module_path = Path(__file__).parent.parent / "src" / \
            "analysis" / "administrative_function_classifier.py"
        if not module_path.exists():
            print(f"Error: Module not found at {module_path}")
            return 1

        spec = importlib.util.spec_from_file_location(
            "administrative_function_classifier", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["administrative_function_classifier"] = module
        spec.loader.exec_module(module)

        # Run complete analysis
        classifier, typology = module.run_phase8_analysis()

        config = get_config()
        output_dir = config.processed_dir / "phase8"

        print("\n" + "=" * 80)
        print("PHASE 8 ANALYSIS COMPLETE")
        print("=" * 80)
        print()
        print(f"Results saved to: {output_dir}")
        print()
        print("Output files:")
        print("  - structural_features.csv")
        print("  - chromatic_features.csv")
        print("  - structural_cluster_assignments.csv")
        print("  - structural_cluster_statistics.csv")
        print("  - administrative_typology.csv")
        print("  - feature_importance_*.csv (3 models)")
        print("  - phase8_metadata.json")
        print()
        print("Next steps:")
        print("  1. Run: python scripts/visualize_phase8_results.py")
        print(f"  2. Review: {output_dir / 'administrative_typology.csv'}")
        print("  3. Expert validation of probabilistic assignments")
        print()

        return 0

    except Exception as e:
        print(f"\nX Error during Phase 8 analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
