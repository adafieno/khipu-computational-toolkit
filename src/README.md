# Source Code Structure

This directory contains the core Python modules used by analysis scripts. All modules are designed as reusable components that scripts import and orchestrate.

---

## Directory Organization

### analysis/ (12 modules, ~5,400 lines)

**Phase 8 - Administrative Function Analysis:**
- `administrative_function_classifier.py` - Classify khipus by administrative function (648 lines)

**Phase 9 - Meta-Analysis Framework (10 modules):**
- `information_capacity.py` - 9.1 Information capacity and encoding efficiency (353 lines)
- `robustness_analysis.py` - 9.2 Pattern robustness under perturbation (364 lines)
- `cognitive_load.py` - 9.3 Visual complexity and cognitive demands (428 lines)
- `minimalism_expressiveness.py` - 9.4 Efficiency vs expressiveness tradeoffs (354 lines)
- `variance_mapping.py` - 9.5 Feature variance patterns (514 lines)
- `boundary_phenomena.py` - 9.6 Edge cases and outlier detection (392 lines)
- `anomaly_taxonomy.py` - 9.7 Anomaly categorization and classification (432 lines)
- `randomness_testing.py` - 9.8 Test if patterns could arise by chance (389 lines)
- `stability_testing.py` - 9.9 Feature ablation and data masking (363 lines)
- `negative_knowledge.py` - 9.10 Document what khipus are NOT (358 lines)

**Phase 3 - Summation Testing:**
- `summation_tester.py` - Hierarchical summation hypothesis testing (315 lines)

### extraction/ (4 modules, ~1,200 lines)

**Phase 2 - Extraction Infrastructure:**
- `khipu_loader.py` - Load khipus from OKR database with metadata (259 lines)
- `cord_extractor.py` - Extract cord hierarchy and relationships (308 lines)
- `knot_extractor.py` - Extract knot configurations and numeric values (275 lines)
- `color_extractor.py` - Extract color codes and patterns (346 lines)

### graph/ (1 module, 311 lines)

**Phase 4 - Pattern Discovery:**
- `graph_builder.py` - Build NetworkX graph representations of khipus (311 lines)

### utils/ (1 module, 487 lines)

**Core Utilities:**
- `arithmetic_validator.py` - Validate numeric consistency and summation (487 lines)

---

## Usage Pattern

Scripts in `scripts/` import these modules:

```python
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from src
from src.analysis.phase9.information_capacity import run_information_capacity_analysis
from src.extraction.khipu_loader import load_all_khipus
from src.graph.graph_builder import build_khipu_graph
```

---

## Module Design Principles

1. **Single Responsibility:** Each module focuses on one analytical task or extraction function
2. **Reusability:** Modules are imported by multiple scripts
3. **Data Pipeline:** Extraction → Graph → Analysis → Output
4. **Phase Organization:** analysis/ subfolder organizes Phase 9 modules separately

---

## Phase-to-Module Mapping

| Phase | Modules | Location |
|-------|---------|----------|
| Phase 2 | Extraction infrastructure | extraction/ (4 files) |
| Phase 3 | Summation testing | analysis/summation_tester.py |
| Phase 4 | Graph construction | graph/graph_builder.py |
| Phase 8 | Administrative function | analysis/administrative_function_classifier.py |
| Phase 9 | Meta-analysis (10 modules) | analysis/phase9/ (10 files) |
| Core | Arithmetic validation | utils/arithmetic_validator.py |

**Note:** Phases 1, 5, 6, 7 analysis is performed directly in scripts without dedicated src/ modules.

---

## File Statistics

- **Total modules:** 23 Python files
- **Total lines:** ~7,200 lines of code
- **Average module size:** 313 lines
- **Largest module:** variance_mapping.py (514 lines)
- **Most complex phase:** Phase 9 (10 modules, ~3,900 lines)

---

## Development Guidelines

### Adding a New Module

1. **Choose appropriate directory:**
   - `extraction/` - Database extraction and data loading
   - `analysis/` - Statistical analysis and hypothesis testing
   - `graph/` - Graph construction and algorithms
   - `utils/` - Cross-cutting utilities

2. **Create module file:**
   ```bash
   touch src/analysis/my_analysis.py
   ```

3. **Follow template:**
   ```python
   """
   Module description and purpose.
   
   Part of Phase X analysis.
   """
   
   import pandas as pd
   from pathlib import Path
   
   def main_analysis_function():
       """
       Main entry point for analysis.
       
       Returns:
           tuple: (analyzer_object, results_dataframe, summary_dict)
       """
       # Implementation
       pass
   
   if __name__ == "__main__":
       # Optional: Allow module to run standalone
       main_analysis_function()
   ```

4. **Update this README** with module description and line count

### Module Requirements

- **Docstrings:** All functions must have docstrings
- **Type hints:** Use type hints for function parameters (preferred)
- **Error handling:** Graceful handling of missing data
- **Output paths:** Use `Path` objects, create directories if missing
- **Imports:** Keep imports at top, group by standard/third-party/local

---

## Dependencies

All modules depend on:
- Python 3.11+
- pandas, numpy (data manipulation)
- networkx (graph operations)
- sqlite3 (database access)
- scikit-learn (ML operations)

See [requirements.txt](../requirements.txt) for complete list.

---

## Testing

Modules are tested indirectly through script execution:

```bash
# Test extraction modules
python scripts/extract_cord_hierarchy.py

# Test analysis modules
python scripts/analyze_information_capacity.py

# Test graph modules
python scripts/build_khipu_graphs.py
```

Direct unit testing infrastructure is not currently implemented.

---

## See Also

- [scripts/README.md](../scripts/README.md) - Documentation for all analysis scripts
- [docs/API_REFERENCE.md](../docs/API_REFERENCE.md) - Detailed API documentation for all modules
- [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) - System architecture and design patterns
- [reports/](../reports/) - Phase reports documenting when each module was developed

---

**Last Updated:** January 1, 2026  
**Module Count:** 23 files  
**Total Size:** ~7,200 lines
