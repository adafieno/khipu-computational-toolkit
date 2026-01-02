# Scripts Directory

This directory contains 31 production-ready Python scripts for khipu analysis, organized by function. All scripts are designed to be run from the repository root directory.

---

## Quick Reference

| Category | Count | Naming Pattern |
|----------|-------|----------------|
| **Extraction** | 4 | `extract_*.py` |
| **Analysis** | 7 | `analyze_*.py` |
| **Testing** | 4 | `test_*.py` |
| **Visualization** | 6 | `visualize_*.py` |
| **Interactive Tools** | 3 | Dashboard, 3D viewer, etc. |
| **Utilities** | 7 | Processing, graph building, etc. |

**Total:** 31 scripts (~310 KB)

---

## Usage Pattern

All scripts should be run from the repository root:

```bash
# Standard pattern
python scripts/script_name.py

# Interactive apps
streamlit run scripts/dashboard_app.py
```

Scripts automatically add `src/` to the Python path for imports.

---

## Extraction Scripts (Phase 1-2)

Extract structured data from the Open Khipu Repository database.

### extract_cord_hierarchy.py
**Purpose:** Extract cord parent-child relationships and hierarchical depth  
**Output:** `data/processed/phase2/cord_hierarchy.csv` (54,403 cords)  
**Phase:** 2 - Extraction Infrastructure  
**Usage:**
```bash
python scripts/extract_cord_hierarchy.py
```

### extract_knot_data.py
**Purpose:** Extract comprehensive knot-level data (23 features per knot)  
**Output:** `data/processed/phase2/knot_data.csv` (215,504 knots)  
**Phase:** 2 - Extraction Infrastructure  
**Usage:**
```bash
python scripts/extract_knot_data.py
```

### extract_color_data.py
**Purpose:** Extract and analyze color patterns across khipus  
**Output:** `data/processed/phase2/color_data.csv`, `white_cords.csv`  
**Phase:** 2 - Extraction Infrastructure  
**Usage:**
```bash
python scripts/extract_color_data.py
```

### extract_templates.py
**Purpose:** Mine structural templates from high-similarity khipu groups  
**Output:** `data/processed/phase4/structural_templates.json`  
**Phase:** 4 - Pattern Discovery  
**Usage:**
```bash
python scripts/extract_templates.py
```

---

## Analysis Scripts (Phases 3-9)

Perform statistical and computational analysis on extracted data.

### analyze_high_match_khipus.py
**Purpose:** Identify khipus with >80% summation match rate  
**Output:** `data/processed/phase4/high_match_khipus.csv` (9 khipus)  
**Phase:** 4 - Pattern Discovery  
**Report:** [phase4_pattern_discovery_report.md](../reports/phase4_pattern_discovery_report.md)  
**Usage:**
```bash
python scripts/analyze_high_match_khipus.py
```

### analyze_geography.py
**Purpose:** Analyze geographic distribution and regional patterns  
**Output:** `data/processed/phase5/geographic_statistics.json`  
**Phase:** 5 - Multi-Model Framework  
**Usage:**
```bash
python scripts/analyze_geography.py
```

### analyze_geographic_correlations.py
**Purpose:** Test correlations between geography and structural features  
**Output:** `data/processed/phase5/geographic_correlations.csv`  
**Phase:** 5 - Multi-Model Framework  
**Usage:**
```bash
python scripts/analyze_geographic_correlations.py
```

### analyze_administrative_function.py
**Purpose:** Classify khipus by administrative function (6 types)  
**Output:** `data/processed/phase8/administrative_classification.csv` (619 khipus)  
**Phase:** 8 - Administrative Function Analysis  
**Report:** [phase8_administrative_function_report.md](../reports/phase8_administrative_function_report.md)  
**Usage:**
```bash
python scripts/analyze_administrative_function.py
```

### analyze_information_capacity.py
**Purpose:** Quantify information capacity and encoding efficiency  
**Output:** `data/processed/phase9/9.1_information_capacity/` (3 files)  
**Phase:** 9.1 - Information Capacity  
**Report:** [phase9_meta_analysis_report.md](../reports/phase9_meta_analysis_report.md)  
**Usage:**
```bash
python scripts/analyze_information_capacity.py
```

### analyze_robustness.py
**Purpose:** Test pattern robustness under noise perturbation  
**Output:** `data/processed/phase9/9.2_robustness/` (2 files)  
**Phase:** 9.2 - Robustness Analysis  
**Usage:**
```bash
python scripts/analyze_robustness.py
```

### analyze_variance.py
**Purpose:** Map variance patterns across structural features  
**Output:** `data/processed/phase9/9.5_variance_mapping/` (2 files)  
**Phase:** 9.5 - Variance Mapping  
**Usage:**
```bash
python scripts/analyze_variance.py
```

---

## Testing Scripts (Phase 3)

Test specific hypotheses about khipu arithmetic and color encoding.

### test_summation_hypotheses.py
**Purpose:** Test basic summation consistency (pendant → sum cord)  
**Output:** `data/processed/phase3/summation_test_results.csv` (619 khipus)  
**Phase:** 3 - Summation Testing  
**Report:** [phase3_summation_testing_report.md](../reports/phase3_summation_testing_report.md)  
**Result:** 26.3% show consistent summation  
**Usage:**
```bash
# With database path
python scripts/test_summation_hypotheses.py --db ../open-khipu-repository/data/khipu.db

# Using KHIPU_DB environment variable
export KHIPU_DB=/path/to/khipu.db
python scripts/test_summation_hypotheses.py
```

### test_alternative_summation.py
**Purpose:** Test non-standard summation strategies (concatenation, multiplication)  
**Output:** `data/processed/phase3/alternative_summation_results.csv`  
**Phase:** 3 - Summation Testing  
**Result:** Both hypotheses rejected (p<0.001)  
**Usage:**
```bash
python scripts/test_alternative_summation.py
```

### test_hierarchical_summation.py
**Purpose:** Test multi-level hierarchical summation patterns  
**Output:** `data/processed/phase4/hierarchical_summation_results.csv` (619 khipus)  
**Phase:** 4 - Pattern Discovery  
**Result:** 22.5% show hierarchical summation  
**Usage:**
```bash
python scripts/test_hierarchical_summation.py
```

### test_color_hypotheses.py
**Purpose:** Test statistical hypotheses about color encoding  
**Output:** `data/processed/phase5/color_hypothesis_tests.json`  
**Phase:** 5 - Multi-Model Framework  
**Key Finding:** White boundary cords show +10.7% summation rate  
**Usage:**
```bash
python scripts/test_color_hypotheses.py
```

---

## Visualization Scripts (Phases 4-8)

Generate publication-ready static visualizations.

### visualize_clusters.py
**Purpose:** Generate cluster visualizations (PCA, dendrograms, silhouettes)  
**Output:** `visualizations/clusters/*.png` (10+ plots)  
**Phase:** 4 - Pattern Discovery  
**Features:** 2D/3D PCA, t-SNE, hierarchical dendrograms  
**Usage:**
```bash
python scripts/visualize_clusters.py
```

### visualize_3d_khipu.py
**Purpose:** Create 3D hierarchical visualizations of individual khipus  
**Output:** `visualizations/3d_khipu/*.png`  
**Phase:** 6 - Advanced Visualizations  
**Features:** Multi-view, color modes, summation flow  
**Usage:**
```bash
# Basic visualization
python scripts/visualize_3d_khipu.py --khipu-id 1000000

# With options
python scripts/visualize_3d_khipu.py --khipu-id 1000000 --color-mode level --multi-view
```
**Guide:** See [VISUALIZATIONS_GUIDE.md](../docs/VISUALIZATIONS_GUIDE.md) for 78-page reference

### visualize_geographic_heatmap.py
**Purpose:** Generate geographic distribution heatmaps  
**Output:** `visualizations/geographic/*.png`  
**Phase:** 6 - Advanced Visualizations  
**Features:** Interactive Folium maps, provenance clustering  
**Usage:**
```bash
python scripts/visualize_geographic_heatmap.py
```

### visualize_geographic_motifs.py
**Purpose:** Visualize region-specific structural motifs  
**Output:** `visualizations/motifs/*.png`  
**Phase:** 5 - Multi-Model Framework  
**Usage:**
```bash
python scripts/visualize_geographic_motifs.py
```

### visualize_ml_results.py
**Purpose:** Visualize machine learning results (classification, anomalies)  
**Output:** `visualizations/ml_results/*.png`  
**Phase:** 7 - ML Extensions  
**Features:** ROC curves, confusion matrices, feature importance  
**Usage:**
```bash
python scripts/visualize_ml_results.py
```

### visualize_phase8_results.py
**Purpose:** Generate visualizations for administrative function analysis  
**Output:** `visualizations/phase8/*.png`  
**Phase:** 8 - Administrative Function Analysis  
**Features:** Function distribution, complexity analysis  
**Usage:**
```bash
python scripts/visualize_phase8_results.py
```

---

## Interactive Tools

Web-based interactive exploration and analysis tools.

### dashboard_app.py
**Purpose:** Streamlit dashboard for interactive khipu exploration  
**Phase:** 6 - Advanced Visualizations  
**Report:** [phase6_advanced_visualizations_report.md](../reports/phase6_advanced_visualizations_report.md)  
**Features:**
- Multi-level drill-down (cluster → provenance → khipu → cord)
- Real-time filtering and selection
- Interactive Plotly visualizations
- Data export capabilities

**Usage:**
```bash
# Basic launch
streamlit run scripts/dashboard_app.py

# With custom database path
streamlit run scripts/dashboard_app.py -- --db /path/to/khipu.db

# Custom port
streamlit run scripts/dashboard_app.py --server.port 8502
```

**Access:** http://localhost:8501

### interactive_3d_viewer.py
**Purpose:** Interactive 3D khipu visualization with rotation  
**Phase:** 6 - Advanced Visualizations  
**Features:** Real-time rotation, zoom, color modes  
**Usage:**
```bash
python scripts/interactive_3d_viewer.py
```

### mine_motifs.py
**Purpose:** Interactive motif mining and pattern discovery  
**Output:** `data/processed/phase4/mined_motifs.json`  
**Phase:** 4 - Pattern Discovery  
**Usage:**
```bash
python scripts/mine_motifs.py
```

---

## Utility Scripts

Infrastructure scripts for data processing and graph building.

### generate_processed_data.py
**Purpose:** Regenerate all processed data files from source database  
**Output:** Recreates entire `data/processed/` directory structure (Phases 1-9)  
**Warning:** Overwrites existing processed data  
**Runtime:** ~5-10 minutes  
**Usage:**
```bash
python scripts/generate_processed_data.py
```

**Use when:**
- Database schema has changed
- Need to rerun entire analysis pipeline
- Data corruption detected

### build_khipu_graphs.py
**Purpose:** Build NetworkX graph representations of all khipus  
**Output:** `data/graphs/khipu_graphs.pkl` (619 graphs)  
**Phase:** 4 - Pattern Discovery  
**Usage:**
```bash
python scripts/build_khipu_graphs.py
```

### compute_graph_similarities.py
**Purpose:** Compute pairwise graph similarity matrix (619×619)  
**Output:** 
- `data/processed/phase4/graph_structural_features.csv` (14 features)
- `data/processed/phase4/graph_similarity_matrix.csv` (619×619)
- `data/processed/phase4/most_similar_khipu_pairs.csv` (top 20)

**Phase:** 4 - Pattern Discovery  
**Algorithm:** Graph Edit Distance (GED)  
**Runtime:** ~30 minutes  
**Usage:**
```bash
python scripts/compute_graph_similarities.py
```

### cluster_khipus.py
**Purpose:** Run clustering analysis (k-means, hierarchical)  
**Output:** 
- `data/processed/phase4/cluster_assignments_kmeans.csv` (612 khipus)
- `data/processed/phase4/cluster_assignments_hierarchical.csv`
- `data/processed/phase4/cluster_pca_coordinates.csv`

**Phase:** 4 - Pattern Discovery  
**Key Result:** 7 optimal clusters (silhouette=0.402)  
**Usage:**
```bash
python scripts/cluster_khipus.py
```

### classify_khipu_function.py
**Purpose:** Train and evaluate khipu function classifier  
**Output:** 
- `models/function_classifier.pkl`
- `data/processed/phase5/function_classification_results.csv`

**Phase:** 5 - Multi-Model Framework  
**Accuracy:** 98.2% (administrative vs ceremonial)  
**Usage:**
```bash
python scripts/classify_khipu_function.py
```

### detect_anomalies.py
**Purpose:** Detect statistical and structural anomalies  
**Output:** 
- `data/processed/phase7/anomaly_detection_results.csv` (619 khipus)
- `data/processed/phase7/high_confidence_anomalies.csv` (13 khipus)

**Phase:** 7 - ML Extensions  
**Report:** [phase7_ml_extensions_report.md](../reports/phase7_ml_extensions_report.md)  
**Methods:** Isolation Forest, LOF, statistical outliers  
**Usage:**
```bash
python scripts/detect_anomalies.py
```

### predict_missing_values.py
**Purpose:** Predict missing cord values using ML  
**Output:** `data/processed/phase7/cord_value_predictions.csv` (17,283 predictions)  
**Phase:** 7 - ML Extensions  
**Accuracy:** 73.2% within ±1 for small values  
**Usage:**
```bash
python scripts/predict_missing_values.py
```

---

## Scripts by Phase

### Phase 1: Baseline Validation
- (Validation performed in notebooks, no standalone scripts)

### Phase 2: Extraction Infrastructure
- `extract_cord_hierarchy.py`
- `extract_knot_data.py`
- `extract_color_data.py`

### Phase 3: Summation Testing
- `test_summation_hypotheses.py`
- `test_alternative_summation.py`

### Phase 4: Pattern Discovery
- `analyze_high_match_khipus.py`
- `test_hierarchical_summation.py`
- `extract_templates.py`
- `build_khipu_graphs.py`
- `compute_graph_similarities.py`
- `cluster_khipus.py`
- `mine_motifs.py`
- `visualize_clusters.py`

### Phase 5: Multi-Model Framework
- `analyze_geography.py`
- `analyze_geographic_correlations.py`
- `test_color_hypotheses.py`
- `visualize_geographic_motifs.py`
- `classify_khipu_function.py`

### Phase 6: Advanced Visualizations
- `dashboard_app.py` ⭐
- `interactive_3d_viewer.py` ⭐
- `visualize_3d_khipu.py`
- `visualize_geographic_heatmap.py`

### Phase 7: ML Extensions
- `detect_anomalies.py`
- `predict_missing_values.py`
- `visualize_ml_results.py`

### Phase 8: Administrative Function
- `analyze_administrative_function.py`
- `visualize_phase8_results.py`

### Phase 9: Meta-Analysis
- `analyze_information_capacity.py` (9.1)
- `analyze_robustness.py` (9.2)
- `analyze_variance.py` (9.5)

---

## Common Patterns

### Database Access

Most extraction scripts support flexible database paths:

```bash
# Option 1: Command-line argument
python scripts/script_name.py --db /path/to/khipu.db

# Option 2: Environment variable
export KHIPU_DB=/path/to/khipu.db
python scripts/script_name.py

# Option 3: Default path (if repository is sibling to toolkit)
python scripts/script_name.py
```

### Output Directories

Scripts automatically create output directories if missing:
- `data/processed/phaseN/` - Processed CSV/JSON files
- `visualizations/category/` - PNG/HTML visualizations
- `models/` - Trained ML models

### Dependencies

All scripts use the `src/` package structure:

```python
from src.analysis.phase9.information_capacity import run_analysis
from src.utils.db import get_db_path
from src.visualization.plotters import create_cluster_plot
```

### Error Handling

Scripts include:
- Database path validation
- Data quality checks
- Graceful error messages
- Progress indicators for long operations

---

## Development

### Adding a New Script

1. **Create script:** `scripts/your_analysis.py`
2. **Follow naming convention:** Use prefix (`analyze_`, `test_`, `visualize_`, `extract_`)
3. **Add imports:**
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   from src.module import function
   ```
4. **Update this README** with script description
5. **Test from repository root:** `python scripts/your_analysis.py`

### Script Template

```python
"""
Brief description of script purpose.

Usage: python scripts/script_name.py [options]
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.db import get_db_path
from src.analysis.module import run_analysis

def main():
    print("Starting analysis...")
    results = run_analysis()
    print("Complete!")
    return results

if __name__ == "__main__":
    main()
```

---

## Troubleshooting

### "Module not found" errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Always run scripts from repository root:
```bash
# ✅ Correct (from repository root)
python scripts/script_name.py

# ❌ Wrong (from scripts directory)
cd scripts
python script_name.py  # Will fail
```

### Database path issues

**Problem:** `FileNotFoundError: khipu.db not found`

**Solution:** Set database path explicitly:
```bash
export KHIPU_DB=/full/path/to/khipu.db
python scripts/script_name.py
```

### Streamlit dashboard won't start

**Problem:** Port already in use

**Solution:** Use custom port:
```bash
streamlit run scripts/dashboard_app.py --server.port 8502
```

---

## Performance Notes

### Fast Scripts (<1 minute)
- All `extract_*` scripts
- All `test_*` scripts
- All `analyze_*` scripts (except robustness)
- All `visualize_*` scripts

### Moderate Scripts (1-10 minutes)
- `generate_processed_data.py` (~5 min)
- `build_khipu_graphs.py` (~3 min)
- `cluster_khipus.py` (~2 min)

### Slow Scripts (>10 minutes)
- `compute_graph_similarities.py` (~30 min) - computes 619×619 matrix

---

## See Also

- [Main README](../README.md) - Project overview and quick start
- [API Reference](../docs/API_REFERENCE.md) - Detailed module documentation
- [Phase Reports](../reports/) - Detailed findings from all 10 phases
- [Visualizations Guide](../docs/VISUALIZATIONS_GUIDE.md) - Comprehensive 78-page guide
- [FAQ](../docs/FAQ.md) - Common questions and troubleshooting

---

**Last Updated:** January 1, 2026  
**Script Count:** 31  
**Total Size:** ~310 KB
