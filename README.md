# Khipu Computational Analysis Toolkit

**A comprehensive computational framework for analyzing Inka khipus**

[![Python](https://img.shields.io/badge/Python-3.11+-blue)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()
[![Status](https://img.shields.io/badge/Status-Research%20Tool-blue)]()
[![Docs](https://img.shields.io/badge/docs-live-blue)](https://adafieno.github.io/khipu-computational-toolkit/?utm_source=github&utm_medium=readme)

## Overview

This standalone toolkit provides computational infrastructure for exploring Inka khipu structure, numeric patterns, color distributions, and hierarchical relationships. It analyzes data from the [Open Khipu Repository](https://github.com/khipulab/open-khipu-repository) and focuses on **computational hypothesis-testing tools for khipu analysis** using rigorous, falsifiable methods.

**This is not a "decipherment" project.** Rather, it provides computational infrastructure to help scholars test hypotheses transparently, quantify uncertainty, and surface structural patterns that may inform future interpretive work. All computational results require expert validation and should be understood as exploratory findings, not definitive conclusions.

### Scope: What This Toolkit Does

1. **Arithmetic pattern analysis** - Tests summation consistency and internal numeric relationships
2. **Graph-based structural analysis** - Converts khipus into hierarchical graphs to identify recurring structural patterns
3. **Hypothesis exploration** - Represents multiple interpretations explicitly (e.g., color semantics as configurable assumptions)
4. **Pattern discovery** - Uses unsupervised learning to surface computational signals across multiple provenances
5. **Multi-modal feature extraction** - Integrates numeric, color, spatial, and structural data with uncertainty tracking
6. **Expert-in-the-loop design** - Provides checkpoints for domain expert review at each analytical stage

### Non-Claims: What This Toolkit Does NOT Do

- ❌ **Semantic decoding** - We do not claim to decode the meaning of numeric values, colors, or structures
- ❌ **Ground truth validation** - Computational results are exploratory; we lack external ground truth for most analyses
- ❌ **Cultural interpretation** - Administrative function classifications are operational typologies requiring expert validation
- ❌ **Definitive conclusions** - All findings are probabilistic signals that require archaeological and anthropological contextualization

### Validation Status

- ✅ **Computationally validated** - Numeric decoding, summation testing, clustering algorithms
- ⚠️ **Requires expert review** - Administrative function classifications, color semantics, structural typologies
- 🔄 **Ongoing research** - Pattern interpretations, geographic correlations, functional hypotheses

### Key Statistics

- **612 khipus** analyzed from the Open Khipu Repository
- **54,403 cords** with hierarchical relationships extracted
- **110,677 knots** decoded (all knot records with sufficient data)
- **7 structural clusters** identified via k-means (moderate separation; see [Phase 8](reports/phase8_administrative_function_report.md))
- **72.2%** exhibit numeric patterns consistent with summation relationships
- **55.7% average confidence** in numeric value extractions (44.2% low-confidence)
- **13 high-confidence structural anomalies** detected using computational outlier methods
- **24,043 predictions** generated for confidence improvement (+0.708 avg gain)
- **100+ datasets** generated for reproducible exploration

**Note:** All counts and percentages reflect computational processing results.

### Research Phases

- **Phase 0:** Reconnaissance - Database exploration and viability assessment
- **Phase 1:** Baseline Validation - Numeric decoding pipeline establishment
- **Phase 2:** Extraction Infrastructure - Hierarchical structure and color data extraction
- **Phase 3:** Summation Testing - Arithmetic relationship pattern exploration
- **Phase 4:** Pattern Discovery - Clustering, motif mining, geographic analysis
- **Phase 5:** Multi-Model Framework - Simultaneous hypothesis testing framework
- **Phase 7:** ML Extensions - Confidence improvement predictions and anomaly detection
- **Phase 8:** Comparative Analysis - Chromatic features and operational typology
- **Phase 9:** Meta-Analysis - Stability testing and robustness validation

See [reports/](reports/) for detailed phase documentation.

## Relation to Prior Work

This project is situated within a growing body of computational research on Andean khipus, most notably the work of Medrano & Khosla (2024), which demonstrates, across a large corpus, that many khipus exhibit structured internal summation relationships consistent with earlier observations by Marcia Ascher.

While that work establishes the viability and prevalence of such numeric regularities, the Khipu Computational Toolkit does not attempt to reinterpret or extend those conclusions. Instead, it focuses on operationalization: transforming published data and hypotheses into an exploratory computational environment that supports systematic analysis, visualization, and experimentation.

In particular, this toolkit emphasizes:
- Reproducible data extraction and transformation pipelines
- Exploratory pattern discovery and structural comparison
- Visualization of cord hierarchies and numeric relationships
- Experimental handling of missing or damaged numeric data
- Pedagogical accessibility for students and non-specialist researchers

All analyses remain strictly non-semantic. Any functional classifications, inferred values, or structural groupings produced by this toolkit are intended as exploratory signals only and require independent expert validation before interpretive use.

## Quick Start

### Prerequisites

**1. Access the Open Khipu Repository Database**

This toolkit requires access to the Open Khipu Repository database:

```bash
# In your projects directory (e.g., C:\code or ~/projects)
git clone https://github.com/khipulab/open-khipu-repository.git
```

**2. Clone this toolkit**

```bash
# In the SAME parent directory
git clone [your-repository-url]
```

**Expected directory structure:**
```
your-projects-directory/
 ├── open-khipu-repository/
 │   └── data/
 │       └── khipu.db           ← Database file
 └── khipu-computational-toolkit/
     ├── src/
     ├── scripts/
     ├── data/
     └── DATA_PATHS.md         ← Path configuration guide
```

**3. Verify configuration**

```bash
# From the toolkit directory
python src/config.py
```

This validates that the database is accessible and directories are properly configured.

See [DATA_PATHS.md](DATA_PATHS.md) for detailed configuration options and troubleshooting.

### Installation

```bash
cd khipu-computational-toolkit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
# IMPORTANT: Set the database path
$env:KHIPU_DB_PATH = "..\open-khipu-repository\data\khipu.db"  # Windows
# export KHIPU_DB_PATH="../open-khipu-repository/data/khipu.db"  # Linux/Mac

# Launch interactive web dashboard
streamlit run scripts/dashboard_app.py

# Launch 3D viewer (on port 8502)
streamlit run scripts/khipu_3d_viewer.py --server.port 8502
```

### Execute Analysis Pipeline

All scripts accept an optional `--db` parameter to specify the database location:

```bash
# Phase 1: Extract and validate numeric data
python scripts/test_summation_hypotheses.py --db ../open-khipu-repository/data/khipu.db

# Or set environment variable once
$env:KHIPU_DB_PATH = "..\open-khipu-repository\data\khipu.db"
python scripts/test_summation_hypotheses.py  # Uses env variable
```

## Documentation

- [**reports/**](reports/) - Phase reports documenting analysis progress (Phases 0-9)
- [**visualizations/README.md**](visualizations/README.md) - Complete index of all 39 visualizations with descriptions
- [**docs/API_REFERENCE.md**](docs/API_REFERENCE.md) - Complete API documentation for all modules
- [**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md) - System architecture and design patterns
- [**docs/CONTRIBUTING.md**](docs/CONTRIBUTING.md) - Contribution guidelines and development setup
- [**docs/FAQ.md**](docs/FAQ.md) - Frequently asked questions and troubleshooting
- [**docs/VISUALIZATIONS_GUIDE.md**](docs/VISUALIZATIONS_GUIDE.md) - 78-page comprehensive visualization guide

## Repository Structure

```
khipu-computational-toolkit/
 scripts/              # Analysis scripts (34 scripts, all operational)
    dashboard_app.py              # Interactive web dashboard
    khipu_3d_viewer.py            # 3D khipu visualization
    detect_anomalies.py           # Outlier detection
    predict_missing_values.py     # ML prediction
    visualize_phase*.py           # Phase visualization generators
    ...
 data/
    processed/        # Analysis outputs (100+ files)
    graphs/           # NetworkX graph structures
 visualizations/       # 39 visualization files (organized by phase)
    phase1_baseline/
    phase2_extraction/
    phase3_summation/
    phase4_patterns/
    phase5_multimodel/
    phase7_ml/
    phase8_comparative/
    phase9_stability/
 notebooks/            # 4 Jupyter notebooks
    01_cluster_explorer.ipynb
    02_geographic_patterns.ipynb
    03_khipu_detail_viewer.ipynb
    04_hypothesis_dashboard.ipynb
 src/                  # Python modules
    extraction/       # Data extraction
    analysis/         # Statistical analysis
    graph/            # Graph algorithms
    utils/            # Utilities
 models/               # Trained ML models
 reports/              # Phase reports (0-9)
 docs/                 # Documentation
```

## Features

### Interactive Tools
- **Web Dashboard** - Streamlit app for khipu exploration
- **3D Viewer** - Interactive visualization of khipu structure
- **Jupyter Notebooks** - 4 interactive analysis notebooks

### Analysis Capabilities
- ✓ Summation hypothesis testing (72.2% exhibit consistent numeric patterns)
- ✓ K-means clustering (7 structural groups with moderate separation)
- ✓ Anomaly detection (Isolation Forest and LOF methods)
- ✓ Confidence improvement prediction (24,043 predictions via constraint, sibling, and ML methods)
- ✓ Operational classification (unsupervised typology requiring expert validation)
- ✓ Motif mining (color and structure pattern discovery)
- ✓ Geographic correlation analysis

### Data Outputs
- **100+ processed data files** - Analysis results (CSV, JSON, pickled graphs)
- **39 visualization files** - Analysis plots organized by research phase
- **10 comprehensive reports** - Phase documentation (Phases 0-9) with detailed findings
- **36 analysis scripts** - Reproducible pipeline for all analyses

## Usage Examples

### Dashboard Exploration

```bash
streamlit run scripts/dashboard_app.py
```

Browse khipus by cluster, provenance, summation behavior, and structural features.

### Anomaly Detection

```bash
python scripts/detect_anomalies.py
```

Identifies outliers using Isolation Forest and Local Outlier Factor.

### Confidence Improvement Prediction

```bash
python scripts/predict_missing_values.py
```

Generates improved predictions for low-confidence cord values (<0.5 confidence) using constraint-based, sibling pattern, and Random Forest ML methods. Produces 24,043 predictions with average +0.708 confidence gain.

### Visualization Generation

```bash
# Generate phase-specific visualizations
python scripts/visualize_phase1_baseline.py
python scripts/visualize_phase2_extraction.py
python scripts/visualize_phase3_summation.py
python scripts/visualize_phase5_hypotheses.py
python scripts/visualize_phase9_meta.py

# Additional visualizations
python scripts/visualize_clusters.py
python scripts/visualize_geographic_heatmap.py
```

Generates comprehensive analysis plots organized by research phase.

## Configuration

### Database Path

The toolkit uses a centralized configuration system (see [DATA_PATHS.md](DATA_PATHS.md)).

**Default:** Looks for `../open-khipu-repository/data/khipu.db` (sibling directory)

**Custom location:** Set environment variable:

```bash
# Windows PowerShell
$env:KHIPU_DB_PATH = "C:\path\to\khipu.db"

# Linux/Mac
export KHIPU_DB_PATH="/path/to/khipu.db"
```

**Validate setup:**

```bash
python src/config.py
```

See [DATA_PATHS.md](DATA_PATHS.md) for complete configuration documentation.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black src/ scripts/
flake8 src/ scripts/
```

## Reproducibility

### Dataset Version

- **Source:** Open Khipu Repository (OKR)
- **Database:** khipu.db from OKR GitHub repository
- **Extraction date:** December 2025
- **Khipus analyzed:** 612 with complete cord data

### Environment

- **Python version:** 3.11+
- **Key dependencies:** See [requirements.txt](requirements.txt)
- **Platform tested:** Windows 11, Ubuntu 22.04, macOS Sonoma

### Regenerating Phase Outputs

All phase outputs can be regenerated from raw data:

```bash
# Validate configuration first
python src/config.py

# Generate all processed data
python scripts/generate_processed_data.py

# Or run individual phase extractions
python scripts/extract_cord_hierarchy.py      # Phase 2
python scripts/extract_knot_data.py           # Phase 2
python scripts/extract_color_data.py          # Phase 2
python scripts/test_summation_hypotheses.py   # Phase 3
python scripts/cluster_khipus.py              # Phase 4
```

See individual phase reports in [reports/](reports/) for detailed methodology.

### Data Provenance

All processed datasets include:
- Generation timestamp
- Source data version
- Processing parameters
- Validation checksums (where applicable)

See [docs/DATA_RECONCILIATION.md](docs/DATA_RECONCILIATION.md) for explanations of count differences across phases.

## Citation

If you use this toolkit in your research, please cite:

```
Da Fieno Delucchi, A. (2026). Khipu Computational Analysis Toolkit.
https://github.com/adafieno/khipu-computational-toolkit
```

And the Open Khipu Repository:

```
OKR Team. (2021). The Open Khipu Repository (v1.0) [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.5037551
```

## License

MIT License - See [LICENSE](LICENSE) for details.

This toolkit is designed to work with data from the Open Khipu Repository.

## Contributing

This is a research project and contributions are welcome. To contribute:

1. Review existing documentation in [docs/](docs/)
2. Follow the code style guidelines (Black formatting, flake8 linting)
3. Add tests for new analytical features
4. Document new hypotheses or analytical approaches
5. Ensure reproducibility by including data provenance

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed guidelines.

## Contact

- **Author:** Agustín Da Fieno Delucchi
- **Project:** Khipu Computational Analysis Toolkit

## Acknowledgments

- OKR Team and Advisory Board for the Open Khipu Repository
- The khipu research community for foundational work on numeric interpretation
- Contributors to NetworkX, scikit-learn, pandas, and matplotlib

---

**Note:** This is a research toolkit under active development. Computational findings should be interpreted with appropriate caution and expert validation.
