# Khipu Computational Analysis Toolkit

**A comprehensive computational framework for analyzing Inka khipus from the Open Khipu Repository**

[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.11+-blue)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

## Overview

This standalone toolkit provides a complete computational analysis pipeline for studying khipu structure, numeric encoding, color semantics, and hierarchical patterns. Built as a research fork of the [Open Khipu Repository](https://github.com/khipulab/open-khipu-repository), it focuses on **computational hypothesis-testing tools for khipu analysis** using rigorous, falsifiable methods.

**This is not a "decipherment" project.** Rather, it provides computational infrastructure to help scholars test hypotheses transparently, quantify uncertainty, and surface structural patterns that may inform future interpretive work.

### Research Goals

1. **Arithmetic validation framework** - Test summation consistency and internal numeric logic
2. **Graph-based structural analysis** - Convert khipus into hierarchical graphs to identify recurring patterns
3. **Hypothesis parameterization** - Represent multiple interpretations explicitly (e.g., color semantics as configurable assumptions)
4. **Pattern discovery with constraints** - Use unsupervised learning while requiring patterns across multiple provenances
5. **Multi-modal feature extraction** - Integrate numeric, color, spatial, and structural data with uncertainty tracking
6. **Expert-in-the-loop validation** - Build checkpoints for domain expert review at each analytical stage

### Key Statistics

- 612 khipus analyzed across 9 comprehensive phases
- 54,403 cords decoded with 68.2% coverage
- 110,151 knots processed with 95.2% numeric values
- 7 structural archetypes identified via clustering
- 98% administrative function confirmed
- 17,321 missing values predicted
- 13 high-confidence anomalies detected

## Quick Start

### Prerequisites

**1. Clone the Open Khipu Repository (OKR)**

This toolkit requires access to the OKR database:

```bash
# In your projects directory (e.g., C:\code or ~/projects)
git clone https://github.com/khipulab/open-khipu-repository.git
```

**2. Clone this toolkit**

```bash
# In the SAME parent directory as OKR
git clone https://github.com/adafieno/khipu-computational-toolkit.git
```

**Expected directory structure:**
```
your-projects-directory/
 open-khipu-repository/
    data/
       khipu.db           Database file
    ...
 khipu-computational-toolkit/   This repo
     scripts/
     data/processed/
     ...
```

**3. Verify database location**

```bash
# From the toolkit directory
ls ../open-khipu-repository/data/khipu.db
# Should show: ../open-khipu-repository/data/khipu.db
```

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
streamlit run scripts/interactive_3d_viewer.py --server.port 8502
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

- [**OVERVIEW.md**](OVERVIEW.md) - Detailed project overview and methodology
- [**reports/**](reports/README.md) - Phase reports documenting analysis progress (Phases 0-7)
- [**docs/VISUALIZATIONS_GUIDE.md**](docs/VISUALIZATIONS_GUIDE.md) - 78-page comprehensive visualization guide

## Repository Structure

```
khipu-computational-toolkit/
 scripts/              # Analysis scripts (26 production tools)
    dashboard_app.py              # Interactive web dashboard
    interactive_3d_viewer.py      # 3D khipu visualization
    detect_anomalies.py           # Outlier detection
    predict_missing_values.py     # ML prediction
    ...
 data/
    processed/        # Analysis outputs (40+ CSV files)
    graphs/           # NetworkX graph structures
 visualizations/       # 100+ publication-quality plots
    clusters/
    geographic/
    ml_results/
    motifs/
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
 reports/              # Phase reports (0-7)
 docs/                 # Documentation
```

## Features

### Interactive Tools
- **Web Dashboard** - Streamlit app for khipu exploration
- **3D Viewer** - Interactive visualization of khipu structure
- **Jupyter Notebooks** - 4 interactive analysis notebooks

### Analysis Capabilities
-  Summation hypothesis testing (75% numeric cord rate)
-  K-means clustering (7 archetypes)
-  Anomaly detection (Isolation Forest)
-  Missing value prediction (3 ML models)
-  Function classification (98% accuracy)
-  Motif mining (color/structure patterns)
-  Geographic correlation analysis

### Data Outputs
- **40+ processed CSV files** - Analysis results
- **100+ PNG visualizations** - Publication-ready plots
- **8 comprehensive reports** - Phase documentation
- **Graph structures** - NetworkX pickled graphs
- **ML models** - Trained classifiers

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

### Missing Value Prediction

```bash
python scripts/predict_missing_values.py
```

Predicts missing numeric values using ML, sibling patterns, and structural constraints.

### Visualization

```bash
python scripts/visualize_clusters.py
python scripts/visualize_geographic_heatmap.py
```

Generates publication-quality plots.

## Database Configuration

### Option 1: Environment Variable (Recommended)

```bash
# Windows PowerShell
$env:KHIPU_DB_PATH = "..\open-khipu-repository\data\khipu.db"

# Linux/Mac
export KHIPU_DB_PATH="../open-khipu-repository/data/khipu.db"
```

### Option 2: Command-Line Parameter

```bash
python scripts/dashboard_app.py --db ../open-khipu-repository/data/khipu.db
```

### Option 3: Update Scripts (Advanced)

Edit individual scripts to set `DB_PATH` constant if needed.

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

## Citation

If you use this toolkit in your research, please cite:

```
Fieno, A. (2025). Khipu Computational Analysis Toolkit.
https://github.com/adafieno/khipu-computational-toolkit
```

And the Open Khipu Repository:

```
Urton, G., & Brezine, C. (2007-present). The Khipu Database Project.
https://github.com/khipulab/open-khipu-repository
```

## License

MIT License - See [LICENSE](LICENSE) for details.

This toolkit is independent of the Open Khipu Repository but designed to work with its data.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Contact

- **Author:** Agustín Da Fieno Delucchi
- **GitHub:** [@adafieno](https://github.com/adafieno)
- **Issues:** [Report bugs or request features](https://github.com/adafieno/khipu-computational-toolkit/issues)

## Acknowledgments

- Gary Urton and Carrie Brezine for the Open Khipu Repository
- The khipu research community for foundational work on numeric interpretation
- Contributors to NetworkX, scikit-learn, pandas, and matplotlib

---

**Status:**  Production Ready |  Research Complete |  Documentation Complete
