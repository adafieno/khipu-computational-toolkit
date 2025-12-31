# Khipu Computational Analysis Toolkit

**AI-Assisted Hypothesis Testing for Inka Khipu Research**

*A computational research fork of the [Open Khipu Repository](https://github.com/khipulab/open-khipu-repository)*

[![Original DOI](https://zenodo.org/badge/296378423.svg)](https://zenodo.org/badge/latestdoi/296378423)

## About This Fork

This repository is a research fork focused on **computational hypothesis-testing tools for khipu analysis**. It builds upon the excellent foundational work of the Open Khipu Repository (OKR) to develop rigorous, falsifiable methods for testing interpretive models of Inka khipus.

**This is not a "decipherment" project.** Rather, it provides computational infrastructure to help scholars test hypotheses transparently, quantify uncertainty, and surface structural patterns that may inform future interpretive work.

### Research Goals

1. **Arithmetic validation framework** - Test summation consistency and internal numeric logic (following Medrano & Khosla 2024)
2. **Graph-based structural analysis** - Convert khipus into hierarchical graphs to identify recurring patterns
3. **Hypothesis parameterization** - Represent multiple interpretations explicitly (e.g., color semantics as configurable assumptions)
4. **Pattern discovery with constraints** - Use unsupervised learning while requiring patterns across multiple provenances
5. **Multi-modal feature extraction** - Integrate numeric, color, spatial, and structural data with uncertainty tracking
6. **Expert-in-the-loop validation** - Build checkpoints for domain expert review at each analytical stage

### Project Status

- ✅ **Phase 0: Reconnaissance** - Database analysis and data model documentation (COMPLETE)
- ✅ **Phase 1: Baseline Validation** - Arithmetic validation and numeric decoding pipeline (COMPLETE)
  - Decoded 54,403 cords with numeric values (68.2% coverage)
  - Validated 612 khipus for arithmetic consistency (95.8% have numeric data, avg confidence 0.947)
  - Exported processed datasets: cord_numeric_values.csv, validation_results_full.json
- ✅ **Phase 2: Extraction Infrastructure** - Cord/knot extractors with validation hooks (COMPLETE)
  - Cord hierarchy extractor: 54,403 cords, 16.9% missing attachments, avg confidence 0.949
  - Knot data extractor: 110,151 knots, 95.2% with numeric values, avg confidence 0.896
  - Color extractor: 56,306 color records, 15,125 white cords (26.8%), RGB mappings
  - Graph builder: 612 NetworkX graphs, 45,204 nodes, 44,592 edges
  - Exported: cord_hierarchy.csv, knot_data.csv, color_data.csv, white_cords.csv, khipu_graphs.pkl
- ✅ **Phase 3: Summation Hypothesis Testing** - Test white cord boundaries, pendant sum patterns (COMPLETE)
  - Tested all 612 khipus for pendant-to-parent summation patterns
  - 26.3% of khipus exhibit summation relationships (161/612)
  - White cords confirmed as boundary markers (+9.1% higher match rates)
  - Identified white cord boundary markers across dataset (454 khipus, 73.3%)
  - Exported: summation_test_results.csv with detailed analysis
- ✅ **Phase 4: Pattern Discovery** - Clustering and motif mining with provenance-aware constraints (COMPLETE)
  - High-match khipu analysis: 9 khipus with ≥80% match rate identified (1.5%)
  - Hierarchical summation testing: Multi-level pendant summation patterns detected (612 tested)
  - Graph similarity analysis: 191,091 comparisons, mean similarity 0.065, 61 perfect matches
  - Clustering analysis: K=7 optimal, 7 structural archetypes (medium khipus 63.2% dominant)
  - Geographic correlation: Significant regional differences (p<0.001), Incahuasi 48.1% summation vs 17.8% Pachacamac
  - Template extraction: 8 perfect-summation khipus (all linear chains), 27 perfect structural matches
  - Motif mining: 7 universal branching patterns, single pendant (1,(0,),True,1) fundamental building block
  - Finding: Hierarchical standardization — universal micro-patterns, empire-wide archetypes, regional adaptations
  - Exported: high_match_khipus.csv, cluster_assignments_kmeans.csv, geographic_correlation_analysis.json, template_analysis.json, motif_mining_results.json
- ✅ **Phase 5: Multi-Model Framework** - Hypothesis evaluation with uncertainty quantification (COMPLETE)
  - Color semantics hypothesis testing: 4 hypotheses tested with statistical validation
    - White boundaries: MIXED verdict (+10.7% summation improvement, 28.9% vs 18.2%)
    - Color-value correlation: NOT SUPPORTED (p=0.92)
    - Color-function patterns: SUPPORTED (+57% more colors in accounting, 5.22 vs 3.33)
    - Provenance semantics: NOT SUPPORTED (p=1.00, uniform across regions)
  - Function classification: Random forest classifier (98% accounting vs 2% narrative)
    - Key discriminators: Numeric coverage (39.9%), color diversity (26.8%), branching (17.9%)
    - All provenances show 95-100% accounting rate
  - Visualization suite: 8 publication-quality visualizations (300 DPI)
    - Cluster PCA plots, geographic heatmaps, motif frequency charts
    - Provenance feature comparisons showing regional specialization
  - Comprehensive research report: RESEARCH_REPORT.md (12 sections, 3 appendices)
  - Exported: color_hypothesis_tests.json, khipu_function_classification.csv, 8 PNG visualizations
- ✅ **Phase 6: Advanced Visualizations** - Interactive dashboards and 3D viewers (COMPLETE)
  - Interactive web dashboard with real-time filtering across 612 khipus
  - Geographic map visualization of the Andes region with 15+ archaeological sites (400+ khipus plotted)
  - 3D khipu structure viewer (command-line + interactive web version)
  - Dropdown selection interface for easy khipu browsing
  - Exported: dashboard_app.py, interactive_3d_viewer.py, visualize_3d_khipu.py
- ✅ **Phase 7: ML Extensions** - Advanced machine learning for quality control and restoration (COMPLETE)
  - **Anomaly Detection:** 13 high-confidence anomalies identified (2.1% of dataset)
    - Three detection methods: Isolation Forest (31 anomalies), statistical outliers (2), topology analysis (126)
    - 2 khipus flagged by all three methods (1000020: 771 nodes, 1000279: 592 nodes)
    - Cluster 5 shows 66.7% anomaly rate suggesting data quality issues
  - **Function Classification:** 98% accounting vs 2% narrative (589 khipus with >90% confidence)
    - Random Forest classifier with cross-validation
    - Key features: Numeric coverage (39.9%), color diversity (26.8%), structural complexity (17.9%)
  - **Sequence Prediction:** 17,321 missing cord values predicted (31.8% of gaps filled)
    - Constraint-based summation inference: 1,295 high-confidence predictions
    - Sibling pattern matching: 773 predictions
    - Random Forest ML baseline: 15,253 predictions (MAE: 258.40)
  - **Visualization Suite:** 5 publication-quality ML results visualizations
    - Anomaly overview with cluster distributions and score analysis
    - High-confidence anomaly details with method agreement
    - Function classification results with confidence metrics
    - Prediction performance statistics and value distributions
  - Exported: anomaly_detection_results.csv, cord_value_predictions.csv, 5 PNG visualizations, ML_RESULTS_SUMMARY.txt

## Repository Structure

```
├── data/              # Processed data and graph representations
│   └── processed/     # Analysis outputs (40+ files)
│       ├── cluster_assignments_kmeans.csv
│       ├── summation_test_results.csv
│       ├── color_hypothesis_tests.json
│       ├── khipu_function_classification.csv
│       ├── anomaly_detection_results.csv
│       ├── cord_value_predictions.csv
│       └── [35+ more analysis files]
├── visualizations/    # Generated visualizations
│   ├── clusters/      # 4 cluster analysis plots + summary table
│   ├── geographic/    # 2 provenance comparison plots
│   ├── motifs/        # 2 motif frequency charts
│   └── ml_results/    # 5 ML visualization plots + summary report
├── notebooks/         # Jupyter notebooks for exploration
├── src/               # Python source code
│   ├── extraction/    # Database extraction tools
│   ├── graph/         # Graph construction and analysis
│   ├── numeric/       # Numeric constraint solving
│   ├── patterns/      # Pattern discovery algorithms
│   └── visualization/ # Visualization tools
├── khipu_computational_toolkit/scripts/           # Analysis scripts
│   ├── test_summation_hypotheses.py
│   ├── test_color_hypotheses.py
│   ├── classify_khipu_function.py
│   ├── visualize_clusters.py
│   ├── visualize_geographic_motifs.py
│   ├── dashboard_app.py           # Interactive web dashboard
│   ├── interactive_3d_viewer.py   # 3D khipu browser
│   ├── detect_anomalies.py        # ML anomaly detection
│   ├── predict_missing_values.py  # Sequence prediction for restoration
│   └── visualize_ml_results.py    # ML results visualization
├── khipu_computational_toolkit/docs/              # Project documentation and findings
├── khipu_computational_toolkit/reports/           # Generated analysis reports
├── RESEARCH_REPORT.md # Comprehensive research report (12 sections)
└── khipu.db          # OKR SQLite database (612 analyzed khipus)
```

## Original Open Khipu Repository

This research project is built entirely upon the [Open Khipu Repository](https://github.com/khipulab/open-khipu-repository), maintained by:

- **Advisory Board:** Carrie Brezine, Jon Clindaniel, Iván Ghezzi, Sabine Hyland, Manuel Medrano
- **Administration:** Open Khipu Research Laboratory (Jon Clindaniel, jclindaniel@uchicago.edu)
- **Contributors:** Robert and Marcia Ascher, Carrie Brezine, Jon Clindaniel, Hugo Pereyra, Kylie Quave, Gary Urton

### About Khipus

Inka khipus were unique pre-Columbian Andean recording devices using three-dimensional signs—knots, cords, and colors—as symbols functionally similar to early writing systems. Spanish chronicles and contemporary studies indicate khipus recorded everything from accounting to historical narratives.

**Status:** The numeric encoding system is well-understood (decimal positional notation established by Locke 1912, refined by Ascher & Ascher). Recent work (Medrano & Khosla 2024) found ~74% of khipus contain internally consistent arithmetic summations. However, semantic meaning beyond accounting remains debated, with no consensus on whether khipus encoded narrative, categorical, or linguistic information.

### Prior Computational Work

This project builds on:
- **Medrano & Khosla (2024):** Algorithmic analysis of 650 khipus, arithmetic consistency testing, white cord boundary detection
- **Clindaniel (2024):** Transformer-based clustering of cord attributes, latent categorical structure discovery
- **Ascher & Ascher:** Foundational comparative datasets and arithmetic structure documentation

Our contribution focuses on **validation infrastructure** and **hypothesis testing frameworks** rather than new interpretive claims.

### Data Source

The OKR contains:
- **612 analyzed khipus** from Harvard Khipu Database (619 total, 7 filtered for having no cord data)
- **45,204 cords** with hierarchical structure and physical attributes (from analyzed khipus)
- **110,677 knots** with type, position, and numeric encoding
- **56,306 color records** with detailed multi-color specifications
- Geographic provenance from 53 archaeological sites

## Visualization Tools

This project includes advanced interactive visualizations:

### 🌐 Interactive Web Dashboard
- **File:** `khipu_computational_toolkit/scripts/dashboard_app.py`
- **Launch:** `streamlit run khipu_computational_toolkit/scripts/dashboard_app.py`
- **Features:** Real-time filtering, multi-tab analysis, **interactive Andes region map** showing 15+ locations, data export
- Explore all 612 khipus with interactive plots and statistical comparisons
- **NEW:** Geographic tab now displays complete Andes region map with 400+ khipus plotted across archaeological sites

### 📐 3D Khipu Structure Viewer
- **Command-line:** `python khipu_computational_toolkit/scripts/visualize_3d_khipu.py --khipu-id 1000000 --multi-view`
- **🌟 Interactive Web Viewer (NEW):** `streamlit run khipu_computational_toolkit/scripts/interactive_3d_viewer.py --server.port 8502`
- **Features:** Dropdown selection of all 612 khipus, live statistics, elevation/azimuth controls, hierarchical 3D layout, summation flow visualization
- **Why use web viewer:** No command-line arguments needed, browse all khipus easily, see provenance and metadata

### 🗺️ Geographic Heatmap
- **File:** `khipu_computational_toolkit/scripts/visualize_geographic_heatmap.py`
- **Usage:** `python khipu_computational_toolkit/scripts/visualize_geographic_heatmap.py`
- **Features:** Interactive maps with provenance statistics, summation rate heatmaps

### 📊 Quick Start

Run multiple viewers simultaneously:
```bash
# Terminal 1: Main dashboard (port 8501)
streamlit run khipu_computational_toolkit/scripts/dashboard_app.py

# Terminal 2: 3D viewer (port 8502)
streamlit run khipu_computational_toolkit/scripts/interactive_3d_viewer.py --server.port 8502
```

📖 **See [khipu_computational_toolkit/docs/VISUALIZATIONS_GUIDE.md](khipu_computational_toolkit/docs/VISUALIZATIONS_GUIDE.md) for complete documentation**

## Interactive Notebooks

Explore khipu data with Jupyter notebooks (requires `ipywidgets`):

1. **Cluster Explorer** (`notebooks/01_cluster_explorer.ipynb`): Filter and compare 7 archetypes
2. **Geographic Patterns** (`notebooks/02_geographic_patterns.ipynb`): Regional variation analysis
3. **Khipu Detail Viewer** (`notebooks/03_khipu_detail_viewer.ipynb`): Deep-dive individual khipus
4. **Hypothesis Dashboard** (`notebooks/04_hypothesis_dashboard.ipynb`): Custom statistical testing

## Citation

If you use this research fork, please cite both:

1. **The original OKR:** See their [Zenodo DOI](https://doi.org/10.5281/zenodo.5037551)
2. **This research project:** (Citation details to be added upon publication)

## License

This research fork maintains the **MIT License** of the original OKR.

Copyright (c) 2022 Open Khipu Repository Team (original data)  
Copyright (c) 2025 Agustín Da Fieno Delucchi (computational methods)

## Reports & Documentation

Comprehensive phase reports are available in [khipu_computational_toolkit/reports/](khipu_computational_toolkit/reports/):

- **[Quick Start Guide](khipu_computational_toolkit/reports/QUICK_START.md)** - New to the project? Start here!
- **[Project Progress Summary](khipu_computational_toolkit/reports/PROJECT_PROGRESS_SUMMARY.md)** - Complete overview of all completed work
- **[Phase 0: Reconnaissance](khipu_computational_toolkit/reports/phase0_reconnaissance_report.md)** - Database analysis (619 khipus, 24 tables)
- **[Phase 1: Baseline Validation](khipu_computational_toolkit/reports/phase1_baseline_validation_report.md)** - Numeric decoding (95.8% success)
- **[Phase 2: Extraction Infrastructure](khipu_computational_toolkit/reports/phase2_extraction_infrastructure_report.md)** - Extractors & graphs (55K nodes)
- **[Phase 3: Summation Testing](khipu_computational_toolkit/reports/phase3_summation_testing_report.md)** - Hypothesis validation (74.2% summation)
- **[Phase 4: Pattern Discovery](khipu_computational_toolkit/reports/)** - Clustering, geographic analysis, motif mining
- **[Phase 5: Multi-Model Framework](khipu_computational_toolkit/reports/)** - Hypothesis testing, function classification
- **[Phase 6: Advanced Visualizations](khipu_computational_toolkit/docs/VISUALIZATIONS_GUIDE.md)** - Interactive dashboards and 3D viewers
- **[Phase 7: ML Extensions](visualizations/ml_results/ML_RESULTS_SUMMARY.txt)** - Anomaly detection, sequence prediction

**Comprehensive Research Report:** [RESEARCH_REPORT.md](RESEARCH_REPORT.md) - Full analysis with 12 sections covering:
- Seven distinct khipu archetypes (K-means clustering, k=7)
- Geographic specialization (Incahuasi 2.7× higher summation than Pachacamac)
- Color semantics (4 hypotheses tested, accounting uses +57% more colors)
- Function classification (98% accounting vs 2% narrative)
- 45× structural complexity range across clusters
- Universal design principles (7 shared motifs)
- Detailed appendices with cluster profiles and statistical tests

## Research Methodology

See [khipu_computational_toolkit/docs/methodology/](khipu_computational_toolkit/docs/methodology/) and [khipu_computational_toolkit/reports/](khipu_computational_toolkit/reports/) for detailed documentation of:
- Database structure and semantics
- Data quality assessments and uncertainty quantification
- Validation strategies and expert-in-the-loop checkpoints
- Hypothesis parameterization and multi-interpretation support
- Overfitting prevention (provenance-aware splitting, cross-validation, perturbation testing)
- Analysis approaches with falsifiable claims
- Findings with confidence intervals and limitations

**Key Principle:** AI surfaces structure and quantifies uncertainty; domain experts provide interpretation and validation.

## Contact & Contributing

This is an active research project. For questions about:
- **Original OKR data:** Contact okr-team@googlegroups.com
- **This research fork:** Agustín Da Fieno Delucchi (adafieno@hotmail.com)

## Acknowledgments

We gratefully acknowledge:
- The Open Khipu Repository team for curating and sharing this invaluable dataset
- The scholars who meticulously recorded khipu data over decades
- The museums and institutions preserving these artifacts
- The Andean communities whose cultural heritage this represents

---

**Note:** This is a research fork for computational analysis. It does not modify or challenge the original OKR data. This project builds **tools for hypothesis testing**, not semantic decipherment. We surface patterns, test falsifiable claims, and quantify uncertainty—interpretation remains the domain of archaeologists, anthropologists, and historians. All findings will be published through appropriate academic channels with full acknowledgment of the OKR's foundational role and prior computational work.
