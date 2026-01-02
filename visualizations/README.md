# Khipu Computational Analysis - Visualizations Index

**Project:** Khipu Computational Analysis Toolkit  
**Last Updated:** January 1, 2026  
**Total Visualizations:** 39 files

## Directory Structure

This folder contains all visualizations generated during the Khipu analysis project, organized by research phase. Each phase corresponds to a specific research objective and deliverable documented in the project reports.

---

## Phase 1: Baseline Validation
**Directory:** `phase1_baseline/`  
**Report:** [reports/phase1_baseline_validation_report.md](../reports/phase1_baseline_validation_report.md)  
**Purpose:** Validate the numeric decoding pipeline and establish baseline data quality metrics

### Visualizations (4 files)

1. **numeric_value_distribution.png**
   - Histogram showing the distribution of decoded numeric values across all cords
   - Demonstrates the range and frequency of Ascher positional notation values
   - Key insight: Most values fall within 0-1000 range, with systematic decimal encoding

2. **confidence_scores.png**
   - Distribution of confidence scores for numeric decoding across all khipus
   - Shows validation quality metrics for the decoding process
   - Key insight: Average confidence of 0.947 indicates high reliability

3. **data_coverage_heatmap.png**
   - Heatmap showing data coverage across khipu collection
   - Visualizes which khipus have complete vs. incomplete numeric data
   - Key insight: 68.2% of cords have numeric values, 95.8% of khipus have some data

4. **validation_summary.png**
   - Multi-panel summary of validation results
   - Shows knot decoding success rates, error types, and validation metrics
   - Key insight: 95.2% of knots successfully decoded

---

## Phase 2: Extraction Infrastructure
**Directory:** `phase2_extraction/`  
**Report:** [reports/phase2_extraction_infrastructure_report.md](../reports/phase2_extraction_infrastructure_report.md)  
**Purpose:** Extract hierarchical structure, knot patterns, and color data from the khipu database

### Visualizations (5 files)

1. **cord_hierarchy_depth.png**
   - Distribution of hierarchical cord levels across all khipus
   - Shows depth of pendant relationships (parent-child cord structures)
   - Key insight: Most khipus have 2-3 levels of hierarchy

2. **knot_types_frequency.png**
   - Bar chart showing frequency of different knot types (single, long, figure-eight)
   - Demonstrates the variety of knot configurations used in khipus
   - Key insight: Single knots are most common, but all types are represented

3. **color_code_distribution.png**
   - Distribution of Ascher 64-color codes across the corpus
   - Shows which colors appear most frequently
   - Key insight: White is most common (26.8%), supporting Medrano boundary hypothesis

4. **extraction_quality.png**
   - Quality metrics for extraction processes (hierarchy, knots, colors)
   - Shows completeness and confidence scores for each extraction module
   - Key insight: High extraction confidence (>0.89) across all modules

5. **khipu_size_distribution.png**
   - Histogram showing the number of cords per khipu
   - Demonstrates size variability across the collection
   - Key insight: Most khipus have 20-100 cords, with some outliers over 500

---

## Phase 3: Summation Hypothesis Testing
**Directory:** `phase3_summation/`  
**Report:** [reports/phase3_summation_testing_report.md](../reports/phase3_summation_testing_report.md)  
**Purpose:** Test computational hypotheses about summation relationships between pendant and subsidiary cords

### Visualizations (5 files)

1. **summation_match_distribution.png**
   - Distribution of pendant summation match rates across all khipus
   - Shows what percentage of khipus exhibit perfect or near-perfect summation
   - Key insight: 73.8% of khipus show pendant summation relationships

2. **white_cord_boundary_effect.png**
   - Comparison of summation accuracy between khipus with/without white cord boundaries
   - Tests the Medrano hypothesis about white cords as section markers
   - Key insight: White boundaries correlate with higher summation accuracy

3. **hierarchical_summation_cascade.png**
   - Visualization of multi-level summation across cord hierarchy levels
   - Shows how summation relationships propagate up the hierarchy
   - Key insight: Summation occurs at multiple hierarchical levels

4. **alternative_hypotheses_rejection.png**
   - Statistical analysis rejecting alternative summation hypotheses (multiplication, averaging, etc.)
   - Demonstrates that simple addition is the dominant computational pattern
   - Key insight: Alternative patterns found in <5% of khipus

5. **summation_by_cluster.png**
   - Summation match rates grouped by khipu structural clusters
   - Shows variation in summation patterns across different khipu types
   - Key insight: Cluster 0 and 2 show highest summation rates (>80%)

---

## Phase 4: Pattern Discovery
**Directory:** `phase4_patterns/`  
**Reports:** [reports/phase4_pattern_discovery_progress.md](../reports/phase4_pattern_discovery_progress.md)  
**Purpose:** Discover structural patterns, motifs, and geographic correlations across the khipu corpus

### Visualizations (11 files)

#### Clustering Analysis
1. **cluster_pca_plot.png**
   - 2D PCA projection of khipus clustered by structural features
   - Shows separation of khipus into distinct structural groups
   - Key insight: 6 distinct clusters identified through k-means

2. **cluster_sizes.png**
   - Bar chart showing the number of khipus in each cluster
   - Demonstrates distribution across structural groups
   - Key insight: Clusters range from 34 to 166 khipus

3. **cluster_summary_table.csv**
   - Tabular data with cluster statistics (centroids, feature means, etc.)
   - Provides quantitative summary of cluster characteristics

4. **feature_distributions.png**
   - Box plots showing distribution of key features across clusters
   - Compares structural properties (depth, cord count, numeric density) by cluster
   - Key insight: Each cluster has distinct structural signature

5. **provenance_pca_plot.png**
   - PCA projection with points colored by geographic provenance
   - Shows geographic clustering vs. structural similarity
   - Key insight: Some provenances cluster together, suggesting regional styles

#### Geographic Analysis
6. **provenance_features.png**
   - Heatmap showing feature values aggregated by geographic provenance
   - Compares structural characteristics across regions
   - Key insight: Coastal vs. highland differences in khipu construction

7. **summation_by_provenance.png**
   - Bar chart showing summation match rates by provenance
   - Tests whether certain regions used summation more consistently
   - Key insight: High summation rates across most provenances (65-85%)

#### Motif Mining
8. **motif_frequencies.png**
   - Bar chart showing frequency of discovered structural motifs
   - Identifies recurring patterns in cord arrangements
   - Key insight: Several motifs appear in >100 khipus

9. **universal_motifs.png**
   - Visualization of the most common motifs across all khipus
   - Shows structural patterns that transcend geographic boundaries
   - Key insight: 3-4 "universal" motifs found in >30% of corpus

---

## Phase 5: Multi-Model Framework
**Directory:** `phase5_multimodel/`  
**Report:** [reports/phase5_multi_model_framework_report.md](../reports/phase5_multi_model_framework_report.md)  
**Purpose:** Test multiple computational hypotheses simultaneously (color-value correlations, functional classification, geographic patterns)

### Visualizations (3 files)

1. **color_hypothesis_tests.png**
   - 4-panel figure testing color-related hypotheses:
     - Panel 1: White boundary summation effect
     - Panel 2: Color-value correlation analysis
     - Panel 3: Regional color diversity
     - Panel 4: Hypothesis verdict summary
   - Key insight: White boundaries confirmed as significant; color-value correlation weak

2. **function_classification.png**
   - 3-panel figure showing khipu functional classification results:
     - Panel 1: Distribution of predicted functions (accounting, calendar, narrative)
     - Panel 2: Classification confidence scores
     - Panel 3: Function distribution by structural cluster
   - Key insight: 78% classified as accounting khipus with high confidence

3. **geographic_cluster_correlation.png**
   - Heatmap showing correlation between geographic provenance and structural clusters
   - Tests whether geographic origin predicts structural type
   - Key insight: Moderate correlation (Cramér's V = 0.42), suggesting regional variation with shared traditions

---

## Phase 7: Machine Learning Extensions
**Directory:** `phase7_ml/`  
**Report:** [reports/phase7_ml_extensions_report.md](../reports/phase7_ml_extensions_report.md)  
**Purpose:** Apply supervised and unsupervised machine learning to predict missing values, classify functions, and detect anomalies

### Visualizations (5 files)

1. **anomaly_overview.png**
   - Multi-panel visualization of anomaly detection results
   - Shows distribution of anomaly scores and flagged khipus
   - Key insight: 27 khipus identified as structural anomalies

2. **function_classification.png**
   - Confusion matrix and classification performance metrics
   - Shows accuracy of ML-based functional classification
   - Key insight: 82% accuracy in predicting khipu function from structure

3. **high_confidence_details.png**
   - Detailed analysis of high-confidence predictions
   - Shows characteristics of khipus with clearest functional signatures
   - Key insight: Accounting khipus have most distinctive features

4. **prediction_results.png**
   - Scatter plots showing predicted vs. actual values for missing data imputation
   - Tests ability to predict missing cord counts, depth, etc.
   - Key insight: R² = 0.71 for cord count prediction

5. **ML_RESULTS_SUMMARY.txt**
   - Text file with detailed performance metrics, model parameters, and results
   - Provides quantitative summary of all ML experiments

---

## Phase 8: Comparative Analysis
**Directory:** `phase8_comparative/`  
**Report:** Not yet documented  
**Purpose:** Comparative analysis of chromatic features, administrative typology, and model performance

### Visualizations (6 files)

1. **01_structural_clusters.png**
   - Detailed visualization of structural clustering with enhanced features
   - Builds on Phase 4 analysis with additional structural metrics

2. **02_chromatic_features.png**
   - Analysis of color-based features and their distribution
   - Explores color as a structural vs. semantic dimension

3. **03_feature_importance.png**
   - Feature importance rankings from ML models
   - Shows which structural features best predict khipu properties

4. **04_administrative_typology.png**
   - Classification of khipus into administrative types
   - Tests hypothesis that structure reflects bureaucratic function

5. **05_model_comparison.png**
   - Performance comparison across different ML models
   - Evaluates trade-offs between accuracy, interpretability, and complexity

6. **06_structure_color_correlation.png**
   - Correlation analysis between structural and chromatic features
   - Tests independence vs. interdependence of these dimensions

---

## Phase 9: Meta-Analysis & Stability Testing
**Directory:** `phase9_stability/`  
**Report:** Not yet documented  
**Purpose:** Evaluate robustness, information capacity, and stability of analysis methods

### Visualizations (4 files)

1. **information_capacity.png**
   - 2-panel analysis of information encoding capacity:
     - Panel 1: Numeric entropy distribution
     - Panel 2: Information capacity vs. khipu size
   - Key insight: Median entropy of 4.2 bits, strong correlation with cord count

2. **robustness_analysis.png**
   - 2-panel robustness testing:
     - Panel 1: Robustness score distribution
     - Panel 2: Error sensitivity analysis
   - Key insight: 82% of khipus show high robustness to data perturbations

3. **stability_testing.png**
   - 4-panel stability analysis:
     - Panel 1: Clustering stability across random seeds
     - Panel 2: Cross-validation performance
     - Panel 3: Feature ablation importance
     - Panel 4: Data masking sensitivity
   - Key insight: Clustering consensus of 0.73, stable across perturbations

4. **anomaly_taxonomy.png**
   - Classification of anomaly types and their frequencies
   - Shows what makes certain khipus unusual
   - Key insight: 3 main anomaly types (structural, chromatic, numeric)

---

## File Organization Principles

1. **Phase-Based Structure:** Visualizations are organized by research phase to align with project reports and maintain chronological coherence

2. **Descriptive Naming:** File names describe the visualization content rather than generic labels

3. **Consistency:** All PNG files are 300 DPI, suitable for publication

4. **Traceability:** Each visualization can be traced to:
   - Source data file in `data/processed/`
   - Generation script in `scripts/`
   - Analysis methodology in phase reports

5. **Reproducibility:** All visualizations can be regenerated by running the corresponding script:
   - `scripts/visualize_phase1_baseline.py`
   - `scripts/visualize_phase2_extraction.py`
   - `scripts/visualize_phase3_summation.py`
   - `scripts/visualize_phase5_hypotheses.py`
   - `scripts/visualize_phase9_meta.py`
   - Additional scripts for Phases 4, 7, and 8

---

## Missing Phases

- **Phase 0:** No visualizations generated (reconnaissance phase was purely analytical)
- **Phase 6:** No phase6 directory (this phase focused on developing visualization tools, not generating final visualizations)

---

## Usage Notes

### For Researchers
- Start with Phase 1-3 visualizations to understand baseline data quality and core hypotheses
- Phase 4 and 5 visualizations show pattern discovery and multi-hypothesis testing
- Phase 7-9 visualizations demonstrate advanced analytical methods

### For Publications
- All PNG files are high-resolution (300 DPI) and publication-ready
- Each visualization includes clear axis labels, legends, and titles
- Multi-panel figures are designed for journal submission (e.g., 2x2 or 1x4 layouts)

### For Reproduction
- Each visualization is generated from processed CSV/JSON data in `data/processed/`
- Source scripts are available in `scripts/visualize_*.py`
- All visualizations use consistent color schemes and styling for coherence

---

## Related Documentation

- **Project Overview:** [OVERVIEW.md](../OVERVIEW.md)
- **Phase Reports:** [reports/](../reports/)
- **Data Files:** [data/processed/](../data/processed/)
- **Analysis Scripts:** [scripts/](../scripts/)
- **Visualizations Guide:** [docs/VISUALIZATIONS_GUIDE.md](../docs/VISUALIZATIONS_GUIDE.md)

---

**Contact:** Agustín Da Fieno Delucchi  
**Repository:** Khipu Computational Analysis Toolkit (Research Fork of Open Khipu Repository)
