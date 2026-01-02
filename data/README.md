# Data Directory Structure

This directory contains all processed analysis outputs from the Khipu Computational Toolkit.

## Directory Layout

```
data/
├── graphs/                    # Graph representations
│   ├── khipu_graphs.pkl      # NetworkX graph objects (612 khipus)
│   └── khipu_graphs_metadata.json
├── processed/                 # Analysis outputs (Phases 1-9)
│   ├── phase1/               # Phase 1: Baseline validation
│   ├── phase2/               # Phase 2: Extraction infrastructure
│   ├── phase3/               # Phase 3: Summation testing
│   ├── phase4/               # Phase 4: Pattern discovery
│   ├── phase5/               # Phase 5: Multi-model framework
│   ├── phase7/               # Phase 7: ML extensions
│   ├── phase8/               # Phase 8: Administrative function analysis
│   └── phase9/               # Phase 9: Meta-analysis (10 modules)
│       ├── 9.1_information_capacity/
│       ├── 9.2_robustness/
│       ├── 9.3_cognitive_load/
│       ├── 9.4_minimalism_expressiveness/
│       ├── 9.5_variance_mapping/
│       ├── 9.6_boundary_phenomena/
│       ├── 9.7_anomaly_taxonomy/
│       ├── 9.8_randomness/
│       ├── 9.9_stability/
│       └── 9.10_negative_knowledge/
└── README.md                  # This file
```

## Phase-Organized Data Files

### Phase 1: Baseline Validation (`processed/phase1/`)

| File | Description | Records |
|------|-------------|---------|
| `cord_numeric_values.csv` | Decoded numeric values for all cords | 54,403 |
| `validation_results_full.json` | Arithmetic validation for all khipus | 612 |
| `validation_results_sample.json` | Sample validation results (10 khipus) | 10 |

### Phase 2: Extraction Infrastructure (`processed/phase2/`)

| File | Description | Records |
|------|-------------|---------|
| `cord_hierarchy.csv` | Hierarchical cord relationships | 54,403 |
| `knot_data.csv` | Individual knot details (type, position, value) | 110,151 |
| `color_data.csv` | Color specifications (RGB + codes) | 56,306 |
| `white_cords.csv` | White cord boundary markers | 15,125 |

### Phase 3: Summation Testing (`processed/phase3/`)

| File | Description | Records |
|------|-------------|---------|
| `summation_test_results.csv` | Pendant-to-primary summation tests | 612 |
| `summation_test_results.json` | Detailed summation analysis | 612 |
| `alternative_summation_results.csv` | Concatenation & multiplication tests | 612 |
| `hierarchical_summation_results.csv` | Multi-level summation patterns | 612 |
| `hierarchical_summation_analysis.json` | Hierarchical summation summary | - |

### Phase 4: Pattern Discovery (`processed/phase4/`)

| File | Description | Records |
|------|-------------|---------|
| `cluster_assignments_kmeans.csv` | K-means cluster assignments (k=7) | 612 |
| `cluster_assignments_hierarchical.csv` | Hierarchical clustering results | 612 |
| `cluster_pca_coordinates.csv` | PCA coordinates for visualization | 612 |
| `cluster_statistics_kmeans.json` | K-means cluster profiles | 7 |
| `cluster_statistics_hierarchical.json` | Hierarchical cluster profiles | varies |
| `high_match_khipus.csv` | Khipus with ≥80% summation match | 9 |
| `high_match_analysis.json` | High-match khipu analysis | - |
| `graph_structural_features.csv` | Graph metrics (nodes, depth, density) | 612 |
| `graph_similarity_matrix.csv` | Pairwise graph similarities | 191,091 |
| `graph_similarity_analysis.json` | Similarity analysis summary | - |
| `most_similar_khipu_pairs.csv` | Top 100 most similar pairs | 100 |
| `template_analysis.json` | Perfect summation templates | 8 |
| `motif_mining_results.json` | Recurring structural patterns | 7 |

### Phase 5: Multi-Model Framework (`processed/phase5/`)

| File | Description | Records |
|------|-------------|---------|
| `color_hypothesis_tests.json` | 4 color semantics hypotheses tested | 4 |
| `geographic_correlation_analysis.json` | Regional pattern analysis | 15 |
| `khipu_function_classification.csv` | Administrative vs narrative (98% vs 2%) | 612 |
| `function_classification_summary.json` | Classification performance metrics | - |

### Phase 6: Advanced Visualizations

No data files - outputs are in `/visualizations/` directory.

### Phase 7: ML Extensions (`processed/phase7/`)

| File | Description | Records |
|------|-------------|---------|
| `anomaly_detection_results.csv` | Outlier detection (3 methods) | 612 |
| `anomaly_detection_detailed.csv` | Per-khipu anomaly scores | 612 |
| `anomaly_detection_summary.json` | Anomaly detection summary | - |
| `high_confidence_anomalies.csv` | High-confidence outliers | 13 |
| `cord_value_predictions.csv` | Missing value predictions | 17,321 |
| `constraint_based_predictions.csv` | Summation-based predictions | 1,295 |
| `sibling_based_predictions.csv` | Sibling pattern predictions | 773 |
| `ml_based_predictions.csv` | Random Forest predictions | 15,253 |
| `value_prediction_summary.json` | Prediction performance summary | - |

## Phase 8: Administrative Function Analysis

Located in `processed/phase8/`:

| File | Description | Records |
|------|-------------|---------|
| `administrative_typology.csv` | 6-class administrative taxonomy | 612 |
| `structural_features.csv` | Structural feature matrix | 612 |
| `chromatic_features.csv` | Color feature matrix | 612 |
| `structural_cluster_assignments.csv` | Structure-based clusters (k=6) | 612 |
| `structural_cluster_statistics.csv` | Cluster profiles | 6 |
| `feature_importance_structure_only.csv` | Structure-only feature importance | 10 |
| `feature_importance_structure_numeric.csv` | Structure+numeric importance | 15 |
| `feature_importance_structure_numeric_color.csv` | Full feature importance | 25 |
| `phase8_metadata.json` | Phase 8 analysis metadata | - |

**Key Finding:** 6 administrative archetypes identified:
- Simple Linear (23.9%)
- Standard Hierarchical (31.2%)
- Complex Hierarchical (18.3%)
- Highly Complex (11.6%)
- Minimal Record (8.5%)
- Deep Hierarchical (6.5%)

## Phase 9: Meta-Analysis Framework

Located in `processed/phase9/` with 10 subdirectories (9.1-9.10):

### 9.1 Information Capacity
- `capacity_metrics.csv` - Information entropy, compression, redundancy (614 khipus)
- `capacity_distribution.csv` - Capacity distribution statistics
- `capacity_summary.json` - Summary statistics

### 9.2 Robustness
- `robustness_metrics.csv` - Consistency under perturbation (612 khipus)
- `robustness_summary.json` - Summary statistics

### 9.3 Cognitive Load
- `cognitive_load_metrics.csv` - Visual complexity scores (619 khipus)
- `cognitive_load_summary.json` - Summary statistics

### 9.4 Minimalism & Expressiveness
- `minimalism_metrics.csv` - Efficiency vs expressiveness (619 khipus)
- `minimalism_summary.json` - Summary statistics

### 9.5 Variance Mapping
- `variance_metrics.csv` - Feature variance analysis (619 khipus)
- `variance_summary.json` - Summary statistics

### 9.6 Boundary Phenomena
- `boundary_metrics.csv` - Edge case detection (619 khipus)
- `boundary_summary.json` - Summary statistics

### 9.7 Anomaly Taxonomy
- `anomaly_taxonomy.csv` - Categorized anomalies (619 khipus)
- `anomaly_categories.json` - Anomaly type definitions
- `analysis_summary.json` - Taxonomy summary

### 9.8 Randomness Testing
- `randomness_metrics.csv` - Statistical randomness tests (612 khipus)
- `randomness_summary.json` - Summary statistics
- `null_comparison.csv` - Comparison with random models

### 9.9 Stability Testing
- `feature_ablation_results.csv` - Feature removal impact (5 features)
- `data_masking_results.csv` - Data corruption robustness (3 levels)
- `clustering_stability.json` - Re-clustering stability (50 runs)
- `cross_validation_results.json` - Classification stability (20 splits)
- `stability_summary.json` - Overall stability assessment

### 9.10 Negative Knowledge Mapping
- `negative_knowledge.json` - What khipus are NOT (comprehensive)
- `failed_hypotheses.csv` - Rejected hypotheses (4 documented)
- `absent_features.csv` - Missing features (4 identified)
- `boundary_conditions.csv` - Confidence levels (5 claims)

## Data Quality Notes

### Coverage
- **612 khipus** consistently analyzed (out of 619 in OKR)
- **7 khipus excluded** due to missing cord data
- **68.2% numeric coverage** (54,403 cords decoded)
- **95.2% knot decoding success** (110,151 knots)

### Confidence Levels
- Numeric validation: 94.7% average confidence
- Cord hierarchy: 94.9% average confidence  
- Knot decoding: 89.6% average confidence

### Missing Data
- 7 khipus lack cord structure data
- 31.8% of cord values missing (17,321 gaps)
- 16.9% of cord attachments unspecified
- 4.8% of knots lack numeric values

## File Format Notes

### CSV Files
- Comma-separated, UTF-8 encoding
- Headers included
- Compatible with pandas, Excel, R

### JSON Files
- Pretty-printed (indent=2)
- UTF-8 encoding
- Nested structures for complex data

### PKL Files
- Python pickle format (NetworkX graphs)
- Requires `networkx` and `pickle` to load
- Not human-readable (use metadata.json for structure)

## Usage Examples

### Load Processed Data (Python)

```python
import pandas as pd
import json
import pickle

# Load CSV from phase directories
numeric_data = pd.read_csv('data/processed/phase1/cord_numeric_values.csv')
color_data = pd.read_csv('data/processed/phase2/color_data.csv')

# Load JSON
with open('data/processed/phase5/color_hypothesis_tests.json') as f:
    hypothesis_results = json.load(f)

# Load graphs
with open('data/graphs/khipu_graphs.pkl', 'rb') as f:
    graphs = pickle.load(f)
```

### Filter by Phase

```python
# Phase 3: Summation analysis
summation = pd.read_csv('data/processed/phase3/summation_test_results.csv')
high_match = summation[summation['match_percentage'] >= 0.8]

# Phase 4: Clustering
clusters = pd.read_csv('data/processed/phase4/cluster_assignments_kmeans.csv')
cluster_5 = clusters[clusters['cluster'] == 5]

# Phase 7: Anomaly detection
anomalies = pd.read_csv('data/processed/phase7/anomaly_detection_results.csv')

# Phase 9: Meta-analysis
capacity = pd.read_csv('data/processed/phase9/9.1_information_capacity/capacity_metrics.csv')
```

## Data Provenance

All data derived from:
- **Source:** Open Khipu Repository (OKR)
- **Database:** `khipu.db` (should be in `../open-khipu-repository/data/`)
- **Version:** 2024 snapshot (619 khipus)
- **Processing:** Khipu Computational Toolkit (2025)
- **DOI:** https://doi.org/10.5281/zenodo.5037551

## Notes

- **Do not commit khipu.db** to this repository (use OKR reference)
- Graph files (`.pkl`) are large (~50MB) - consider `.gitignore`
- JSON files removed if CSV equivalents exist (saves space)
- All analysis reproducible from scripts in `/scripts/`

## Contact

Questions about data structure or files:
- **Repository:** https://github.com/adafieno/khipu-computational-toolkit
- **Issues:** https://github.com/adafieno/khipu-computational-toolkit/issues
