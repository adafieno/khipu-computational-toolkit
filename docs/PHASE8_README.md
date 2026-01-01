# Phase 8: Administrative Function & Encoding Strategies

## Overview

Phase 8 classifies khipus by administrative function using structural, chromatic, and numeric affordances. The analysis explicitly avoids semantic decoding, focusing instead on *how* khipus were used as administrative tools.

## Framing Principles

1. **No semantic decoding** - Operational features only
2. **Function before interpretation** - How used, not what said  
3. **Expert-in-the-loop validation** - Probabilistic assignments

## Three-Stage Analysis

### 8.1: Structural Typology (Color-Agnostic Baseline)
Identifies administrative artifact types based purely on structure and numeric behavior:
- Hierarchy metrics
- Summation patterns
- Numeric coverage
- Structural complexity

### 8.2: Chromatic Encoding (Administrative Affordances)
Analyzes how color usage reinforces administrative function:
- Color diversity and entropy
- Color positioning (primary, pendant, subsidiary)
- Color transitions
- Boundary alignment

### 8.3: Integrated Classification
Combines structure + color + numeric features:
- Random Forest classifier
- Feature importance analysis (SHAP)
- Confidence scoring
- Administrative typology generation

## Quick Start

### Prerequisites

Requires completed Phases 1-7:
- Phase 1-2: Numeric decoding and color extraction
- Phase 3: Summation testing results
- Phase 4: Structural clustering data
- Phase 5: Function classification
- Phase 7: Anomaly detection

### Required Data Files

Ensure these files exist in `data/processed/`:
```
graph_structural_features.csv
summation_test_results.csv
color_data.csv
cord_numeric_values.csv
cluster_assignments_kmeans.csv
cord_hierarchy.csv
khipu_function_classification.csv (optional)
```

### Execution

**Step 1: Run Phase 8 Analysis**
```bash
cd c:\code\khipu-computational-toolkit
python scripts/analyze_administrative_function.py
```

This will:
- Extract structural features (8.1)
- Extract chromatic features (8.2)
- Perform structural clustering
- Build integrated classifiers (3 models)
- Generate final administrative typology
- Save results to `data/processed/phase8/`

**Expected runtime:** ~5-10 minutes for 612 khipus

**Step 2: Generate Visualizations**
```bash
python scripts/visualize_phase8_results.py
```

This generates 6 publication-quality plots in `visualizations/phase8/`:
1. Structural cluster distribution
2. Chromatic feature analysis
3. Feature importance comparison
4. Administrative typology distribution
5. Model performance comparison
6. Structure-color correlation

**Expected runtime:** ~2-3 minutes

## Outputs

### Data Files (`data/processed/phase8/`)

| File | Description |
|------|-------------|
| `structural_features.csv` | Color-agnostic structural features |
| `chromatic_features.csv` | Color affordance features |
| `structural_cluster_assignments.csv` | Cluster labels with quality scores |
| `structural_cluster_statistics.csv` | Cluster centroids and statistics |
| `administrative_typology.csv` | **Final typology with confidence scores** |
| `feature_importance_*.csv` | Feature importance (3 models) |
| `phase8_metadata.json` | Analysis metadata |

### Key Output File: `administrative_typology.csv`

Columns:
- `khipu_id` - Khipu identifier
- `structural_cluster` - Cluster ID (0-6)
- `predicted_function` - Accounting or Narrative
- `confidence` - Classification confidence (0-1)
- `administrative_type` - Final type assignment
- `cord_count`, `hierarchy_depth`, `summation_match_rate` - Key metrics
- `unique_color_count`, `color_entropy` - Chromatic features

### Visualizations (`visualizations/phase8/`)

1. `01_structural_clusters.png` - Cluster distribution and characteristics
2. `02_chromatic_features.png` - Color usage patterns
3. `03_feature_importance.png` - Feature importance (3 models)
4. `04_administrative_typology.png` - Typology distribution
5. `05_model_comparison.png` - Model performance
6. `06_structure_color_correlation.png` - Structure × color

## Expected Results

### Administrative Types

The analysis identifies ~7-10 administrative types, such as:
- **Local Operational Record** - Small, high summation, minimal color
- **Aggregated Summary** - Large, hierarchical, multi-level
- **Standard Administrative Record** - Medium size, accounting function
- **Multi-Category Record** - High color diversity
- **Lateral Category Tracking** - Wide, shallow hierarchy
- **Exceptional/Anomalous** - Outliers

### Model Performance

- **Structure only:** ~95% accuracy
- **Structure + numeric:** ~96% accuracy  
- **Structure + numeric + color:** ~98% accuracy

Color features add ~2-3% accuracy improvement.

### Confidence Scores

- **High confidence (>0.8):** ~70-80% of khipus
- **Medium confidence (0.6-0.8):** ~15-20%
- **Low confidence (<0.6):** ~5-10%

Low-confidence cases require expert validation.

## Interpretation Guidelines

### What the Results Mean

✅ **This analysis identifies:**
- Structural patterns consistent with administrative functions
- Color usage as procedural affordances (visual organization)
- Probabilistic role assignments for expert review

❌ **This analysis does NOT claim:**
- Semantic decoding of khipu content
- Ground truth about specific information encoded
- Definitive functional classifications without expert validation

### Confidence Scores

- **>0.8:** High-confidence assignments, likely correct
- **0.6-0.8:** Medium confidence, review recommended
- **<0.6:** Low confidence, requires expert validation

### Color Interpretation

Color features indicate:
- **Low color diversity:** Pure accounting (numeric records)
- **Medium color diversity:** Multi-category tracking
- **High color diversity:** Narrative or ceremonial function

Color does NOT encode:
- Specific values or quantities
- Geographic locations
- Personal names or semantic content

## Validation Workflow

### Priority 1: Low-Confidence Assignments
Review khipus with confidence < 0.6:
```python
import pandas as pd
typology = pd.read_csv("data/processed/phase8/administrative_typology.csv")
low_conf = typology[typology['confidence'] < 0.6]
print(f"Low confidence: {len(low_conf)} khipus")
print(low_conf[['khipu_id', 'administrative_type', 'confidence']])
```

### Priority 2: Exceptional/Anomalous Type
Examine outliers:
```python
exceptional = typology[typology['administrative_type'] == 'Exceptional/Anomalous']
print(f"Exceptional: {len(exceptional)} khipus")
```

### Priority 3: Archaeological Context
Cross-reference with provenance data to validate administrative types match expected site functions.

## Troubleshooting

### Error: "File not found"
- Ensure all prerequisite phases (1-7) have been completed
- Check that data files exist in `data/processed/`
- Verify you're running from the toolkit root directory

### Error: "No ground truth labels"
- Phase 8 can run without Phase 5 function predictions
- Will use cluster-based heuristic instead
- Results will have slightly different interpretation

### Low cluster quality scores
- Silhouette < 0.3 indicates weak clustering
- May need to adjust number of clusters
- Some overlap between administrative types is expected

### Memory issues
- Analysis requires ~2-4 GB RAM
- Close other applications if needed
- Process can be run in batches if necessary

## Integration with Other Phases

### Builds On:
- **Phase 1-2:** Numeric and color data extraction
- **Phase 3:** Summation hypothesis testing
- **Phase 4:** Structural clustering framework
- **Phase 5:** Function classification baseline
- **Phase 7:** Anomaly detection

### Feeds Into:
- Expert validation workflow
- Archaeological interpretation
- Publication preparation
- Future geographic/temporal submodels

## Citation

When referencing Phase 8 results:

> Da Fieno Delucchi, A. (2026). Administrative function classification 
> using structural and chromatic affordances. In *Khipu Computational 
> Analysis Toolkit*, Phase 8 Report.

## Contact

For questions about Phase 8 methodology or results:
- **Email:** adafieno@hotmail.com
- **GitHub:** [Repository URL]

---

**Last Updated:** January 1, 2026  
**Version:** 1.0  
**Status:** Ready for execution
