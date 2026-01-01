# Phase 8 Quick Start Guide

**Execute Phase 8 in 3 simple steps**

## Prerequisites

✅ Completed Phases 1-7  
✅ Data files in `data/processed/`  
✅ Python 3.11+ with required packages

## Step 1: Run Analysis (5-10 minutes)

```bash
cd c:\code\khipu-computational-toolkit
python scripts/analyze_administrative_function.py
```

**What it does:**
- Extracts 11 structural features (color-agnostic)
- Extracts 9 chromatic features (administrative affordances)
- Performs clustering on structural features
- Trains 3 Random Forest models (structure, +numeric, +color)
- Generates final administrative typology with confidence scores

**Output location:** `data/processed/phase8/`

## Step 2: Generate Visualizations (2-3 minutes)

```bash
python scripts/visualize_phase8_results.py
```

**What it does:**
- Creates 6 publication-quality plots
- Analyzes cluster distributions
- Compares feature importance
- Visualizes administrative typology

**Output location:** `visualizations/phase8/`

## Step 3: Review Results

```python
import pandas as pd

# Load final typology
typology = pd.read_csv("data/processed/phase8/administrative_typology.csv")

# View distribution
print(typology['administrative_type'].value_counts())

# Check confidence
print(f"Average confidence: {typology['confidence'].mean():.3f}")
print(f"High confidence (>0.8): {(typology['confidence'] > 0.8).sum()} khipus")

# View low-confidence cases
low_conf = typology[typology['confidence'] < 0.6]
print(f"\nLow confidence: {len(low_conf)} khipus requiring expert review")
```

## Key Output File

**`data/processed/phase8/administrative_typology.csv`**

Contains for each khipu:
- `khipu_id` - Identifier
- `administrative_type` - Final classification
- `confidence` - Classification confidence (0-1)
- `structural_cluster` - Cluster ID (0-6)
- `predicted_function` - Accounting or Narrative
- `cord_count`, `hierarchy_depth`, `summation_match_rate` - Structural metrics
- `unique_color_count`, `color_entropy` - Chromatic metrics

## Expected Results

### Administrative Types (~7-10 types)
- Local Operational Record
- Aggregated Summary
- Standard Administrative Record
- Compact Operational Record
- Lateral Category Tracking
- Multi-Level Aggregation
- Exceptional/Anomalous

### Model Performance
- Structure only: ~95% accuracy
- Structure + numeric + color: ~98% accuracy
- Color adds ~2-3% improvement

### Confidence Distribution
- High (>0.8): ~70-80% of khipus
- Medium (0.6-0.8): ~15-20%
- Low (<0.6): ~5-10% (require expert validation)

## Framing Principles

Remember Phase 8:
1. **Does NOT perform semantic decoding**
2. **Identifies how khipus were used** (not what they said)
3. **Generates probabilistic assignments** (requires expert validation)

## Troubleshooting

**Error: "File not found"**
→ Ensure Phases 1-7 are complete and data files exist

**Low cluster quality (silhouette < 0.3)**
→ Some overlap expected; adjust `n_clusters` parameter if needed

**Missing ground truth**
→ Phase 8 can run without Phase 5; uses cluster heuristic

## Next Steps

1. ✅ Review visualizations
2. ✅ Check confidence scores
3. ✅ Identify low-confidence cases
4. ✅ Begin expert validation workflow
5. ✅ Update report template with actual values
6. ✅ Update PROJECT_PROGRESS_SUMMARY.md

## Documentation

- Full guide: `docs/PHASE8_README.md`
- Report template: `reports/phase8_administrative_function_report.md`
- Implementation: `PHASE8_IMPLEMENTATION_SUMMARY.md`

---

**Ready to start?** Run: `python scripts/analyze_administrative_function.py`
