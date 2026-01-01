# Phase 9 Data Schema Reference

**Generated:** January 1, 2026  
**Purpose:** Canonical reference for column names across all Phase 9 input files

---

## Critical Note: Case Sensitivity

⚠️ **UPPERCASE vs lowercase inconsistency exists between files:**
- `hierarchy` file uses **UPPERCASE** (KHIPU_ID, CORD_ID, CORD_LEVEL, etc.)
- All other files use **lowercase** (khipu_id, cord_id, etc.)

Always use correct case when merging/joining dataframes!

---

## Input Files

### graph_structural_features.csv (Phase 4)
```python
columns = [
    'khipu_id',           # lowercase
    'num_nodes',
    'num_edges',
    'avg_degree',
    'max_degree',
    'density',
    'depth',
    'width',
    'avg_branching',
    'num_roots',
    'num_leaves',
    'has_numeric',
    'has_color',
    'avg_numeric_value',
    'std_numeric_value'
]
```

### summation_test_results.csv (Phase 3)
```python
columns = [
    'khipu_id',           # lowercase
    'has_pendant_summation',
    'pendant_match_rate',
    'num_pendant_groups',
    'has_white_boundaries',
    'num_white_boundaries'
]
```

### color_data.csv (Phase 2)
```python
columns = [
    'color_id',
    'khipu_id',           # lowercase
    'cord_id',            # lowercase
    'color_cd_1',
    'operator_1',
    'color_cd_2',
    'operator_2',
    'color_cd_3',
    'full_color',
    'color_range',
    'range_beg',
    'range_end',
    'pcord_flag',
    'description',
    'red',
    'green',
    'blue',
    'color_category'
]
```

### cord_numeric_values.csv (Phase 1)
```python
columns = [
    'khipu_id',           # lowercase
    'cord_id',            # lowercase
    'numeric_value',
    'confidence',
    'num_clusters',
    'validation_notes'
]
```

### cord_hierarchy.csv (Phase 2)
⚠️ **UPPERCASE columns!**
```python
columns = [
    'CORD_ID',            # UPPERCASE!
    'KHIPU_ID',           # UPPERCASE!
    'CORD_CLASSIFICATION',
    'CORD_LEVEL',
    'PENDANT_FROM',
    'ATTACHED_TO',
    'CORD_ORDINAL',
    'CORD_LENGTH',
    'TWIST',
    'FIBER',
    'num_knots',          # lowercase (inconsistent)
    'num_valued_knots',   # lowercase
    'has_numeric_value',  # lowercase
    'has_missing_attachment',
    'has_missing_ordinal',
    'confidence'
]
```

### administrative_typology.csv (Phase 8)
```python
columns = [
    'khipu_id',           # lowercase
    'structural_cluster',
    'predicted_function',
    'confidence',
    'administrative_type',
    'cord_count',
    'hierarchy_depth',
    'summation_match_rate',
    'numeric_coverage',
    'unique_color_count',
    'color_entropy'
]
```

### anomaly_detection_results.csv (Phase 7)
```python
columns = [
    'khipu_id',           # lowercase
    'PROVENANCE',         # UPPERCASE!
    'cluster',
    'anomaly_score',
    'is_anomaly_isolation',
    'is_anomaly_statistical',
    'num_outlier_flags',
    'is_anomaly_topology',
    'num_topology_flags',
    'num_methods_flagged',
    'high_confidence_anomaly'
]
```

---

## Common Join Operations

### Merging with cord_hierarchy.csv
```python
# WRONG (will fail):
df.merge(hierarchy, on='khipu_id')

# CORRECT:
df.merge(hierarchy, left_on='khipu_id', right_on='KHIPU_ID')
df.merge(hierarchy, left_on='cord_id', right_on='CORD_ID')
```

### Merging color with hierarchy
```python
# color_data has: khipu_id (lowercase), cord_id (lowercase)
# hierarchy has: KHIPU_ID (UPPERCASE), CORD_ID (UPPERCASE)

merged = color_data.merge(
    hierarchy,
    left_on=['khipu_id', 'cord_id'],
    right_on=['KHIPU_ID', 'CORD_ID'],
    how='left'
)
```

---

## Knot Data Access

⚠️ **Important:** There is no processed knot-level CSV file. Knot data must be:
1. Extracted from database: `data/khipu.db` → `knot` table
2. Or computed from `cord_hierarchy.csv` columns:
   - `num_knots`: Total knots on cord
   - `num_valued_knots`: Knots contributing to numeric value
   - `has_numeric_value`: Boolean flag

---

## Quick Reference Card

| File | khipu_id | cord_id | Notes |
|------|----------|---------|-------|
| structural_features | lowercase | N/A | ✅ Standard |
| summation_results | lowercase | N/A | ✅ Standard |
| color_data | lowercase | lowercase | ✅ Standard |
| numeric_values | lowercase | lowercase | ✅ Standard |
| **cord_hierarchy** | **UPPERCASE** | **UPPERCASE** | ⚠️ Different! |
| typology | lowercase | N/A | ✅ Standard |
| anomalies | lowercase | N/A | PROVENANCE is UPPERCASE |

---

## Validation Checklist

Before running any Phase 9 analysis:
- [ ] Verify column names with `df.columns.tolist()`
- [ ] Use `left_on/right_on` when merging with hierarchy
- [ ] Check for `KeyError` in all merge operations
- [ ] Test with 5 sample khipus before full run

---

**Last Updated:** January 1, 2026  
**Status:** Canonical reference for Phase 9 implementation
