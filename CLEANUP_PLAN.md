# Cleanup Plan for Khipu Toolkit
# Before Phase 1 Implementation

## Files to DELETE (Temporary/Debug)

### Temporary test files created during debugging
- scripts/check_colors.py
- scripts/check_hierarchy_cols.py
- scripts/check_khipu_colors.py
- scripts/check_knot_fields.py
- scripts/check_knot_type_docs.py
- scripts/check_schema.py
- scripts/explore_db_schema.py
- scripts/explore_knot_data.py
- scripts/debug_3d.py
- scripts/test_knot_rendering.py
- scripts/test_knot_spacing.py
- scripts/test_summation_single.py
- scripts/investigate_knot_values.py
- scripts/verify_ascher_notation.py
- scripts/temp_fix.txt

### Failed/incomplete summation analysis (to be replaced by Phase 10)
- scripts/analyze_summation_patterns.py (buggy, wrong schema)
- scripts/test_alternative_summation.py
- scripts/test_hierarchical_summation.py
- scripts/test_summation_hypotheses.py
- scripts/generate_summation_report.py

### PDF extraction tools (not needed for core analysis)
- scripts/extract_pdf_text.py
- scripts/search_pdf_methodology.py

## Files to DEPRECATE (Keep but mark as old)

### Old 3D viewers (replaced by plotly_3d_viewer_new.py)
- scripts/visualize_3d_khipu.py → DEPRECATE
- scripts/interactive_3d_viewer.py → DEPRECATE

Add deprecation notice at top of each:
```python
# DEPRECATED: This viewer has been replaced by plotly_3d_viewer_new.py
# This file is kept for historical reference only.
# Please use: streamlit run scripts/plotly_3d_viewer_new.py
```

## Files to RENAME for Clarity

### Make the new 3D viewer the primary
- plotly_3d_viewer_new.py → khipu_3d_viewer.py

## Files to KEEP (Core functionality)

### Analysis modules (will be enhanced in Phase 10)
- analyze_administrative_function.py
- analyze_geographic_correlations.py
- analyze_geography.py
- analyze_high_match_khipus.py
- analyze_information_capacity.py
- analyze_robustness.py
- analyze_variance.py
- classify_khipu_function.py
- cluster_khipus.py
- compute_graph_similarities.py
- detect_anomalies.py
- mine_motifs.py
- predict_missing_values.py

### Extraction (Phase 2 - keep as-is)
- extract_color_data.py
- extract_cord_hierarchy.py
- extract_knot_data.py
- extract_templates.py
- generate_processed_data.py

### Graph building (Phase 2 - keep as-is)
- build_khipu_graphs.py

### Visualizations (Phase 6-8 - keep as-is)
- visualize_clusters.py
- visualize_geographic_heatmap.py
- visualize_geographic_motifs.py
- visualize_ml_results.py
- visualize_phase1_baseline.py
- visualize_phase2_extraction.py
- visualize_phase3_summation.py
- visualize_phase5_hypotheses.py
- visualize_phase8_results.py
- visualize_phase9_meta.py

### Color hypotheses (Phase 5 - keep as-is)
- test_color_hypotheses.py

### Dashboard (keep but may enhance later)
- dashboard_app.py

## Cleanup Actions

Execute these commands:
