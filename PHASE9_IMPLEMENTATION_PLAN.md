# Phase 9 Implementation Plan: Systemic, Cognitive, and Robustness Analysis

**Version:** 1.0  
**Date:** January 1, 2026  
**Status:** Planning

---

## Overview

Phase 9 characterizes **khipus as engineered information systems** by analyzing their efficiency, robustness, variability, and cognitive affordances — **without semantic or cultural interpretation**.

**Core Principle:** Treat khipus as designed artifacts with measurable properties common to all information systems.

---

## Dependencies from Previous Phases

### Required Data Files
- **Phase 1-2:** `cord_numeric_values.csv`, `cord_hierarchy.csv`, `color_data.csv`
- **Phase 3:** `summation_test_results.csv`
- **Phase 4:** `graph_structural_features.csv`, `cluster_assignments_kmeans.csv`
- **Phase 7:** `anomaly_detection_results.csv`
- **Phase 8:** `administrative_typology.csv`, `structural_cluster_assignments.csv`

### Required Metrics/Functions
- Graph traversal algorithms
- Shannon entropy calculations
- Statistical distance measures
- Clustering quality metrics
- Random graph generation

---

## Implementation Order & Priority

### **Tier 1: Foundation (Weeks 1-2)**
- 9.1 Information Capacity & Efficiency
- 9.5 Variance Mapping
- 9.8 Randomness & Intentional Design Testing

### **Tier 2: Core Analysis (Weeks 3-4)**
- 9.2 Error Detection & Robustness
- 9.4 Structural Minimalism vs. Expressiveness
- 9.6 Boundary Phenomena Analysis

### **Tier 3: Advanced (Weeks 5-6)**
- 9.3 Cognitive Load & Usability
- 9.7 Anomaly Taxonomy
- 9.9 Stability & Stress Testing

### **Tier 4: Synthesis (Week 7)**
- 9.10 Negative Knowledge Mapping
- Final report compilation

---

## 9.1 Information Capacity & Efficiency

### Objective
Quantify how much information khipus encode and how efficiently.

### Implementation Steps

1. **Information-Theoretic Metrics**
   - Compute Shannon entropy for:
     - Numeric value distributions
     - Color distributions
     - Structural patterns (branching, depth)
   - Calculate entropy per cord, per knot, per level
   
2. **Compression Efficiency**
   - Compare information density vs. structural depth
   - Calculate redundancy ratios: `actual_bits / minimum_bits`
   - Measure compressibility using Lempel-Ziv complexity

3. **Capacity Bounds**
   - Lower bound: Minimum bits to represent observed data
   - Upper bound: Maximum distinguishable states
   - Utilization rate: `actual_usage / theoretical_capacity`

4. **Archetype Comparison**
   - Compare efficiency across Phase 8 administrative types
   - Identify over-engineered vs. under-engineered designs

### Algorithms
```python
# Shannon entropy
H = -sum(p_i * log2(p_i))

# Redundancy ratio
redundancy = 1 - (H_actual / H_theoretical)

# Compression ratio
compression = len(compressed(data)) / len(data)
```

### Outputs
- `information_capacity.csv`: Per-khipu metrics
- `efficiency_by_type.csv`: Aggregate by administrative type
- `capacity_bounds.json`: System-wide estimates
- Visualization: Efficiency curves by archetype

### Success Criteria
- Capacity bounds differ by >2x across types
- Clear efficiency clusters emerge
- Results position khipus relative to other pre-modern systems

---

## 9.2 Error Detection, Correction, and Robustness

### Objective
Measure resilience to error, damage, or mis-recording.

### Implementation Steps

1. **Perturbation Analysis**
   - Single-knot error injection (change value by ±1, ±10%)
   - Measure impact on summation checks
   - Calculate error propagation distance

2. **Self-Correction via Summation**
   - Identify khipus with summation hierarchies (Phase 3)
   - Test if errors are detectable at aggregation points
   - Measure correction coverage: `detectable_errors / total_errors`

3. **Boundary Cord Analysis**
   - Test if boundary cords (white cords) localize errors
   - Measure error containment: `errors_within_segment / total_errors`

4. **Redundancy vs. Brittleness**
   - Compare redundancy levels (from 9.1) with error tolerance
   - Identify trade-offs: high redundancy but low robustness?

5. **Robustness Scoring**
   ```python
   robustness_score = (
       0.4 * error_detection_rate +
       0.3 * error_localization_rate +
       0.2 * redundancy_factor +
       0.1 * self_correction_capability
   )
   ```

### Simulations
- Monte Carlo: 1,000 random perturbations per khipu
- Systematic: Perturb each cord once
- Cascading: Test multi-error scenarios

### Outputs
- `robustness_metrics.csv`: Per-khipu scores
- `error_sensitivity.csv`: Perturbation results
- `failure_modes.json`: Taxonomy of failure types
- Visualization: Robustness heatmap by type

### Success Criteria
- Clear robustness differences across administrative types
- Summation khipus show higher error detection (>70%)
- Boundary cords demonstrate error localization

---

## 9.3 Cognitive Load & Usability Modeling

### Objective
Estimate cognitive demands imposed on khipu users.

### Implementation Steps

1. **Working Memory Load**
   - Estimate based on hierarchy depth and branching
   - Model: `WM_load = depth * log(branching_factor) + cord_count / 10`
   - Account for chunking: groups of 10, color boundaries

2. **Hierarchy Traversal Cost**
   - Calculate path length from root to leaves
   - Measure backtracking requirements
   - Count "context switches" (level changes)

3. **Visual Parsing Complexity**
   - Color entropy as visual load proxy
   - Transition count as attention-switching cost
   - Measure time-to-locate (simulated): random target search

4. **Sequential vs. Parallel Interpretability**
   - Identify parallel-readable sections (same level)
   - Measure serial dependencies (parent-child chains)
   - Calculate parallelization potential

5. **Specialist Operation Evidence**
   - Identify khipus requiring expert knowledge (high complexity)
   - Threshold: `cognitive_load > 90th percentile`

### Cognitive Models
```python
# Miller's Law: 7±2 items in working memory
chunks_required = cord_count / 7

# Traversal cost
traversal_cost = sum(depth_at_node for node in tree)

# Visual load
visual_load = color_entropy + transition_frequency
```

### Outputs
- `cognitive_complexity.csv`: Per-khipu indices
- `usability_scores.csv`: By administrative type
- `specialist_indicators.json`: High-complexity khipus
- Visualization: Complexity vs. administrative function

### Success Criteria
- Cognitive load correlates with khipu size (r > 0.6)
- Aggregated summaries show higher load than compact records
- >20% of khipus exceed typical working memory capacity

---

## 9.4 Structural Minimalism vs. Expressiveness

### Objective
Determine if khipus are minimally sufficient or intentionally redundant.

### Implementation Steps

1. **Marginal Expressiveness**
   - For each structural layer (depth level), calculate information gain
   - Measure: `ΔH = H(depth=n) - H(depth=n-1)`
   - Identify diminishing returns point

2. **Complexity vs. Scale**
   - Plot structural complexity vs. administrative scale
   - Test if complexity grows linearly, logarithmically, or exponentially
   - Identify over-engineered outliers

3. **Template Matching**
   - Cluster khipus by structural similarity (Phase 4 refined)
   - Identify standard templates (>10 instances)
   - Measure template adherence: `similarity_to_template > 0.8`

4. **Design Efficiency Curves**
   - For each administrative type, plot:
     - X-axis: Structural complexity
     - Y-axis: Information capacity
   - Identify Pareto-optimal designs

### Metrics
```python
# Marginal expressiveness
marginal_info = entropy(level_n) - entropy(level_n-1)

# Template distance
template_dist = edit_distance(khipu_structure, template_structure)

# Efficiency ratio
efficiency = information_capacity / structural_complexity
```

### Outputs
- `expressiveness_curves.csv`: Marginal info gain per level
- `template_analysis.csv`: Standard vs. bespoke khipus
- `efficiency_frontier.json`: Pareto-optimal designs
- Visualization: Diminishing returns curves

### Success Criteria
- Diminishing returns evident after depth=3
- >60% of khipus match identified templates
- Clear efficiency frontier separates optimized from complex designs

---

## 9.5 Variance Mapping: Constraint vs. Flexibility

### Objective
Identify which dimensions are standardized vs. flexible.

### Implementation Steps

1. **Variance by Dimension**
   - Calculate coefficient of variation (CV) for:
     - Structural: depth, branching, cord count
     - Chromatic: color count, entropy, transitions
     - Numeric: value distributions, summation rates
   - Compare CV across dimensions: `CV = σ / μ`

2. **Regional Comparison**
   - Group by region (if available from database)
   - Test variance differences: `ANOVA(variance ~ region)`
   - Identify standardized (low CV) vs. variable (high CV) features

3. **Archetype-Specific Variance**
   - Compare variance within vs. between Phase 8 types
   - High within-type variance → local autonomy
   - Low within-type variance → imperial standardization

4. **Constraint Maps**
   - Visualize as heatmap:
     - Rows: Features
     - Columns: Administrative types or regions
     - Color: Coefficient of variation

### Statistical Tests
```python
# Variance ratio test
F_statistic = var(group_A) / var(group_B)

# Levene's test for homogeneity of variance
levene_test(group_A, group_B, group_C)

# Coefficient of variation
CV = std(data) / mean(data)
```

### Outputs
- `variance_by_dimension.csv`: CV for all features
- `constraint_map.csv`: Standardization scores
- `flexibility_zones.json`: High-variance features
- Visualization: Constraint heatmap

### Success Criteria
- Structural features show lower variance (CV < 0.3) than chromatic (CV > 0.5)
- Clear regional patterns if geographic data available
- Evidence of empire-wide structural standards

---

## 9.6 Boundary Phenomena Analysis

### Objective
Characterize what happens at structural boundaries within khipus.

### Implementation Steps

1. **Boundary Identification**
   - Extract boundaries from `cord_hierarchy.csv`:
     - White cords (color-based boundaries)
     - Level transitions (structural boundaries)
     - Summation points (functional boundaries)

2. **Knot Density at Boundaries**
   - Calculate knot density: `knots_per_cord`
   - Compare boundary vs. non-boundary cords
   - Test: `t_test(density_boundary, density_interior)`

3. **Color Transitions**
   - Count color changes at boundaries vs. interior
   - Measure alignment: `boundary_transitions / total_transitions`

4. **Numeric Discontinuities**
   - Test for value resets at boundaries: `value_after - value_before`
   - Identify numeric "chapters" separated by boundaries

5. **Boundary Function Typology**
   - Classify boundaries by properties:
     - Separating (high discontinuity)
     - Summarizing (summation present)
     - Marking (color change only)
     - Structural (depth change)

### Analyses
```python
# Boundary score
boundary_score = (
    0.3 * has_color_change +
    0.3 * has_summation +
    0.2 * has_depth_change +
    0.2 * numeric_discontinuity
)

# Discontinuity measure
discontinuity = abs(mean(after_boundary) - mean(before_boundary))
```

### Outputs
- `boundary_catalog.csv`: All identified boundaries
- `boundary_functions.json`: Typology classification
- `discontinuity_scores.csv`: Numeric resets
- Visualization: Boundary types by administrative function

### Success Criteria
- Boundaries show 2x higher knot density than interior
- >70% of white cords align with functional boundaries
- Clear boundary typology emerges (3-5 types)

---

## 9.7 Anomaly Taxonomy (Pre-Interpretive)

### Objective
Classify anomalies by deviation type without causal explanation.

### Implementation Steps

1. **Anomaly Subclustering**
   - Load Phase 7 anomalies: `anomaly_detection_results.csv`
   - Apply hierarchical clustering on anomaly features
   - Use silhouette score to determine optimal subclusters

2. **Deviation Type Classification**
   - **Numeric:** Extreme values, impossible summations, missing values
   - **Structural:** Unusual depth, extreme branching, graph cycles
   - **Chromatic:** Rare colors, extreme entropy, missing color data
   - **Hybrid:** Multiple deviation types

3. **Coherence Testing**
   - Test if anomalies form coherent subgroups
   - Measure intra-cluster similarity vs. inter-cluster distance
   - Identify transitional artifacts (between normal and anomalous)

4. **Cross-Reference with Robustness**
   - Correlate anomaly types with robustness scores (9.2)
   - Test if anomalies are fragile or robust designs

### Clustering
```python
# Anomaly feature vector
anomaly_features = [
    numeric_zscore,
    structural_outlier_score,
    chromatic_entropy,
    robustness_score
]

# Hierarchical clustering
linkage_matrix = hierarchical_cluster(anomaly_features)
clusters = cut_tree(linkage_matrix, n_clusters=k)
```

### Outputs
- `anomaly_subclusters.csv`: Refined anomaly classification
- `deviation_taxonomy.json`: Types and characteristics
- `transitional_artifacts.csv`: Boundary cases
- Visualization: Anomaly dendrogram

### Success Criteria
- 3-5 distinct anomaly classes identified
- Subclusters show higher internal coherence than Phase 7
- No causal or historical explanations offered (guardrail maintained)

---

## 9.8 Randomness & Intentional Design Testing

### Objective
Demonstrate khipus are far from random construction.

### Implementation Steps

1. **Synthetic Khipu Generation**
   - Generate 1,000 random khipus per model:
     - **Random Structure:** Uniform random depth, branching
     - **Random Color:** Random color assignment per cord
     - **Random Numeric:** Random values from empirical distribution
   - Match size distribution to real khipus

2. **Statistical Distance Measures**
   - Calculate for each random set vs. real khipus:
     - Kolmogorov-Smirnov distance (numeric distributions)
     - Jensen-Shannon divergence (color distributions)
     - Graph edit distance (structural)
     - Multi-dimensional Mahalanobis distance

3. **Forbidden Design Regions**
   - Identify combinations never observed:
     - High depth + low branching?
     - High color entropy + no summation?
   - Map forbidden vs. allowed parameter space

4. **Intentionality Scoring**
   ```python
   intentionality = (
       0.4 * distance_from_random +
       0.3 * forbidden_region_avoidance +
       0.3 * pattern_consistency
   )
   ```

### Null Models
- **Erdős-Rényi:** Random graphs with same node count
- **Preferential Attachment:** Scale-free structure
- **Uniform Random:** All parameters uniform random
- **Constrained Random:** Random within observed ranges

### Outputs
- `synthetic_khipus/`: Generated random khipus (1,000 per model)
- `statistical_distances.csv`: Distance measures
- `forbidden_regions.json`: Impossible design combinations
- `intentionality_scores.csv`: Per-khipu design scores
- Visualization: Real vs. random in PCA space

### Success Criteria
- Real khipus are >5σ from random in all models
- Clear forbidden regions identified (>20% of parameter space)
- Strong evidence of design constraints

---

## 9.9 Stability & Stress Testing

### Objective
Test robustness of discovered patterns under data loss.

### Implementation Steps

1. **Feature Ablation**
   - Remove features one at a time:
     - Numeric values only
     - Color information only
     - Structural depth only
   - Re-run Phase 4 clustering and Phase 8 classification
   - Measure agreement: `adjusted_rand_index(original, ablated)`

2. **Data Masking**
   - Randomly mask 10%, 20%, 30% of:
     - Cords
     - Knot values
     - Color records
   - Re-run all analyses
   - Track classification drift

3. **Re-clustering Stability**
   - Repeat Phase 4 clustering with:
     - Different random seeds (100 runs)
     - Different k values (k=5,6,7,8,9)
     - Different algorithms (k-means, DBSCAN, hierarchical)
   - Measure consensus: `normalized_mutual_information`

4. **Cross-Validation**
   - Split dataset 80/20 train/test
   - Train classifiers on subset, test on holdout
   - Repeat 100 times, measure variance

### Metrics
```python
# Stability score
stability = mean(adjusted_rand_index(run_i, run_j) for all pairs)

# Classification drift
drift = hamming_distance(original_labels, masked_labels) / n

# Consensus strength
consensus = mean(nmi(clustering_i, clustering_j) for all pairs)
```

### Outputs
- `feature_ablation_results.csv`: Impact per feature
- `data_masking_results.csv`: Drift by masking level
- `clustering_stability.json`: Consensus measures
- `confidence_bounds.csv`: Per-analysis uncertainty
- Visualization: Stability curves

### Success Criteria
- Classification stable under 10% masking (drift < 5%)
- Feature ablation: no single feature causes >20% drift
- Clustering consensus: NMI > 0.85 across runs

---

## 9.10 Negative Knowledge Mapping

### Objective
Document what can be confidently ruled out.

### Implementation Steps

1. **Tested Hypotheses That Failed**
   - Review all Phase 1-8 analyses
   - Document hypotheses that failed empirically:
     - No phonetic encoding evidence
     - No symbolic color lexicon
     - No free-form narrative structure (in most cases)

2. **Absence Testing**
   - Test for features NOT present:
     - No alphabetic structure (N-gram analysis)
     - No linguistic patterns (Zipf's law violations)
     - No semantic color coding (color-meaning consistency tests)

3. **Boundary Conditions**
   - Define confidence levels:
     - **Confidently ruled out:** p < 0.001 against
     - **Unlikely:** p < 0.05 against
     - **Cannot determine:** insufficient evidence

4. **Documentation Standards**
   - For each negative finding:
     - State null hypothesis clearly
     - Show statistical test results
     - Specify confidence level
     - Note alternative explanations

### Statistical Tests
```python
# Test for linguistic structure (Zipf's law)
zipf_test = correlation(log(rank), log(frequency))
# If r < 0.5, not linguistic

# Test for symbolic consistency
color_meaning_consistency = chi_square(color, numeric_pattern)
# If p > 0.05, no consistent symbolic meaning

# Test for phonetic encoding
ngram_entropy = entropy(bigrams) / entropy(unigrams)
# If ratio < 1.2, no phonetic structure
```

### Outputs
- `negative_findings.json`: Ruled-out hypotheses
- `boundary_conditions.csv`: Confidence levels
- `cannot_determine.json`: Insufficient evidence cases
- Report section: "What Khipus Are NOT"

### Success Criteria
- >10 confidently ruled-out hypotheses documented
- Clear statistical evidence for each (p-values reported)
- No unsupported negative claims

---

## Module Structure

### Core Modules

```
src/analysis/
├── phase9/
│   ├── __init__.py
│   ├── information_capacity.py      # 9.1
│   ├── robustness_analysis.py       # 9.2
│   ├── cognitive_load.py            # 9.3
│   ├── minimalism_analysis.py       # 9.4
│   ├── variance_mapping.py          # 9.5
│   ├── boundary_analysis.py         # 9.6
│   ├── anomaly_taxonomy.py          # 9.7
│   ├── randomness_testing.py        # 9.8
│   ├── stability_testing.py         # 9.9
│   └── negative_knowledge.py        # 9.10
```

### Script Structure

```
scripts/
├── run_phase9_full.py               # Main pipeline
├── analyze_information_capacity.py  # 9.1
├── test_robustness.py               # 9.2
├── model_cognitive_load.py          # 9.3
├── analyze_minimalism.py            # 9.4
├── map_variance.py                  # 9.5
├── analyze_boundaries.py            # 9.6
├── refine_anomalies.py              # 9.7
├── test_randomness.py               # 9.8
├── test_stability.py                # 9.9
└── document_negative_knowledge.py   # 9.10
```

### Output Structure

```
data/processed/phase9/
├── 9.1_information_capacity/
│   ├── capacity_metrics.csv
│   ├── efficiency_by_type.csv
│   └── capacity_bounds.json
├── 9.2_robustness/
│   ├── robustness_metrics.csv
│   ├── error_sensitivity.csv
│   └── failure_modes.json
├── 9.3_cognitive_load/
│   ├── cognitive_complexity.csv
│   └── usability_scores.csv
├── 9.4_minimalism/
│   ├── expressiveness_curves.csv
│   └── template_analysis.csv
├── 9.5_variance/
│   ├── variance_by_dimension.csv
│   └── constraint_map.csv
├── 9.6_boundaries/
│   ├── boundary_catalog.csv
│   └── boundary_functions.json
├── 9.7_anomalies/
│   ├── anomaly_subclusters.csv
│   └── deviation_taxonomy.json
├── 9.8_randomness/
│   ├── synthetic_khipus/
│   ├── statistical_distances.csv
│   └── forbidden_regions.json
├── 9.9_stability/
│   ├── ablation_results.csv
│   └── clustering_stability.json
├── 9.10_negative_knowledge/
│   ├── negative_findings.json
│   └── boundary_conditions.csv
└── phase9_metadata.json
```

---

## Visualization Suite

### Core Visualizations

1. **Information Capacity Dashboard**
   - Entropy distributions by type
   - Efficiency curves
   - Capacity utilization heatmap

2. **Robustness Heatmap**
   - Khipu type × robustness metric
   - Error propagation networks
   - Failure mode taxonomy tree

3. **Cognitive Load Profiles**
   - Complexity vs. administrative function
   - Working memory load distributions
   - Traversal cost networks

4. **Variance Constraint Map**
   - Feature × type heatmap
   - Standardization vs. flexibility zones
   - Regional variation (if available)

5. **Boundary Analysis**
   - Boundary type distributions
   - Discontinuity profiles
   - Function typology tree

6. **Randomness Testing**
   - Real vs. synthetic PCA projection
   - Distance distributions
   - Forbidden region visualization

7. **Stability Dashboard**
   - Feature ablation impact
   - Data masking drift curves
   - Clustering consensus matrix

---

## Quality Assurance

### Validation Checks

1. **Sanity Checks**
   - Entropy values: 0 ≤ H ≤ log₂(n_states)
   - Probabilities sum to 1.0
   - Distance metrics: 0 ≤ d ≤ d_max

2. **Consistency Checks**
   - Cross-phase agreement (Phase 4 ↔ Phase 8 ↔ Phase 9)
   - Metric correlations make sense (e.g., size vs. complexity)

3. **Reproducibility**
   - Fixed random seeds for all simulations
   - Version-controlled synthetic data generation
   - Documented parameter choices

4. **Edge Cases**
   - Handle khipus with missing data gracefully
   - Document outliers explicitly
   - Test algorithms on minimal/maximal khipus

---

## Timeline

### Week 1-2: Foundation (Tier 1)
- Implement 9.1 (Information Capacity)
- Implement 9.5 (Variance Mapping)
- Implement 9.8 (Randomness Testing)
- **Milestone:** Baseline system characterization complete

### Week 3-4: Core Analysis (Tier 2)
- Implement 9.2 (Robustness)
- Implement 9.4 (Minimalism)
- Implement 9.6 (Boundaries)
- **Milestone:** Core system properties quantified

### Week 5-6: Advanced (Tier 3)
- Implement 9.3 (Cognitive Load)
- Implement 9.7 (Anomaly Taxonomy)
- Implement 9.9 (Stability Testing)
- **Milestone:** Advanced analyses complete

### Week 7: Synthesis
- Implement 9.10 (Negative Knowledge)
- Generate all visualizations
- Compile Phase 9 report
- Cross-validation with Phase 8
- **Milestone:** Phase 9 complete

---

## Success Criteria

### Minimum Viable Phase 9
- ✅ Information capacity bounds established
- ✅ Robustness metrics computed for all khipus
- ✅ Variance maps show standardization patterns
- ✅ Randomness testing shows intentional design (>5σ)
- ✅ At least 5 negative findings documented

### Full Phase 9 Success
- All 10 sub-analyses complete
- >30 visualizations generated
- Comprehensive Phase 9 report (100+ pages)
- Stability testing confirms all prior phases
- Reusable metrics library for future research

### Publication-Ready
- Standalone paper: "Khipus as Information Systems"
- Standalone paper: "Robustness and Error Tolerance in Khipus"
- All analyses peer-review ready
- Open-source code repository published

---

## Risk Mitigation

### Technical Risks
- **Risk:** Synthetic khipus don't capture structural constraints
  - **Mitigation:** Use multiple null models, validate against real data
  
- **Risk:** Cognitive load models are speculative
  - **Mitigation:** Ground in established cognitive science (Miller's Law, etc.)

- **Risk:** Stability testing reveals fragile patterns
  - **Mitigation:** Document honestly, adjust confidence claims

### Interpretive Risks
- **Risk:** Analyses drift toward semantic claims
  - **Mitigation:** Explicit guardrails in code comments and report
  
- **Risk:** Negative knowledge overstated
  - **Mitigation:** Require p < 0.001 for "confidently ruled out"

---

## Next Steps

1. **Review and Approve** this implementation plan
2. **Prioritize** which sub-analyses to implement first (recommend Tier 1)
3. **Set up** Phase 9 module structure (`src/analysis/phase9/`)
4. **Begin** with 9.1 (Information Capacity) as foundation
5. **Iterate** through tiers systematically

---

**Document Status:** Ready for implementation  
**Estimated Effort:** 7 weeks full-time, or 14 weeks part-time  
**Blocking Dependencies:** None (all Phase 0-8 outputs available)
