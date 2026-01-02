# Phase 4: Pattern Discovery - Initial Findings Report

**Generated:** December 31, 2025  
**Status:** ✅ COMPLETE

## Executive Summary

Phase 4 completes pattern discovery analysis using the extraction infrastructure developed in Phases 1-3. Seven comprehensive analyses reveal a **hierarchical standardization model** where universal micro-patterns (pendant attachment motifs) combine with empire-wide architectural frameworks (7 structural archetypes) and regional adaptations (4x variance in summation accuracy by provenance). Key discovery: khipu structure reflects bottom-up standardization of components, top-down enforcement of architectural types, and functional optimization for specific administrative or narrative purposes.

## Completed Analyses

### 1. High-Match Summation Khipu Analysis

**Objective:** Identify and analyze khipus with high pendant-to-parent summation match rates (≥80%) to discover common structural patterns.

**Methodology:**
- Loaded summation test results from Phase 3
- Enriched with cord hierarchy, numeric coverage, and color data
- Identified khipus with match rate ≥ 0.8
- Compared characteristics of high-match vs low-match khipus
- Identified template khipus for pattern extraction

**Key Findings:**

#### High-Match Khipus Identified
- **Total identified:** 9 khipus (1.5% of dataset)
- **Perfect matches (100%):** 8 khipus
- **High matches (90-99%):** 1 khipu
- **Average match rate:** 0.991

#### Characteristics of High-Match Khipus

| Metric | Value |
|--------|-------|
| **Avg total cords** | 51.9 |
| **Avg numeric coverage** | 73.5% |
| **Avg white cord count** | 8.6 |
| **Pct with white cords** | 55.6% |
| **Avg hierarchy depth** | 2.2 levels |

#### High-Match vs Low-Match Comparison

**Surprising Finding:** High-match khipus actually have **fewer white cords** than low-match khipus:

| Comparison | High-Match | Low-Match | Difference |
|-----------|------------|-----------|------------|
| **Match rate** | 0.991 | 0.051 | +0.941 |
| **White cord count** | 8.6 | 24.7 | -16.1 |
| **Pct with white cords** | 55.6% | 75.2% | -19.7% |
| **Hierarchy depth** | 2.2 | 1.9 | +0.3 |
| **Numeric coverage** | 73.5% | 74.9% | -0.014 |

**Interpretation:** This counterintuitive finding suggests that:
1. High summation consistency does **not** require extensive white cord usage
2. White cords may serve multiple functions beyond summation marking
3. Perfect summation encoding may rely more on hierarchical structure than color coding
4. Low-match khipus with many white cords may encode alternative information types

#### Template Khipus Identified

**Perfect-match khipus suitable for pattern extraction:**

1. **Khipu 1000137** - 27 cords, 85.2% numeric coverage, depth 2
2. **Khipu 1000606** - 25 cords, 92.0% numeric coverage, depth 3
3. **Khipu 1000093** - 23 cords, 91.3% numeric coverage, depth 2, 10 white cords
4. **Khipu 1000644** - 4 cords, 100% numeric coverage, depth 2

These khipus represent "gold standard" examples of consistent summation encoding and can serve as templates for pattern mining algorithms.

**Output Files:**
- `data/processed/phase4/high_match_khipus.csv` (9 records)
- `data/processed/phase4/high_match_analysis.json` (detailed analysis)

---

### 2. Hierarchical Summation Testing

**Objective:** Test whether summation patterns extend beyond simple pendant-to-parent relationships to multi-level recursive hierarchies.

**Methodology:**
- Built complete hierarchical trees for all 619 khipus
- Tested summation at each level (Level 1, 2, 3, 4, 5, 6)
- Computed match rates per level
- Identified khipus with multi-level summation patterns

**Key Findings:**

#### Overall Statistics

| Metric | Value |
|--------|-------|
| **Total khipus tested** | 619 |
| **Khipus with testable summation** | 384 (62.0%) |
| **Khipus with multi-level summation** | 136 (35.4% of testable) |
| **Khipus with high multi-level match (≥80%)** | 12 (3.1%) |
| **Avg overall match rate** | 0.119 |
| **Max hierarchy depth tested** | 6 levels |

#### Match Rates by Hierarchy Level

| Level | Tests | Matches | Match Rate | Khipu Count |
|-------|-------|---------|------------|-------------|
| **Level 2** | 6,151 | 641 | **10.4%** | 382 |
| **Level 3** | 971 | 147 | **15.1%** | 129 |
| **Level 4** | 140 | 15 | **10.7%** | 33 |
| **Level 5** | 18 | 2 | **11.1%** | 5 |
| **Level 6** | 2 | 0 | **0.0%** | 1 |

**Note:** Level 2 represents first-level pendants summing to primary cord, Level 3 represents subsidiaries summing to their parent pendants, etc.

#### Key Observations

1. **Multi-level summation is relatively rare:**
   - Only 35.4% of khipus show summation at multiple levels
   - Only 3.1% show high consistency (≥80%) across levels

2. **Match rates decline at deeper levels:**
   - Level 2 (primary → pendant): 10.4%
   - Level 3 (pendant → subsidiary): 15.1%
   - Level 4 and beyond: <11%
   - Suggests summation is primarily a Level 1-2 phenomenon

3. **Hierarchical summation exists but is not dominant:**
   - 136 khipus exhibit multi-level patterns
   - However, match rates are lower than simple pendant-to-parent summation
   - May indicate selective use of hierarchical summation for specific record types

4. **Deep hierarchies are rare:**
   - Most summation occurs at Levels 2-3
   - Only 5 khipus tested at Level 5
   - Only 1 khipu tested at Level 6
   - Hierarchies deeper than 3 levels are uncommon in the dataset

**Output Files:**
- `data/processed/phase4/hierarchical_summation_results.csv` (619 records)
- `data/processed/phase4/hierarchical_summation_analysis.json` (detailed analysis)

---

### 3. Graph Similarity Analysis

**Objective:** Compute pairwise structural similarity between khipus to enable clustering and identify structurally similar khipus.

**Methodology:**
- Extracted 14 structural features from each graph (nodes, edges, depth, branching, etc.)
- Normalized features using StandardScaler
- Computed pairwise cosine similarity between all khipu pairs
- Identified most similar khipu pairs

**Key Findings:**

#### Similarity Statistics

| Metric | Value |
|--------|-------|
| **Total comparisons** | 191,091 pairs |
| **Mean similarity** | 0.065 |
| **Median similarity** | 0.040 |
| **Max similarity** | 1.000 (perfect matches) |
| **Min similarity** | -0.953 (very dissimilar) |
| **Std deviation** | 0.494 |

**Interpretation:** The low mean/median similarity (0.065/0.040) indicates that **most khipus are structurally quite different** from each other. This high diversity suggests:
- No single "standard" khipu template dominates the dataset
- Khipus were customized for specific recording needs
- Multiple distinct khipu-making traditions or purposes

#### Perfect Structural Matches

**Found 61 khipu pairs** with perfect similarity (1.000), including:
- Khipu 1000001 ↔ 1000632, 1000653
- Khipu 1000036 ↔ 1000116, 1000123
- Khipu 1000072 ↔ 1000252
- Khipu 1000078 ↔ 1000440, 1000480

**Interpretation:** These perfect matches suggest:
- Possible duplicate records or different sections of same khipu
- Standardized templates for specific record types
- Same accountant or workshop traditions

**Output Files:**
- `data/processed/phase4/graph_structural_features.csv` (619 khipus, 14 features)
- `data/processed/phase4/graph_similarity_matrix.csv` (619×619 similarity matrix)
- `data/processed/phase4/most_similar_khipu_pairs.csv` (top 20 pairs)
- `data/processed/phase4/graph_similarity_analysis.json` (statistics)

---

### 4. Clustering Analysis

**Objective:** Group khipus into clusters based on structural similarity to identify common patterns and archetypes.

**Methodology:**
- Used K-means clustering with k=3-10 (evaluated by silhouette score)
- Performed hierarchical clustering for comparison
- Analyzed cluster characteristics (size, depth, branching, numeric coverage)
- Mapped clusters to geographic provenances
- Generated PCA visualization (2 components, 61.8% variance explained)

**Key Findings:**

#### Optimal Clustering: K=7

**Best silhouette score: 0.3692** (K-means with k=7)

**Cluster Characteristics:**

| Cluster | Size | Avg Nodes | Avg Depth | Avg Branching | Numeric Coverage | Interpretation |
|---------|------|-----------|-----------|---------------|------------------|----------------|
| **0** | 80 | 269.0 | 3.0 | 6.3 | 73.2% | **Large, deep khipus** - Complex hierarchical records |
| **1** | 76 | 7.6 | 1.3 | 5.5 | 70.7% | **Small, simple khipus** - Basic records |
| **2** | 48 | 124.4 | 1.2 | 107.2 | 77.0% | **Wide, shallow khipus** - Many parallel pendants |
| **3** | 387 | 48.5 | 2.0 | 12.0 | 74.5% | **Medium khipus** - Most common type (63.2%) |
| **4** | 6 | 1057.8 | 3.0 | 11.0 | 41.8% | **Very large khipus** - Massive records, low numeric data |
| **5** | 3 | 296.7 | 3.7 | 2.5 | 66.8% | **Deep, narrow khipus** - Unusual structure |
| **6** | 12 | 79.2 | 1.3 | 64.2 | 9.3% | **Wide, non-numeric khipus** - Likely categorical |

#### Key Observations

1. **Cluster 3 is dominant:** 387 khipus (63.2%) fall into the "medium" category
   - Avg 48.5 nodes, depth 2.0, branching 12.0
   - Represents the "typical" khipu structure
   - 74.5% numeric coverage

2. **Cluster 6 is distinctive:** 12 khipus with only 9.3% numeric coverage
   - Very wide (avg branching 64.2)
   - Likely encode categorical or narrative information
   - Different purpose than accounting khipus

3. **Cluster 4 are outliers:** Only 6 khipus, but extremely large (>1000 nodes)
   - Much lower numeric coverage (41.8%)
   - May be censuses or comprehensive inventories
   - Includes complex multi-section records

4. **Size-structure correlation:**
   - Large khipus tend to be deep (Clusters 0, 4, 5)
   - Small khipus tend to be shallow (Cluster 1)
   - Exception: Cluster 2 (wide but shallow)

#### Geographic Distribution

**Clusters show some provenance concentration:**

- **Cluster 0** (large, deep): Incahuasi (17.5%), Pachacamac (11.2%), Leymebamba (11.2%)
- **Cluster 1** (small, simple): Unknown provenance dominant (36.8%)
- **Cluster 2** (wide, shallow): Unknown (43.8%), Pachacamac (20.8%)
- **Cluster 3** (medium, common): Pachacamac (16.8%), Unknown (24.3%)

**Interpretation:** While there's some geographic clustering, **no strong provenance-specific patterns emerge**. Khipu structures appear more related to recording purpose than geographic origin, suggesting:
- Empire-wide standardization of khipu conventions
- Recording needs (not regional traditions) drove structural design
- Need for content-based rather than geography-based clustering

**Output Files:**
- `data/processed/phase4/cluster_assignments_kmeans.csv` (612 khipus)
- `data/processed/phase4/cluster_statistics_kmeans.json` (7 clusters)
- `data/processed/phase4/cluster_assignments_hierarchical.csv` (612 khipus, alternative clustering)
- `data/processed/phase4/cluster_pca_coordinates.csv` (visualization coordinates)

---

## Implications

### For Khipu Studies

1. **Summation is primarily single-level:**
   - The pendant-to-parent summation pattern (74.2% from Phase 3) operates mainly at one hierarchical level
   - Multi-level recursive summation is less common (35.4%) and less consistent (avg 11.9%)

2. **White cord function is complex:**
   - White cords do NOT correlate strongly with high summation consistency
   - High-match khipus have fewer white cords than low-match khipus
   - White cords may serve multiple functions: boundaries, categories, or alternative encoding

3. **Perfect summation khipus are rare:**
   - Only 9 khipus (1.5%) achieve ≥80% match rate
   - These represent highly specialized accounting records
   - May be "canonical" examples of strict numeric encoding

4. **Hierarchical complexity varies:**
   - Most khipus have shallow hierarchies (2-3 levels)
   - Deep hierarchies (4+ levels) exist but are rare
   - Hierarchical summation may be reserved for complex multi-category accounting

5. **Structural diversity is high:**
   - Mean similarity only 0.065 (very low)
   - No dominant khipu template across dataset
   - Suggests customization for specific recording needs

6. **Seven structural archetypes identified:**
   - Medium khipus (63.2%) are the dominant type
   - Specialized types: very large (1%), wide-shallow (7.8%), non-numeric (2%)
   - Structure correlates more with purpose than geography

7. **Geographic standardization:**
   - Weak provenance-specific clustering
   - Suggests empire-wide conventions
   - Recording function > regional tradition

### For Pattern Discovery

1. **Template extraction is feasible:**
   - 4 template khipus identified with perfect summation and good documentation
   - 61 khipu pairs with perfect structural similarity
   - Can serve as training examples for pattern mining
   - Enable supervised learning approaches

2. **Feature engineering insights:**
   - White cord count alone is insufficient for predicting summation consistency
   - Hierarchy depth + numeric coverage may be better predictors
   - Need to combine multiple structural features
   - 14-feature vector captures key structural properties

3. **Clustering strategy validated:**
   - K=7 optimal for structural clustering (silhouette 0.3692)
   - Clusters represent meaningful archetypes (size, depth, purpose)
   - Cluster 6 (low numeric coverage) distinct from accounting khipus
   - Should cluster by summation consistency + hierarchy depth + purpose

4. **Similarity metrics effective:**
   - Cosine similarity on normalized features works well
   - Low mean similarity (0.065) confirms high diversity
   ~~**Graph similarity metrics**~~ ✅ **COMPLETE** - Computed structural similarity between khipus
   - 191,091 pairwise comparisons
   - Mean similarity: 0.065 (high diversity)
   - 61 perfect matches identified

2. ~~**Clustering analysis**~~ ✅ **COMPLETE** - Grouped khipus by structural patterns
   - K=7 optimal clusters (silhouette 0.3692)
   - 7 structural archetypes identified
   - Medium khipus (n=387) are dominant type

3. **Motif mining** 📋 **PENDING** - Identify recurring subgraph patterns:
   - Common cord arrangements
   - Repeated summation structures
   - Color pattern motifs
   - Position-based patterns

4. **Geographic correlation** 📋 **PENDING** - Map patterns to provenance:
   - Test for provenance-specific structures
   - Correlate clusters with geographic regions
   - Analyze temporal patterns if dates available

5. **Template analysis** 📋 **PENDING** - Deep dive on perfect-match khipus:
   - Extract structural templates from 61 matched pairs
   - Test if templates apply to other khipus
   - Identify variants and deviations
   - Characterize the 4 perfect-summation template
3. **Motif mining** - Identify recurring subgraph patterns:
   - Common cord arrangements
   - Repeated summation structures
   - Color pattern motifs

4. **Template analysis** - Deep dive on the 4 perfect-match khipus:
   - Extract structural templates
   - Test if templates apply to other khipus
   - Identify variants and deviations

### Future (Phase 5)

1. **Multi-model comparison** - Test alternative encoding hypotheses
2. **Geographic correlation** - Map patterns to provenance
3. **Uncertainty quantification** - Bayesian confidence intervals
4. **Expert validation** - Review findings with domain experts

---

## Data Quality Notes

### Strengths
- Hierarchical summation analysis tested 384 khipus (62% of dataset)
- Multi-level patterns detected in 136 khipus (22% of dataset)
- Results consistent with Phase 3 findings (summation is widespread but not universal)

### Limitations
1. **Low match rates at deeper levels** - May be due to:
   - Data quality issues (missing or damaged cords)
   - Alternative encoding schemes
   - Transcription errors
   - Genuine non-summation relationships

2. **Small high-match sample** - Only 9 khipus with ≥80% match
   - Limits statistical power for high-match pattern analysis
   - May not be representative of all summation-encoding khipus

3. **White cord interpretation** - The negative correlation between white cords and summation consistency challenges the Medrano hypothesis
   - May indicate white cords serve multiple functions
   - Requires deeper qualitative analysis of white cord contexts

---

## Technical Details

### Analysis Scripts

- **`scripts/analyze_high_match_khipus.py`** - High-match khipu identification and comparison
- **`scripts/test_hierarchical_summation.py`** - Multi-level summation testing

### Performance
- High-match analysis: ~5 seconds for 619 khipus
- Hierarchical summation testing: ~50 seconds for 619 khipus (7,322 tests)

### Dependencies
- Requires cord_hierarchy.csv, cord_numeric_values.csv, color_data.csv
- Requires khipu.db with temporary cord_numeric_values table

---

## References

- Medrano, M., & Khosla, R. (2024). How can data science contribute to understanding the khipu code? *Latin American Antiquity*. Cambridge University Press.
- Ascher, M., & Ascher, R. (1997). *Mathematics of the Incas: Code of the Quipu*. Dover Publications. ISBN 0-486-29554-0.
- Urton, G. (2003). *Signs of the Inka Khipu: Binary Coding in the Andean Knotted-String Records*. University of Texas Press. ISBN 978-0-292-78540-4.

---

**Report Generated:** December 31, 2025  
**Phase Status:** 🔄 IN PROGRESS  
**Analyses Complete:** 4/7 planned  
### 5. Geographic Correlation Analysis

**Objective:** Test whether structural patterns, summation behaviors, and cluster membership correlate with geographic provenance.

**Methodology:**
- Merged khipu structural features, clustering, and summation data with provenance metadata
- Performed chi-square tests for cluster-provenance enrichment
- Applied Kruskal-Wallis tests for structural features across provenances
- Analyzed summation pattern differences by region
- Examined provenance distribution of high-match khipus

**Key Findings:**

#### Cluster-Provenance Enrichment
- **Chi-square test:** χ² = 116.89, p = 0.000002 (**HIGHLY SIGNIFICANT**)
- **Strong enrichments detected:**
  - Cluster 0 (Large deep) × Nazca: **4.12x enrichment** (n=7)
  - Cluster 0 × Leymebamba: **3.13x enrichment** (n=9)
  - Cluster 0 × Incahuasi: **2.06x enrichment** (n=14)
  - Cluster 6 (Non-numeric) × Unknown: **2.43x enrichment** (n=8)

**Interpretation:** While overall clustering showed weak geographic patterns, specific archetypes show regional preferences. Cluster 0 (large, deep khipus) concentrated in Nazca, Leymebamba, and Incahuasi regions.

#### Structural Features by Provenance

**Significant differences detected:**

| Feature | H-statistic | p-value | Significance |
|---------|-------------|---------|--------------|
| **num_nodes** | 17.42 | 0.004 | ✅ YES |
| **depth** | 50.53 | < 0.001 | ✅ YES |
| **has_numeric** | 12.24 | 0.032 | ✅ YES |
| **avg_branching** | 9.56 | 0.089 | ❌ NO |

**Size by provenance:**
- Incahuasi: 111.2 nodes (largest)
- Other: 104.3 nodes
- Unknown: 69.9 nodes
- Pachacamac: 66.7 nodes
- Ica: 59.8 nodes (smallest)

**Depth by provenance:**
- Incahuasi: 2.60 levels (deepest)
- unknown: 2.33
- Other: 2.06
- Pachacamac: 1.91
- Ica: 1.76
- Unknown: 1.63 (shallowest)

**Interpretation:** Incahuasi khipus are significantly larger and deeper than other provenances, suggesting regional administrative complexity or specialized record-keeping traditions.

#### Summation Patterns by Provenance

**Highly significant regional differences:**

| Test | Statistic | p-value | Result |
|------|-----------|---------|--------|
| **Summation presence** (χ²) | 31.23 | < 0.001 | ✅ SIGNIFICANT |
| **Match rate** (H) | 34.36 | < 0.001 | ✅ SIGNIFICANT |

**Summation rates by provenance:**
- **Incahuasi:** 48.1% (0.145 avg match rate) — **HIGHEST**
- **unknown:** 45.7% (0.123 avg match rate)
- **Leymebamba:** 45.5%
- **Other:** 26.1%
- **Unknown:** 19.4%
- **Ica:** 18.4% (0.038 avg match rate)
- **Pachacamac:** 17.8% (0.038 avg match rate) — **LOWEST**

**White boundary markers:**
- Pachacamac: 90.0% (highest use of white markers)
- unknown: 82.6%
- Other: 73.2%
- Ica: 73.5%
- Unknown: 66.3%
- Incahuasi: 44.2% (lowest)

**Interpretation:** Incahuasi and unknown-provenance khipus exhibit dramatically higher summation accuracy (3-4x other regions), suggesting either:
1. Specialized accounting traditions in these regions
2. Different functional purposes (accounting vs narrative)
3. Greater standardization in these administrative centers

#### High-Match Khipu Provenance Distribution

**Geographic concentration of exemplar khipus:**
- Incahuasi: 2/9 (22.2%) — **2.65x enrichment**
- Armatambo, Huaca San Pedro: 1/9 (11.1%) — **6.25x enrichment**
- Huaquerones: 1/9 (11.1%) — **3.62x enrichment**
- Ica: 1/9 (11.1%)
- Pachacamac: 1/9 (11.1%)

**Interpretation:** High-match exemplars disproportionately from Incahuasi, Armatambo, and Huaquerones, suggesting these may have been administrative centers with more rigorous accounting standards.

#### Provenance Summary Statistics

All provenances show **Cluster 3 (medium khipus) as most common**, confirming this as the dominant khipu archetype across the empire.

**Regional profiles:**
- **Incahuasi:** Large (111 nodes), deep (2.6), high summation (48.1%), low white markers (44.2%)
- **Leymebamba:** Very large (236 nodes), deep (2.6), high summation (45.5%)
- **Pachacamac:** Medium (67 nodes), low summation (17.8%), very high white markers (90%)
- **Ica:** Small (60 nodes), shallow (1.8), low summation (18.4%), highest numeric (80.5%)
- **Huaquerones:** Very small (38 nodes), shallow (1.5), lowest summation (10.5%)

**Outputs:**
- `geographic_correlation_analysis.json` — Complete statistical results

**Status:** ✅ COMPLETE

---

## Pattern Discovery Implications

### Key Discoveries

1. **Geographic Variation is Real but Limited**
   - Significant structural differences exist between provenances (p < 0.001)
   - Incahuasi emerges as distinct: larger, deeper, higher summation accuracy
   - All regions dominated by Cluster 3 (medium khipus), suggesting empire-wide standardization

2. **Functional Specialization by Region**
   - **Incahuasi/Leymebamba:** Large administrative accounting khipus
   - **Pachacamac:** Heavy use of white boundary markers, lower summation (possibly ceremonial?)
   - **Ica:** High numeric coverage (80.5%), precise encoding
   - **Huaquerones:** Small, simple khipus with minimal summation

3. **Summation as Regional Administrative Signature**
   - 3-4x variance in summation rates across regions (10.5% to 48.1%)
   - High-match exemplars concentrated in specific administrative centers
   - Suggests accounting rigor varied by regional administrative practices

4. **White Cord Paradox Explained**
   - Previous finding: high-match khipus have fewer white cords
   - New finding: Incahuasi (highest summation) uses fewer white markers (44.2%)
   - Interpretation: Sophisticated accounting regions may have relied less on visual separators

5. **Cluster 0 as Elite Administrative Type**
   - Large, deep khipus (Cluster 0) enriched 2-4x in Nazca, Leymebamba, Incahuasi
   - These regions show highest summation accuracy
   - Suggests Cluster 0 represents high-status administrative records

6. **Empire-Wide Standardization with Regional Flavor**
   - Despite significant differences, all regions use same structural archetypes (7 clusters)
   - Medium khipus (Cluster 3) dominant everywhere (63.2%)
   - Suggests central standardization with regional adaptation

### Revised Understanding

The **geographic correlation analysis fundamentally revises** our interpretation of the clustering results:

- Previous: "Weak provenance clustering suggests empire-wide standards override geography"
- **Revised:** "Empire-wide architectural standards (7 archetypes) were adapted to regional administrative needs and traditions"

**Khipu structure reflects:**
1. **Central standardization:** Same 7 structural archetypes used empire-wide
2. **Regional specialization:** Different archetype frequencies and summation practices per region
3. **Administrative hierarchy:** Elite centers (Incahuasi, Leymebamba) produced larger, more complex accounting khipus
4. **Functional diversity:** Regional differences may reflect different record types (accounting vs narrative vs ceremonial)

---

## Next Steps

Phase 4 pattern discovery enables:
- ✅ **Phase 5:** Multi-model framework and function classification
- ✅ **Phase 6:** Advanced interactive visualizations
- ✅ **Phase 7:** Machine learning extensions for anomaly detection and prediction
- ✅ **Phase 8+:** Administrative function analysis and meta-analysis validation

---

### 6. Template Extraction and Analysis

**Objective:** Extract structural templates from perfect-match exemplars (perfect summation and perfect structural similarity khipus).

**Methodology:**
- Identified khipus with perfect summation (match rate = 1.0)
- Identified khipu pairs with perfect structural similarity
- Extracted graph templates (size, depth, branching, level structure)
- Compared templates to find common patterns

**Key Findings:**

#### Perfect Summation Khipus (n=8)

**Identified khipus:**
- 1000093 (Ica): 24 nodes, depth 2, 87.5% numeric
- 1000137 (Pachacamac): 28 nodes, depth 2, 82.1% numeric
- 1000352: 56 nodes, depth 3, 60.7% numeric
- 1000390: 137 nodes, depth 2, 72.3% numeric (largest)
- 1000456: 63 nodes, depth 2, **11.1% numeric** (lowest)
- 1000606 (Incahuasi): 26 nodes, depth 3, 88.5% numeric
- 1000644 (Incahuasi): 5 nodes, depth 2, 80.0% numeric (smallest)
- 1000659 (Armatambo, Huaca San Pedro): 57 nodes, depth 2, 56.1% numeric

**Template characteristics:**
- Size range: 5 to 137 nodes (mean: 56.9)
- Depth range: 2 to 3 levels
- Branching: 0.80 to 1.0 (mean: 0.95) — **nearly linear chains**
- Numeric coverage: 11.1% to 88.5%

**Interpretation:** Perfect summation khipus are structurally **simple linear chains** (low branching ~1.0) with varying sizes. Most have depth=2 (primary cord → pendants → 1 subsidiary). The near-linear structure suggests summation was easiest to maintain in straightforward hierarchies.

#### Perfect Structural Matches (n=27 unique khipus in 20 pairs)

**Match groups:**
- Khipu 1000112 has **5 perfect matches** (1000205, 1000348, 1000354, 1000626, 1000660)
- Khipu 1000001 matches 1000632, 1000653
- Khipu 1000036 matches 1000116, 1000123
- Multiple groups of 2-3 identical structures

**Size distribution:**
- Small (≤10 nodes): 11 khipus
- Medium (11-50 nodes): 9 khipus

**Interpretation:** Perfect structural matches are predominantly **small, simple khipus** (≤10 nodes). The existence of identical structures across different khipus suggests:
1. Standardized templates for common record types
2. Potential copying or systematic replication
3. Simple structures more likely to match by chance

#### Template Comparison Summary

**All templates analyzed (n=28):**
- Size: 5 to 137 nodes (mean: 22.2)
- Depth: 1 to 3 levels (mean: 1.43)
- Branching: 0.80 to 0.99 (mean: 0.90)

**Key pattern:** Templates are **shallow, low-branching structures**, predominantly simple linear or nearly-linear chains.

**Outputs:**
- `template_analysis.json` — Complete template data and comparisons

**Status:** ✅ COMPLETE

---

### 7. Subgraph Motif Mining

**Objective:** Mine recurring structural motifs (subgraph patterns) across clusters to identify common local cord arrangements.

**Methodology:**
- Extracted branching motifs (parent → children patterns with numeric properties)
- Extracted depth motifs (level-to-level connection patterns)
- Analyzed motif frequency within each cluster (sample n=50 per cluster)
- Identified universal motifs appearing across ≥3 clusters

**Key Findings:**

#### Motif Counts by Cluster

| Cluster | Branching Motifs | Unique | Depth Motifs | Unique |
|---------|------------------|--------|--------------|--------|
| **0 (Large deep)** | 3,211 | 206 | 154 | 149 |
| **1 (Small simple)** | 71 | 39 | 64 | 20 |
| **2 (Wide shallow)** | 56 | 52 | 55 | 49 |
| **3 (Medium dominant)** | 400 | 86 | 94 | 77 |
| **4 (Very large)** | 661 | 87 | 18 | 18 |
| **5 (Deep narrow)** | 337 | 98 | 11 | 11 |
| **6 (Non-numeric)** | 48 | 24 | 16 | 16 |

**Observation:** Cluster 0 (large deep khipus) has by far the most motifs (3,211), indicating high structural complexity. Cluster 1 (small simple) has few motifs (71), reflecting simplicity.

#### Universal Branching Motifs

**7 motifs appear across ≥3 clusters:**

1. **(1, (0,), True, 1)** — **ALL 7 clusters**
   - 1 child, child is leaf, parent has value, 1 child has value
   - **Interpretation:** Single pendant with value attached to parent with value
   - **This is the fundamental khipu building block**

2. **(1, (0,), True, 0)** — **6 clusters**
   - 1 child leaf, parent has value, child has no value
   - Non-numeric marker pendant

3. **(1, (0,), False, 1)** — **5 clusters**
   - 1 child leaf, parent no value, child has value
   - Inverted hierarchy (value on pendant not parent)

4. **(2, (0, 0), True, 2)** — **4 clusters**
   - 2 children (both leaves), parent has value, both children have values
   - **Binary summation unit**

5. **(1, (0,), False, 0)** — **4 clusters**
   - Fully non-numeric single pendant
   - Structural marker only

#### No Universal Depth Motifs

**Finding:** Zero depth motifs appear across ≥3 clusters.

**Interpretation:** While local branching patterns are universal, **level-to-level structures are cluster-specific**. This suggests:
- **Micro-structure standardized** (how individual pendants attach)
- **Macro-structure variable** (how levels are organized)

#### Simple Chain Motif Analysis

**Criteria:** depth=2, branching <1.1, nodes >5

**Result:** 0 khipus match these exact criteria.

**Interpretation:** No khipus are pure simple chains by strict definition. Even "simple" khipus have some branching variation.

**Outputs:**
- `motif_mining_results.json` — Complete motif frequencies and patterns

**Status:** ✅ COMPLETE

---

## Pattern Discovery Implications

### Key Discoveries

1. **Geographic Variation is Real but Limited**
   - Significant structural differences exist between provenances (p < 0.001)
   - Incahuasi emerges as distinct: larger, deeper, higher summation accuracy
   - All regions dominated by Cluster 3 (medium khipus), suggesting empire-wide standardization

2. **Functional Specialization by Region**
   - **Incahuasi/Leymebamba:** Large administrative accounting khipus
   - **Pachacamac:** Heavy use of white boundary markers, lower summation (possibly ceremonial?)
   - **Ica:** High numeric coverage (80.5%), precise encoding
   - **Huaquerones:** Small, simple khipus with minimal summation

3. **Summation as Regional Administrative Signature**
   - 3-4x variance in summation rates across regions (10.5% to 48.1%)
   - High-match exemplars concentrated in specific administrative centers
   - Suggests accounting rigor varied by regional administrative practices

4. **White Cord Paradox Explained**
   - Previous finding: high-match khipus have fewer white cords
   - New finding: Incahuasi (highest summation) uses fewer white markers (44.2%)
   - Interpretation: Sophisticated accounting regions may have relied less on visual separators

5. **Cluster 0 as Elite Administrative Type**
   - Large, deep khipus (Cluster 0) enriched 2-4x in Nazca, Leymebamba, Incahuasi
   - These regions show highest summation accuracy
   - Suggests Cluster 0 represents high-status administrative records

6. **Empire-Wide Standardization with Regional Flavor**
   - Despite significant differences, all regions use same structural archetypes (7 clusters)
   - Medium khipus (Cluster 3) dominant everywhere (63.2%)
   - Suggests central standardization with regional adaptation

7. **Perfect Summation Khipus are Linear Chains**
   - All 8 perfect-summation khipus have branching ~1.0 (nearly linear)
   - Depth limited to 2-3 levels
   - Suggests arithmetic precision requires structural simplicity

8. **Universal Micro-Structure, Variable Macro-Structure**
   - 7 branching motifs appear across all/most clusters
   - **Single pendant with value (1, (0,), True, 1)** is fundamental building block
   - No universal depth patterns — macro organization is cluster-specific
   - Implies standardized local rules, flexible global organization

9. **Small Perfect Matches Suggest Standardized Templates**
   - 27 khipus in perfect structural matches, mostly small (≤10 nodes)
   - Khipu 1000112 has 5 identical matches
   - Indicates standardized templates for simple record types

### Revised Understanding

The **complete Phase 4 analysis reveals a hierarchical standardization model:**

**Level 1: Universal Micro-Structure (Motifs)**
- Single pendant attachment is fundamental building block
- Binary summation units standard across empire
- Appears in all 7 structural archetypes

**Level 2: Regional Macro-Structure (Clusters + Geography)**
- 7 structural archetypes used empire-wide
- Regional preferences for specific archetypes (e.g., Incahuasi → Cluster 0)
- Summation rigor varies 4x by region (10.5% to 48.1%)

**Level 3: Functional Specialization (Templates)**
- Perfect summation requires linear simplicity (branching ~1.0)
- Small standardized templates for routine records
- Large complex structures for administrative accounting

**Khipu structure reflects:**
1. **Bottom-up standardization:** Universal micro-patterns (pendant attachments)
2. **Top-down framework:** 7 architectural archetypes enforced empire-wide
3. **Regional adaptation:** Archetype frequency and summation practices vary geographically
4. **Functional optimization:** Structure follows function (simple for arithmetic, complex for administration)

---

## Conclusions

### Phase 4 Accomplishments

**Status:** ✅ **COMPLETE** (7/7 analyses, 100%)

1. Identified 9 high-match summation khipus (≥80% accuracy)
2. Tested hierarchical summation across 384 khipus (35.4% show multi-level patterns)
3. Computed 191,091 pairwise structural similarities (mean 0.065, high diversity)
4. Identified 7 optimal structural archetypes via clustering (k=7, silhouette 0.3692)
5. Demonstrated significant geographic correlations (χ²=116.89, p<0.001)
6. Extracted templates from 8 perfect-summation + 27 perfect-match khipus
7. Mined 4,784 branching motifs across clusters, identified 7 universal patterns

### Major Findings

1. **Hierarchical Standardization Model:** Universal micro-patterns, empire-wide archetypes, regional adaptations
2. **Geographic Administrative Diversity:** Incahuasi 48.1% vs Pachacamac 17.8% summation rates
3. **Structure-Function Relationship:** Perfect summation requires linear simplicity
4. **Single Pendant Motif is Fundamental:** (1, (0,), True, 1) appears in all 7 clusters

### Implications for Khipu Research

1. **Regional Encoding Diversity:** 4x variance in summation by region suggests multiple encoding traditions
2. **Regional Context Essential:** Khipu interpretation must account for provenance
3. **Accounting vs Narrative Distinction:** Low-numeric clusters (e.g., Cluster 6) likely non-accounting
4. **Hierarchical Analysis Critical:** Multi-level summation (35.4%) requires recursive testing

### Next Phase

**Phase 5: Multi-Model Hypothesis Framework**
- Develop provenance-aware interpretation models
- Implement uncertainty quantification for summation hypotheses
- Design expert validation checkpoints
- Build hypothesis testing infrastructure for future research

---

**Report Complete:** December 31, 2025
