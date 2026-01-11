# Khipu Toolkit: Comprehensive Integration Plan

**Date:** January 10, 2026  
**Author:** GitHub Copilot  
**Purpose:** Holistic analysis of toolkit structure and Ascher framework integration

---

## Current State Assessment

### Existing Research Phases (0-9)

#### Phase 0-3: Foundation & Summation
- **Phase 0:** Database reconnaissance
- **Phase 1:** Baseline validation of Ascher's summation rules
- **Phase 2:** Extraction infrastructure (cords, knots, colors)
- **Phase 3:** Summation hypothesis testing
  - **Finding:** ~18% basic summation match (lower than Ascher's 25%)
  - **Issue:** Current implementation doesn't account for Ascher's full categorization framework

#### Phase 4: Pattern Discovery
- Structural clustering using graph metrics
- Template extraction
- Motif mining
- **Finding:** Identified 8-12 structural clusters

#### Phase 5: Multi-Model Hypothesis Testing
Four tested hypotheses:
1. **White Boundaries (MIXED):** +10.7pp improvement in summation with white cords
2. **Color-Value Correlation (REJECTED):** No statistical link (p=0.92)
3. **Color-Function Patterns (SUPPORTED):** Accounting khipus use 57% more colors
4. **Provenance Semantics (REJECTED):** Color meanings standardized empire-wide

#### Phase 6-7: Advanced Analysis
- Geographic correlations
- Advanced visualizations (3D viewer we just built)
- ML extensions for prediction

#### Phase 8: Administrative Function
- Classification of administrative vs narrative khipus
- 98% accuracy in function prediction

#### Phase 9: Meta-Analysis & Validation
- Robustness testing
- Information capacity (8.47 bits avg, 67% redundancy)
- Cognitive load analysis
- Negative knowledge boundaries

---

## The Ascher Gap: What's Missing

### Current Understanding vs Ascher's Framework

**What We Have:**
- ✅ Cord hierarchy extraction
- ✅ Knot value computation (TYPE_CODE, NUM_TURNS)
- ✅ Basic summation testing (pendant sum = main cord?)
- ✅ Color distribution analysis
- ✅ Structural clustering

**What Ascher Adds:**
- ❌ **Cross-categorization detection** (p_ij, p_ijk structures)
- ❌ **Spatial grouping analysis** (gaps in CORD_ORDINAL as category boundaries)
- ❌ **Categorical sums** (row/column totals beyond simple hierarchy)
- ❌ **Format classification** (identifying which khipus are "formatted records")
- ❌ **Logical structure validation** (is summation internally consistent?)

### Why Phase 3 Found Only 18% (Not 25%)

**Root Cause:** Phase 3 tested **hierarchical summation only**
- Checked: Do pendant knot values sum to main cord value?
- Missed: Cross-categorical summation patterns (groups × colors)
- Missed: Partial summation (some cords are sums, others are data)
- Missed: Multi-level summation (subsidiaries → pendants → groups → grand total)

**Ascher's 25% likely includes:**
1. Pure hierarchical (what we found: ~18%)
2. Cross-categorical (groups with row/column sums)
3. Hybrid patterns (some hierarchical, some categorical)
4. Partial summation (main cord sums only certain pendant groups)

---

## Proposed Integration: 10-Phase Structure

### Phase 10: Ascher Logical Framework (NEW)

**Purpose:** Implement complete Ascher categorization system

#### Module 10.1: Value Computation (CORRECTED)
**Current Issues:**
- Wrong column names (KNOT_TYPE → TYPE_CODE)
- Wrong position field (KNOT_POSITION → CLUSTER_ORDINAL)
- Wrong hierarchy field (PARENT_CORD_ID → PENDANT_FROM/ATTACHED_TO)

**New Implementation:**
```python
def compute_cord_value_corrected(cord_id, db_conn):
    """Compute decimal value using correct schema"""
    query = """
    SELECT k.TYPE_CODE, k.CLUSTER_ORDINAL, k.NUM_TURNS
    FROM knot k
    JOIN cord c ON k.CORD_ID = c.CORD_ID
    WHERE c.CORD_ID = ?
    ORDER BY k.CLUSTER_ORDINAL DESC
    """
    knots = execute_query(query, cord_id)
    value = 0
    for knot in knots:
        position = int(knot['CLUSTER_ORDINAL'])
        if knot['TYPE_CODE'] == 'S':
            value += 1 * (10 ** position)
        elif knot['TYPE_CODE'] == 'L':
            turns = knot['NUM_TURNS'] or 1
            value += int(turns) * (10 ** position)
        elif knot['TYPE_CODE'] == 'E' and position == 0:
            pass  # Explicit zero
    return value
```

#### Module 10.2: Spatial Grouping Detection
**Method:** Detect category boundaries from CORD_ORDINAL gaps

```python
def detect_spatial_groups(khipu_id, gap_threshold=2.0):
    """
    Find groups of pendants separated by gaps.
    Returns: List of cord groups
    """
    pendants = get_pendants_ordered(khipu_id)
    groups = []
    current_group = []
    
    for i, pendant in enumerate(pendants):
        if i == 0:
            current_group = [pendant]
        else:
            gap = pendant['CORD_ORDINAL'] - pendants[i-1]['CORD_ORDINAL']
            if gap > gap_threshold:
                groups.append(current_group)
                current_group = [pendant]
            else:
                current_group.append(pendant)
    
    if current_group:
        groups.append(current_group)
    
    return groups
```

#### Module 10.3: Cross-Categorization Analysis
**Method:** Matrix structure detection (spatial × color)

```python
def detect_cross_categorization(khipu_id):
    """
    Detect p_ij structures: spatial groups × color categories
    """
    spatial_groups = detect_spatial_groups(khipu_id)
    colors = get_unique_colors(khipu_id)
    
    # Build matrix: rows=spatial groups, cols=colors
    matrix = {}
    for g_idx, group in enumerate(spatial_groups):
        for pendant in group:
            color = pendant['COLOR_CD_1']
            matrix[(g_idx, color)] = compute_cord_value(pendant['CORD_ID'])
    
    # Check if this is a structured p_ij arrangement
    has_structure = (
        len(spatial_groups) > 1 and
        len(colors) > 1 and
        matrix_is_complete(matrix, spatial_groups, colors)
    )
    
    return {
        'is_cross_categorized': has_structure,
        'spatial_dim': len(spatial_groups),
        'color_dim': len(colors),
        'matrix': matrix
    }
```

#### Module 10.4: Categorical Summation Verification
**Method:** Check row/column/grand totals

```python
def verify_categorical_sums(khipu_id, matrix_structure):
    """
    For p_ij structures, verify:
    - Row sums (sum across colors for each spatial group)
    - Column sums (sum across spatial groups for each color)
    - Grand total (sum of all values)
    """
    results = {
        'row_sums': [],
        'col_sums': [],
        'grand_total': None,
        'accuracy': {}
    }
    
    # Find sum cords (often subsidiaries or special position cords)
    for group_idx in range(matrix_structure['spatial_dim']):
        expected_sum = sum_matrix_row(matrix_structure['matrix'], group_idx)
        actual_sum_cord = find_sum_cord_for_group(khipu_id, group_idx)
        if actual_sum_cord:
            accuracy = match_accuracy(expected_sum, actual_sum_cord['value'])
            results['row_sums'].append({
                'group': group_idx,
                'expected': expected_sum,
                'actual': actual_sum_cord['value'],
                'match': accuracy > 0.95
            })
    
    # Similar for column sums and grand total
    return results
```

#### Module 10.5: Format Classification
**Method:** Classify khipus by Ascher format types

Categories:
1. **Formatted Accounting** (p_ij with sums) - Ascher's 25%
2. **Simple Hierarchical** (pendant → main, no categories)
3. **Color-Coded Lists** (categories but no summation)
4. **Narrative/Mnemonic** (irregular structure)
5. **Hybrid** (mixed patterns)

---

## Integration with Existing Phases

### Enhanced Phase Structure

```
Phase 0-2: Foundation (KEEP AS-IS)
   └─ Data extraction, baseline infrastructure

Phase 3: Summation Testing (DEPRECATE → Merge into Phase 10)
   └─ Current version is incomplete
   └─ Findings (18%) superseded by Phase 10

Phase 4: Pattern Discovery (ENHANCE)
   └─ Add Ascher format types as clustering features
   └─ Compare structural clusters to logical formats

Phase 5: Hypothesis Testing (ENHANCE)
   ├─ Keep existing 4 hypotheses
   └─ ADD: H5: Categorical Summation (Ascher's p_ij patterns)

Phase 6-7: Visualizations (ENHANCE)
   └─ 3D Viewer: Show computed values, summation flows, category boundaries
   └─ Matrix Viewer: Display p_ij structures as tables

Phase 8: Administrative Function (INTEGRATE)
   └─ Use Phase 10 format classification as feature
   └─ "Formatted Accounting" → administrative function

Phase 9: Meta-Analysis (EXTEND)
   └─ Validate Phase 10 findings with robustness tests

Phase 10: Ascher Logical Framework (NEW)
   └─ Complete implementation as described above
```

---

## Revised Toolkit Architecture

### New Module: `src/analysis/ascher_logic.py`

```python
class AscherAnalyzer:
    """
    Complete implementation of Ascher's logical framework
    from Chapter 5: Format, Category, and Summation
    """
    
    def __init__(self, db_path):
        self.db = Database(db_path)
        self.value_cache = {}
    
    def analyze_khipu(self, khipu_id):
        """Full Ascher analysis pipeline"""
        
        # Step 1: Compute all cord values (corrected schema)
        values = self.compute_all_cord_values(khipu_id)
        
        # Step 2: Detect spatial grouping
        spatial = self.detect_spatial_groups(khipu_id)
        
        # Step 3: Detect color categories
        colors = self.detect_color_categories(khipu_id)
        
        # Step 4: Check for cross-categorization
        cross_cat = self.detect_cross_categorization(
            khipu_id, spatial, colors
        )
        
        # Step 5: Verify summation patterns
        if cross_cat['is_cross_categorized']:
            sums = self.verify_categorical_sums(khipu_id, cross_cat)
        else:
            sums = self.verify_hierarchical_sums(khipu_id, values)
        
        # Step 6: Classify format
        format_type = self.classify_format(
            spatial, colors, cross_cat, sums
        )
        
        return {
            'khipu_id': khipu_id,
            'cord_values': values,
            'spatial_groups': spatial,
            'color_categories': colors,
            'cross_categorization': cross_cat,
            'summation': sums,
            'format_type': format_type,
            'ascher_formatted': format_type in ['p_ij', 'p_ijk', 'hybrid']
        }
```

### Updated Data Model

Add computed tables:

```sql
-- Cord values (computed from knots)
CREATE TABLE cord_computed_values (
    cord_id INTEGER PRIMARY KEY,
    khipu_id INTEGER,
    decimal_value INTEGER,
    computation_method TEXT,
    confidence REAL,
    FOREIGN KEY (cord_id) REFERENCES cord(CORD_ID)
);

-- Spatial groups
CREATE TABLE spatial_groups (
    group_id INTEGER PRIMARY KEY,
    khipu_id INTEGER,
    group_index INTEGER,
    start_ordinal REAL,
    end_ordinal REAL,
    cord_count INTEGER
);

-- Cross-categorization structures
CREATE TABLE khipu_categories (
    khipu_id INTEGER PRIMARY KEY,
    has_spatial_groups BOOLEAN,
    num_spatial_groups INTEGER,
    has_color_categories BOOLEAN,
    num_color_categories INTEGER,
    is_cross_categorized BOOLEAN,
    category_notation TEXT,  -- 'p_ij', 'p_ijk', etc.
    FOREIGN KEY (khipu_id) REFERENCES cord(KHIPU_ID)
);

-- Summation verification
CREATE TABLE summation_analysis (
    khipu_id INTEGER PRIMARY KEY,
    has_hierarchical_summation BOOLEAN,
    hierarchical_accuracy REAL,
    has_categorical_summation BOOLEAN,
    categorical_accuracy REAL,
    has_row_sums BOOLEAN,
    has_column_sums BOOLEAN,
    has_grand_total BOOLEAN,
    format_classification TEXT,
    FOREIGN KEY (khipu_id) REFERENCES cord(KHIPU_ID)
);
```

---

## Implementation Priorities

### CRITICAL PATH (Week 1-2)

**Priority 1: Fix Schema Issues**
- Update all analysis scripts to use correct column names
- Create `cord_value_computer.py` with corrected implementation
- Validate on 10 known examples from Ascher's book

**Priority 2: Implement Module 10.2-10.4**
- Spatial grouping detection
- Cross-categorization analysis  
- Categorical summation verification

**Priority 3: Re-run Phase 3 Analysis**
- Compare old (18%) vs new findings
- Document which khipus moved from "no summation" to "categorical summation"
- Publish updated statistics

### IMPORTANT (Week 3-4)

**Priority 4: Integration**
- Enhance Phase 4 clustering with Ascher features
- Add H5 to Phase 5 hypothesis testing
- Update Phase 8 function classifier

**Priority 5: Visualization Updates**
- 3D Viewer: Show computed values on hover
- 3D Viewer: Highlight sum cords
- New Matrix Viewer for p_ij structures

**Priority 6: Documentation**
- SUMMATION_GUIDE.md with examples
- Update ARCHITECTURE.md with Module 10
- Update API_REFERENCE.md

### NICE-TO-HAVE (Week 5+)

**Priority 7: Advanced Analysis**
- Three-way categorization (p_ijk) detection
- Partial summation patterns
- Hybrid format identification

**Priority 8: Publication**
- Compare findings to Ascher's book examples
- Validate with archaeological literature
- Prepare dataset for release

---

## Expected Outcomes

### Quantitative Goals

1. **Summation Rate:** 18% → 22-28% (closer to Ascher's 25%)
   - Breakdown by type: hierarchical vs categorical vs hybrid

2. **Format Classification:** Categorize all 612 khipus
   - Formatted Accounting: ~150-170 khipus (25-28%)
   - Simple Hierarchical: ~250 khipus (40%)
   - Color Lists: ~100 khipus (16%)
   - Narrative: ~50 khipus (8%)
   - Hybrid: ~40 khipus (7%)

3. **Cross-Categorization:** Identify ~50-80 khipus with p_ij or p_ijk structures

### Qualitative Goals

1. **Theoretical Alignment:** Match Ascher's framework explicitly
2. **Reproducibility:** All findings validated with public code
3. **Archaeological Impact:** Provide tools for field researchers
4. **Educational Value:** Clear examples for students

---

## Risk Assessment

### Technical Risks

**Risk 1: Interpretation Ambiguity**
- **Issue:** Multiple valid interpretations of same structure
- **Mitigation:** Document all assumptions, provide confidence scores
- **Impact:** Medium - affects specific classifications, not overall patterns

**Risk 2: Data Quality**
- **Issue:** Missing/incomplete cord or knot data
- **Mitigation:** Flag low-confidence khipus, exclude from statistics
- **Impact:** Low - affects ~5% of dataset

**Risk 3: Computational Complexity**
- **Issue:** Matrix detection is O(n³) for large khipus
- **Mitigation:** Heuristic pre-filtering, parallel processing
- **Impact:** Low - performance acceptable on modern hardware

### Scientific Risks

**Risk 4: Overfitting Ascher's Framework**
- **Issue:** Finding patterns because we're looking for them
- **Mitigation:** Phase 9 robustness testing, negative controls
- **Impact:** High - addressed by rigorous validation

**Risk 5: Cultural Context**
- **Issue:** Mathematical interpretation may miss cultural meaning
- **Mitigation:** Clearly label as "structural analysis," defer to archaeologists
- **Impact:** Medium - this toolkit is ONE lens, not the only lens

---

## Success Metrics

### Technical Metrics
- ✅ All 612 khipus processed without errors
- ✅ >95% of cords with knots have computed values
- ✅ Cross-categorization detection accuracy >90% on manual test set
- ✅ Summation verification runs in <5 minutes for full dataset

### Scientific Metrics
- ✅ Summation rate: 22-28% (matches Ascher's 25% ± 3pp)
- ✅ Identify ≥3 new exemplar p_ij khipus not in Ascher's book
- ✅ Phase 9 robustness tests show patterns stable under perturbation
- ✅ Results validated against published archaeological interpretations

### Impact Metrics
- ✅ Toolkit used by ≥3 independent researchers
- ✅ Documentation clear enough for undergraduates to use
- ✅ Code/data cited in ≥1 peer-reviewed publication
- ✅ Open-source contributions from community

---

## Conclusion

### The Big Picture

The toolkit has achieved remarkable technical sophistication (Phases 0-9) but **missed the logical foundation** that Ascher established. Phase 10 integration will:

1. **Correct technical errors** (schema issues in value computation)
2. **Fill conceptual gaps** (cross-categorization, categorical sums)
3. **Align with theory** (explicit Ascher framework implementation)
4. **Validate findings** (re-run analyses with complete methodology)

### The Path Forward

1. **Immediate:** Fix schema issues, implement Module 10.1-10.4
2. **Short-term:** Integrate with Phases 4-5, update visualizations
3. **Medium-term:** Full validation, documentation, exemplar identification
4. **Long-term:** Publication, community engagement, ongoing refinement

**This is not a rewrite** - it's **strategic enhancement** that makes explicit what was implicit, corrects what was wrong, and completes what was incomplete.

The toolkit will emerge as a **comprehensive computational framework** that bridges:
- Archaeological evidence (OKR database)
- Logical structure (Ascher's framework)
- Pattern discovery (clustering, ML)
- Validation (robustness, meta-analysis)
- Visualization (3D viewer, matrix displays)

**Goal:** Make khipu analysis accessible, reproducible, and rigorous.

---

## Next Actions (Immediate)

1. ☐ Review this plan with stakeholders
2. ☐ Prioritize Phase 10 modules (10.1-10.4 first)
3. ☐ Create GitHub issues for each module
4. ☐ Set up test dataset (10-20 known khipus from Ascher's book)
5. ☐ Begin implementation with value computation fixes
6. ☐ Document as we build (not after!)

---

**END OF INTEGRATION PLAN**
