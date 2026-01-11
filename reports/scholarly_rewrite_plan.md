# Khipu Computational Toolkit: Scholarly Rewrite & Enhancement Plan

**Date:** January 10, 2026  
**Context:** Building on Ascher (1997, 2002), Medrano & Khosla (2024), and OKR Database  
**Goal:** Create a rigorous, multi-method toolkit for scholars and students

---

## Foundational Premise

**From your statement:**
> "This project is situated within a growing body of computational research on Andean khipus, most notably the work of Medrano & Khosla (2024), which demonstrates, across a large corpus, that many khipus exhibit structured internal summation relationships consistent with earlier observations by Marcia Ascher."

**What this means for the toolkit:**
1. We're **validating and extending** established findings, not discovering from scratch
2. **Multiple interpretation methods** must coexist (summation is ONE lens)
3. **Computational validation** of archaeological observations is the core mission
4. **Pedagogical clarity** matters as much as research capability

---

## Critical Analysis: What Needs Rewriting

### 1. Core Value Computation (MUST REWRITE)

**Current Problem:**
- Wrong schema (KNOT_TYPE vs TYPE_CODE, KNOT_POSITION vs CLUSTER_ORDINAL)
- Single interpretation method (decimal positional)
- No uncertainty quantification
- No alternative readings

**Rewrite: Multi-Method Value Computer**

```python
class KhipuValueComputer:
    """
    Multiple interpretation methods for computing cord values.
    Follows Ascher (1981, 2002) and Medrano & Khosla (2024).
    """
    
    def compute_value(self, cord_id: int, method: str = 'ascher_decimal') -> ValueResult:
        """
        Compute cord value using specified method.
        
        Methods:
            'ascher_decimal': Ascher's base-10 positional system
                S knot in position n = 1 × 10^n
                L knot with k turns in position n = k × 10^n
                E knot in position 0 = explicit zero
                
            'medrano_summation': Focused on summation verification
                Same as ascher_decimal but with match scoring
                
            'locke_decimal': Alternative position assignments
                Some scholars assign positions differently
                
            'urton_signs': Considers knot direction (S vs Z)
                May encode binary or signed values
                
            'cluster_analysis': Uses cluster structure without decimal assumption
                For non-numeric interpretations
        
        Returns:
            ValueResult with:
                - primary_value: Most likely numeric interpretation
                - alternative_values: Dict of other method results
                - confidence: 0-1 score based on knot clarity
                - uncertainty: List of ambiguous elements
                - raw_knots: Original knot data for inspection
        """
```

**Why Multiple Methods:**
- Ascher's decimal is widely accepted but **not proven** for all khipus
- Some khipus may be non-numeric (narrative, mnemonic)
- Scholarly debate ongoing - toolkit should support exploration
- Students need to understand interpretation is analytical choice

### 2. Summation Analysis (MUST REWRITE)

**Current Problem:**
- Tests only one pattern (pendant sum → main cord)
- No concept of partial summation
- Doesn't model Medrano & Khosla's corpus findings
- Binary pass/fail instead of confidence scoring

**Rewrite: Comprehensive Summation Framework**

```python
class SummationAnalyzer:
    """
    Multi-level summation detection following Ascher and Medrano & Khosla.
    Implements five summation pattern types with confidence scoring.
    """
    
    def analyze_khipu(self, khipu_id: int) -> SummationReport:
        """
        Full summation analysis with multiple pattern types.
        
        Pattern Types (from Medrano & Khosla 2024):
        
        1. Simple Hierarchical (Type A)
           - Main cord = sum of all pendants
           - Most common pattern in corpus
           
        2. Grouped Hierarchical (Type B)
           - Spatial groups each sum to a marker cord
           - Group sums may sum to main cord
           
        3. Categorical Matrix (Type C) - Ascher's p_ij
           - Rows = spatial groups
           - Columns = color categories
           - Row/column totals present
           
        4. Partial Summation (Type D)
           - Only some cords participate in sums
           - Others are descriptive/categorical
           
        5. Nested Multi-Level (Type E)
           - Subsidiaries → pendants → groups → main
           - Multiple summation layers
        
        Returns:
            SummationReport with:
                - detected_patterns: List of pattern types found
                - confidence_scores: Per-pattern confidence (0-1)
                - match_accuracy: How well sums match (%)
                - participating_cords: Which cords are in summation
                - non_participating_cords: Which are excluded
                - interpretation_notes: Ambiguities and alternatives
        """
```

**Critical Feature: Match Tolerance**

Ascher and Medrano & Khosla note that **exact matches are rare**. We must model:
- Missing/damaged cords (archaeological reality)
- Transcription errors in database
- Alternative reading possibilities
- Cultural conventions (rounding, approximation)

```python
def compute_match_score(expected: int, actual: int, 
                       tolerance: float = 0.05) -> MatchScore:
    """
    Score summation match with archaeological realism.
    
    Scoring:
        - Exact match (expected == actual): 1.0
        - Within 5% (default tolerance): 0.8-0.95
        - Within 10%: 0.6-0.79
        - Within 20%: 0.4-0.59
        - >20% off: 0.0-0.39
        
    Also considers:
        - Possible missing cords (gaps in ordinals)
        - Alternative readings (if E vs S ambiguous)
        - Round number bias (300 vs 297 → cultural rounding?)
    """
```

### 3. Category Detection (NEW MODULE NEEDED)

**Current Problem:**
- Doesn't exist at all
- Ascher's core concept (p_ij) not implemented
- Can't identify formatted vs unformatted khipus

**Write from Scratch: Category Analyzer**

```python
class CategoryAnalyzer:
    """
    Detect logical categories following Ascher's Chapter 5.
    Identifies spatial grouping, color coding, and cross-categorization.
    """
    
    def detect_categories(self, khipu_id: int) -> CategoryStructure:
        """
        Multi-dimensional category detection.
        
        Detection Methods:
        
        1. Spatial Categories (Primary Dimension)
           Algorithm: Gap-based clustering of CORD_ORDINAL
           - Compute gaps between sequential pendants
           - Threshold: μ + 2σ of all gaps
           - Result: Groups of spatially clustered cords
           
        2. Color Categories (Secondary Dimension)
           Algorithm: Unique color assignment per cord
           - Extract COLOR_CD_1 from ascher_cord_color
           - Group cords with same color across spatial groups
           - Result: Color classes that cross-cut spatial groups
           
        3. Position Categories (Tertiary Dimension)
           Algorithm: Sequential position within group
           - 1st, 2nd, 3rd... cord in each spatial group
           - May represent temporal or logical sequence
           
        4. Attachment Categories (Structural Dimension)
           Algorithm: Pendant vs Subsidiary vs Top cord
           - CORD_LEVEL from database
           - PENDANT_FROM relationships
           - Structural role in hierarchy
        
        Returns:
            CategoryStructure with:
                - spatial_groups: List of cord groups with boundaries
                - color_categories: Dict of color → cord lists
                - position_index: Dict of (group, position) → cord
                - cross_categorization: Matrix if p_ij detected
                - dimensionality: 1D (list), 2D (p_ij), 3D (p_ijk)
        """
    
    def classify_format(self, categories: CategoryStructure,
                       summation: SummationReport) -> FormatType:
        """
        Classify khipu by Ascher's logical format types.
        
        Format Types:
        
        1. Formatted Accounting Record (Ascher's ~25%)
           - Cross-categorization present (p_ij or p_ijk)
           - Summation relationships verified
           - Structured, repeating pattern
           - Example: Inventory with (items × locations)
           
        2. Simple Numerical List
           - Single dimension (no cross-categorization)
           - May have hierarchical summation
           - Example: Population counts by village
           
        3. Color-Coded Classification
           - Multiple colors but no summation
           - Categories may be qualitative
           - Example: Tribute types, social groups
           
        4. Hierarchical Narrative
           - Complex subsidiary structure
           - Minimal numeric regularity
           - May be mnemonic device
           
        5. Hybrid/Indeterminate
           - Mixed patterns or insufficient data
        """
```

### 4. Statistical Validation (ENHANCE PHASE 9)

**Current Status:** Good foundation in Phase 9 meta-analysis

**Enhancement Needed:**

```python
class ValidationFramework:
    """
    Statistical validation of all findings with archaeological context.
    Extends Phase 9 with summation-specific tests.
    """
    
    def validate_summation_claim(self, khipu_id: int, 
                                 pattern: SummationPattern) -> Validation:
        """
        Rigorous statistical test of summation claim.
        
        Tests Applied:
        
        1. Null Hypothesis Test
           H0: Observed match is due to chance
           Method: Monte Carlo simulation
           - Randomly permute cord values
           - Compute "accidental" match rate
           - Compare to observed match
           - Reject H0 if p < 0.05
           
        2. Consistency Check
           - Are all group sums correct, or just some?
           - Pattern of errors: random or systematic?
           - Missing cords: do gaps explain discrepancies?
           
        3. Alternative Explanation Test
           - Could this be a different pattern type?
           - Test competing hypotheses
           - Report likelihood ratios
           
        4. Corpus Comparison (Medrano & Khosla 2024)
           - How does this khipu compare to known examples?
           - Is pattern typical or unusual?
           - Similarity to validated exemplars
        
        Returns:
            Validation with:
                - p_value: Statistical significance
                - confidence_level: High/Medium/Low/Uncertain
                - alternative_hypotheses: Other possible patterns
                - corpus_percentile: Where this ranks in dataset
                - archaeological_notes: Context from literature
        """
```

---

## New Module Structure

### Module 1: Core Value Computation (REWRITE)
**File:** `src/analysis/value_computation.py`

Classes:
- `KnotExtractor` - Get knots with correct schema (TYPE_CODE, CLUSTER_ORDINAL)
- `ValueComputer` - Multi-method value computation
- `ValueResult` - Data class with uncertainty
- `AlternativeReadings` - Document interpretation choices

**Key Features:**
- Correct database schema usage
- Multiple interpretation methods
- Confidence scoring
- Raw data preservation for verification

### Module 2: Summation Analysis (REWRITE)
**File:** `src/analysis/summation_patterns.py`

Classes:
- `SummationAnalyzer` - Main analysis engine
- `PatternDetector` - Identify summation types A-E
- `MatchScorer` - Compute match quality with tolerance
- `SummationReport` - Complete results with uncertainty

**Key Features:**
- Five pattern types (Medrano & Khosla taxonomy)
- Partial summation support
- Match tolerance modeling
- Corpus-aware interpretation

### Module 3: Category Detection (NEW)
**File:** `src/analysis/category_detection.py`

Classes:
- `CategoryAnalyzer` - Detect spatial, color, position categories
- `CategoryStructure` - Data class for multi-dimensional categories
- `FormatClassifier` - Ascher's format types
- `MatrixBuilder` - Construct p_ij, p_ijk representations

**Key Features:**
- Gap-based spatial clustering
- Multi-dimensional categorization
- Ascher format classification
- Matrix representation of cross-categorization

### Module 4: Validation Framework (ENHANCE)
**File:** `src/analysis/validation.py`

Classes:
- `ValidationFramework` - Statistical testing
- `NullHypothesisTester` - Monte Carlo simulation
- `CorpusComparator` - Compare to known examples
- `ConfidenceScorer` - Unified confidence assessment

**Key Features:**
- Statistical significance testing
- Alternative hypothesis evaluation
- Corpus-level comparison
- Archaeological context integration

### Module 5: Multi-Method Pipeline (NEW)
**File:** `src/analysis/khipu_analyzer.py`

```python
class KhipuAnalyzer:
    """
    Unified analysis pipeline combining all methods.
    Main interface for scholars and students.
    """
    
    def analyze(self, khipu_id: int, 
               methods: List[str] = ['all']) -> AnalysisReport:
        """
        Complete khipu analysis with multiple approaches.
        
        Pipeline:
        1. Value Computation (all methods)
        2. Category Detection (spatial, color, position)
        3. Summation Analysis (all pattern types)
        4. Statistical Validation (significance tests)
        5. Corpus Comparison (Medrano & Khosla context)
        6. Format Classification (Ascher types)
        7. Interpretation Synthesis (narrative summary)
        
        Returns:
            AnalysisReport suitable for:
                - Research publication (full statistical detail)
                - Student learning (pedagogical explanations)
                - Database query (structured data)
                - Visualization (3D viewer integration)
        """
    
    def compare_methods(self, khipu_id: int) -> MethodComparison:
        """
        Compare results across interpretation methods.
        Educational tool showing how choices affect conclusions.
        """
    
    def batch_analyze(self, khipu_ids: List[int], 
                     parallel: bool = True) -> CorpusReport:
        """
        Analyze multiple khipus for corpus-level patterns.
        Replicates Medrano & Khosla (2024) methodology.
        """
```

---

## Data Model Revision

### New Tables (Add to Database or Create Views)

```sql
-- Computed values with method transparency
CREATE TABLE cord_values (
    cord_id INTEGER PRIMARY KEY,
    khipu_id INTEGER NOT NULL,
    
    -- Ascher decimal interpretation
    ascher_decimal_value INTEGER,
    ascher_confidence REAL,
    ascher_uncertainty TEXT,
    
    -- Alternative interpretations
    medrano_value INTEGER,
    locke_value INTEGER,
    urton_signed_value INTEGER,
    
    -- Raw data for verification
    knot_sequence TEXT,  -- JSON: [{type, position, turns}]
    computation_notes TEXT,
    
    -- Quality indicators
    has_ambiguous_knots BOOLEAN,
    missing_data_flags TEXT,
    
    FOREIGN KEY (cord_id) REFERENCES cord(CORD_ID)
);

-- Category assignments
CREATE TABLE cord_categories (
    cord_id INTEGER PRIMARY KEY,
    khipu_id INTEGER NOT NULL,
    
    -- Spatial categories
    spatial_group_id INTEGER,
    spatial_group_position INTEGER,  -- 1st, 2nd, 3rd in group
    
    -- Color categories
    primary_color TEXT,
    color_category_id INTEGER,
    
    -- Structural categories
    cord_level INTEGER,  -- 0=main, 1=pendant, 2+=subsidiary
    parent_cord_id INTEGER,
    attachment_type TEXT,
    
    -- Cross-categorization
    matrix_row INTEGER,  -- For p_ij structures
    matrix_col INTEGER,
    matrix_page INTEGER,  -- For p_ijk structures
    
    FOREIGN KEY (cord_id) REFERENCES cord(CORD_ID)
);

-- Summation patterns
CREATE TABLE summation_patterns (
    pattern_id INTEGER PRIMARY KEY,
    khipu_id INTEGER NOT NULL,
    
    -- Pattern type
    pattern_type TEXT,  -- 'simple_hierarchical', 'grouped', 'matrix', 'partial', 'nested'
    pattern_description TEXT,
    
    -- Participants
    sum_cord_id INTEGER,  -- The cord holding the sum
    component_cord_ids TEXT,  -- JSON array of cords being summed
    
    -- Match quality
    expected_sum INTEGER,
    actual_sum INTEGER,
    match_score REAL,  -- 0-1
    match_percentage REAL,  -- 0-100%
    
    -- Statistical validation
    p_value REAL,  -- Null hypothesis test
    confidence_level TEXT,  -- 'high', 'medium', 'low', 'uncertain'
    
    -- Context
    notes TEXT,
    verified_by TEXT,  -- 'algorithm', 'manual', 'literature'
    
    FOREIGN KEY (khipu_id) REFERENCES cord(KHIPU_ID),
    FOREIGN KEY (sum_cord_id) REFERENCES cord(CORD_ID)
);

-- Khipu format classification
CREATE TABLE khipu_formats (
    khipu_id INTEGER PRIMARY KEY,
    
    -- Ascher classification
    ascher_format TEXT,  -- 'formatted_accounting', 'simple_list', 'color_coded', 'narrative', 'hybrid'
    ascher_confidence REAL,
    
    -- Dimensionality
    is_cross_categorized BOOLEAN,
    spatial_dimension INTEGER,  -- Number of spatial groups
    color_dimension INTEGER,     -- Number of color categories
    notation TEXT,  -- 'p_ij', 'p_ijk', 'simple', etc.
    
    -- Summation presence
    has_summation BOOLEAN,
    summation_type TEXT,
    summation_coverage REAL,  -- % of cords participating
    
    -- Corpus comparison (Medrano & Khosla 2024)
    corpus_percentile REAL,  -- Where this ranks in structural complexity
    similar_khipus TEXT,  -- JSON array of similar khipu_ids
    
    -- Quality indicators
    completeness_score REAL,  -- % of cords with complete data
    data_quality_flags TEXT,
    
    FOREIGN KEY (khipu_id) REFERENCES cord(KHIPU_ID)
);

-- Analysis metadata (reproducibility)
CREATE TABLE analysis_metadata (
    analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
    khipu_id INTEGER NOT NULL,
    analysis_date TEXT NOT NULL,
    
    -- Methods used
    value_computation_method TEXT,
    summation_detection_method TEXT,
    category_detection_params TEXT,  -- JSON
    
    -- Software version
    toolkit_version TEXT,
    database_version TEXT,
    
    -- Results hash (for reproducibility)
    results_hash TEXT,
    
    -- Attribution
    analyst TEXT,
    notes TEXT,
    
    FOREIGN KEY (khipu_id) REFERENCES cord(KHIPU_ID)
);
```

---

## Visualization Updates

### 1. Enhanced 3D Viewer

**Current:** Shows structure with colors  
**Add:**
- **Value Display:** Show computed values on hover
- **Summation Flow:** Animate arrows showing sum relationships
- **Category Highlighting:** Color-code by spatial group
- **Match Indicators:** Green=good match, yellow=partial, red=mismatch
- **Method Selector:** Switch between interpretation methods
- **Confidence Display:** Visual indicators of uncertainty

```python
# In plotly_3d_viewer_new.py
def create_3d_plot_enhanced(khipu_data, analysis_report):
    """
    Enhanced 3D viewer integrated with analysis results.
    
    Features:
    - Cord values displayed on hover
    - Summation relationships shown as connecting lines
    - Category groups visually distinguished
    - Match quality color-coded
    - Toggle between multiple interpretation methods
    """
```

### 2. Matrix Viewer (NEW)

For khipus with p_ij or p_ijk structures:

```python
class MatrixViewer:
    """
    Display cross-categorized khipus as tables.
    Shows Ascher's subscript notation visually.
    """
    
    def display_p_ij(self, khipu_id, category_structure):
        """
        Show 2D matrix (spatial groups × colors).
        Includes row sums, column sums, grand total.
        Highlights which sums are verified.
        """
    
    def display_p_ijk(self, khipu_id, category_structure):
        """
        Show 3D structure as set of 2D tables.
        One table per "page" (3rd dimension).
        Shows inter-table summation relationships.
        """
```

### 3. Method Comparison Dashboard (NEW)

Educational tool showing how interpretation choices matter:

```python
class MethodComparisonDashboard:
    """
    Side-by-side comparison of interpretation methods.
    Pedagogical tool for students.
    """
    
    def show_interpretation_impact(self, khipu_id):
        """
        Display:
        - Same khipu analyzed with different methods
        - How results differ
        - Which methods agree/disagree
        - Statistical confidence for each
        - Scholarly rationale for each approach
        """
```

---

## Educational Components

### 1. Tutorial Notebooks (NEW)

Create Jupyter notebooks for learning:

```
notebooks/
├── 01_intro_to_khipus.ipynb
│   - What are khipus?
│   - Basic structure
│   - Our dataset
│
├── 02_reading_values.ipynb
│   - Ascher's decimal system
│   - Step-by-step value computation
│   - Ambiguities and choices
│
├── 03_detecting_summation.ipynb
│   - What is summation?
│   - Pattern types A-E
│   - Medrano & Khosla examples
│
├── 04_categories_and_format.ipynb
│   - Ascher's p_ij notation
│   - Category detection
│   - Format classification
│
├── 05_statistical_validation.ipynb
│   - How do we know summation is real?
│   - Null hypothesis testing
│   - Confidence assessment
│
└── 06_full_analysis_pipeline.ipynb
    - Complete analysis of one khipu
    - All methods, all tests
    - Interpretation synthesis
```

### 2. Exemplar Library (NEW)

Curated set of well-understood khipus:

```python
class ExemplarLibrary:
    """
    Collection of validated examples from literature.
    Reference set for learning and comparison.
    """
    
    EXEMPLARS = {
        'simple_summation': [
            'AS100',  # Ascher's example with perfect hierarchical sum
            'UR004',  # Urton's administrative khipu
        ],
        'matrix_p_ij': [
            'AS135',  # Ascher's 2D categorization example
        ],
        'narrative': [
            'UR253',  # Non-numeric mnemonic khipu
        ],
        # ... etc
    }
    
    def get_exemplar(self, exemplar_id: str) -> ExemplarPackage:
        """
        Load exemplar with:
        - Full analysis results
        - Literature references
        - Pedagogical notes
        - Known ambiguities
        """
```

### 3. Interactive Glossary (NEW)

```python
class KhipuGlossary:
    """
    Interactive terminology reference.
    Links terms to visualizations and examples.
    """
    
    TERMS = {
        'khipu': {
            'definition': 'Inca recording device made of knotted cords',
            'quechua': 'khipu = "knot"',
            'image': 'assets/khipu_photo.jpg',
            'example_ids': ['AS100', 'UR004'],
        },
        'pendant': {
            'definition': 'Cord hanging from main cord (CORD_LEVEL=1)',
            'visualization': '3d_viewer_highlight_pendants',
            'related_terms': ['main_cord', 'subsidiary'],
        },
        # ... full glossary
    }
```

---

## Implementation Priority

### PHASE 1: Foundation (Weeks 1-2) - CRITICAL

**Goal:** Get basic infrastructure correct

1. ✅ **Fix Schema Issues**
   - Create `value_computation.py` with correct column names
   - Test on 10 khipus manually verified from Ascher's book
   - Document knot → value mapping transparently

2. ✅ **Core Value Computer**
   - Implement Ascher decimal method (primary)
   - Add confidence scoring
   - Preserve raw knot data

3. ✅ **Basic Summation Detector**
   - Pattern Type A (simple hierarchical)
   - Match scoring with tolerance
   - Statistical significance test

**Deliverable:** `cord_values` table populated for all 612 khipus

### PHASE 2: Multi-Method Analysis (Weeks 3-4)

**Goal:** Implement complete analytical framework

4. ✅ **Category Detection**
   - Spatial grouping (gap-based clustering)
   - Color categories
   - Cross-categorization detection

5. ✅ **Advanced Summation**
   - Pattern Types B-E
   - Partial summation support
   - Format classification

6. ✅ **Validation Framework**
   - Monte Carlo null hypothesis tests
   - Corpus comparison
   - Confidence scoring

**Deliverable:** Complete analysis of all 612 khipus with validation

### PHASE 3: Visualization & Interface (Week 5-6)

**Goal:** Make results accessible

7. ✅ **Enhanced 3D Viewer**
   - Integrate computed values
   - Show summation relationships
   - Category highlighting

8. ✅ **Matrix Viewer**
   - Display p_ij structures
   - Show sum verification

9. ✅ **Method Comparison Dashboard**
   - Side-by-side interpretation comparison
   - Educational explanations

**Deliverable:** Complete interactive toolkit

### PHASE 4: Education & Documentation (Week 7-8)

**Goal:** Enable scholarly and student use

10. ✅ **Tutorial Notebooks**
    - 6 progressive learning notebooks
    - Based on real data

11. ✅ **Exemplar Library**
    - Curated validated examples
    - Literature integration

12. ✅ **Documentation**
    - Complete API reference
    - Methodological guide
    - Uncertainty documentation

**Deliverable:** Full pedagogical package

### PHASE 5: Validation & Publication (Week 9-10)

**Goal:** Scientific validation

13. ✅ **Corpus Analysis**
    - Replicate Medrano & Khosla findings
    - Compare to published percentages
    - Identify novel patterns

14. ✅ **Peer Review**
    - Share with khipu scholars
    - Incorporate feedback
    - Validate against archaeological interpretations

15. ✅ **Publication Preparation**
    - Write methodology paper
    - Prepare dataset release
    - Create permanent DOI

**Deliverable:** Published, peer-reviewed toolkit

---

## Success Criteria

### Technical Success
- ✅ All 612 khipus processed without errors
- ✅ >95% of cords with knots have computed values
- ✅ Multiple interpretation methods implemented
- ✅ Statistical validation for all claims
- ✅ Reproducible results (version-controlled, documented)

### Scientific Success
- ✅ Replicate Medrano & Khosla's summation percentages (±3pp)
- ✅ Match Ascher's examples from published work
- ✅ Identify ≥5 new well-validated summation khipus
- ✅ Corpus patterns consistent with archaeological literature
- ✅ Used by ≥3 independent research groups

### Educational Success
- ✅ Tutorials understandable to undergraduates
- ✅ Clear documentation of all analytical choices
- ✅ Transparent uncertainty quantification
- ✅ Multiple learning pathways (notebooks, docs, videos)
- ✅ Adopted in ≥2 university courses

---

## Philosophical Approach

### What This Toolkit IS:

1. **Computational Validation** of archaeological observations
   - Tests Ascher's framework computationally
   - Validates Medrano & Khosla across full corpus
   - Quantifies confidence in interpretations

2. **Multi-Method Framework**
   - Supports multiple interpretation approaches
   - Documents analytical choices explicitly
   - Allows scholarly debate through code

3. **Educational Platform**
   - Makes khipu analysis accessible
   - Teaches computational archaeology
   - Bridges mathematics and anthropology

4. **Open Science Infrastructure**
   - Reproducible, version-controlled
   - Publicly available code and data
   - Transparent methodology

### What This Toolkit IS NOT:

1. **Not "Deciphering" Khipus**
   - We don't claim to know "what they mean"
   - Structural analysis ≠ semantic interpretation
   - Cultural context beyond our scope

2. **Not Single-Method Dogma**
   - No claim that Ascher decimal is "correct"
   - Multiple methods = multiple possibilities
   - Interpretation is scholarly choice

3. **Not Replacement for Archaeology**
   - Computational analysis is ONE lens
   - Physical examination remains primary
   - Cultural/historical context essential

4. **Not Claiming Completeness**
   - Database has gaps (damage, uncertainty)
   - Not all patterns detectable computationally
   - Always provisional, always revisable

---

## Integration with Existing Work

### Relationship to Prior Phases

**Phase 0-2: Keep as documentation**
- Show evolution of understanding
- Document what we learned
- Pedagogical value

**Phase 3: Deprecate, redirect to Phase 10**
- Mark as "preliminary"
- Note limitations
- Link to improved analysis

**Phase 4-9: Enhance with Phase 10 outputs**
- Use new value computation
- Integrate format classification
- Add validation scores

**Phase 10: New comprehensive analysis**
- Multi-method value computation
- Complete summation framework
- Category detection
- Statistical validation

---

## Deliverables Summary

### Code Modules (5 new/rewritten)
1. `src/analysis/value_computation.py` (REWRITE)
2. `src/analysis/summation_patterns.py` (REWRITE)
3. `src/analysis/category_detection.py` (NEW)
4. `src/analysis/validation.py` (ENHANCE)
5. `src/analysis/khipu_analyzer.py` (NEW)

### Data Tables (4 new)
1. `cord_values` - Multi-method value computation
2. `cord_categories` - Category assignments
3. `summation_patterns` - Verified summation relationships
4. `khipu_formats` - Ascher classification

### Visualizations (3 enhanced/new)
1. Enhanced 3D Viewer (with values, summation flows)
2. Matrix Viewer (p_ij, p_ijk display)
3. Method Comparison Dashboard (educational)

### Documentation (6 notebooks + guides)
1. 6 Tutorial Notebooks (progressive learning)
2. Methodological Guide (detailed methods)
3. API Reference (complete function docs)
4. Uncertainty Guide (how to interpret confidence scores)
5. Exemplar Catalog (validated examples)
6. Glossary (interactive terminology)

### Reports (3 new analyses)
1. Corpus Summation Report (replicate Medrano & Khosla)
2. Format Classification Report (Ascher types in corpus)
3. Validation Report (statistical confidence for all findings)

---

## Timeline: 10 Weeks

**Weeks 1-2:** Foundation (value computation, basic summation)  
**Weeks 3-4:** Multi-method analysis (categories, validation)  
**Weeks 5-6:** Visualization & interface  
**Weeks 7-8:** Education & documentation  
**Weeks 9-10:** Validation & publication prep

**Total Effort:** ~400 hours (~40 hrs/week × 10 weeks)

---

## Conclusion

This is not just an enhancement - it's a **paradigm completion**. The toolkit has good technical infrastructure but lacks:

1. **Correct schema usage** (critical bug fixes)
2. **Theoretical alignment** (Ascher framework)
3. **Empirical validation** (Medrano & Khosla replication)
4. **Methodological transparency** (multiple approaches)
5. **Pedagogical clarity** (learning pathways)

The rewrite creates a **scholarly instrument**: rigorous enough for research, transparent enough for peer review, accessible enough for education.

**We're building computational infrastructure for an entire field.**

---

## Next Immediate Steps

1. ☐ Review this plan with stakeholders
2. ☐ Set up test dataset (10-20 exemplar khipus from literature)
3. ☐ Begin Phase 1: value_computation.py with correct schema
4. ☐ Test against Ascher's published examples
5. ☐ Document as we build (literate programming)
6. ☐ Share early results with khipu research community

---

**END OF REWRITE PLAN**
