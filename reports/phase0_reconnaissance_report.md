# Open Khipu Repository - Phase 0 Reconnaissance Report

**Generated:** December 30, 2025  
**Database:** khipu.db (SQLite)

## Executive Summary

The OKR database contains **619 khipus** in the original dataset, with **612 valid khipus** containing cord data used for analysis. Detailed structural, numeric, and color information is encoded across 24 tables with over 280,000 records. The data is well-structured for hierarchical graph representation.

## Database Overview

### Core Data Volume
- **619 unique khipus** in original dataset (612 with cord data analyzed)
- **54,403 cords** (cord)
- **110,677 knots** (knot)
- **56,306 color records** (ascher_cord_color)
- **14,847 cord clusters** (cord_cluster)
- **60,169 knot clusters** (knot_cluster)

### Key Tables for Graph Construction

#### 1. **khipu_main** - Metadata Layer
**Fields:** 
- KHIPU_ID (primary key)
- PROVENANCE, REGION (geographic)
- MUSEUM_NAME, MUSEUM_NUM (curation)
- INVESTIGATOR_NUM, OKR_NUM (identification)
- NOTES (narrative descriptions)

**Data Quality:**
- Complete records: 84.3% (522/619 original dataset)
- Missing values primarily in dates and condition fields

#### 2. **primary_cord** - Main Cord Structure
**Fields:**
- PCORD_ID, KHIPU_ID
- STRUCTURE (plied/braid/wrapped)
- THICKNESS, PCORD_LENGTH
- FIBER, TWIST (S/Z direction)
- BEGINNING, TERMINATION

**Count:** 633 primary cords  
**Data Quality:** 
- 93.2% missing ATTACHED_TO_ID (expected - most are standalone)
- Good coverage of physical attributes

#### 3. **cord** - Pendant/Subsidiary Hierarchy ⭐
**Fields:**
- CORD_ID (primary), KHIPU_ID
- PENDANT_FROM, ATTACHED_TO (hierarchy)
- CORD_LEVEL (depth in tree)
- ATTACHMENT_TYPE, ATTACH_POS
- CORD_LENGTH, THICKNESS
- FIBER, TWIST, TERMINATION
- CLUSTER_ID, CORD_ORDINAL

**Count:** 54,403 cords  
**Critical for Graph:** PENDANT_FROM + ATTACHED_TO define parent-child relationships

**Data Quality Issues:**
- 16.9% missing TWIST_ANGLE, ATTACHED_TO
- 17.0% missing INVESTIGATOR_CORD_NUM (canuto cords)
- Generally excellent for structural analysis

#### 4. **knot** - Numeric Encoding Layer ⭐⭐
**Fields:**
- KNOT_ID, CORD_ID
- TYPE_CODE (L=long, E=figure-eight, S=single, etc.)
- DIRECTION (S/Z twist)
- knot_value_type (numeric value: 1, 8, 10, etc.)
- NUM_TURNS (for long knots)
- CLUSTER_ID, KNOT_ORDINAL
- AXIS_ORIENTATION

**Count:** 110,677 knots  
**Critical for Numeric Constraint Solving**

**Data Quality:**
- 23.2% missing KNOT_ORDINAL, NUM_TURNS, CLUSTER_ORDINAL
- 18.1% missing AXIS_ORIENTATION
- Excellent coverage for type and direction

**Knot Types Available:**
```
L  = long (tens position)
E  = figure eight (units)
S  = single (hundreds)
8  = figure eight variant
EE = double figure eight
LL = double long
```

#### 5. **ascher_cord_color** - Color Encoding ⭐
**Fields:**
- color_id, CORD_ID, KHIPU_ID
- COLOR_CD_1 through COLOR_CD_5 (up to 5 colors)
- OPERATOR_1 through OPERATOR_5 (color mixing: -, :, *, etc.)
- FULL_COLOR (concatenated representation)
- COLOR_RANGE, RANGE_BEG, RANGE_END (position along cord)
- PIGMENTATION_CD_1 through CD_5

**Count:** 56,306 color records  
**64 distinct color codes** in ascher_color_dc with RGB mappings

**Color Examples:**
- AB = light brown
- MB = grayish yellowish brown
- KB = olive brown  
- W = white
- B = moderate yellowish brown
- PR = ?purple/red?

**Color Operators:**
- `-` = barber pole (twisted together)
- `:` = mottled (irregular appearance)
- `*` = special (see databooks)

#### 6. **cord_cluster** - Grouping Patterns
**Fields:**
- CLUSTER_ID, CORD_ID, KHIPU_ID
- ORDINAL, CLUSTER_LEVEL
- START_POSITION, END_POSITION, SPACING
- BEG_CORD, END_CORD, NUM_CORDS
- GROUPING_CLASS (T=top cords, PA=loop pendants, M=marker)

**Count:** 14,847 clusters  
**Critical for Pattern Discovery:** Identifies recurring structural motifs

#### 7. **knot_cluster** - Positional Numeric Groups
**Fields:**
- CLUSTER_ID, CORD_ID
- START_POS, END_POS
- TOTAL_VALUE (sum of knots)
- NUM_KNOTS, ORDINAL

**Count:** 60,169 knot clusters  
**Critical for Numeric Analysis:** Pre-computed totals

### Data Dictionary Tables (Well-Defined)

- **fiber_dc**: 11 fiber types (cotton, alpaca, llama, etc.)
- **knot_type_dc**: 8 knot types with descriptions
- **termination_dc**: 7 termination types (broken, cut, ravelled, knotted)
- **color_operator_dc**: 6 color operators
- **ascher_color_dc**: 64 color codes with ISCC-NBS numbers and RGB values
- **grouping_class_dc**: 12 grouping classifications
- **structure_dc**: 3 structure types (plied, braid, wrapped)
- **regions_dc**: 53 provenances with north/south coding

## Data Model Assessment

### Strengths ✓

1. **Hierarchical Structure is Explicit**
   - `PENDANT_FROM` and `ATTACHED_TO` fields define tree structure
   - `CORD_LEVEL` provides depth information
   - Perfect for graph conversion

2. **Numeric Conventions Well-Encoded**
   - Knot types (L/E/S) map to positional decimal
   - `knot_value_type` field directly encodes numeric values
   - Position information available for validation

3. **Rich Color Information**
   - Multi-color cords supported (up to 5 colors)
   - Mixing operators documented
   - RGB values available for visualization
   - Position ranges for color changes along cord

4. **Spatial Information Preserved**
   - Attachment positions
   - Cord lengths
   - Knot positions along cords
   - Cluster spacing measurements

5. **Provenance Well-Documented**
   - Geographic regions
   - Museum attributions
   - Investigator tracking
   - Creation/change history

### Limitations ⚠️

1. **Missing Data Patterns**
   - ~17% of cords missing some structural attributes
   - ~23% of knots missing ordinal/turn counts
   - ~15-20% of khipu metadata incomplete

2. **Sparse Canuto Data**
   - Only 70 canuto_cluster records
   - 465 canutito_color records
   - 994 canuto_color records
   - Most khipus don't have canuto (bead-like) encoding

3. **No Built-in Similarity Metrics**
   - Requires computation
   - No pre-computed nearest neighbors

4. **Naming Convention Complexity**
   - Multiple ID systems (INVESTIGATOR_NUM, OKR_NUM, MUSEUM_NUM)
   - Some duplication flags present

5. **Limited Provenance Constraints**
   - ~16% missing region/provenance
   - Date information sparse (most "0000-00-00")

## Recommendations for Phase 1

### Immediate Actions

1. **Build Core Data Extraction Pipeline**
   ```
   src/extraction/
   ├── khipu_loader.py       # Load khipu metadata
   ├── cord_extractor.py     # Extract cord hierarchy
   ├── knot_extractor.py     # Extract knot data
   ├── color_extractor.py    # Extract color information
   └── cluster_extractor.py  # Extract clustering patterns
   ```

2. **Create Graph Conversion Module**
   ```
   src/graph/
   ├── graph_builder.py      # NetworkX graph construction
   ├── node_features.py      # Feature engineering
   ├── edge_attributes.py    # Relationship encoding
   └── graph_validator.py    # Validate hierarchy integrity
   ```

3. **Implement Numeric Constraint Solver**
   ```
   src/numeric/
   ├── conventions.py        # Encode positional decimal rules
   ├── validator.py          # Check numeric consistency
   ├── classifier.py         # Numeric vs non-numeric cords
   └── uncertainty.py        # Confidence scoring
   ```

### Critical Questions to Resolve

1. **Knot Value Interpretation**
   - Is `knot_value_type` already the interpreted decimal value?
   - How to handle missing `NUM_TURNS` for long knots?
   - What do knot_value_type values like 10, 8, 1 represent?

2. **Color Semantics**
   - Which color codes are most reliable?
   - How to handle multi-color (operator) cords?
   - Should we treat color ranges separately?

3. **Hierarchy Validation**
   - How to handle cords with missing `ATTACHED_TO`?
   - Are `PENDANT_FROM` and `ATTACHED_TO` redundant or complementary?
   - What does `CORD_LEVEL` represent precisely?

4. **Clustering Ground Truth**
   - Are `cord_cluster` groupings scholar-identified or algorithmic?
   - Should we trust existing clusters or re-discover?
   - What is the GROUPING_CLASS taxonomy?

### Data Quality Priorities

1. **Validate hierarchical integrity** (check for cycles, orphans)
2. **Cross-reference ID systems** (INVESTIGATOR_NUM ↔ OKR_NUM)
3. **Assess color code consistency** across contributors
4. **Check numeric encoding completeness** per khipu
5. **Identify high-quality "gold standard" khipus** for testing

## Next Steps (Phase 1 - Week 1)

1. ✅ Set up Python environment with pandas, networkx, sqlite3
2. ⬜ Create `src/extraction/khipu_loader.py` - load all khipus with metadata
3. ⬜ Create `src/extraction/cord_extractor.py` - build cord hierarchy
4. ⬜ Create initial Jupyter notebook: `notebooks/01_data_exploration.ipynb`
5. ⬜ Generate sample visualizations of 3-5 khipus
6. ⬜ Document data model ambiguities in `docs/methodology/`
7. ⬜ Identify "test set" khipus (complete, well-documented, diverse)

## Assessment: Viability for AI Decipherment

**Rating: 8.5/10 - Highly Viable**

**Positive Indicators:**
- ✅ Sufficient volume (612 analyzed khipus, 45K cords)
- ✅ Structured hierarchy (graph-ready)
- ✅ Rich multi-modal data (numeric, color, spatial)
- ✅ Well-documented encoding conventions
- ✅ Geographic diversity for provenance analysis
- ✅ Domain expert curation (high quality)

**Challenges:**
- ⚠️ Limited external ground truth (no confirmed decipherments)
- ⚠️ Missing data in ~15-20% of records
- ⚠️ Sparse temporal information
- ⚠️ Need domain expert validation for numeric conventions
- ⚠️ Small dataset by modern ML standards (but adequate for symbolic methods)

**Recommendation:** The proposed approach is well-aligned with the data structure. Proceed with Phase 1 infrastructure development.
