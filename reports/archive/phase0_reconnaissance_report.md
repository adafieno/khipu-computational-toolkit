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

## Next Steps

Phase 0 reconnaissance enables:
- ✅ **Phase 1:** Numeric data extraction and validation pipeline
- ✅ **Phase 2:** Comprehensive extraction infrastructure for hierarchical, knot, and color data
- ✅ **Phase 3:** Statistical hypothesis testing framework
- ✅ **Phase 4+:** Advanced pattern discovery and structural analysis

## Assessment: Research Viability

**Rating: 8.5/10 - Highly Viable**

**Positive Indicators:**
- ✅ Sufficient volume (612 analyzed khipus, 45K cords)
- ✅ Structured hierarchy (graph-ready)
- ✅ Rich multi-modal data (numeric, color, spatial)
- ✅ Well-documented encoding conventions
- ✅ Geographic diversity for provenance analysis
- ✅ Domain expert curation (high quality)

**Challenges:**
- ⚠️ Limited external validation sources
- ⚠️ Missing data in ~15-20% of records
- ⚠️ Sparse temporal information
- ⚠️ Need domain expert validation for numeric conventions
- ⚠️ Small dataset by modern ML standards (but adequate for symbolic methods)

---

## References

- OKR Team. (2021). *The Open Khipu Repository* (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5037551
- Ascher, M., & Ascher, R. (1997). *Mathematics of the Incas: Code of the Quipu*. Dover Publications. ISBN 0-486-29554-0.
- Locke, L. L. (1912). *The Ancient Quipu, a Peruvian Knot Record*. American Anthropologist, 14(2), 325-332.
- Urton, G. (2003). *Signs of the Inka Khipu: Binary Coding in the Andean Knotted-String Records*. University of Texas Press. ISBN 978-0-292-78540-4.

---
