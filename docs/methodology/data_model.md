# Open Khipu Repository - Data Model Documentation

## Overview

The OKR database uses a relational model to capture the hierarchical, spatial, and symbolic properties of khipus. This document provides detailed semantics for the core tables and their relationships.

## Table Relationships

```
khipu_main (619 original, 612 analyzed)
    ↓ 1:N
primary_cord (633)
    ↓ 1:N
cord (54,403)
    ↓ 1:N
    ├─→ knot (110,677)
    │       ↓ grouped by
    │   knot_cluster (60,169)
    │
    ├─→ ascher_cord_color (56,306)
    │
    └─→ cord_cluster (14,847)
```

## Core Tables

### khipu_main
**Purpose:** Top-level khipu metadata and provenance

**Key Fields:**
- `KHIPU_ID`: Primary key (INTEGER)
- `PROVENANCE`: Geographic origin (TEXT)
- `REGION`: Broader geographic classification (TEXT) - joins to regions_dc
- `MUSEUM_NAME`: Current location (TEXT)
- `MUSEUM_NUM`: Museum catalog number (TEXT)
- `INVESTIGATOR_NUM`: Original researcher's ID (TEXT) - legacy naming
- `OKR_NUM`: Standardized OKR identifier (TEXT) - format: "KH####"
- `EARLIEST_AGE`, `LATEST_AGE`: Dating information (TEXT) - often sparse
- `NICKNAME`: Informal name (TEXT)
- `COMPLETE`: Completeness indicator (REAL) - 0.0 or NULL for incomplete
- `DUPLICATE_FLAG`, `DUPLICATE_ID`: Track duplicate entries (REAL)
- `NOTES`: Free-text observations (TEXT)

**Data Completeness:** 84% have complete metadata; 16% missing dates and condition

**Critical IDs:**
- Internal processing: use `KHIPU_ID`
- External references: prefer `OKR_NUM` (standardized)
- Historical citations: `INVESTIGATOR_NUM` (researcher-specific)

### primary_cord
**Purpose:** Main structural cord from which pendants hang

**Key Fields:**
- `PCORD_ID`: Primary key (INTEGER)
- `KHIPU_ID`: Parent khipu (INTEGER) - FK to khipu_main
- `STRUCTURE`: Construction type (TEXT) - FK to structure_dc
  - P = plied cord (most common)
  - B = braid
  - W = wrapped
- `PCORD_LENGTH`: Total length in cm (REAL)
- `THICKNESS`: Diameter in mm (REAL)
- `FIBER`: Material type (TEXT) - FK to fiber_dc
  - CN = cotton
  - CL = cotton + llama mix (?)
  - A = alpaca
  - L = llama
- `TWIST`: Spin direction (TEXT)
  - S = left-twisted
  - Z = right-twisted
- `BEGINNING`, `TERMINATION`: Cord end treatments (TEXT) - FK to termination_dc
  - K = knotted
  - B = broken
  - C = cut
  - T = tassel
  - R = ravelled
- `ATTACHED_TO_ID`: Parent cord if this is a branch (REAL) - mostly NULL
- `NOTES`: Observations, often in Spanish (TEXT)

**Semantics:**
- Typically 1 primary cord per khipu (some have branches)
- Spatial reference: position 0.0 starts at BEGINNING
- All pendant cords attach to this reference line

### cord
**Purpose:** Individual pendant, subsidiary, and top cords

**Key Fields:**

**Identity:**
- `CORD_ID`: Primary key (INTEGER) - globally unique
- `KHIPU_ID`: Parent khipu (INTEGER) - FK to khipu_main
- `INVESTIGATOR_CORD_NUM`: Scholar's original numbering (INTEGER) - may have gaps

**Hierarchy:**
- `PENDANT_FROM`: Parent primary cord ID (INTEGER) - FK to primary_cord.PCORD_ID
- `ATTACHED_TO`: Parent cord ID (INTEGER) - FK to cord.CORD_ID
  - If pendant: ATTACHED_TO = 0 or NULL
  - If subsidiary: ATTACHED_TO = parent CORD_ID
- `CORD_LEVEL`: Depth in tree (INTEGER)
  - 1 = pendant (hangs from primary)
  - 2 = subsidiary (hangs from pendant)
  - 3 = sub-subsidiary, etc.
- `ATTACHMENT_TYPE`: How cord is attached (TEXT)
  - V = verso (standard attachment)
  - R = recto (reverse)
  - U = undetermined

**Spatial:**
- `ATTACH_POS`: Position along parent cord in cm (REAL)
- `CORD_LENGTH`: Length of this cord in cm (REAL)

**Physical:**
- `THICKNESS`: Diameter in mm (REAL)
- `FIBER`: Material (TEXT) - FK to fiber_dc
- `TWIST`: Spin direction (TEXT) - S or Z
- `TERMINATION`: End treatment (TEXT) - FK to termination_dc

**Clustering:**
- `CLUSTER_ID`: Member of which cluster (INTEGER) - FK to cord_cluster.CLUSTER_ID
- `CLUSTER_ORDINAL`: Position within cluster (INTEGER)
- `CORD_ORDINAL`: Sequential position among all cords (INTEGER)

**Canuto (beads):**
- `CANUTO_ID`, `CANUTO_ORDINAL`: If cord passes through decorative beads (INTEGER)

**Classification:**
- `CORD_CLASSIFICATION`: Functional category (TEXT) - sparse
- `CORD_NOTES`: Free-text observations (TEXT)

**Critical Semantics:**
- **Hierarchy Rule:** If `CORD_LEVEL = 1`, cord is a pendant from primary
- **Hierarchy Rule:** If `CORD_LEVEL > 1`, cord is subsidiary; `ATTACHED_TO` points to parent
- **Missing `ATTACHED_TO`:** ~17% missing - likely primary pendants with CORD_LEVEL=1
- **Canuto Data:** 83% have NULL canuto fields (canutos are rare decorative elements)

### knot
**Purpose:** Individual knots encoding numeric or symbolic information

**Key Fields:**

**Identity:**
- `KNOT_ID`: Primary key (INTEGER) - globally unique
- `CORD_ID`: Parent cord (INTEGER) - FK to cord.CORD_ID

**Knot Type:**
- `TYPE_CODE`: Knot classification (TEXT) - FK to knot_type_dc
  - **L** = long (multiple wraps) → typically tens position
  - **E** = figure-eight → typically units position
  - **S** = single (overhand) → typically hundreds position
  - **8** = figure-eight variant (?)
  - **EE** = double figure-eight (?)
  - **LL** = double long (?)
  - **X** = damaged/indeterminate

**Numeric Value:**
- `knot_value_type`: Interpreted numeric value (INTEGER)
  - Appears to be the actual number represented (1-9 typically)
  - Examples: 10, 8, 1, 5, etc.
  - **Question:** Is this pre-computed or raw encoding?
- `NUM_TURNS`: Number of wraps for long knots (REAL)
  - Typically 1-9 for position encoding
  - NULL for non-long knots (23% of dataset)

**Spatial:**
- `DIRECTION`: Twist direction (TEXT)
  - S = left-twisted
  - Z = right-twisted
- `AXIS_ORIENTATION`: Knot axis relative to cord (TEXT)
  - AXD = axis diagonal (?)
  - 18% missing

**Clustering:**
- `CLUSTER_ID`: Member of knot cluster (INTEGER) - FK to knot_cluster.CLUSTER_ID
- `KNOT_ORDINAL`: Position within cord (REAL) - 23% missing
- `CLUSTER_ORDINAL`: Position within cluster (REAL) - 23% missing

**Critical Semantics:**
- **Positional Decimal Encoding:** Knots encode base-10 numbers
  - Position along cord determines place value
  - Knot type determines digit (via NUM_TURNS or knot_value_type)
  - Typical pattern: [S knots] [L knots] [E knots] = [hundreds][tens][units]
- **Missing Ordinals:** 23% lack KNOT_ORDINAL - may need to be computed from positions
- **knot_value_type Interpretation:** Requires domain validation
  - Is this the raw encoding or already decoded value?
  - How does it relate to TYPE_CODE and NUM_TURNS?

### ascher_cord_color
**Purpose:** Color information along cord segments

**Key Fields:**

**Identity:**
- `color_id`: Primary key (INTEGER)
- `CORD_ID`: Parent cord (INTEGER) - FK to cord.CORD_ID
- `KHIPU_ID`: Parent khipu (INTEGER)
- `PCORD_FLAG`: Is this a primary cord color? (INTEGER) - 1=yes, 0=no

**Color Encoding:**
- `COLOR_CD_1` through `COLOR_CD_5`: Up to 5 color codes (TEXT)
  - FK to ascher_color_dc.AS_COLOR_CD
  - Examples: AB, MB, KB, W, B, PR, YB, VB
- `OPERATOR_1` through `OPERATOR_5`: Color mixing operators (TEXT)
  - `-` = barber pole (twisted together)
  - `:` = mottled (irregular)
  - `*` = special (see databooks)
- `FULL_COLOR`: Concatenated representation (TEXT)
  - Example: "KB:W" = olive brown mottled with white
- `PIGMENTATION_CD_1` through `CD_5`: Pigmentation notes (TEXT)
  - Mostly empty strings

**Spatial:**
- `COLOR_RANGE`: Which range # for this cord (INTEGER)
  - Cords can have multiple color segments
- `RANGE_BEG`, `RANGE_END`: Start/end position in cm (REAL)

**Critical Semantics:**
- **Multi-Color Cords:** Some cords have 2-5 colors mixed or in sequence
- **Color Ranges:** Track color changes along cord length
- **Color Code System:** Ascher notation (64 codes) based on ISCC-NBS standard
  - Includes RGB mappings in ascher_color_dc
- **Operator Semantics:**
  - No operator = single solid color
  - `:` = irregular mixing within ply
  - `-` = structured barber-pole pattern
  - `*` = see researcher's notes (special cases)

### ascher_color_dc
**Purpose:** Color code dictionary with RGB values

**Key Fields:**
- `AS_COLOR_CD`: Color code (TEXT) - Primary key
- `COLOR_DESCR`: Human-readable description (TEXT)
- `ISCC_NBS_NUM`: ISCC-NBS color standard number (INTEGER)
- `R_DEC`, `G_DEC`, `B_DEC`: RGB values 0.0-1.0 (REAL)
- `COLOR`: Base color category (TEXT)
  - B = brown, G = green, Y = yellow, R = red, etc.
- `INTENSITY`: Darkness level 1-5 (INTEGER)
  - 1 = light, 5 = dark/black

**Usage:** 
- Maps symbolic color codes to visualizable RGB
- Enables computational color similarity analysis
- Based on standardized color science (ISCC-NBS)

### cord_cluster
**Purpose:** Groups of adjacent cords forming structural patterns

**Key Fields:**

**Identity:**
- `CLUSTER_ID`: Primary key (INTEGER)
- `CORD_ID`: Parent cord (primary cord to which cluster is attached) (INTEGER)
- `KHIPU_ID`: Parent khipu (INTEGER)

**Cluster Definition:**
- `ORDINAL`: Cluster sequence number (INTEGER)
- `CLUSTER_LEVEL`: Depth in hierarchy (INTEGER)
- `NUM_CORDS`: Count of cords in cluster (INTEGER)
- `BEG_CORD`, `END_CORD`: Cord ordinal range (INTEGER)
- `BEG_INV_CORD`, `END_INV_CORD`: Investigator numbering range (INTEGER)

**Spatial:**
- `START_POSITION`, `END_POSITION`: Positions on parent cord (cm) (REAL)
- `SPACING`: Average inter-cord spacing (cm) (REAL)

**Classification:**
- `GROUPING_CLASS`: Functional classification (TEXT) - FK to grouping_class_dc
  - T = top cords (hang opposite direction)
  - PA = loop pendants (attached at both ends)
  - M = markers (tassels, decorations)
  - Mostly NULL (99%)

**Critical Semantics:**
- **Scholar-Identified:** These clusters are researcher annotations, not algorithmic
- **Hierarchical:** Clusters can contain sub-clusters
- **Motif Discovery:** Valuable for finding recurring patterns
- **Sparse Classification:** Only 1% have GROUPING_CLASS - most are structural groupings

### knot_cluster
**Purpose:** Groups of knots encoding positional decimal numbers

**Key Fields:**

**Identity:**
- `CLUSTER_ID`: Primary key (INTEGER)
- `CORD_ID`: Parent cord (INTEGER) - FK to cord.CORD_ID

**Cluster Definition:**
- `ORDINAL`: Sequence number on cord (INTEGER)
- `NUM_KNOTS`: Count of knots in cluster (INTEGER)

**Spatial:**
- `START_POS`, `END_POS`: Positions along cord (cm) (REAL)
  - 20.5% missing END_POS

**Numeric:**
- `TOTAL_VALUE`: Sum of all knots in cluster (INTEGER)
  - Pre-computed by researchers
  - 20.5% missing (same as END_POS)
  - **Critical for validation:** Should match sum of knot_value_type

**Critical Semantics:**
- **Decimal Register:** Each cluster = one decimal number
- **Position Ordering:** START_POS determines which number comes first
- **Validation Target:** TOTAL_VALUE should be computable from constituent knots
- **Missing Values:** ~20% incomplete - may need reconstruction

## Derived Relationships

### Hierarchy Extraction
```sql
-- Get all pendants for a khipu
SELECT * FROM cord 
WHERE KHIPU_ID = ? AND CORD_LEVEL = 1;

-- Get subsidiaries of a cord
SELECT * FROM cord
WHERE ATTACHED_TO = ? AND CORD_LEVEL > 1;

-- Build full tree recursively via ATTACHED_TO
```

### Numeric Reconstruction
```sql
-- Get all knots for a cord in positional order
SELECT * FROM knot
WHERE CORD_ID = ?
ORDER BY KNOT_ORDINAL, CLUSTER_ORDINAL;

-- Get pre-computed cluster totals
SELECT ORDINAL, TOTAL_VALUE 
FROM knot_cluster
WHERE CORD_ID = ?
ORDER BY ORDINAL;
```

### Color Patterns
```sql
-- Get full color profile for a cord
SELECT COLOR_RANGE, FULL_COLOR, RANGE_BEG, RANGE_END
FROM ascher_cord_color
WHERE CORD_ID = ?
ORDER BY COLOR_RANGE;
```

## Data Quality Flags

### Known Issues
1. **Missing Ordinals:** 23% of knots lack KNOT_ORDINAL
   - Impact: Positional ordering may need reconstruction
   - Mitigation: Use START_POS from knot_cluster

2. **Incomplete Clusters:** 20% of knot_clusters missing TOTAL_VALUE
   - Impact: Can't validate numeric encoding
   - Mitigation: Compute from constituent knots

3. **Sparse Metadata:** 16% of khipus missing provenance
   - Impact: Geographic analysis limited
   - Mitigation: Use available REGION field

4. **Attachment Ambiguity:** 17% of cords missing ATTACHED_TO
   - Impact: Hierarchy reconstruction uncertain
   - Mitigation: Infer from CORD_LEVEL and PENDANT_FROM

### Integrity Checks Needed
- [ ] Verify CORD_LEVEL consistency with ATTACHED_TO
- [ ] Check for cycles in cord hierarchy
- [ ] Validate TOTAL_VALUE against sum of knots
- [ ] Confirm KHIPU_ID consistency across tables
- [ ] Check color code validity (all codes in ascher_color_dc)
- [ ] Verify spatial coherence (positions non-negative, orderings consistent)

## Next Steps
1. Implement data loaders respecting these semantics
2. Build validation suite checking integrity constraints
3. Document ambiguous cases requiring domain expert clarification
4. Create example queries for common access patterns
