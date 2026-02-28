# KFG Database Schema Design

**Purpose:** Native schema for Khipu Field Guide data  
**Last Updated:** February 25, 2026

## Design Philosophy

The KFG schema is designed to **match the Excel format's natural structure** rather than force-fit into OKR's schema. This preserves KFG's data model and makes ETL simpler.

## Excel Source Format

Each khipu is one `.xlsx` file with 3 sheets:
- **Khipu:** Metadata (key:value pairs)
- **PrimaryCord:** Primary cord properties (key:value pairs)
- **Cords:** Pendant/subsidiary cords (table format)

## Proposed Tables

### 1. `khipu_metadata`
**Purpose:** One row per khipu with identifying information and provenance

| Column | Type | Source | Notes |
|--------|------|--------|-------|
| `kfg_id` | TEXT PRIMARY KEY | Filename | e.g., "KH0001", "KH0525" |
| `kfg_name` | TEXT | Khipu sheet: KFG_Name | Official KFG identifier |
| `aliases` | TEXT | Khipu sheet: Aliases | Comma-separated (e.g., "LL01,UR176") |
| `contributors` | TEXT | Khipu sheet: Contributors | KFG contributors |
| `kfg_url` | TEXT | Khipu sheet: KFG URL | Link to KFG entry |
| `museum_name` | TEXT | Khipu sheet: Museum Name | Holding institution |
| `museum_number` | TEXT | Khipu sheet: Museum Number | Museum catalog number |
| `museum_city_state` | TEXT | Khipu sheet: Museum City/State | Location |
| `museum_country` | TEXT | Khipu sheet: Museum Country | Country |
| `museum_url` | TEXT | Khipu sheet: Museum URL | Museum collection link |
| `provenance` | TEXT | Khipu sheet: Provenance | Origin location |
| `region` | TEXT | Khipu sheet: Region | Geographic region |
| `creation_date` | TEXT | Khipu sheet: Creation_Date | Estimated date of creation |
| `excel_write_date` | TEXT | Khipu sheet: Excel Write Date | When Excel was created |
| `excel_creator` | TEXT | Khipu sheet: Excel File Creator | Who created the Excel |

**Difference from OKR:** KFG has richer provenance and museum metadata

---

### 2. `primary_cord`
**Purpose:** One row per khipu describing the primary (main) cord

| Column | Type | Source | Notes |
|--------|------|--------|-------|
| `kfg_id` | TEXT PRIMARY KEY | Foreign key to khipu_metadata | |
| `structure` | TEXT | PrimaryCord: Structure | P=plied, B=braid, W=wrapped |
| `thickness` | REAL | PrimaryCord: Thickness | In cm |
| `length` | REAL | PrimaryCord: Length | In cm |
| `color` | TEXT | PrimaryCord: Color | Color code (e.g., "W", "W:MB") |
| `fiber` | TEXT | PrimaryCord: Fiber | e.g., "CN" (cotton) |

**Difference from OKR:** Separated out as distinct entity; KFG has more detailed primary cord measurements

---

### 3. `cords`
**Purpose:** All pendant and subsidiary cords

| Column | Type | Source | Notes |
|--------|------|--------|-------|
| `cord_id` | INTEGER PRIMARY KEY AUTOINCREMENT | Generated | Unique cord identifier |
| `kfg_id` | TEXT NOT NULL | Foreign key | References khipu_metadata |
| `cord_name` | TEXT NOT NULL | Cords: Cord_Name | e.g., "p1", "p6s1", "p10s1s1" |
| `pendant_num` | INTEGER | Parsed from cord_name | Base pendant number |
| `hierarchy_level` | INTEGER | Parsed from cord_name | 0=pendant, 1=subsidiary, 2=sub-subsidiary |
| `parent_cord` | TEXT | Parsed from cord_name | Parent cord name (NULL for pendants) |
| `twist` | TEXT | Cords: Twist | Z/S/U twist direction |
| `attachment` | TEXT | Cords: Attachment | Attachment style (R/V/U) |
| `knots` | TEXT | Cords: Knots | Raw knot string (e.g., "5S(0.0,U),50") |
| `length` | REAL | Cords: Length | Cord length in cm |
| `termination` | TEXT | Cords: Termination | How cord ends |
| `thickness` | REAL | Cords: Thickness | Cord thickness in cm |
| `color` | TEXT | Cords: Color | Color code  |
| `value` | INTEGER | Cords: Value | Numeric value (KFG's computed value) |
| `alt_value` | INTEGER | Cords: Alt_Value | Alternative interpretation |
| `position` | REAL | Cords: Position | Position on primary cord |
| `notes` | TEXT | Cords: Notes | Free-text notes |

**Difference from OKR:** 
- `cord_name` is primary identifier (not synthetic CORD_ID from different system)
- Includes both raw knot string AND computed value
- Has alt_value for alternative interpretations
- Simpler hierarchy parsing (p6s1 is self-documenting)

---

### 4. `knot_clusters`
**Purpose:** Parsed knot clusters (one row per cluster)

| Column | Type | Source | Notes |
|--------|------|--------|-------|
| `cluster_id` | INTEGER PRIMARY KEY AUTOINCREMENT | Generated | |
| `cord_id` | INTEGER NOT NULL | Foreign key | References cords |
| `cluster_ordinal` | INTEGER | Parsed from Knots | Position in sequence (0-based) |
| `knot_type` | TEXT | Parsed from Knots | S, L, E, EE, LL, BL, SP, TF |
| `num_knots` | INTEGER | Parsed from Knots | Count of knots in cluster |
| `position_cm` | REAL | Parsed from Knots | Distance from attachment |
| `direction` | TEXT | Parsed from Knots | Z, S, U |
| `cluster_value` | INTEGER | Parsed from Knots | Numeric value (if encoded in string) |
| `axis_orientation` | TEXT | Parsed from Knots | D/U (if present) |

**Difference from OKR:**  
- More granular than OKR's `knot` table
- Preserves cluster structure (multiple knots per cluster)
- Includes optional axis orientation

---

### 5. `cord_colors`
**Purpose:** Color sequences for multi-color cords

| Column | Type | Source | Notes |
|--------|------|--------|-------|
| `color_id` | INTEGER PRIMARY KEY AUTOINCREMENT | Generated | |
| `cord_id` | INTEGER NOT NULL | Foreign key | References cords |
| `color_code` | TEXT | Parsed from Color | e.g., "W", "MB", "AB" |
| `sequence_ord` | INTEGER | Parsed from Color | Position in color sequence (0-based) |

**Difference from OKR:** Normalized table for multi-color cords (W:MB:BG)

---

## Schema Comparison: KFG vs OKR

| Aspect | KFG Schema | OKR Schema |
|--------|------------|------------|
| **Khipu table** | `khipu_metadata` (15 fields) | `khipu_main` (~8 fields) |
| **Primary cord** | Separate `primary_cord` table | Combined with khipu_main |
| **Cord identifiers** | String names (`p6s1`) | Numeric IDs |
| **Hierarchy** | Self-documenting names | Separate ATTACHED_TO, CORD_LEVEL columns |
| **Knots** | Both raw string + parsed clusters | Fully normalized knot table |
| **Colors** | Both raw code + normalized table | Single color field |
| **Values** | Both KFG-computed + alternative | Single numeric field |
| **Provenance** | Rich museum + geographic metadata | Basic provenance |

## Key Advantages of KFG-Native Schema

1. **Simpler ETL:** Direct mapping from Excel→SQL (no translation layer needed)
2. **Preserves Semantics:** `p6s1` is more meaningful than `CORD_ID=4728`
3. **Richer Metadata:** Full museum provenance and KFG documentation links
4. **Dual Values:** Preserves both KFG's computed value and alternatives
5. **Raw + Parsed:** Keeps original knot strings for verification + parsed clusters for analysis
6. **Query-Friendly:** `WHERE cord_name LIKE 'p6%'` finds all subsidiaries of pendant 6

## Implementation Priority

**Phase 0 - Core Tables (Required for basic analysis):**
1. `khipu_metadata` ✓ Required
2. `cords` ✓ Required  
3. `knot_clusters` ✓ Required

**Phase 1 - Enhanced Tables (Nice to have):**
4. `primary_cord` ⚡ Adds primary cord analysis capability
5. `cord_colors` ⚡ Enables color pattern analysis

## Next Steps

1. **Review & Approve:** Does this schema make sense for KFG data?
2. **Build Importer:** Create `scripts/build_kfg_database.py`
3. **Test on Sample:** Import 10 khipus and validate
4. **Full Import:** Process all 650+ khipus
5. **Update Extractors:** Adapt existing analysis scripts to query this schema

---

**Decision Needed:** Should we proceed with this KFG-native schema, or match OKR's structure for compatibility?
