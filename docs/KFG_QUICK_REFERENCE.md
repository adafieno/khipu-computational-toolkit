# KFG Integration Quick Reference

**Last Updated:** February 25, 2026

## 📋 Quick Facts

- **KFG Files:** 700+ Excel files (`.xlsx`) in `data/kfg/KFG/KFG/`
- **File naming:** KH#### format (e.g., KH0001.xlsx, KH0525.xlsx)
- **Additional formats:** Also includes CM, HM, KT, RA, RS prefixes
- **Parser library:** `src/extraction/kfg_parsers.py`

## 🚀 Getting Started

Test the parsers to see KFG format parsing in action:
```bash
python src/extraction/kfg_parsers.py
```
This validates that cord hierarchy, knot, and color parsers work correctly

## 📊 Data Format Cheat Sheet

### Knot Format
```
5S(0.0,U),50;3S(7.0,Z),30;6L(23.5,Z),6
│││ │   │  │
││└─Type (S=Single, L=Long)
│└──Count
└───Position (cm), Direction, Value
```

### Cord Hierarchy
```
p1       Level 0 (pendant)
p6s1     Level 1 (subsidiary of p6)
p10s1s1  Level 2 (sub-subsidiary of p10s1)
```

### Color Codes
```
W        Single color (White)
W:MB     Multi-color (White with Mottled Brown)
```

## 🔧 Parser Library

**File:** `src/extraction/kfg_parsers.py`

**Functions:**
- `parse_cord_hierarchy(cord_name)` → Dict with hierarchy info
- `parse_kfg_knots(knot_string)` → List of knot clusters
- `parse_kfg_color(color_string)` → List of color codes
- `parse_kfg_metadata(khipu_df)` → Dict of metadata
- `parse_primary_cord(primary_cord_df)` → Dict of primary cord properties
- `compute_cord_value(knots)` → Dict with value and confidence

**Example Usage:**
```python
from src.extraction.kfg_parsers import parse_cord_hierarchy, parse_kfg_knots

# Parse cord hierarchy
hierarchy = parse_cord_hierarchy("p6s1")
print(f"Level: {hierarchy['level']}, Parent: {hierarchy['parent']}")
# Output: Level: 1, Parent: p6

# Parse knots
knots = parse_kfg_knots("5S(0.0,U),50;3S(7.0,Z),30")
total_value = sum(k['cluster_value'] for k in knots)
print(f"Total value: {total_value}")
# Output: Total value: 80
```

## 🎯 Example: Process Single Khipu

```python
import pandas as pd
from pathlib import Path
from src.extraction.kfg_parsers import (
    parse_kfg_metadata, parse_primary_cord, 
    parse_cord_hierarchy, parse_kfg_knots, parse_kfg_color
)

# Load KH0001
kfg_file = Path("data/kfg/KFG/KFG/KH0001.xlsx")

# Read sheets
khipu_df = pd.read_excel(kfg_file, sheet_name='Khipu')
primary_cord_df = pd.read_excel(kfg_file, sheet_name='PrimaryCord')
cords_df = pd.read_excel(kfg_file, sheet_name='Cords')

# Parse metadata
metadata = parse_kfg_metadata(khipu_df)
print(f"Khipu: {metadata['kfg_name']}")
print(f"Provenance: {metadata['provenance']}")

# Parse primary cord
primary = parse_primary_cord(primary_cord_df)
print(f"Primary cord: {primary['length']}cm, {primary['color']}")

# Parse first few cords
for _, row in cords_df.head(5).iterrows():
    cord_name = row['Cord_Name']
    hierarchy = parse_cord_hierarchy(cord_name)
    knots = parse_kfg_knots(row['Knots'])
    colors = parse_kfg_color(row['Color'])
    
    value = sum(k['cluster_value'] for k in knots)
    print(f"{cord_name}: Level {hierarchy['level']}, Value {value}, Colors {[c['color_code'] for c in colors]}")
```

## 📚 File Structure

```
khipu-computational-toolkit/
├── data/
│   └── kfg/
│       └── KFG/
│           └── KFG/
│               ├── KH0001.xlsx  ← 700+ khipu files
│               ├── KH0002.xlsx
│               ├── CM009.xlsx
│               ├── HM45419A.xlsx
│               ├── RA001.xlsx
│               └── ...
├── docs/
│   └── KFG_QUICK_REFERENCE.md  ← This file
└── src/
    └── extraction/
        ├── kfg_parsers.py      ← KFG format parsers
        ├── khipu_loader.py     ← General khipu loading
        ├── cord_extractor.py   ← Cord extraction
        ├── knot_extractor.py   ← Knot extraction
        └── color_extractor.py  ← Color extraction
```
