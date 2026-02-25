# KFG Integration Quick Reference

**Last Updated:** February 22, 2026

## 📋 Quick Facts

- **KFG Files:** 709 Excel files (`.xlsx`)
- **Overlap with OKR:** 616 khipus (99.5%)
- **KFG-Exclusive:** 66 khipus
- **Implementation Status:** Parsers complete, extractors needed
- **Estimated Effort:** 28-39 hours remaining

## 🚀 Getting Started

### 1. Investigate the Data
```bash
python scripts/investigate_kfg_format.py
```
This script examines 50 KFG files and reports on format, completeness, and overlap with OKR.

### 2. Test the Parsers
```bash
python src/extraction/kfg_parsers.py
```
Validates that cord hierarchy, knot, and color parsers work correctly.

### 3. Read the Documentation
- **Start here:** [KFG_INTEGRATION_SUMMARY.md](KFG_INTEGRATION_SUMMARY.md) - Executive summary
- **Technical details:** [KFG_INTEGRATION_ASSESSMENT.md](KFG_INTEGRATION_ASSESSMENT.md) - Full assessment
- **Format specs:** [KFG_INVESTIGATION_FINDINGS.md](KFG_INVESTIGATION_FINDINGS.md) - Detailed findings

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

## 🔧 Available Tools

### Investigation Script
**File:** `scripts/investigate_kfg_format.py`

**What it does:**
- Scans KFG Excel files
- Reports metadata completeness
- Analyzes cord hierarchy patterns
- Examines knot and color formats
- Checks overlap with OKR database

**Usage:**
```bash
python scripts/investigate_kfg_format.py
```

### Parser Library
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

## 📝 Implementation Checklist

### ✅ Phase 0: Investigation (Complete)
- [x] Examine KFG file structure
- [x] Document format specifications
- [x] Check overlap with OKR
- [x] Identify parsing requirements

### ✅ Phase 1: Core Parsers (Complete)
- [x] Implement cord hierarchy parser
- [x] Implement knot parser
- [x] Implement color parser
- [x] Implement metadata parsers
- [x] Test all parsers

### ⬜ Phase 2: Extractors (To Do)
- [ ] Build KFGKhipuLoader
- [ ] Build KFGCordExtractor
- [ ] Build KFGKnotExtractor
- [ ] Build KFGColorExtractor
- [ ] Process all 709 files
- [ ] Export to CSV format

### ⬜ Phase 3: Validation (To Do)
- [ ] Compare with OKR for 616 overlapping khipus
- [ ] Validate cord counts
- [ ] Validate numeric values
- [ ] Generate data quality report

### ⬜ Phase 4: Analysis Integration (To Do)
- [ ] Reproduce Phase 3 summation testing
- [ ] Reproduce Phase 4 clustering
- [ ] Compare results with OKR analyses
- [ ] Update documentation

## 📈 Key Metrics

### Dataset Comparison
| Metric | OKR | KFG | Advantage |
|--------|-----|-----|-----------|
| Total khipus | 612 | 703 | KFG +14.9% |
| Metadata completeness | 84% | 100% | KFG +19% |
| Knot spatial data | No | Yes | KFG ✓ |
| Last updated | Static | 2025-2026 | KFG ✓ |

### Overlap Analysis
- **Both datasets:** 616 khipus (90% of KFG, 99.5% of OKR)
- **KFG-only:** 66 khipus (9.5% additional coverage)
- **OKR-only:** 3 khipus (0.5% loss if switching)

## 🎯 Quick Win: Process Single Khipu

Want to see the parsers in action? Here's how to process one KFG file:

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

## 🔍 Common Questions

**Q: Why use KFG instead of OKR?**  
A: KFG has 66 more khipus, better metadata completeness (100% vs 84%), spatial knot positions, and active maintenance (2025-2026 updates).

**Q: Will this break existing analyses?**  
A: No. Extractors will export to same CSV format as OKR, so analysis scripts won't need changes.

**Q: How long to implement?**  
A: 28-39 hours for complete pipeline (parsers done, extractors and validation remain).

**Q: Can we use both datasets?**  
A: Yes. Abstraction layer approach allows switching between sources or cross-validation.

**Q: What about the 3 OKR-only khipus?**  
A: Very minor loss (0.5%). If needed, could maintain OKR as supplementary source.

## 📚 File Structure

```
khipu-computational-toolkit/
├── data/
│   └── kfg/
│       └── KFG/
│           └── KFG/
│               ├── KH0001.xlsx  ← 709 khipu files
│               ├── KH0002.xlsx
│               └── ...
├── docs/
│   ├── KFG_INTEGRATION_SUMMARY.md       ← Executive summary
│   ├── KFG_INTEGRATION_ASSESSMENT.md    ← Technical details
│   ├── KFG_INVESTIGATION_FINDINGS.md    ← Format specifications
│   └── KFG_QUICK_REFERENCE.md           ← This file
├── src/
│   └── extraction/
│       ├── kfg_parsers.py               ← ✅ Complete
│       ├── kfg_loader.py                ← ⬜ To build
│       ├── kfg_cord_extractor.py        ← ⬜ To build
│       ├── kfg_knot_extractor.py        ← ⬜ To build
│       └── kfg_color_extractor.py       ← ⬜ To build
└── scripts/
    └── investigate_kfg_format.py        ← ✅ Complete
```

## 🚦 Status

**Overall Progress:** 40% complete

- ✅ Investigation: 100%
- ✅ Documentation: 100%
- ✅ Parsers: 100%
- ⬜ Extractors: 0%
- ⬜ Validation: 0%
- ⬜ Integration: 0%

**Next Action:** Build KFGCordExtractor (estimated 4-6 hours)

---

**Need Help?**
- Review [KFG_INTEGRATION_SUMMARY.md](KFG_INTEGRATION_SUMMARY.md) for executive overview
- Check [KFG_INVESTIGATION_FINDINGS.md](KFG_INVESTIGATION_FINDINGS.md) for format details
- Run `python scripts/investigate_kfg_format.py` to explore the data
- Test parsers with `python src/extraction/kfg_parsers.py`
