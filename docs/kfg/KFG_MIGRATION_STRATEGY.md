# KFG Data Migration Strategy

**Status:** 🚧 IN PROGRESS  
**Branch:** `kfg-integration`  
**Target:** Replace OKR data with authoritative KFG (Khipu Field Guide) data  
**Timeline:** Phase 1 (Setup) → Phase 2 (Migration) → Phase 3 (Validation) → Phase 4 (Replace)

---

## Background

Per MIT feedback (February 2026), the **Open Khipu Repository (OKR) data is deprecated**. The Khipu Field Guide (KFG) represents 3-4 person-years of corrections and is now the authoritative source. Our current toolkit uses OKR data exclusively.

**Key Issue:** MIT reports **60-70% of khipus have summation**, while our OKR-based analysis initially found only 27.9%. After algorithm correction, we now find 69.5% - validating our algorithms work correctly, but we still need to migrate to KFG data for maximum accuracy.

---

## Migration Strategy Overview

### Phase 1: Setup Dual Configuration ✅ COMPLETE

**Goal:** Enable side-by-side operation of OKR and KFG data sources without breaking existing work.

**Implementation:**
- Created `src/config_kfg.py` - KFG-specific configuration wrapper
- KFG processed data stored in `data/processed_kfg/` (separate from OKR's `data/processed/`)
- Both configurations coexist safely

**Usage:**
```python
# Existing OKR code (unchanged)
from src.config import get_config
config = get_config()  # Uses OKR database

# New KFG code
from src.config_kfg import get_kfg_config
config = get_kfg_config()  # Uses KFG database
```

**File Structure:**
```
data/
├── kfg/
│   └── khipu_database.db          # Authoritative KFG database
├── processed/                      # OKR-derived results (legacy)
│   ├── phase2/
│   ├── phase3/
│   └── ...
└── processed_kfg/                  # KFG-derived results (NEW)
    ├── phase2/
    ├── phase3/
    └── ...
```

---

### Phase 2: Run KFG Pipeline 🚧 NEXT

**Goal:** Generate complete processed datasets using KFG database.

**Tasks:**

1. **Create KFG Integration Branch**
   ```bash
   git checkout -b kfg-integration
   git push -u origin kfg-integration
   ```

2. **Update All Extraction Scripts**
   
   Modify each script to accept `--kfg` flag:
   
   ```python
   # Example: scripts/extract_cord_hierarchy.py
   import argparse
   from src.config import get_config
   from src.config_kfg import get_kfg_config
   
   parser = argparse.ArgumentParser()
   parser.add_argument('--kfg', action='store_true', 
                      help='Use KFG database instead of OKR')
   args = parser.parse_args()
   
   # Select configuration
   config = get_kfg_config() if args.kfg else get_config()
   ```
   
   **Scripts to Update:**
   - [ ] `extract_cord_hierarchy.py`
   - [ ] `extract_knot_data.py`
   - [ ] `extract_color_data.py`
   - [ ] `build_khipu_graphs.py`
   - [ ] `test_value_computation.py`
   - [ ] `test_color_hypotheses.py`
   - [ ] `analyze_geographic_correlations.py`
   - [ ] `cluster_khipus.py`
   - [ ] All Phase 4, 7, 8, 9 scripts

3. **Run Complete KFG Pipeline**
   
   ```bash
   # Phase 2: Extraction
   python scripts/extract_cord_hierarchy.py --kfg
   python scripts/extract_knot_data.py --kfg
   python scripts/extract_color_data.py --kfg
   
   # Phase 3: Summation Testing
   python scripts/test_value_computation.py --kfg
   
   # Phase 4: Pattern Discovery
   python scripts/build_khipu_graphs.py --kfg
   python scripts/cluster_khipus.py --kfg
   
   # Phase 5: Multi-Model Framework
   python scripts/test_color_hypotheses.py --kfg
   python scripts/analyze_geographic_correlations.py --kfg
   
   # Continue through all phases...
   ```

4. **Generate KFG Visualizations**
   
   All visualization scripts should also accept `--kfg` flag and output to separate directories:
   - OKR visualizations: `visualizations/phase*/`
   - KFG visualizations: `visualizations_kfg/phase*/`

---

### Phase 3: Validation & Comparison 📊 FUTURE

**Goal:** Compare OKR vs KFG results to validate migration and identify improvements.

**Comparison Metrics:**

| Metric | OKR (Legacy) | KFG (Authoritative) | Change |
|--------|--------------|---------------------|--------|
| Total khipus | 619 | TBD | TBD |
| Khipus with summation | 430 (69.5%) | TBD | TBD |
| First-position white cords | 332 (53.6%) | TBD | TBD |
| Average knots per khipu | TBD | TBD | TBD |
| Color accuracy | TBD | TBD | TBD |

**Validation Tasks:**
- [ ] Compare database schemas (OKR vs KFG)
- [ ] Validate summation rate matches MIT expectation (60-70%)
- [ ] Check for data quality improvements
- [ ] Identify any khipus present in KFG but missing from OKR
- [ ] Document algorithm adjustments needed for KFG schema
- [ ] Generate comparison report

**Deliverable:** `reports/kfg_migration_validation_report.md`

---

### Phase 4: Replace & Archive 🔄 FUTURE

**Goal:** Make KFG the default data source and archive OKR results.

**Tasks:**

1. **Archive OKR Results**
   ```bash
   # Create archive directory
   mkdir -p data/archive_okr
   
   # Move OKR processed data
   mv data/processed data/archive_okr/processed_okr
   mv visualizations visualizations_okr
   
   # Rename KFG data to main
   mv data/processed_kfg data/processed
   mv visualizations_kfg visualizations
   ```

2. **Update Default Configuration**
   ```python
   # src/config.py - Change default database path
   def get_database_path(self) -> Path:
       # Use KFG as default
       default_path = self.data_dir / "kfg" / "khipu_database.db"
       return default_path.resolve()
   ```

3. **Update All Documentation**
   - README.md - Update database setup instructions
   - Installation guide - Remove OKR references
   - API documentation - Update schema references
   - All reports - Add "KFG Data" badge

4. **Deprecation Notice**
   
   Add to README:
   ```markdown
   ## ⚠️ Data Source Migration
   
   **As of March 2026, this toolkit uses the Khipu Field Guide (KFG) database
   as the authoritative data source.** The Open Khipu Repository (OKR) data
   has been archived and is no longer maintained.
   
   Legacy OKR results are preserved in `data/archive_okr/` for reference.
   ```

5. **Merge to Main**
   ```bash
   git checkout main
   git merge kfg-integration
   git push origin main
   git tag v2.0.0-kfg
   ```

---

## Benefits of Dual Configuration Approach

✅ **No Breaking Changes:** Existing OKR scripts continue to work unchanged  
✅ **Side-by-Side Testing:** Can run same analysis on both datasets for validation  
✅ **Safe Migration:** KFG work happens in isolated branch and directories  
✅ **Easy Rollback:** If issues arise, revert is simple  
✅ **Clear History:** Git branch preserves entire migration process

---

## Current Status

**✅ Phase 1 Complete:**
- `src/config_kfg.py` created and tested
- Dual configuration validated
- File structure planned

**🚧 Phase 2 In Progress:**
- Ready to update extraction scripts
- Awaiting decision to proceed with full pipeline run

**📋 Next Immediate Actions:**
1. Create `kfg-integration` branch
2. Update first extraction script (`extract_cord_hierarchy.py`)
3. Test KFG extraction on single script
4. Once validated, update remaining scripts systematically

---

## Questions & Decisions

**Q: Should we maintain OKR support long-term?**  
A: No. Once KFG migration is validated and merged, remove `--kfg` flags and make KFG the only option. Simplify codebase.

**Q: What if KFG schema is different from OKR?**  
A: Document differences in migration report. Update extraction logic as needed. This is expected and part of why we're migrating.

**Q: Timeline?**  
A: Phase 2 (pipeline run): 1-2 days. Phase 3 (validation): 2-3 days. Phase 4 (replace): 1 day. Total: ~1 week.

---

## References

- **MIT Feedback:** `docs/MIT_FEEDBACK_AND_CORRECTIONS.md`
- **KFG Database:** `data/kfg/khipu_database.db`
- **KFG Config:** `src/config_kfg.py`
- **OKR Config:** `src/config.py` (legacy)
