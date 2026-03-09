# Khipu Computational Analysis Toolkit (K-CAT)

**Computational infrastructure for analyzing Inka khipus using the Khipu Field Guide dataset**

[![Python](https://img.shields.io/badge/Python-3.11+-blue)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()
[![Status](https://img.shields.io/badge/Status-Research%20Tool-blue)]()

## Overview

K-CAT is a research toolkit for computational analysis of Inka khipus. It is built on the [Khipu Field Guide (KFG)](https://khipufieldguide.com) dataset — 709 khipus with carefully corrected fieldmarks representing approximately 3–4 person-years of expert annotation.

The toolkit focuses on **falsifiable, reproducible hypothesis testing**: summation pattern detection, structural typology, and geographic analysis. All findings are exploratory and require expert validation before interpretive use.

> **Not a decipherment project.** K-CAT does not claim to decode khipu meaning. It provides computational infrastructure for scholars to test hypotheses transparently and surface structural patterns.

---

## Quick Start

```bash
# 1. Place the KFG database at data/kfg/khipu_database.db
#    (gitignored — obtain from KFG team)

# 2. Set up environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt

# 3. Build the SQLite database from KFG Excel files
python scripts/build_kfg_database.py

# 4. Launch the local corpus browser
streamlit run scripts/browse.py
```

The browser provides: **Corpus Browser** (filterable table of 709 khipus), **3D Viewer** (Plotly cord structure), and **X-Ray View** (color grid with summation arc overlays).

---

## Research Phases

K-CAT organizes analysis into numbered phases. Each phase has a script entry-point, processed outputs, and a report.

| Phase | Topic | Status | Script | Report |
|-------|-------|--------|--------|--------|
| 1 | Corpus Foundation | ? Complete | `scripts/corpus_statistics.py` | [phase1_corpus_foundation.md](reports/phase1_corpus_foundation.md) |
| 2 | Summation Patterns | ? Complete | `scripts/test_kfg_summation_detector.py` | [phase2_summation_patterns.md](reports/phase2_summation_patterns.md) |
| 3 | Structural Typology | ? Complete | `scripts/run_phase3_typology.py` | [phase3_structural_typology.md](reports/phase3_structural_typology.md) |

### Phase 3 key findings (current)

- **709 khipus** in corpus; **80.3%** carry at least one summation pattern
- **Best k = 2** clusters (silhouette = 0.37): 591 Simple (avg 45 cords, ~2 pattern types) vs 118 Complex (avg 304 cords, ~6 pattern types)
- **Chachapoyas 52% Complex**, Central Coast 8% — strongest geographic signal
- UMAP projections: `visualizations/phase3/`

---

## Repository Structure

```
data/
  kfg/                    # KFG Excel source files + SQLite DB (gitignored)
  processed/              # Pipeline outputs (CSV)
    kfg_fieldmarks_reconciliation.csv
    phase3_feature_matrix.csv
    phase3_clusters.csv
    phase3_silhouette.csv

docs/
  kfg/                    # KFG-specific documentation
    KFG_DATABASE_SCHEMA.md
    KFG_MIGRATION_STRATEGY.md
    KFG_QUICK_REFERENCE.md
    MIT_FEEDBACK_AND_CORRECTIONS.md

reports/                  # Phase reports (Phases 1–3)
scripts/                  # Analysis entry-points
src/
  config_kfg.py           # Path configuration
  analysis/
    kfg_summation_detector.py
    kfg_relation_loader.py
    feature_matrix.py
  extraction/
    kfg_cord_extractor.py
    kfg_parsers.py
  utils/
    arithmetic_validator.py

visualizations/
  phase3/                 # 5 PNGs (heatmap, silhouette, UMAP ×3)

legacy/                   # Frozen OKR-era code, data, reports, and visualizations
                          # (gitignored — preserved in git history)
```

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `build_kfg_database.py` | Parse KFG Excel files ? SQLite |
| `reconcile_kfg_fieldmarks.py` | Cross-check KFG fieldmarks against K-CAT detections |
| `calibrate_detector_threshold.py` | Tune summation detector thresholds |
| `test_kfg_summation_detector.py` | Run summation detection; write Phase 2 outputs |
| `run_phase3_typology.py` | Build feature matrix, cluster, generate UMAP figures |
| `corpus_statistics.py` | Phase 1 corpus baseline statistics |
| `browse.py` | Streamlit local corpus browser |
| `import_kfg_summation_checks.py` | Ingest KFG expert summation annotations |
| `migrate_provenance_labels.py` | Load provenance label table into DB |
| `migrate_cord_groups.py` | Load cord group assignments into DB |

---

## Configuration

Database path is managed by `src/config_kfg.py`. The KFG database defaults to `data/kfg/khipu_database.db` (gitignored — must be generated locally via `build_kfg_database.py`).

---

## Status and Caveats

- Phase 2 has two open questions pending KFG team response (PP threshold, PSN interpretation) that may shift ~150 binary pattern flags. Phase 3 cluster boundaries may adjust accordingly.
- Phase 3 results are **not for publication** until Phase 2 open questions are resolved.
- `museum_country` / `museum_name` are intentionally excluded from geographic analysis — they record current exhibition location, not origin.

---

## Legacy (OKR-era)

The `legacy/` directory contains the prior OKR-based pipeline (Phases 0–9), including scripts, processed data, notebooks, and reports built on the [Open Khipu Repository](https://github.com/khipulab/open-khipu-repository) database. That work is frozen; all active development uses the KFG dataset.

---

## Citations and Acknowledgments

### Citing This Toolkit

`
Da Fieno Delucchi, A. (2026). Khipu Computational Analysis Toolkit (K-CAT).
https://github.com/adafieno/khipu-computational-toolkit
`

### Primary Data Source

All analyses use the **Khipu Field Guide (KFG)** database.

`
Khosla, A., & Medrano, M. (2020–present). Khipu Field Guide.
https://khipufieldguide.com
`

The KFG was created and is edited by **Ashok Khosla**. Substantial database curation and correction work was contributed by **Karen Thompson** (Senior Research Data Specialist, University of Melbourne), along with KFG affiliates **Manuel Medrano** (Harvard University), **Kylie Quave** (George Washington University), **Mack FitzPatrick** (Harvard University), **Saoirse Byrne**, and **Andrés Chirinos**. Per Ashok Khosla: “Karen Thompson and I both have invested at least 3 or 4 person-years of effort in improving and correcting the database.”

### Numeric Decoding Methodology

Cord values are decoded using the Ascher & Ascher positional notation system:

> Ascher, Marcia and Robert Ascher. *Mathematics of the Incas: Code of the Quipu*. Dover Publications, 1997. (Reprint of the 1981 edition.)

> Ascher, Marcia and Robert Ascher. “Code of the Quipu: Databook.” Cornell University, 1978.

### Published Research

> Khosla, Ashok and Manuel Medrano. “How Can Data Science Contribute to Understanding the Khipu Code?” *Latin American Antiquity*, 2023.

Karen Thompson’s work on KFG Ascher khipus (including the relationship between KH0082 and KH0083) has been published in *Nawpa Pacha* (Journal of Andean Archaeology).

MIT Khipu Lab provided feedback on summation detection.

## License

MIT — see [LICENSE](LICENSE).
