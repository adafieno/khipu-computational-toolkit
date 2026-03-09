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

## Live Demo

The K-CAT analytics dashboard is also available as a **hosted cloud app** — no installation required:

> **[https://khipu-explorer.greenrock-570e1f4a.westus2.azurecontainerapps.io/](https://khipu-explorer.greenrock-570e1f4a.westus2.azurecontainerapps.io/)**

The cloud app (K-CAT Khipu Explorer) exposes the same four views as the local browser and is free to use. The source lives in the companion repository [`khipu-explorer`](https://github.com/your-org/khipu-explorer).

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

The browser provides four views: **Corpus Browser** (filterable table of 709 khipus), **Analytics** (pattern statistics dashboard), **3D Viewer** (Plotly cord structure), and **Summation Arcs** (cord-grid map with togglable arc overlays).

---

## Research Phases

K-CAT organizes analysis into numbered phases. Each phase has a script entry-point, processed outputs, and a report.

| Phase | Topic | Script | Report |
|-------|-------|--------|--------|
| 1 | Corpus Foundation | `scripts/corpus_statistics.py` | [phase1_corpus_foundation.md](reports/phase1_corpus_foundation.md) |
| 2 | Summation Patterns | `scripts/test_kfg_summation_detector.py` | [phase2_summation_patterns.md](reports/phase2_summation_patterns.md) |
| 3 | Structural Typology | `scripts/run_phase3_typology.py` | [phase3_structural_typology.md](reports/phase3_structural_typology.md) |
| 4 | Geographic Patterns | `scripts/run_phase4_geography.py` | [phase4_geographic_patterns.md](reports/phase4_geographic_patterns.md) |
| 5 | Color Analysis | `scripts/run_phase5_color.py` | [phase5_color_analysis.md](reports/phase5_color_analysis.md) |
| 6 | Anomaly Detection | `scripts/run_phase6_anomaly.py` | [phase6_anomaly_detection.md](reports/phase6_anomaly_detection.md) |
| 7 | Multi-feature Typology | `scripts/run_phase7_typology.py` | [phase7_typology_report.md](reports/phase7_typology_report.md) |
| 8 | Behavioral Analysis | `scripts/run_phase8_behavior.py` | [phase8_behavioral_analysis.md](reports/phase8_behavioral_analysis.md) |
| 9 | Graph Topology | `scripts/run_phase9_graph.py` | [phase9_graph_topology_report.md](reports/phase9_graph_topology_report.md) |
| 10 | Summation Compliance | `scripts/run_phase10_summation.py` | [phase10_summation_analysis_report.md](reports/phase10_summation_analysis_report.md) |
| 11 | Color Value | `scripts/run_phase11_color.py` | [phase11_color_value_report.md](reports/phase11_color_value_report.md) |

### Key findings

- **709 khipus**, 62,746 cords, 70,143 knot clusters; 98.2% of khipus have ≥1 decoded cord value
- **72.6%** of khipus carry at least one summation pattern across 9 detector types (Phase 2)
- **Best k = 2** structural clusters (silhouette = 0.37): 591 Simple (mean 45 cords, ~2 pattern types) vs 118 Complex (mean 304 cords, ~6 pattern types) (Phase 3)
- **Chachapoyas 52% Complex**, Central Coast 8% — strongest geographic signal; PP and IS rates are statistically significant by zone (Phase 4)
- **White** is the dominant color code (27.5% of cord entries, 77.7% of khipus); color diversity is 3× higher in Complex vs Simple khipus (Phase 5)
- **90** anomalous khipus identified; 43 high-confidence (all three methods agree) (Phase 6)
- Phase 7 T2 typology (n = 56) concentrates at Chachapoyas; 85.7% are Phase 3 Complex
- **6 behavioral clusters** (B1–B6) cross-cut structural typology; B3 (n = 245) is the dominant recording style (Phase 8)
- **80.3%** of pendants carry no subsidiaries; motif-8 (8 subsidiaries) is an unexplained frequency spike (Phase 9)
- Summation constraint (parent = sum of children) holds for only **6.6%** of parent-child groups; median summation ratio is **0.35** (Phase 10)
- Primary color predicts cord value order-of-magnitude (Kruskal-Wallis p = 3.6 × 10⁻²⁶⁷); white cords carry significantly lower values than brown-family cords (Phase 11)

---

## Repository Structure

```
data/
  kfg/                    # KFG Excel source files + SQLite DB (gitignored)
  processed/              # Pipeline outputs (CSV) — phases 3–11

docs/
  VISUALIZATIONS_GUIDE.md # Interactive browser + static figure reference
  kfg/                    # KFG-specific documentation
    KFG_DATABASE_SCHEMA.md
    KFG_MIGRATION_STRATEGY.md
    KFG_QUICK_REFERENCE.md
    MIT_FEEDBACK_AND_CORRECTIONS.md

reports/                  # Phase reports (Phases 1–11)
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
  phase3/ … phase11/      # PNG figures for each analysis phase

legacy/                   # Frozen OKR-era code, data, reports, and visualizations
                          # (gitignored — preserved in git history)
```

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `build_kfg_database.py` | Parse KFG Excel files → SQLite |
| `corpus_statistics.py` | Phase 1: corpus baseline statistics |
| `test_kfg_summation_detector.py` | Phase 2: summation detection; write pattern CSVs |
| `run_phase3_typology.py` | Phase 3: feature matrix, k-means clusters, UMAP figures |
| `run_phase4_geography.py` | Phase 4: geographic zone analysis, chi-square, NN attribution |
| `run_phase5_color.py` | Phase 5: color vocabulary, diversity, white-cord hypothesis |
| `run_phase6_anomaly.py` | Phase 6: multi-method anomaly detection |
| `run_phase7_typology.py` | Phase 7: multi-feature typology (T1/T2) |
| `run_phase8_behavior.py` | Phase 8: behavioral cluster analysis (B1–B6) |
| `run_phase9_graph.py` | Phase 9: graph topology metrics and motif catalog |
| `run_phase10_summation.py` | Phase 10: summation compliance ratios |
| `run_phase11_color.py` | Phase 11: color–value correlations |
| `browse.py` | Streamlit local corpus browser (4 views) |
| `reconcile_kfg_fieldmarks.py` | Cross-check KFG fieldmarks against K-CAT detections |
| `calibrate_detector_threshold.py` | Tune summation detector thresholds |
| `import_kfg_summation_checks.py` | Ingest KFG expert summation annotations |
| `migrate_provenance_labels.py` | Load provenance label table into DB |
| `migrate_cord_groups.py` | Load cord group assignments into DB |

---

## Configuration

Database path is managed by `src/config_kfg.py`. The KFG database defaults to `data/kfg/khipu_database.db` (gitignored — must be generated locally via `build_kfg_database.py`).

---

## Status and Caveats

- Phases 1–11 are complete. All findings are exploratory and require expert validation before interpretive use.
- Phase 2 has two open questions pending KFG team response (PP threshold, PSN interpretation) that may shift ~150 binary pattern flags. Downstream cluster boundaries (Phases 3, 7) may adjust accordingly.
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
