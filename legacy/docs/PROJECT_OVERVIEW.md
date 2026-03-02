# Khipu Computational Toolkit — Project Overview

*Last updated: March 1, 2026*

---

## What It Is

An open-source Python toolkit for computational analysis of Inka khipus — knotted-cord artifacts that served as the primary recording medium of the Inka empire. The toolkit ingests data from two authoritative scholarly datasets, applies a layered pipeline of data extraction, arithmetic pattern detection, structural analysis, and visualization, and presents results through an interactive local browser and a planned cloud-hosted application. It is designed for researchers, students, and computational humanists who want to engage with the khipu corpus rigorously without requiring deep domain expertise to get started.

The project is explicitly non-interpretive: it does not claim to decode or translate khipus. All outputs are exploratory computational signals that require independent expert validation.

---

## Data Sources

### Open Khipu Repository (OKR)
- 612 khipus, 54,403 cords, 110,677 knots in a community-maintained SQLite database
- Used for the earlier research phases (clustering, anomaly detection, ML extensions)
- Still fully supported for algorithmic summation detection and structural analysis

### Khipu Field Guide (KFG)
- 709 khipus in a purpose-built SQLite database derived from the Khipu Field Guide project
- More recent, more accurate, and richer than OKR — includes expert-identified summation pattern annotations at full cord-level resolution
- The primary dataset for all current development
- Includes a `checks/` directory with 9 pattern-specific relation CSVs (authoritative ground truth) and 9 summary CSVs

---

## Core Analytical Capabilities

### 1. Numeric Decoding and Extraction
- Translates raw knot records (type + position on cord) into positional-decimal numeric values following Ascher conventions
- Reconstructs the full cord hierarchy (primary cord → pendant → subsidiary → further subsidiaries) from flat database records
- Extracts and normalizes color data including Ascher color codes, compound codes (e.g. `MB:W`), and multi-color cords
- Assigns a confidence score to every numeric value based on completeness of knot data; the OKR corpus is bimodal — 55.5% high-confidence, 44.2% low-confidence

### 2. Summation Pattern Detection
The most developed component. Nine distinct arithmetic relationship types are detected:

| Short | Full name | What it means |
|---|---|---|
| PP | pendant_pendant_sum | A cord = sum of an adjacent contiguous window of pendants |
| IP | indexed_pendant_sum | A cord = sum of same-position pendants across all cord groups |
| CP | colored_pendant_sum | A cord = sum of all pendants sharing the same color |
| SP | subsidiary_pendant_sum | A subsidiary cord = sum of a pendant window |
| IS | indexed_subsidiary_sum | Subsidiary + position-indexed cross-group relationship |
| GG | group_group_sum | One cord group's total = sum of another group's total |
| GSB | group_sum_bands | A cord = sum of corresponding positions across split group bands |
| ADG | ascher_decreasing_groups | Groups decrease monotonically by a consistent factor |
| PSN | pendant_sub_neighbor | A pendant = sum of its own subsidiary plus a neighboring cord |

Two detection paths exist:

- **KFG path**: For the 709 KFG khipus, the `KFGRelationLoader` reads the authoritative `*_relation.csv` files directly. This achieves 99.4% agreement with KFG ground truth (8 of 9 patterns at 100%; the PSN counting-unit definition remains an open question with the KFG authors).
- **Algorithmic path**: For khipus outside the KFG corpus, a rule-based detector applies the same nine pattern definitions computationally. Used for OKR khipus and any future corpora.

An important discovery from this project: the KFG records the same cord in multiple pattern tables simultaneously (a cord can be a PP sum cord, an IP sum cord, and a CP sum cord with different summand sets — all simultaneously true). The system preserves this multi-label representation. A separate exclusivity function is available for use cases that require a single classification label per cord.

Across the OKR corpus: 69.5% of khipus exhibit at least one Ascher summation pattern.

### 3. Missing Value Prediction
Three complementary methods recover numeric values for low-confidence cords:
- **Constraint propagation**: derives bounds from known summation relationships
- **Sibling interpolation**: infers from neighboring cords in the same group
- **ML model** (trained on OKR): predicts values from structural and color features

Combined: 24,043 predictions generated, average confidence gain of +0.708.

### 4. Graph-Based Structural Analysis
- Converts each khipu into a directed hierarchical graph (NetworkX) with nodes for each cord and edges encoding attachment type, position, and color
- Computes per-graph structural features: depth, branching factor, subsidiary ratios, pendant density, color diversity
- Mines recurring structural sub-graph motifs across the corpus
- Computes pairwise graph similarity scores for corpus-level structural comparison
- Extracts structural templates — common cord-group configurations that recur across multiple khipus

### 5. Clustering and Operational Typology (OKR corpus)
- K-means clustering identifies 7 structural groups across the OKR corpus (silhouette = 0.339)
- Summation features are excluded from clustering to prevent them dominating the structural solution
- An unsupervised 8-class operational typology is derived by combining cluster membership with summation presence/absence and cord-count ranges — explicitly marked as exploratory, not validated

### 6. Anomaly Detection
- Isolation Forest and Local Outlier Factor methods identify structural outliers
- 13 high-confidence anomalies identified in the OKR corpus

### 7. Multi-Model Hypothesis Framework
- Five competing color-semantics models (color-as-value, color-as-category, color-as-modifier, color-as-ply, color-as-hierarchy) are evaluated simultaneously against summation evidence
- No single model is assumed correct; all are scored and compared on fit to observed patterns
- Model parameters are externalized — alternative interpretations can be tested without code changes

### 8. Geographic and Comparative Analysis
- Where provenance data is available, summation patterns and structural clusters are compared across Andean geographic regions and site types
- Geographic heatmaps of pattern prevalence by coordinates
- Variance and robustness testing of pattern detection under threshold perturbations

---

## Interactive Tools

### Local Browser (`browse.py`) — current primary tool

A single Streamlit application with three views, running locally against the KFG database:

- **Corpus Browser**: Filterable, sortable table of all 709 KFG khipus. Filter by provenance, country, minimum cord count, or free-text search. Columns include KFG ID, name, provenance, region, museum, cord count, and a direct link to the khipu's page on the Khipu Field Guide website.

- **3D Viewer**: Interactive Plotly 3D rendering of the cord hierarchy for any selected khipu. Nodes represent pendant and subsidiary cords, colored by Ascher color codes. Edges show structural attachment. Hover tooltips show cord name, color, decoded value, length, and knot data. Handles khipus with 200+ cords smoothly.

- **X-Ray View**: A 2D color grid showing every pendant cord as a colored square, organized by cord group (x-axis) and position within group (y-axis). Hovering reveals cord name, color, and value. Includes a group summary table with cord count, colors present, numeric cord count, and sum of values per group. **Summation arc overlays are planned** — arcs will be drawn between sum cords and their summands once the detection layer is wired into this view.

### Legacy OKR Dashboard (`dashboard_app.py`)

A Streamlit app built for the OKR corpus, showing cluster scatter plots, geographic maps, summation statistics, and per-khipu drill-down based on Phase 3/4 OKR analysis results. **Planned for removal**: it uses the older OKR dataset and predates the KFG integration. It will be superseded by the Analytics tab described in the development plan below.

### Jupyter Notebooks (4)

Cluster explorer, geographic patterns, khipu detail viewer, hypothesis dashboard — all interactive, all OKR-based.

---

## Outputs and Reproducibility
- 100+ processed data files (CSV, JSON) covering each analysis phase
- 39 visualization files organized by research phase (PNG/HTML)
- 10 phase reports (Phases 0–9) documenting methods, results, and limitations
- Trained ML models saved for missing-value prediction
- Full pipeline re-runnable end-to-end from the raw database

---

## Technology Stack

Python 3.11 · SQLite · Pandas · NumPy · scikit-learn · NetworkX · Streamlit · Plotly · Matplotlib · Seaborn · Jupyter

---

## Development Plan — Next Phase

### Goal

Consolidate the toolkit around the KFG dataset and upgrade `browse.py` into a comprehensive analytical tool. Retire the legacy OKR dashboard. Lay the groundwork for a cloud-hosted version.

---

### Step 1 — Add Analytics Tab to `browse.py`

Add a fourth view, **Analytics**, to the existing three-tab browser. All data sourced from the KFG ground truth (the `ascher_sums_overview.csv` and `*_summary.csv` files in `checks/`), not from the old OKR processed files.

**Panel A — Pattern Prevalence**
- Horizontal bar chart: for each of the 9 patterns, how many of the 709 KFG khipus have at least one detected instance
- Secondary metric: percentage of the corpus each represents
- Ordered by prevalence (PP highest → IS lowest)

**Panel B — Pattern Co-occurrence Heatmap**
- 9×9 symmetric heatmap: each cell = number of khipus that have both pattern X and pattern Y
- Diagonal = khipus with that single pattern
- Color scale from 0 to max co-occurrence count
- Reveals which patterns are structurally linked (e.g. PP and IP typically co-occur)

Both panels use only the `ascher_sums_overview.csv` and two additional summary CSVs for IS and PSN (which are not in the overview file). No new computation required — the data is already in the checks directory.

---

### Step 2 — Wire Summation Arcs into X-Ray View

Complete the "coming soon" annotation in the X-Ray view:
- Load matched summation results for the selected khipu from the `KFGRelationLoader`
- For each detected PP/IP/CP/SP/IS relationship, draw a curved arc on the color grid connecting the sum cord to each of its summand cords
- Color-code arcs by pattern type (PP=blue, IP=orange, CP=green, etc.)
- Add a legend and per-pattern toggle checkboxes to show/hide individual pattern types

---

### Step 3 — Retire `dashboard_app.py`

Delete the file. Update the README and scripts README to remove references to it. Visualizations it generated are kept as archived static outputs but the script that regenerates them is removed.

---

### Step 4 — Cloud Application (Future)

The enhanced `browse.py` (all four tabs including Analytics and summation arcs) forms the specification for the cloud version:
- FastAPI backend serving the KFG database queries as REST endpoints
- React frontend replicating the four-tab layout with equivalent visualizations
- Hosted on Azure Container Apps
- Publicly accessible without a local install

This step is deferred until Steps 1–3 are complete and the local app is stable.

---

## References

### Primary Data Sources

**Khosla, Ashok.** *The Khipu Field Guide.* [khipufieldguide.com](https://khipufieldguide.com), 2020–present.
With contributions from Karen Thompson (University of Melbourne), Manuel Medrano (Harvard University), Kylie Quave (George Washington University), Mack FitzPatrick (Harvard University), Saoirse Byrne, and Andrés Chirinos. The KFG is the authoritative digital database of Inka khipus: 709 khipus with approximately 3–4 person-years of systematic quality correction by Khosla and Thompson.

**Open Khipu Repository (OKR) Team.** *The Open Khipu Repository (v1.0).* Harvard Dataverse / Zenodo, 2021.
[https://doi.org/10.5281/zenodo.5037551](https://doi.org/10.5281/zenodo.5037551)
An earlier community-maintained open dataset of 612 khipus. Per the KFG team, the KFG now supersedes the OKR as the authoritative source for computational khipu research.

---

### Foundational Khipu Scholarship

**Ascher, Marcia and Robert Ascher.** *Code of the Quipu: Databook.* Cornell University, 1978.
Defines the positional-decimal notation system used to decode knot values from knot type and position on a cord — the basis of all numeric decoding in this toolkit.

**Ascher, Marcia and Robert Ascher.** *Mathematics of the Incas: Code of the Quipu.* Dover Publications, 1997. (Reprint of the 1981 first edition.)
Establishes the seven core Ascher summation fieldmarks (PP, IP, CP, SP, GG, GSB, ADG) used by the toolkit's summation detector.

---

### Computational Khipu Research

**Khosla, Ashok and Manuel Medrano.** "How Can Data Science Contribute to Understanding the Khipu Code?" *Latin American Antiquity*, 2023.
The primary publication applying computational methods to the KFG corpus. Defines the operationalization of the eight Ascher summation types (including `ascher_decreasing_group` as the eighth) and establishes corpus-scale prevalence figures used as validation baselines in this toolkit.

**Medrano, Manuel and Ashok Khosla.** Related corpus analysis work. Harvard University / MIT Khipu Lab, 2020–2024.
MIT Khipu Lab feedback on summation detection algorithms and validation approaches is referenced in the Phase 2 report.

---

### Additional Scholarly Context

**Thompson, Karen.** Work on KFG Ascher khipus, including the relationship between KH0082 and KH0083. Published in *Nawpa Pacha: Journal of Andean Archaeology.* University of Melbourne.

**Urton, Gary.** Cord-level annotations referenced in individual KFG records (visible in the KFG data as "per Urton" notes in cord observation fields). Urton's physical examinations of specific khipus contributed alternative knot counts and cord observations that the KFG records alongside Ascher readings.

**Hyland, Sabine.** Non-numeric semiotic interpretation research — cited in the project's limitations sections as an example of alternative interpretive frameworks in which the toolkit's "summation patterns" carry different significance.

---

### This Toolkit

**Da Fieno Delucchi, A.** *Khipu Computational Analysis Toolkit (K-CAT).* [github.com/adafieno/khipu-computational-toolkit](https://github.com/adafieno/khipu-computational-toolkit), 2026.

---

*If you use this toolkit in research, please also cite the KFG primary source and the Ascher & Ascher reference for the decoding methodology.*
