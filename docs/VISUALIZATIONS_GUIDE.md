# Visualizations Guide

This guide covers all visualization tools available in K-CAT: the interactive
local browser (`scripts/browse.py`) and the static phase outputs under
`visualizations/`.

> **Cloud version available.** The K-CAT Khipu Explorer is also hosted online at
> **[https://khipu-explorer.greenrock-570e1f4a.westus2.azurecontainerapps.io/](https://khipu-explorer.greenrock-570e1f4a.westus2.azurecontainerapps.io/)** — free to use, no setup required. It exposes the same four views as the local browser.

---

## Khipu Explorer (`scripts/browse.py`)

The primary interactive tool. A four-view Streamlit app backed directly by the
KFG SQLite database.

### Requirements

All dependencies are already covered by `requirements.txt`:

```
streamlit
plotly
pandas
numpy
```

### Setup

```bash
# 1. Build the database (first time only, or after KFG Excel files change)
python scripts/build_kfg_database.py

# 2. Launch the browser
streamlit run scripts/browse.py
```

Opens at `http://localhost:8501`.

### Navigation

The app uses a fixed icon-based left navigation bar. You can also jump directly
to any view via query parameter:

| Icon | View | URL |
|------|------|-----|
| 🔎 | Corpus Browser | `?v=corpus` |
| 💡 | Analytics | `?v=analytics` |
| 🧊 | 3D Viewer | `?v=3dviewer` |
| Σ | Summation Arcs | `?v=arcs` |

---

### Corpus Browser (`?v=corpus`)

A filterable, sortable table of all 709 KFG khipus.

**Header metrics:** total khipus · total cords · provenance count · country count.

**Table columns:** KFG ID, Provenance, Region, Country, Museum, Cords.

**Interaction:** Click any row to open a full-detail modal for that khipu. The
modal shows:
- Key metadata (provenance, region, country, museum, primary cord length/color)
- Cord summary counts (pendants, subsidiaries, groups)
- Expandable full cord data table
- "View on KFG ↗" link to the KFG web viewer

---

### Analytics (`?v=analytics`)

Corpus-wide statistics on the nine summation pattern detectors. Requires the KFG
`checks/` directory to be present.

**Header metrics:** total khipus · khipus with ≥1 pattern · pattern coverage % ·
khipus with no pattern · most common pattern.

Four tabs:

#### 📊 Overview

| Chart | What it shows |
|-------|---------------|
| **Pattern Prevalence** | Bar chart — number of khipus exhibiting ≥1 instance of each pattern, sorted by prevalence |
| **Pattern Co-occurrence** | 9×9 heatmap — how many khipus simultaneously express both patterns (diagonal = single-pattern count) |
| **Pattern Complexity** | Histogram — distribution of how many distinct patterns each khipu carries (0 = no summation structure) |

An expandable legend explains all nine pattern codes (PP, IP, CP, SP, IS, GG,
GSB, ADG, PSN).

#### 🔬 Deep Dive

| Chart | What it shows |
|-------|---------------|
| **Handedness** | Left (←) vs right (→) summation direction across all cord-level patterns (PP · IP · CP · SP · IS) |
| **Instance-Count Distribution** | Box plots of how many summation instances each positive khipu carries, per pattern |
| **Sum Magnitude Distribution** | Box plots of mean cord-value sum per khipu, for patterns that report numeric magnitudes |
| **Dual- & Multi-Summand Breakdown** | PP / IP / CP: instances split into regular (A+B), dual-summand (cord in two relations), and multi-summand (A+B+C+…) |

#### 🌍 Geography

| Chart | What it shows |
|-------|---------------|
| **Pattern Rate by Provenance** | Heatmap — each row is a find site (top 25 by khipu count), each column is a pattern code; cell = % of khipus from that site exhibiting that pattern |

#### 🧮 Pattern Space

| Chart | What it shows |
|-------|---------------|
| **Khipu Pattern-Space (PCA)** | Scatter plot — each dot is one khipu projected from the 9-dimensional boolean flag space onto PC1/PC2; coloured by number of distinct patterns |
| **Pattern Detail Table** | Per-pattern statistics: khipu count, coverage %, average instances per positive khipu, average sum magnitude |

---

### 3D Viewer (`?v=3dviewer`)

Interactive Plotly 3D visualization of a single khipu's cord hierarchy.

**Provenance filter + khipu selector** sit in the top bar. The selected khipu
shows header metrics (KFG ID, pendants, subsidiaries, knots, primary cord length)
and a "View on KFG ↗" link.

**What is rendered:**
- Main cord (horizontal, at top)
- Pendant cords (hanging vertically)
- Subsidiary cords (branching with elbow joints, indented by level)
- Knots shown as shaped markers: ● S-knot · ◆ L-knot · ■ E-knot
- Cords coloured by their Ascher color code

**Mouse controls:**
- Rotate: left-click and drag
- Zoom: mouse wheel or trackpad pinch
- Pan: right-click and drag (or Ctrl + drag)
- Hover: detailed cord/knot info in tooltip

An expandable **Raw cord data** table below the figure lists all cords with their
structural attributes.

---

### Summation Arcs (`?v=arcs`)

2D cord-group map with Bézier arc overlays showing detected summation relations
for a single khipu.

**Provenance filter + khipu selector** sit in the top bar. Header metrics show
KFG ID, pendants, subsidiaries, and cord group count.

**Cord map layout:**
- Circles arranged in a pendant × group grid, coloured by Ascher code
- **Gold ring** = sum cord (the cord whose value equals the sum of others)
- **Cyan ring** = summand cord (contributes to a sum)
- Bézier arcs bow above the grid connecting each sum cord to its summands

**Pattern toggles:** For each detected cord-level pattern (PP, IP, CP, SP), a
card shows the pattern abbreviation, full name, and number of relations. A
checkbox below each card enables or disables that pattern's arcs independently.

Supported arc patterns:

| Code | Full name | Arc colour |
|------|-----------|------------|
| PP | Pendant–Pendant Sum | Blue (`#3b82f6`) |
| IP | Indexed Pendant Sum | Orange (`#f97316`) |
| CP | Colored Pendant Sum | Green (`#22c55e`) |
| SP | Subsidiary–Pendant Sum | Purple (`#a855f7`) |
| IS | Indexed Subsidiary Sum | Rose (`#f43f5e`) |

**Summation relations table:** Lists every arc as a row — pattern code, sum cord
name/value, and the summand cord names/values joined with `+`.

**Group summary table** (expandable): per-group breakdown of cord count, colors
present, cords with numeric values, and total group sum.

> **Note:** GG, GSB, ADG, and PSN do not produce arc overlays — they operate at
> the group level rather than the cord level. Their presence is shown in the Analytics view.

---

## Static Phase Outputs (`visualizations/`)

Each analysis phase script writes PNG figures to a subdirectory. Re-run the
corresponding script to regenerate all figures for that phase.

### Phase 3 — Structural Typology (`visualizations/phase3/`)

```bash
python scripts/run_phase3_typology.py
```

| File | Description |
|------|-------------|
| `silhouette_curve.png` | Silhouette score vs k — used to choose k=2 |
| `heatmap_cluster_patterns.png` | Pattern-type rates by cluster (Simple vs Complex) |
| `pca_by_cluster.png` | PCA scatter coloured by cluster |
| `pca_by_n_types.png` | PCA scatter coloured by number of pattern types |
| `pca_by_region.png` | PCA scatter coloured by geographic region |
| `umap_by_cluster.png` | UMAP projection coloured by cluster |
| `umap_by_n_types.png` | UMAP projection coloured by number of pattern types |
| `umap_by_region.png` | UMAP projection coloured by geographic region |

### Phase 4 — Geographic Patterns (`visualizations/phase4/`)

```bash
python scripts/run_phase4_geography.py
```

| File | Description |
|------|-------------|
| `pattern_heatmap_by_zone.png` | Pattern prevalence rates by geographic zone |
| `complexity_by_zone.png` | Simple vs Complex cluster share per zone |
| `structural_by_zone.png` | Cord count and subsidiary ratio by zone |
| `nn_attribution.png` | Nearest-neighbor provenance attribution confidence |

### Phase 5 — Color Analysis (`visualizations/phase5/`)

```bash
python scripts/run_phase5_color.py
```

| File | Description |
|------|-------------|
| `color_vocab.png` | Top color codes by frequency |
| `color_cooccurrence.png` | Color co-occurrence heatmap |
| `white_cord_analysis.png` | White-cord prevalence by cluster and position |
| `color_diversity_by_cluster.png` | Unique color count: Simple vs Complex khipus |
| `color_value_correlation.png` | Color code vs numeric cord value |

### Phase 6 — Anomaly Detection (`visualizations/phase6/`)

```bash
python scripts/run_phase6_anomaly.py
```

| File | Description |
|------|-------------|
| `anomaly_scatter.png` | Multi-method anomaly scores (2D scatter) |
| `anomaly_profiles.png` | Feature profiles for Normal / Candidate / High-confidence |
| `anomaly_features.png` | Feature importance for anomaly classification |
| `anomaly_method_venn.png` | Agreement between detection methods |

### Phase 7 — Extended Typology (`visualizations/phase7/`)

```bash
python scripts/run_phase7_typology.py
```

| File | Description |
|------|-------------|
| `silhouette_curve.png` | Silhouette score vs k for the Phase 7 clustering |
| `profile_heatmap.png` | Feature profile heatmap for T1 vs T2 typology labels |
| `umap_typology.png` | UMAP projection coloured by typology label |
| `cluster_complexity.png` | Pattern complexity distribution by typology |
| `cluster_zone.png` | Geographic zone distribution by typology |

### Phase 8 — Behavioral Analysis (`visualizations/phase8/`)

```bash
python scripts/run_phase8_behavior.py
```

| File | Description |
|------|-------------|
| `silhouette_curve.png` | Silhouette score vs k for behavioral clustering |
| `behavioral_heatmap.png` | Feature profile heatmap across B1–B6 behavioral clusters |
| `value_register.png` | Value-register distribution per cluster |
| `round_number_zone.png` | Round-number rate by geographic zone |
| `cross_structural.png` | Cross-tabulation: behavioral × structural typology |

### Phase 9 — Graph Topology (`visualizations/phase9/`)

```bash
python scripts/run_phase9_graph.py
```

| File | Description |
|------|-------------|
| `topology_heatmap.png` | Graph metric heatmap across khipus |
| `branching_distribution.png` | Branching factor and entropy distributions |
| `b4_vs_b5_topology.png` | Topology contrast between B4 and B5 behavioral clusters |
| `zone_topology.png` | Graph topology metrics by geographic zone |

### Phase 10 — Summation Compliance (`visualizations/phase10/`)

```bash
python scripts/run_phase10_summation.py
```

| File | Description |
|------|-------------|
| `ratio_distribution.png` | Distribution of summation compliance ratios |
| `compliance_by_cluster.png` | Compliance rates by typology cluster |
| `compliance_predictors.png` | Feature predictors of high/low compliance |
| `zero_cord_patterns.png` | Zero-value cord placement patterns |

### Phase 11 — Color Value (`visualizations/phase11/`)

```bash
python scripts/run_phase11_color.py
```

| File | Description |
|------|-------------|
| `color_value_boxplot.png` | Numeric cord value distribution by color code |
| `color_by_level.png` | Color usage by hierarchy level (pendant vs subsidiary) |
| `attachment_color.png` | Attachment type vs color co-occurrence |
| `color_cluster_heatmap.png` | Color profile heatmap across clusters |
| `color_compliance.png` | Color-pattern compliance rates |

---

## Recommended Exploration Sequence

1. **Start with Corpus Browser** to orient yourself to the 709-khipu dataset.
   Filter by provenance to browse a regional subset.

2. **Open Analytics → Overview** to see which patterns are most prevalent and
   which khipus carry the richest summation structures.

3. **Analytics → Geography** to identify provenances with unusually high or low
   pattern rates — these are the most analytically interesting sites.

4. **Pick a high-pattern khipu** from the Corpus Browser detail modal and
   examine it in **3D Viewer**, then switch to **Summation Arcs** to see its
   summation relations overlaid on the cord grid.

5. **Review static PNGs** under `visualizations/` for publication-quality
   versions of the corpus-wide statistical findings from each phase.
