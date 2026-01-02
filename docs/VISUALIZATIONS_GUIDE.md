# Advanced Visualizations Guide

This guide explains how to use the advanced visualization tools created for the Khipu Analysis project.

## Configuration Note

All scripts use the centralized configuration system. See [DATA_PATHS.md](../DATA_PATHS.md) for complete setup instructions.

**Quick setup validation:**
```bash
python src/config.py
```

## 🌐 Interactive Web Dashboard

**File:** `scripts/dashboard_app.py`

A comprehensive Streamlit web application for exploring khipu data with real-time filtering and interactive visualizations.

![Dashboard Overview](images/dashboard_overview.png)
*The main dashboard interface showing PCA scatter plot and cluster distribution*

### Features:
- **Real-time Filtering:** Select clusters, provenances, size ranges, and summation patterns
- **Multi-tab Interface:**
  - **Overview:** PCA scatter, size vs depth, cluster distribution
  - **Geographic:** Interactive Andes region map showing all 612 khipus across 15+ locations, summation rates by provenance, structural features, enrichment heatmap
  - **Clusters:** Detailed cluster analysis with feature distributions
  - **Features:** Correlation analysis and feature relationships
- **Data Export:** Download filtered data and summary statistics as CSV
- **Geographic Map:** Plotly scatter_geo showing complete distribution with fuzzy provenance matching

### Database Access:

The dashboard requires access to the Open Khipu Repository database for provenance data:

- **Default:** Uses `../open-khipu-repository/data/khipu.db` (sibling directory)
- **Custom location:** Set `KHIPU_DB_PATH` environment variable
- **Fallback:** If database not found, provenance shows as "Unknown" but other features work

See [DATA_PATHS.md](../DATA_PATHS.md) for configuration details.

### Usage:
```bash
streamlit run scripts/dashboard_app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

### Controls:
- Use the **sidebar** to filter data by cluster, provenance, size, and summation pattern
- Switch between **tabs** to explore different aspects of the data
- **Hover** over plot elements for detailed information
- Use **export buttons** at the bottom to download filtered data

![Dashboard Geographic Tab](images/dashboard_geographic.png)
*Geographic tab showing khipu distribution across the Andes region*

---

## 📐 3D Khipu Structure Viewer

### Command-Line Viewer

**File:** `scripts/visualize_3d_khipu.py`

Creates interactive 3D visualizations of individual khipu hierarchical structures.

**Features:**
- Hierarchical layout showing parent-child cord relationships
- Interactive rotation and zoom (in matplotlib window)
- Three color modes:
  - `value`: Color by numeric value (gradient)
  - `level`: Color by hierarchy level
  - `color`: Uniform coloring
- Multiple viewing angles
- Summation flow visualization (highlights summation relationships in red)

**Usage:**

**Basic visualization:**
```bash
python scripts/visualize_3d_khipu.py --khipu-id 1000000
```

**Note:** Khipu IDs start at 1000000 (not 1). Use the interactive viewer (below) for easier browsing.

### 🌟 Interactive 3D Viewer (NEW)

**File:** `scripts/interactive_3d_viewer.py`
![3D Viewer Interface](images/3d_viewer_interface.png)
*Interactive 3D viewer showing hierarchical khipu structure with color-coded cord levels*


Streamlit-based web interface for browsing 3D khipu structures with dropdown selection.

**Features:**
- **Dropdown menu** with all 612 khipus (no command-line arguments needed!)
- **Live statistics** panel showing cord count, hierarchy depth, numeric values
- **Interactive controls:** Elevation and azimuth sliders to rotate view
- **Color modes:** Switch between numeric value and hierarchy level coloring
- **Provenance display:** See location and cord count for each khipu
- **Auto-refresh:** Visualization updates instantly when you change selections

**Requirements:**
- Database file at `data/khipu.db` (for khipu list and provenance)

**Usage:**
```bash
streamlit run scripts/interactive_3d_viewer.py --server.port 8502
```

The viewer will open at `http://localhost:8502`

**Interface:**
- **Left sidebar:** Select khipu from dropdown, choose color mode, adjust view angles
- **Main panel:** Interactive 3D visualization
- **Right panel:** Real-time statistics and hierarchy level distribution

**Why use this instead of command-line?**
- No need to remember khipu IDs
- Browse through all khipus quickly
![3D Khipu Detail](images/3d_khipu_detail.png)
*Example khipu showing hierarchical cord structure with summation relationships*

- See metadata (provenance, cord count) before viewing
- Instant visual feedback when adjusting parameters

**With specific color mode:**
```bash
python scripts/visualize_3d_khipu.py --khipu-id 1000000 --color-mode level
```

**Save to file:**
```bash
python scripts/visualize_3d_khipu.py --khipu-id 1000000 --output outputs/khipu_1000000.png
```

**Multi-view (4 angles):**
```bash
python scripts/visualize_3d_khipu.py --khipu-id 1000000 --multi-view
```

**Summation flow (highlight summation edges):**
```bash
python scripts/visualize_3d_khipu.py --khipu-id 1000000 --summation-flow
```

### Running Multiple Viewers

You can run both the dashboard and 3D viewer simultaneously:

```bash
# Terminal 1: Main dashboard
streamlit run scripts/dashboard_app.py

# Terminal 2: 3D viewer
streamlit run scripts/interactive_3d_viewer.py --server.port 8502
```

- Dashboard: http://localhost:8501
- 3D Viewer: http://localhost:8502

This allows you to browse the dataset in the dashboard, then inspect interesting khipus in the 3D viewer.

---

![Geographic Heatmap](images/geographic_heatmap.png)
*Interactive map showing summation rate intensity across Andean archaeological sites*

## 🗺️ Geographic Heatmap

**File:** `scripts/visualize_geographic_heatmap.py`

Creates interactive maps showing geographic distribution of khipu patterns across archaeological sites in the Andes region.

### Features:
- **Heatmap overlay:** Intensity based on summation rate
- **Interactive markers:** Click for detailed statistics per provenance
- **Color-coded bubbles:**
  - 🔴 Red: >40% summation rate
  - 🟠 Orange: 25-40% summation rate
  - 🔵 Blue: <25% summation rate
- **Bubble size:** Proportional to number of khipus
- **Two maps:**
  1. Summation heatmap (pattern intensity)
  2. Cluster distribution (dominant archetype per region)

### Requirements:
- Database file at `data/khipu.db` (for provenance data)

### Usage:
```bash
python scripts/visualize_geographic_heatmap.py
```

**Outputs:**
- `outputs/visualizations/geographic_heatmap.html` - Summation rate heatmap
![Cluster Geographic Distribution](images/cluster_geographic_map.png)

*Cluster distribution map showing dominant khipu archetypes by region*

- `outputs/visualizations/geographic_heatmap_statistics.csv` - Aggregated statistics
- `outputs/visualizations/cluster_geographic_map.html`
*Cluster distribution map*

### Viewing:
Open the `.html` files in any web browser. The maps are fully interactive:
- **Zoom:** Mouse wheel or +/- buttons
- **Pan:** Click and drag
- **Info:** Click markers for detailed statistics

---

## 🎨 Visualization Workflow

### Recommended Exploration Sequence:

1. **Start with the Dashboard** to get an overview:
   ```bash
   streamlit run scripts/dashboard_app.py
   ```
   - Filter to specific clusters or provenances of interest
   - Export filtered data for focused analysis

2. **Explore Geographic Patterns**:
   ```bash
   python scripts/visualize_geographic_heatmap.py
   ```
   - Identify regional variations in summation patterns
   - Note provenances with high summation rates

3. **Deep Dive into Individual Khipus**:
   ```bash
   python scripts/visualize_3d_khipu.py --khipu-id <ID> --multi-view
   python scripts/visualize_3d_khipu.py --khipu-id <ID> --summation-flow
   ```
   - Choose khipus from interesting clusters/provenances
   - Examine hierarchical structure in 3D
   - Visualize summation relationships

4. **Use Interactive Notebooks** for hypothesis testing:
   - Open `notebooks/03_khipu_detail_viewer.ipynb` for comprehensive khipu analysis
   - Open `notebooks/04_hypothesis_dashboard.ipynb` for custom statistical tests

---

## 📊 Data Sources

All visualizations use processed data from:
- `data/khipu.db` - SQLite database with khipu metadata and provenance
- `data/processed/phase1/cord_numeric_values.csv` - Numeric values
- `data/processed/phase2/cord_hierarchy.csv` - Hierarchical structure (54,404 cords)
- `data/processed/phase3/summation_test_results.csv` - Summation testing
- `data/processed/phase4/cluster_assignments_kmeans.csv` - 7 archetypes (613 khipus)
- `data/processed/phase4/cluster_pca_coordinates.csv` - PCA projections
- `data/processed/phase4/graph_structural_features.csv` - Structural metrics

---

## 🛠️ Technical Requirements

### Python Packages:
- `streamlit` - Web dashboard framework
- `plotly` - Interactive plotting
- `folium` - Interactive maps
- `matplotlib` - 3D visualization
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `networkx` - Graph analysis
- `statsmodels` - Statistical analysis
- `sqlite3` - Database access (standard library)

Install required packages:
```bash
pip install streamlit plotly folium matplotlib pandas numpy networkx statsmodels
```

### System Requirements:
- **RAM:** 2GB minimum (4GB recommended for dashboard)
- **Browser:** Modern browser (Chrome, Firefox, Edge) for HTML maps
- **Display:** 1920×1080 recommended for dashboard

---

## 🎯 Use Cases

### Archaeological Research:
- Compare khipu structures across regions
- Identify regional traditions and variations
- Discover outliers and unique specimens

### Data Quality:
- Visual inspection of complex hierarchies
- Verification of summation patterns
- Identification of potential transcription errors

### Publications:
- High-resolution 3D visualizations (300 DPI)
- Interactive supplements (HTML maps)
- Summary statistics for tables

### Teaching:
- Interactive demonstrations of khipu structure
- Real-time exploration in classroom settings
- Hands-on data analysis exercises

---

## 📝 Notes

- **Dashboard:** First launch may take 10-15 seconds to load data
- **3D Viewer:** Matplotlib 3D is interactive - click and drag to rotate
- **Maps:** Require internet connection for base map tiles
- **Export:** All visualizations can be saved as high-resolution images (PNG, 300 DPI)

---

## 🐛 Troubleshooting

**Dashboard won't start:**
- Ensure port 8501 is available: `netstat -ano | findstr :8501`
- Try alternate port: `streamlit run scripts/dashboard_app.py --server.port 8502`
- Kill existing Streamlit: `Stop-Process -Name streamlit -Force`

**3D viewer shows empty plot:**
- Use interactive viewer (`interactive_3d_viewer.py`) instead of command-line version
- Khipu IDs start at 1000000, not 1
- Verify khipu ID exists: Check dropdown list in interactive viewer
- Ensure khipu has cord data (not one of the 7 filtered empty records)

**Geographic map shows only 3 locations:**
- This was a bug - now fixed! Map uses full dataset, not filtered data
- Fuzzy matching covers 15+ locations (Pachacamac, Ica, Nazca, Leymebamba, etc.)
- Refresh dashboard if you still see old version
- Empty provenance strings are filtered out by design

**Database errors:**
- Ensure `data/khipu.db` exists (copy from Open Khipu Repository)
- Expected path: `C:\code\khipu-computational-toolkit\data\khipu.db`
- Without database, visualizations will work but provenance will show "Unknown"
- Download OKR database from: https://zenodo.org/record/5037551

**Maps show no data:**
- Check that geographic heatmap script completed successfully
- Verify provenance names match `PROVENANCE_LOCATIONS` dictionary
- Dashboard map uses fuzzy matching and shows ~400+ khipus across locations
- Requires database file with provenance information

**Memory issues:**
- Close other applications
- Filter to smaller subsets in dashboard
- Process khipus individually in 3D viewer

---

## 🚀 Next Steps

After exploring these visualizations, consider:
1. **ML Extensions** (Task 7):
   - Function prediction (accounting vs narrative)
   - Anomaly detection (outliers and errors)
   - Sequence prediction for restoration

2. **Custom Analysis:**
   - Modify dashboard to add new visualizations
   - Create animated sequences in 3D viewer
   - Add temporal dimension if dating data available

3. **Publication:**
   - Export high-resolution figures
   - Create interactive supplements
   - Generate summary tables

---

*For questions or issues, refer to the main project documentation in `README_FORK.md`*
