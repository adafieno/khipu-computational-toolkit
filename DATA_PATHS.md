# Data Paths and Configuration

**Single Source of Truth for File Locations**

This document defines the canonical file path structure used throughout the Khipu Computational Toolkit.

## Configuration System

All scripts use **centralized path management** via `src/config.py`. This ensures consistency and allows flexible deployment.

### Database Path Resolution

The Open Khipu Repository database path is resolved in this order:

1. **Environment Variable** (highest priority):
   ```bash
   export KHIPU_DB_PATH="/path/to/your/khipu.db"
   ```
   
2. **Default Path** (if env var not set):
   ```
   ../open-khipu-repository/data/khipu.db
   ```
   This assumes the standard sibling directory structure:
   ```
   your-projects-directory/
   ├── open-khipu-repository/
   │   └── data/
   │       └── khipu.db
   └── khipu-computational-toolkit/
       ├── src/
       ├── scripts/
       └── data/
   ```

### Validating Your Configuration

Run the configuration validator:

```bash
python src/config.py
```

This will check:
- ✅ Database file exists and is accessible
- ✅ Required directories are present
- ⚠️  Any configuration warnings

## Directory Structure

### Input Data

| Path | Description | Source |
|------|-------------|--------|
| `../open-khipu-repository/data/khipu.db` | SQLite database | External (OKR) |

### Processed Data

Scripts write extracted/processed data to phase-specific directories:

| Path | Phase | Contents |
|------|-------|----------|
| `data/processed/` | All | Root for processed outputs |
| `data/processed/phase1/` | Phase 1 | Numeric decoding results |
| `data/processed/phase2/` | Phase 2 | Hierarchical structures, color data |
| `data/processed/phase3/` | Phase 3 | Summation test results |
| `data/processed/phase4/` | Phase 4 | Clusters, motifs, anomalies |
| `data/processed/phase5/` | Phase 5 | Multi-model hypothesis results |
| `data/processed/phase7/` | Phase 7 | ML predictions and classifications |
| `data/processed/phase8/` | Phase 8 | Administrative function analysis |
| `data/processed/phase9/` | Phase 9 | Stability and robustness tests |

**Note:** Some legacy scripts may read from the root `data/processed/` directory for convenience. See "Cross-Phase File Access" below.

### Graph Data

| Path | Description |
|------|-------------|
| `data/graphs/` | NetworkX graph structures (pickled) |
| `data/graphs/khipu_graphs.pkl` | Individual khipu graphs |
| `data/graphs/similarity_matrix.npy` | Graph similarity computations |

### Outputs

| Path | Description |
|------|-------------|
| `outputs/` | Root for all generated outputs |
| `outputs/visualizations/` | Plot images and interactive HTML |
| `models/` | Trained ML models (.pkl files) |
| `visualizations/phase*/` | Phase-specific visualization collections |

### Reports

| Path | Description |
|------|-------------|
| `reports/` | Markdown reports for each phase |
| `reports/phase0_reconnaissance_report.md` | Database exploration |
| `reports/phase1_baseline_validation_report.md` | Numeric decoding validation |
| ... | (9 phase reports total) |

## Key Data Files

### Phase 2 Extraction Outputs

These are the most commonly referenced files across scripts:

| File | Location | Rows | Description |
|------|----------|------|-------------|
| `cord_hierarchy.csv` | `data/processed/phase2/` | 54,403 | Parent-child relationships, depth |
| `cord_numeric_values.csv` | `data/processed/phase2/` | 54,403 | Decoded numeric values per cord |
| `knot_data.csv` | `data/processed/phase2/` | 110,151 | Individual knot records |
| `color_data.csv` | `data/processed/phase2/` | 56,306 | Color codes per cord |

**Important:** These Phase 2 files are also copied to `data/processed/` (root) for backward compatibility with some analysis scripts.

### Phase 4 Analysis Outputs

| File | Location | Description |
|------|----------|-------------|
| `khipu_clusters.csv` | `data/processed/phase4/` | K-means cluster assignments |
| `anomalies.csv` | `data/processed/phase4/` | Detected structural anomalies |
| `motifs.csv` | `data/processed/phase4/` | Recurring subgraph patterns |

### Phase 7 ML Outputs

| File | Location | Description |
|------|----------|-------------|
| `predicted_values.csv` | `data/processed/phase7/` | Missing value predictions |
| `function_predictions.csv` | `data/processed/phase7/` | Administrative function classifications |
| `missing_value_predictor.pkl` | `models/` | Trained Random Forest model |
| `function_classifier.pkl` | `models/` | Trained classification model |

## Cross-Phase File Access

Some scripts need to read outputs from earlier phases. Standard pattern:

```python
from src.config import get_config

config = get_config()

# Read from specific phase
hierarchy_path = config.get_processed_file('cord_hierarchy.csv', phase=2)
hierarchy = pd.read_csv(hierarchy_path)

# Read from root processed directory (backward compatibility)
legacy_path = config.get_processed_file('cord_hierarchy.csv')
data = pd.read_csv(legacy_path)
```

## Using Configuration in Your Scripts

### Standard Import Pattern

```python
from src.config import get_config

# Initialize config
config = get_config()

# Get database path
db_path = config.get_database_path()

# Get processed file from specific phase
cord_hierarchy = pd.read_csv(
    config.get_processed_file('cord_hierarchy.csv', phase=2)
)

# Get output path
output_path = config.get_output_file('result.csv', subdir='visualizations')
```

### Creating Directories

```python
config = get_config()
config.ensure_directories()  # Creates all necessary directories
```

### Validation in Scripts

```python
from src.config import get_config

config = get_config()
results = config.validate_setup()

if not results['valid']:
    print("Configuration errors:")
    for error in results['errors']:
        print(f"  • {error}")
    sys.exit(1)
```

## Environment Setup Examples

### Standard Setup (Recommended)

```bash
# Directory structure is correct, no environment variables needed
cd khipu-computational-toolkit
python scripts/extract_cord_hierarchy.py
```

### Custom Database Location

```bash
# Database is in a non-standard location
export KHIPU_DB_PATH="/custom/path/to/khipu.db"
python scripts/extract_cord_hierarchy.py
```

### Windows PowerShell

```powershell
# Set environment variable in PowerShell
$env:KHIPU_DB_PATH = "C:\custom\path\to\khipu.db"
python scripts\extract_cord_hierarchy.py
```

## Troubleshooting

### "Database not found" Error

**Solution 1:** Check directory structure
```bash
ls ../open-khipu-repository/data/khipu.db
```

**Solution 2:** Set environment variable
```bash
export KHIPU_DB_PATH="/path/to/khipu.db"
```

**Solution 3:** Run validation
```bash
python src/config.py
```

### "Processed file not found" Error

Files are created by extraction scripts. Run in order:
1. `python scripts/extract_cord_hierarchy.py` (Phase 2)
2. `python scripts/extract_knot_data.py` (Phase 2)
3. `python scripts/extract_color_data.py` (Phase 2)

Or use the batch generation:
```bash
python scripts/generate_processed_data.py
```

### Inconsistent Paths Between Scripts

If you encounter hardcoded paths in older scripts, they should be updated to use `src/config.py`. Please report these as issues or update them to use the configuration system.

## Migration Guide for Legacy Scripts

If you have custom scripts with hardcoded paths:

**Before:**
```python
hierarchy = pd.read_csv("data/processed/cord_hierarchy.csv")
conn = sqlite3.connect("../open-khipu-repository/data/khipu.db")
```

**After:**
```python
from src.config import get_config

config = get_config()
hierarchy = pd.read_csv(config.get_processed_file('cord_hierarchy.csv', phase=2))
conn = sqlite3.connect(config.get_database_path())
```

## Related Documentation

- [src/config.py](src/config.py) - Configuration implementation
- [docs/DATA_RECONCILIATION.md](docs/DATA_RECONCILIATION.md) - Explanation of count differences across phases
- [README.md](README.md) - Quick start and setup guide
- [scripts/README.md](scripts/README.md) - Individual script documentation
