# API Reference

Comprehensive reference for the Khipu Computational Toolkit Python modules.

## Table of Contents

- [Core Modules](#core-modules)
  - [extraction](#extraction) - Database extraction tools
  - [analysis](#analysis) - Statistical analysis and testing
  - [graph](#graph) - Graph construction and algorithms
  - [numeric](#numeric) - Numeric constraint solving
  - [patterns](#patterns) - Pattern discovery
  - [hypothesis](#hypothesis) - Hypothesis testing
  - [visualization](#visualization) - Visualization tools
  - [utils](#utils) - Utility functions

---

## extraction

Module for extracting data from the Open Khipu Repository database.

### `khipu_loader.py`

**KhipuLoader**

Main interface for loading khipu data from SQLite database.

```python
from src.extraction.khipu_loader import KhipuLoader

loader = KhipuLoader(db_path="../open-khipu-repository/data/khipu.db")
```

**Methods:**

- `get_all_khipus()` → `pd.DataFrame`
  - Returns all khipus with metadata
  - Columns: khipu_id, provenance, museum, date, etc.

- `get_khipu_by_id(khipu_id: int)` → `dict`
  - Returns single khipu with complete data
  - Includes cords, knots, colors

- `get_cords(khipu_id: int = None)` → `pd.DataFrame`
  - Returns cord hierarchy for khipu(s)
  - Columns: cord_id, khipu_id, pendant_from, attached_to, cord_level

- `get_knots(khipu_id: int = None)` → `pd.DataFrame`
  - Returns knots with types and values
  - Columns: knot_id, cord_id, knot_type, position, value

- `get_colors(khipu_id: int = None)` → `pd.DataFrame`
  - Returns color specifications
  - Columns: cord_id, color_cd_1, color_cd_2, rgb_1, rgb_2

### `cord_extractor.py`

**CordExtractor**

Extracts hierarchical cord structure with validation.

```python
from src.extraction.cord_extractor import CordExtractor

extractor = CordExtractor(db_path="...")
hierarchy = extractor.extract_hierarchy()
```

**Methods:**

- `extract_hierarchy()` → `pd.DataFrame`
  - Builds parent-child relationships
  - Validates attachment integrity
  - Returns: cord_id, parent_id, level, attachment_confidence

- `validate_hierarchy(hierarchy: pd.DataFrame)` → `dict`
  - Checks for orphaned cords, cycles, invalid levels
  - Returns validation report with issues

### `knot_extractor.py`

**KnotExtractor**

Decodes numeric values from knot data.

```python
from src.extraction.knot_extractor import KnotExtractor

extractor = KnotExtractor(db_path="...")
numeric_values = extractor.extract_numeric_values()
```

**Methods:**

- `extract_numeric_values()` → `pd.DataFrame`
  - Decodes knot types to decimal values
  - Columns: cord_id, numeric_value, confidence, num_knots

- `decode_knot_cluster(cluster: list)` → `float`
  - Converts knot cluster to decimal value
  - Handles positional notation (S, L, E)

### `color_extractor.py`

**ColorExtractor**

Extracts and normalizes color data.

```python
from src.extraction.color_extractor import ColorExtractor

extractor = ColorExtractor(db_path="...")
colors = extractor.extract_colors()
```

**Methods:**

- `extract_colors()` → `pd.DataFrame`
  - Returns color specifications with RGB
  - Handles multi-color cords (up to 6 colors)

- `normalize_color_codes(colors: pd.DataFrame)` → `pd.DataFrame`
  - Maps color codes to standardized palette
  - Returns normalized codes with RGB values

- `identify_white_cords(colors: pd.DataFrame)` → `pd.DataFrame`
  - Finds white/blank cord boundary markers
  - Returns cord_ids with white designation

---

## analysis

Statistical analysis and hypothesis testing modules.

### `summation_tester.py`

**SummationTester**

Tests pendant-to-parent summation hypotheses.

```python
from src.analysis.summation_tester import SummationTester

tester = SummationTester()
results = tester.test_summation(khipu_graph, numeric_values)
```

**Methods:**

- `test_summation(graph: nx.DiGraph, values: dict)` → `dict`
  - Tests if pendant values sum to parent
  - Returns: match_rate, total_tests, matches, mismatches

- `test_hierarchical_summation(graph: nx.DiGraph, values: dict)` → `dict`
  - Tests multi-level summation patterns
  - Returns summation depth and consistency

- `compute_match_percentage(results: dict)` → `float`
  - Calculates percentage of matching summations
  - Range: 0.0 to 1.0

### `administrative_function_classifier.py`

**AdministrativeFunctionClassifier**

Classifies khipus by administrative function (Phase 8).

```python
from src.analysis.administrative_function_classifier import AdministrativeFunctionClassifier

classifier = AdministrativeFunctionClassifier()
classifier.train(features, labels)
predictions = classifier.predict(test_features)
```

**Methods:**

- `train(X: pd.DataFrame, y: pd.Series)` → `None`
  - Trains Random Forest classifier
  - Features: structure + color + numeric

- `predict(X: pd.DataFrame)` → `np.ndarray`
  - Predicts administrative function
  - Classes: Simple Linear, Standard Hierarchical, Complex, etc.

- `get_feature_importance()` → `pd.DataFrame`
  - Returns feature importance scores
  - Sorted by contribution to classification

### `phase9/` modules

Phase 9 meta-analysis modules (10 total):

- **`information_capacity.py`** - Entropy and information metrics
- **`robustness_analysis.py`** - Perturbation testing
- **`cognitive_load.py`** - Visual complexity scoring
- **`minimalism_expressiveness.py`** - Efficiency analysis
- **`variance_mapping.py`** - Feature variance analysis
- **`boundary_phenomena.py`** - Edge case detection
- **`anomaly_taxonomy.py`** - Anomaly categorization
- **`randomness_testing.py`** - Statistical randomness tests
- **`stability_testing.py`** - Feature ablation and cross-validation
- **`negative_knowledge.py`** - Negative findings documentation

Each module follows similar pattern:
```python
from src.analysis.phase9.module_name import AnalysisClass

analyzer = AnalysisClass()
results = analyzer.run_analysis()
```

---

## graph

Graph construction and analysis tools.

### `graph_builder.py`

**KhipuGraphBuilder**

Converts khipus to NetworkX graphs.

```python
from src.graph.graph_builder import KhipuGraphBuilder

builder = KhipuGraphBuilder()
graph = builder.build_graph(cords, numeric_values, colors)
```

**Methods:**

- `build_graph(cords: pd.DataFrame, values: dict, colors: pd.DataFrame)` → `nx.DiGraph`
  - Creates directed graph from cord hierarchy
  - Nodes: cords with attributes (value, color, level)
  - Edges: parent-child relationships

- `add_numeric_attributes(graph: nx.DiGraph, values: dict)` → `None`
  - Adds numeric_value attribute to nodes
  - Handles missing values gracefully

- `add_color_attributes(graph: nx.DiGraph, colors: pd.DataFrame)` → `None`
  - Adds color_code and rgb attributes
  - Supports multi-color cords

- `compute_structural_features(graph: nx.DiGraph)` → `dict`
  - Returns: num_nodes, depth, width, density, avg_branching

**Graph Attributes:**

Node attributes:
- `cord_id`: Unique cord identifier
- `level`: Hierarchical level (0=primary, 1=pendant, etc.)
- `numeric_value`: Decoded numeric value (or None)
- `color_code`: Primary color code
- `rgb`: RGB tuple
- `is_white`: Boolean for white boundary cords

Edge attributes:
- `relationship`: 'pendant' or 'subsidiary'

---

## numeric

Numeric constraint solving and inference.

### `constraint_solver.py`

**ConstraintSolver**

Infers missing values using summation constraints.

```python
from src.numeric.constraint_solver import ConstraintSolver

solver = ConstraintSolver()
predictions = solver.solve_missing_values(graph, known_values)
```

**Methods:**

- `solve_missing_values(graph: nx.DiGraph, known: dict)` → `dict`
  - Infers missing values from summation rules
  - Returns: {cord_id: predicted_value, ...}

- `apply_summation_constraint(parent_id: int, children: list, known: dict)` → `float | None`
  - Calculates missing parent from children or vice versa
  - Returns predicted value or None if unsolvable

---

## patterns

Pattern discovery and motif mining.

### `motif_miner.py`

**MotifMiner**

Discovers recurring structural patterns.

```python
from src.patterns.motif_miner import MotifMiner

miner = MotifMiner()
motifs = miner.mine_motifs(graphs)
```

**Methods:**

- `mine_motifs(graphs: list[nx.DiGraph])` → `dict`
  - Finds recurring subgraph patterns
  - Returns: motif_id, frequency, structure

- `extract_subtree(graph: nx.DiGraph, root: int, depth: int)` → `nx.DiGraph`
  - Extracts subtree rooted at node
  - Used for motif comparison

### `template_extractor.py`

**TemplateExtractor**

Identifies structural templates.

```python
from src.patterns.template_extractor import TemplateExtractor

extractor = TemplateExtractor()
templates = extractor.extract_templates(graphs, threshold=0.95)
```

**Methods:**

- `extract_templates(graphs: list, threshold: float)` → `pd.DataFrame`
  - Finds perfect or near-perfect structural matches
  - Returns template definitions with examples

---

## hypothesis

Hypothesis testing framework.

### `color_hypothesis_tester.py`

**ColorHypothesisTester**

Tests color semantics hypotheses (Phase 5).

```python
from src.hypothesis.color_hypothesis_tester import ColorHypothesisTester

tester = ColorHypothesisTester()
results = tester.test_white_boundary_hypothesis(khipus, summation_results)
```

**Methods:**

- `test_white_boundary_hypothesis(khipus: pd.DataFrame, summation: pd.DataFrame)` → `dict`
  - Tests if white cords mark boundaries
  - Returns statistical significance and effect size

- `test_color_value_correlation(colors: pd.DataFrame, values: pd.DataFrame)` → `dict`
  - Tests if colors correlate with numeric values
  - Returns correlation coefficient and p-value

---

## visualization

Visualization tools (used by scripts/).

### `cluster_visualizer.py`

**ClusterVisualizer**

Creates cluster analysis plots.

```python
from src.visualization.cluster_visualizer import ClusterVisualizer

viz = ClusterVisualizer()
viz.plot_pca_clusters(features, labels, output_path="clusters.png")
```

**Methods:**

- `plot_pca_clusters(X: pd.DataFrame, labels: np.ndarray, output: str)` → `None`
  - Creates 2D PCA plot with cluster colors
  - Saves to file

- `plot_cluster_statistics(stats: dict, output: str)` → `None`
  - Creates bar chart of cluster profiles
  - Shows mean features per cluster

---

## utils

Utility functions and helpers.

### `arithmetic_validator.py`

**ArithmeticValidator**

Validates numeric consistency.

```python
from src.utils.arithmetic_validator import ArithmeticValidator

validator = ArithmeticValidator()
is_valid = validator.validate_summation(parent_value, children_values, tolerance=0.01)
```

**Methods:**

- `validate_summation(parent: float, children: list[float], tolerance: float)` → `bool`
  - Checks if children sum to parent within tolerance
  - Returns True if valid

- `compute_match_confidence(observed: float, expected: float)` → `float`
  - Calculates confidence score (0-1)
  - Based on relative difference

### `database_utils.py`

**Utility Functions**

Database connection and query helpers.

```python
from src.utils.database_utils import connect_db, execute_query

conn = connect_db(db_path)
results = execute_query(conn, "SELECT * FROM khipu WHERE provenance = ?", ("Pachacamac",))
```

**Functions:**

- `connect_db(path: str)` → `sqlite3.Connection`
  - Opens database connection with error handling

- `execute_query(conn: Connection, query: str, params: tuple)` → `list[dict]`
  - Executes parameterized query safely
  - Returns results as list of dicts

---

## Usage Examples

### Complete Analysis Pipeline

```python
# 1. Load data
from src.extraction.khipu_loader import KhipuLoader
loader = KhipuLoader(db_path="../open-khipu-repository/data/khipu.db")
khipus = loader.get_all_khipus()

# 2. Extract features
from src.extraction.cord_extractor import CordExtractor
from src.extraction.knot_extractor import KnotExtractor
cord_extractor = CordExtractor(db_path)
knot_extractor = KnotExtractor(db_path)

cords = cord_extractor.extract_hierarchy()
numeric_values = knot_extractor.extract_numeric_values()

# 3. Build graphs
from src.graph.graph_builder import KhipuGraphBuilder
builder = KhipuGraphBuilder()
graphs = {}
for khipu_id in khipus['khipu_id']:
    khipu_cords = cords[cords['khipu_id'] == khipu_id]
    khipu_values = numeric_values[numeric_values['khipu_id'] == khipu_id]
    graphs[khipu_id] = builder.build_graph(khipu_cords, khipu_values)

# 4. Test hypotheses
from src.analysis.summation_tester import SummationTester
tester = SummationTester()
results = {}
for khipu_id, graph in graphs.items():
    results[khipu_id] = tester.test_summation(graph, numeric_values)

# 5. Classify function
from src.analysis.administrative_function_classifier import AdministrativeFunctionClassifier
classifier = AdministrativeFunctionClassifier()
# ... train and predict

# 6. Visualize
from src.visualization.cluster_visualizer import ClusterVisualizer
viz = ClusterVisualizer()
viz.plot_pca_clusters(features, labels, "output.png")
```

### Phase 9 Meta-Analysis

```python
# Run all Phase 9 modules
from src.analysis.phase9 import (
    InformationCapacity,
    RobustnessAnalyzer,
    CognitiveLoadAnalyzer,
    # ... etc
)

# Information capacity
capacity = InformationCapacity()
capacity_results = capacity.run_analysis()

# Robustness
robustness = RobustnessAnalyzer()
robustness_results = robustness.run_analysis()

# ... continue with other modules
```

---

## Module Dependencies

```
extraction/
  └─ (depends on: sqlite3, pandas)

graph/
  └─ (depends on: extraction, networkx)

numeric/
  └─ (depends on: graph, numpy)

analysis/
  └─ (depends on: graph, numeric, sklearn)

patterns/
  └─ (depends on: graph, analysis)

hypothesis/
  └─ (depends on: analysis, scipy.stats)

visualization/
  └─ (depends on: analysis, matplotlib, seaborn)
```

---

## Error Handling

All modules raise informative exceptions:

- `DatabaseConnectionError` - Cannot connect to khipu.db
- `DataValidationError` - Invalid or inconsistent data
- `GraphConstructionError` - Cannot build valid graph
- `InsufficientDataError` - Not enough data for analysis

Example:
```python
from src.extraction.khipu_loader import KhipuLoader, DatabaseConnectionError

try:
    loader = KhipuLoader(db_path="invalid.db")
except DatabaseConnectionError as e:
    print(f"Failed to connect: {e}")
```

---

## Configuration

Most modules accept configuration dictionaries:

```python
config = {
    'summation_tolerance': 0.01,
    'min_confidence': 0.8,
    'enable_validation': True
}

tester = SummationTester(config=config)
```

See individual module docstrings for available configuration options.

---

## Testing

Run unit tests for modules:

```bash
pytest tests/test_extraction.py
pytest tests/test_graph.py
pytest tests/test_analysis.py
```

---

## Contributing

When adding new modules:
1. Follow existing naming conventions
2. Add type hints for all functions
3. Include docstrings with examples
4. Add unit tests
5. Update this API reference

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

MIT License - See [LICENSE](../LICENSE) for details.
