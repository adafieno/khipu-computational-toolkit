# System Architecture

Technical architecture of the Khipu Computational Toolkit.

## Table of Contents

- [System Overview](#system-overview)
- [Module Architecture](#module-architecture)
- [Data Flow](#data-flow)
- [Phase Pipeline](#phase-pipeline)
- [Design Patterns](#design-patterns)
- [Technology Stack](#technology-stack)
- [Extension Points](#extension-points)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACES                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  CLI Scripts │  │  Dashboard   │  │  Notebooks   │       │
│  │  (scripts/)  │  │  (Streamlit) │  │  (Jupyter)   │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────┐
│                      CORE MODULES (src/)                    │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    ANALYSIS LAYER                     │  │
│  │  ┌──────────┐  ┌──────────┐   ┌──────────────┐        │  │
│  │  │Summation │  │Clustering│   │Classification│        │  │
│  │  │ Testing  │  │          │   │              │        │  │
│  │  └─────┬────┘  └────┬─────┘   └──────┬───────┘        │  │
│  └────────┼────────────┼────────────────┼────────────────┘  │
│           │            │                │                   │
│  ┌────────┴────────────┴────────────────┴──────────────┐    │
│  │              PATTERN DISCOVERY LAYER                │    │
│  │  ┌──────────┐  ┌──────────┐   ┌──────────────┐      │    │
│  │  │  Motif   │  │Template  │   │  Similarity  │      │    │
│  │  │  Mining  │  │Extraction│   │   Analysis   │      │    │
│  │  └─────┬────┘  └────┬─────┘   └──────┬───────┘      │    │
│  └────────┼────────────┼────────────────┼──────────────┘    │
│           │            │                │                   │
│  ┌────────┴────────────┴────────────────┴──────────────┐    │
│  │                GRAPH LAYER                          │    │
│  │  ┌──────────────────────────────────────────────┐   │    │
│  │  │          KhipuGraphBuilder                   │   │    │
│  │  │  - Hierarchical graph construction           │   │    │
│  │  │  - Structural feature computation            │   │    │
│  │  └────────────────┬─────────────────────────────┘   │    │
│  └───────────────────┼─────────────────────────────────┘    │
│                      │                                      │
│  ┌───────────────────┴────────────────────────────────┐     │
│  │              EXTRACTION LAYER                      │     │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │     │
│  │  │ Cord    │  │ Knot    │  │ Color   │  │ Khipu  │ │     │
│  │  │Extractor│  │Extractor│  │Extractor│  │ Loader │ │     │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬───┘ │     │
│  └───────┼────────────┼────────────┼───────────┼──────┘     │
└──────────┼────────────┼────────────┼───────────┼────────────┘
           │            │            │           │
           └────────────┴────────────┴───────────┘
                        │
┌───────────────────────┴──────────────────────────────────┐
│                  DATA LAYER                              │
│  ┌────────────────────────────────────────────────────┐  │
│  │        Open Khipu Repository Database              │  │
│  │          (khipu.db - SQLite)                       │  │
│  │  - 612 khipus, 54K cords, 110K knots               │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │        Processed Data (data/processed/)            │  │
│  │  - Phase outputs (CSV/JSON)                        │  │
│  │  - Graph structures (NetworkX)                     │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## Module Architecture

### Core Modules

#### 1. Extraction (`src/extraction/`)

**Purpose:** Extract and normalize data from OKR database

**Components:**
- `KhipuLoader` - Main database interface
- `CordExtractor` - Hierarchical cord structure
- `KnotExtractor` - Numeric value decoding
- `ColorExtractor` - Color data normalization

**Design:** Single Responsibility Principle - each extractor handles one data type

#### 2. Graph (`src/graph/`)

**Purpose:** Convert khipus to graph representations

**Components:**
- `KhipuGraphBuilder` - Constructs NetworkX DiGraph
- Graph utilities for traversal and analysis

**Design:** Builder pattern for flexible graph construction

#### 3. Analysis (`src/analysis/`)

**Purpose:** Statistical analysis and hypothesis testing

**Components:**
- `SummationTester` - Arithmetic validation
- `AdministrativeFunctionClassifier` - ML classification
- `phase9/` - 10 meta-analysis modules

**Design:** Strategy pattern for different analysis types

#### 4. Patterns (`src/patterns/`)

**Purpose:** Discover recurring structures

**Components:**
- `MotifMiner` - Subgraph pattern discovery
- `TemplateExtractor` - Structural template identification

**Design:** Template Method pattern

#### 5. Hypothesis (`src/hypothesis/`)

**Purpose:** Test falsifiable hypotheses

**Components:**
- `ColorHypothesisTester` - Color semantics tests
- Statistical testing framework

**Design:** Test suite pattern with p-value reporting

#### 6. Visualization (`src/visualization/`)

**Purpose:** Generate publication-quality plots

**Components:**
- `ClusterVisualizer` - Cluster analysis plots
- `GeographicVisualizer` - Map-based visualizations
- `InteractiveDashboard` - Streamlit components

**Design:** Facade pattern for complex plotting

#### 7. Utils (`src/utils/`)

**Purpose:** Shared utilities

**Components:**
- `ArithmeticValidator` - Summation validation
- `DatabaseUtils` - Connection management
- Logging, configuration, helpers

**Design:** Utility/helper functions

---

## Data Flow

### Phase 1-2: Extraction Pipeline

```
OKR Database (khipu.db)
    ↓
KhipuLoader.get_all_khipus()
    ↓
CordExtractor.extract_hierarchy()
    ↓
KnotExtractor.extract_numeric_values()
    ↓
ColorExtractor.extract_colors()
    ↓
data/processed/phase1/, phase2/
```

### Phase 3: Hypothesis Testing

```
data/processed/phase2/cord_hierarchy.csv
data/processed/phase2/knot_data.csv
    ↓
KhipuGraphBuilder.build_graph()
    ↓
SummationTester.test_summation()
    ↓
data/processed/phase3/summation_test_results.csv
```

### Phase 4: Pattern Discovery

```
data/graphs/khipu_graphs.pkl
    ↓
sklearn.cluster.KMeans.fit()
    ↓
MotifMiner.mine_motifs()
    ↓
data/processed/phase4/
    - cluster_assignments_kmeans.csv
    - motif_mining_results.json
```

### Phase 5-7: ML Pipeline

```
data/processed/phase1-4/
    ↓
FeatureEngineer.extract_features()
    ↓
RandomForestClassifier.fit()
    ↓
data/processed/phase5/, phase7/
```

### Phase 8-9: Meta-Analysis

```
data/processed/phase1-7/
    ↓
Phase8Classifier / Phase9Analyzers
    ↓
data/processed/phase8/, phase9/
```

---

## Phase Pipeline

### Sequential Processing

Phases build on each other:

```
Phase 0: Reconnaissance
    ↓ (database schema)
Phase 1: Baseline Validation
    ↓ (numeric values)
Phase 2: Extraction Infrastructure
    ↓ (cords, knots, colors, graphs)
Phase 3: Summation Testing
    ↓ (arithmetic patterns)
Phase 4: Pattern Discovery
    ↓ (clusters, motifs, templates)
Phase 5: Multi-Model Framework
    ↓ (hypotheses, classification)
Phase 6: Advanced Visualizations
    ↓ (interactive tools)
Phase 7: ML Extensions
    ↓ (anomalies, predictions)
Phase 8: Administrative Function
    ↓ (typology, affordances)
Phase 9: Meta-Analysis
    ↓ (robustness, stability, negative knowledge)
```

### Phase Dependencies

```
Phase 1 ────┐
            ├──→ Phase 3 ──→ Phase 4 ──→ Phase 5 ──→ Phase 7 ──→ Phase 8 ──→ Phase 9
Phase 2 ────┘                    │
                                 ↓
                            Phase 6 (Visualizations)
```

### Idempotency

All phases are **idempotent** - can be re-run safely:
- Check for existing outputs
- Overwrite with confirmation
- Maintain data versioning

---

## Design Patterns

### 1. Builder Pattern (Graph Construction)

```python
class KhipuGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def add_cords(self, cords: pd.DataFrame):
        """Add cord nodes."""
        return self
        
    def add_hierarchy(self, hierarchy: pd.DataFrame):
        """Add parent-child edges."""
        return self
        
    def add_attributes(self, **kwargs):
        """Add node attributes."""
        return self
        
    def build(self) -> nx.DiGraph:
        """Construct final graph."""
        return self.graph
```

### 2. Strategy Pattern (Analysis Methods)

```python
class AnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass

class SummationStrategy(AnalysisStrategy):
    def analyze(self, data: pd.DataFrame) -> dict:
        # Summation-specific analysis
        pass

class ClusteringStrategy(AnalysisStrategy):
    def analyze(self, data: pd.DataFrame) -> dict:
        # Clustering-specific analysis
        pass
```

### 3. Template Method (Phase Execution)

```python
class PhaseAnalyzer(ABC):
    def run_analysis(self):
        """Template method."""
        self.load_data()
        results = self.execute_analysis()
        self.validate_results(results)
        self.save_results(results)
        
    @abstractmethod
    def execute_analysis(self):
        """Subclass implements specific analysis."""
        pass
```

### 4. Facade Pattern (Visualization)

```python
class VisualizationFacade:
    """Simplifies complex visualization pipeline."""
    
    def create_cluster_plot(self, data, labels, output):
        # Orchestrates multiple plotting steps
        self._prepare_data(data)
        self._compute_pca()
        self._create_figure()
        self._add_clusters(labels)
        self._save(output)
```

---

## Technology Stack

### Core Dependencies

| Technology | Purpose | Version |
|------------|---------|---------|
| Python | Core language | 3.11+ |
| pandas | Data manipulation | 2.0+ |
| NumPy | Numerical computing | 1.24+ |
| NetworkX | Graph analysis | 3.0+ |
| scikit-learn | Machine learning | 1.3+ |
| matplotlib | Visualization | 3.7+ |
| seaborn | Statistical plots | 0.12+ |
| Streamlit | Interactive dashboards | 1.25+ |
| SQLite | Database | 3.0+ |

### Development Tools

- **pytest** - Testing framework
- **black** - Code formatting
- **flake8** - Linting
- **mypy** - Type checking (optional)

### Optional Dependencies

- **jupyter** - Interactive notebooks
- **plotly** - Interactive plots (Phase 6)
- **scipy** - Statistical tests

---

## Extension Points

### Adding New Analysis Modules

1. **Create module:** `src/analysis/new_module.py`
2. **Implement interface:**
   ```python
   class NewAnalyzer:
       def run_analysis(self) -> dict:
           pass
   ```
3. **Add script:** `scripts/run_new_analysis.py`
4. **Update docs:** API_REFERENCE.md

### Adding New Extractors

1. **Extend base extractor:**
   ```python
   class NewExtractor(BaseExtractor):
       def extract(self) -> pd.DataFrame:
           pass
   ```
2. **Implement validation:**
   ```python
       def validate(self, data: pd.DataFrame) -> dict:
           pass
   ```

### Adding New Visualizations

1. **Create visualizer:**
   ```python
   class NewVisualizer:
       def plot(self, data, output_path):
           pass
   ```
2. **Add to dashboard:** `scripts/dashboard_app.py`

---

## Performance Considerations

### Scalability

- **Database:** SQLite efficient for 600 khipus, consider PostgreSQL for 10K+
- **Graphs:** NetworkX memory-intensive, consider graph-tool for large graphs
- **Clustering:** K-means scales linearly, hierarchical is O(n²)

### Optimization Strategies

1. **Caching:** Cache extracted data to avoid re-extraction
2. **Parallelization:** Use multiprocessing for independent khipu analysis
3. **Incremental:** Process khipus incrementally, save checkpoints
4. **Lazy loading:** Load data on-demand rather than all at once

### Memory Management

```python
# Good: Process in batches
for khipu_batch in chunked(khipu_ids, batch_size=100):
    process_batch(khipu_batch)
    
# Bad: Load all at once
all_data = load_all_khipus()  # Memory intensive
```

---

## Security Considerations

### Database Access

- Use parameterized queries to prevent SQL injection
- Never execute raw SQL from user input
- Limit database permissions to read-only

### Data Privacy

- No personal data collected
- Archaeological data is public domain
- Respect museum data usage policies

---

## Future Architecture Evolution

### Potential Improvements

1. **Microservices:** Split into extraction, analysis, visualization services
2. **API Layer:** REST API for programmatic access
3. **Database Migration:** Move to PostgreSQL for better concurrency
4. **Distributed Computing:** Use Dask or Spark for large-scale analysis
5. **Web Interface:** Full web application (currently Streamlit)

### Plugin System

Consider plugin architecture for community contributions:

```python
class AnalysisPlugin(ABC):
    @property
    def name(self) -> str:
        pass
        
    @abstractmethod
    def analyze(self, khipu_data) -> dict:
        pass
        
# Load plugins dynamically
plugin_manager.register_plugin(CustomAnalyzer())
```

---

## Contact

For architecture questions or proposals:
- **GitHub Issues:** Technical questions
- **Email:** adafieno@hotmail.com for architectural discussions

---

**Last Updated:** January 2026
