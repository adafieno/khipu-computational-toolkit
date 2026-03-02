# Frequently Asked Questions (FAQ)

Common questions about the Khipu Computational Toolkit.

## Table of Contents

- [General Questions](#general-questions)
- [Installation & Setup](#installation--setup)
- [Data & Database](#data--database)
- [Analysis & Methods](#analysis--methods)
- [Interpretation & Claims](#interpretation--claims)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## General Questions

### What is this project?

A computational research toolkit for analyzing Inka khipus from the Open Khipu Repository. It provides tools for hypothesis testing, pattern discovery, and statistical analysis—NOT semantic decoding.

### Is this a khipu decipherment project?

**No.** This is a **tools project** that:
- ✅ Surfaces structural patterns
- ✅ Tests falsifiable hypotheses
- ✅ Quantifies uncertainty
- ❌ Does NOT claim to decode meaning
- ❌ Does NOT make historical claims

### Who is this for?

- Researchers studying khipu structure and patterns
- Archaeologists testing hypotheses about khipu use
- Students learning about computational archaeology
- Developers building analysis tools

### What's been completed?

**9 phases complete (Phases 0-9):**
- ✅ Phase 0: Database reconnaissance (619 khipus analyzed)
- ✅ Phase 1: Numeric validation (95.8% success rate)
- ✅ Phase 2: Data extraction (54K cords, 110K knots)
- ✅ Phase 3: Summation testing (69.5% exhibit summation, corrected Feb 2026)
- ✅ Phase 4: Pattern discovery (7 archetypes found)
- ✅ Phase 5: Hypothesis testing (4 color hypotheses tested)
- ✅ Phase 6: Interactive visualizations (dashboards & 3D viewer)
- ✅ Phase 7: ML extensions (anomaly detection, predictions)
- ✅ Phase 8: Administrative function (8 typologies, re-run Feb 2026)
- ✅ Phase 9: Meta-analysis (10 modules: robustness, stability, etc.)

### How is this different from the Open Khipu Repository?

**OKR** (Open Khipu Repository):
- Curated database of khipu records
- Raw data: knots, cords, colors, provenance
- Foundation for all khipu research

**This Toolkit:**
- Analysis tools built ON TOP of OKR data
- Computational methods and pipelines
- Research findings and visualizations
- Does NOT modify OKR data

Think of it as: **OKR = Data | This Toolkit = Analysis Tools**

---

## Installation & Setup

### What are the prerequisites?

1. **Python 3.11+** (required)
2. **Open Khipu Repository** cloned adjacent to this repo
3. **Git** for version control
4. **Virtual environment** (recommended)

### How do I install the toolkit?

```bash
# Clone both repositories (in same parent directory)
git clone https://github.com/khipulab/open-khipu-repository.git
git clone https://github.com/adafieno/khipu-computational-toolkit.git

cd khipu-computational-toolkit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Why do I need the Open Khipu Repository?

The toolkit analyzes data from the OKR database (`khipu.db`). You need to clone OKR to access this database.

**Expected structure:**
```
your-projects-folder/
├── open-khipu-repository/
│   └── data/
│       └── khipu.db          ← Database
└── khipu-computational-toolkit/
    └── scripts/               ← Analysis scripts
```

### Can I use my own database path?

Yes! Set the `KHIPU_DB_PATH` environment variable:

```bash
# Windows PowerShell
$env:KHIPU_DB_PATH = "C:\path\to\khipu.db"

# Linux/Mac
export KHIPU_DB_PATH="/path/to/khipu.db"
```

Or pass `--db` parameter to scripts:
```bash
# browse.py reads KHIPU_DB_PATH automatically; no --db flag needed
streamlit run scripts/browse.py
```

---

## Data & Database

### How many khipus are analyzed?

**612 khipus** (out of 619 in OKR)
- 7 excluded due to missing cord data
- 54,403 cords decoded (68.2% coverage)
- 110,151 knots processed (95.2% success)

### Where is the processed data?

All outputs are in `data/processed/`, organized by phase:

```
data/processed/
├── phase1/    # Numeric validation
├── phase2/    # Extraction outputs
├── phase3/    # Summation tests
├── phase4/    # Clustering, motifs
├── phase5/    # Hypotheses, classification
├── phase7/    # Anomaly detection
├── phase8/    # Administrative typology
└── phase9/    # Meta-analysis (10 modules)
```

See [data/README.md](../data/README.md) for file descriptions.

### What format are the outputs?

- **CSV** - Tabular data (pandas compatible)
- **JSON** - Structured data (summaries, metadata)
- **PKL** - NetworkX graphs (Python pickle)
- **PNG** - Visualizations (300 DPI)

### Can I use the data in my research?

**Yes!** All processed data is available. Please cite:

1. **Open Khipu Repository** (original data)
2. **This toolkit** (computational methods)

See [README.md](../README.md#citation) for citation details.

### How do I load processed data?

```python
import pandas as pd

# Load any CSV
summation = pd.read_csv('data/processed/phase3/summation_test_results.csv')

# Load graphs
import pickle
with open('data/graphs/khipu_graphs.pkl', 'rb') as f:
    graphs = pickle.load(f)
```

---

## Analysis & Methods

### What is "summation testing"?

Testing whether pendant cord values sum to their primary cord value. Example:

```
Primary cord: 100
Pendant cords: [60, 40]
Result: MATCH (60 + 40 = 100)
```

**Finding:** 69.5% of khipus exhibit summation (430/619), corrected Feb 2026

### What are the "8 administrative types"?

Phase 8 clustering (re-run Feb 2026) identified 8 administrative types:

1. **Aggregated Summary** (45.6%) - Most common, standard administrative records
2. **Standard Administrative Record** (18.6%) - Well-structured routine accounting
3. **Compact Operational Record** (11.3%) - Small, focused local record-keeping
4. **Local Operational Record** (9.5%) - Context-specific operational tracking
5. **Exceptional/Anomalous** (8.4%) - Outliers with atypical characteristics
6. **Lateral Category Tracking** (5.3%) - Specialized category-based organization
7. **Multi-Level Aggregation** (1.1%) - Deep hierarchies, complex aggregation (rare)
8. **Multi-Category Record** (0.2%) - Unique structural combination (1 khipu)

### What is "Phase 9 meta-analysis"?

Phase 9 assesses the **robustness and validity** of Phases 1-8 findings:

- **9.1 Information Capacity** - Entropy, compression
- **9.2 Robustness** - Consistency under perturbation
- **9.3 Cognitive Load** - Visual complexity
- **9.4 Minimalism** - Efficiency vs expressiveness
- **9.5 Variance Mapping** - Feature variability
- **9.6 Boundary Phenomena** - Edge cases
- **9.7 Anomaly Taxonomy** - Categorized outliers
- **9.8 Randomness Testing** - vs random null models
- **9.9 Stability Testing** - Feature ablation, cross-validation
- **9.10 Negative Knowledge** - What khipus are NOT

**Purpose:** Validate that patterns are real, not artifacts.

### How accurate are the classifications?

**Function Classification (Phase 5):**
- 98% accuracy with cross-validation
- 98% administrative vs 2% narrative

**Administrative Typology (Phase 8):**
- 6 distinct types identified
- Random Forest with feature importance

**Confidence levels documented** for all findings.

### Can I add my own analysis?

**Yes!** See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Create new module in `src/analysis/` and add script in `scripts/`.

---

## Interpretation & Claims

### Does this project decode what khipus "say"?

**No.** This project:
- ✅ Identifies structural patterns
- ✅ Tests arithmetic consistency
- ✅ Classifies administrative function
- ❌ Does NOT decode semantic content
- ❌ Does NOT claim to know what khipus "mean"

### What CAN you claim about khipus?

**Empirical findings:**
1. 69.5% exhibit summation patterns (corrected Feb 2026)
2. 8 distinct administrative types exist (re-run Feb 2026)
3. 98% appear administrative (vs narrative)
4. First-position white cords correlate with summation marking (+11.3%)
5. Depth is most stable structural feature
6. NOT random (p<0.0001 vs random models)

**Confidence levels provided** for all claims.

### What CAN'T you claim?

**Avoid:**
- "Khipus encode language X"
- "This color means Y"
- "Khipus record Z historical event"
- Any causal or semantic interpretations

**Why?** Insufficient evidence. Computational analysis surfaces patterns but cannot determine *meaning* without additional context.

### What about color semantics?

**Tested 4 hypotheses:**
1. White boundaries: MIXED (some support)
2. Color-value correlation: NOT SUPPORTED (p=0.92)
3. Color-function patterns: SUPPORTED (accounting uses +57% more colors)
4. Provenance semantics: NOT SUPPORTED (p=1.00)

**Conclusion:** Colors correlate with function, not fixed meanings.

### Is this culturally sensitive?

**Yes.** This project:
- Acknowledges khipus as Andean cultural heritage
- Avoids appropriative or colonialist claims
- Focuses on operational/structural analysis
- Defers interpretation to domain experts
- Respects contemporary Andean communities

---

## Troubleshooting

### "Database not found" error

**Problem:** Cannot find `khipu.db`

**Solution:**
1. Ensure OKR is cloned adjacent to toolkit
2. Set environment variable:
   ```bash
   $env:KHIPU_DB_PATH = "..\open-khipu-repository\data\khipu.db"
   ```
3. Or pass `--db` parameter to scripts

### "Module not found" error

**Problem:** Import error when running scripts

**Solution:**
1. Activate virtual environment:
   ```bash
   .venv\Scripts\Activate.ps1
   ```
2. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Verify PYTHONPATH includes `src/`:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
   ```

### "No processed data found"

**Problem:** Scripts can't find phase outputs

**Solution:**
Run data generation first:
```bash
python scripts/generate_processed_data.py
```

This creates all required processed data files.

### Dashboard won't start

**Problem:** Streamlit error

**Solution:**
1. Install Streamlit:
   ```bash
   pip install streamlit
   ```
2. Check port availability (default: 8501)
3. Try different port:
   ```bash
   streamlit run scripts/browse.py --server.port 8502
   ```

### Linting errors

**Problem:** Code style issues

**Solution:**
```bash
# Auto-fix with black
black src/ scripts/

# Check remaining issues
flake8 src/ scripts/ --select=E9,F63,F7,F82
```

### Tests failing

**Problem:** pytest errors

**Solution:**
1. Ensure test database exists:
   ```bash
   cp ../open-khipu-repository/data/khipu.db tests/fixtures/
   ```
2. Run specific test:
   ```bash
   pytest tests/test_extraction.py -v
   ```
3. Check dependencies are installed

---

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

**Quick start:**
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

### What contributions are welcome?

- New analysis modules
- Bug fixes
- Documentation improvements
- Visualization enhancements
- Performance optimizations
- Test coverage

### Do I need to be a Python expert?

No! Contributions welcome at all skill levels:
- **Beginner:** Documentation, bug reports
- **Intermediate:** Bug fixes, tests
- **Advanced:** New features, architecture

### How long does review take?

- Initial response: **1-2 business days**
- Full review: **3-5 business days**
- Merge: After approval and CI passes

### Can I request features?

**Yes!** Open an [Issue](https://github.com/adafieno/khipu-computational-toolkit/issues) with:
- Clear description
- Use case
- Expected behavior
- Example data (if applicable)

---

## More Questions?

### Documentation

- [README.md](../README.md) - Project overview
- [API_REFERENCE.md](API_REFERENCE.md) - Module documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [data/README.md](../data/README.md) - Data structure
- [reports/](../reports/) - Phase reports

### Support Channels

- **GitHub Issues:** Bug reports, feature requests
- **GitHub Discussions:** General questions, ideas
- **Email:** adafieno@hotmail.com (for private inquiries)

### Academic Contact

For research collaboration or citation questions:
- **Author:** Agustín Da Fieno Delucchi
- **Email:** adafieno@hotmail.com
- **GitHub:** [@adafieno](https://github.com/adafieno)

---

**Can't find your answer?** [Open an issue](https://github.com/adafieno/khipu-computational-toolkit/issues) or [start a discussion](https://github.com/adafieno/khipu-computational-toolkit/discussions)!
