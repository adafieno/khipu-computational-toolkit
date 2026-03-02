# Contributing to Khipu Computational Toolkit

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

---

## Code of Conduct

### Core Principles

This project adheres to strict research ethics:

1. **No semantic decoding claims** - This is a tools project, not a decipherment effort
2. **Pre-interpretive analysis** - Surface patterns, don't assign meaning
3. **Transparency** - All methods must be reproducible and documented
4. **Uncertainty quantification** - Always report confidence levels
5. **Expert validation** - Computational findings require domain expert review

### Research Integrity

- Never claim to have "decoded" khipus
- Avoid causal or historical claims beyond data
- Acknowledge limitations explicitly
- Respect cultural heritage and contemporary Andean communities
- Credit prior work appropriately (OKR, Medrano & Khosla, Clindaniel, Aschers)

---

## Getting Started

### Prerequisites

1. **Python 3.11+** required
2. **Open Khipu Repository** - Clone adjacent to this repo
3. **Git** for version control
4. **Virtual environment** strongly recommended

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/adafieno/khipu-computational-toolkit.git
cd khipu-computational-toolkit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### Verify Setup

```bash
# Run existing tests
pytest tests/

# Check code style
flake8 src/ scripts/

# Verify database connection
python -c "from src.extraction.khipu_loader import KhipuLoader; KhipuLoader()"
```

---

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

**Branch naming conventions:**
- `feature/` - New features or enhancements
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or fixes

### 2. Make Changes

- Work in small, focused commits
- Test frequently during development
- Document as you go

### 3. Test Your Changes

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_extraction.py

# Run with coverage
pytest --cov=src tests/
```

### 4. Format and Lint

```bash
# Format code
black src/ scripts/ tests/

# Check linting
flake8 src/ scripts/ --count --select=E9,F63,F7,F82,E712,F401,F541

# Type checking (optional but recommended)
mypy src/
```

### 5. Commit Changes

```bash
git add .
git commit -m "Brief description of changes"
```

**Commit message guidelines:**
- Use present tense ("Add feature" not "Added feature")
- Be concise but descriptive
- Reference issue numbers when applicable (#123)

Example:
```
Add Phase 9.11 complexity metrics analysis

- Implement ComplexityAnalyzer class
- Add metrics for structural complexity
- Include unit tests and documentation
- Fixes #456
```

---

## Code Standards

### Python Style

Follow **PEP 8** with these specifics:

- **Line length:** 88 characters (Black default)
- **Indentation:** 4 spaces
- **Quotes:** Double quotes for strings
- **Imports:** Grouped (standard library, third-party, local)

### Type Hints

Use type hints for all function signatures:

```python
from typing import Dict, List, Optional
import pandas as pd
import networkx as nx

def extract_hierarchy(
    khipu_id: int,
    validate: bool = True
) -> pd.DataFrame:
    """Extract hierarchical cord structure.
    
    Args:
        khipu_id: Unique khipu identifier
        validate: Whether to validate hierarchy integrity
        
    Returns:
        DataFrame with cord hierarchy
        
    Raises:
        DataValidationError: If hierarchy is invalid
    """
    pass
```

### Docstrings

Use **Google-style** docstrings:

```python
def compute_summation_match(
    parent_value: float,
    children_values: List[float],
    tolerance: float = 0.01
) -> Dict[str, any]:
    """Compute summation match between parent and children.
    
    Tests whether children values sum to parent value within
    specified tolerance. Returns match status and confidence.
    
    Args:
        parent_value: Numeric value of parent cord
        children_values: List of children cord values
        tolerance: Acceptable deviation (default: 0.01)
        
    Returns:
        Dictionary with keys:
            - is_match: bool
            - confidence: float (0-1)
            - deviation: float
            
    Example:
        >>> compute_summation_match(100.0, [60.0, 40.0])
        {'is_match': True, 'confidence': 1.0, 'deviation': 0.0}
    """
    pass
```

### Error Handling

- Use specific exception types
- Provide helpful error messages
- Clean up resources in finally blocks

```python
try:
    conn = connect_db(db_path)
    results = execute_query(conn, query)
except DatabaseConnectionError as e:
    logger.error(f"Database connection failed: {e}")
    raise
finally:
    if conn:
        conn.close()
```

### Logging

Use Python's `logging` module:

```python
import logging

logger = logging.getLogger(__name__)

def analyze_khipu(khipu_id: int) -> dict:
    logger.info(f"Analyzing khipu {khipu_id}")
    try:
        results = perform_analysis(khipu_id)
        logger.debug(f"Analysis complete: {len(results)} findings")
        return results
    except Exception as e:
        logger.error(f"Analysis failed for khipu {khipu_id}: {e}")
        raise
```

---

## Testing Guidelines

### Test Structure

```
tests/
├── test_extraction.py      # Extraction module tests
├── test_graph.py            # Graph construction tests
├── test_analysis.py         # Analysis module tests
├── test_phase9.py           # Phase 9 modules tests
└── fixtures/                # Test data fixtures
    ├── sample_khipu.json
    └── test_database.db
```

### Writing Tests

Use **pytest** with fixtures:

```python
import pytest
from src.extraction.cord_extractor import CordExtractor

@pytest.fixture
def sample_cords():
    """Fixture providing sample cord data."""
    return pd.DataFrame({
        'cord_id': [1, 2, 3],
        'khipu_id': [1000, 1000, 1000],
        'pendant_from': [None, 1, 1],
        'level': [0, 1, 1]
    })

def test_extract_hierarchy(sample_cords):
    """Test hierarchy extraction."""
    extractor = CordExtractor(db_path="tests/fixtures/test_database.db")
    hierarchy = extractor.extract_hierarchy()
    
    assert len(hierarchy) == 3
    assert hierarchy.iloc[0]['level'] == 0
    assert hierarchy['parent_id'].notna().sum() == 2

def test_validate_hierarchy_no_cycles(sample_cords):
    """Test that hierarchy validation detects cycles."""
    # Create cyclic data
    cyclic_cords = sample_cords.copy()
    cyclic_cords.loc[0, 'pendant_from'] = 3  # Creates cycle
    
    extractor = CordExtractor()
    validation = extractor.validate_hierarchy(cyclic_cords)
    
    assert not validation['is_valid']
    assert 'cycle' in validation['errors'][0].lower()
```

### Test Coverage

Aim for **80%+ coverage** on new code:

```bash
pytest --cov=src --cov-report=html tests/
# View coverage report at htmlcov/index.html
```

### Integration Tests

Test end-to-end workflows:

```python
def test_complete_analysis_pipeline():
    """Test full analysis from extraction to visualization."""
    # Load data
    loader = KhipuLoader(db_path=TEST_DB)
    khipus = loader.get_all_khipus()
    
    # Extract
    extractor = CordExtractor(db_path=TEST_DB)
    cords = extractor.extract_hierarchy()
    
    # Build graph
    builder = KhipuGraphBuilder()
    graph = builder.build_graph(cords)
    
    # Analyze
    tester = SummationTester()
    results = tester.test_summation(graph)
    
    assert results['match_rate'] >= 0.0
    assert results['total_tests'] > 0
```

---

## Documentation

### Code Documentation

- All modules must have module-level docstrings
- All classes must have class docstrings
- All public functions must have docstrings
- Complex algorithms should have inline comments

### README Updates

When adding features, update:
- Main [README.md](../README.md) if user-facing
- [API_REFERENCE.md](API_REFERENCE.md) for new modules
- [data/README.md](../data/README.md) for new data outputs

### Report Generation

Phase reports go in `reports/`:
- Use markdown format
- Include methodology, findings, visualizations
- Document limitations and confidence levels
- Reference specific data files produced

Example structure:
```markdown
# Phase N: Title

## Overview
Brief description

## Methodology
How analysis was performed

## Results
Key findings with statistics

## Limitations
What to be cautious about

## Outputs
- data/processed/phaseN/file1.csv
- visualizations/phaseN/plot1.png
```

---

## Submitting Changes

### Pull Request Process

1. **Update Documentation**
   - Update relevant README files
   - Add docstrings to new functions
   - Update API reference if needed

2. **Run Full Test Suite**
   ```bash
   pytest tests/
   flake8 src/ scripts/
   black src/ scripts/ tests/ --check
   ```

3. **Create Pull Request**
   - Use descriptive title
   - Reference related issues
   - Describe changes made
   - List any breaking changes
   - Include screenshots for UI changes

4. **PR Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Motivation
   Why is this change needed?
   
   ## Changes Made
   - Added X feature
   - Fixed Y bug
   - Updated Z documentation
   
   ## Testing
   - [ ] All tests pass
   - [ ] Added new tests for changes
   - [ ] Manually tested feature
   
   ## Related Issues
   Closes #123
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] Tests added/updated
   - [ ] No linting errors
   ```

### Review Process

- Maintainers will review within 5 business days
- Address review comments
- Once approved, changes will be merged

---

## New Feature Guidelines

### Adding a New Analysis Module

1. **Create module in appropriate directory**
   ```
   src/analysis/new_module.py
   ```

2. **Follow class structure pattern**
   ```python
   class NewAnalyzer:
       def __init__(self, config: dict = None):
           self.config = config or {}
           
       def run_analysis(self) -> dict:
           """Main entry point."""
           pass
           
       def save_results(self, results: dict, output_dir: Path):
           """Save analysis outputs."""
           pass
   ```

3. **Create script wrapper**
   ```
   scripts/run_new_analysis.py
   ```

4. **Add tests**
   ```
   tests/test_new_module.py
   ```

5. **Document**
   - Add to API_REFERENCE.md
   - Create report in reports/
   - Update main README.md

### Adding a Phase 9 Module

Phase 9 modules follow strict conventions:

1. **Location:** `src/analysis/phase9/module_name.py`
2. **Output:** `data/processed/phase9/9.X_module_name/`
3. **Naming:** Use descriptive names (e.g., `9.11_complexity_metrics`)
4. **Class name:** `ModuleNameAnalyzer`
5. **Method:** `run_analysis() -> None` (saves to disk)

---

## Questions or Issues?

- **General questions:** Open a [Discussion](https://github.com/adafieno/khipu-computational-toolkit/discussions)
- **Bug reports:** Open an [Issue](https://github.com/adafieno/khipu-computational-toolkit/issues)
- **Feature requests:** Open an [Issue](https://github.com/adafieno/khipu-computational-toolkit/issues) with `enhancement` label
- **Security issues:** Email adafieno@hotmail.com privately

---

## Attribution

Contributors will be acknowledged in:
- `CONTRIBUTORS.md` file
- Release notes
- Academic publications (if applicable)

By contributing, you agree that your contributions will be licensed under the project's MIT License.

---

**Thank you for contributing to khipu research!** 🎉
