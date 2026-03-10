# Phase 9: Graph Topology Analysis

**Generated:** 2026-03-08  
**Database:** K-CAT SQLite database (built from KFG source data)  
**Script:** `scripts/run_phase9_graph.py`  
**Inputs:** `data/kfg/khipu_database.db` · Phase 7/8 cluster assignments  
**Status:** ✅ Complete

---

## Research Question

How do khipus vary in their hierarchical tree shape — depth, branching, and balance — and do these topology metrics align with or cut across the structural (T1/T2) and behavioral (B1–B6) partitions?

**Data scope:** 709 khipus, 62,746 cords, 45,119 pendant nodes.

---

## Methods

### Tree construction

Each khipu is modeled as a `networkx.DiGraph`:

```
virtual_root
  ├── pendant_p1        (hierarchy_level = 0)
  │     ├── p1s1        (level 1)
  │     └── p1s2
  ├── pendant_p2
  └── …
```

Level-0 cords with NULL `parent_cord` attach to the virtual root. All other cords attach to their named parent within the same `kfg_id`.

### Topology metrics (per khipu)

| Metric | Description |
|---|---|
| `depth` | Maximum hierarchy level |
| `leaf_ratio` | Fraction of cords with no children |
| `mean_branching` | Mean out-degree of level-0 cords |
| `branching_cv` | Coefficient of variation of pendant out-degrees |
| `branching_entropy` | Shannon entropy (bits) of pendant out-degree distribution |
| `balance_score` | 1 − min(subtree-size CV, 1); 1 = uniform, 0 = lopsided |
| `subtree_size_cv` | CV of pendant subtree sizes |

### Pendant motif

A pendant's motif is its out-degree: the number of direct subsidiary cords. Motif 0 = pure pendant (no subsidiaries); motif k = k subsidiaries.

---

## Results

### 1. Pendant Motif Distribution

| Motif | Count | Share |
|---|---|---|
| 0 (pure pendant) | 36,243 | 80.3% |
| 1 subsidiary | 6,090 | 13.5% |
| 2 subsidiaries | 1,437 | 3.2% |
| 3 subsidiaries | 543 | 1.2% |
| 4 subsidiaries | 273 | 0.6% |
| 5 subsidiaries | 166 | 0.4% |
| 8 subsidiaries | 145 | 0.3% |
| ≥ 9 subsidiaries | ~322 | ~0.7% |

80.3% of pendants carry no subsidiaries. There is a notable frequency at motif-8 (145 occurrences, vs. 42 at motif-7 and 60 at motif-9).

### 2. Topology by Behavioral Cluster

Median values:

| Cluster | depth | leaf_ratio | mean_branch | branch_entropy | balance |
|---|---|---|---|---|---|
| B1 | 0 | 1.000 | 0.000 | 0.000 | 1.000 |
| B2 | 1 | 0.893 | 0.195 | 0.508 | 0.715 |
| B3 | 1 | 0.938 | 0.076 | 0.314 | 0.778 |
| B4 | 2 | 0.741 | 0.617 | 1.280 | 0.409 |
| B5 | 0 | 1.000 | 0.000 | 0.000 | 1.000 |
| B6 | 1 | 0.927 | 0.118 | 0.455 | 0.717 |

B1 and B5 are topologically identical — both perfectly flat with no subsidiary structure. Their separation in Phase 8 was driven entirely by value content (B1 = no numeric values; B5 = high-entropy values). B4 is the clear outlier: deeper trees (median depth = 2), more branching, and lower balance scores.

### 3. B4 vs B5: Direct Topology Comparison

| Metric | B4 median | B5 median | p-value | Sig |
|---|---|---|---|---|
| depth | 2.000 | 0.000 | < 0.0001 | *** |
| leaf_ratio | 0.741 | 1.000 | < 0.0001 | *** |
| mean_branching | 0.617 | 0.000 | < 0.0001 | *** |
| branching_cv | 1.557 | 0.000 | < 0.0001 | *** |
| branching_entropy | 1.280 | 0.000 | < 0.0001 | *** |
| balance_score | 0.409 | 1.000 | < 0.0001 | *** |

All Mann-Whitney U tests, two-sided. B4 and B5 are topological opposites.

### 4. Geographic Zone Effect on Branching Entropy

**Kruskal-Wallis: H = 41.80, p < 0.0001**

Branching entropy differs significantly across geographic zones, consistent with Phase 8 H2's finding that multi-tier hierarchy concentrates in coastal zones.

### 5. T1 vs T2 Topology

| Metric | T1 median | T2 median | p-value | Sig |
|---|---|---|---|---|
| depth | 1.000 | 2.000 | < 0.0001 | *** |
| mean_branching | 0.118 | 0.847 | < 0.0001 | *** |
| branching_entropy | 0.424 | 1.447 | < 0.0001 | *** |
| balance_score | 0.730 | 0.354 | < 0.0001 | *** |

T2 khipus have deeper trees, more branching per pendant, higher entropy, and lower balance than T1.

---

## Limitations

- **Parent-matching fidelity.** Some pendant attachments may reflect data entry inconsistencies in the KFG schema; orphaned subsidiaries are treated as pendants.
- **Small behavioral clusters.** B1 (n = 15) is too small for robust topology comparison.
- **Cord color not included.** Cord attachment hierarchy may co-vary with color coding; that interaction is not modeled here.

---

## Outputs

| File | Description |
|---|---|
| `data/processed/phase9_graph_metrics.csv` | Per-khipu topology metrics merged with B-label and T-label |
| `data/processed/phase9_motif_catalog.csv` | Corpus-wide pendant motif frequency table |
| `visualizations/phase9/topology_heatmap.png` | Behavioral cluster × topology metric heatmap |
| `visualizations/phase9/branching_distribution.png` | Pendant motif histogram + top-12 bar chart |
| `visualizations/phase9/b4_vs_b5_topology.png` | Direct B4 vs B5 boxplot comparison |
| `visualizations/phase9/zone_topology.png` | Topology metrics by geographic zone |

---

*Corpus sweep run against K-CAT SQLite database. Re-run with `scripts/run_phase9_graph.py` to refresh.*
