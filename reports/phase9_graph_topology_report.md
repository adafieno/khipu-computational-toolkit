# Phase 9: Graph Topology Analysis

*Khipu Computational Toolkit — kfg-integration branch*

---

## Overview

Phase 9 interprets each khipu as a **rooted directed tree** and extracts structural
topology metrics that are independent of the cord values analyzed in Phase 8 and the
size-based structural features of Phase 7.  The goal is to test whether the
hierarchical *shape* of a khipu encodes organizational information beyond what raw
cord counts or accounting signals reveal.

**Data scope:** 709 khipus, 62,746 cords, 45,119 pendant nodes.

---

## Methods

### Tree construction

For each khipu, a `networkx.DiGraph` is built as follows:

```
virtual_root
  ├── pendant_p1        (hierarchy_level = 0, parent_cord = NULL)
  │     ├── p1s1        (level 1)
  │     └── p1s2
  ├── pendant_p2
  └── …
```

- Level-0 cords whose `parent_cord` is NULL or references a cord not present in the
  same khipu are attached directly to the virtual root.
- All other cords are attached to their named parent within the same `kfg_id`.

### Topology metrics (per khipu)

| Metric | Description |
|--------|-------------|
| `depth` | Maximum hierarchy level in the tree |
| `leaf_ratio` | Fraction of cords with no children |
| `mean_branching` | Mean out-degree of pendant (level-0) cords |
| `branching_cv` | Coefficient of variation of pendant out-degrees |
| `branching_entropy` | Shannon entropy (bits) of pendant out-degree distribution |
| `balance_score` | 1 − min(subtree-size CV, 1) — 1 = uniform, 0 = lopsided |
| `subtree_size_cv` | CV of pendant subtree sizes |

### Pendant motif

A "pendant motif" is simply the out-degree of a level-0 cord: the number of
subsidiary cords it directly supports.  Motif 0 = a pure value-recording pendant;
motif *k* = a pendant that groups *k* subsidiaries beneath it.

---

## Results

### 1. Corpus-wide pendant motif distribution

The vast majority of pendant cords carry **no subsidiaries** (pure value recorders):

| Motif | Count | Share |
|-------|------:|------:|
| 0 subsidiaries (pure pendant) | 36,243 | 80.3 % |
| 1 subsidiary | 6,090 | 13.5 % |
| 2 subsidiaries | 1,437 | 3.2 % |
| 3 subsidiaries | 543 | 1.2 % |
| 4 subsidiaries | 273 | 0.6 % |
| 5 subsidiaries | 166 | 0.4 % |
| 8 subsidiaries | 145 | 0.3 % |
| ≥9 subsidiaries | ~322 | ~0.7 % |

The spike at **8 subsidiaries** (145 occurrences) is notable and likely reflects a
standard administrative grouping unit — possibly a decimal sub-unit (groups of 8
reporting positions under a pendant total).

### 2. Topology by behavioral cluster

Median values across the seven topology metrics:

| Cluster | depth | leaf_ratio | mean_branch | branch_entropy | balance | Interpretation |
|---------|------:|----------:|------------:|---------------:|--------:|----------------|
| **B1** | 0 | 1.000 | 0.000 | 0.000 | 1.000 | Entirely flat — no subsidiaries |
| **B2** | 1 | 0.893 | 0.195 | 0.508 | 0.715 | Shallow, moderate branching |
| **B3** | 1 | 0.938 | 0.076 | 0.314 | 0.778 | Mostly flat, sparse branching |
| **B4** | 2 | 0.741 | 0.617 | 1.280 | 0.409 | **Deep, branched, unbalanced** |
| **B5** | 0 | 1.000 | 0.000 | 0.000 | 1.000 | Flat — identical topology to B1 |
| **B6** | 1 | 0.927 | 0.118 | 0.455 | 0.717 | Shallow single-tier grouping |

**Key observation:** B1 and B5 are **topologically indistinguishable** — both are
perfectly flat trees with no subsidiary structure.  Their separation in Phase 8 was
driven entirely by *value content* (B1 = no numeric values; B5 = high-entropy
multi-commodity numerics).  Tree topology alone cannot resolve them, confirming that
recording function is encoded at the value layer, not the structural layer, for flat
khipus.

B4 is the clear outlier: deeper trees (median depth = 2), more branching, and
substantially lower balance scores, consistent with hierarchical aggregation
accounting where subsidiary cords summarise sub-totals into pendant totals.

### 3. B4 vs B5: direct topology comparison

All six topology metrics distinguish B4 from B5 at the highest significance level:

| Metric | B4 median | B5 median | p-value | Sig |
|--------|----------:|----------:|--------:|-----|
| depth | 2.000 | 0.000 | < 0.0001 | *** |
| leaf_ratio | 0.741 | 1.000 | < 0.0001 | *** |
| mean_branching | 0.617 | 0.000 | < 0.0001 | *** |
| branching_cv | 1.557 | 0.000 | < 0.0001 | *** |
| branching_entropy | 1.280 | 0.000 | < 0.0001 | *** |
| balance_score | 0.409 | 1.000 | < 0.0001 | *** |

*All Mann-Whitney U tests, two-sided.*

B4 and B5 represent genuine **topological antipodes** in the corpus: one class
structures information hierarchically; the other collapses everything to a single
flat tier.

### 4. Geographic zone effect on branching entropy

Kruskal-Wallis test: **H = 41.80, p < 0.0001**

Geographic zone has a statistically significant association with cord-branching
structure.  Khipus from different ecozones do not share the same hierarchical
complexity distributions.  This aligns with Phase 8's H2 result (coastal khipus
tend toward deeper hierarchies, p = 0.0005 by chi-squared) and extends it using
a continuous topology metric rather than a discrete depth category.

### 5. Structural type (T1 vs T2) vs topology

| Metric | T1 median | T2 median | p-value | Sig |
|--------|----------:|----------:|--------:|-----|
| depth | 1.000 | 2.000 | < 0.0001 | *** |
| mean_branching | 0.118 | 0.847 | < 0.0001 | *** |
| branching_entropy | 0.424 | 1.447 | < 0.0001 | *** |
| balance_score | 0.730 | 0.354 | < 0.0001 | *** |

T2 khipus (large, multi-group) are not only larger (Phase 7 finding) but are
also structurally more complex: deeper trees, more branching per pendant, higher
entropy, and lower balance.  This suggests T2 represents a qualitatively different
administrative tier — not simply a bigger version of T1 but an organiser of
sub-accounts that T1 khipus may individually record.

---

## Data Signals and Potential Avenues for Exploration

### Hierarchical structure and administrative tiering

The B4 topology profile (depth ≥ 2, branching_entropy > 1.0) is structurally consistent
with what one would expect of an aggregation ledger: pendant totals composed from
subsidiary partial counts beneath them.  The unbalanced balance scores (median 0.41)
suggest sub-groups of unequal size.  Whether this reflects census or tribute accounting
hierarchies is a question for archaeologically-informed interpretation.

### Flat topology does not imply simple content

B5 and B1 share identical topologies but opposite value-layer profiles.  A flat
khipu can be either an empty ceremonial/non-numeric object (B1) or a dense
multi-commodity inventory (B5).  Topology and value complexity are therefore
**orthogonal information dimensions** — the shape of the tree tells you the
accounting *depth* but not its *content diversity*.

### The 8-subsidiary spike and a potential standardized unit

The unexpected peak at motif-8 (145 occurrences, vs. 42 at motif-7 and 60 at
motif-9) is unlikely to be random.  One candidate avenue for investigation is whether
this corresponds to a decimal sub-unit — the Inca decimal system included divisions
such as groups of 40 (5 × 8) households under a pachaka (100-household officer).  If
pendant totals represent sub-group entries, motif-8 may mark a standard reporting
sub-unit.  Cross-referencing motif-8 pendants with geographic zone and khipu provenance
would be a productive step for specialists.

### T2 topology and a potential higher-tier accounting role

The elevated branching entropy and depth of T2 khipus (entropy median 1.45
vs. T1’s 0.42) is consistent with a reading of T2 as higher-tier aggregators —
objects that bring together records from multiple T1-level registers.
The low balance scores for T2 (0.354 median) further suggest that T2 khipus
consolidated sub-accounts of heterogeneous sizes rather than uniform
reporting pools.  Whether this constitutes a distinct administrative tier is a
question for archaeologically-grounded investigation.

---

## Limitations

- **Parent-matching fidelity:** Some pendant attachments may reflect data entry
  inconsistencies in the KFG schema rather than true hierarchy gaps; orphaned
  subsidiaries are treated as pendants.
- **Small behavioral clusters:** B1 (n = 15) is too small for robust topology
  comparison; its results are directional only.
- **Cord color not included:** Cord attachment hierarchy may co-vary with color
  coding practices (Phase 5); that interaction is not modeled here.

---

## Outputs

| File | Description |
|------|-------------|
| `data/processed/phase9_graph_metrics.csv` | Per-khipu topology metrics merged with B-label and T-label |
| `data/processed/phase9_motif_catalog.csv` | Corpus-wide pendant motif frequency table |
| `visualizations/phase9/topology_heatmap.png` | Behavioral cluster × topology metric heatmap |
| `visualizations/phase9/branching_distribution.png` | Pendant motif histogram + top-12 bar chart |
| `visualizations/phase9/b4_vs_b5_topology.png` | Direct B4 vs B5 boxplot comparison |
| `visualizations/phase9/zone_topology.png` | Topology metrics by geographic zone |
