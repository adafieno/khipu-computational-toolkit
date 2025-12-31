# Quipu Decipherment Project  
## Literature Review, Expert Integration, and Validation Strategy

This document outlines prior computational work on Inka quipus, identifies open debates and constraints in the domain, and proposes a validation-first strategy suitable for a personal exploratory AI project grounded in the Open Khipu Repository (OKR).

---

## 1. Literature / Prior Work Review

### 1.1 Existing Computational Attempts

Computational and quantitative approaches to quipu analysis date back over a century. In 1912, L. Leland Locke demonstrated that quipu knots encode numbers in a decimal positional system, establishing the numeric foundation of quipus. This insight was later expanded by Marcia and Robert Ascher, who compiled large comparative datasets and showed that many quipus exhibit arithmetic structure, including summation relationships across cords.

More recently, computational scale has increased significantly. Medrano and Khosla (2024) analyzed approximately 650 quipus using algorithmic methods and found that around 74% contain internally consistent summation structures. Their work also identified non-obvious conventions, such as white cords marking boundaries between summed groups, and enabled the probabilistic reassembly of fragmented quipus based on arithmetic consistency.

Machine learning approaches have begun to appear, though cautiously. Clindaniel (2024) applied transformer-based models (BERT-style architectures) to structured quipu data and found statistically significant clustering of cord colors and attributes, suggesting latent categorical systems. Other efforts have used clustering, dimensionality reduction, and motif mining to explore structural similarities across quipus.

Crucially, none of these approaches claim a full “decipherment.” Computational methods so far detect structure, regularity, and anomaly — not definitive semantic meaning.

---

### 1.2 Established vs. Debated Domain Constraints

**Well-established constraints:**
- Quipus encode numbers using a decimal positional system.
- Knot type and position correspond to numeric value.
- Quipus have a hierarchical structure (primary cord → pendant cords → subsidiaries).
- Many quipus contain internal arithmetic relations (e.g., subtotals and totals).
- Quipus were used for administrative and accounting purposes (census, tribute, inventory).

**Debated or unresolved constraints:**
- Whether quipus encode narrative or linguistic information.
- The semantic meaning of cord color, fiber type, ply direction, and knot orientation.
- Whether non-numeric cords encode categories, names, or phonetic elements.
- The degree of regional or temporal standardization versus local conventions.

Some scholars (e.g., Gary Urton, Sabine Hyland) argue for limited non-numeric or syllabic encoding in specific contexts, while others maintain that quipus functioned primarily as numeric–mnemonic devices. No consensus exists.

---

### 1.3 Tested Hypotheses and Methods

Several hypotheses have been tested computationally:

- **Numeric encoding hypothesis:** Confirmed via knot-position analysis and arithmetic consistency.
- **Summation structure hypothesis:** Tested by verifying whether certain cords equal the sum of others (confirmed at scale).
- **Categorical encoding hypothesis:** Explored via clustering of cord attributes (color, material, structure).
- **Provenance consistency hypothesis:** Tested by grouping quipus by region and identifying shared structural motifs.
- **Document alignment hypothesis:** Attempted probabilistic matching between quipus and colonial-era administrative records, with partial but inconclusive success.

Overall, hypotheses about structure and arithmetic are testable and frequently validated; hypotheses about semantics remain speculative and probabilistic.

---

## 2. Domain Expert Integration

### 2.1 Validation of Numeric Constraint Rules

Numeric constraint rules (e.g., decimal place inference, knot-type interpretation, summation logic) must be validated by domain experts — archaeologists, anthropologists, and historians specializing in Andean record-keeping.

Experts play a critical role in:
- Confirming that algorithmic rules align with known quipu conventions.
- Identifying edge cases (e.g., unusual knot forms, damaged cords).
- Preventing silent propagation of incorrect assumptions.

In practice, this means experts should review:
- Sample numeric decodings
- Constraint definitions
- Anomaly classifications

before these rules are treated as stable.

---

### 2.2 Handling Disputed Interpretations

Disputed interpretations (e.g., meaning of color or ply direction) should not be hard-coded as facts. Instead:

- Represent multiple interpretations explicitly.
- Annotate features as “disputed” or “hypothetical.”
- Run analyses under multiple assumptions and compare outcomes.
- Track expert disagreement as metadata, not noise.

This allows the system to remain scientifically honest and flexible.

---

### 2.3 Expert-in-the-Loop Validation

Expert-in-the-loop validation is essential at every stage:

1. **Data ingestion:** Verify correct understanding of OKR schema and physical features.
2. **Feature extraction:** Confirm that computational features correspond to meaningful physical properties.
3. **Pattern discovery:** Evaluate whether discovered patterns are archaeologically plausible.
4. **Interpretation:** Ground any semantic claims in cultural and historical context.

Experts are not just validators at the end, but collaborators throughout the pipeline.

---

## 3. Validation Strategy

### 3.1 Measuring Success by Phase

Because full ground truth is unavailable, success must be defined locally at each stage:

- **Data accuracy:** Internal consistency (e.g., numeric sums match).
- **Pattern robustness:** Patterns recur across subsets and provenances.
- **Expert agreement:** Independent experts recognize outputs as plausible.
- **Stability:** Results are insensitive to small perturbations in data or parameters.

Success is incremental, not absolute.

---

### 3.2 Train/Test Strategy with Limited Data

Given the small dataset size:

- Use k-fold cross-validation or leave-one-out validation.
- Avoid splitting quipus from the same provenance across train/test.
- Prefer task-based evaluation (e.g., “does the model rediscover known summation rules?”).
- Consider bootstrapping and resampling for stability testing.

Traditional 80/20 splits are often inappropriate in this domain.

---

### 3.3 Preventing Overfitting

To reduce overfitting:

- Prefer simple, interpretable models.
- Apply strong regularization.
- Limit feature space using domain knowledge.
- Require patterns to appear across multiple quipus.
- Use perturbation tests to assess sensitivity.
- Reject results that hinge on a single artifact.

Expert review serves as an additional safeguard against overinterpretation.

---

## Closing Note

This project should be framed not as “deciphering quipus,” but as:
> **Building computational tools that help scholars test hypotheses rigorously and transparently.**

AI’s role is to surface structure, quantify uncertainty, and support falsifiable claims — not to replace domain expertise or assert definitive meaning.

