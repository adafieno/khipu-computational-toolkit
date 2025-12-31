# Quick Start Guide - Reading the Reports

## Where to Start

If you're new to this project, start here:

### 1. For Project Overview
→ **[PROJECT_PROGRESS_SUMMARY.md](PROJECT_PROGRESS_SUMMARY.md)**

This gives you the complete picture: what's been done, key findings, statistics, and next steps.

### 2. For Understanding the Data
→ **[phase0_reconnaissance_report.md](phase0_reconnaissance_report.md)**

Learn about the database structure, what data is available, and data quality assessment.

### 3. For Numeric Encoding Details
→ **[phase1_baseline_validation_report.md](phase1_baseline_validation_report.md)**

Understand how knots encode numbers and how we validated the numeric system.

### 4. For Extraction Methods
→ **[phase2_extraction_infrastructure_report.md](phase2_extraction_infrastructure_report.md)**

See how we extracted cords, knots, colors, and built graph representations.

### 5. For Hypothesis Testing
→ **[phase3_summation_testing_report.md](phase3_summation_testing_report.md)**

Learn about summation patterns and white cord boundary markers.

---

## Key Questions Answered

### "What are khipus?"
Inka recording devices using knots, cords, and colors. Think of them as pre-Columbian "databases" encoded in fiber.

**Answer:** See [README_FORK.md](../README_FORK.md) - "About Khipus" section

### "How many khipus do you have?"
619 khipus with 54,403 cords and 110,677 knots

**Answer:** See [PROJECT_PROGRESS_SUMMARY.md](PROJECT_PROGRESS_SUMMARY.md) - "Complete Dataset Summary"

### "Can you read khipus?"
Numbers: Yes (95.8% success rate). Semantic meaning: Not yet - that's what we're working toward.

**Answer:** See [phase1_baseline_validation_report.md](phase1_baseline_validation_report.md)

### "What's the white cord hypothesis?"
White cords mark structural boundaries and summation groups. We confirmed this: white cords show +9.1% higher summation match rates.

**Answer:** See [phase3_summation_testing_report.md](phase3_summation_testing_report.md) - "White Cord Analysis"

### "Do khipus encode arithmetic?"
Yes! 74.2% of khipus show pendant-to-parent summation patterns.

**Answer:** See [phase3_summation_testing_report.md](phase3_summation_testing_report.md) - "Results"

### "What's the data quality like?"
Excellent. Average confidence scores of 0.947 for numeric data, 0.949 for hierarchical structure.

**Answer:** See [PROJECT_PROGRESS_SUMMARY.md](PROJECT_PROGRESS_SUMMARY.md) - "Data Quality Metrics"

### "What file formats do you use?"
CSV for tabular data, JSON for metadata, NetworkX pickle for graphs.

**Answer:** See [phase2_extraction_infrastructure_report.md](phase2_extraction_infrastructure_report.md) - "Output Files"

### "What's next?"
Phase 4: Pattern discovery using graph analysis, clustering, and motif mining.

**Answer:** See [PROJECT_PROGRESS_SUMMARY.md](PROJECT_PROGRESS_SUMMARY.md) - "Next Steps: Phase 4 & 5"

---

## Reading by Interest

### For Archaeologists/Anthropologists
1. [PROJECT_PROGRESS_SUMMARY.md](PROJECT_PROGRESS_SUMMARY.md) - Overview
2. [phase3_summation_testing_report.md](phase3_summation_testing_report.md) - Hypothesis validation
3. [phase2_extraction_infrastructure_report.md](phase2_extraction_infrastructure_report.md) - Color analysis

**Focus on:** Key findings, white cord analysis, summation patterns

### For Data Scientists/ML Engineers
1. [phase0_reconnaissance_report.md](phase0_reconnaissance_report.md) - Data model
2. [phase2_extraction_infrastructure_report.md](phase2_extraction_infrastructure_report.md) - Graph construction
3. [PROJECT_PROGRESS_SUMMARY.md](PROJECT_PROGRESS_SUMMARY.md) - Technical infrastructure

**Focus on:** Graph representations, data quality, extraction pipeline

### For Khipu Domain Experts
1. [phase3_summation_testing_report.md](phase3_summation_testing_report.md) - Validation of Medrano hypothesis
2. [phase2_extraction_infrastructure_report.md](phase2_extraction_infrastructure_report.md) - Color findings
3. [phase1_baseline_validation_report.md](phase1_baseline_validation_report.md) - Numeric reliability

**Focus on:** Hypothesis validation, white cord patterns, data coverage

### For Software Engineers
1. [phase0_reconnaissance_report.md](phase0_reconnaissance_report.md) - Database schema
2. [PROJECT_PROGRESS_SUMMARY.md](PROJECT_PROGRESS_SUMMARY.md) - Codebase structure
3. [phase2_extraction_infrastructure_report.md](phase2_extraction_infrastructure_report.md) - API documentation

**Focus on:** Code architecture, performance metrics, dependencies

---

## Report Structure (Consistent Across All Phases)

Each phase report contains:

1. **Executive Summary** - 2-3 paragraph overview
2. **Objectives** - What we aimed to accomplish
3. **Methodology** - How we did it
4. **Results** - What we found (with statistics)
5. **Key Findings** - Most important discoveries
6. **Output Files** - What data files were generated
7. **Validation Checks** - Quality assurance
8. **Limitations & Caveats** - What we can't claim
9. **Next Steps** - How this enables future work

---

## Data File Locations

### Processed Data
`data/processed/`
- `cord_numeric_values.csv` (54,403 cords)
- `cord_hierarchy.csv` (54,403 cords)
- `knot_data.csv` (110,151 knots)
- `color_data.csv` (56,306 records)
- `white_cords.csv` (15,125 segments)
- `summation_test_results.csv` (619 khipus)

### Graph Data
`data/graphs/`
- `khipu_graphs.pkl` (619 NetworkX graphs)
- `khipu_graphs_metadata.json` (statistics)

### Original Database
`khipu.db` (SQLite, 619 khipus)

---

## Key Statistics at a Glance

| Metric | Value |
|--------|-------|
| **Khipus** | 619 |
| **Cords** | 54,403 |
| **Knots** | 110,677 |
| **Colors** | 56,306 records (66 unique codes) |
| **White cords** | 15,125 (26.8%) |
| **Numeric coverage** | 95.8% of khipus |
| **Avg confidence** | 0.947 |
| **Summation khipus** | 459 (74.2%) |
| **High match rate** | 187 (30.2%) |

---

## Contact

**Questions about the reports?**  
Agustín Da Fieno Delucchi - adafieno@hotmail.com

**Questions about the original data?**  
Open Khipu Repository - okr-team@googlegroups.com

---

**Last Updated:** December 31, 2025
