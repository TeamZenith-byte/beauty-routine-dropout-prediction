# Week 6 Compliance Checklist

Report: `Week_6_Model_Evaluation_Report.docx`. Status of the whole report: **evaluation framework plus simulated code demonstration. No real model results exist.**

## Official requirements → report sections
| # | Official requirement | Section | Status |
|---|---|---|---|
| 1 | Introduction and project overview | 2 (and 1 Executive Summary) | Done |
| 2 | Objectives of model evaluation | 4 | Done |
| 3 | Overview of Week 5 modelling strategy | 5 | Done |
| 4 | Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC | 6.2, 6.3 | Done (formulas, plain meaning; invented example labelled) |
| 5 | Justification for each metric | 7 | Done |
| 6 | Confusion matrix and interpretation | 8 (Figure 2, Tables 15-17) | Done (blank counts) |
| 7 | Model comparison framework (LR, DT, RF) | 9 (Tables 18-20) | Done (blank results) |
| 8 | Comprehensive validation plan | 10 | Done |
| 9 | Training, validation, CV, test stages | 11 (Figures 3-4, Table 22) | Done |
| 10 | Leakage prevention, over/underfitting | 12 | Done |
| 11 | Threshold selection and business significance | 13 (Figure 5) | Done |
| 12 | Visualization and reporting | 14 (Figure 6, Table 26) | Done |
| 13 | Business impact in beauty/wellness | 16 (Figure 7, Table 29) | Done, as hypotheses |
| 14 | Recommendations (marketing, product, support, decision-makers) | 17 (Table 30, Figure 8) | Done, conditional |
| 15 | Limitations, risks, ethics | 18 | Done |
| 16 | Future improvements, implementation plan | 15 (Part B), 19 | Done |
| 17 | Final conclusion, technical and non-technical | 20 | Done |

## Required explanations
TP/TN/FP/FN: Table 8. Confusion matrix: Section 8. Threshold, cross-validation, stratified sampling, overfitting, underfitting, data leakage: Table 11 (and Sections 12, 13). Metric formulas: Table 9.

## Required visuals
| Visual | Where |
|---|---|
| Model evaluation workflow diagram | Figure 1 |
| Confusion matrix illustration | Figure 2 |
| Model comparison table | Table 18 (properties) and Table 20 (blank ranking) |
| Evaluation metrics comparison table | Table 19 (blank) |
| Validation process flowchart | Figure 4 (splits: Figure 3) |
| ROC curve explanation (labelled illustrative) | Figure 6 left |
| Precision-Recall curve explanation (labelled illustrative) | Figure 6 right |
| Business impact workflow | Figure 7 |
| Final recommendation framework | Figure 8 |
| Captions below every figure | Yes |

## Extra instructions
- Audit of Week 4/5 files: Section 3.3 (Table 5). 
- Part A (Work Completed): Section 3. Part B (Future Evaluation and Deployment): Section 15.
- Formatting: cover, real TOC field, numbered headings, page numbers, references (Section 21), executive summary, conclusion.

## Methodology consistency (14 Nov 2026)
Sections 5.3, 10.3-10.4, 11 (incl. new 11.1), 12.1, 13.2, 15.3-15.4 and Figures 3-4 describe the nested threshold selection used by `week5_modeling_template.py` (v3) and `week6_evaluation_template.py` (v2). TOC: 23/23 entries verified against PDF page footers.

## Honesty checks
- No accuracy, precision, recall, F1, AUC, confusion counts, rankings, customer findings or financial figures from the project: none reported.
- Only invented numbers: Tables 10, 13 and the cost-unit example in 13.2, each labelled "ILLUSTRATIVE SIMULATED EXAMPLE – NOT ACTUAL PROJECT RESULTS".
- ROC/PR and threshold figures are hand-drawn schematics, labelled, not tied to the three models.
- Simulated demo numbers from the Python template are not quoted anywhere.
- BRFS and H1-H5 described as proposed/untested. Kaggle licence not claimed verified.

## Needs you manually
Open the DOCX in Word, accept the "update fields" prompt (refreshes TOC page numbers), check pagination once. Update stale GitHub files (see correction log).
