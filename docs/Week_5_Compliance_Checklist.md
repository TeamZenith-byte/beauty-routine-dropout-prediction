# Week 5 compliance checklist (self-audit)

Status: **Fully** = covered in the report; **Partial** = covered but limited by missing real data or unchecked items; **Not** = not covered.

| Requirement | Status | Where / note |
|---|---|---|
| Business problem, target, binary classification vs regression/clustering, business value | Fully | Sections 3, 4 |
| Feature readiness in 3 groups; label construction | Fully | Section 5 (Tables 7 to 12) |
| Logistic Regression: sigmoid, coefficients, regularisation, imbalance, baseline role | Fully | Section 7 |
| Decision Tree: Gini/entropy, gain, depth, pruning, hypothetical path | Fully | Section 8 |
| Random Forest: bootstrap, random features, voting, variance, importance, hyper-parameters | Fully | Section 9 |
| Detailed comparison table and selection rationale (no performance claims) | Fully | Section 10, CSV |
| Preprocessing: missing, encoding, scaling, selection, split, imbalance, leakage, pipelines | Fully | Section 11 |
| Libraries pandas, NumPy, Matplotlib, Seaborn, scikit-learn, imbalanced-learn | Fully | Section 12 |
| Python implementation, all listed steps | Partial | Section 13; executed on simulated rows only (plain, grouped, nested); group-aware folds and one-time test scoring verified; SMOTE path not executed |
| Metrics incl. TP/TN/FP/FN, ROC-AUC, PR-AUC, RMSE/MAE, thresholds | Fully | Section 14 |
| Validation and robustness; procedure for real data | Fully (plan) | Section 15 |
| Overfitting/underfitting, bias-variance, learning curves | Fully | Section 16 |
| Business interpretation, potential vs proven | Fully | Section 17 |
| Limitations: the 8 mandatory facts, consequences, future work | Fully | Sections 18, 19 |
| Workflow diagram, equations, captions, automatic TOC, page numbers | Partial | 34 pages; TOC is a real Word field; cached numbers match the final PDF (0 mismatches); Word may re-paginate (see below) |
| References: credible, none invented | Partial | All from memory of well-known sources; web addresses and page numbers not opened in the build session |
| Submission description 200+ words | Fully | 256 words, Section 22 and `Week_5_Submission_Description.txt` |
| Consistency of F18 to F30 names across report, table and template | Fully | F20 = `cost_per_use`; see feature table columns `template_column` / `data_status` |
| Real model results / best model | Not covered (by design) | No real label exists |

## Needs real data or your manual action

1. Real routine-dropout labels (Layer 2 survey/diary). Nothing can be trained or scored before this.
2. Run the Week 3 pipeline on the real Sephora / Open Beauty Facts files, then Week 4 EDA.
3. Fix the dropout rule, windows and cost ratio (missed dropout vs wasted nudge) with your mentor.
4. Verify dataset licences (Kaggle licence is unverified) and the price currency.
5. Add the submission date to the cover page (removed because it was uncertain).
5b. Open the DOCX in Word: accept the "update fields" prompt (or right-click the TOC, Update Field) so page numbers match Word's pagination.
6. Click-check the reference web addresses and citation details before submitting.
7. Check data-protection obligations (for example India's DPDP Act, 2023) before collecting survey data.
8. Optional: `pip install imbalanced-learn` and test `--resample smote`.
9. GitHub upload: nothing was uploaded or committed. Add the files yourself (or give write access). The repo currently holds the Week 4 report as `reports/Week_4_EDA_Feature_Engineering_Report-3.docx`, and the main README lists Week 3 oddly; consider tidying.
