# Week 6 Correction and Limitations Log

Version 2, 14 November 2026 (final technical correction). Version 1 (13 November) content is kept in Part 2.

# PART 1 - Final technical correction (14 Nov 2026)

## 1.1 The problem
In the Week 5 template (and the Week 6 template built on it) hyper-parameters were chosen with GridSearchCV on the training folds, and then the threshold was chosen from `cross_val_predict(best_model, ..., cv=same folds)`. Those folds had already selected the best settings, so the "out-of-fold" probabilities came from a model picked because it did well on them. They are slightly optimistic, and so is a threshold chosen from them. The same issue applied to the label-shuffle check (real score from the tuned model on the selecting folds, compared with shuffled scores with fixed settings) and to the training-side CV scores in the Week 6 template (`repeated_cv_summary`).

## 1.2 What was changed (exactly)
| # | File | Change |
|---|---|---|
| 1 | `week5_modeling_template.py` (now `week5-v3`) | New function `nested_oof()`: for each outer fold, tune with GridSearchCV on the outer-training rows only (inner folds built inside those rows), then predict the untouched outer fold. Stratified and group-aware for both loops; every row is predicted exactly once. |
| 2 | same | Threshold now comes from `nested_oof()` predictions (`threshold_from_nested_oof`). The final model is still tuned once on all training rows and scored once on the locked test set. |
| 3 | same | `label_shuffle_check()` rewritten: runs the same nested procedure on shuffled labels (whole participants if grouped) and compares pooled nested PR-AUC with the real one. Signature changed (needs pipeline and grid). |
| 4 | same | Manifest: `nested_cv_pr_auc`, `nested_cv_roc_auc`, `tuning_cv_pr_auc_optimistic` (the old GridSearch score, renamed so it is not mistaken for an honest estimate), `outer_fold_best_params`, `share_outer_folds_choosing_final_params`, `threshold_method`, `outer_folds`, `inner_folds`, `script_version`. |
| 5 | same | Old `--nested` flag (ungrouped only, side estimate) removed because nesting is now the default and works with groups. New flags `--outer-folds` (5), `--inner-folds` (3), `--shuffle-repeats` (3, was 10 non-nested shuffles). Docstring updated. |
| 6 | `week6_evaluation_template.py` (now `week6-v2`) | Uses `w5.nested_oof` for the threshold; `repeated_cv_summary` replaced by `nested_fold_summary` (outer-fold mean and SD; each outer fold uses a threshold chosen from the OTHER folds); optional `--shuffle-repeats` calls the Week 5 function; threshold table now written twice: `..._threshold_table_train_nested_oof.csv` (for choosing) and `..._test_DESCRIPTIVE_ONLY.csv` (never for choosing). Metric CSV columns renamed `nested_cv_mean_*`, `nested_cv_sd_*`. Manifest records the method. |
| 7 | `test_week5_6_nested.py`, `pytest.ini` | New: 9 automated tests (see 1.3). |
| 8 | Report (`Week_6_Model_Evaluation_Report.docx`) | Edited only the passages describing the old method: Section 1 none; 3.1 (new row "5/6" in Table 3); 5.3 (two bullets); 10.3 (rewritten); 10.4; 11 (Table 22 rows 2-3, new Section 11.1 "Why the threshold uses nested predictions"); 12.1 (shuffle paragraph); 12.2 table; 13.2 and 13.4; Table 20 header; Table 26 threshold row; 9.2 rules 4 and the pre-set rule; Section 15.3-15.4; Table 31/32 wording; 20.2; Table 35; testing-status paragraph. Figures 3 and 4 redrawn to show the nested procedure. All other text, tables, figures and the structure are unchanged. |
| 9 | TOC | **Defect found and fixed:** the previous version restarted page numbering after the cover, but the PDF rendering numbers pages continuously, so the TOC numbers were one lower than the printed page footers. The restart was removed (cover = page 1, hidden footer; Table of Contents = page 2) so Word and LibreOffice agree, and the cached TOC numbers were recomputed from the PDF. The TOC is still a real Word field that refreshes on open. |

Not changed: the Week 5 report DOCX. It still describes and quotes the old v2 code (threshold from out-of-fold predictions of the tuned model; `--nested` flag; old `label_shuffle_check` snippet). Passages affected: the evaluation-approach paragraph near the start, the leakage table rows on groups/test set, the code listings for `make_splits`, `choose_threshold` and `label_shuffle_check`, the testing table (nested flag, `--nested with groups` error) and the numbered threshold step near the end. Replace or annotate them before resubmitting Week 5; see `Week_5_Methodology_Addendum_v3.md`.

## 1.3 Tests executed (all on simulated data only)
Run in this workspace on 14 Nov 2026 (scikit-learn 1.9.1, pandas 3.0.5, Python 3, pytest 9.1.1):
| Test | Result |
|---|---|
| `test_tuning_never_sees_the_rows_it_predicts` (ungrouped and grouped): a spy on `tune` confirms the tuning data equal the outer-training rows, never include the rows then predicted, inner folds lie inside them, every row is predicted exactly once, no participant crosses an outer train/validation split | passed (2 cases) |
| `test_old_pattern_is_not_used_for_threshold`: no `cross_val_predict(` call outside docstrings in either template; both call `nested_oof(` | passed |
| `test_threshold_is_in_range_and_deterministic` | passed |
| `test_fold_summary_uses_leave_fold_out_thresholds`: outer-fold threshold equals the threshold chosen without that fold | passed |
| `test_locked_test_set_second_use_raises` | passed |
| `test_real_run_refused_without_verified_label`: exit code 2, with and without `--label-source` | passed |
| `test_blank_templates_hold_no_results`: blank CSVs contain no decimal numbers | passed |
| `test_end_to_end_demo_labels_everything_simulated` (Week 6 `--demo`, 3 outer / 2 inner folds): exit 0, every output prefixed `SIMULATED_`, `data_source` = SIMULATED, test set used once per model, descriptive test threshold table written | passed (about 2 min) |
Additional runs: Week 5 `--demo` (default 5/3 folds, 5 shuffles) exit 0; Week 5 `--demo --demo-grouped --shuffle-repeats 2` exit 0; Week 6 `--demo-grouped --outer-folds 3 --inner-folds 2 --shuffle-repeats 1 --cost-fn 3 --cost-fp 1` exit 0 with all outputs SIMULATED_. Report checks: DOCX passes the schema validator; TOC (23 entries) compared with the page footer of each heading in the PDF: 23 of 23 match.

Observation (simulated data, code sanity check only, not a project result): in the Week 5 demo the nested PR-AUC was lower than the old GridSearch tuning score for all three models, which is the direction the correction predicts. No number from any simulated run is quoted in the report.

## 1.4 What the correction does not fix
- Nested CV costs more time (Random Forest with the full grid takes minutes on 600 rows).
- With a small real sample the nested folds may hold very few dropouts; the code does not rescue that. Use fewer folds and report the spread.
- The final model is tuned on 100% of the training rows while each nested prediction comes from a model tuned on about 80%; hyper-parameters can differ between outer folds (the manifest reports how many folds agree with the final choice). This is standard but not exact.
- Permutation importance still fits with the final hyper-parameters on the training folds (a ranking aid, not a performance estimate).
- Everything remains unvalidated on real data: there are no verified real labels.

## 1.5 Honesty checks (unchanged)
No accuracy, precision, recall, F1, AUC, confusion counts, rankings, customer findings or financial figures from the project appear in the report. The only invented numbers are the three boxes labelled "ILLUSTRATIVE SIMULATED EXAMPLE - NOT ACTUAL PROJECT RESULTS". Simulated demo output files are prefixed `SIMULATED_`.

## 1.6 Manual actions for you
1. Replace `src/week5_modeling_template.py` in the repo with the new file (same name, new version), add `week6_evaluation_template.py` and the test file; commit yourself (nothing was uploaded).
2. Decide how to handle the Week 5 report (see 1.2, "Not changed").
3. Open the DOCX in Word and accept the field-update prompt; the TOC should already show correct numbers.
4. Still stale on GitHub: `docs/feature_engineering_table_week4.csv`, `README.md`.

# PART 2 - Version 1 log (13 Nov 2026)

## A. Audit of existing files
Inspected: repository (public, readable), all delivered Week 1-5 files, Week 5 report text/tables, Week 3 and Week 5 scripts. Not available: the original build sources for the Week 4/5 reports (helper and figure scripts). I did not push or change anything on GitHub.

| Finding | Action for you |
|---|---|
| Repo `docs/feature_engineering_table_week4.csv` is the OLD version (F01 `price_usd`); corrected file uses `price` | Replace on GitHub |
| Repo `README.md` progress list stale | Update |
| Week 4 report stored as `...Report-3.docx` in repo | Rename (optional) |
| Repo Week 1 Final docx is 87,137 bytes vs 87,197 delivered | Decide which is intended |
| Week 5 report, script (v2), readiness table, Week 3 script: identical to repo | None |

The readiness table shows 11 (public data exist, not computed), 9 (need extra info) and 10 (planned) features; the report uses these counts.

## B. Assumptions
Outcome window and definition of "dropped" not fixed; cost ratio unknown (arbitrary units only); Table 27 criteria are proposed rules; the 10-20 events per predictor rule is a general guideline; Table 18 reflects general algorithm properties.

## C. Limitations
No verified labels; proxy public data; pipeline and templates tested only on simulated data; no feature computed on real data; BRFS weights unvalidated; H1-H5 untested; expected small sample; association not causation; Kaggle licence not verified.

## D. Not done
No performance numbers, rankings, confusion counts, savings or revenue figures. No GitHub upload.

## E. Verification limits
Not every reference link was opened from my workspace. LibreOffice pagination differs from Word.
