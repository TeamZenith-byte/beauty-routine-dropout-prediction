# Week 5 technical correction log (final pre-submission pass)

| # | Issue found | What I changed | Where |
|---|-------------|----------------|-------|
| 1 | Threshold selection used `cross_val_predict` with ordinary `StratifiedKFold`, so rows of one participant could sit on both sides of a fold. | One function, `make_splits`, builds group-aware folds (`StratifiedGroupKFold`) and is reused for tuning, out-of-fold threshold, label-shuffle check and permutation importance. A runtime assertion stops the run if a participant is in both train and validation. The label-shuffle check now shuffles whole participants (the old `permutation_test_score` shuffled within groups, which does nothing when a participant's label is constant). Checked on a simulated grouped table: the old folds shared 80 to 86 participants in every fold, the new folds share 0. | `src/week5_modeling_template.py`; report Sections 11.5, 11.7, 13.5, 13.6, 13.7, 15.1, Table 25 |
| 2 | Permutation importance used the final test set. | Importance is now computed on the validation folds of the training data. The test rows are held in `LockedTestSet`: each model is scored once, after hyper-parameters and threshold are frozen; a second use raises an error. The manifest records `used_for_selection_or_tuning: false`. The report states that the test set is spent after one use. | script; Sections 9.4, 13.6, 13.7, 15, 15.3 |
| 3 | F20 had two names (`cost_per_use` in Week 4 and the report, `routine_cost_month` in the template). | Canonical name is `cost_per_use` (the Week 4 name) everywhere. Monthly cost is not a separate feature; derive it as cost_per_use x uses per month if needed. Report text and examples updated; template and demo data use `cost_per_use`. | script, Tables 4, 9, 10, 11, Section 13.1, feature table |
| 4 | Planned and available features were not clearly separated, and column names differed (`skin_concern` vs F26 `skin_concern_category`; an unlisted `skin_type`). | Feature table gains `data_status`, `in_week5_template`, `template_column`, `canonical_column_name`, `week5_note`. Template uses `skin_concern_category`; unlisted `skin_type` removed. New `docs/data_dictionary_TEMPLATE.csv` marks every template column PLANNED. Report states which features are in the template (F18, F19, F20, F24, F26, F27, F29, F30) and which are not (F21, F25, F28, public-data features). | feature table, data dictionary, Sections 5.2, 13.1 |
| 5 | Cover said "Submission date: October 30, 2026". | Removed, because the real submission date is not known. Add it when you submit. | cover page |
| 6 | TOC page numbers after edits. | The TOC is still a real Word field. The cached numbers were recomputed from the final PDF (34 pages) and checked against every heading: 0 mismatches. | report |
| 7 | Demo outputs must be clearly simulated. | All demo output files are prefixed `SIMULATED_`, CSVs carry `data_source = SIMULATED`, the figure title and the manifest carry the warning. Demo re-run after the fixes (plain, grouped, nested, dictionary+group path, error paths). No outputs or scores are shipped or quoted. | script |

## What still needs real labelled data or your action
- No model can be trained or scored until a real `routine_dropout` label exists (future Layer 2 survey/diary). Every F18 to F30 column is planned; none exists in any dataset.
- The test of the group-aware code on real data is still to be done; I only tested on simulated tables.
- `--resample smote` (imbalanced-learn) is still not executed: the library could not be installed here.
- Dropout rule, window lengths and the cost ratio (missed dropout vs wasted nudge) still need your mentor's decision.
- Word pagination: I cannot run Word. The TOC field refreshes when Word asks to update fields (or right-click, Update Field).
- Add the submission date to the cover yourself.
- Reference web addresses were not opened in this environment; click-check them.
- Nothing was uploaded or committed to GitHub.
