# Week 5 Methodology Addendum (v3, 14 November 2026)

The Week 5 template `week5_modeling_template.py` was corrected from v2 to **v3**.

**Changed:** the threshold is now chosen from NESTED out-of-fold predictions (`nested_oof`): in each outer fold the models are tuned only on the other folds and predict the held-out fold, so no probability comes from a model tuned with that row. The label-shuffle check runs the same nested procedure on shuffled labels. The old `--nested` flag is gone (nesting is default and works with groups). New flags: `--outer-folds`, `--inner-folds`, `--shuffle-repeats`.

**Unchanged:** the three models and grids, the stratified and group-aware hold-out, LockedTestSet (test set scored once per model), PR-AUC tuning, cost-based threshold rule, SIMULATED_ labelling, leakage guard, data dictionary rules.

**Why:** v2 chose the threshold from `cross_val_predict` of the already-tuned model on the folds that had selected its hyper-parameters, which is slightly optimistic.

**Status:** tested on simulated data only (see Week_6_Correction_Log.md, Part 1). No real-data result exists.

**The submitted Week 5 DOCX still describes v2** (evaluation-approach paragraph, leakage table rows, code listings for make_splits / choose_threshold / label_shuffle_check, the testing table, and the final threshold step). It was not regenerated. Suggested one-line annotation if you keep it as is: "Superseded by week5-v3: threshold selection and the label-shuffle check use nested cross-validation (see addendum)."
