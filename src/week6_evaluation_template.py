#!/usr/bin/env python3
"""
Week 6 - Model evaluation & reporting template
Project: Beauty Routine Dropout Prediction (YuvaIntern Data Science with Python)

WHAT THIS FILE IS
  An evaluation and reporting layer on top of the Week 5 modeling template
  (week5_modeling_template.py, same folder). It uses the same three models
  (Logistic Regression, Decision Tree, Random Forest), the same stratified /
  group-aware splits, the same locked test set and the same cost-based threshold rule, and --
  since the 14 Nov 2026 correction -- the same NESTED threshold selection (week5-v3): hyper-parameters
  are tuned only on outer-training rows, and the threshold comes from the predictions that tuned
  model makes on the untouched outer-validation rows. The older code reused the tuned model's own
  out-of-fold predictions, which are slightly optimistic.

WHAT THIS FILE IS NOT
  It has produced NO project results. No verified real routine_dropout label exists yet.
  The only runnable mode today is --demo, which uses SIMULATED data whose target comes from an
  INVENTED rule. Demo numbers test the code only. They say nothing about real customers
  or real model quality, and every demo output file starts with "SIMULATED_".

SAFETY GATE (real data)
  A real run is refused unless you pass BOTH
     --label-verified                 (you confirm the label is a verified, outcome-window label)
     --label-source "<how it was collected>"
  Public Sephora / Open Beauty Facts files contain no routine_dropout label, so they cannot be used.

USAGE
  python week6_evaluation_template.py --demo                  # code test on simulated data
  python week6_evaluation_template.py --blank-templates       # only write empty report tables
  python week6_evaluation_template.py --data survey.csv --dictionary dict.csv \
         --label-verified --label-source "Layer 2 survey + 8-week diary" [--group-col participant_id]
Exit codes: 0 ok, 2 input/config error (same as Week 5).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

import week5_modeling_template as w5          # Week 5 functions are reused, not re-written

SIM_LABEL = "ILLUSTRATIVE SIMULATED EXAMPLE - NOT ACTUAL PROJECT RESULTS"
MODELS = ["logistic_regression", "decision_tree", "random_forest"]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc_average_precision"]
NOT_YET = "Not yet evaluated"


# --8<-- H_blank_templates
def write_blank_templates(outdir, pre=""):
    """Empty tables for the report. They contain no numbers on purpose."""
    cols = ["model", "role"] + METRICS + ["threshold", "status"]
    rows = [[m, r] + [NOT_YET] * len(METRICS) + [NOT_YET, "Waiting for verified real labels"]
            for m, r in [("logistic_regression", "primary baseline"),
                         ("decision_tree", "interpretable comparison"),
                         ("random_forest", "ensemble comparison"),
                         ("dummy_prior_baseline", "reference (must be beaten)")]]
    pd.DataFrame(rows, columns=cols).to_csv(outdir / f"{pre}evaluation_metrics_BLANK.csv", index=False)
    cm = pd.DataFrame({"model": MODELS, "TN": NOT_YET, "FP": NOT_YET, "FN": NOT_YET, "TP": NOT_YET})
    cm.to_csv(outdir / f"{pre}confusion_matrix_BLANK.csv", index=False)
    cost = pd.DataFrame({"item": ["cost_of_missed_dropout (FN)", "cost_of_wasted_outreach (FP)",
                                  "implied_threshold = c_FP / (c_FP + c_FN)"],
                         "value": ["to be set by the brand", "to be set by the brand", NOT_YET]})
    cost.to_csv(outdir / f"{pre}cost_assumptions_BLANK.csv", index=False)
# --8<-- end


# --8<-- I_nested_fold_summary
def nested_fold_summary(y, oof, outer_splits, beta=2.0, cost_fn=None, cost_fp=None):
    """Spread of each metric across the OUTER folds of the nested procedure (training data only).
    For outer fold k the threshold is chosen from the nested predictions of the OTHER folds only, then
    applied to fold k, so no fold judges a threshold that it helped to choose. Returns (mean, sd, thresholds)."""
    y = np.asarray(y); rows, thrs = [], []
    for tr, va in outer_splits:
        thr = w5.choose_threshold(y[tr], oof[tr], beta, cost_fn, cost_fp); thrs.append(float(thr))
        yhat = (oof[va] >= thr).astype(int)
        rows.append({"accuracy": float((yhat == y[va]).mean()),
                     "precision": precision_score(y[va], yhat, zero_division=0),
                     "recall": recall_score(y[va], yhat), "f1": f1_score(y[va], yhat),
                     "roc_auc": roc_auc_score(y[va], oof[va]),
                     "pr_auc_average_precision": average_precision_score(y[va], oof[va])})
    d = pd.DataFrame(rows)
    return d.mean().to_dict(), d.std(ddof=1).to_dict(), thrs
# --8<-- end


# --8<-- J_bootstrap
def bootstrap_ci(y, p, thr, n=500, seed=w5.SEED, groups=None):
    """95% percentile interval for test-set metrics. Resamples whole participants when groups exist."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    idx_by_group = None if groups is None else pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).apply(list)
    out = {k: [] for k in ["precision", "recall", "f1", "pr_auc_average_precision"]}
    for _ in range(n):
        if idx_by_group is None:
            i = rng.integers(0, len(y), len(y))
        else:
            pick = rng.choice(len(idx_by_group), len(idx_by_group))
            i = np.concatenate([idx_by_group.iloc[k] for k in pick])
        if y[i].min() == y[i].max():
            continue
        yh = (p[i] >= thr).astype(int)
        out["precision"].append(precision_score(y[i], yh, zero_division=0))
        out["recall"].append(recall_score(y[i], yh))
        out["f1"].append(f1_score(y[i], yh))
        out["pr_auc_average_precision"].append(average_precision_score(y[i], p[i]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}
# --8<-- end


# --8<-- K_threshold_table
def threshold_table(y, p, c_fn=None, c_fp=None, grid=None):
    """What each possible threshold would mean: flagged share, precision, recall, optional cost.
    Build it from the NESTED training predictions when it will guide a choice. A table built from
    the test set is descriptive only: never pick or change a threshold from it."""
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2) if grid is None else grid
    rows = []
    for t in grid:
        yh = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
        row = {"threshold": t, "flagged_share": float(yh.mean()), "TP": int(tp), "FP": int(fp), "FN": int(fn),
               "TN": int(tn), "precision": precision_score(y, yh, zero_division=0), "recall": recall_score(y, yh)}
        if c_fn is not None and c_fp is not None:
            row["total_cost_units"] = float(c_fn * fn + c_fp * fp)
        rows.append(row)
    return pd.DataFrame(rows)
# --8<-- end


# --8<-- L_plots
def plot_report(results, y_te, outdir, pre, simulated):
    """ROC, PR, calibration, confusion matrices. On simulated data every panel carries a red banner."""
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))
    for name, r in results.items():
        f, t, _ = roc_curve(y_te, r["proba"]); ax[0, 0].plot(f, t, label=name)
        pr, rc, _ = precision_recall_curve(y_te, r["proba"]); ax[0, 1].plot(rc, pr, label=name)
        fr, mp = calibration_curve(y_te, r["proba"], n_bins=5, strategy="quantile"); ax[0, 2].plot(mp, fr, "o-", label=name)
    ax[0, 0].plot([0, 1], [0, 1], "k:"); ax[0, 0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    ax[0, 1].axhline(float(np.mean(y_te)), color="k", ls=":"); ax[0, 1].set(title="PR curve (dotted = prevalence)", xlabel="Recall", ylabel="Precision")
    ax[0, 2].plot([0, 1], [0, 1], "k:"); ax[0, 2].set(title="Calibration", xlabel="Mean predicted risk", ylabel="Observed dropout share")
    ax[0, 0].legend(fontsize=8)
    for k, (name, r) in enumerate(results.items()):
        cm = confusion_matrix(y_te, r["proba"] >= r["metrics"]["threshold"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[1, k],
                    xticklabels=["continue", "dropout"], yticklabels=["continue", "dropout"])
        ax[1, k].set(title=name, xlabel="Predicted", ylabel="Actual")
    if simulated:
        fig.suptitle(SIM_LABEL + "  (code test on invented data)", color="red", fontsize=11)
    fig.tight_layout(); fig.savefig(outdir / f"{pre}evaluation_report_panels.png", dpi=130); plt.close(fig)
# --8<-- end


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="SIMULATED data, code test only")
    ap.add_argument("--demo-grouped", action="store_true", help="SIMULATED data with several rows per participant")
    ap.add_argument("--blank-templates", action="store_true", help="write empty report tables and stop")
    ap.add_argument("--data"); ap.add_argument("--dictionary"); ap.add_argument("--group-col")
    ap.add_argument("--target", default=w5.TARGET)
    ap.add_argument("--label-verified", action="store_true")
    ap.add_argument("--label-source", default="")
    ap.add_argument("--cost-fn", type=float); ap.add_argument("--cost-fp", type=float)
    ap.add_argument("--outer-folds", type=int, default=5); ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--shuffle-repeats", type=int, default=0,
                    help="optional label-shuffle check (Week 5 function; each repeat is a full nested run)")
    ap.add_argument("--outdir"); ap.add_argument("--seed", type=int, default=w5.SEED)
    a = ap.parse_args(argv)

    simulated = a.demo or a.demo_grouped
    pre = "SIMULATED_" if simulated else ""
    outdir = Path(a.outdir or ("week6_outputs_SIMULATED_DEMO" if simulated else "week6_outputs")); outdir.mkdir(exist_ok=True)
    if a.blank_templates:
        write_blank_templates(outdir); print("blank templates written to", outdir); return 0

    # ---- Gate: never evaluate against an unverified target label.
    try:
        if simulated:
            print(w5.SIM_BANNER)
            df = w5.make_simulated_data(seed=a.seed, grouped=a.demo_grouped)
            numeric, categorical, id_col = w5.DEFAULT_NUMERIC, w5.DEFAULT_CATEGORICAL, "person_id"
            group_col = "participant_id" if a.demo_grouped else None
        else:
            if not (a.label_verified and a.label_source.strip()):
                raise ValueError("Real run refused: pass --label-verified and --label-source describing how the "
                                 "routine_dropout label was collected. Public proxy data have no such label.")
            if not a.data:
                raise ValueError("Give --data")
            df = pd.read_csv(a.data)
            if a.dictionary:
                numeric, categorical, id_col, group_col = w5.read_dictionary(a.dictionary, df)
            else:
                numeric = [c for c in w5.DEFAULT_NUMERIC if c in df.columns]
                categorical = [c for c in w5.DEFAULT_CATEGORICAL if c in df.columns]
                id_col, group_col = None, None
            group_col = a.group_col or group_col
        if a.target not in df.columns:
            raise ValueError(f"Target column '{a.target}' not found")
        if not set(df[a.target].dropna().unique()) <= {0, 1}:
            raise ValueError("Target must be 0/1")
        df = df.dropna(subset=[a.target]).copy()
        features = numeric + categorical
        w5.guard_leakage(features, a.target, id_col, group_col)
        if group_col and (df.groupby(group_col)[a.target].nunique() > 1).any():
            raise ValueError("Outcome differs between rows of one participant")
    except Exception as e:
        print("ERROR:", e); return 2

    X, y = df[features], df[a.target].astype(int)
    groups = df[group_col] if group_col else None
    X_tr, X_te, y_tr, y_te = w5.stratified_holdout(X, y, groups, 0.2, a.seed)      # stage 1: locked test set
    g_tr = groups.loc[X_tr.index] if groups is not None else None
    g_te = groups.loc[X_te.index] if groups is not None else None
    locked = w5.LockedTestSet(X_te, y_te)
    summary, results = {}, {}
    for name, (pipe, grid) in w5.make_models(numeric, categorical, a.seed).items():
        splits = w5.make_splits(X_tr, y_tr, g_tr, 5, a.seed)
        gs = w5.tune(name, pipe, grid, X_tr, y_tr, splits)                         # FINAL model: all training rows
        best = gs.best_estimator_
        # stage 2+3: NESTED out-of-fold predictions -> threshold and training-side spread
        oof, outer_splits, fold_params = w5.nested_oof(pipe, grid, X_tr, y_tr, g_tr, a.outer_folds, a.inner_folds, a.seed)
        thr = w5.choose_threshold(y_tr.values, oof, 2.0, a.cost_fn, a.cost_fp)
        mean, sd, fold_thr = nested_fold_summary(y_tr.values, oof, outer_splits, 2.0, a.cost_fn, a.cost_fp)
        entry_extra = {}
        if a.shuffle_repeats > 0:
            entry_extra["label_shuffle_check"] = w5.label_shuffle_check(
                pipe, grid, X_tr, y_tr, g_tr, average_precision_score(y_tr, oof), a.outer_folds, a.inner_folds,
                a.shuffle_repeats, a.seed)
        metrics, proba = locked.evaluate_once(name, best, thr)                      # stage 4: once on test
        ci = bootstrap_ci(y_te.values, proba, thr, 300, a.seed, g_te)
        summary[name] = {"best_params": gs.best_params_, "threshold": thr, "outer_fold_thresholds": fold_thr,
                         "outer_fold_best_params": fold_params, "nested_cv_mean": mean, "nested_cv_sd": sd,
                         "test": metrics, "test_ci95": ci, **entry_extra}
        results[name] = {"proba": proba, "metrics": metrics}
        src_ = "SIMULATED" if simulated else "REAL"
        threshold_table(y_tr.values, oof, a.cost_fn, a.cost_fp).assign(data_source=src_, basis="training nested out-of-fold").to_csv(
            outdir / f"{pre}{name}_threshold_table_train_nested_oof.csv", index=False)
        threshold_table(y_te.values, proba, a.cost_fn, a.cost_fp).assign(data_source=src_, basis="test set, descriptive only").to_csv(
            outdir / f"{pre}{name}_threshold_table_test_DESCRIPTIVE_ONLY.csv", index=False)

    # ---- Reporting tables (nothing below feeds back into any modelling choice)
    src = "SIMULATED" if simulated else "REAL"
    rows = []
    for name, s in summary.items():
        t, m, sd = s["test"], s["nested_cv_mean"], s["nested_cv_sd"]
        rows.append({"data_source": src, "model": name, "threshold": t["threshold"],
                     **{f"test_{k}": round(t[k], 3) for k in METRICS},
                     **{f"nested_cv_mean_{k}": round(m[k], 3) for k in METRICS},
                     **{f"nested_cv_sd_{k}": round(sd[k], 3) for k in METRICS},
                     "tn": t["tn"], "fp": t["fp"], "fn": t["fn"], "tp": t["tp"]})
    pd.DataFrame(rows).to_csv(outdir / f"{pre}evaluation_metrics.csv", index=False)
    write_blank_templates(outdir, pre="")
    plot_report(results, y_te.values, outdir, pre, simulated)
    manifest = {"data_source": src, "label_source": "INVENTED simulation rule" if simulated else a.label_source,
                "script_version": "week6-v2 (uses week5-v3 nested threshold selection)",
                "threshold_method": "nested out-of-fold; outer-fold spread uses leave-fold-out thresholds",
                "test_set_use": locked.uses, "summary": summary,
                "warning": SIM_LABEL if simulated else "Real run: report only after leakage review and label audit."}
    (outdir / f"{pre}week6_manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    for name, s in summary.items():
        print(f"{name:20s} threshold={s['threshold']:.3f}  test recall={s['test']['recall']:.2f}  "
              f"precision={s['test']['precision']:.2f}  PR-AUC={s['test']['pr_auc_average_precision']:.3f}"
              + ("   [SIMULATED]" if simulated else ""))
    print("outputs written to", outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
