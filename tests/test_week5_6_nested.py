"""Tests for the 14 Nov 2026 correction (nested threshold selection). Simulated data only.
Run:  python -m pytest -q test_week5_6_nested.py        (the slow end-to-end test is marked and takes a few minutes)"""
import json, re, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest
import week5_modeling_template as w5
import week6_evaluation_template as w6

HERE = Path(__file__).parent


def small(grouped=False):
    df = w5.make_simulated_data(n=300, seed=7, grouped=grouped)
    feats = w5.DEFAULT_NUMERIC + w5.DEFAULT_CATEGORICAL
    return df[feats], df[w5.TARGET].astype(int), (df["participant_id"] if grouped else None)


def spy_tune(monkeypatch):
    calls = []
    real = w5.tune
    def spy(name, pipe, grid, X, y, splits):
        calls.append({"idx": set(X.index), "inner_val": [set(X.index[va]) for _, va in splits]})
        return real(name, pipe, {k: v[:2] for k, v in grid.items()}, X, y, splits)   # tiny grid keeps the test fast
    monkeypatch.setattr(w5, "tune", spy)
    return calls


@pytest.mark.parametrize("grouped", [False, True])
def test_tuning_never_sees_the_rows_it_predicts(monkeypatch, grouped):
    X, y, g = small(grouped)
    calls = spy_tune(monkeypatch)
    pipe, grid = w5.make_models(w5.DEFAULT_NUMERIC, w5.DEFAULT_CATEGORICAL)["logistic_regression"]
    oof, outer, _ = w5.nested_oof(pipe, grid, X, y, g, outer=4, inner=3, seed=1)
    assert len(calls) == 4 and not np.isnan(oof).any()
    seen_as_val = np.zeros(len(y), int)
    for (tr, va), c in zip(outer, calls):
        assert c["idx"] == set(X.index[tr])                       # tuning data = outer-training rows only
        assert not c["idx"] & set(X.index[va])                    # ...and none of the rows it then predicts
        assert all(v <= c["idx"] for v in c["inner_val"])         # inner folds live inside outer-training rows
        seen_as_val[va] += 1
        if g is not None:                                         # participants never straddle outer train/val
            assert not set(g.iloc[tr]) & set(g.iloc[va])
    assert (seen_as_val == 1).all()                               # every row predicted exactly once


def test_old_pattern_is_not_used_for_threshold():
    for f in ("week5_modeling_template.py", "week6_evaluation_template.py"):
        code = (HERE / f).read_text()
        body = re.sub(r'""".*?"""', "", code, flags=re.S)
        assert "cross_val_predict(" not in body, f
    assert "nested_oof(" in (HERE / "week5_modeling_template.py").read_text()
    assert "nested_oof(" in (HERE / "week6_evaluation_template.py").read_text()


def test_threshold_is_in_range_and_deterministic(monkeypatch):
    X, y, g = small()
    spy_tune(monkeypatch)
    pipe, grid = w5.make_models(w5.DEFAULT_NUMERIC, w5.DEFAULT_CATEGORICAL)["decision_tree"]
    o1, _, _ = w5.nested_oof(pipe, grid, X, y, None, 3, 2, 5)
    o2, _, _ = w5.nested_oof(pipe, grid, X, y, None, 3, 2, 5)
    assert np.allclose(o1, o2)
    assert 0 < w5.choose_threshold(y.values, o1) <= 1


def test_fold_summary_uses_leave_fold_out_thresholds(monkeypatch):
    X, y, _ = small()
    spy_tune(monkeypatch)
    pipe, grid = w5.make_models(w5.DEFAULT_NUMERIC, w5.DEFAULT_CATEGORICAL)["logistic_regression"]
    oof, outer, _ = w5.nested_oof(pipe, grid, X, y, None, 4, 2, 3)
    mean, sd, thr = w6.nested_fold_summary(y, oof, outer)
    for (tr, va), t in zip(outer, thr):
        assert t == w5.choose_threshold(y.values[tr], oof[tr])    # chosen without fold k's own rows
    assert set(mean) >= set(w6.METRICS)


def test_locked_test_set_second_use_raises():
    X, y, _ = small()
    pipe, _ = w5.make_models(w5.DEFAULT_NUMERIC, w5.DEFAULT_CATEGORICAL)["logistic_regression"]
    lock = w5.LockedTestSet(X.iloc[:60], y.iloc[:60]); pipe.fit(X.iloc[60:], y.iloc[60:])
    lock.evaluate_once("m", pipe, 0.5)
    with pytest.raises(RuntimeError):
        lock.evaluate_once("m", pipe, 0.4)


def run(*args):
    return subprocess.run([sys.executable, str(HERE / "week6_evaluation_template.py"), *args], capture_output=True, text=True, cwd=HERE)


def test_real_run_refused_without_verified_label():
    r = run("--data", "x.csv"); assert r.returncode == 2 and "refused" in r.stdout
    r = run("--data", "x.csv", "--label-verified"); assert r.returncode == 2          # source text missing


def test_blank_templates_hold_no_results(tmp_path):
    assert run("--blank-templates", "--outdir", str(tmp_path)).returncode == 0
    for f in tmp_path.glob("*.csv"):
        text = f.read_text()
        assert "Not yet evaluated" in text or "to be set" in text
        assert not re.search(r"\b0\.\d+\b", text), f.name


@pytest.mark.slow
def test_end_to_end_demo_labels_everything_simulated(tmp_path):
    r = run("--demo", "--outer-folds", "3", "--inner-folds", "2", "--outdir", str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    files = [p for p in tmp_path.iterdir() if "BLANK" not in p.name]
    assert files and all(p.name.startswith("SIMULATED_") for p in files)
    m = json.loads((tmp_path / "SIMULATED_week6_manifest.json").read_text())
    assert m["data_source"] == "SIMULATED" and "NOT ACTUAL PROJECT RESULTS" in m["warning"]
    assert all(v["test"] for v in m["summary"].values()) and set(m["test_set_use"].values()) == {1}
    assert (pd.read_csv(tmp_path / "SIMULATED_evaluation_metrics.csv").data_source == "SIMULATED").all()
    assert list(tmp_path.glob("SIMULATED_*_threshold_table_test_DESCRIPTIVE_ONLY.csv"))
