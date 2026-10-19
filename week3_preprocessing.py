#!/usr/bin/env python3
"""
week3_preprocessing.py
======================
Week 3 - Data Cleaning and Preprocessing Pipeline
Project: Beauty Routine Dropout Prediction (YuvaIntern Data Science internship)
Author : Zishan Alam
Repo   : https://github.com/TeamZenith-byte/beauty-routine-dropout-prediction

What this script does
---------------------
Cleans PUBLIC product / review / ingredient tables (Layer 1 of the data strategy)
and writes separate cleaned files. Raw input files are never modified.

What this script does NOT do
----------------------------
* It does NOT create a routine_dropout label. Public review data does not record
  whether someone abandoned a routine. The true target comes from the future
  survey / routine diary (Layer 2), which is not handled here.
* It has NOT been run on the real Sephora or Open Beauty Facts files. Column
  names below are candidates and must be checked with --inspect first.
  The built-in --demo mode uses SIMULATED rows only, to test the code.

Quick start (run from the repository root; script assumed to be in src/)
-----------
    python src/week3_preprocessing.py --demo
    python src/week3_preprocessing.py --inspect --products data/raw/products.csv
    python src/week3_preprocessing.py --products P.csv --reviews R.csv --output-dir out

Exit codes: 0 = finished; 1 = --strict and at least one validation check FAILED;
2 = input or configuration error (missing file, required column not found,
forbidden label column, output folder equal to the raw data folder).
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import html
import json
import logging
import platform
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("week3")

# ---------------------------------------------------------------------------
# 1. CONFIGURATION  (edit here, or override with --config my_config.json)
# ---------------------------------------------------------------------------
CONFIG = {
    "paths": {"products": None, "reviews": None, "obf": None},
    "separators": {"products": ",", "reviews": ",", "obf": "\t"},
    "output_dir": "data/processed",
    # text values that really mean "missing" ("unknown" is NOT in this list)
    "na_tokens": ["", "na", "n/a", "#n/a", "nan", "null", "-", "--", "?",
                  "not available"],
    "rating_range": [1, 5],          # documented validity rule for ratings
    "price_min": 0,                  # negative prices are invalid; 0 is flagged
    "date_min": "2000-01-01",        # earlier or later (future) dates are invalid
    "review_rating_integer_only": True,
    "iqr_k": 1.5,                    # outlier fence multiplier (flag only)
    "outlier_columns": ["price", "review_count", "loves_count",
                        "review_word_count"],
    "outlier_log_columns": ["price", "review_count", "loves_count"],
    "key_duplicate_policy": "flag",  # "flag" or "keep_most_complete"
    "very_short_review_words": 3,
    "max_missing_pct_warn": 50.0,
    # column meaning decides the missing-value action (see report section 9)
    "missing_policy": {
        "products": {"price": "flag", "rating": "flag", "review_count": "flag",
                     "loves_count": "flag", "brand_name": "unknown_category",
                     "category": "unknown_category",
                     "secondary_category": "unknown_category",
                     "ingredients": "leave"},
        "reviews": {"rating": "flag", "review_text": "flag",
                    "review_date": "flag", "is_recommended": "flag",
                    "skin_type": "unknown_category"},
        "obf": {"brand_name": "unknown_category", "category": "unknown_category",
                "countries": "unknown_category", "ingredients": "leave"},
    },
    # optional numeric imputation; empty = no imputation (default)
    # example: {"products": {"price": {"group_by": "category"}}}
    "impute_numeric": {},
    # privacy: profile columns excluded from the cleaned output
    "exclude_sensitive": True,
    "sensitive_columns": ["skin_tone", "eye_color", "hair_color"],
    # the public pipeline must never create or accept these columns
    "forbidden_columns": ["routine_dropout", "dropout_reason", "adherence_days"],
    # identifiers stay as text and are never used as predictors
    "identifier_columns": ["product_id", "author_id", "review_id", "barcode"],
    # category cleanup: EXAMPLE synonyms, extend after reviewing value counts
    "category_synonyms": {"moisturizers": "moisturizer",
                          "moisturiser": "moisturizer",
                          "moisturisers": "moisturizer",
                          "cleansers": "cleanser", "serums": "serum",
                          "face wash": "cleanser", "sunscreens": "sunscreen"},
    # ordered keyword rules; first match wins (draft, check against real data)
    "step_type_rules": [
        ["eye_care", ["eye"]], ["sunscreen", ["sunscreen", "spf"]],
        ["cleanser", ["cleanser", "face wash", "cleansing"]],
        ["exfoliant", ["exfoliat", "peel", "scrub"]],
        ["toner", ["toner", "essence"]], ["serum", ["serum", "ampoule"]],
        ["mask", ["mask"]],
        ["moisturizer", ["moisturi", "cream", "lotion"]],
    ],
    # explicit overrides if auto-detection picks the wrong column:
    # {"products": {"price": "price_inr"}}
    "column_overrides": {},
}

# candidate column names (after snake_case). NOT verified on real files.
COLUMN_CANDIDATES = {
    "products": {
        "product_id": ["product_id", "productid", "id", "sku"],
        "product_name": ["product_name", "name", "title"],
        "brand_name": ["brand_name", "brand"],
        "category": ["primary_category", "category", "product_category"],
        "secondary_category": ["secondary_category", "sub_category"],
        "price": ["price_usd", "price", "list_price"],
        "rating": ["rating", "avg_rating", "average_rating"],
        "review_count": ["reviews", "review_count", "n_reviews"],
        "loves_count": ["loves_count", "likes", "favorites"],
        "ingredients": ["ingredients", "ingredient_list", "ingredients_text"],
    },
    "reviews": {
        "review_id": ["review_id"],
        "author_id": ["author_id", "user_id", "reviewer_id"],
        "product_id": ["product_id", "productid"],
        "rating": ["rating", "stars", "review_rating"],
        "is_recommended": ["is_recommended", "recommended"],
        "review_date": ["submission_time", "review_date", "date", "created_at"],
        "review_text": ["review_text", "text", "review", "content"],
        "review_title": ["review_title", "title", "summary"],
        "skin_type": ["skin_type"],
    },
    "obf": {
        "barcode": ["code", "barcode", "ean"],
        "product_name": ["product_name", "name"],
        "brand_name": ["brands", "brand"],
        "category": ["categories", "categories_en"],
        "ingredients": ["ingredients_text", "ingredients"],
        "countries": ["countries", "countries_en"],
    },
}
REQUIRED = {"products": ["product_id"], "reviews": ["product_id"],
            "obf": ["barcode"]}
KEY_COLUMN = {"products": "product_id", "obf": "barcode"}


# ---------------------------------------------------------------------------
# 2. CLEANING LOG  (every decision is recorded with the number of rows affected)
# ---------------------------------------------------------------------------
class CleaningLog:
    def __init__(self):
        self.records = []

    def add(self, dataset, step, rule, column, n_affected, action, note=""):
        rec = {"timestamp_utc": datetime.now(timezone.utc).isoformat(
                   timespec="seconds"),
               "dataset": dataset, "step": step, "rule": rule,
               "column": column or "", "n_affected": int(n_affected),
               "action": action, "note": note}
        self.records.append(rec)
        LOGGER.info("[%s] %s | %s | col=%s | n=%d | %s", dataset, step, rule,
                    column or "-", n_affected, action)

    def to_frame(self):
        return pd.DataFrame(self.records)


# ---------------------------------------------------------------------------
# 3. LOADING, PROFILING, SCHEMA
# ---------------------------------------------------------------------------
def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv_safely(path, sep=",", nrows=None):
    """Read everything as text so invalid values stay visible, not silently coerced."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    last_error = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False,
                             encoding=enc, nrows=nrows, on_bad_lines="warn")
            meta = {"path": str(path), "encoding": enc, "separator": sep,
                    "sha256": sha256_of_file(path), "n_rows_read": len(df),
                    "n_cols": int(df.shape[1])}
            return df, meta
        except UnicodeDecodeError as err:
            last_error = err
    raise ValueError(f"Could not decode {path}: {last_error}")


def to_snake_case(name):
    s = unicodedata.normalize("NFKC", str(name)).strip()
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s)
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s or "unnamed"


def standardize_column_names(df, log, dataset):
    """snake_case names; duplicates get a suffix; old->new mapping is returned."""
    new_names, seen = [], set()
    for col in df.columns:
        base = to_snake_case(col)
        name, k = base, 1
        while name in seen:
            k += 1
            name = f"{base}_{k}"
        seen.add(name)
        new_names.append(name)
    mapping = pd.DataFrame({"original_name": list(df.columns),
                            "standard_name": new_names})
    n_changed = int((mapping.original_name != mapping.standard_name).sum())
    out = df.copy()
    out.columns = new_names
    log.add(dataset, "schema", "snake_case column names", None, n_changed,
            "renamed (mapping saved)")
    return out, mapping


def drop_export_index_columns(df, log, dataset):
    """Drop 'unnamed_0'-style columns only if they are just a row counter."""
    out = df
    for col in [c for c in df.columns if re.fullmatch(r"unnamed(_\d+)?", c)]:
        num = pd.to_numeric(df[col], errors="coerce")
        if num.notna().all() and num.is_unique:
            out = out.drop(columns=[col])
            log.add(dataset, "schema", "export index column", col, len(df),
                    "dropped (unique row counter, no information)")
    return out


def normalize_na_tokens(df, tokens, log, dataset):
    """Strip whitespace and turn disguised missing values into real NA."""
    token_set = {t.lower() for t in tokens}
    out = df.copy()
    for col in out.columns:
        s = out[col].astype("string").str.strip()
        mask = s.isna() | s.str.lower().isin(token_set)
        n_disguised = int((mask & df[col].astype("string").str.strip().notna()
                           & (df[col].astype("string").str.strip() != "")).sum())
        out[col] = s.mask(mask, pd.NA)
        if n_disguised:
            log.add(dataset, "missing", "disguised missing token -> NA", col,
                    n_disguised, "converted to NA")
    return out


def resolve_columns(df, dataset, overrides=None):
    """Map canonical names to the real columns; first match wins, no reuse."""
    resolved, claimed = {}, set()
    overrides = overrides or {}
    for canon, names in COLUMN_CANDIDATES[dataset].items():
        pick = overrides.get(canon)
        if pick is None:
            pick = next((n for n in names
                         if n in df.columns and n not in claimed), None)
        if pick is not None and pick in df.columns:
            claimed.add(pick)
            resolved[canon] = pick
        else:
            resolved[canon] = None
    return resolved


def apply_canonical_names(df, resolved, log, dataset):
    rename = {actual: canon for canon, actual in resolved.items()
              if actual and actual != canon}
    clash = [c for c in rename.values() if c in df.columns and c not in rename]
    if clash:
        raise ValueError(f"Rename would overwrite existing columns: {clash}")
    out = df.rename(columns=rename)
    for canon, actual in resolved.items():
        if actual is None:
            log.add(dataset, "schema", "candidate column not found", canon, 0,
                    "dependent steps skipped")
    return out, rename


def profile_dataframe(df):
    """Per-column profile. Genuine NA (n_missing) and blank/whitespace-only text
    (n_blank) are counted SEPARATELY. A raw file is read with keep_default_na=False,
    so its empty cells are blank strings, not NA; n_blank makes them visible."""
    text = df.astype("string")
    if len(df) == 0:
        blank = pd.DataFrame(False, index=df.index, columns=df.columns)
    else:
        blank = text.apply(lambda s: s.str.strip().eq("")).fillna(False)
    n = max(len(df), 1)
    prof = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "n_blank": blank.sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "pct_blank": (blank.sum() / n * 100).round(2),
        "pct_missing_or_blank": ((df.isna() | blank).sum() / n * 100).round(2),
        "n_unique": [df[c][~blank[c]].nunique(dropna=True) for c in df.columns],
        "example_value": [(df[c][~blank[c]].dropna().iloc[0]
                           if (~blank[c] & df[c].notna()).any() else "")
                          for c in df.columns],
    })
    prof.index.name = "column"
    summary = {"n_rows": int(len(df)), "n_cols": int(df.shape[1]),
               "n_exact_duplicate_rows": int(df.duplicated().sum())}
    return prof.reset_index(), summary


def check_required(resolved, dataset, found_columns=None):
    missing = [c for c in REQUIRED[dataset] if resolved.get(c) is None]
    if missing:
        raise KeyError(
            f"[{dataset}] required column(s) not found: {missing}. "
            f"Columns found: {[] if found_columns is None else list(found_columns)}. If the whole header "
            "is one column, the separator is probably wrong (set 'separators' in "
            "the config). Otherwise run --inspect and set column_overrides.")


# ---------------------------------------------------------------------------
# 4. DUPLICATES
# ---------------------------------------------------------------------------
def drop_exact_duplicates(df, subset, dataset, log, label):
    """Remove repeated rows. `subset` is the documented definition of duplicate."""
    mask = df.duplicated(subset=subset, keep="first")
    n = int(mask.sum())
    where = "all columns" if subset is None else ", ".join(subset)
    log.add(dataset, "duplicates", f"{label} (same {where})", None, n,
            "removed, first copy kept" if n else "none found")
    return df.loc[~mask].copy(), df.loc[mask].copy()


def handle_key_duplicates(df, key, policy, dataset, log):
    """Same identifier but different content: flag (default) or keep most complete."""
    dup_mask = df[key].notna() & df.duplicated(subset=[key], keep=False)
    n_rows = int(dup_mask.sum())
    out = df.copy()
    out[f"{key}_duplicate_flag"] = dup_mask
    removed = out.iloc[0:0].copy()
    if policy == "keep_most_complete" and n_rows:
        completeness = out.notna().sum(axis=1)
        order = completeness.sort_values(ascending=False, kind="stable").index
        out = out.loc[order]
        drop_mask = out.duplicated(subset=[key], keep="first") & out[key].notna()
        removed = out.loc[drop_mask].copy()
        out = out.loc[~drop_mask].sort_index()
        action = "kept the most complete row per id"
    else:
        action = "flagged for manual review, nothing removed"
    log.add(dataset, "duplicates", f"repeated {key} with different content", key,
            n_rows, action)
    return out, removed


# ---------------------------------------------------------------------------
# 5. NUMERIC / BINARY / DATE VALIDATION
# ---------------------------------------------------------------------------
def parse_numeric(series):
    """Assumes '.' decimal and ',' thousands separator; strips currency symbols."""
    s = series.astype("string")
    s = s.str.replace(r"[\s$€£₹¥]", "", regex=True).str.replace(",", "", regex=True)
    return pd.to_numeric(s, errors="coerce").astype("Float64")


def apply_range_rule(df, col, low, high, integer_only, dataset, log, rule_name):
    """Parse a numeric column, set invalid values to NA, keep the raw text."""
    raw = df[col].astype("string")
    num = parse_numeric(raw)
    unparseable = (raw.notna() & num.isna()).fillna(False).astype(bool)
    bad = pd.Series(False, index=df.index)
    if low is not None:
        bad |= (num < low).fillna(False).astype(bool)
    if high is not None:
        bad |= (num > high).fillna(False).astype(bool)
    if integer_only:
        bad |= (num.notna() & (num != num.round())).fillna(False).astype(bool)
    invalid = unparseable | bad
    out = df.copy()
    out[f"{col}_raw"] = raw
    out[col] = num.mask(invalid, pd.NA)
    out[f"{col}_invalid_flag"] = invalid
    log.add(dataset, "validity", f"{rule_name}: not a number", col,
            unparseable.sum(), "set to NA, raw text kept in *_raw")
    log.add(dataset, "validity", f"{rule_name}: out of range / not whole", col,
            bad.sum(), "set to NA, raw text kept in *_raw")
    return out


def to_nullable_int(df, col):
    df[col] = df[col].round().astype("Int64")
    return df


def parse_binary(df, col, dataset, log):
    mapping = {"1": 1, "1.0": 1, "true": 1, "yes": 1, "y": 1, "t": 1,
               "0": 0, "0.0": 0, "false": 0, "no": 0, "n": 0, "f": 0}
    s = df[col].astype("string").str.lower()
    values = [mapping.get(v, pd.NA) if pd.notna(v) else pd.NA for v in s]
    parsed = pd.array(values, dtype="Int64")
    invalid = (s.notna() & pd.Series(parsed, index=df.index).isna()).to_numpy()
    out = df.copy()
    out[f"{col}_raw"] = df[col].astype("string")
    out[col] = pd.Series(parsed, index=df.index)
    out[f"{col}_invalid_flag"] = invalid
    log.add(dataset, "validity", "binary field must be yes/no/0/1", col,
            invalid.sum(), "unrecognised values set to NA")
    return out


def parse_dates(df, col, cfg, dataset, log):
    """Parse mixed date formats. Unparseable, too old or future dates become NaT."""
    raw = df[col].astype("string")
    try:
        parsed = pd.to_datetime(raw, errors="coerce", format="mixed", utc=True)
    except (TypeError, ValueError):       # older pandas without format="mixed"
        parsed = pd.to_datetime(raw, errors="coerce", utc=True)
    parsed = parsed.dt.tz_localize(None)
    unparseable = (raw.notna() & parsed.isna()).to_numpy()
    lo = pd.Timestamp(cfg["date_min"])
    hi = pd.Timestamp.now(tz="UTC").tz_localize(None)
    window = ((parsed < lo) | (parsed > hi)).fillna(False).to_numpy()
    out = df.copy()
    out[f"{col}_raw"] = raw
    out[col] = parsed.mask(window)
    out[f"{col}_invalid_flag"] = unparseable | window
    log.add(dataset, "validity", "date could not be parsed", col,
            unparseable.sum(), "set to NaT, raw text kept")
    log.add(dataset, "validity", f"date outside {cfg['date_min']}..today", col,
            window.sum(), "set to NaT, raw text kept")
    return out


# ---------------------------------------------------------------------------
# 6. OUTLIERS  (flag and investigate; never delete automatically)
# ---------------------------------------------------------------------------
def flag_outliers_iqr(series, k=1.5, use_log=False):
    x = pd.to_numeric(series, errors="coerce").astype("float64")
    v = np.log1p(x.clip(lower=0)) if use_log else x
    valid = v.dropna()
    summary = {"n_valid": int(valid.size), "method": "IQR on log1p" if use_log
               else "IQR", "fence_low": np.nan, "fence_high": np.nan,
               "n_flagged": 0}
    if valid.size < 8:
        return pd.Series(False, index=series.index), summary
    q1, q3 = valid.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        summary["method"] += " (IQR is 0, not applicable)"
        return pd.Series(False, index=series.index), summary
    lo, hi = q1 - k * iqr, q3 + k * iqr
    flags = ((v < lo) | (v > hi)).fillna(False)
    summary.update(
        fence_low=float(np.expm1(lo) if use_log else lo),
        fence_high=float(np.expm1(hi) if use_log else hi),
        n_flagged=int(flags.sum()), min=float(x.min()),
        median=float(x.median()), p99=float(x.quantile(0.99)),
        max=float(x.max()))
    return flags, summary


def investigate_outliers(df, cfg, dataset, log):
    out, rows = df.copy(), []
    for col in cfg["outlier_columns"]:
        if col not in out.columns:
            continue
        use_log = col in cfg["outlier_log_columns"]
        flags, summ = flag_outliers_iqr(out[col], cfg["iqr_k"], use_log)
        out[f"{col}_outlier_flag"] = flags.to_numpy()
        summ.update(dataset=dataset, column=col)
        rows.append(summ)
        log.add(dataset, "outliers", summ["method"], col, summ["n_flagged"],
                "flagged only, nothing removed or capped")
    return out, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. MISSING VALUES AND OPTIONAL IMPUTATION
# ---------------------------------------------------------------------------
def apply_missing_policy(df, policy, dataset, log):
    """leave / flag / unknown_category, chosen per column meaning."""
    out = df.copy()
    for col, action in policy.items():
        if col not in out.columns:
            continue
        n = int(out[col].isna().sum())
        if n == 0:
            continue
        if action == "flag":
            out[f"{col}_missing_flag"] = out[col].isna()
            log.add(dataset, "missing", "numeric/date/text left missing", col, n,
                    "kept as NA + missing indicator")
        elif action == "unknown_category":
            out[f"{col}_missing_flag"] = out[col].isna()
            out[col] = out[col].fillna("unknown")
            log.add(dataset, "missing", "categorical missing", col, n,
                    "filled with 'unknown' + missing indicator")
        else:
            log.add(dataset, "missing", "left missing (handled elsewhere)", col, n,
                    "no change")
    return out


def impute_numeric_group_median(df, col, group_col, dataset, log):
    """Optional, off by default. Writes new columns; never overwrites the original."""
    out = df.copy()
    values = out[col].astype("float64")
    if group_col in out.columns:
        grp = out.groupby(group_col)[col].transform(
            lambda s: s.astype("float64").median())
    else:
        grp = pd.Series(np.nan, index=out.index)
    filled = values.fillna(grp.astype("float64"))
    n_group = int((values.isna() & filled.notna()).sum())
    filled = filled.fillna(values.median())
    n_global = int(filled.notna().sum() - values.notna().sum() - n_group)
    out[f"{col}_imputed"] = filled
    out[f"{col}_was_imputed"] = values.isna() & filled.notna()
    log.add(dataset, "missing", f"median imputation by {group_col}", col, n_group,
            "new column *_imputed (EDA use only)")
    log.add(dataset, "missing", "median imputation, global fallback", col,
            n_global, "new column *_imputed (EDA use only)")
    return out


# ---------------------------------------------------------------------------
# 8. TEXT, CATEGORY AND INGREDIENT STANDARDIZATION
# ---------------------------------------------------------------------------
def tidy_text(value):
    if pd.isna(value):
        return pd.NA
    t = unicodedata.normalize("NFKC", str(value))
    t = re.sub(r"\s+", " ", t).strip()
    return t if t else pd.NA


def standardize_category(df, col, synonyms, dataset, log):
    """Lower-case key + synonym map. Mapping table is returned for human review."""
    out = df.copy()
    before = out[col].map(tidy_text)
    key = before.map(lambda v: v.casefold() if pd.notna(v) else v)
    mapped = key.map(lambda v: synonyms.get(v, v) if pd.notna(v) else v)
    changed = int((mapped.astype("string") != before.astype("string"))
                  .fillna(False).sum())
    out[f"{col}_raw"] = df[col].astype("string")
    out[col] = mapped.astype("string")
    report = (pd.DataFrame({"raw": df[col].astype("string"),
                            "standardized": out[col]})
              .value_counts(dropna=False).reset_index(name="n_rows"))
    log.add(dataset, "standardize", "category case/spacing + synonym map", col,
            changed, "values standardized (mapping report saved)")
    return out, report


def map_step_type(text, rules):
    if pd.isna(text):
        return "unknown"
    low = str(text).casefold()
    for step_type, keywords in rules:
        if any(k in low for k in keywords):
            return step_type
    return "other"


MARKER_RE = re.compile(r"^(\+/-|\(\+/-\)|may contain)", re.I)


def _split_top_level(text):
    """Split on commas/semicolons that are not inside brackets."""
    parts, buf, depth = [], [], 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch in ",;\n" and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def parse_ingredients(value):
    """Return (list or None, status). Status separates missing from empty list."""
    if value is None or value is pd.NA or (
            isinstance(value, float) and np.isnan(value)):
        return None, "missing"
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if text == "":
        return None, "missing"
    items = None
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                items = [str(p) for p in parsed]
        except (ValueError, SyntaxError):
            items = None
    if items is None:
        items = _split_top_level(text.strip("[]"))
    cleaned = []
    for item in items:
        item = re.sub(r"\s+", " ", item).strip(" .*'\"")
        if MARKER_RE.match(item):          # 'may contain' / '+/-' ends the list
            break
        if item:
            cleaned.append(item.lower())
    unique = list(dict.fromkeys(cleaned))
    if not unique:
        return [], "empty_list"
    return unique, "ok"


def add_ingredient_features(df, col, dataset, log):
    out = df.copy()
    parsed = out[col].map(parse_ingredients)
    items = parsed.map(lambda r: r[0])
    status = parsed.map(lambda r: r[1])
    out["ingredient_status"] = status.astype("string")
    out["ingredient_count"] = pd.array(
        [len(i) if i is not None else pd.NA for i in items], dtype="Int64")
    out["ingredients_clean"] = pd.Series(
        [" | ".join(i) if i else pd.NA for i in items], index=out.index,
        dtype="string")
    for name, n in status.value_counts().items():
        log.add(dataset, "ingredients", f"status = {name}", col, n,
                "parsed; missing is NOT treated as an empty list")
    return out


def clean_review_text(text):
    """Safe normalization: keeps case, punctuation, negations and emojis."""
    if pd.isna(text):
        return pd.NA
    t = html.unescape(str(text))
    t = re.sub(r"<[^>]+>", " ", t)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"(https?://\S+|www\.\S+)", " ", t, flags=re.I)
    t = t.replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"')
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"([!?.])\1{3,}", r"\1\1\1", t)
    return t if t else pd.NA


def add_text_features(df, col, cfg, dataset, log):
    out = df.copy()
    raw = out[col].astype("string")
    clean = raw.map(clean_review_text).astype("string")
    out[f"{col}_clean"] = clean
    words = clean.str.split().map(lambda w: len(w) if isinstance(w, list) else 0)
    out["review_word_count"] = words.astype("Int64")
    out["review_char_count"] = clean.str.len().astype("Int64")
    out["review_text_missing_flag"] = clean.isna().to_numpy()
    out["review_very_short_flag"] = (clean.notna() & (
        words < cfg["very_short_review_words"])).to_numpy()
    long_enough = clean.notna() & (clean.str.len() >= 20)
    repeats = clean.where(long_enough).map(
        clean.where(long_enough).value_counts())
    out["review_text_repeated_flag"] = (repeats.fillna(0) > 1).to_numpy()
    n_changed = int((raw.notna() & (raw != clean).fillna(True)).sum())
    log.add(dataset, "text", "safe normalization (html, urls, spaces, quotes)",
            col, n_changed, "written to *_clean; original column kept")
    log.add(dataset, "text", "identical long text appears more than once", col,
            out["review_text_repeated_flag"].sum(), "flagged only, not removed")
    return out


def normalize_barcode(df, col, dataset, log):
    out = df.copy()
    raw = out[col].astype("string")
    digits = raw.str.replace(r"\D", "", regex=True)
    digits = digits.mask(digits == "", pd.NA)
    changed = int((raw.notna() & (raw != digits).fillna(True)).sum())
    out[f"{col}_raw"] = raw
    out[col] = digits
    out[f"{col}_length_invalid_flag"] = (
        digits.notna() & ~digits.str.len().isin([8, 12, 13, 14])).to_numpy()
    log.add(dataset, "identifiers", "barcode kept as digits-only text", col,
            changed, "normalized (never converted to a number)")
    log.add(dataset, "identifiers", "barcode length not in 8/12/13/14", col,
            out[f"{col}_length_invalid_flag"].sum(), "flagged only")
    return out


# ---------------------------------------------------------------------------
# 9. DATASET-LEVEL CLEANING FUNCTIONS
# ---------------------------------------------------------------------------
def _prepare(df_raw, dataset, cfg, log):
    df, name_map = standardize_column_names(df_raw, log, dataset)
    df = drop_export_index_columns(df, log, dataset)
    df = normalize_na_tokens(df, cfg["na_tokens"], log, dataset)
    resolved = resolve_columns(
        df, dataset, cfg["column_overrides"].get(dataset))
    df, _ = apply_canonical_names(df, resolved, log, dataset)
    check_required(resolved, dataset, df.columns)
    guard_forbidden(df, cfg)
    return df, name_map, resolved


def guard_forbidden(df, cfg):
    bad = [c for c in cfg["forbidden_columns"] if c in df.columns]
    if bad:
        raise ValueError(
            f"Public input contains {bad}. Public review data should not contain "
            "a dropout label. Check the source before continuing.")


def drop_sensitive_columns(df, cfg, dataset, log):
    if not cfg["exclude_sensitive"]:
        return df
    cols = [c for c in cfg["sensitive_columns"] if c in df.columns]
    for c in cols:
        log.add(dataset, "privacy", "sensitive profile column excluded", c,
                len(df), "dropped from cleaned output (raw file unchanged)")
    return df.drop(columns=cols)


def reject_missing_key(df, key, dataset, log):
    mask = df[key].isna()
    rejected = df.loc[mask].copy()
    rejected["reject_reason"] = f"missing {key}"
    log.add(dataset, "identifiers", f"missing {key} (cannot be linked)", key,
            mask.sum(), "moved to rejected file, not deleted silently")
    return df.loc[~mask].copy(), rejected


def clean_products(df_raw, cfg, log, dataset="products"):
    df, name_map, resolved = _prepare(df_raw, dataset, cfg, log)
    n_raw = len(df)
    df = drop_sensitive_columns(df, cfg, dataset, log)
    df, dup_exact = drop_exact_duplicates(df, None, dataset, log,
                                          "exact duplicate row")
    df, rejected = reject_missing_key(df, "product_id", dataset, log)
    df, dup_key = handle_key_duplicates(
        df, "product_id", cfg["key_duplicate_policy"], dataset, log)
    lo, hi = cfg["rating_range"]
    if "price" in df.columns:
        df = apply_range_rule(df, "price", cfg["price_min"], None, False,
                              dataset, log, "price >= 0")
        df["price_zero_flag"] = (df["price"] == 0).fillna(False).to_numpy()
        log.add(dataset, "validity", "price equals 0", "price",
                df["price_zero_flag"].sum(), "flagged only (could be a sample)")
    if "rating" in df.columns:
        df = apply_range_rule(df, "rating", lo, hi, False, dataset, log,
                              f"rating in [{lo}, {hi}]")
    for col in ("review_count", "loves_count"):
        if col in df.columns:
            df = apply_range_rule(df, col, 0, None, True, dataset, log,
                                  "count >= 0 and whole")
            df = to_nullable_int(df, col)
    for col in ("product_name", "brand_name"):
        if col in df.columns:
            df[col] = df[col].map(tidy_text).astype("string")
    cat_report = pd.DataFrame()
    if "category" in df.columns:
        df, cat_report = standardize_category(
            df, "category", cfg["category_synonyms"], dataset, log)
        df["routine_step_type"] = df["category"].map(
            lambda v: map_step_type(v, cfg["step_type_rules"])).astype("string")
    if "ingredients" in df.columns:
        df = add_ingredient_features(df, "ingredients", dataset, log)
    df = apply_missing_policy(df, cfg["missing_policy"][dataset], dataset, log)
    for col, spec in cfg["impute_numeric"].get(dataset, {}).items():
        if col in df.columns:
            df = impute_numeric_group_median(
                df, col, spec.get("group_by", "category"), dataset, log)
    df, outliers = investigate_outliers(df, cfg, dataset, log)
    acct = {"n_raw": n_raw, "n_exact_duplicates_removed": len(dup_exact),
            "n_key_duplicates_removed": len(dup_key),
            "n_rejected": len(rejected), "n_clean": len(df)}
    return {"clean": df, "rejected": rejected,
            "removed_duplicates": pd.concat([dup_exact, dup_key]),
            "outliers": outliers, "category_report": cat_report,
            "name_map": name_map, "accounting": acct, "resolved": resolved}


def clean_reviews(df_raw, cfg, log, dataset="reviews"):
    df, name_map, resolved = _prepare(df_raw, dataset, cfg, log)
    n_raw = len(df)
    df = drop_sensitive_columns(df, cfg, dataset, log)
    subset = [c for c in ("author_id", "product_id", "review_date",
                          "review_text", "rating") if c in df.columns]
    if "review_text" in subset and "product_id" in subset:
        df, dup_exact = drop_exact_duplicates(
            df, subset, dataset, log, "duplicate review")
    else:   # no review_text column: fall back to fully identical rows
        df, dup_exact = drop_exact_duplicates(
            df, None, dataset, log, "exact duplicate row (no review_text column)")
    df, rejected = reject_missing_key(df, "product_id", dataset, log)
    lo, hi = cfg["rating_range"]
    if "rating" in df.columns:
        df = apply_range_rule(df, "rating", lo, hi,
                              cfg["review_rating_integer_only"], dataset, log,
                              f"review rating in [{lo}, {hi}]")
    if "is_recommended" in df.columns:
        df = parse_binary(df, "is_recommended", dataset, log)
    if "review_date" in df.columns:
        df = parse_dates(df, "review_date", cfg, dataset, log)
    if "skin_type" in df.columns:
        df, _ = standardize_category(df, "skin_type", {}, dataset, log)
    if "review_text" in df.columns:
        df = add_text_features(df, "review_text", cfg, dataset, log)
    if "review_title" in df.columns:
        df["review_title_clean"] = df["review_title"].map(
            clean_review_text).astype("string")
    df = apply_missing_policy(df, cfg["missing_policy"][dataset], dataset, log)
    df, outliers = investigate_outliers(df, cfg, dataset, log)
    acct = {"n_raw": n_raw, "n_exact_duplicates_removed": len(dup_exact),
            "n_key_duplicates_removed": 0, "n_rejected": len(rejected),
            "n_clean": len(df)}
    return {"clean": df, "rejected": rejected, "removed_duplicates": dup_exact,
            "outliers": outliers, "category_report": pd.DataFrame(),
            "name_map": name_map, "accounting": acct, "resolved": resolved}


def clean_obf(df_raw, cfg, log, dataset="obf"):
    df, name_map, resolved = _prepare(df_raw, dataset, cfg, log)
    n_raw = len(df)
    df = drop_sensitive_columns(df, cfg, dataset, log)
    df, dup_exact = drop_exact_duplicates(df, None, dataset, log,
                                          "exact duplicate row")
    df = normalize_barcode(df, "barcode", dataset, log)
    df, rejected = reject_missing_key(df, "barcode", dataset, log)
    df, dup_key = handle_key_duplicates(
        df, "barcode", cfg["key_duplicate_policy"], dataset, log)
    for col in ("product_name", "brand_name", "countries"):
        if col in df.columns:
            df[col] = df[col].map(tidy_text).astype("string")
    cat_report = pd.DataFrame()
    if "category" in df.columns:
        df, cat_report = standardize_category(
            df, "category", cfg["category_synonyms"], dataset, log)
    if "ingredients" in df.columns:
        df = add_ingredient_features(df, "ingredients", dataset, log)
    df = apply_missing_policy(df, cfg["missing_policy"][dataset], dataset, log)
    acct = {"n_raw": n_raw, "n_exact_duplicates_removed": len(dup_exact),
            "n_key_duplicates_removed": len(dup_key), "n_rejected": len(rejected),
            "n_clean": len(df)}
    return {"clean": df, "rejected": rejected,
            "removed_duplicates": pd.concat([dup_exact, dup_key]),
            "outliers": pd.DataFrame(), "category_report": cat_report,
            "name_map": name_map, "accounting": acct, "resolved": resolved}


def check_join_feasibility(products, obf, log):
    """Report only. No join is performed: no shared identifier is assumed."""
    shared = sorted(set(products.columns) & set(obf.columns)
                    & {"barcode", "product_id"})
    if not shared:
        log.add("join", "enrichment", "no shared identifier column", None, 0,
                "join skipped; tables stay separate")
        return
    for key in shared:
        overlap = len(set(products[key].dropna()) & set(obf[key].dropna()))
        log.add("join", "enrichment", f"overlap on {key}", key, overlap,
                "report only; a join needs a documented mapping")


# ---------------------------------------------------------------------------
# 10. VALIDATION (run on the cleaned output)
# ---------------------------------------------------------------------------
def validate_output(result, dataset, cfg):
    df, acct = result["clean"], result["accounting"]
    checks = []

    def add(name, ok, detail="", warn=False):
        status = "PASS" if ok else ("WARN" if warn else "FAIL")
        checks.append({"dataset": dataset, "check": name, "status": status,
                       "detail": detail})

    key = KEY_COLUMN.get(dataset)
    for col in REQUIRED[dataset]:
        add(f"required column '{col}' present", col in df.columns)
    if key and key in df.columns:
        add(f"no missing {key}", df[key].notna().all())
    add("no exact duplicate rows", not df.duplicated().any())
    lo, hi = cfg["rating_range"]
    if "rating" in df.columns:
        r = df["rating"].dropna().astype("float64")
        add("ratings inside allowed range", bool(((r >= lo) & (r <= hi)).all()),
            f"allowed [{lo}, {hi}]")
    if "price" in df.columns:
        p = df["price"].dropna().astype("float64")
        add("prices non-negative", bool((p >= 0).all()))
    if "review_date" in df.columns:
        d = df["review_date"].dropna()
        add("dates inside allowed window",
            bool(d.empty or ((d >= pd.Timestamp(cfg["date_min"])).all()
                             and (d <= pd.Timestamp.now()).all())))
    bad = [c for c in cfg["forbidden_columns"] if c in df.columns]
    add("no dropout label columns in public data", not bad, str(bad))
    if cfg["exclude_sensitive"]:
        left = [c for c in cfg["sensitive_columns"] if c in df.columns]
        add("sensitive profile columns excluded", not left, str(left))
    expected = (acct["n_raw"] - acct["n_exact_duplicates_removed"]
                - acct["n_key_duplicates_removed"] - acct["n_rejected"])
    add("row accounting reconciles", expected == acct["n_clean"],
        f"{acct['n_raw']} raw = {acct['n_clean']} clean + "
        f"{acct['n_exact_duplicates_removed']} dup + "
        f"{acct['n_key_duplicates_removed']} key-dup + "
        f"{acct['n_rejected']} rejected")
    if "ingredient_status" in df.columns:
        miss = df["ingredient_status"] == "missing"
        empty = df["ingredient_status"] == "empty_list"
        ok = bool(df.loc[miss, "ingredient_count"].isna().all()
                  and (df.loc[empty, "ingredient_count"] == 0).all())
        add("ingredient missing vs empty list kept apart", ok)
    for col in cfg["identifier_columns"]:
        if col in df.columns:
            add(f"identifier '{col}' stored as text",
                str(df[col].dtype) in ("string", "object", "str"))
    for col, pct in (df.isna().mean() * 100).items():
        if pct > cfg["max_missing_pct_warn"]:
            add(f"missingness of '{col}' <= {cfg['max_missing_pct_warn']}%",
                False, f"{pct:.1f}% missing", warn=True)
    return checks


# ---------------------------------------------------------------------------
# 11. OUTPUT, MANIFEST, RUNNER
# ---------------------------------------------------------------------------
def deep_update(base, extra):
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def safe_output_path(out_dir, name, input_paths):
    target = (Path(out_dir) / name).resolve()
    if target in {Path(p).resolve() for p in input_paths if p}:
        raise ValueError(f"Refusing to overwrite an input file: {target}")
    return target


def run_pipeline(cfg, strict=False, nrows=None):
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = {k: v for k, v in cfg["paths"].items() if v}
    if not inputs:
        raise ValueError("No input files given. Use --products/--reviews/--obf.")
    for k, v in inputs.items():
        if Path(v).resolve().parent == out_dir.resolve():
            raise ValueError("output_dir must be different from the raw data folder.")
    log, manifest = CleaningLog(), {"inputs": {}, "outputs": [], "datasets": {}}
    results, all_checks = {}, []
    cleaners = {"products": clean_products, "reviews": clean_reviews,
                "obf": clean_obf}
    for dataset, path in inputs.items():
        raw, meta = load_csv_safely(path, cfg["separators"][dataset], nrows)
        manifest["inputs"][dataset] = meta
        prof, summ = profile_dataframe(raw)
        prof.to_csv(safe_output_path(out_dir, f"profile_before_{dataset}.csv",
                                     inputs.values()), index=False)
        log.add(dataset, "profile", "raw shape / exact duplicate rows", None,
                summ["n_exact_duplicate_rows"],
                f"{summ['n_rows']} rows x {summ['n_cols']} cols loaded as text")
        res = cleaners[dataset](raw, cfg, log)
        results[dataset] = res
        checks = validate_output(res, dataset, cfg)
        all_checks += checks
        files = {
            f"{dataset}_clean.csv": res["clean"],
            f"{dataset}_rejected_rows.csv": res["rejected"],
            f"{dataset}_removed_duplicates.csv": res["removed_duplicates"],
            f"outlier_report_{dataset}.csv": res["outliers"],
            f"category_mapping_{dataset}.csv": res["category_report"],
            f"column_name_map_{dataset}.csv": res["name_map"],
            f"profile_after_{dataset}.csv": profile_dataframe(res["clean"])[0],
        }
        for name, frame in files.items():
            if frame is not None and len(frame.columns):
                p = safe_output_path(out_dir, name, inputs.values())
                frame.to_csv(p, index=False)
                manifest["outputs"].append(str(p))
        manifest["datasets"][dataset] = res["accounting"]
    if "products" in results and "obf" in results:
        check_join_feasibility(results["products"]["clean"],
                               results["obf"]["clean"], log)
    log.to_frame().to_csv(out_dir / "cleaning_log.csv", index=False)
    checks_df = pd.DataFrame(all_checks)
    checks_df.to_csv(out_dir / "validation_report.csv", index=False)
    manifest.update(
        nrows_limit=nrows,
        run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        python=platform.python_version(), pandas=pd.__version__,
        numpy=np.__version__, config=cfg,
        validation_summary=checks_df["status"].value_counts().to_dict())
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    n_fail = int((checks_df["status"] == "FAIL").sum())
    LOGGER.info("Done. Validation: %s", manifest["validation_summary"])
    if strict and n_fail:
        raise SystemExit(f"{n_fail} validation check(s) failed (strict mode).")
    return results, checks_df


def inspect_files(cfg, nrows=2000):
    """Show columns, blank/NA counts and which canonical names were detected."""
    if not any(cfg["paths"].values()):
        print("Nothing to inspect: give --products, --reviews and/or --obf "
              "(or use --demo).")
        return
    for dataset, path in cfg["paths"].items():
        if not path:
            continue
        raw, meta = load_csv_safely(path, cfg["separators"][dataset], nrows)
        prof, _ = profile_dataframe(raw)
        cells = prof.set_index("column")[["n_blank", "n_missing"]]
        log = CleaningLog()
        df, _ = standardize_column_names(raw, log, dataset)
        resolved = resolve_columns(df, dataset,
                                   cfg["column_overrides"].get(dataset))
        print(f"\n=== {dataset}: {path} ({meta['n_rows_read']} rows read, "
              f"encoding {meta['encoding']}) ===")
        print("columns:", list(df.columns))
        print("blank cells per column:",
              {c: int(v) for c, v in cells["n_blank"].items() if v})
        print("detected :", {k: v for k, v in resolved.items() if v})
        print("NOT found:", [k for k, v in resolved.items() if not v])
        print("Fix wrong or missing matches with column_overrides in the config.")


def make_demo_inputs(folder):
    """SIMULATED rows with deliberate problems. Not Sephora, not real data."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    products = pd.DataFrame({
        "Unnamed: 0": range(13),
        "Product ID": ["SIM_P001", "SIM_P002", "SIM_P003", "SIM_P003", "SIM_P004",
                       "SIM_P005", "SIM_P005", "SIM_P006", "SIM_P007", "SIM_P008",
                       "SIM_P009", "", "SIM_P010"],
        "Product Name": ["Gentle Foam", "Dew Serum", "Day Cream", "Day Cream",
                         "Light Lotion", "Night Gel", "Night Gel ", "Eye Balm",
                         "Luxury Set", "Clay Mask", "Toner X", "Ghost Row",
                         "Plain Lotion"],
        "Brand": ["BrandA", "BrandB", "BrandA", "BrandA", "BrandC", "BrandB",
                  "BrandB", "N/A", "BrandD", "BrandC", "BrandA", "BrandA",
                  "BrandB"],
        "Primary Category": ["Cleansers", " moisturizer ", "MOISTURISER",
                             "MOISTURISER", "Moisturizers", "Moisturizers",
                             "Moisturizers", "Eye Care", "Skincare Sets",
                             "Masks", "Toners", "Masks", "Moisturizers"],
        "Price (USD)": ["18.00", "42.00", "30.00", "30.00", "", "28.00", "29.50",
                        "-5.00", "$1,250.00", "22.00", "19.00", "10.00", "15.00"],
        "Rating": ["4.5", "4.2", "7.2", "7.2", "N/A", "3.9", "3.9", "4.0", "4.8",
                   "4.1", "4.4", "4.0", "3.5"],
        "Reviews": ["120", "88", "45", "45", "30", "-3", "60", "12", "5", "70",
                    "1.5k", "9", "40"],
        "Loves Count": ["900", "700", "300", "300", "250", "410", "410", "80",
                        "20", "500", "610", "5", "330"],
        "Ingredients": ["['Water', 'Glycerin', 'Niacinamide']",
                        "Water, Glycerin, Phenoxyethanol (and) Ethylhexylglycerin,"
                        " +/- Mica", "[]", "[]", "N/A",
                        "Water; Dimethicone; Parfum", "Water; Dimethicone; Parfum",
                        "", "['Water', 'Gold']", "Kaolin, Water", "Water, Aloe",
                        "Water", "Water, Glycerin, Water"],
    })
    reviews = pd.DataFrame({
        "Unnamed: 0": range(14),
        "author_id": ["A1", "A2", "A3", "A3", "A4", "A5", "A6", "A7", "A8", "A9",
                      "A10", "A11", "A12", "A13"],
        "product_id": ["SIM_P001", "SIM_P001", "SIM_P002", "SIM_P002", "SIM_P003",
                       "SIM_P004", "SIM_P005", "SIM_P005", "SIM_P006", "SIM_P007",
                       "SIM_P008", "SIM_P008", "", "SIM_P009"],
        "rating": ["5", "4", "2", "2", "0", "6", "3", "4.5", "1", "5", "4", "4",
                   "3", "N/A"],
        "is_recommended": ["1.0", "yes", "0.0", "0.0", "0", "maybe", "0", "1",
                           "0", "1", "1", "1", "1", ""],
        "submission_time": ["2023-04-01", "03/15/2023", "15 Jan 2022", "15 Jan 2022",
                            "not a date", "2031-01-01", "2022-07-09", "2022-08-10",
                            "2021-12-24", "2020-02-02", "2023-01-05", "2023-01-06",
                            "2022-05-05", ""],
        "review_text": [
            "Loved it. It didn’t irritate my skin!!!!!!",
            "Too many steps<br/>  and it takes  too long. See https://x.test/a",
            "Not worth the price.", "Not worth the price.", "ok", "",
            "Burned a little, would not buy again.", "Works well &amp; smells nice",
            "Very sticky and messy to apply every night",
            "Good value, texture is light and easy to use.",
            "Great product, great product, will repurchase every month",
            "Great product, great product, will repurchase every month",
            "Missing product id on this one", "great"],
        "review_title": ["Nice", "Too long", "Pricey", "Pricey", "", "", "Meh",
                         "Good", "Messy", "Good", "Great", "Great", "x", ""],
        "skin_type": ["Dry", "dry ", "OILY", "OILY", "Combination", "", "Oily",
                      "Normal", "Dry", "Oily", "Dry", "Dry", "Dry", "Normal"],
        "skin_tone": ["light", "medium", "deep", "deep", "light", "", "tan",
                      "light", "medium", "deep", "tan", "tan", "light", "medium"],
    })
    obf = pd.DataFrame({
        "code": ["3600550123456", "3 600550-123457", "3600550123456", "12345",
                 "3600550123458", ""],
        "product_name": ["Face Wash A", "Cream B", "Face Wash A", "Odd Item",
                         "Serum C", "No Code"],
        "brands": ["BrandA", "BrandB", "BrandA", "BrandX", "BrandC", "BrandZ"],
        "categories": ["Cleansers", "Moisturizers", "Cleansers", "Other",
                       "Serums", "Other"],
        "ingredients_text": ["Aqua, Glycerin", "Aqua; Parfum", "Aqua, Glycerin",
                             "", "[]", "Aqua"],
        "countries": ["France", "France", "France", "", "Spain", "Italy"],
    })
    p = {"products": folder / "SIMULATED_products.csv",
         "reviews": folder / "SIMULATED_reviews.csv",
         "obf": folder / "SIMULATED_obf.tsv"}
    products.to_csv(p["products"], index=False)
    reviews.to_csv(p["reviews"], index=False)
    obf.to_csv(p["obf"], index=False, sep="\t")
    return {k: str(v) for k, v in p.items()}


def build_sklearn_preprocessor(numeric_cols, categorical_cols):
    """FOR FUTURE MODELLING. Not fitted here. Fit on the training split only."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    numeric = Pipeline([("impute", SimpleImputer(strategy="median",
                                                 add_indicator=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, numeric_cols),
                              ("cat", categorical, categorical_cols)])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Week 3 cleaning pipeline")
    p.add_argument("--products")
    p.add_argument("--reviews")
    p.add_argument("--obf")
    p.add_argument("--output-dir")
    p.add_argument("--config", help="JSON file that overrides CONFIG keys")
    p.add_argument("--inspect", action="store_true",
                   help="only show columns and detected mapping")
    p.add_argument("--demo", action="store_true",
                   help="run on SIMULATED data to test the code")
    p.add_argument("--strict", action="store_true",
                   help="exit with an error if a validation check fails")
    p.add_argument("--nrows", type=int, help="read only the first N rows")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = copy.deepcopy(CONFIG)
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            deep_update(cfg, json.load(f))
    for key in ("products", "reviews", "obf"):
        if getattr(args, key):
            cfg["paths"][key] = getattr(args, key)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.demo:
        cfg["output_dir"] = args.output_dir or "demo_output"
        cfg["paths"] = make_demo_inputs(Path(cfg["output_dir"]) / "simulated_inputs")
        cfg["impute_numeric"] = {"products": {"price": {"group_by": "category"}}}
    handlers = [logging.StreamHandler(sys.stdout)]
    if not args.inspect:           # --inspect writes no files
        Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
        handlers.insert(0, logging.FileHandler(
            Path(cfg["output_dir"]) / "pipeline.log", mode="w", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers)
    if args.demo:
        LOGGER.warning("DEMO MODE: all rows are SIMULATED. Counts are not "
                       "results from any real dataset.")
    try:
        if args.inspect:
            inspect_files(cfg, nrows=args.nrows or 2000)
            return 0
        run_pipeline(cfg, strict=args.strict, nrows=args.nrows)
    except (FileNotFoundError, KeyError, ValueError) as err:
        message = err.args[0] if isinstance(err, KeyError) and err.args else err
        LOGGER.error("%s: %s", type(err).__name__, message)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
