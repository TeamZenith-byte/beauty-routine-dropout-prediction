# Week 3 – Data Cleaning and Preprocessing Pipeline

Project: **Beauty Routine Dropout Prediction** (YuvaIntern, Data Science with Python)
Author: Zishan Alam
Repository: https://github.com/TeamZenith-byte/beauty-routine-dropout-prediction

`week3_preprocessing.py` cleans the **public** product, review and ingredient tables
(Layer 1 of the Week 2 data strategy) and writes separate cleaned files.
Raw input files are never modified.

> **Honest status.** The script has been tested only on the built-in **simulated**
> demo rows (`--demo`). It has **not** been run on the real Sephora or Open Beauty
> Facts files. Column names in the script are candidates and must be checked with
> `--inspect` first. The script does **not** create a `routine_dropout` label; that
> label can only come from the future survey / routine diary (Layer 2).

## 1. Requirements

- Python 3.9+
- `pandas`, `numpy` (required)
- `scikit-learn` (optional, only for `build_sklearn_preprocessor`, which is not run by the pipeline)

```bash
pip install pandas numpy scikit-learn
```

## 2. Folder layout (the commands below assume this)

```
beauty-routine-dropout-prediction/      <- run every command from this folder
├── data/raw/                           # downloaded files, never edited
├── data/processed/                     # pipeline output (default --output-dir)
├── src/week3_preprocessing.py
├── reports/figures/flowchart_preprocessing.png
└── README_Week3.md
```

All relative paths (`data/raw/...`, `data/processed`, `demo_output`) are relative to the
folder you run the command from. If you keep the script in the repository root instead,
just drop `src/` from the commands.

## 3. Run it

```bash
# (a) Self-test on SIMULATED rows (no downloads needed). Writes ./demo_output/
python src/week3_preprocessing.py --demo --strict
echo $?        # 0 = finished and all validation checks passed (on simulated rows)

# (b) Look at the real files first: columns, blank counts, detected names
python src/week3_preprocessing.py --inspect \
    --products data/raw/products.csv --reviews data/raw/reviews.csv --obf data/raw/obf.csv

# (c) Quick trial on the first 1000 rows of each file
python src/week3_preprocessing.py --products data/raw/products.csv --nrows 1000 \
    --output-dir data/processed_trial

# (d) Full run
python src/week3_preprocessing.py \
    --products data/raw/products.csv \
    --reviews  data/raw/reviews.csv \
    --obf      data/raw/obf.csv \
    --output-dir data/processed --strict
```

Options:

| Option | Behaviour |
|---|---|
| `--demo` | Writes SIMULATED input files to `<output-dir>/simulated_inputs/` and cleans them. Default output folder is `demo_output`. The demo also switches on the optional price imputation to test it (off in a normal run). |
| `--inspect` | Cleans nothing and writes no files. Prints the columns, the number of blank cells per column, and which canonical names were detected or not found. Uses the first 2000 rows unless `--nrows` is given. |
| `--strict` | Exit code 1 if any validation check is FAIL (outputs are still written so you can read `validation_report.csv`). WARN does not fail the run. |
| `--nrows N` | Read only the first N rows of every input (also used by `--inspect`). The value is saved in `run_manifest.json`. Row counts of such a trial run are not full-data counts. |
| `--config file.json` | Override settings in `CONFIG` (see section 4). |
| `--output-dir` | Output folder. It must not be the folder that holds an input file. |

Exit codes: `0` finished; `1` `--strict` and a validation check failed; `2` input or
configuration error (file not found, required column not found, label-like column such as
`routine_dropout` in public input, output folder equal to the input folder). Any of the three
inputs can be left out.

Open Beauty Facts exports are usually tab-separated; the default separator for `obf` is
a tab. Change it in the config if your file differs.

## 4. Config file example

Only the keys you want to change are needed; they are merged into the defaults.

```json
{
  "separators": {"obf": "\t"},
  "column_overrides": {"products": {"price": "price_usd"}},
  "rating_range": [1, 5],
  "key_duplicate_policy": "flag",
  "missing_policy": {"products": {"brand_name": "unknown_category"}},
  "impute_numeric": {"products": {"price": {"group_by": "category"}}}
}
```

- `column_overrides`: use when `--inspect` shows that auto-detection chose the wrong column.
  Write the column name as shown in `--inspect` after snake_case conversion
  (for example `price_usd`, not `Price (USD)`).
- `impute_numeric` is **empty by default**. If switched on, the imputed values go into a new
  `*_imputed` column and the original column stays untouched (for EDA only).
- `category_synonyms` and `step_type_rules` are draft examples; extend them after looking at
  the real value counts.

## 5. Outputs (in `--output-dir`)

| File | Content |
|---|---|
| `{dataset}_clean.csv` | cleaned table with `*_raw`, `*_invalid_flag`, `*_missing_flag`, `*_outlier_flag` columns |
| `{dataset}_rejected_rows.csv` | rows moved out because the key identifier was missing |
| `{dataset}_removed_duplicates.csv` | exact duplicates that were removed |
| `outlier_report_{dataset}.csv` | IQR fences and number of flagged values (flag only) |
| `category_mapping_{dataset}.csv` | original vs standardized category values |
| `column_name_map_{dataset}.csv` | original column name vs snake_case name |
| `profile_before_/profile_after_{dataset}.csv` | per-column profile before and after cleaning. `n_missing` counts genuine NA; `n_blank` counts empty or whitespace-only text separately (a raw file is read as text, so its empty cells show up under `n_blank`) |
| `cleaning_log.csv` | every cleaning decision with the number of rows affected |
| `validation_report.csv` | PASS / WARN / FAIL checks on the cleaned output |
| `run_manifest.json` | package versions, SHA-256 of inputs, config used, row accounting |
| `pipeline.log` | text log of the run |

`{dataset}` is `products`, `reviews` or `obf`. Files for a table that has no columns
to report (for example the outlier report of `obf`) are not written.

## 6. What the script deliberately does not do

- No `routine_dropout` label. Columns named `routine_dropout`, `dropout_reason` or
  `adherence_days` in a public input stop the run with an error (target-leakage guard).
- No joins between products and Open Beauty Facts. It only reports whether a shared
  identifier exists.
- No automatic deletion of outliers; they are flagged.
- No imputation unless configured.
- Sensitive profile columns (`skin_tone`, `eye_color`, `hair_color`) are dropped from the output.
- It does not fit scaling or encoding. `build_sklearn_preprocessor` only builds a
  `ColumnTransformer` that must be fitted on the training split later.

## 7. After the real run

1. Read `validation_report.csv` (every FAIL must be fixed) and `cleaning_log.csv`.
2. Copy the real row counts into the Week 3/Week 4 report; the numbers in the Week 3 report's
   demo section are simulated and must not be quoted as real results.
3. Commit the script, README and the config you used. Do not commit raw data if its licence
   does not allow redistribution.
