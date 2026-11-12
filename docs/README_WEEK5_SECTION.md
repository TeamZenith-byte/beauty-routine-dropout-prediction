## Week 5: Predictive Modeling and Algorithm Selection

Week 5 is a **design and planning** report. It chooses and explains the models that would estimate dropout risk once real labelled data exist. It does **not** contain real-data results, model scores or a "best model" claim. There is currently **no verified real-world `routine_dropout` label**: the public Sephora and Open Beauty Facts data are contextual or proxy sources, the Week 3 pipeline was tested on simulated rows only, the public datasets have not yet been processed through it, BRFS is my own proposed framework with unvalidated weights, and H1 to H5 are untested hypotheses. Genuine dropout prediction needs a future survey, diary or longitudinal routine-usage dataset.

### Work completed

- Defined the proposed target `routine_dropout` (1 = dropout, 0 = continuation) as a binary classification task and justified it against regression and clustering.
- Explained three fixed models for this problem: **Logistic Regression** (primary baseline), **Decision Tree** (interpretable comparison) and **Random Forest** (ensemble comparison), with equations, hyper-parameters, overfitting controls and hypothetical skincare examples.
- Sorted the Week 4 features F01 to F23 by readiness and proposed seven survey/diary features (F24 to F30).
- Wrote the preprocessing, leakage-prevention, evaluation (recall, F1, PR-AUC, ROC-AUC, confusion matrix; RMSE and MAE as regression-only metrics), validation and over/underfitting plans.
- Wrote a scikit-learn template (`src/week5_modeling_template.py`). Every split is group-aware when a participant has several rows, and the test set is scored once per model after everything else is frozen. It was run end to end on **simulated** rows only, to check the code; demo outputs carry a `SIMULATED_` prefix. Its demo output is not a project result. The optional SMOTE path was not executed (imbalanced-learn could not be installed in the build environment).

### Project files

| File | Purpose |
|------|---------|
| `reports/Week_5_Predictive_Modeling_Report.docx` | Week 5 report (22 sections, table of contents, references) |
| `reports/figures/week5_modeling_workflow.png` | Proposed modelling workflow diagram |
| `src/week5_modeling_template.py` | Modelling template (`--demo` uses simulated data) |
| `docs/model_comparison_table_week5.csv` | Three-model comparison table |
| `docs/feature_readiness_table_week5.csv` | Feature readiness (F01 to F30): planned vs available, and template column names |
| `docs/data_dictionary_TEMPLATE.csv` | Data dictionary template; every column is PLANNED, none exists in real data yet |
| `docs/Week_5_Compliance_Checklist.md` | Requirement checklist and items needing real data |
| `docs/Week_5_Technical_Correction_Log.md` | Final technical corrections |
| `requirements-week5.txt` | Library versions |

### Run the code check

```bash
pip install -r requirements-week5.txt
python src/week5_modeling_template.py --demo                 # SIMULATED data: tests the code, not results
python src/week5_modeling_template.py --demo --demo-grouped  # SIMULATED, several rows per participant
```

### Limitations

No model has been trained on real data. Public data cannot supply a dropout label. All business uses described in the report are potential applications, not proven findings. The Kaggle dataset licence has not been verified.
