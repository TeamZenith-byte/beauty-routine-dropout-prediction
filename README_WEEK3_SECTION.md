## Week 3: Data Cleaning and Preprocessing Pipeline

Week 3 designs and implements a documented cleaning pipeline for the **public** product, review and ingredient data (Layer 1 of the data strategy). Public data contains only proxy signals; it has no verified `routine_dropout` label. That label is planned to come from the future survey and routine diary (Layer 2). BRFS remains my own proposed framework, and the five Week 1 hypotheses (H1 to H5) are still untested.

**Status:** the pipeline has been tested only on built-in **simulated** rows (`--demo`). It has **not** yet been run on the real Sephora or Open Beauty Facts files, so no real-data results are reported.

| Item | Link |
|---|---|
| Week 3 report (Word) | [reports/Week_3_Data_Cleaning_Preprocessing_Final.docx](reports/Week_3_Data_Cleaning_Preprocessing_Final.docx) |
| Cleaning script | [src/week3_preprocessing.py](src/week3_preprocessing.py) |
| Run instructions | [README_Week3.md](README_Week3.md) |
| Pipeline flowchart | [reports/figures/flowchart_preprocessing.png](reports/figures/flowchart_preprocessing.png) |

Quick check (from the repository root, needs `pandas` and `numpy`):

```bash
python src/week3_preprocessing.py --demo --strict
```
