# Beauty Routine Dropout Prediction

A Data Science project developed during the YuvaIntern Data Science with Python – Analyst Internship.

## Project Overview

This project investigates why customers abandon multi-step skincare and beauty routines.

The study focuses on factors such as:

- Number of routine steps
- Time burden
- Cost
- Product compatibility
- Irritation or dissatisfaction
- Product ratings and reviews
- Replacement difficulty

## Proposed Analytical Framework

### Beauty Routine Friction Score (BRFS)

BRFS is a proposed analytical framework developed for this project to organize and analyze factors that may contribute to beauty routine friction and eventual dropout.

It is not presented as an established industry metric.

## Internship Progress

- ✅ Week 1 — Problem Definition & Research Planning
- ✅ Week 2 — Data Identification, Sourcing Strategy & Initial Analysis Planning
- ⏳ Week 3 — 
- 🟡 Week 3 — Data Cleaning and Preprocessing Pipeline (simulated-data demo completed; real-data testing pending)
## Data Strategy

The project uses publicly available secondary data as a reference for product, ingredient, rating and review-related analysis.

Potential sources include:

- Kaggle — Sephora Products and Skincare Reviews
- Open Beauty Facts
- INCIDB — optional ingredient enrichment

For actual routine-dropout ground truth, a future primary survey and routine diary approach is planned.

## Important Methodological Note

Public product and review datasets do not directly provide a verified routine-dropout label.

Therefore, public data will be treated as contextual or proxy data rather than being presented as actual customer dropout ground truth.

---

Week 3: Data Cleaning and Preprocessing Pipeline

Objective

Develop a reproducible pipeline to prepare skincare product and review data for future analysis.

Work Completed

- Developed a Python preprocessing script using pandas and NumPy.
- Added data profiling, duplicate detection, missing-value assessment, standardisation, and validation checks.
- Created a preprocessing flowchart and technical report.
- Tested the workflow using simulated data.

Project Files

- "week3_preprocessing.py"
- "Week_3_Data_Cleaning_Preprocessing_Final.docx"
- "flowchart_preprocessing.png"
- "README_Week3.md"

Limitations

The pipeline has been tested using simulated data only. The actual Sephora and Open Beauty Facts datasets have not yet been processed. Therefore, the demo results must not be interpreted as real-world findings.

The public datasets do not provide a verified routine-dropout label. The proposed Beauty Routine Friction Score (BRFS) remains a framework for future investigation.

Next Steps

- Inspect the actual datasets and verify their licences.
- Configure the pipeline for the available columns.
- Run validation on real data.
- Conduct exploratory data analysis after checking data quality.

## Repository Structure

```text
beauty-routine-dropout-prediction/
│
├── docs/
│   ├── .gitkeep
│   └── Week_2_Data_Sourcing_Strategy_Final.docx
│
├── README.md
└── Week_1_Data_Science_Report.docx
