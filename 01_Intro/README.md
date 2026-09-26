# 01_Intro

Introductory module: pandas basics and exploratory data analysis, using the **Car Fuel Efficiency 2026** dataset.

## Contents

- [Car Fuel Efficiency Analysis.ipynb](Car%20Fuel%20Efficiency%20Analysis.ipynb) — Notebook answering the module's homework questions:
  - **Q2** — Number of records in the dataset
  - **Q3** — Number of distinct fuel types (Gasoline, Diesel, Hybrid)
  - **Q4** — Number of columns with missing values
  - **Q5** — Maximum fuel efficiency among cars from Asia
  - **Q6** — Median horsepower before/after filling missing values with the mode
  - **Q7** — Sum of weights `w` from a manual linear regression (normal equation) using `vehicle_weight` and `model_year`
- [data/](data) — Raw dataset(s) used in the analysis. Not tracked in git (see root `.gitignore`).

## Data source

```bash
curl -o data/car_fuel_efficiency_2026.csv https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv
```
