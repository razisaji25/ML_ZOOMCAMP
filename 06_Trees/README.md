# 06_Trees

Module 6 of Machine Learning Zoomcamp: decision trees, random forests and gradient boosting (XGBoost), applied to the 2026 car fuel-efficiency dataset (`fuel_efficiency_mpg` as the target).

## Contents

- [hw6.py](hw6.py) - Homework 6 solution script.
- `car_fuel_efficiency_2026.csv` - dataset, not tracked in git (see root `.gitignore`). Download it with:

```bash
wget https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv
```

## Setup

Missing values are filled with zeros. The data is split 60/20/20 with `random_state=1` (train / validation / test), and features are vectorized with `DictVectorizer(sparse=True)`.

## Run

XGBoost is not listed in the root `pyproject.toml`, so add it for this run:

```bash
uv run --with xgboost python hw6.py
```

## Homework 6 answers

| Question | Result | Answer |
|---|---|---|
| Q1 | A depth-1 tree splits on `model_year` | `model_year` |
| Q2 | Random forest (10 trees) validation RMSE = 1.833 | 1.837 (closest option) |
| Q3 | RMSE 1.833 / 1.774 / 1.770 / 1.769 for 10 / 50 / 100 / 150 trees | 150 |
| Q4 | Mean RMSE 1.754 / 1.784 / 1.787 / 1.786 for depth 10 / 15 / 20 / 25 | 10 |
| Q5 | Among the options: `vehicle_weight` 0.203, `horsepower` 0.078, `engine_displacement` 0.061 (`model_year` ranks higher overall at 0.296 but is not an option) | `vehicle_weight` |
| Q6 | Validation RMSE after 100 rounds: 1.826 (eta 0.3) vs 1.725 (eta 0.1) | 0.1 |
