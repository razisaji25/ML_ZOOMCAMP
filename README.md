# MLZOOMCAMP

Personal workspace for following [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) (DataTalksClub).

## Environment

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies.

- **Python version:** 3.11 (pinned in `.python-version`)
- **Dependency file:** `pyproject.toml` (locked in `uv.lock`)

Setup:

```bash
uv sync
```

Run a script:

```bash
uv run python main.py
```

Run JupyterLab:

```bash
uv run jupyter lab
```

A Jupyter kernel named **"Python (mlzoomcamp)"** is registered from this environment and includes:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Jupyter / IPykernel

## Folders

| Folder | Description |
|---|---|
| [01_Intro](01_Intro) | Introduction to ML: pandas basics, exploratory analysis, and a linear regression exercise on the Car Fuel Efficiency dataset. |

Each folder has its own `README.md` describing its specific contents in more detail.
