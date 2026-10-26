# Homework 5: Deploying Machine Learning Models — My Notes

Module 5 of ML Zoomcamp 2026. I serve a frozen lead-scoring model (`DictVectorizer` + `LogisticRegression`) through a FastAPI service and run it in a Docker container. The assignment is in [homework.md](homework.md).

## Setup

Before using the model artifact, I checked its integrity as the homework requires:

```bash
sha256sum pipeline.bin
# 1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1  pipeline.bin
```

The checksum matches the one in the homework and in `model_metadata.json`.

I installed the environment with `uv sync --locked --python 3.11`. The homework pins Python 3.11.15, but my `uv` (0.9.13) cannot download that build yet, so I used Python 3.11.14. The lockfile accepts any 3.11 and the pinned packages are identical (scikit-learn 1.7.2, numpy 2.3.3). `smoke_test.py` passes.

Git on Windows converted `feature_defaults.json` to CRLF line endings when I cloned, which changed its SHA-256 (the `/health` endpoint reported `5cb9195a...` instead of `1694f842...` from `model_metadata.json`). I converted the file back to LF and the checksum matches again. Predictions were not affected.

## Answers

| Question | What I did | Answer |
|---|---|---|
| Q1 | Run `uv --version`| `uv 0.9.13` |
| Q2 | Opened `pyproject.toml` (`scikit-learn==1.7.2`) and checked `uv.lock` (`name = "scikit-learn"`, `version = "1.7.2"`). Both agree, and the model file itself was trained with 1.7.2. | `1.7.2` |
| Q3 | Loaded `pipeline.bin` with `pickle` and called `predict_proba` on the first lead (script: [q3.py](q3.py)). Probability of conversion = 0.532994. | `0.533` |
| Q4 | Started the API with `uv run uvicorn predict:app --host 0.0.0.0 --port 9696` and sent the second lead to `POST /predict` with `curl`. Response: `{"conversion_probability":0.769799,"conversion":true}`. | `0.770` |
| Q5 | Read the `FROM` line of the `Dockerfile`: `python:3.11.15-slim-bookworm` (pinned by digest `sha256:d29f48a3...`). This is a declaration in version control; no image was built. | `python:3.11.15-slim-bookworm` |
| Q6 | Built the image (`docker build -t zoomcamp-model:2026-hw5 .`), ran it with `docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5`, and ran `python q6_test.py` against it. Response: `{'conversion_probability': 0.769799, 'conversion': True}`, identical to Q4. `GET /health` returned `status: ok` with `model_sha256` and `feature_defaults_sha256` equal to the values in `model_metadata.json`. | `0.770` |
