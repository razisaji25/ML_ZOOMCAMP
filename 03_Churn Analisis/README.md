# 03_Churn Analisis

Module: Classification — data preparation, correlation, mutual information, dan logistic regression (one-hot encoding, feature elimination, regularisasi `C`) pada dataset **Lead Scoring 2026**. Target: `converted`.

## Contents

- [Lead Scoring Classification.ipynb](Lead%20Scoring%20Classification.ipynb) — Notebook jawaban Homework 3.
- [data/](data) — Dataset. Tidak di-track di git (lihat root `.gitignore`).

## Dataset

```bash
wget -P data https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv
```

## Jawaban Homework 3

| Soal | Jawaban |
|---|---|
| Q1 — mode `industry` | `technology` |
| Q2 — korelasi terbesar | `interaction_count` & `lead_score` (0.916) |
| Q3 — mutual information terbesar | `lead_source` (0.03) |
| Q4 — akurasi validasi | 0.65 |
| Q5 — selisih terkecil | `number_of_courses_viewed` (0.002) |
| Q6 — `C` terbaik | 0.001 (akurasi 0.645) |

## Catatan

- Di pandas 3 kolom string bertipe `str`, bukan `object` — pilih kolom kategorikal dengan `df.select_dtypes(exclude='number')`.
- Q5: "smallest difference" berarti nilai `base_acc - acc` paling kecil (boleh negatif), bukan nilai absolut.
