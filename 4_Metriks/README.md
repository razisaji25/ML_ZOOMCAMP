# 4_Metriks

Module: Evaluation Metrics for Classification — ROC AUC, precision, recall, F1, cross-validation, dan tuning parameter `C` pada dataset **Course Lead Scoring 2026**.

## Contents

- [Homework 4 Metriks.ipynb](Homework%204%20Metriks.ipynb) — Notebook jawaban Homework 4 (sudah dijalankan, kernel `mlzoomcamp`).
- [data/](data) — Dataset `course_lead_scoring_2026.csv`. Tidak di-track di git (lihat root `.gitignore`).

## Jawaban Homework 4

Missing values: ada di `lead_source` (148), `industry` (240), `employment_status` (194), `location` (208), `annual_income` (369), dan `lead_score` (35). Kategorikal diisi `'NA'`, numerik diisi `0.0`.

| Soal | Hasil | Jawaban |
|---|---|---|
| Q1 | AUC tertinggi: `lead_score` (0.788), lalu `interaction_count` (0.765), `number_of_courses_viewed` (0.723), `annual_income` (0.608) | `lead_score` |
| Q2 | AUC validasi = 0.732 | **0.732** |
| Q3 | Precision dan recall berpotongan di threshold 0.63 (selisih terkecil 0.0075) | **0.63** |
| Q4 | F1 maksimal (0.758) di threshold 0.41 | **0.41** |
| Q5 | Rata-rata AUC 5-fold = 0.732, std = 0.007 | **0.007** |
| Q6 | C=0.000001 -> 0.617 (std 0.019); C=0.001 -> 0.749 (std 0.010); C=1 -> 0.732 (std 0.007) | **0.001** |

Catatan: std dihitung dengan `np.std` (ddof=0).
