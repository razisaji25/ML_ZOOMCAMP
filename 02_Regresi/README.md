# 02_Regresi

Module: Linear Regression — exploratory data analysis, handling long tail distribution, dan implementasi regresi linear pada dataset **Car Fuel Efficiency 2026**.

## Contents

- [Regresi Car Fuel.ipynb](Regresi%20Car%20Fuel.ipynb) — Notebook analisis dan regresi.
- [data/](data) — Dataset. Tidak di-track di git (lihat root `.gitignore`).

## Long Tail Distribution

Cara mengecek long tail pada suatu kolom:

```python
df['kolom'].skew()          # >1 atau <-1 = long tail kuat
df['kolom'].hist(bins=50)   # cek visual, ekor panjang di satu sisi
```

Interpretasi nilai skew:

- **≈ 0** → distribusi hampir simetris / mendekati normal
- **> 1** → long tail kuat ke kanan (skew positif)
- **< -1** → long tail kuat ke kiri (skew negatif)

Long tail menandakan sebagian besar data terkumpul di satu area, tapi ada sedikit **outlier** dengan nilai sangat besar/kecil yang menarik ekor distribusi memanjang. Ini bisa mengganggu model ML — terutama model linear/regresi yang sensitif terhadap outlier — sehingga biasanya perlu **transformasi** sebelum data dipakai untuk training.

## Transformasi untuk Mengatasi Skewness

| Transformasi | Kapan Dipakai | Syarat |
|---|---|---|
| **Log transform** (`np.log1p`) | Skew positif kuat (long tail kanan), umum untuk harga/pendapatan/populasi. Paling sering dipakai untuk target di linear regression. | Nilai ≥ 0 (`log1p` aman untuk 0) |
| **Square root** (`np.sqrt`) | Skew positif ringan-sedang, cocok untuk count data (data hasil hitungan). | Nilai ≥ 0 |
| **Box-Cox** | Skew tidak diketahui bentuknya, ingin dicari transformasi optimal otomatis (parameter λ). | Semua nilai harus > 0 (strictly positive) |
| **Yeo-Johnson** | Sama seperti Box-Cox tapi data mengandung nilai negatif atau nol. | Bisa untuk nilai negatif/nol |
| **Reciprocal** (`1/x`) | Skew sangat ekstrem, log saja belum cukup. Cocok untuk hubungan inverse (mis. waktu vs kecepatan). | Nilai tidak boleh 0 |
| **Clipping / Winsorizing** | Bukan untuk skew keseluruhan, tapi ketika masalahnya cuma beberapa outlier ekstrem (mis. error input/sensor). | Tidak ada syarat distribusi tertentu |

**Rule of thumb:** cek `abs(skew)` dulu — kalau `> 1`, log transform (`log1p`) biasanya sudah cukup efektif untuk kebanyakan kasus.
