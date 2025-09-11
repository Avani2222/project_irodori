# irodori: Hyperspectral Data Analysis Library

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  
[![GitHub Issues](https://img.shields.io/badge/library-irodori-green)](https://github.com/milk0707/ANSWER-Library/edit/master/common/project_irodori)

---

## 📌 Overview
`irodori` is a comprehensive Python library for **Irodori data (HSD) processing**, covering the full workflow from **data loading and preprocessing** to **visualization, analysis, and classification**.  
Developed initially in Jupyter Notebooks, the code has been refactored into **modular, reusable Python scripts** for production-ready usage.

---

## 📌 Installation
Clone the repository and navigate to the `project_irodori` folder:

```bash
git clone https://github.com/Avani2222/project_irodori.git
cd project_irodori
```
---

## 📂 Folder Structure

```plaintext
common/
└── project_irodori/
    ├── docs/                              # Documentation files
    │   ├── README.md
    │   ├── API_REFERENCE/                 # API reference for irodori library
    │       ├── irodori_core.md
    │       ├── irodori_io.md
    │       ├── irodori_analysis.md
    │       ├── irodori_indices.md
    │       ├── irodori_preprocessing.md
    |       ├──irodori_dimensionality_reduction.md
    │       ├── irodori_similarity.md
    │       └── irodori_classification.md
    │
    ├── irodori/                           # Main library source code
    │   ├── __init__.py
    │   ├── analysis.py
    │   ├── classification.py
    │   ├── core.py
    │   ├── indices.py
    │   ├── io.py
    |   ├──dimensionality_reduction.py 
    │   ├── preprocessing.py
    │   └── similarity.py
    │
    ├── tests/                             # Unit tests
    │   ├── __init__.py
    │   ├── test_analysis.py
    │   ├── test_classification.py
    │   ├── test_indices.py
    │   ├── test_io.py
    |   ├── test_dimensionality_reduction.py
    │   ├── test_preprocessing.py
    │   └── test_similarity.py
    │
    ├── .gitignore
    └── pyproject.toml
```
---
# Hyperspectral Analysis Library

## core.py — Core Irodori Data Structure

### Main Features
- Standard container (`HyperTable` class) for storing Irodori data and metadata.
- Validates shape, band count, and wavelength metadata.
- Ensures cross-module compatibility.
- Extract individual spectral bands by wavelength.
- Developer-friendly object summary with `__repr__`.

### Key Class & Methods

**HyperTable**
- `__init__(data, wavelengths=None, metadata=None)` — Initializes hyperspectral cube.
- `shape` (property) — Returns cube dimensions `(rows, cols, bands)`.
- `bands` (property) — Returns number of spectral bands.
- `get_band(wavelength)` — Retrieves 2D image for closest band.
- `__repr__()` — Summary of cube data and metadata.

---

## io.py — Data Input/Output

### Main Features
- Unified interface for loading/saving hyperspectral datasets.
- Supports `.hsd`, `.dat`, `.npy`, and CSV formats.
- Converts datasets into `HyperTable` objects.

### Key Functions
- `load_csv(filepath)` — Loads CSV as `HyperTable`.
- `save_csv(cube, filename)` — Saves `HyperTable` as CSV.
- `_load_npy()`, `_load_hsd()` — Internal parsers for specific formats.

---

## analysis.py — Spectral Analysis

### Main Features
- PCA and ICA for dimensionality reduction and component extraction.
- Spectral heatmaps and loadings for visualization.
- Band ratios, ANOVA F-tests, and peak detection.
- Pixel-level and average spectral plotting.

### Key Functions
- `first_derivative()`, `second_derivative()`, `smooth_spectra()`
- `plot_pixel_spectrum()`, `plot_average_spectrum()`, `plot_band_image()`
- `pca_outlier_detection()`, `cluster_bands()`, `spectral_entropy()`
- `spectral_angle_mapper()`, `spectral_information_divergence()`
- `band_ratio()`, `continuum_removal()`, `anova_f_test()`

---

## preprocessing.py — Preprocessing Tasks

### Main Features
- Denoising, normalization, baseline correction.
- Savitzky–Golay smoothing and spectral derivatives.
- Wavelength selection, mixup, and noise addition.
- Outlier removal and scatter correction.

### Key Functions
- `minmax_scale()`, `standardize()`, `vector_normalize()`
- `apply_savgol_filter()`, `baseline_als()`, `apply_baseline_correction()`
- `spectral_shift()`, `resample_spectra()`, `remove_outliers_zscore()`

---

## indices.py — Vegetation & Water Indices

### Key Functions
- `compute_ndvi()`, `compute_gndvi()`, `compute_savi()`, `compute_evi()`
- `compute_arvi()`, `compute_mndwi()`, `compute_ndwi()`, `compute_ndsi()`
- `compute_custom_index()`

---

## similarity.py — Spectral Similarity Metrics

### Key Functions
- `spectral_angle_mapper()`, `spectral_information_divergence()`
- `euclidean_distance()`, `cosine_similarity()`, `spectral_correlation()`
- `sam_heatmap()`, `similarity_dashboard()`

---

## classification.py — Supervised & Unsupervised Classification

### Key Functions
- `supervised_classification()` — Random Forest, SVM, etc.
- `unsupervised_classification()` — KMeans and other clustering methods.
- Supports pixel-wise or full scene classification.

---

## requirements.txt
