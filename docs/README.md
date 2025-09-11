# Irodori: Irodori Data Analysis Library

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  
[![GitHub Issues](https://img.shields.io/badge/library-irodori-green)](https://github.com/Avani2222/project_irodori)

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
- **HyperTable**: Main container for hyperspectral/Irodori data with metadata validation and band access.
  - `__init__(data, wavelengths=None, metadata=None)` — Initialize hyperspectral cube.
  - `shape` — Returns cube dimensions `(rows, cols, bands)`.
  - `bands` — Returns number of spectral bands.
  - `get_band(wavelength)` — Retrieve 2D image for the closest wavelength.
  - `__repr__()` — Summary of cube data and metadata.

---

## io.py — Data Input/Output
- Unified interface for loading/saving Irodori datasets, converting to `HyperTable`.
  - `load_csv(filepath)` — Load CSV as HyperTable.
  - `save_csv(cube, filename)` — Save HyperTable as CSV.
  - `_load_npy()`, `_load_hsd()` — Internal parsers for `.npy` and `.hsd` formats.

---

## analysis.py — Spectral Analysis
- Tools for derivatives, PCA/ICA, band statistics, spectral plots, and similarity.
  - `first_derivative()`, `second_derivative()`, `smooth_spectra()` — Compute derivatives and smooth spectra.
  - `plot_pixel_spectrum()`, `plot_average_spectrum()`, `plot_band_image()`, `plot_band_histograms()`, `plot_spectral_signatures()`, `plot_pca()` — Visualize spectra and PCA/ICA results.
  - `pca_outlier_detection()` — Detect spectral outliers via PCA.
  - `cluster_bands()` — Cluster spectral bands.
  - `spectral_entropy()` — Compute entropy per spectrum.
  - `spectral_angle_mapper()`, `spectral_information_divergence()` — Measure spectral similarity.
  - `band_ratio()` — Compute ratio between spectral bands.
  - `continuum_removal()` — Normalize spectra continuum.
  - `anova_f_test()` — Perform F-test for band significance.

---

## preprocessing.py — Preprocessing Tasks
- Functions for denoising, normalization, baseline correction, scatter correction, outlier removal.
  - `minmax_scale()`, `standardize()`, `vector_normalize()` — Scale/normalize spectra.
  - `apply_savgol_filter()` — Smooth spectra with Savitzky–Golay filter.
  - `baseline_als()`, `apply_baseline_correction()` — Correct spectral baseline.
  - `spectral_shift()` — Shift spectra along wavelength.
  - `resample_spectra()` — Resample spectra to new wavelengths.
  - `remove_outliers_zscore()` — Remove outlier pixels based on Z-score.
  - `_rebuild_hypertable()` — Internal HyperTable reconstruction.
  - `pca_denoise()`, `remove_noisy_bands()` — Denoise via PCA or remove noisy bands.
  - `select_wavelength_range()` — Select subset of wavelengths.
  - `mahalanobis_distance()`, `isolation_forest_filter()` — Outlier filtering.
  - `normalize_vector()` — Vector normalization.
  - `mixup()` — Data augmentation by mixing spectra.
  - `spectral_derivative()`, `savgol_first_derivative()`, `savgol_second_derivative()` — Derivative calculations.
  - `add_noise()` — Add synthetic noise to spectra.
  - `spectral_index()` — Compute indices from spectra.
  - `estimate_snr()` — Estimate signal-to-noise ratio.
  - `multiplicative_scatter_correction()`, `standard_normal_variate()` — Scatter correction methods.
  - `resample_wavelengths()` — Resample spectra to uniform wavelength grid.

---

## indices.py — Vegetation & Water Indices
- Compute vegetation and water indices for Irodori data.
  - `compute_ndvi()`, `compute_gndvi()`, `compute_savi()`, `compute_evi()` — Vegetation indices.
  - `compute_arvi()`, `compute_mndwi()`, `compute_ndwi()`, `compute_ndsi()` — Water/soil indices.
  - `compute_custom_index()` — Compute user-defined index.

---

## similarity.py — Spectral Similarity Metrics
- Compute and visualize spectral similarity.
  - `spectral_angle_mapper()`, `spectral_information_divergence()` — SAM/SID similarity measures.
  - `euclidean_distance()` — Euclidean distance between spectra.
  - `cosine_similarity()` — Cosine similarity between spectra.
  - `spectral_correlation()` — Pearson correlation between spectra.
  - `band_ratio()` — Band ratio similarity metric.
  - `sam_heatmap()`, `similarity_dashboard()` — Visualization tools.

---

## classification.py — Supervised & Unsupervised Classification
- Functions for training, evaluating, and deploying classifiers.
  - `supervised_classification()`, `unsupervised_classification()` — Train models or cluster data.
  - `split_data()`, `scale_data()`, `apply_pca()` — Prepare data for modeling.
  - `train_classifier()`, `cross_validate_classifier()`, `full_classification_pipeline()` — Train and validate models.
  - `evaluate_classifier()`, `plot_feature_importance()`, `plot_precision_recall()`, `classwise_metrics()` — Evaluate and visualize performance.
  - `top_k_accuracy()`, `plot_calibration_curve()` — Performance metrics.
  - `plot_tsne()`, `plot_umap()` — Visualize high-dimensional embeddings.
  - `permutation_importance_plot()` — Feature importance analysis.
  - `build_voting_classifier()`, `build_stacking_classifier()` — Ensemble methods.
  - `one_vs_rest_classifier()`, `one_vs_one_classifier()` — Multi-class strategies.
  - `compare_classifiers()` — Compare multiple classifiers.
  - `optimize_threshold()` — Adjust probability thresholds.
  - `apply_smote()` — Oversample minority classes.
  - `batch_predict_and_report()` — Predict and report for multiple datasets.

---

## dimensionality_reduction.py — Dimensionality Reduction & Clustering
- Transform, embed, and cluster Irodori data.
  - `pca_transform()`, `ica_transform()`, `lda_transform()`, `kernel_pca_transform()`, `factor_analysis_transform()` — Linear/non-linear transforms.
  - `nmf_decomposition()`, `svd_transform()` — Matrix decomposition methods.
  - `isomap_transform()`, `spectral_embedding_transform()`, `mds_transform()` — Manifold learning techniques.
  - `visualize_embedding()` — Visualize low-dimensional embeddings.
  - `kmeans_clustering()`, `gmm_clustering()` — Cluster transformed data.
  - `compute_mutual_info()`, `variance_per_band()` — Feature importance metrics.
  - `dr_anova_f_test()`, `dr_smooth_spectra()` — Dimensionality reduction specific F-test and smoothing.

---

## requirements.txt
- `numpy>=1.21`, `pandas>=1.3`, `matplotlib>=3.4`, `seaborn>=0.11`, `scipy>=1.7`, `scikit-learn>=0.24`, `opencv-python>=4.5`, `pywavelets>=1.1`

---
## Usage

### Basic Usage Example

```python
from irodori.core import HyperTable
from irodori.classification import supervised_classification, unsupervised_classification
import numpy as np

# Fake hyperspectral cube (50x50 pixels, 100 bands)
data = np.random.rand(50, 50, 100)
wavelengths = np.linspace(400, 1000, 100)
cube = HyperTable(data, wavelengths=wavelengths)

# Labels for supervised classification
labels = np.random.randint(0, 3, size=(50, 50))

# Run supervised classification
result_sup = supervised_classification(cube, labels, model='RandomForest')
print("Supervised classification shape:", result_sup.shape)

# Run unsupervised classification (KMeans)
result_unsup = unsupervised_classification(cube, n_clusters=4)
print("Unsupervised classification shape:", result_unsup.shape)
```
## Features

- **End-to-End Workflow** — Load, preprocess, analyze, and classify Irodori data.  
- **Multi-Camera Support** — Supports other devices similar to irodori device.  
- **Standardized Data Structure** — `HyperTable` ensures compatibility across modules.  
- **Preprocessing** — Denoising, calibration, wavelength correction, derivatives, continuum removal.  
- **Spectral Analysis** — PCA, ICA, ANOVA, similarity metrics, band ratios.  
- **Indices** — NDVI, GNDVI, SAVI, PRI, NDWI, MSI, Chlorophyll Index, ARI.  
- **Classification** — Supervised (RF, SVM) and unsupervised (KMeans, GMM) pipelines.  
- **Visualization** — Spectral plots, PCA scatter plots, heatmaps, RGB composites.  
- **Modular Design** — Each function organized in separate modules for maintainability.  

## Notes

- **Dependencies** — Install all packages listed in `requirements.txt`.  
- **Data Handling** — Functions expect `HyperTable` objects; raw arrays may need conversion.  
- **Performance** — Large HSD datasets are memory-intensive; downsample if needed.  
- **Extensibility** — Easy to add new preprocessing, analysis, or classification functions.  
- **License** — Check repository license before commercial use.  
- **Contributions** — Pull requests are welcome; follow contribution guidelines.
  
## License

This library is provided for educational and research purposes. 
