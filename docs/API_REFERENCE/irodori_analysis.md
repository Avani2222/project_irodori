# API Reference – Hyperspectral Analysis Module

This document describes the available functions in the Hyperspectral Analysis module.  
Each function operates on a `HyperTable` object (or its data matrix) unless otherwise specified.  
Use cases are provided to help guide practical applications.

---

## 📊 Spectral Derivatives

### `first_derivative(hyper_table, visualize=False)`
Compute the first derivative of spectral signatures along the wavelength axis.

- **Parameters**
  - `hyper_table (HyperTable)`: Input dataset.
  - `visualize (bool, optional)`: If `True`, plots the derivative spectra. Default: `False`.

- **Returns**
  - `np.ndarray`: First derivative spectra.

- **Use case**  
  Helps identify subtle absorption features and highlight inflection points in vegetation or mineral spectra.

---

### `second_derivative(hyper_table, visualize=False)`
Compute the second derivative of spectral signatures.

- **Parameters**
  - `hyper_table (HyperTable)`: Input dataset.
  - `visualize (bool, optional)`: If `True`, plots the derivative spectra.

- **Returns**
  - `np.ndarray`: Second derivative spectra.

- **Use case**  
  Enhances detection of narrow spectral features, such as pigments in vegetation or mineral-specific absorption bands.

---

## 📈 Visualization

### `plot_spectra(hyper_table, sample_indices=None, title="Spectral Signatures")`
Plot spectral signatures for selected samples.

- **Parameters**
  - `hyper_table (HyperTable)`: Input dataset.
  - `sample_indices (list[int], optional)`: Row indices to plot. If `None`, plots all.
  - `title (str)`: Plot title.

- **Returns**
  - `None`

- **Use case**  
  Quick inspection of spectra to visually compare different classes (e.g., healthy vs. stressed crops).

---

## 📑 Statistical Analysis

### `anova_f_test(hyper_table, visualize=False)`
Perform ANOVA F-test across spectral bands.

- **Parameters**
  - `hyper_table (HyperTable)`: Input dataset.
  - `visualize (bool, optional)`: If `True`, plots F-scores.

- **Returns**
  - `np.ndarray`: F-scores for each band.

- **Use case**  
  Identifies wavelengths that significantly differ across classes (e.g., distinguishing soil types).

---

## 🔎 Spectral Similarity Measures

### `spectral_angle_mapper(sig1, sig2)`
Compute the Spectral Angle Mapper (SAM) similarity between two spectra.

- **Parameters**
  - `sig1, sig2 (np.ndarray)`: Input spectra.

- **Returns**
  - `float`: SAM angle in radians.

- **Use case**  
  Robustly compare spectra under varying illumination conditions.

---

### `spectral_information_divergence(sig1, sig2)`
Compute Spectral Information Divergence (SID).

- **Parameters**
  - `sig1, sig2 (np.ndarray)`: Input spectra.

- **Returns**
  - `float`: SID value.

- **Use case**  
  Measure how different two spectra are, useful in classification and anomaly detection.

---

### `euclidean_distance(sig1, sig2)`
Compute Euclidean distance between two spectra.

- **Parameters**
  - `sig1, sig2 (np.ndarray)`: Input spectra.

- **Returns**
  - `float`: Euclidean distance.

- **Use case**  
  Baseline similarity metric, simple but sensitive to scaling.

---

## 🔢 Band Operations

### `band_ratios(hyper_table, pairs, visualize=False)`
Compute band ratios for given wavelength/band pairs.

- **Parameters**
  - `hyper_table (HyperTable)`: Input dataset.
  - `pairs (list[tuple[int,int]])`: List of `(band1, band2)` pairs.
  - `visualize (bool)`: If `True`, plots histograms of computed ratios.

- **Returns**
  - `pd.DataFrame`: DataFrame of ratio values.

- **Use case**  
  Widely used in vegetation indices (e.g., NDVI = NIR/Red).

---

### `continuum_removal(hyper_table, sample_index=0, visualize=False)`
Perform continuum removal for a single sample.

- **Parameters**
  - `hyper_table (HyperTable)`: Input dataset.
  - `sample_index (int)`: Index of the sample to process.
  - `visualize (bool)`: If `True`, plots original vs. continuum-removed spectra.

- **Returns**
  - `np.ndarray`: Continuum-removed spectrum.

- **Use case**  
  Isolate and compare absorption features independent of baseline effects.

---

## 🧮 Dimensionality Reduction

### `pca_transform(hyper_table, n_components=2, visualize=False)`
Perform PCA on hyperspectral data.

- **Parameters**
  - `hyper_table (HyperTable)`: Input dataset.
  - `n_components (int)`: Number of PCA components.
  - `visualize (bool)`: If `True`, plots PCA scatter plot.

- **Returns**
  - `(np.ndarray, PCA)`: Transformed data and fitted PCA model.

- **Use case**  
  Reduce dimensionality for visualization and noise reduction.

---

### `ica_transform(hyper_table, n_components=2, visualize=False)`
Perform Independent Component Analysis (ICA).

- **Parameters**
  - `hyper_table (HyperTable)`: Input dataset.
  - `n_components (int)`: Number of components.
  - `visualize (bool)`: If `True`, scatter plot of ICA results.

- **Returns**
  - `(np.ndarray, FastICA)`: Transformed data and fitted ICA model.

- **Use case**  
  Separate mixed signals, such as removing background noise.

---

### `nmf_transform(hyper_table, n_components=2, visualize=False)`
Perform Non-Negative Matrix Factorization (NMF).

- **Parameters**
  - `hyper_table (HyperTable)`: Input dataset.
  - `n_components (int)`: Number of components.
  - `visualize (bool)`: If `True`, barplot of component loadings.

- **Returns**
  - `(np.ndarray, NMF)`: Transformed data and fitted NMF model.

- **Use case**  
  Useful for feature extraction where only additive parts are meaningful (e.g., endmember extraction).

---

### `tsne_transform(hyper_table, n_components=2, visualize=False)`
Apply t-SNE embedding for visualization.

- **Parameters**
  - `hyper_table (HyperTable)`: Input dataset.
  - `n_components (int)`: Embedding dimension.
  - `visualize (bool)`: If `True`, scatter plot of t-SNE results.

- **Returns**
  - `np.ndarray`: Transformed embedding.

- **Use case**  
  Nonlinear visualization of high-dimensional data to reveal clusters.

---

### `umap_transform(hyper_table, n_components=2, visualize=False)`
Apply UMAP embedding for visualization.

- **Parameters**
  - `hyper_table (HyperTable)`: Input dataset.
  - `n_components (int)`: Embedding dimension.
  - `visualize (bool)`: If `True`, scatter plot of UMAP results.

- **Returns**
  - `np.ndarray`: Transformed embedding.

- **Use case**  
  Fast, scalable alternative to t-SNE for large hyperspectral datasets.

---

## 📐 Spectral Features

### `spectral_entropy(hyper_table, visualize=False)`
Compute entropy of spectra across bands.

- **Parameters**
  - `hyper_table (HyperTable)`: Input dataset.
  - `visualize (bool)`: If `True`, plots entropy distribution.

- **Returns**
  - `np.ndarray`: Entropy values.

- **Use case**  
  Detects variability/uncertainty in spectral signatures, useful in anomaly detection.

---

### `spectral_snr(hyper_table, visualize=False)`
Compute Signal-to-Noise Ratio (SNR) per band.

- **Parameters**
  - `hyper_table (HyperTable)`: Input dataset.
  - `visualize (bool)`: If `True`, plots SNR profile.

- **Returns**
  - `np.ndarray`: SNR values.

- **Use case**  
  Identify low-quality bands to discard in preprocessing.

---

### `spectral_peaks(hyper_table, sample_index=0, visualize=False)`
Find local maxima (peaks) in a spectrum.

- **Parameters**
  - `hyper_table (HyperTable)`: Input dataset.
  - `sample_index (int)`: Sample index.
  - `visualize (bool)`: If `True`, plots peaks on spectrum.

- **Returns**
  - `list[int]`: Indices of detected peaks.

- **Use case**  
  Pinpoint specific absorption/emission features (e.g., chlorophyll peaks).

---

## 🧹 Preprocessing

### `smooth_spectra(hyper_table, window_length=7, polyorder=3, visualize=False)`
Apply Savitzky–Golay filter to smooth spectra.

- **Parameters**
  - `hyper_table (HyperTable)`: Input dataset.
  - `window_length (int)`: Window size.
  - `polyorder (int)`: Polynomial order.
  - `visualize (bool)`: If `True`, plots smoothed spectra.

- **Returns**
  - `np.ndarray`: Smoothed spectra.

- **Use case**  
  Reduce noise while preserving absorption features.

---

### `pca_outlier_detection(hyper_table, n_components=2, visualize=False)`
Detect outliers in PCA space.

- **Parameters**
  - `hyper_table (HyperTable)`: Input dataset.
  - `n_components (int)`: Number of PCA components.
  - `visualize (bool)`: If `True`, scatter plot with outliers highlighted.

- **Returns**
  - `np.ndarray`: Boolean mask of outliers.

- **Use case**  
  Detect faulty or rare spectral measurements.

---

## 📉 Band Clustering

### `cluster_bands(hyper_table, n_clusters=5, visualize=False)`
Cluster bands into groups based on correlation.

- **Parameters**
  - `hyper_table (HyperTable)`: Input dataset.
  - `n_clusters (int)`: Number of clusters.
  - `visualize (bool)`: If `True`, plots band-cluster assignments.

- **Returns**
  - `np.ndarray`: Cluster labels for each band.

- **Use case**  
  Reduce redundancy in bands and select representative wavelengths for modeling.

---

# Notes
- Most functions accept a `visualize` flag for quick exploratory plots.
- Returned `np.ndarray` objects align with the shape of `hyper_table.spectra`.
