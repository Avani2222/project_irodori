# API Reference – Dimensionality Reduction Module

This document describes the available functions in the dimensionality reduction module for hyperspectral (Irodori) data, including their purpose and use cases.

---

## 1. PCA Transformation

### `pca_transform(hyper_table: HyperTable, n_components: int = 3, visualize: bool = True, figsize: tuple = (7, 5)) -> np.ndarray`

- **Description:** Reduce hyperspectral data dimensionality using Principal Component Analysis (PCA).  
- **Use/Helps:** Helps to compress high-dimensional spectral data into a smaller number of principal components while retaining maximum variance. Useful for visualization, preprocessing before classification, or exploratory analysis.  
- **Returns:** PCA-transformed data `(samples, n_components)`  

---

## 2. ICA Transformation

### `ica_transform(hyper_table: HyperTable, n_components: int = 3, visualize: bool = True, figsize: tuple = (7, 5)) -> np.ndarray`

- **Description:** Perform Independent Component Analysis (ICA) for hyperspectral data.  
- **Use/Helps:** Helps to separate statistically independent sources or features in spectral data, which can reveal underlying factors or pure spectral signatures.  
- **Returns:** ICA-transformed data `(samples, n_components)`  

---

## 3. t-SNE / UMAP Embedding

### `visualize_embedding(hyper_table: HyperTable, method: str = "tsne", n_components: int = 2, perplexity: float = 30, figsize: tuple = (7, 5), random_state: int = 42) -> np.ndarray`

- **Description:** Reduce dimensionality for visualization using t-SNE or UMAP.  
- **Use/Helps:** Helps visualize complex hyperspectral data in 2D or 3D for clustering, anomaly detection, or understanding class separability.  
- **Returns:** Low-dimensional embedding `(samples, n_components)`  

---

## 4. Non-negative Matrix Factorization

### `nmf_decomposition(hyper_table: HyperTable, n_components: int = 5, visualize: bool = True, figsize: tuple = (10,5), random_state: int = 42) -> tuple`

- **Description:** Perform NMF to extract endmembers and abundances.  
- **Use/Helps:** Useful for unmixing hyperspectral data into pure spectral components (endmembers) and their contributions (abundances). Common in material identification and remote sensing.  
- **Returns:** `(W, H)`  
  - `W`: Abundance matrix `(samples, n_components)`  
  - `H`: Endmember spectra `(n_components, bands)`  

---

## 5. Mutual Information

### `compute_mutual_info(hyper_table: HyperTable, y: np.ndarray, task: Literal["classification", "regression"] = "classification", n_neighbors: int = 3, plot: bool = True) -> np.ndarray`

- **Description:** Compute mutual information between bands and target variable.  
- **Use/Helps:** Identifies which spectral bands are most informative for predicting a target, aiding in feature selection and dimensionality reduction for supervised tasks.  
- **Returns:** Mutual information scores per band `(n_bands,)`  

---

## 6. Linear Discriminant Analysis

### `lda_transform(hyper_table: HyperTable, n_components: int = 2, visualize: bool = True, figsize: tuple = (7, 5)) -> np.ndarray`

- **Description:** Apply LDA for dimensionality reduction using class labels.  
- **Use/Helps:** Helps maximize class separability in supervised tasks, making it useful for visualization and preprocessing for classification.  
- **Returns:** LDA-transformed data `(samples, n_components)`  

---

## 7. Kernel PCA

### `kernel_pca_transform(hyper_table: HyperTable, n_components: int = 3, kernel: str = "rbf", gamma: Optional[float] = None, visualize: bool = True, figsize: tuple = (7,5)) -> np.ndarray`

- **Description:** Apply Kernel PCA using specified kernel.  
- **Use/Helps:** Captures non-linear relationships in spectral data that PCA cannot, useful for complex data patterns and visualization.  
- **Returns:** Kernel PCA-transformed data `(samples, n_components)`  

---

## 8. Factor Analysis

### `factor_analysis_transform(hyper_table: HyperTable, n_components: int = 3, visualize: bool = True, figsize: tuple = (7,5)) -> np.ndarray`

- **Description:** Apply Factor Analysis to extract latent factors.  
- **Use/Helps:** Helps discover hidden latent variables that explain the covariance in spectral data, useful for data compression and interpretation.  
- **Returns:** FA-transformed data `(samples, n_components)`  

---

## 9. Isomap

### `isomap_transform(hyper_table: HyperTable, n_components: int = 2, n_neighbors: int = 5, visualize: bool = True, figsize: tuple = (7,5)) -> np.ndarray`

- **Description:** Non-linear dimensionality reduction using Isomap.  
- **Use/Helps:** Preserves geodesic distances in data manifold, useful for visualizing and understanding intrinsic geometry of hyperspectral data.  
- **Returns:** Isomap embedding `(samples, n_components)`  

---

## 10. Truncated SVD

### `svd_transform(hyper_table: HyperTable, n_components: int = 3, visualize: bool = True, figsize: tuple = (7,5)) -> np.ndarray`

- **Description:** Apply Truncated SVD for dimensionality reduction (suitable for sparse/high-dimensional data).  
- **Use/Helps:** Efficiently reduces dimensions in large spectral datasets or sparse representations while retaining most variance.  
- **Returns:** SVD-transformed data `(samples, n_components)`  

---

## 11. Spectral Embedding

### `spectral_embedding_transform(hyper_table: HyperTable, n_components: int = 2, n_neighbors: int = 5, visualize: bool = True, figsize: tuple = (7,5)) -> np.ndarray`

- **Description:** Apply Laplacian Eigenmaps (Spectral Embedding).  
- **Use/Helps:** Maps high-dimensional spectral data to lower dimensions preserving local neighborhood relationships. Useful for visualization and manifold learning.  
- **Returns:** Spectral embedding `(samples, n_components)`  

---

## 12. Multidimensional Scaling

### `mds_transform(hyper_table: HyperTable, n_components: int = 2, visualize: bool = True, figsize: tuple = (7,5)) -> np.ndarray`

- **Description:** Distance-preserving embedding using MDS.  
- **Use/Helps:** Useful for visualizing spectral data while preserving pairwise distances between samples.  
- **Returns:** MDS-transformed data `(samples, n_components)`  

---

## 13. KMeans Clustering

### `kmeans_clustering(hyper_table: HyperTable, n_clusters: int = 5, visualize: bool = True, figsize: tuple = (7,5)) -> np.ndarray`

- **Description:** Cluster samples using KMeans.  
- **Use/Helps:** Groups similar spectral signatures together for segmentation, material identification, or exploratory analysis.  
- **Returns:** Cluster labels `(samples,)`  

---

## 14. Gaussian Mixture Model Clustering

### `gmm_clustering(hyper_table: HyperTable, n_components: int = 5, visualize: bool = True, figsize: tuple = (7,5)) -> np.ndarray`

- **Description:** Cluster samples using GMM.  
- **Use/Helps:** Provides soft clustering (probabilistic) for spectral data, useful when clusters overlap.  
- **Returns:** Cluster labels `(samples,)`  

---

## 15. Variance per Band

### `variance_per_band(hyper_table: HyperTable) -> np.ndarray`

- **Description:** Compute and plot variance for each spectral band.  
- **Use/Helps:** Identifies informative bands with high variability, aiding feature selection and noise analysis.  
- **Returns:** Variance array `(n_bands,)`  

---

## 16. ANOVA F-test

### `anova_f_test(hyper_table: HyperTable, y: np.ndarray, plot: bool = True) -> np.ndarray`

- **Description:** Perform ANOVA F-test for supervised band ranking.  
- **Use/Helps:** Ranks spectral bands by their discriminatory power for classification, useful in feature selection.  
- **Returns:** F-scores per band `(n_bands,)`  

---

## 17. Savitzky–Golay Smoothing

### `smooth_spectra(hyper_table: HyperTable, window_length: int = 11, polyorder: int = 2) -> np.ndarray`

- **Description:** Smooth hyperspectral spectra using Savitzky–Golay filter.  
- **Use/Helps:** Reduces noise while preserving spectral features, helpful before derivative analysis or feature extraction.  
- **Returns:** Smoothed spectra `(samples, bands)`  
