# API Reference – Dimensionality Reduction Module

This document describes the available functions in the dimensionality reduction module for hyperspectral (Irodori) data.

---

## 1. PCA Transformation

### `pca_transform(hyper_table: HyperTable, n_components: int = 3, visualize: bool = True, figsize: tuple = (7, 5)) -> np.ndarray`

- **Description:** Reduce hyperspectral data dimensionality using Principal Component Analysis (PCA).  
- **Parameters:**  
  - `hyper_table`: `HyperTable` object containing spectral data.  
  - `n_components`: Number of principal components to retain (default 3).  
  - `visualize`: Show explained variance and scatter plot (default True).  
  - `figsize`: Plot figure size (default (7,5)).  
- **Returns:** PCA-transformed data `(samples, n_components)`  

---

## 2. ICA Transformation

### `ica_transform(hyper_table: HyperTable, n_components: int = 3, visualize: bool = True, figsize: tuple = (7, 5)) -> np.ndarray`

- **Description:** Perform Independent Component Analysis (ICA) for hyperspectral data.  
- **Returns:** ICA-transformed data `(samples, n_components)`  

---

## 3. t-SNE / UMAP Embedding

### `visualize_embedding(hyper_table: HyperTable, method: str = "tsne", n_components: int = 2, perplexity: float = 30, figsize: tuple = (7, 5), random_state: int = 42) -> np.ndarray`

- **Description:** Reduce dimensionality for visualization using t-SNE or UMAP.  
- **Returns:** Low-dimensional embedding `(samples, n_components)`  

---

## 4. Non-negative Matrix Factorization

### `nmf_decomposition(hyper_table: HyperTable, n_components: int = 5, visualize: bool = True, figsize: tuple = (10,5), random_state: int = 42) -> tuple`

- **Description:** Perform NMF to extract endmembers and abundances.  
- **Returns:** `(W, H)`  
  - `W`: Abundance matrix `(samples, n_components)`  
  - `H`: Endmember spectra `(n_components, bands)`  

---

## 5. Mutual Information

### `compute_mutual_info(hyper_table: HyperTable, y: np.ndarray, task: Literal["classification", "regression"] = "classification", n_neighbors: int = 3, plot: bool = True) -> np.ndarray`

- **Description:** Compute mutual information between bands and target variable.  
- **Returns:** Mutual information scores per band `(n_bands,)`  

---

## 6. Linear Discriminant Analysis

### `lda_transform(hyper_table: HyperTable, n_components: int = 2, visualize: bool = True, figsize: tuple = (7, 5)) -> np.ndarray`

- **Description:** Apply LDA for dimensionality reduction using class labels.  
- **Returns:** LDA-transformed data `(samples, n_components)`  

---

## 7. Kernel PCA

### `kernel_pca_transform(hyper_table: HyperTable, n_components: int = 3, kernel: str = "rbf", gamma: Optional[float] = None, visualize: bool = True, figsize: tuple = (7,5)) -> np.ndarray`

- **Description:** Apply Kernel PCA using specified kernel.  
- **Returns:** Kernel PCA-transformed data `(samples, n_components)`  

---

## 8. Factor Analysis

### `factor_analysis_transform(hyper_table: HyperTable, n_components: int = 3, visualize: bool = True, figsize: tuple = (7,5)) -> np.ndarray`

- **Description:** Apply Factor Analysis to extract latent factors.  
- **Returns:** FA-transformed data `(samples, n_components)`  

---

## 9. Isomap

### `isomap_transform(hyper_table: HyperTable, n_components: int = 2, n_neighbors: int = 5, visualize: bool = True, figsize: tuple = (7,5)) -> np.ndarray`

- **Description:** Non-linear dimensionality reduction using Isomap.  
- **Returns:** Isomap embedding `(samples, n_components)`  

---

## 10. Truncated SVD

### `svd_transform(hyper_table: HyperTable, n_components: int = 3, visualize: bool = True, figsize: tuple = (7,5)) -> np.ndarray`

- **Description:** Apply Truncated SVD for dimensionality reduction (suitable for sparse/high-dimensional data).  
- **Returns:** SVD-transformed data `(samples, n_components)`  

---

## 11. Spectral Embedding

### `spectral_embedding_transform(hyper_table: HyperTable, n_components: int = 2, n_neighbors: int = 5, visualize: bool = True, figsize: tuple = (7,5)) -> np.ndarray`

- **Description:** Apply Laplacian Eigenmaps (Spectral Embedding).  
- **Returns:** Spectral embedding `(samples, n_components)`  

---

## 12. Multidimensional Scaling

### `mds_transform(hyper_table: HyperTable, n_components: int = 2, visualize: bool = True, figsize: tuple = (7,5)) -> np.ndarray`

- **Description:** Distance-preserving embedding using MDS.  
- **Returns:** MDS-transformed data `(samples, n_components)`  

---

## 13. KMeans Clustering

### `kmeans_clustering(hyper_table: HyperTable, n_clusters: int = 5, visualize: bool = True, figsize: tuple = (7,5)) -> np.ndarray`

- **Description:** Cluster samples using KMeans.  
- **Returns:** Cluster labels `(samples,)`  

---

## 14. Gaussian Mixture Model Clustering

### `gmm_clustering(hyper_table: HyperTable, n_components: int = 5, visualize: bool = True, figsize: tuple = (7,5)) -> np.ndarray`

- **Description:** Cluster samples using GMM.  
- **Returns:** Cluster labels `(samples,)`  

---

## 15. Variance per Band

### `variance_per_band(hyper_table: HyperTable) -> np.ndarray`

- **Description:** Compute and plot variance for each spectral band.  
- **Returns:** Variance array `(n_bands,)`  

---

## 16. ANOVA F-test

### `anova_f_test(hyper_table: HyperTable, y: np.ndarray, plot: bool = True) -> np.ndarray`

- **Description:** Perform ANOVA F-test for supervised band ranking.  
- **Returns:** F-scores per band `(n_bands,)`  

---

## 17. Savitzky–Golay Smoothing

### `smooth_spectra(hyper_table: HyperTable, window_length: int = 11, polyorder: int = 2) -> np.ndarray`

- **Description:** Smooth hyperspectral spectra using Savitzky–Golay filter.  
- **Returns:** Smoothed spectra `(samples, bands)`  
