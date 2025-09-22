# Irodori Similarity Module API Reference

This module provides functions for measuring similarity or difference between hyperspectral data stored in a `HyperTable` object. It includes spectral angle mapper, Euclidean distance, spectral information divergence, band ratio, spectral correlation, cosine similarity, and visualization tools.

---

**1. spectral_angle_mapper**  
**Signature:**
```python
spectral_angle_mapper(
    hyper_table: HyperTable,
    reference: np.ndarray,
    visualize: bool = True,
    in_degrees: bool = True,
    figsize: Tuple[int,int] = (8,4)
) -> np.ndarray
```
**Description:**
Compute SAM similarity between each spectrum in a HyperTable and a reference spectrum.
**Parameters:**

hyper_table (HyperTable): Hyperspectral data container.
reference (np.ndarray): Reference spectrum (1D array of length = number of bands).
visualize (bool, default=True): Plot SAM angles for all samples.
in_degrees (bool, default=True): Return angles in degrees or radians.
figsize (tuple, default=(8,4)): Figure size.
**Returns:**
np.ndarray — SAM angles per sample.
**Use Case:**
Compare each spectrum to a reference and identify similar or dissimilar samples.

---

**2. euclidean_distance**
**Signature:**
```python
euclidean_distance(
    hyper_table: HyperTable,
    reference: np.ndarray,
    visualize: bool = True,
    figsize: Tuple[int,int] = (8,4)
) -> np.ndarray
```
**Description:**
Compute Euclidean distance between each spectrum and a reference spectrum.
**Returns:**
np.ndarray — Distance per sample (lower = more similar).

**Use Case:**
Measure absolute spectral difference from a reference.

---

**3. spectral_information_divergence**
**Signature:**
```python
spectral_information_divergence(
    hyper_table: HyperTable,
    reference: np.ndarray,
    visualize: bool = True,
    figsize: Tuple[int,int] = (8,4)
) -> np.ndarray
```
**Description:**
Compute symmetric KL divergence (SID) between spectra and reference.
**Returns:**
np.ndarray — SID per sample (lower = more similar).

**Use Case:**
Outlier detection or similarity assessment based on information theory.

---

**4. band_ratio**
**Signature:**
```python
band_ratio(
    hyper_table: HyperTable,
    band1: int,
    band2: int,
    visualize: bool = True,
    figsize: Tuple[int,int] = (8,4)
) -> np.ndarray
```
**Description:**
Compute the ratio of two spectral bands for each sample.
**Returns:**
np.ndarray — Band ratio per sample.

**Use Case:**
Calculate vegetation indices or other spectral ratios.

---

**5. spectral_correlation**
**Signature:**
```python
spectral_correlation(
    hyper_table: HyperTable,
    reference: np.ndarray,
    visualize: bool = True,
    figsize: Tuple[int,int] = (8,4)
) -> np.ndarray
```
**Description:**
Compute Pearson correlation coefficient between each spectrum and reference.
**Returns:**
np.ndarray — Correlation per sample (closer to 1 = more similar).

**Use Case:**
Assess spectral similarity in terms of linear relationship.

---

**6. cosine_similarity**
**Signature:**
```python
cosine_similarity(
    hyper_table: HyperTable,
    reference: np.ndarray,
    visualize: bool = True,
    figsize: Tuple[int,int] = (8,4)
) -> np.ndarray
```
**Description:**
Compute cosine similarity between spectra and reference.
**Returns:**
np.ndarray — Cosine similarity per sample (1 = identical, -1 = opposite).

**Use Case:**
Compare spectral shapes independently of magnitude.

---

**7. sam_heatmap**
**Signature:**
```python
sam_heatmap(
    image_cube: np.ndarray,
    reference: np.ndarray,
    in_degrees: bool = True,
    figsize: Tuple[int,int] = (6,5)
) -> np.ndarray
```
**Description:**
Compute SAM angles for each pixel in a 3D hyperspectral cube.
**Returns:**
np.ndarray — 2D SAM map of shape (H x W).

**Use Case:**
Visualize spectral similarity across spatial hyperspectral images.

---

**8. similarity_dashboard**
**Signature:**
```python
similarity_dashboard(
    hyper_table: HyperTable,
    reference: np.ndarray,
    figsize: Tuple[int,int] = (12,6)
) -> pd.DataFrame
```
**Description:**
Compute multiple similarity metrics (SAM, Euclidean, Correlation, Cosine) and visualize them.
**Returns:**
pd.DataFrame — DataFrame containing all similarity metrics per sample.

**Use Case:**
Quick comparison of spectra using multiple metrics with visualization.
