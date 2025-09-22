# Irodori Preprocessing Module API Reference

Module containing functions for preprocessing hyperspectral (Irodori) data, including normalization, smoothing, denoising, outlier detection, spectral transformations, and baseline correction. All functions operate on `HyperTable` objects.

---

## 1. `_rebuild_hypertable`

**Signature:**
```python
_rebuild_hypertable(
    hyper_table,
    new_df: pd.DataFrame,
    metadata_updates: Optional[dict] = None,
    filtered_indices: Optional[np.ndarray] = None,
    wavelengths: Optional[np.ndarray] = None,
) -> HyperTable
```
**Description:**
Build a new HyperTable from a filtered DataFrame. Ensures labels and wavelengths are properly set, and metadata is updated.
**Parameters:**

hyper_table (HyperTable): Original HyperTable.
new_df (pd.DataFrame): Filtered data.
metadata_updates (dict, optional): Metadata updates.
filtered_indices (np.ndarray, optional): Indices of rows from original HyperTable.
wavelengths (np.ndarray, optional): Explicit wavelengths for new HyperTable.
**Returns:**
HyperTable — New HyperTable object.
**Use Case:**
Internal helper for rebuilding HyperTable after preprocessing steps.

---

**2. minmax_scale**
**Signature:**
minmax_scale(hyper_table: HyperTable, feature_range=(0, 1), axis: int = 0) -> HyperTable
**Description:**
Applies Min–Max scaling to spectral data.
**Parameters:**

hyper_table (HyperTable): Input HyperTable.
feature_range (tuple, default=(0,1)): Desired output range.
axis (int, default=0): 0 = per-band, 1 = per-sample.
**Returns:**
HyperTable — Scaled data.
**Use Case:**
Normalize spectra to a fixed range, useful before PCA or machine learning.

---

**3. standardize**
**Signature:**
standardize(hyper_table: HyperTable, axis: int = 0) -> HyperTable
Description:
Applies Z-score standardization.
**Parameters:**

hyper_table (HyperTable): Input data.
axis (int): 0 = per-band, 1 = per-sample.
**Returns:**
HyperTable — Standardized spectra.
**Use Case:**
Removes scale differences, commonly used before clustering or classification.

---

**4. vector_normalize**
**Signature:**
vector_normalize(hyper_table: HyperTable) -> HyperTable
**Description:**
Normalize each spectrum to unit Euclidean norm (L2 normalization).
**Returns:**
HyperTable — Row-normalized spectra.

**Use Case:**
Used when magnitude differences are not important, e.g., spectral similarity calculations.

---

**5. apply_savgol_filter**
**Signature:**
apply_savgol_filter(
    hyper_table: HyperTable,
    window_length: int = 11,
    polyorder: int = 2,
    deriv: int = 0,
    axis: int = 1
) -> HyperTable
**Description:**
Apply Savitzky–Golay smoothing or derivative filtering.
**Parameters:**

window_length: Odd integer window size.
polyorder: Polynomial order.
deriv: Derivative order (0 = smoothing).
axis: 1 = across bands, 0 = across samples.
**Returns:**
HyperTable — Smoothed or derivative spectra.
**Use Case:**
Remove high-frequency noise or compute spectral derivatives.

---

**6. band_average**
**Signature:**
band_average(hyper_table: HyperTable, window_size: int = 3) -> HyperTable
**Description:**
Reduce noise by averaging adjacent bands.
**Parameters:**

window_size: Number of adjacent bands to average.
**Returns:**
HyperTable — Smoothed spectra with fewer bands.
**Use Case:**
Denoising spectra while reducing dimensionality.

---

**7. pca_denoise**
**Signature:**
pca_denoise(hyper_table: HyperTable, n_components: int) -> HyperTable
Description:
Apply PCA to denoise and reconstruct spectra.
**Parameters:**

n_components: Number of principal components to retain.
**Returns:**
HyperTable — Reconstructed, denoised spectra.
**Use Case:**
Remove noise while retaining major spectral features.

---

**8. remove_noisy_bands**
**Signature:**
remove_noisy_bands(
    hyper_table: HyperTable,
    wavelength_range: tuple = None,
    variance_threshold: float = None
) -> HyperTable
**Description:**
Remove low-variance or out-of-range spectral bands.
Returns:
HyperTable — Filtered dataset.

**Use Case:**
Preprocessing to keep only informative bands.

---

**9. select_wavelength_range**
**Signature:**
select_wavelength_range(hyper_table: HyperTable, ranges: list[tuple[float, float]]) -> HyperTable
**Description:**
Select specific wavelength ranges.
**Returns:**
HyperTable — Filtered bands.

**Use Case:**
Focus analysis on wavelength ranges of interest.

---

**10. mahalanobis_distance**
**Signature:**
mahalanobis_distance(hyper_table) -> np.ndarray
**Description:**
Compute Mahalanobis distance of each sample.
**Returns:**
np.ndarray — Distance per sample.

**Use Case:**
Outlier detection in multivariate spectral space.

---

**11. isolation_forest_filter**
**Signature:**
isolation_forest_filter(
    hyper_table: HyperTable,
    contamination: float = 0.05,
    random_state: int = 42,
    return_mask: bool = False
) -> HyperTable or tuple(HyperTable, np.ndarray)
**Description:**
Detect and remove outliers using Isolation Forest.
**Returns:**
Filtered HyperTable; optionally with mask.

**Use Case:**
Automatic removal of anomalous spectra before analysis.

---

**12. correct_baseline**
**Signature:**
correct_baseline(ht: HyperTable, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> HyperTable
**Description:**
Apply baseline correction to spectra using ALS method.
**Use Case:**
Correct background effects in hyperspectral data.

---

**13. normalize_vector**
**Signature:**
normalize_vector(ht: HyperTable) -> HyperTable
**Description:**
Normalize each spectrum to unit vector (L2 norm = 1).
**Use Case:**
Useful when comparing spectral shapes independent of intensity.

---

**14. spectral_shift**
**Signature:**
spectral_shift(ht: HyperTable, shift: float) -> HyperTable
**Description:**
Shift spectra along bands (supports interpolation).
**Use Case:**
Simulate spectral misalignment or correct for small shifts.

---

**15. mixup**
**Signature:**
mixup(ht: HyperTable, alpha: float = 0.4, n_samples: int = None) -> HyperTable
**Description:**
Mixup augmentation: generate synthetic spectra by linear combination.
**Use Case:**
Data augmentation for training machine learning models.

---

**16. spectral_derivative**
**Signature:**
spectral_derivative(ht: HyperTable, order: int = 1, window_length: int = 11, polyorder: int = 2) -> HyperTable
**Description:**
Compute first or second derivative spectra using Savitzky–Golay filter.
**Use Case:**
Highlight spectral features and remove baseline drift.

---

**17. add_noise**
**Signature:**
add_noise(ht: HyperTable, noise_level: float = 0.01) -> HyperTable
**Description:**
Add Gaussian noise to spectra.
**Use Case:**
Augmentation or simulation of sensor noise.

---

**18. spectral_index**
**Signature:**
spectral_index(ht: HyperTable, band1: int, band2: int, index_name: str = None) -> HyperTable
**Description:**
Compute normalized difference index between two bands.
**Use Case:**
Calculate NDVI-like or custom spectral indices.

---

**19. resample_spectra**
**Signature:**
resample_spectra(hyper_table: HyperTable, new_wavelengths: np.ndarray) -> HyperTable
**Description:**
Resample spectra to new wavelengths using interpolation.
**Use Case:**
Align spectra from different sensors or resolutions.

---

**20. estimate_snr**
**Signature:**
estimate_snr(ht: HyperTable, band_range: tuple = None) -> np.ndarray
**Description:**
Estimate SNR per sample as mean/std within band range.
**Use Case:**
Quality assessment of hyperspectral data.

---

**21. multiplicative_scatter_correction**
**Signature:**
multiplicative_scatter_correction(ht: HyperTable) -> HyperTable
**Description:**
Apply MSC to reduce scatter effects.
**Use Case:**
Correct multiplicative and additive effects in spectra.

---

**22. standard_normal_variate**
**Signature:**
standard_normal_variate(ht: HyperTable) -> HyperTable
**Description:**
Apply SNV normalization to spectra.
**Use Case:**
Remove scaling effects per spectrum for chemometric analysis.

---

**23. savgol_first_derivative**
**Signature:**
savgol_first_derivative(ht: HyperTable, window_length: int = 11, polyorder: int = 2) -> HyperTable
**Description:**
Compute first derivative using Savitzky–Golay filter.
**Use Case:**
Enhance spectral features, remove baseline drift.

---

**24. savgol_second_derivative**
**Signature:**
savgol_second_derivative(ht: HyperTable, window_length: int = 11, polyorder: int = 2) -> HyperTable
**Description:**
Compute second derivative using Savitzky–Golay filter.
**Use Case:**
Highlight subtle spectral features for analysis.

---

**25. baseline_als**
**Signature:**
baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray
**Description:**
Compute baseline using Asymmetric Least Squares.
**Use Case:**
Correct spectral baseline in individual spectra.

---

**26. apply_baseline_correction**
**Signature:**
apply_baseline_correction(ht: HyperTable, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> HyperTable
**Description:**
Apply ALS baseline correction to all spectra.
**Use Case:**
Remove baseline drift in hyperspectral datasets.

---

**27. resample_wavelengths**
**Signature:**
resample_wavelengths(ht: HyperTable, new_wavelengths: np.ndarray) -> HyperTable
**Description:**
Resample hyperspectral data to new wavelengths.
**Use Case:**
Interpolate data to match specific wavelength grids.

---

**28. remove_outliers_zscore**
**Signature:**
remove_outliers_zscore(ht: HyperTable, threshold: float = 3.0) -> HyperTable
**Description:**
Remove samples with any band having Z-score above threshold.
**Use Case:**
Clean datasets by removing extreme spectral outliers.
