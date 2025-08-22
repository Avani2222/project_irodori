import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.ensemble import IsolationForest


# ==============================
# Helper to rebuild HyperTable
# ==============================
def _rebuild_hypertable(hyper_table: "HyperTable", new_data: pd.DataFrame, new_metadata: dict = None) -> "HyperTable":
    """
    Rebuild a HyperTable by combining labels with transformed band data.

    Parameters
    ----------
    hyper_table : HyperTable
        Original HyperTable object.
    new_data : pd.DataFrame
        Transformed spectral data (bands only, without labels).
    new_metadata : dict, optional
        Extra metadata to merge into the new HyperTable.

    Returns
    -------
    HyperTable
        A new HyperTable with labels reattached, wavelengths preserved,
        and metadata updated.
    """
    df = pd.concat(
        [pd.Series(hyper_table.labels, name="Label").reset_index(drop=True),
         new_data.reset_index(drop=True)],
        axis=1
    )
    return HyperTable(
        df,
        wavelengths=hyper_table.wavelengths,
        metadata={**hyper_table.metadata, **(new_metadata or {})}
    )


# ==============================
# Normalization / Standardization
# ==============================
def minmax_scale(hyper_table: "HyperTable", feature_range=(0, 1), axis: int = 0) -> "HyperTable":
    """
    Apply Min–Max scaling to spectral data in a HyperTable.

    Parameters
    ----------
    hyper_table : HyperTable
        Input HyperTable object.
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    axis : int, default=0
        Axis along which to scale:
        - 0 → Per-band (column-wise, across samples).
        - 1 → Per-sample (row-wise, across bands).

    Returns
    -------
    HyperTable
        New HyperTable with scaled data.
    """
    min_val, max_val = feature_range

    if axis == 0:
        data_min = hyper_table.data.min(axis=0)
        data_max = hyper_table.data.max(axis=0)
        scaled_data = (hyper_table.data - data_min) / (data_max - data_min).replace(0, 1)
    elif axis == 1:
        data_min = hyper_table.data.min(axis=1)
        data_max = hyper_table.data.max(axis=1)
        scaled_data = (hyper_table.data.T - data_min).T / (data_max - data_min).replace(0, 1)
    else:
        raise ValueError("axis must be 0 (per band) or 1 (per sample).")

    scaled_data = scaled_data * (max_val - min_val) + min_val
    return _rebuild_hypertable(hyper_table, scaled_data, {"scaling": "minmax"})


def standardize(hyper_table: "HyperTable", axis: int = 0) -> "HyperTable":
    """
    Apply Z-score standardization to spectral data in a HyperTable.

    Parameters
    ----------
    hyper_table : HyperTable
        Input HyperTable object.
    axis : int, default=0
        Axis along which to standardize:
        - 0 → Per-band (column-wise).
        - 1 → Per-sample (row-wise).

    Returns
    -------
    HyperTable
        New HyperTable with standardized data.
    """
    if axis == 0:
        mean = hyper_table.data.mean(axis=0)
        std = hyper_table.data.std(axis=0).replace(0, 1)
        standardized_data = (hyper_table.data - mean) / std
    elif axis == 1:
        mean = hyper_table.data.mean(axis=1)
        std = hyper_table.data.std(axis=1).replace(0, 1)
        standardized_data = ((hyper_table.data.T - mean).T) / std
    else:
        raise ValueError("axis must be 0 (per band) or 1 (per sample).")

    return _rebuild_hypertable(hyper_table, standardized_data, {"scaling": "zscore"})


def vector_normalize(hyper_table: "HyperTable") -> "HyperTable":
    """
    Apply vector normalization (L2 norm) per spectrum (row).

    Each sample spectrum is scaled so its Euclidean norm equals 1:
        x' = x / ||x||_2

    Parameters
    ----------
    hyper_table : HyperTable
        Input HyperTable object.

    Returns
    -------
    HyperTable
        New HyperTable with row-normalized spectra.
    """
    norms = np.linalg.norm(hyper_table.data.values, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid divide by zero
    normalized = hyper_table.data.values / norms

    normalized_df = pd.DataFrame(normalized, columns=hyper_table.data.columns)
    return _rebuild_hypertable(hyper_table, normalized_df, {"scaling": "vector"})


# ==============================
# Spectral Smoothing / Filtering
# ==============================
def apply_savgol_filter(hyper_table: "HyperTable", window_length: int = 11,
                        polyorder: int = 2, deriv: int = 0, axis: int = 1) -> "HyperTable":
    """
    Apply Savitzky–Golay filter to smooth spectra.

    Parameters
    ----------
    hyper_table : HyperTable
        Input dataset.
    window_length : int, default=11
        Size of the smoothing window (odd integer).
    polyorder : int, default=2
        Polynomial order for fitting.
    deriv : int, default=0
        Derivative order (0 = smoothing).
    axis : int, default=1
        Direction of filter:
        - 1 → across bands (smooth spectra).
        - 0 → across samples.

    Returns
    -------
    HyperTable
        Smoothed HyperTable.
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (samples) or 1 (bands).")

    smoothed_data = savgol_filter(
        hyper_table.data.values,
        window_length=window_length,
        polyorder=polyorder,
        deriv=deriv,
        axis=axis
    )

    smoothed_df = pd.DataFrame(smoothed_data, columns=hyper_table.data.columns)
    return _rebuild_hypertable(hyper_table, smoothed_df, {"filter": f"savgol(window={window_length}, poly={polyorder})"})


def band_average(hyper_table: "HyperTable", window_size: int = 3) -> "HyperTable":
    """
    Apply band averaging (spectral smoothing) to reduce noise.

    Parameters
    ----------
    hyper_table : HyperTable
        Input dataset.
    window_size : int, default=3
        Number of adjacent bands to average.

    Returns
    -------
    HyperTable
        Smoothed HyperTable with fewer bands.
    """
    if window_size < 2:
        raise ValueError("window_size must be >= 2")

    data = hyper_table.data.values
    n_samples, n_bands = data.shape
    n_new_bands = n_bands // window_size

    smoothed_data = np.zeros((n_samples, n_new_bands))
    new_wavelengths = np.zeros(n_new_bands) if hyper_table.wavelengths is not None else None

    for i in range(n_new_bands):
        start, end = i * window_size, (i + 1) * window_size
        smoothed_data[:, i] = data[:, start:end].mean(axis=1)
        if hyper_table.wavelengths is not None:
            new_wavelengths[i] = hyper_table.wavelengths[start:end].mean()

    smoothed_df = pd.DataFrame(smoothed_data, columns=[f"band_{i}" for i in range(n_new_bands)])
    return _rebuild_hypertable(hyper_table, smoothed_df, {"smoothing": f"band_average(window={window_size})"})


# ==============================
# Denoising / Dimensionality Reduction
# ==============================
def pca_denoise(hyper_table: "HyperTable", n_components: int) -> "HyperTable":
    """
    Apply PCA-based denoising and reconstruction.

    Parameters
    ----------
    hyper_table : HyperTable
        Input dataset.
    n_components : int
        Number of principal components to retain.

    Returns
    -------
    HyperTable
        Reconstructed dataset with noise reduced.
    """
    if n_components <= 0 or n_components > hyper_table.bands:
        raise ValueError("n_components must be between 1 and number of bands.")

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(hyper_table.data.values)
    reconstructed = pca.inverse_transform(transformed)

    reconstructed_df = pd.DataFrame(reconstructed, columns=hyper_table.data.columns)
    return _rebuild_hypertable(hyper_table, reconstructed_df, {"denoising": f"PCA (n={n_components})"})


# ==============================
# Band Selection
# ==============================
def remove_noisy_bands(hyper_table: "HyperTable", wavelength_range: tuple = None,
                       variance_threshold: float = None) -> "HyperTable":
    """
    Remove noisy or irrelevant spectral bands.

    Parameters
    ----------
    hyper_table : HyperTable
        Input dataset.
    wavelength_range : tuple (min_wl, max_wl), optional
        Keep only bands in this wavelength range.
    variance_threshold : float, optional
        Remove bands with variance below this value.

    Returns
    -------
    HyperTable
        Dataset with filtered bands.
    """
    data = hyper_table.data.copy()
    wavelengths = hyper_table.wavelengths
    keep_mask = np.ones(data.shape[1], dtype=bool)

    if wavelength_range is not None and wavelengths is not None:
        min_wl, max_wl = wavelength_range
        keep_mask &= (wavelengths >= min_wl) & (wavelengths <= max_wl)

    if variance_threshold is not None:
        band_variances = data.var(axis=0).values
        keep_mask &= band_variances > variance_threshold

    filtered_data = data.iloc[:, keep_mask]
    return _rebuild_hypertable(hyper_table, filtered_data,
                               {"filter": "remove_noisy_bands", "variance_threshold": variance_threshold})


def select_wavelength_range(hyper_table: "HyperTable", ranges: list[tuple[float, float]]) -> "HyperTable":
    """
    Select bands within given wavelength ranges.

    Parameters
    ----------
    hyper_table : HyperTable
        Input dataset.
    ranges : list of (min_wl, max_wl)
        Wavelength ranges to keep.

    Returns
    -------
    HyperTable
        Dataset with only selected bands.
    """
    if hyper_table.wavelengths is None:
        raise ValueError("Wavelengths are not defined in this HyperTable.")

    keep_mask = np.zeros(hyper_table.bands, dtype=bool)
    for min_wl, max_wl in ranges:
        keep_mask |= (hyper_table.wavelengths >= min_wl) & (hyper_table.wavelengths <= max_wl)

    filtered_data = hyper_table.data.iloc[:, keep_mask]
    return _rebuild_hypertable(hyper_table, filtered_data, {"filter": f"selected_ranges={ranges}"})


# ==============================
# Outlier Detection
# ==============================
def mahalanobis_distance(hyper_table: "HyperTable") -> np.ndarray:
    """
    Compute Mahalanobis distance for each sample.

    Parameters
    ----------
    hyper_table : HyperTable
        Input dataset.

    Returns
    -------
    np.ndarray
        Array of distances per sample.
    """
    X = hyper_table.data.values
    mean_vec = np.mean(X, axis=0)
    cov_matrix = np.cov(X, rowvar=False)
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
    return np.array([distance.mahalanobis(x, mean_vec, inv_cov_matrix) for x in X])


def isolation_forest_filter(hyper_table: "HyperTable", contamination: float = 0.05,
                            random_state: int = 42, return_mask: bool = False) -> "HyperTable":
    """
    Apply Isolation Forest to detect and filter outlier samples.

    Parameters
    ----------
    hyper_table : HyperTable
        Input dataset.
    contamination : float, default=0.05
        Proportion of expected outliers.
    random_state : int, default=42
        Random seed.
    return_mask : bool, default=False
        If True, return (HyperTable, mask).

    Returns
    -------
    HyperTable
        Dataset with outliers removed.
    mask : np.ndarray, optional
        Boolean mask of kept samples.
    """
    X = hyper_table.data.values
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    preds = iso.fit_predict(X)
    mask = preds == 1

    filtered_data = hyper_table.data[mask]
    new_ht = _rebuild_hypertable(hyper_table, filtered_data,
                                 {"filter": "isolation_forest", "contamination": contamination})

    return (new_ht, mask) if return_mask else new_ht
