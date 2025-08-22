import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.ensemble import IsolationForest

def minmax_scale(hyper_table: "HyperTable", feature_range=(0, 1), axis: int = 0) -> "HyperTable":
    """
    Apply Min–Max scaling to a HyperTable object.

    Parameters
    ----------
    hyper_table : HyperTable
        Input HyperTable object.
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    axis : int, default=0
        Axis along which to scale:
        - 0 → Column-wise scaling (per band, across all samples).
        - 1 → Row-wise scaling (per sample, across all bands).

    Returns
    -------
    HyperTable
        New HyperTable object with scaled data.
    """
    min_val, max_val = feature_range

    if axis == 0:
        # Per-band scaling
        data_min = hyper_table.data.min(axis=0)
        data_max = hyper_table.data.max(axis=0)
        scaled_data = (hyper_table.data - data_min) / (data_max - data_min).replace(0, 1)

    elif axis == 1:
        # Per-sample scaling
        data_min = hyper_table.data.min(axis=1)
        data_max = hyper_table.data.max(axis=1)
        scaled_data = (hyper_table.data.T - data_min).T / (data_max - data_min).replace(0, 1)

    else:
        raise ValueError("axis must be 0 (per band) or 1 (per sample).")

    # Rescale to feature_range
    scaled_data = scaled_data * (max_val - min_val) + min_val

    return HyperTable(
        scaled_data,
        wavelengths=hyper_table.wavelengths,
        metadata=hyper_table.metadata.copy()
    )

def standardize(hyper_table: "HyperTable", axis: int = 0) -> "HyperTable":
    """
    Apply Z-score standardization to a HyperTable object.

    Parameters
    ----------
    hyper_table : HyperTable
        Input HyperTable object.
    axis : int, default=0
        Axis along which to standardize:
        - 0 → Column-wise (per band, across all samples).
        - 1 → Row-wise (per sample, across all bands).

    Returns
    -------
    HyperTable
        New HyperTable object with standardized data.
    """
    if axis == 0:
        # Per-band standardization
        mean = hyper_table.data.mean(axis=0)
        std = hyper_table.data.std(axis=0).replace(0, 1)  # avoid div by zero
        standardized_data = (hyper_table.data - mean) / std

    elif axis == 1:
        # Per-sample standardization
        mean = hyper_table.data.mean(axis=1)
        std = hyper_table.data.std(axis=1).replace(0, 1)
        standardized_data = ((hyper_table.data.T - mean).T) / std

    else:
        raise ValueError("axis must be 0 (per band) or 1 (per sample).")

    return HyperTable(
        standardized_data,
        wavelengths=hyper_table.wavelengths,
        metadata=hyper_table.metadata.copy()
    )

def apply_savgol_filter(
    hyper_table: "HyperTable",
    window_length: int = 11,
    polyorder: int = 2,
    deriv: int = 0,
    axis: int = 1
) -> "HyperTable":
    """
    Apply Savitzky–Golay filter to hyperspectral data in a HyperTable object.

    Parameters
    ----------
    hyper_table : HyperTable
        Input HyperTable object.
    window_length : int, default=11
        Length of the filter window (number of coefficients).
        Must be a positive odd integer.
    polyorder : int, default=2
        Polynomial order to fit. Must be less than `window_length`.
    deriv : int, default=0
        Order of derivative to compute. Default 0 means smoothing.
    axis : int, default=1
        Axis to apply filter:
        - 1 → across spectral bands (smoothing spectra of each sample row)
        - 0 → across samples (smoothing each band across rows)

    Returns
    -------
    HyperTable
        New HyperTable object with smoothed (or derivative) data.
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (samples) or 1 (bands).")

    # Apply filter
    smoothed_data = savgol_filter(
        hyper_table.data.values,
        window_length=window_length,
        polyorder=polyorder,
        deriv=deriv,
        axis=axis
    )

    # Wrap back into HyperTable
    return HyperTable(
        pd.DataFrame(smoothed_data, columns=hyper_table.data.columns),
        wavelengths=hyper_table.wavelengths,
        metadata=hyper_table.metadata.copy()
    )

def pca_denoise(
    hyper_table: "HyperTable",
    n_components: int
) -> "HyperTable":
    """
    Apply PCA-based denoising to a HyperTable object.

    Parameters
    ----------
    hyper_table : HyperTable
        Input hyperspectral dataset.
    n_components : int
        Number of principal components to retain. Higher = less denoising, 
        lower = more aggressive denoising.

    Returns
    -------
    HyperTable
        New HyperTable object with denoised data reconstructed from PCA.
    """
    if n_components <= 0 or n_components > hyper_table.bands:
        raise ValueError("n_components must be between 1 and number of bands.")

    # Fit PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(hyper_table.data.values)

    # Reconstruct data using only top components
    reconstructed = pca.inverse_transform(transformed)

    return HyperTable(
        pd.DataFrame(reconstructed, columns=hyper_table.data.columns),
        wavelengths=hyper_table.wavelengths,
        metadata={**hyper_table.metadata, "denoising": f"PCA (n={n_components})"}
    )

def band_average(
    hyper_table: "HyperTable",
    window_size: int = 3
) -> "HyperTable":
    """
    Apply band averaging (spectral smoothing) to reduce noise.

    Parameters
    ----------
    hyper_table : HyperTable
        Input hyperspectral dataset.
    window_size : int, default=3
        Number of adjacent bands to average. Must be >= 2.

    Returns
    -------
    HyperTable
        New HyperTable object with smoothed spectral bands.
    """
    if window_size < 2:
        raise ValueError("window_size must be >= 2")

    data = hyper_table.data.values
    n_samples, n_bands = data.shape

    # Number of resulting bands after averaging
    n_new_bands = n_bands // window_size

    # Average adjacent bands
    smoothed_data = np.zeros((n_samples, n_new_bands))
    new_wavelengths = np.zeros(n_new_bands)

    for i in range(n_new_bands):
        start = i * window_size
        end = start + window_size
        smoothed_data[:, i] = data[:, start:end].mean(axis=1)

        # Average the corresponding wavelengths too
        if hyper_table.wavelengths is not None:
            new_wavelengths[i] = hyper_table.wavelengths[start:end].mean()

    # Create DataFrame with new band labels
    smoothed_df = pd.DataFrame(
        smoothed_data,
        index=hyper_table.data.index,
        columns=[f"band_{i}" for i in range(n_new_bands)]
    )

    return HyperTable(
        smoothed_df,
        wavelengths=new_wavelengths if hyper_table.wavelengths is not None else None,
        metadata={**hyper_table.metadata, "smoothing": f"band_average(window={window_size})"}
    )



def remove_noisy_bands(
    hyper_table: "HyperTable",
    wavelength_range: tuple = None,
    variance_threshold: float = None
) -> "HyperTable":
    """
    Remove noisy or irrelevant spectral bands.

    Parameters
    ----------
    hyper_table : HyperTable
        Input hyperspectral dataset.
    wavelength_range : tuple (min_wl, max_wl), optional
        Keep only bands within this wavelength range.
        Example: (450, 900) keeps visible + NIR.
    variance_threshold : float, optional
        Remove bands whose variance is below this threshold (flat/noisy).

    Returns
    -------
    HyperTable
        New HyperTable object with noisy bands removed.
    """
    data = hyper_table.data.copy()
    wavelengths = hyper_table.wavelengths

    keep_mask = np.ones(data.shape[1], dtype=bool)

    # Filter by wavelength range
    if wavelength_range is not None and wavelengths is not None:
        min_wl, max_wl = wavelength_range
        keep_mask &= (wavelengths >= min_wl) & (wavelengths <= max_wl)

    # Filter by variance threshold
    if variance_threshold is not None:
        band_variances = data.var(axis=0).values
        keep_mask &= band_variances > variance_threshold

    # Apply mask
    filtered_data = data.iloc[:, keep_mask]

    if wavelengths is not None:
        filtered_wavelengths = wavelengths[keep_mask]
    else:
        filtered_wavelengths = None

    return HyperTable(
        filtered_data,
        wavelengths=filtered_wavelengths,
        metadata={**hyper_table.metadata, "filter": "remove_noisy_bands"}
    )


def select_wavelength_range(
    hyper_table: "HyperTable",
    ranges: list[tuple[float, float]]
) -> "HyperTable":
    """
    Select spectral bands within one or more wavelength ranges.

    Parameters
    ----------
    hyper_table : HyperTable
        Input hyperspectral dataset.
    ranges : list of (min_wl, max_wl)
        List of wavelength ranges to keep.
        Example: [(400, 700), (750, 900)] keeps visible + NIR.

    Returns
    -------
    HyperTable
        New HyperTable object with selected wavelength ranges.
    """
    if hyper_table.wavelengths is None:
        raise ValueError("Wavelengths are not defined in this HyperTable.")

    keep_mask = np.zeros(hyper_table.bands, dtype=bool)

    # Build a mask for all selected ranges
    for min_wl, max_wl in ranges:
        keep_mask |= (hyper_table.wavelengths >= min_wl) & (hyper_table.wavelengths <= max_wl)

    # Apply mask
    filtered_data = hyper_table.data.iloc[:, keep_mask]
    filtered_wavelengths = hyper_table.wavelengths[keep_mask]

    return HyperTable(
        filtered_data,
        wavelengths=filtered_wavelengths,
        metadata={**hyper_table.metadata, "filter": f"selected_ranges={ranges}"}
    )

def mahalanobis_distance(hyper_table: "HyperTable") -> np.ndarray:
    """
    Compute Mahalanobis distance of each sample (row) in a HyperTable.

    Parameters
    ----------
    hyper_table : HyperTable
        Input hyperspectral dataset.

    Returns
    -------
    np.ndarray
        Array of Mahalanobis distances for each sample.
    """
    X = hyper_table.data.values

    # Mean of each band
    mean_vec = np.mean(X, axis=0)

    # Covariance matrix of bands
    cov_matrix = np.cov(X, rowvar=False)

    # Inverse covariance (handle singular case safely)
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        inv_cov_matrix = np.linalg.pinv(cov_matrix)

    # Compute distances
    m_dist = [distance.mahalanobis(x, mean_vec, inv_cov_matrix) for x in X]

    return np.array(m_dist)

def isolation_forest_filter(
    hyper_table: "HyperTable",
    contamination: float = 0.05,
    random_state: int = 42,
    return_mask: bool = False
) -> "HyperTable":
    """
    Apply Isolation Forest anomaly detection to remove noisy/outlier samples 
    from a HyperTable object.

    Parameters
    ----------
    hyper_table : HyperTable
        Input HyperTable object.
    contamination : float, default=0.05
        Proportion of expected outliers in the data (between 0 and 0.5).
    random_state : int, default=42
        Random seed for reproducibility.
    return_mask : bool, default=False
        If True, returns (HyperTable, mask) where mask is a boolean array
        indicating which samples were kept.

    Returns
    -------
    HyperTable
        New HyperTable object with outliers removed.
    mask : np.ndarray, optional
        Boolean mask of samples (only if return_mask=True).
    """
    X = hyper_table.data.values

    # Fit Isolation Forest
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )
    preds = iso.fit_predict(X)  # -1 = outlier, 1 = inlier

    # Mask to select inliers
    mask = preds == 1
    filtered_data = hyper_table.data[mask]

    new_ht = HyperTable(
        filtered_data,
        wavelengths=hyper_table.wavelengths,
        metadata=hyper_table.metadata.copy()
    )

    if return_mask:
        return new_ht, mask
    else:
        return new_ht
