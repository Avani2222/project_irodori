"""
Module with functions for preprocessing tasks on irodori data including _rebuild_hypertable, minmax_scale, standardize, vector_normalize, 
apply_savgol_filter, band_average, pca_denoise, remove_noisy_bands, select_wavelength_range, mahalanobis_distance, isolation_forest_filter, correct_baseline, 
normalize_vector, spectral_shift, mixup, spectral_derivative, add_noise, spectral_index, resample_spectra, estimate_snr, multiplicative_scatter_correction, 
standard_normal_variate, savgol_first_derivative, 
savgol_second_derivative, baseline_als, apply_baseline_correction, resample_wavelengths, and remove_outliers_zscore.
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.ensemble import IsolationForest
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .core import HyperTable
import matplotlib.pyplot as plt          
from scipy.spatial import ConvexHull     
from sklearn.feature_selection import f_classif  # For ANOVA F-test
from typing import List, Tuple, Optional, Union     

# ==============================
# Helper to rebuild HyperTable
# ==============================
def _rebuild_hypertable(hyper_table, new_df: pd.DataFrame, metadata_updates: Optional[dict] = None, filtered_indices: Optional[np.ndarray] = None):
    """
    Build a new HyperTable from a DataFrame `new_df`. Ensures that the labels match the filtered rows
    and computes a new wavelengths array if needed.

    Parameters
    ----------
    hyper_table : HyperTable
        Original HyperTable.
    new_df : pd.DataFrame
        Filtered data (without label column or with label column).
    metadata_updates : dict, optional
        Updates to original metadata.
    filtered_indices : np.ndarray, optional
        Row indices from the original hyper_table that correspond to new_df rows.

    Returns
    -------
    HyperTable
    """
    df = new_df.copy()

    # handle labels
    if hasattr(hyper_table, "labels") and hyper_table.labels is not None:
        if filtered_indices is not None:
            filtered_labels = np.array(hyper_table.labels)[filtered_indices]
        else:
            filtered_labels = np.array(hyper_table.labels)

        # insert labels if not already present
        if "Label" not in df.columns and df.shape[1] == getattr(hyper_table, "bands", df.shape[1]):
            df.insert(0, "Label", filtered_labels)
        else:
            # if label column exists, replace with filtered labels
            df.iloc[:, 0] = filtered_labels

    # compute new wavelengths
    old_wl = getattr(hyper_table, "wavelengths", None)
    n_new_bands = df.shape[1] - (1 if "Label" in df.columns else 0)

    new_wavelengths = None
    if old_wl is not None:
        old_wl = np.asarray(old_wl)
        n_old = len(old_wl)
        if n_new_bands == n_old:
            new_wavelengths = old_wl.copy()
        else:
            # fallback: group old wavelengths evenly
            edges = np.linspace(0, n_old, n_new_bands + 1, dtype=int)
            new_wavelengths = np.array([
                old_wl[edges[i]:edges[i+1]].mean() if edges[i+1] > edges[i] else old_wl[min(edges[i], n_old-1)]
                for i in range(n_new_bands)
            ])

    # merge metadata
    new_meta = dict(getattr(hyper_table, "metadata", {}) or {})
    if metadata_updates:
        new_meta.update(metadata_updates)

    return HyperTable(df, wavelengths=new_wavelengths, metadata=new_meta)


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
def mahalanobis_distance(hyper_table):
    spectra = getattr(hyper_table, "spectra", None)
    if spectra is None:
        spectra = getattr(hyper_table, "data").values

    # compute covariance and invert using pinv (robust for singular)
    cov = np.cov(spectra, rowvar=False)
    # small regularization to stabilize numerical issues
    reg = 1e-8 * np.eye(cov.shape[0])
    inv_cov = np.linalg.pinv(cov + reg)

    mean = np.mean(spectra, axis=0)
    diffs = spectra - mean
    d2 = np.sum(diffs @ inv_cov * diffs, axis=1)  # squared distances
    return np.sqrt(np.maximum(d2, 0.0))


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
    new_ht = _rebuild_hypertable(hyper_table, filtered_data, filtered_indices=mask.nonzero()[0])

    return (new_ht, mask) if return_mask else new_ht

def correct_baseline(ht: HyperTable, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> HyperTable:
    """
    Apply baseline correction to all spectra in a HyperTable.

    Parameters
    ----------
    ht : HyperTable
        Hyperspectral dataset.
    lam : float
        Smoothness parameter.
    p : float
        Asymmetry parameter.
    niter : int
        Number of iterations.

    Returns
    -------
    HyperTable
        New HyperTable with baseline-corrected spectra.
    """
    corrected_data = ht.data.copy()
    for i in range(ht.samples):
        spectrum = ht.get_pixel(i)
        baseline = baseline_als(spectrum, lam=lam, p=p, niter=niter)
        corrected_data.iloc[i, :] = spectrum - baseline

    # Construct a new HyperTable
    return HyperTable(
        data=pd.concat([pd.Series(ht.labels, name="label"), corrected_data], axis=1),
        wavelengths=ht.wavelengths,
        metadata={**ht.metadata, "baseline_corrected": True}
    )

def normalize_vector(ht: HyperTable) -> HyperTable:
    """
    Apply vector normalization (L2 norm = 1) to all spectra in a HyperTable.

    Parameters
    ----------
    ht : HyperTable
        Hyperspectral dataset.

    Returns
    -------
    HyperTable
        New HyperTable with vector-normalized spectra.
    """
    normalized_data = ht.data.copy()
    for i in range(ht.samples):
        spectrum = ht.get_pixel(i)
        norm = np.linalg.norm(spectrum, ord=2)
        if norm > 0:
            normalized_data.iloc[i, :] = spectrum / norm
        else:
            normalized_data.iloc[i, :] = spectrum  # leave unchanged if zero vector

    return HyperTable(
        data=pd.concat([pd.Series(ht.labels, name="label"), normalized_data], axis=1),
        wavelengths=ht.wavelengths,
        metadata={**ht.metadata, "vector_normalized": True}
    )

def spectral_shift(ht: HyperTable, shift: float) -> HyperTable:
    """
    Apply spectral shift to all spectra in a HyperTable.

    Parameters
    ----------
    ht : HyperTable
        Hyperspectral dataset.
    shift : float
        Number of bands to shift (positive = right, negative = left).
        Can be fractional (uses interpolation).

    Returns
    -------
    HyperTable
        New HyperTable with shifted spectra.
    """
    shifted_data = ht.data.copy()
    bands = ht.bands
    x = np.arange(bands)

    for i in range(ht.samples):
        spectrum = ht.get_pixel(i)
        f = interp1d(x, spectrum, kind="linear", bounds_error=False, fill_value="extrapolate")
        shifted_x = x - shift
        shifted_data.iloc[i, :] = f(shifted_x)

    return HyperTable(
        data=pd.concat([pd.Series(ht.labels, name="label"), shifted_data], axis=1),
        wavelengths=ht.wavelengths,  # wavelengths stay same, only spectra shifted
        metadata={**ht.metadata, "spectral_shift": shift}
    )

def mixup(ht: HyperTable, alpha: float = 0.4, n_samples: int = None) -> HyperTable:
    """
    Apply Mixup data augmentation to a HyperTable.

    Parameters
    ----------
    ht : HyperTable
        Hyperspectral dataset.
    alpha : float, default=0.4
        Beta distribution parameter (controls interpolation strength).
        alpha=0 -> no mixup, higher alpha -> more aggressive mixing.
    n_samples : int, optional
        Number of new samples to generate.
        If None, will generate the same number as in ht.

    Returns
    -------
    HyperTable
        New HyperTable with mixed samples.
    """
    X = ht.data.values
    y = ht.labels
    n = ht.samples

    if n_samples is None:
        n_samples = n

    mixed_X = []
    mixed_y = []

    for _ in range(n_samples):
        i, j = np.random.choice(n, 2, replace=False)
        lam = np.random.beta(alpha, alpha)

        x_new = lam * X[i] + (1 - lam) * X[j]
        # mix labels (supports numeric labels)
        if np.issubdtype(y.dtype, np.number):
            y_new = lam * y[i] + (1 - lam) * y[j]
        else:
            # For categorical labels: store tuple or string
            y_new = f"mix({y[i]},{y[j]})"

        mixed_X.append(x_new)
        mixed_y.append(y_new)

    mixed_df = pd.DataFrame(mixed_X, columns=ht.data.columns)
    mixed_df.insert(0, "label", mixed_y)

    return HyperTable(
        data=mixed_df,
        wavelengths=ht.wavelengths,
        metadata={**ht.metadata, "augmented": "mixup", "alpha": alpha}
    )

def spectral_derivative(ht: HyperTable, order: int = 1, window_length: int = 11, polyorder: int = 2) -> HyperTable:
    """
    Compute derivative spectra using Savitzky–Golay filter.

    Parameters
    ----------
    ht : HyperTable
        Hyperspectral dataset.
    order : int, default=1
        Derivative order (1 = first derivative, 2 = second derivative).
    window_length : int
        Smoothing window.
    polyorder : int
        Polynomial order.

    Returns
    -------
    HyperTable
        HyperTable with derivative spectra.
    """
    derivative_data = ht.data.copy()
    for i in range(ht.samples):
        spectrum = ht.get_pixel(i)
        derivative_data.iloc[i, :] = savgol_filter(spectrum, window_length=window_length, polyorder=polyorder, deriv=order)
    return HyperTable(
        data=pd.concat([pd.Series(ht.labels, name="label"), derivative_data], axis=1),
        wavelengths=ht.wavelengths,
        metadata={**ht.metadata, "derivative": order}
    )

def add_noise(ht: HyperTable, noise_level: float = 0.01) -> HyperTable:
    """
    Add Gaussian noise to spectra.

    Parameters
    ----------
    ht : HyperTable
    noise_level : float
        Standard deviation of noise relative to max spectrum.

    Returns
    -------
    HyperTable
    """
    noisy_data = ht.data.copy()
    for i in range(ht.samples):
        spectrum = ht.get_pixel(i)
        noise = np.random.normal(0, noise_level * spectrum.max(), spectrum.shape)
        noisy_data.iloc[i, :] = spectrum + noise

    return HyperTable(
        data=pd.concat([pd.Series(ht.labels, name="label"), noisy_data], axis=1),
        wavelengths=ht.wavelengths,
        metadata={**ht.metadata, "noise_added": noise_level}
    )

def spectral_index(ht: HyperTable, band1: int, band2: int, index_name: str = None) -> HyperTable:
    """
    Compute normalized difference index between two bands.

    Parameters
    ----------
    ht : HyperTable
    band1 : int
    band2 : int
    index_name : str, optional

    Returns
    -------
    HyperTable
        HyperTable with one column representing the index.
    """
    X = ht.data.values
    ndi = (X[:, band1] - X[:, band2]) / (X[:, band1] + X[:, band2] + 1e-12)
    df = pd.DataFrame(ndi, columns=[index_name or f"NDI_{band1}_{band2}"])
    df.insert(0, "label", ht.labels)
    return HyperTable(data=df, wavelengths=None, metadata={**ht.metadata, "spectral_index": f"{band1}-{band2}"})

def resample_spectra(ht: HyperTable, new_wavelengths: np.ndarray) -> HyperTable:
    """
    Interpolate spectra to new wavelength positions.

    Parameters
    ----------
    ht : HyperTable
    new_wavelengths : np.ndarray
        New wavelength grid.

    Returns
    -------
    HyperTable
    """
    if ht.wavelengths is None:
        raise ValueError("Original wavelengths not defined.")

    resampled_data = []
    for i in range(ht.samples):
        spectrum = ht.get_pixel(i)
        f = interp1d(ht.wavelengths, spectrum, kind="linear", bounds_error=False, fill_value="extrapolate")
        resampled_data.append(f(new_wavelengths))

    df = pd.DataFrame(resampled_data, columns=[f"band_{i}" for i in range(len(new_wavelengths))])
    df.insert(0, "label", ht.labels)
    return HyperTable(data=df, wavelengths=new_wavelengths, metadata={**ht.metadata, "resampled": True})

def estimate_snr(ht: HyperTable, band_range: tuple = None) -> np.ndarray:
    """
    Estimate SNR as mean / std in selected band range.

    Parameters
    ----------
    ht : HyperTable
    band_range : tuple, optional
        (start_band, end_band) to compute SNR.

    Returns
    -------
    np.ndarray
        SNR per sample.
    """
    X = ht.data.values
    if band_range:
        X = X[:, band_range[0]:band_range[1]]
    snr = X.mean(axis=1) / (X.std(axis=1) + 1e-12)
    return snr

def multiplicative_scatter_correction(ht: HyperTable) -> HyperTable:
    """
    Apply Multiplicative Scatter Correction (MSC) to reduce scatter effects.
    """
    data = ht.data.values.astype(float)
    mean_spectrum = data.mean(axis=0)
    
    corrected = np.zeros_like(data)
    
    for i, spectrum in enumerate(data):
        # Linear regression of spectrum vs mean_spectrum
        fit = np.polyfit(mean_spectrum, spectrum, 1)
        slope, intercept = fit
        corrected[i] = (spectrum - intercept) / slope

    corrected_df = pd.DataFrame(corrected, columns=ht.data.columns)
    return _rebuild_hypertable(ht, corrected_df, {"preprocessing": "MSC"})

def standard_normal_variate(ht: HyperTable) -> HyperTable:
    """
    Apply SNV (Standard Normal Variate) to each spectrum (row-wise).
    """
    data = ht.data.values.astype(float)
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std == 0] = 1  # avoid divide by zero
    snv_data = (data - mean) / std

    snv_df = pd.DataFrame(snv_data, columns=ht.data.columns)
    return _rebuild_hypertable(ht, snv_df, {"preprocessing": "SNV"})

def savgol_first_derivative(ht: HyperTable, window_length: int = 11, polyorder: int = 2) -> HyperTable:
    """
    Compute first derivative using Savitzky–Golay filter.
    """
    deriv_data = savgol_filter(ht.data.values, window_length, polyorder, deriv=1, axis=1)
    deriv_df = pd.DataFrame(deriv_data, columns=ht.data.columns)
    return _rebuild_hypertable(ht, deriv_df, {"preprocessing": "first_derivative"})

def savgol_second_derivative(ht: HyperTable, window_length: int = 11, polyorder: int = 2) -> HyperTable:
    """
    Compute second derivative using Savitzky–Golay filter.
    """
    deriv_data = savgol_filter(ht.data.values, window_length, polyorder, deriv=2, axis=1)
    deriv_df = pd.DataFrame(deriv_data, columns=ht.data.columns)
    return _rebuild_hypertable(ht, deriv_df, {"preprocessing": "second_derivative"})

def baseline_als(y, lam=1e5, p=0.01, niter=10):
    """
    Asymmetric least squares baseline correction.
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def apply_baseline_correction(ht: HyperTable, lam=1e5, p=0.01, niter=10) -> HyperTable:
    """
    Apply ALS baseline correction to all spectra in a HyperTable.
    """
    corrected_data = ht.data.copy()
    for i in range(ht.samples):
        spectrum = ht.get_pixel(i)
        baseline = baseline_als(spectrum, lam, p, niter)
        corrected_data.iloc[i, :] = spectrum - baseline
    return _rebuild_hypertable(ht, corrected_data, {"preprocessing": "baseline_als"})

def resample_wavelengths(ht: HyperTable, new_wavelengths: np.ndarray) -> HyperTable:
    """
    Resample hyperspectral data to a new set of wavelengths using linear interpolation.
    """
    if ht.wavelengths is None:
        raise ValueError("Original wavelengths not defined.")

    resampled_data = np.zeros((ht.samples, len(new_wavelengths)))
    for i in range(ht.samples):
        f = interp1d(ht.wavelengths, ht.get_pixel(i), kind='linear', bounds_error=False, fill_value='extrapolate')
        resampled_data[i, :] = f(new_wavelengths)

    resampled_df = pd.DataFrame(resampled_data, columns=[f"wl_{int(wl)}" for wl in new_wavelengths])
    return _rebuild_hypertable(ht, resampled_df, {"preprocessing": "resampled", "new_wavelengths": new_wavelengths.tolist()})

def remove_outliers_zscore(ht: HyperTable, threshold: float = 3.0) -> HyperTable:
    """
    Remove samples with any band having a Z-score above a threshold.
    """
    data = ht.data.values
    z_scores = (data - data.mean(axis=0)) / data.std(axis=0)
    mask = ~(np.abs(z_scores) > threshold).any(axis=1)
    filtered_df = ht.data[mask]
    return _rebuild_hypertable(ht, filtered_df, {"preprocessing": "zscore_outlier_removed"})

