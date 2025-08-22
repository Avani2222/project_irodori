import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.feature_selection import f_classif


def compute_ndvi(hyper_table: "HyperTable",
                 red_wavelength: float = 660,
                 nir_wavelength: float = 800,
                 image_shape: tuple = None,
                 cmap: str = "RdYlGn") -> np.ndarray:
    """
    Compute the Normalized Difference Vegetation Index (NDVI).

    NDVI = (NIR - RED) / (NIR + RED)

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container with wavelengths and spectra.
    red_wavelength : float, default=660
        Approximate wavelength (nm) for the RED band.
    nir_wavelength : float, default=800
        Approximate wavelength (nm) for the NIR band.
    image_shape : tuple of int, optional
        Shape of the spatial image (rows, cols). If provided, NDVI will be reshaped
        and displayed as an image.
    cmap : str, default="RdYlGn"
        Colormap for visualization.

    Returns
    -------
    np.ndarray
        NDVI values per pixel.
    """
    if hyper_table.wavelengths is None:
        raise ValueError("Wavelengths are not defined in the HyperTable.")

    red_idx = np.argmin(np.abs(hyper_table.wavelengths - red_wavelength))
    nir_idx = np.argmin(np.abs(hyper_table.wavelengths - nir_wavelength))

    red_band = hyper_table.get_band(red_idx)
    nir_band = hyper_table.get_band(nir_idx)

    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10)

    if image_shape is not None:
        plt.imshow(ndvi.reshape(image_shape), cmap=cmap)
        plt.colorbar(label="NDVI")
        plt.title("NDVI Heatmap")
        plt.axis("off")
        plt.show()

    return ndvi


def compute_ndwi(hyper_table: "HyperTable",
                 green_wavelength: float = 560,
                 nir_wavelength: float = 860,
                 image_shape: tuple = None,
                 cmap: str = "Blues") -> np.ndarray:
    """
    Compute the Normalized Difference Water Index (NDWI).

    NDWI = (GREEN - NIR) / (GREEN + NIR)

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container with wavelengths and spectra.
    green_wavelength : float, default=560
        Approximate wavelength (nm) for the GREEN band.
    nir_wavelength : float, default=860
        Approximate wavelength (nm) for the NIR band.
    image_shape : tuple of int, optional
        Shape of the spatial image (rows, cols). If provided, NDWI will be reshaped
        and displayed as an image.
    cmap : str, default="Blues"
        Colormap for visualization.

    Returns
    -------
    np.ndarray
        NDWI values per pixel.
    """
    if hyper_table.wavelengths is None:
        raise ValueError("Wavelengths are not defined in the HyperTable.")

    green_idx = np.argmin(np.abs(hyper_table.wavelengths - green_wavelength))
    nir_idx = np.argmin(np.abs(hyper_table.wavelengths - nir_wavelength))

    green_band = hyper_table.get_band(green_idx)
    nir_band = hyper_table.get_band(nir_idx)

    ndwi = (green_band - nir_band) / (green_band + nir_band + 1e-10)

    if image_shape is not None:
        plt.imshow(ndwi.reshape(image_shape), cmap=cmap)
        plt.colorbar(label="NDWI")
        plt.title("NDWI Heatmap")
        plt.axis("off")
        plt.show()

    return ndwi


def compute_savi(hyper_table: "HyperTable",
                 red_wavelength: float = 670,
                 nir_wavelength: float = 860,
                 L: float = 0.5,
                 image_shape: tuple = None,
                 cmap: str = "YlGn") -> np.ndarray:
    """
    Compute the Soil Adjusted Vegetation Index (SAVI).

    SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container with wavelengths and spectra.
    red_wavelength : float, default=670
        Approximate wavelength (nm) for the RED band.
    nir_wavelength : float, default=860
        Approximate wavelength (nm) for the NIR band.
    L : float, default=0.5
        Soil brightness correction factor.
    image_shape : tuple of int, optional
        Shape of the spatial image (rows, cols). If provided, SAVI will be reshaped
        and displayed as an image.
    cmap : str, default="YlGn"
        Colormap for visualization.

    Returns
    -------
    np.ndarray
        SAVI values per pixel.
    """
    if hyper_table.wavelengths is None:
        raise ValueError("Wavelengths are not defined in the HyperTable.")

    red_idx = np.argmin(np.abs(hyper_table.wavelengths - red_wavelength))
    nir_idx = np.argmin(np.abs(hyper_table.wavelengths - nir_wavelength))

    red_band = hyper_table.get_band(red_idx)
    nir_band = hyper_table.get_band(nir_idx)

    savi = ((nir_band - red_band) / (nir_band + red_band + L)) * (1 + L)

    if image_shape is not None:
        plt.imshow(savi.reshape(image_shape), cmap=cmap)
        plt.colorbar(label="SAVI")
        plt.title("SAVI Heatmap")
        plt.axis("off")
        plt.show()

    return savi


def compute_custom_index(hyper_table: "HyperTable",
                         formula: str,
                         band_map: dict,
                         image_shape: tuple = None,
                         cmap: str = "RdYlGn") -> np.ndarray:
    """
    Compute a user-defined spectral index.

    Example
    -------
    formula = "(NIR - RED) / (NIR + RED)"
    band_map = {"RED": 670, "NIR": 860, "L": 0.5}

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container with wavelengths and spectra.
    formula : str
        Mathematical expression defining the index. Band variables must be present
        in `band_map`.
    band_map : dict
        Mapping of band names to wavelengths or constants. Example:
        {"RED": 670, "NIR": 860, "L": 0.5}.
    image_shape : tuple of int, optional
        Shape of the spatial image (rows, cols). If provided, index values will be reshaped
        and displayed as an image.
    cmap : str, default="RdYlGn"
        Colormap for visualization.

    Returns
    -------
    np.ndarray
        Computed index values per pixel.
    """
    if hyper_table.wavelengths is None:
        raise ValueError("Wavelengths are not defined in the HyperTable.")

    local_vars = {}
    for name, value in band_map.items():
        if isinstance(value, (int, float)):
            local_vars[name] = value
        else:
            band_idx = np.argmin(np.abs(hyper_table.wavelengths - value))
            local_vars[name] = hyper_table.get_band(band_idx)

    try:
        index_values = eval(formula, {"np": np}, local_vars)
    except Exception as e:
        raise ValueError(f"Error evaluating formula: {e}")

    if image_shape is not None:
        plt.imshow(index_values.reshape(image_shape), cmap=cmap)
        plt.colorbar(label="Custom Index")
        plt.title("Custom Index Heatmap")
        plt.show()

    return index_values


def first_derivative(hyper_table: "HyperTable",
                     show_plot: bool = False,
                     sample_indices: list = None) -> "HyperTable":
    """
    Compute the first derivative of hyperspectral spectra.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container with wavelengths and spectra.
    show_plot : bool, default=False
        If True, plots the original and derivative spectra for selected samples.
    sample_indices : list of int, optional
        Indices of samples to plot when `show_plot=True`.

    Returns
    -------
    HyperTable
        New HyperTable containing first derivative spectra.
    """
    data = hyper_table.data.values.astype(float)

    if hyper_table.wavelengths is not None:
        derivative_data = np.gradient(data, hyper_table.wavelengths, axis=1)
    else:
        derivative_data = np.diff(data, axis=1)
        derivative_data = np.hstack([derivative_data,
                                     derivative_data[:, -1][:, None]])

    derivative_df = pd.DataFrame(derivative_data, index=hyper_table.data.index)
    derivative_ht = HyperTable(derivative_df,
                               wavelengths=hyper_table.wavelengths,
                               metadata={**hyper_table.metadata, "processed": "first_derivative"})

    if show_plot:
        if sample_indices is None:
            sample_indices = [0]
        wl = hyper_table.wavelengths if hyper_table.wavelengths is not None else np.arange(hyper_table.bands)
        plt.figure(figsize=(8, 5))
        for idx in sample_indices:
            plt.plot(wl, hyper_table.get_pixel(idx), label=f"Original {idx}", alpha=0.6)
            plt.plot(wl, derivative_ht.get_pixel(idx), "--", label=f"Derivative {idx}")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance / Derivative")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return derivative_ht

def plot_spectral_signatures(hyper_table: "HyperTable",
                             sample_indices: list = None,
                             labels: bool = True,
                             title: str = "Spectral Signatures",
                             figsize: tuple = (8, 6),
                             alpha: float = 0.8,
                             cmap: str = "tab10"):
    """
    Plot spectral signatures for selected samples from a HyperTable.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container with wavelengths and spectra.
    sample_indices : list of int, optional
        List of sample indices to plot. If None, a few random samples are chosen.
    labels : bool, default=True
        Whether to show sample labels in the legend (if available).
    title : str, default="Spectral Signatures"
        Title of the plot.
    figsize : tuple of int, default=(8, 6)
        Size of the figure.
    alpha : float, default=0.8
        Transparency of the plotted curves.
    cmap : str, default="tab10"
        Colormap used to differentiate curves.

    Returns
    -------
    None
        Displays a matplotlib plot.
    """
    if hyper_table.wavelengths is None:
        wavelengths = np.arange(hyper_table.bands)
    else:
        wavelengths = hyper_table.wavelengths

    if sample_indices is None:
        sample_indices = np.random.choice(hyper_table.samples,
                                          size=min(5, hyper_table.samples),
                                          replace=False)

    plt.figure(figsize=figsize)
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(sample_indices)))

    for i, idx in enumerate(sample_indices):
        spectrum = hyper_table.get_pixel(idx)
        if labels and hyper_table.labels is not None:
            lbl = f"Sample {idx} ({hyper_table.labels.iloc[idx]})"
        else:
            lbl = f"Sample {idx}"
        plt.plot(wavelengths, spectrum,
                 label=lbl,
                 color=colors[i],
                 alpha=alpha)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def second_derivative(hyper_table: "HyperTable",
                      window_length: int = 7,
                      polyorder: int = 2) -> "HyperTable":
    """
    Compute the second derivative spectra of all samples using Savitzky–Golay filter.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    window_length : int, default=7
        The length of the filter window (number of coefficients).
        Must be odd and >= polyorder + 2.
    polyorder : int, default=2
        The order of the polynomial used to fit the samples.

    Returns
    -------
    HyperTable
        A new HyperTable with second derivative spectra.
    """
    from scipy.signal import savgol_filter

    if window_length >= hyper_table.bands:
        raise ValueError("window_length must be smaller than number of bands.")
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd.")

    # Apply Savitzky–Golay filter for 2nd derivative along each row
    deriv_data = savgol_filter(hyper_table.data.values,
                               window_length=window_length,
                               polyorder=polyorder,
                               deriv=2,
                               axis=1)

    # Create new DataFrame preserving column names
    deriv_df = pd.DataFrame(deriv_data, columns=hyper_table.data.columns)

    return HyperTable(
        data=pd.concat([pd.Series(hyper_table.labels, name="Label"), deriv_df], axis=1),
        wavelengths=hyper_table.wavelengths,
        metadata={**hyper_table.metadata, "transform": "second_derivative"}
    )



def anova_f_test(hyper_table: "HyperTable",
                 top_k: int = 10,
                 visualize: bool = True,
                 figsize: tuple = (10, 5)) -> pd.DataFrame:
    """
    Perform ANOVA F-test to rank spectral bands by discriminative power.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container with labels.
    top_k : int, default=10
        Number of top-ranked bands to return.
    visualize : bool, default=True
        Whether to visualize F-values across wavelengths.
    figsize : tuple, default=(10, 5)
        Figure size for visualization.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['Band_Index', 'Wavelength', 'F_value', 'p_value'],
        sorted by F_value (descending).
    """
    if hyper_table.labels is None:
        raise ValueError("Labels are required for ANOVA F-test.")

    X = hyper_table.data.values
    y = hyper_table.labels

    # Perform one-way ANOVA F-test
    F_vals, p_vals = f_classif(X, y)

    # Build result table
    result = pd.DataFrame({
        "Band_Index": np.arange(hyper_table.bands),
        "Wavelength": (hyper_table.wavelengths
                       if hyper_table.wavelengths is not None
                       else np.arange(hyper_table.bands)),
        "F_value": F_vals,
        "p_value": p_vals
    }).sort_values(by="F_value", ascending=False)

    # Visualization
    if visualize:
        wavelengths = (hyper_table.wavelengths
                       if hyper_table.wavelengths is not None
                       else np.arange(hyper_table.bands))

        plt.figure(figsize=figsize)
        plt.plot(wavelengths, F_vals, label="F-value", color="steelblue")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("F-value")
        plt.title("ANOVA F-test across spectral bands")
        plt.grid(True, alpha=0.3)

        # Highlight top_k bands
        top_bands = result.head(top_k)
        plt.scatter(top_bands["Wavelength"], top_bands["F_value"],
                    color="red", label=f"Top {top_k} bands", zorder=5)

        for _, row in top_bands.iterrows():
            plt.text(row["Wavelength"], row["F_value"],
                     f"{int(row['Band_Index'])}", fontsize=8,
                     ha="center", va="bottom", color="darkred")

        plt.legend()
        plt.tight_layout()
        plt.show()

    return result.head(top_k)



def spectral_angle_mapper(hyper_table: "HyperTable",
                          reference: np.ndarray,
                          visualize: bool = True,
                          in_degrees: bool = True,
                          figsize: tuple = (8, 4)) -> np.ndarray:
    """
    Perform Spectral Angle Mapper (SAM) similarity calculation.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    reference : np.ndarray
        Reference spectrum (1D array of length = number of bands).
    visualize : bool, default=True
        Whether to visualize SAM scores across samples.
    in_degrees : bool, default=True
        Whether to return angles in degrees (otherwise radians).
    figsize : tuple, default=(8, 4)
        Figure size for visualization.

    Returns
    -------
    np.ndarray
        SAM angles for each sample (lower = more similar).
    """
    X = hyper_table.data.values
    ref = np.asarray(reference)

    if ref.shape[0] != hyper_table.bands:
        raise ValueError("Reference spectrum must match number of bands.")

    # Normalize spectra
    dot_products = np.sum(X * ref, axis=1)
    norms = np.linalg.norm(X, axis=1) * np.linalg.norm(ref)
    cos_theta = np.clip(dot_products / norms, -1, 1)

    # Convert to angle
    angles = np.arccos(cos_theta)
    if in_degrees:
        angles = np.degrees(angles)

    # Visualization
    if visualize:
        plt.figure(figsize=figsize)
        plt.plot(angles, "o-", color="darkorange", label="SAM Angle")
        plt.xlabel("Sample Index")
        plt.ylabel("Angle (degrees)" if in_degrees else "Angle (radians)")
        plt.title("Spectral Angle Mapper (SAM)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return angles

def spectral_information_divergence(hyper_table: "HyperTable",
                                    reference: np.ndarray,
                                    visualize: bool = True,
                                    figsize: tuple = (8, 4)) -> np.ndarray:
    """
    Perform Spectral Information Divergence (SID) similarity calculation.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    reference : np.ndarray
        Reference spectrum (1D array of length = number of bands).
    visualize : bool, default=True
        Whether to visualize SID scores across samples.
    figsize : tuple, default=(8, 4)
        Figure size for visualization.

    Returns
    -------
    np.ndarray
        SID values for each sample (lower = more similar).
    """
    X = hyper_table.data.values
    ref = np.asarray(reference, dtype=float)

    if ref.shape[0] != hyper_table.bands:
        raise ValueError("Reference spectrum must match number of bands.")

    # Normalize to probability distributions
    X_prob = X / X.sum(axis=1, keepdims=True)
    ref_prob = ref / ref.sum()

    # Avoid division by zero
    eps = 1e-12
    X_prob = np.clip(X_prob, eps, 1)
    ref_prob = np.clip(ref_prob, eps, 1)

    # Compute SID (symmetric KL divergence)
    sid_scores = []
    for p in X_prob:
        sid = np.sum(p * np.log(p / ref_prob)) + np.sum(ref_prob * np.log(ref_prob / p))
        sid_scores.append(sid)

    sid_scores = np.array(sid_scores)

    # Visualization
    if visualize:
        plt.figure(figsize=figsize)
        plt.plot(sid_scores, "o-", color="teal", label="SID")
        plt.xlabel("Sample Index")
        plt.ylabel("SID Value")
        plt.title("Spectral Information Divergence (SID)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return sid_scores

def euclidean_distance(hyper_table: "HyperTable",
                       reference: np.ndarray,
                       visualize: bool = True,
                       figsize: tuple = (8, 4)) -> np.ndarray:
    """
    Compute Euclidean distance between each pixel spectrum and a reference spectrum.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    reference : np.ndarray
        Reference spectrum (1D array of length = number of bands).
    visualize : bool, default=True
        Whether to visualize distances across samples.
    figsize : tuple, default=(8, 4)
        Figure size for visualization.

    Returns
    -------
    np.ndarray
        Euclidean distances for each sample (lower = more similar).
    """
    X = hyper_table.data.values
    ref = np.asarray(reference, dtype=float)

    if ref.shape[0] != hyper_table.bands:
        raise ValueError("Reference spectrum must match number of bands.")

    # Euclidean distance calculation
    distances = np.linalg.norm(X - ref, axis=1)

    # Visualization
    if visualize:
        plt.figure(figsize=figsize)
        plt.plot(distances, "o-", color="purple", label="Euclidean Distance")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        plt.title("Euclidean Distance to Reference Spectrum")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return distances


def band_ratio(hyper_table: "HyperTable",
               band1: int,
               band2: int,
               visualize: bool = True,
               figsize: tuple = (8, 4)) -> np.ndarray:
    """
    Compute band ratio (R_band1 / R_band2) for each sample.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    band1 : int
        Index of the numerator band (0-based index).
    band2 : int
        Index of the denominator band (0-based index).
    visualize : bool, default=True
        Whether to visualize the band ratio values across samples.
    figsize : tuple, default=(8, 4)
        Figure size for visualization.

    Returns
    -------
    np.ndarray
        Band ratio values for each sample.
    """
    if band1 >= hyper_table.bands or band2 >= hyper_table.bands:
        raise ValueError("Band indices must be within the available range.")

    band1_values = hyper_table.get_band(band1).astype(float)
    band2_values = hyper_table.get_band(band2).astype(float)

    # Avoid division by zero
    eps = 1e-12
    ratios = band1_values / (band2_values + eps)

    # Visualization
    if visualize:
        plt.figure(figsize=figsize)
        plt.plot(ratios, "o-", color="darkgreen",
                 label=f"Band Ratio (Band {band1} / Band {band2})")
        plt.xlabel("Sample Index")
        plt.ylabel("Ratio Value")
        plt.title("Band Ratio Analysis")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return ratios

def continuum_removal(hyper_table: "HyperTable",
                      sample_index: int,
                      visualize: bool = True,
                      figsize: tuple = (8, 5)) -> np.ndarray:
    """
    Apply continuum removal to a given pixel spectrum.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    sample_index : int
        Index of the sample (row) to process.
    visualize : bool, default=True
        Whether to plot the original spectrum, continuum, and removed spectrum.
    figsize : tuple, default=(8, 5)
        Size of the plot when visualizing.

    Returns
    -------
    np.ndarray
        Continuum-removed spectrum (values between 0–1).
    """
    if hyper_table.wavelengths is None:
        raise ValueError("Wavelengths must be defined for continuum removal.")

    # Extract spectrum and wavelengths
    spectrum = hyper_table.get_pixel(sample_index).astype(float)
    wl = hyper_table.wavelengths

    # Compute convex hull (continuum line)
    points = np.column_stack((wl, spectrum))
    hull = ConvexHull(points)

    # Get upper hull indices (continuum points)
    hull_indices = hull.vertices
    hull_indices = np.sort(hull_indices)
    wl_hull = wl[hull_indices]
    spec_hull = spectrum[hull_indices]

    # Linear interpolation of continuum
    continuum = np.interp(wl, wl_hull, spec_hull)

    # Continuum removal
    continuum_removed = spectrum / (continuum + 1e-12)

    # Visualization
    if visualize:
        plt.figure(figsize=figsize)
        plt.plot(wl, spectrum, label="Original Spectrum", color="blue")
        plt.plot(wl, continuum, label="Continuum", color="red", linestyle="--")
        plt.plot(wl, continuum_removed, label="Continuum Removed", color="green")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance (a.u.)")
        plt.title(f"Continuum Removal (Sample {sample_index})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return continuum_removed

