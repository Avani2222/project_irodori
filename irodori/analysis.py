import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull

def compute_ndvi(
    hyper_table: "HyperTable",
    red_wavelength: float = 660,
    nir_wavelength: float = 800,
    visualize: bool = False,
    cmap: str = "RdYlGn"
) -> np.ndarray:
    """
    Compute NDVI (Normalized Difference Vegetation Index) from a HyperTable object.
    
    Parameters
    ----------
    hyper_table : HyperTable
        Input hyperspectral data.
    red_wavelength : float, default=660
        Wavelength (in nm) corresponding to the red band.
    nir_wavelength : float, default=800
        Wavelength (in nm) corresponding to the NIR band.
    visualize : bool, default=False
        If True, display NDVI as a heatmap.
    cmap : str, default="RdYlGn"
        Colormap for heatmap visualization.
    
    Returns
    -------
    np.ndarray
        NDVI values for each sample (row).
    """
    if hyper_table.wavelengths is None:
        raise ValueError("Wavelengths are not defined in the HyperTable.")

    # Find closest available wavelengths
    red_idx = np.argmin(np.abs(hyper_table.wavelengths - red_wavelength))
    nir_idx = np.argmin(np.abs(hyper_table.wavelengths - nir_wavelength))

    red_band = hyper_table.data.iloc[:, red_idx].values
    nir_band = hyper_table.data.iloc[:, nir_idx].values

    # Compute NDVI
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10)

    # Optional visualization
    if visualize:
        n_samples = hyper_table.samples
        side = int(np.sqrt(n_samples))
        if side * side == n_samples:  # check if reshape is possible
            ndvi_img = ndvi.reshape(side, side)
            plt.figure(figsize=(6, 6))
            sns.heatmap(ndvi_img, cmap=cmap, cbar=True, square=True)
            plt.title("NDVI Heatmap")
            plt.axis("off")
            plt.show()
        else:
            print("Warning: NDVI heatmap visualization requires samples to form a square grid.")

    return ndvi

def compute_ndwi(
    hyper_table: "HyperTable",
    green_wavelength: float = 560,
    nir_wavelength: float = 860,
    visualize: bool = False,
    cmap: str = "Blues"
) -> np.ndarray:
    """
    Compute NDWI (Normalized Difference Water Index) from a HyperTable object.
    
    Parameters
    ----------
    hyper_table : HyperTable
        Input hyperspectral data.
    green_wavelength : float, default=560
        Wavelength (in nm) corresponding to the green band.
    nir_wavelength : float, default=860
        Wavelength (in nm) corresponding to the near-infrared band.
    visualize : bool, default=False
        If True, display NDWI as a heatmap.
    cmap : str, default="Blues"
        Colormap for heatmap visualization.
    
    Returns
    -------
    np.ndarray
        NDWI values for each sample (row).
    """
    if hyper_table.wavelengths is None:
        raise ValueError("Wavelengths are not defined in the HyperTable.")

    # Find closest available wavelengths
    green_idx = np.argmin(np.abs(hyper_table.wavelengths - green_wavelength))
    nir_idx = np.argmin(np.abs(hyper_table.wavelengths - nir_wavelength))

    green_band = hyper_table.data.iloc[:, green_idx].values
    nir_band = hyper_table.data.iloc[:, nir_idx].values

    # Compute NDWI
    ndwi = (green_band - nir_band) / (green_band + nir_band + 1e-10)

    # Optional visualization
    if visualize:
        n_samples = hyper_table.samples
        side = int(np.sqrt(n_samples))
        if side * side == n_samples:  # check if samples form a square grid
            ndwi_img = ndwi.reshape(side, side)
            plt.figure(figsize=(6, 6))
            sns.heatmap(ndwi_img, cmap=cmap, cbar=True, square=True)
            plt.title("NDWI Heatmap")
            plt.axis("off")
            plt.show()
        else:
            print("Warning: NDWI heatmap visualization requires samples to form a square grid.")

    return ndwi


def compute_savi(
    hyper_table: "HyperTable",
    red_wavelength: float = 670,
    nir_wavelength: float = 860,
    L: float = 0.5,
    visualize: bool = False,
    cmap: str = "YlGn"
) -> np.ndarray:
    """
    Compute SAVI (Soil Adjusted Vegetation Index) from a HyperTable object.

    Parameters
    ----------
    hyper_table : HyperTable
        Input hyperspectral data.
    red_wavelength : float, default=670
        Wavelength (in nm) corresponding to the red band.
    nir_wavelength : float, default=860
        Wavelength (in nm) corresponding to the near-infrared band.
    L : float, default=0.5
        Soil adjustment factor (0 = equivalent to NDVI, 1 = strong correction).
    visualize : bool, default=False
        If True, display SAVI as a heatmap.
    cmap : str, default="YlGn"
        Colormap for heatmap visualization.

    Returns
    -------
    np.ndarray
        SAVI values for each sample (row).
    """
    if hyper_table.wavelengths is None:
        raise ValueError("Wavelengths are not defined in the HyperTable.")

    # Find closest available wavelengths
    red_idx = np.argmin(np.abs(hyper_table.wavelengths - red_wavelength))
    nir_idx = np.argmin(np.abs(hyper_table.wavelengths - nir_wavelength))

    red_band = hyper_table.data.iloc[:, red_idx].values
    nir_band = hyper_table.data.iloc[:, nir_idx].values

    # Compute SAVI
    savi = ((nir_band - red_band) / (nir_band + red_band + L)) * (1 + L)

    # Optional visualization
    if visualize:
        n_samples = hyper_table.samples
        side = int(np.sqrt(n_samples))
        if side * side == n_samples:  # check if samples form a square grid
            savi_img = savi.reshape(side, side)
            plt.figure(figsize=(6, 6))
            sns.heatmap(savi_img, cmap=cmap, cbar=True, square=True)
            plt.title("SAVI Heatmap")
            plt.axis("off")
            plt.show()
        else:
            print("Warning: SAVI heatmap visualization requires samples to form a square grid.")

    return savi


def compute_custom_index(
    hyper_table: "HyperTable",
    formula: str,
    band_map: dict,
    img_shape: tuple = None,
    show_heatmap: bool = False,
    cmap: str = "RdYlGn"
) -> np.ndarray:
    """
    Compute a user-defined spectral index from a HyperTable object.
    Optionally display a heatmap if data is spatial.

    Parameters
    ----------
    hyper_table : HyperTable
        Input hyperspectral data.
    formula : str
        Mathematical expression for the index.
        Example: "(NIR - RED) / (NIR + RED)"
    band_map : dict
        Mapping of variable names in formula to target wavelengths or constants.
        Example: {"RED": 670, "NIR": 860, "L": 0.5}
    img_shape : tuple, optional
        Shape of the image (rows, cols). Required for heatmap visualization.
    show_heatmap : bool, default=False
        If True, displays a heatmap of the computed index.
    cmap : str, default="RdYlGn"
        Colormap for visualization.

    Returns
    -------
    np.ndarray
        Computed index values for each sample (row).
    """
    if hyper_table.wavelengths is None:
        raise ValueError("Wavelengths are not defined in the HyperTable.")

    # Dictionary to hold actual band values
    local_vars = {}

    for name, value in band_map.items():
        if isinstance(value, (int, float)):  
            # constants like L=0.5
            local_vars[name] = value
        else:
            # assume it's a wavelength
            band_idx = np.argmin(np.abs(hyper_table.wavelengths - value))
            local_vars[name] = hyper_table.data.iloc[:, band_idx].values

    try:
        # Evaluate formula safely
        index_values = eval(formula, {"np": np}, local_vars)
    except Exception as e:
        raise ValueError(f"Error evaluating formula: {e}")

    # Heatmap visualization if requested
    if show_heatmap:
        if img_shape is None:
            raise ValueError("img_shape must be provided for heatmap visualization.")
        
        plt.imshow(index_values.reshape(img_shape), cmap=cmap)
        plt.colorbar(label="Custom Index")
        plt.title("Custom Index Heatmap")
        plt.show()

    return index_values


def spectral_angle_mapper(
    hyper_table: "HyperTable",
    reference: np.ndarray,
    in_degrees: bool = True,
    show_heatmap: bool = False,
    image_shape: tuple = None,
    cmap: str = "viridis"
) -> np.ndarray:
    """
    Compute Spectral Angle Mapper (SAM) between each pixel spectrum and a reference spectrum.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral dataset.
    reference : np.ndarray
        Reference spectrum (1D array with length equal to number of bands).
    in_degrees : bool, default=True
        If True, returns angle in degrees. Otherwise, in radians.
    show_heatmap : bool, default=False
        If True, displays SAM values as a heatmap (requires image_shape).
    image_shape : tuple, optional
        Shape of the original image (rows, cols). Required if show_heatmap=True.
    cmap : str, default="viridis"
        Colormap to use for heatmap.

    Returns
    -------
    np.ndarray
        Array of SAM values for each pixel (row).
    """
    if reference.shape[0] != hyper_table.bands:
        raise ValueError(
            f"Reference spectrum must have length {hyper_table.bands}, "
            f"but got {reference.shape[0]}."
        )

    # Convert to numpy
    data = hyper_table.data.values
    ref = reference.reshape(1, -1)

    # SAM calculation
    numerator = np.sum(data * ref, axis=1)
    denominator = np.linalg.norm(data, axis=1) * np.linalg.norm(ref)

    cos_theta = np.clip(numerator / (denominator + 1e-12), -1, 1)
    angles = np.arccos(cos_theta)

    if in_degrees:
        angles = np.degrees(angles)

    # Optional visualization
    if show_heatmap:
        if image_shape is None:
            raise ValueError("image_shape must be provided to visualize heatmap.")

        sam_image = angles.reshape(image_shape)
        plt.figure(figsize=(6, 5))
        plt.imshow(sam_image, cmap=cmap)
        plt.colorbar(label="SAM (degrees)" if in_degrees else "SAM (radians)")
        plt.title("Spectral Angle Mapper (SAM)")
        plt.axis("off")
        plt.show()

    return angles'


def spectral_information_divergence(
    hyper_table: "HyperTable",
    reference: np.ndarray,
    show_heatmap: bool = False,
    image_shape: tuple = None,
    cmap: str = "magma"
) -> np.ndarray:
    """
    Compute Spectral Information Divergence (SID) between each pixel spectrum 
    and a reference spectrum.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral dataset.
    reference : np.ndarray
        Reference spectrum (1D array with length equal to number of bands).
    show_heatmap : bool, default=False
        If True, displays SID values as a heatmap (requires image_shape).
    image_shape : tuple, optional
        Shape of the original image (rows, cols). Required if show_heatmap=True.
    cmap : str, default="magma"
        Colormap to use for heatmap.

    Returns
    -------
    np.ndarray
        Array of SID values for each pixel (row).
    """
    if reference.shape[0] != hyper_table.bands:
        raise ValueError(
            f"Reference spectrum must have length {hyper_table.bands}, "
            f"but got {reference.shape[0]}."
        )

    data = hyper_table.data.values.astype(float)
    ref = reference.astype(float)

    # Normalize to probability distributions
    data = data / (np.sum(data, axis=1, keepdims=True) + 1e-12)
    ref = ref / (np.sum(ref) + 1e-12)

    # Compute SID for each pixel
    sid = np.sum(data * np.log((data + 1e-12) / (ref + 1e-12)), axis=1) + \
          np.sum(ref * np.log((ref + 1e-12) / (data + 1e-12).T), axis=0)

    # Optional visualization
    if show_heatmap:
        if image_shape is None:
            raise ValueError("image_shape must be provided to visualize heatmap.")

        sid_image = sid.reshape(image_shape)
        plt.figure(figsize=(6, 5))
        plt.imshow(sid_image, cmap=cmap)
        plt.colorbar(label="Spectral Information Divergence (SID)")
        plt.title("SID Heatmap")
        plt.axis("off")
        plt.show()

    return sid


def euclidean_distance(
    hyper_table: "HyperTable",
    reference: np.ndarray,
    show_heatmap: bool = False,
    image_shape: tuple = None,
    cmap: str = "viridis"
) -> np.ndarray:
    """
    Compute Euclidean Distance (ED) between each pixel spectrum 
    and a reference spectrum.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral dataset.
    reference : np.ndarray
        Reference spectrum (1D array with length equal to number of bands).
    show_heatmap : bool, default=False
        If True, displays ED values as a heatmap (requires image_shape).
    image_shape : tuple, optional
        Shape of the original image (rows, cols). Required if show_heatmap=True.
    cmap : str, default="viridis"
        Colormap to use for heatmap.

    Returns
    -------
    np.ndarray
        Array of ED values for each pixel (row).
    """
    if reference.shape[0] != hyper_table.bands:
        raise ValueError(
            f"Reference spectrum must have length {hyper_table.bands}, "
            f"but got {reference.shape[0]}."
        )

    data = hyper_table.data.values.astype(float)
    ref = reference.astype(float)

    # Compute Euclidean distance per row (pixel)
    diff = data - ref
    ed = np.sqrt(np.sum(diff ** 2, axis=1))

    # Optional visualization
    if show_heatmap:
        if image_shape is None:
            raise ValueError("image_shape must be provided to visualize heatmap.")

        ed_image = ed.reshape(image_shape)
        plt.figure(figsize=(6, 5))
        plt.imshow(ed_image, cmap=cmap)
        plt.colorbar(label="Euclidean Distance")
        plt.title("Euclidean Distance Heatmap")
        plt.axis("off")
        plt.show()

    return ed

def first_derivative(
    hyper_table: "HyperTable",
    show_plot: bool = False,
    sample_indices: list = None
) -> "HyperTable":
    """
    Compute the first derivative of spectra in a HyperTable.

    Parameters
    ----------
    hyper_table : HyperTable
        Input hyperspectral dataset.
    show_plot : bool, default=False
        If True, plots the original and first derivative spectra.
    sample_indices : list of int, optional
        Indices of samples/pixels to visualize. 
        If None, only the first pixel is plotted.

    Returns
    -------
    HyperTable
        New HyperTable containing first-derivative spectra.
    """
    data = hyper_table.data.values.astype(float)

    # If wavelengths are available, use spacing, otherwise assume step=1
    if hyper_table.wavelengths is not None:
        x = hyper_table.wavelengths
        derivative_data = np.gradient(data, x, axis=1)
    else:
        derivative_data = np.diff(data, axis=1)
        # Pad to keep same shape (append NaN or last value)
        derivative_data = np.hstack([derivative_data, derivative_data[:, -1][:, None]])

    # Create new HyperTable
    derivative_ht = HyperTable(
        pd.DataFrame(derivative_data, index=hyper_table.data.index),
        wavelengths=hyper_table.wavelengths,
        metadata={**hyper_table.metadata, "processed": "first_derivative"}
    )

    # Visualization
    if show_plot:
        if sample_indices is None:
            sample_indices = [0]  # default to first pixel

        plt.figure(figsize=(8, 5))
        for idx in sample_indices:
            original = hyper_table.get_pixel(idx)
            derived = derivative_ht.get_pixel(idx)
            wl = hyper_table.wavelengths if hyper_table.wavelengths is not None else np.arange(hyper_table.bands)

            plt.plot(wl, original, label=f"Original Pixel {idx}", alpha=0.6)
            plt.plot(wl, derived, label=f"Derivative Pixel {idx}", linestyle="--")

        plt.xlabel("Wavelength (nm)" if hyper_table.wavelengths is not None else "Band Index")
        plt.ylabel("Reflectance / Derivative")
        plt.title("First Derivative of Spectra")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return derivative_ht


def plot_all_spectral_signatures(hyper_table: "HyperTable", max_samples: int = None):
    """
    Plot spectral signatures of all (or a subset of) samples in a HyperTable.

    Parameters
    ----------
    hyper_table : HyperTable
        Input hyperspectral dataset.
    max_samples : int, optional
        Maximum number of samples to plot. If None, all samples are plotted.
    """
    data = hyper_table.data.values
    wl = hyper_table.wavelengths if hyper_table.wavelengths is not None else np.arange(hyper_table.bands)

    # Subset samples if requested
    if max_samples is not None and max_samples < hyper_table.samples:
        data = data[:max_samples, :]

    plt.figure(figsize=(10, 6))
    for i in range(data.shape[0]):
        plt.plot(wl, data[i, :], linewidth=1, alpha=0.6)

    plt.xlabel("Wavelength (nm)" if hyper_table.wavelengths is not None else "Band Index")
    plt.ylabel("Reflectance")
    plt.title("Spectral Signatures of Samples")
    plt.grid(True, alpha=0.3)
    plt.show()

def second_derivative(hyper_table: "HyperTable", visualize: bool = False, max_samples: int = None) -> "HyperTable":
    """
    Compute the second derivative of spectral signatures in a HyperTable.

    Parameters
    ----------
    hyper_table : HyperTable
        Input hyperspectral dataset.
    visualize : bool, optional (default=False)
        Whether to plot the second derivative spectra.
    max_samples : int, optional
        Maximum number of samples to visualize (for clarity).

    Returns
    -------
    HyperTable
        New HyperTable object with second derivative spectra.
    """
    # Compute second derivative along spectral axis (columns)
    second_deriv = np.gradient(np.gradient(hyper_table.data.values, axis=1), axis=1)

    # Wrap in DataFrame with same column structure
    second_deriv_df = pd.DataFrame(
        second_deriv,
        columns=hyper_table.data.columns,
        index=hyper_table.data.index
    )

    result = HyperTable(
        second_deriv_df,
        wavelengths=hyper_table.wavelengths,
        metadata=hyper_table.metadata.copy()
    )

    # Visualization
    if visualize:
        wl = hyper_table.wavelengths if hyper_table.wavelengths is not None else np.arange(hyper_table.bands)
        plot_data = second_deriv
        if max_samples is not None and max_samples < hyper_table.samples:
            plot_data = plot_data[:max_samples, :]

        plt.figure(figsize=(10, 6))
        for i in range(plot_data.shape[0]):
            plt.plot(wl, plot_data[i, :], linewidth=1, alpha=0.7)
        plt.xlabel("Wavelength (nm)" if hyper_table.wavelengths is not None else "Band Index")
        plt.ylabel("Second Derivative (Δ² Reflectance)")
        plt.title("Second Derivative Spectra")
        plt.grid(True, alpha=0.3)
        plt.show()

    return result

def continuum_removal(hyper_table: "HyperTable", visualize: bool = False, sample_index: int = 0) -> "HyperTable":
    """
    Apply continuum removal to hyperspectral data in a HyperTable.

    Parameters
    ----------
    hyper_table : HyperTable
        Input hyperspectral dataset.
    visualize : bool, optional (default=False)
        Whether to visualize the continuum removal for a sample.
    sample_index : int, optional (default=0)
        Index of the sample to visualize (used if visualize=True).

    Returns
    -------
    HyperTable
        New HyperTable object with continuum-removed spectra.
    """
    wl = hyper_table.wavelengths if hyper_table.wavelengths is not None else np.arange(hyper_table.bands)
    spectra = hyper_table.data.values

    # Compute continuum for each sample
    cont_removed = []
    for spectrum in spectra:
        # Build convex hull of (wavelength, reflectance) points
        points = np.column_stack((wl, spectrum))
        hull = ConvexHull(points)

        # Extract upper hull indices (sorted by wavelength)
        hull_indices = np.unique(hull.vertices)
        hull_indices = hull_indices[np.argsort(wl[hull_indices])]

        # Interpolate continuum line
        continuum = np.interp(wl, wl[hull_indices], spectrum[hull_indices])

        # Normalize spectrum by continuum
        cr_spectrum = spectrum / continuum
        cont_removed.append(cr_spectrum)

    cont_removed = np.array(cont_removed)

    # Wrap in DataFrame
    cont_removed_df = pd.DataFrame(
        cont_removed,
        columns=hyper_table.data.columns,
        index=hyper_table.data.index
    )

    result = HyperTable(
        cont_removed_df,
        wavelengths=hyper_table.wavelengths,
        metadata=hyper_table.metadata.copy()
    )

    # Visualization
    if visualize:
        spec = spectra[sample_index]
        cr_spec = cont_removed[sample_index]

        # Continuum for visualization
        points = np.column_stack((wl, spec))
        hull = ConvexHull(points)
        hull_indices = np.unique(hull.vertices)
        hull_indices = hull_indices[np.argsort(wl[hull_indices])]
        continuum = np.interp(wl, wl[hull_indices], spec[hull_indices])

        plt.figure(figsize=(12, 6))

        # Original spectrum with continuum
        plt.subplot(1, 2, 1)
        plt.plot(wl, spec, label="Original Spectrum")
        plt.plot(wl, continuum, "--", label="Continuum", color="red")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.title(f"Continuum (Sample {sample_index})")
        plt.legend()
        plt.grid(alpha=0.3)

        # Continuum-removed spectrum
        plt.subplot(1, 2, 2)
        plt.plot(wl, cr_spec, label="Continuum Removed Spectrum", color="green")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Normalized Reflectance")
        plt.title(f"Continuum Removed (Sample {sample_index})")
        plt.grid(alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.show()

    return result

def band_ratio(
    hyper_table: "HyperTable",
    num_band: int,
    den_band: int,
    visualize: bool = False
) -> pd.Series:
    """
    Compute a band ratio for all samples in a HyperTable.

    Parameters
    ----------
    hyper_table : HyperTable
        Input hyperspectral dataset.
    num_band : int
        Column index (or wavelength index) for numerator band.
    den_band : int
        Column index (or wavelength index) for denominator band.
    visualize : bool, optional (default=False)
        If True, show a heatmap of the band ratio.

    Returns
    -------
    pd.Series
        Band ratio values for all samples (length = number of rows).
    """
    if den_band >= hyper_table.bands or num_band >= hyper_table.bands:
        raise ValueError("Band indices out of range.")

    numerator = hyper_table.get_band(num_band)
    denominator = hyper_table.get_band(den_band)

    # Avoid division by zero
    ratio = numerator / np.where(denominator == 0, np.nan, denominator)
    ratio_series = pd.Series(ratio, index=hyper_table.data.index, name=f"BR_{num_band}/{den_band}")

    # Visualization
    if visualize:
        plt.figure(figsize=(8, 6))
        plt.imshow(ratio.reshape(-1, 1), cmap="viridis", aspect="auto")
        plt.colorbar(label=f"Band Ratio {num_band}/{den_band}")
        plt.title("Band Ratio Heatmap")
        plt.xlabel("Band Ratio")
        plt.ylabel("Samples")
        plt.show()

    return ratio_series

def anova_f_test(
    hyper_table: "HyperTable",
    labels: np.ndarray,
    top_k: int = None,
    visualize: bool = False
) -> pd.DataFrame:
    """
    Perform ANOVA F-test to rank spectral bands by discriminative power.

    Parameters
    ----------
    hyper_table : HyperTable
        Input hyperspectral dataset.
    labels : np.ndarray
        Class labels for each sample (length must match number of rows).
    top_k : int, optional
        If specified, return only the top_k bands ranked by F-score.
    visualize : bool, optional (default=False)
        If True, plot F-scores across all bands.

    Returns
    -------
    pd.DataFrame
        DataFrame with band indices, wavelengths, F-scores, and p-values.
    """
    if len(labels) != hyper_table.samples:
        raise ValueError("Length of labels must match number of samples in HyperTable.")

    # Perform ANOVA F-test
    F_scores, p_values = f_classif(hyper_table.data.values, labels)

    # Build results DataFrame
    results = pd.DataFrame({
        "Band_Index": np.arange(hyper_table.bands),
        "Wavelength": hyper_table.wavelengths if hyper_table.wavelengths is not None else np.arange(hyper_table.bands),
        "F_Score": F_scores,
        "p_Value": p_values
    })

    # Sort by F-score
    results = results.sort_values(by="F_Score", ascending=False).reset_index(drop=True)

    # If top_k specified, trim results
    if top_k is not None:
        results = results.head(top_k)

    # Visualization
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.plot(
            np.arange(hyper_table.bands),
            F_scores,
            marker="o",
            linestyle="-",
            color="b"
        )
        plt.title("ANOVA F-test Scores per Band")
        plt.xlabel("Band Index")
        plt.ylabel("F-Score")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    return resultsthe
