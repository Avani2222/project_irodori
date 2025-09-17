"""
  Module containing functions for analysis of irodori data including first_derivative, second_derivative, smooth_spectra, 
  plot_spectral_signatures, plot_pca, plot_pixel_spectrum, plot_average_spectrum, plot_band_image, plot_band_histograms, 
  anova_f_test, mutual_info_band_selection, band_correlation, spectral_entropy, cluster_bands, spectral_snr, 
  spectral_peaks, spectral_angle_mapper, spectral_information_divergence, euclidean_distance, band_ratio, 
  continuum_removal, pca_outlier_detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from .core import HyperTable
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


def first_derivative(
    hyper_table: "HyperTable",
    show_plot: bool = False,
    sample_indices: list = None
) -> "HyperTable":
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
        derivative_data = np.hstack(
            [derivative_data, derivative_data[:, -1][:, None]]
        )

    derivative_df = pd.DataFrame(
        derivative_data, index=hyper_table.data.index
    )
    derivative_ht = HyperTable(
        derivative_df,
        wavelengths=hyper_table.wavelengths,
        metadata={**hyper_table.metadata, "processed": "first_derivative"},
    )

    if show_plot:
        if sample_indices is None:
            sample_indices = [0]
        wl = (
            hyper_table.wavelengths
            if hyper_table.wavelengths is not None
            else np.arange(hyper_table.bands)
        )
        plt.figure(figsize=(8, 5))
        for idx in sample_indices:
            plt.plot(
                wl,
                hyper_table.get_pixel(idx),
                label=f"Original {idx}",
                alpha=0.6,
            )
            plt.plot(
                wl,
                derivative_ht.get_pixel(idx),
                "--",
                label=f"Derivative {idx}",
            )
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
            lbl = f"Sample {idx} ({hyper_table.labels[idx]})"
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

def plot_pca(hyper_table: "HyperTable",
             n_components: int = 3,
             figsize: tuple = (7, 5)) -> None:
    """
    Visualize PCA results for hyperspectral data.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    n_components : int, default=3
        Number of components to visualize (must be >= 2).
    figsize : tuple, default=(7, 5)
        Figure size for plots.

    Returns
    -------
    None
        Displays explained variance and scatter plots.
    """
    X = hyper_table.data.values.astype(float)

    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # 1. Explained variance plot
    plt.figure(figsize=figsize)
    plt.plot(np.cumsum(pca.explained_variance_ratio_[:n_components]) * 100,
             marker="o", color="darkblue")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.title("PCA Explained Variance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. Scatter plot (PC1 vs PC2)
    if n_components >= 2:
        plt.figure(figsize=figsize)
        plt.scatter(X_pca[:, 0], X_pca[:, 1],
                    c="darkgreen", alpha=0.7, edgecolor="k")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Scatter Plot (PC1 vs PC2)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # 3. Optional 3D scatter (PC1 vs PC2 vs PC3)
    if n_components >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                   c="teal", alpha=0.7, edgecolor="k")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("PCA 3D Scatter Plot")
        plt.tight_layout()
        plt.show()

import matplotlib.pyplot as plt


def plot_pixel_spectrum(ht: HyperTable, index: int, show_baseline: bool = False) -> None:
    """
    Plot the spectral signature of a single pixel/sample.

    Parameters
    ----------
    ht : HyperTable
        Hyperspectral dataset.
    index : int
        Row index of the sample to plot.
    show_baseline : bool, default=False
        If True, also plot the mean spectrum for reference.
    """
    spectrum = ht.get_pixel(index)
    
    if ht.wavelengths is not None:
        x_axis = ht.wavelengths
        xlabel = "Wavelength (nm)"
    else:
        x_axis = range(ht.bands)
        xlabel = "Band Index"
    
    plt.figure(figsize=(8, 4))
    plt.plot(x_axis, spectrum, label=f"Pixel {index}", color="blue")

    if show_baseline:
        mean_spec = ht.data.mean(axis=0).values
        plt.plot(x_axis, mean_spec, label="Mean Spectrum", color="red", linestyle="--")
    
    plt.xlabel(xlabel)
    plt.ylabel("Intensity")
    plt.title(f"Spectral Signature (Pixel {index})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_average_spectrum(ht: HyperTable, by_label: bool = False) -> None:
    """
    Plot the average spectrum of the dataset.

    Parameters
    ----------
    ht : HyperTable
        Hyperspectral dataset.
    by_label : bool, default=False
        If True, plots average spectrum separately for each label.
        If False, plots a single overall average.
    """
    if ht.wavelengths is not None:
        x_axis = ht.wavelengths
        xlabel = "Wavelength (nm)"
    else:
        x_axis = np.arange(ht.bands)
        xlabel = "Band Index"

    plt.figure(figsize=(8, 4))

    if by_label:
        unique_labels = np.unique(ht.labels)
        for label in unique_labels:
            spectra = ht.data[ht.labels == label]
            mean_spec = spectra.mean(axis=0).values
            plt.plot(x_axis, mean_spec, label=f"Label {label}")
    else:
        mean_spec = ht.data.mean(axis=0).values
        plt.plot(x_axis, mean_spec, label="Average Spectrum", color="blue")

    plt.xlabel(xlabel)
    plt.ylabel("Intensity")
    plt.title("Average Spectrum" + (" (per Label)" if by_label else ""))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_band_image(ht: HyperTable, band_index: int, image_shape: tuple, cmap: str = "viridis") -> None:
    """
    Plot a 2D image of a single spectral band.

    Parameters
    ----------
    ht : HyperTable
        Hyperspectral dataset.
    band_index : int
        Index of the band to plot.
    image_shape : tuple
        Shape of the original image (rows, cols).
    cmap : str, default="viridis"
        Colormap for visualization.
    """
    if band_index < 0 or band_index >= ht.bands:
        raise ValueError(f"band_index must be in [0, {ht.bands-1}]")

    # Extract values for the band
    band_values = ht.get_band(band_index)

    # Reshape into 2D image
    try:
        band_image = band_values.reshape(image_shape)
    except ValueError:
        raise ValueError("image_shape does not match the number of samples in HyperTable.")

    # Define x-axis label
    if ht.wavelengths is not None:
        band_label = f"Wavelength: {ht.wavelengths[band_index]:.1f} nm"
    else:
        band_label = f"Band {band_index}"

    plt.figure(figsize=(5, 5))
    plt.imshow(band_image, cmap=cmap)
    plt.title(f"Band Image ({band_label})")
    plt.axis("off")
    plt.colorbar(label="Intensity")
    plt.tight_layout()
    plt.show()

def band_correlation(hyper_table: "HyperTable",
                     method: str = "pearson",
                     figsize: tuple = (10, 6)) -> pd.DataFrame:
    """
    Compute correlation between spectral bands.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    method : str, default="pearson"
        Correlation method: 'pearson', 'spearman', or 'kendall'.
    figsize : tuple, default=(10, 6)
        Figure size for heatmap visualization.

    Returns
    -------
    pd.DataFrame
        Correlation matrix between bands.
    """
    corr_matrix = hyper_table.data.corr(method=method)

    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, cbar_kws={"label": f"{method} correlation"})
    plt.title(f"Band-to-Band Correlation ({method.title()})")
    plt.tight_layout()
    plt.show()

    return corr_matrix

def mutual_info_band_selection(hyper_table: "HyperTable",
                               top_k: int = 10,
                               figsize: tuple = (10, 5)) -> pd.DataFrame:
    """
    Rank spectral bands using Mutual Information with class labels.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container with labels.
    top_k : int, default=10
        Number of top-ranked bands to return.
    figsize : tuple, default=(10, 5)
        Plot size for visualization.

    Returns
    -------
    pd.DataFrame
        Top ranked bands with MI scores.
    """
    if hyper_table.labels is None:
        raise ValueError("Labels are required for Mutual Information.")

    X = hyper_table.data.values
    y = hyper_table.labels

    mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)

    result = pd.DataFrame({
        "Band_Index": np.arange(hyper_table.bands),
        "Wavelength": (hyper_table.wavelengths if hyper_table.wavelengths is not None
                       else np.arange(hyper_table.bands)),
        "MI_Score": mi_scores
    }).sort_values(by="MI_Score", ascending=False)

    plt.figure(figsize=figsize)
    plt.plot(result["Wavelength"], result["MI_Score"], color="darkorange", label="MI Score")
    plt.scatter(result.head(top_k)["Wavelength"], result.head(top_k)["MI_Score"],
                color="red", label=f"Top {top_k} Bands")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mutual Information Score")
    plt.title("Mutual Information Ranking of Spectral Bands")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return result.head(top_k)

def spectral_entropy(hyper_table: "HyperTable",
                     visualize: bool = True,
                     figsize: tuple = (10, 4)) -> np.ndarray:
    """
    Compute spectral entropy for each band.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    visualize : bool, default=True
        Whether to visualize entropy across bands.
    figsize : tuple, default=(10, 4)
        Plot size.

    Returns
    -------
    np.ndarray
        Entropy values per band.
    """
    X = hyper_table.data.values
    # Normalize each band into probabilities
    band_probs = (X.T / (X.sum(axis=1) + 1e-12)).T
    entropies = entropy(band_probs.T)

    if visualize:
        wl = hyper_table.wavelengths if hyper_table.wavelengths is not None else np.arange(hyper_table.bands)
        plt.figure(figsize=figsize)
        plt.plot(wl, entropies, color="purple")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Entropy")
        plt.title("Spectral Entropy Across Bands")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return entropies

def cluster_bands(hyper_table: "HyperTable",
                  n_clusters: int = 5,
                  figsize: tuple = (8, 5)) -> dict:
    """
    Cluster spectral bands using KMeans on band similarity.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    n_clusters : int, default=5
        Number of clusters.
    figsize : tuple, default=(8, 5)
        Plot size.

    Returns
    -------
    dict
        Mapping of cluster_id -> band indices.
    """
    X = hyper_table.data.values.T  # transpose: bands × samples
    km = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = km.fit_predict(X)

    cluster_map = {}
    for i in range(n_clusters):
        cluster_map[i] = np.where(clusters == i)[0]

    plt.figure(figsize=figsize)
    plt.scatter(np.arange(hyper_table.bands), [clusters[i] for i in range(hyper_table.bands)],
                c=clusters, cmap="tab10")
    plt.xlabel("Band Index")
    plt.ylabel("Cluster ID")
    plt.title("Clustering of Spectral Bands")
    plt.grid(alpha=0.3)
    plt.show()

    return cluster_map

def spectral_snr(hyper_table: "HyperTable",
                 visualize: bool = True,
                 figsize: tuple = (10, 4)) -> np.ndarray:
    """
    Compute signal-to-noise ratio (SNR) for each band.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    visualize : bool, default=True
        Show SNR plot across bands.
    figsize : tuple, default=(10, 4)
        Plot size.

    Returns
    -------
    np.ndarray
        SNR values for each band.
    """
    X = hyper_table.data.values
    mean_signal = X.mean(axis=0)
    std_noise = X.std(axis=0)
    snr = mean_signal / (std_noise + 1e-12)

    if visualize:
        wl = hyper_table.wavelengths if hyper_table.wavelengths is not None else np.arange(hyper_table.bands)
        plt.figure(figsize=figsize)
        plt.plot(wl, snr, color="darkblue")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("SNR")
        plt.title("Signal-to-Noise Ratio Across Bands")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return snr

def spectral_peaks(hyper_table: "HyperTable",
                   prominence: float = 0.01,
                   visualize: bool = True,
                   sample_indices: list = None,
                   figsize: tuple = (8, 4)) -> dict:
    """
    Detect spectral peaks in hyperspectral spectra.

    Parameters
    ----------
    hyper_table : HyperTable
    prominence : float, default=0.01
        Minimum prominence of peaks.
    visualize : bool, default=True
        Plot spectra with peaks.
    sample_indices : list of int, optional
        Specific samples to plot.
    figsize : tuple

    Returns
    -------
    dict
        sample_index -> peak band indices
    """
    peaks_dict = {}
    if sample_indices is None:
        sample_indices = [0]

    wl = hyper_table.wavelengths if hyper_table.wavelengths is not None else np.arange(hyper_table.bands)

    plt.figure(figsize=figsize)
    for idx in sample_indices:
        spectrum = hyper_table.get_pixel(idx)
        peaks, _ = find_peaks(spectrum, prominence=prominence)
        peaks_dict[idx] = peaks

        if visualize:
            plt.plot(wl, spectrum, label=f"Sample {idx}")
            plt.scatter(wl[peaks], spectrum[peaks], color="red", marker="x", label=f"Peaks {idx}")

    if visualize:
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.title("Spectral Peaks Detection")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return peaks_dict

def pca_outlier_detection(hyper_table: "HyperTable",
                          n_components: int = 5,
                          threshold: float = 2.0,
                          visualize: bool = True,
                          figsize: tuple = (8, 4)) -> np.ndarray:
    """
    Detect outliers based on PCA reconstruction error.

    Parameters
    ----------
    hyper_table : HyperTable
    n_components : int
        Number of PCA components.
    threshold : float
        Z-score threshold for outlier detection.
    visualize : bool, default=True
    figsize : tuple

    Returns
    -------
    np.ndarray
        Boolean mask of outliers (True = outlier)
    """
    X = hyper_table.data.values
    pca = PCA(n_components=n_components)
    X_proj = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_proj)

    errors = np.linalg.norm(X - X_recon, axis=1)
    zscores = (errors - errors.mean()) / errors.std()
    outliers = np.abs(zscores) > threshold

    if visualize:
        plt.figure(figsize=figsize)
        plt.plot(errors, "o-", label="Reconstruction Error")
        plt.axhline(threshold*errors.std() + errors.mean(), color="red", linestyle="--", label="Threshold")
        plt.xlabel("Sample Index")
        plt.ylabel("Error")
        plt.title("PCA-Based Outlier Detection")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return outliers

def smooth_spectra(hyper_table: "HyperTable",
                   window_length: int = 7,
                   polyorder: int = 2,
                   visualize: bool = True,
                   sample_indices: list = None,
                   figsize: tuple = (8, 4)) -> "HyperTable":
    """
    Apply Savitzky–Golay smoothing to spectra.

    Returns new HyperTable with smoothed spectra.
    """
    data = hyper_table.data.values
    smoothed_data = savgol_filter(data, window_length=window_length, polyorder=polyorder, axis=1)
    smoothed_ht = HyperTable(pd.DataFrame(smoothed_data, index=hyper_table.data.index),
                             wavelengths=hyper_table.wavelengths,
                             metadata={**hyper_table.metadata, "processed": "smoothed"})

    if visualize:
        if sample_indices is None:
            sample_indices = [0]
        wl = hyper_table.wavelengths if hyper_table.wavelengths is not None else np.arange(hyper_table.bands)
        plt.figure(figsize=figsize)
        for idx in sample_indices:
            plt.plot(wl, hyper_table.get_pixel(idx), label=f"Original {idx}", alpha=0.5)
            plt.plot(wl, smoothed_ht.get_pixel(idx), label=f"Smoothed {idx}", alpha=0.8)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.title("Savitzky-Golay Smoothing")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return smoothed_ht

def plot_band_histograms(hyper_table: "HyperTable",
                         band_indices: list = None,
                         bins: int = 30,
                         figsize: tuple = (12, 6)) -> None:
    """
    Plot histogram for selected spectral bands.
    """
    if band_indices is None:
        band_indices = list(range(min(5, hyper_table.bands)))

    wl = hyper_table.wavelengths if hyper_table.wavelengths is not None else np.arange(hyper_table.bands)

    plt.figure(figsize=figsize)
    for b in band_indices:
        plt.hist(hyper_table.get_band(b), bins=bins, alpha=0.6, label=f"Band {b} ({wl[b]:.1f} nm)")
    plt.xlabel("Reflectance")
    plt.ylabel("Frequency")
    plt.title("Band-Wise Histograms")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
