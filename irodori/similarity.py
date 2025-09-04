import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.interpolate import interp1d
from typing import Optional, Tuple, List
from .core import HyperTable

# ==============================
# Spectral Angle Mapper (SAM)
# ==============================
def spectral_angle_mapper(hyper_table: HyperTable,
                          reference: np.ndarray,
                          visualize: bool = True,
                          in_degrees: bool = True,
                          figsize: Tuple[int,int] = (8,4)) -> np.ndarray:
    """
    Compute SAM similarity between each spectrum in HyperTable and a reference spectrum.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    reference : np.ndarray
        Reference spectrum (1D array of length = number of bands).
    visualize : bool, default=True
        Whether to plot SAM angles for all samples.
    in_degrees : bool, default=True
        Return angles in degrees (True) or radians (False).
    figsize : tuple, default=(8,4)
        Figure size.

    Returns
    -------
    np.ndarray
        SAM angles per sample (lower = more similar to reference).
    """
    X = hyper_table.data.values
    ref = np.asarray(reference)

    if ref.shape[0] != hyper_table.bands:
        raise ValueError("Reference spectrum must have same number of bands as HyperTable.")

    dot_products = np.sum(X * ref, axis=1)
    norms = np.linalg.norm(X, axis=1) * np.linalg.norm(ref)
    cos_theta = np.clip(dot_products / norms, -1, 1)
    angles = np.arccos(cos_theta)
    if in_degrees:
        angles = np.degrees(angles)

    if visualize:
        plt.figure(figsize=figsize)
        plt.plot(angles, "o-", color="darkorange", label="SAM Angle")
        plt.xlabel("Sample Index")
        plt.ylabel("Angle (degrees)" if in_degrees else "Angle (radians)")
        plt.title("Spectral Angle Mapper (SAM)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return angles


# ==============================
# Euclidean Distance
# ==============================
def euclidean_distance(hyper_table: HyperTable,
                       reference: np.ndarray,
                       visualize: bool = True,
                       figsize: Tuple[int,int] = (8,4)) -> np.ndarray:
    """
    Compute Euclidean distance between each sample spectrum and a reference spectrum.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    reference : np.ndarray
        Reference spectrum (1D array of length = number of bands).
    visualize : bool, default=True
        Plot distances if True.
    figsize : tuple, default=(8,4)
        Figure size.

    Returns
    -------
    np.ndarray
        Euclidean distances per sample (lower = more similar).
    """
    X = hyper_table.data.values
    ref = np.asarray(reference, dtype=float)

    if ref.shape[0] != hyper_table.bands:
        raise ValueError("Reference spectrum must match number of bands.")

    distances = np.linalg.norm(X - ref, axis=1)

    if visualize:
        plt.figure(figsize=figsize)
        plt.plot(distances, "o-", color="purple", label="Euclidean Distance")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        plt.title("Euclidean Distance to Reference Spectrum")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return distances


# ==============================
# Spectral Information Divergence (SID)
# ==============================
def spectral_information_divergence(hyper_table: HyperTable,
                                    reference: np.ndarray,
                                    visualize: bool = True,
                                    figsize: Tuple[int,int] = (8,4)) -> np.ndarray:
    """
    Compute SID (Symmetric KL divergence) between each spectrum and a reference.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    reference : np.ndarray
        Reference spectrum.
    visualize : bool, default=True
        Plot SID scores if True.
    figsize : tuple, default=(8,4)
        Figure size.

    Returns
    -------
    np.ndarray
        SID values per sample (lower = more similar).
    """
    X = hyper_table.data.values
    ref = np.asarray(reference, dtype=float)

    if ref.shape[0] != hyper_table.bands:
        raise ValueError("Reference spectrum must match number of bands.")

    # Normalize to probability distributions
    eps = 1e-12
    X_prob = np.clip(X / X.sum(axis=1, keepdims=True), eps, 1)
    ref_prob = np.clip(ref / ref.sum(), eps, 1)

    sid_scores = np.array([np.sum(p * np.log(p / ref_prob)) + np.sum(ref_prob * np.log(ref_prob / p)) for p in X_prob])

    if visualize:
        plt.figure(figsize=figsize)
        plt.plot(sid_scores, "o-", color="teal", label="SID")
        plt.xlabel("Sample Index")
        plt.ylabel("SID Value")
        plt.title("Spectral Information Divergence (SID)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return sid_scores


# ==============================
# Band Ratio
# ==============================
def band_ratio(hyper_table: HyperTable,
               band1: int,
               band2: int,
               visualize: bool = True,
               figsize: Tuple[int,int] = (8,4)) -> np.ndarray:
    """
    Compute band ratio R_band1 / R_band2 for each sample.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral dataset.
    band1 : int
        Index of numerator band.
    band2 : int
        Index of denominator band.
    visualize : bool, default=True
        Plot ratio values.
    figsize : tuple, default=(8,4)
        Figure size.

    Returns
    -------
    np.ndarray
        Band ratio per sample.
    """
    if band1 >= hyper_table.bands or band2 >= hyper_table.bands:
        raise ValueError("Band indices exceed number of bands.")

    band1_values = hyper_table.get_band(band1).astype(float)
    band2_values = hyper_table.get_band(band2).astype(float)

    eps = 1e-12
    ratios = band1_values / (band2_values + eps)

    if visualize:
        plt.figure(figsize=figsize)
        plt.plot(ratios, "o-", color="darkgreen", label=f"Band Ratio (Band {band1}/{band2})")
        plt.xlabel("Sample Index")
        plt.ylabel("Ratio Value")
        plt.title("Band Ratio Analysis")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return ratios

def spectral_correlation(hyper_table: HyperTable,
                         reference: np.ndarray,
                         visualize: bool = True,
                         figsize: Tuple[int,int] = (8,4)) -> np.ndarray:
    """
    Compute Pearson correlation coefficient between each spectrum and reference.

    Parameters
    ----------
    hyper_table : HyperTable
    reference : np.ndarray
    visualize : bool
    figsize : tuple

    Returns
    -------
    np.ndarray
        Correlation coefficients per sample (closer to 1 = more similar).
    """
    X = hyper_table.data.values
    ref = np.asarray(reference)

    if ref.shape[0] != hyper_table.bands:
        raise ValueError("Reference spectrum must match number of bands.")

    corrs = np.array([np.corrcoef(spectrum, ref)[0, 1] for spectrum in X])

    if visualize:
        plt.figure(figsize=figsize)
        plt.plot(corrs, "o-", color="blue", label="Correlation")
        plt.xlabel("Sample Index")
        plt.ylabel("Correlation Coefficient")
        plt.title("Spectral Correlation with Reference")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return corrs

def cosine_similarity(hyper_table: HyperTable,
                      reference: np.ndarray,
                      visualize: bool = True,
                      figsize: Tuple[int,int] = (8,4)) -> np.ndarray:
    """
    Compute cosine similarity between each spectrum and reference.

    Returns
    -------
    np.ndarray
        Cosine similarity per sample (1 = identical, -1 = opposite).
    """
    X = hyper_table.data.values
    ref = np.asarray(reference)

    if ref.shape[0] != hyper_table.bands:
        raise ValueError("Reference spectrum must match number of bands.")

    norms_X = np.linalg.norm(X, axis=1)
    norm_ref = np.linalg.norm(ref)
    sim = (X @ ref) / (norms_X * norm_ref + 1e-12)

    if visualize:
        plt.figure(figsize=figsize)
        plt.plot(sim, "o-", color="magenta", label="Cosine Similarity")
        plt.xlabel("Sample Index")
        plt.ylabel("Similarity")
        plt.title("Cosine Similarity with Reference")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return sim

def sam_heatmap(image_cube: np.ndarray,
                reference: np.ndarray,
                in_degrees: bool = True,
                figsize: Tuple[int,int] = (6,5)) -> np.ndarray:
    """
    Compute SAM for each pixel in a 3D hyperspectral cube.

    Parameters
    ----------
    image_cube : np.ndarray
        3D hyperspectral image (H x W x Bands)
    reference : np.ndarray
        Reference spectrum
    in_degrees : bool
    figsize : tuple

    Returns
    -------
    np.ndarray
        2D array (H x W) of SAM angles per pixel
    """
    H, W, B = image_cube.shape
    if reference.shape[0] != B:
        raise ValueError("Reference must match number of bands in image cube.")

    ref_norm = np.linalg.norm(reference)
    sam_map = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            pixel = image_cube[i, j, :]
            cos_theta = np.dot(pixel, reference) / (np.linalg.norm(pixel) * ref_norm + 1e-12)
            cos_theta = np.clip(cos_theta, -1, 1)
            angle = np.arccos(cos_theta)
            if in_degrees:
                angle = np.degrees(angle)
            sam_map[i, j] = angle

    plt.figure(figsize=figsize)
    plt.imshow(sam_map, cmap="viridis")
    plt.colorbar(label="SAM Angle (degrees)" if in_degrees else "SAM (radians)")
    plt.title("SAM Heatmap")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return sam_map

def similarity_dashboard(hyper_table: HyperTable,
                         reference: np.ndarray,
                         figsize: Tuple[int,int] = (12,6)) -> pd.DataFrame:
    """
    Compute SAM, Euclidean, Correlation, Cosine similarity and visualize together.

    Returns
    -------
    pd.DataFrame
        DataFrame with similarity metrics per sample.
    """
    sam_vals = spectral_angle_mapper(hyper_table, reference, visualize=False)
    euclid_vals = euclidean_distance(hyper_table, reference, visualize=False)
    corr_vals = spectral_correlation(hyper_table, reference, visualize=False)
    cos_vals = cosine_similarity(hyper_table, reference, visualize=False)

    df = pd.DataFrame({
        "SAM": sam_vals,
        "Euclidean": euclid_vals,
        "Correlation": corr_vals,
        "Cosine": cos_vals
    })

    # Visualization
    plt.figure(figsize=figsize)
    for col in df.columns:
        plt.plot(df[col], "o-", label=col)
    plt.xlabel("Sample Index")
    plt.ylabel("Similarity / Distance")
    plt.title("Similarity Dashboard")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df
