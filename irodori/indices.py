"""
Module containing functions for calculating indices including compute_ndvi, compute_ndwi, compute_savi, compute_custom_index, 
compute_evi, compute_gndvi, compute_arvi, compute_mndwi, compute_ndsi
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any

from .core import HyperTable

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

def compute_evi(hyper_table: "HyperTable",
                blue_wavelength: float = 470,
                red_wavelength: float = 660,
                nir_wavelength: float = 860,
                G: float = 2.5,
                C1: float = 6.0,
                C2: float = 7.5,
                L: float = 1.0,
                image_shape: tuple = None,
                cmap: str = "Greens") -> np.ndarray:
    """
    Enhanced Vegetation Index (EVI).

    EVI = G * (NIR - RED) / (NIR + C1*RED - C2*BLUE + L)
    """
    if hyper_table.wavelengths is None:
        raise ValueError("Wavelengths are not defined in the HyperTable.")

    blue_idx = np.argmin(np.abs(hyper_table.wavelengths - blue_wavelength))
    red_idx = np.argmin(np.abs(hyper_table.wavelengths - red_wavelength))
    nir_idx = np.argmin(np.abs(hyper_table.wavelengths - nir_wavelength))

    blue_band = hyper_table.get_band(blue_idx)
    red_band = hyper_table.get_band(red_idx)
    nir_band = hyper_table.get_band(nir_idx)

    evi = G * (nir_band - red_band) / (nir_band + C1 * red_band - C2 * blue_band + L + 1e-10)

    if image_shape is not None:
        plt.imshow(evi.reshape(image_shape), cmap=cmap)
        plt.colorbar(label="EVI")
        plt.title("EVI Heatmap")
        plt.axis("off")
        plt.show()

    return evi

def compute_gndvi(hyper_table: "HyperTable",
                  green_wavelength: float = 550,
                  nir_wavelength: float = 860,
                  image_shape: tuple = None,
                  cmap: str = "YlGn") -> np.ndarray:
    """
    Green Normalized Difference Vegetation Index (GNDVI).

    GNDVI = (NIR - GREEN) / (NIR + GREEN)
    """
    if hyper_table.wavelengths is None:
        raise ValueError("Wavelengths are not defined in the HyperTable.")

    green_idx = np.argmin(np.abs(hyper_table.wavelengths - green_wavelength))
    nir_idx = np.argmin(np.abs(hyper_table.wavelengths - nir_wavelength))

    green_band = hyper_table.get_band(green_idx)
    nir_band = hyper_table.get_band(nir_idx)

    gndvi = (nir_band - green_band) / (nir_band + green_band + 1e-10)

    if image_shape is not None:
        plt.imshow(gndvi.reshape(image_shape), cmap=cmap)
        plt.colorbar(label="GNDVI")
        plt.title("GNDVI Heatmap")
        plt.axis("off")
        plt.show()

    return gndvi
python

def compute_arvi(hyper_table: "HyperTable",
                 red_wavelength: float = 660,
                 blue_wavelength: float = 470,
                 nir_wavelength: float = 860,
                 gamma: float = 1.0,
                 image_shape: tuple = None,
                 cmap: str = "YlGn") -> np.ndarray:
    """
    Atmospherically Resistant Vegetation Index (ARVI).

    ARVI = (NIR - (RED - gamma*(BLUE - RED))) / (NIR + (RED - gamma*(BLUE - RED)))
    """
    if hyper_table.wavelengths is None:
        raise ValueError("Wavelengths are not defined in the HyperTable.")

    red_idx = np.argmin(np.abs(hyper_table.wavelengths - red_wavelength))
    blue_idx = np.argmin(np.abs(hyper_table.wavelengths - blue_wavelength))
    nir_idx = np.argmin(np.abs(hyper_table.wavelengths - nir_wavelength))

    red_band = hyper_table.get_band(red_idx)
    blue_band = hyper_table.get_band(blue_idx)
    nir_band = hyper_table.get_band(nir_idx)

    rb_corr = red_band - gamma * (blue_band - red_band)
    arvi = (nir_band - rb_corr) / (nir_band + rb_corr + 1e-10)

    if image_shape is not None:
        plt.imshow(arvi.reshape(image_shape), cmap=cmap)
        plt.colorbar(label="ARVI")
        plt.title("ARVI Heatmap")
        plt.axis("off")
        plt.show()

    return arvi

def compute_mndwi(hyper_table: "HyperTable",
                  green_wavelength: float = 560,
                  swir_wavelength: float = 1650,
                  image_shape: tuple = None,
                  cmap: str = "Blues") -> np.ndarray:
    """
    Modified Normalized Difference Water Index (MNDWI).

    MNDWI = (GREEN - SWIR) / (GREEN + SWIR)
    """
    if hyper_table.wavelengths is None:
        raise ValueError("Wavelengths are not defined in the HyperTable.")

    green_idx = np.argmin(np.abs(hyper_table.wavelengths - green_wavelength))
    swir_idx = np.argmin(np.abs(hyper_table.wavelengths - swir_wavelength))

    green_band = hyper_table.get_band(green_idx)
    swir_band = hyper_table.get_band(swir_idx)

    mndwi = (green_band - swir_band) / (green_band + swir_band + 1e-10)

    if image_shape is not None:
        plt.imshow(mndwi.reshape(image_shape), cmap=cmap)
        plt.colorbar(label="MNDWI")
        plt.title("MNDWI Heatmap")
        plt.axis("off")
        plt.show()

    return mndwi

def compute_ndsi(hyper_table: "HyperTable",
                 swir1_wavelength: float = 1640,
                 swir2_wavelength: float = 2130,
                 image_shape: tuple = None,
                 cmap: str = "Oranges") -> np.ndarray:
    """
    Normalized Difference Soil Index (NDSI).

    NDSI = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
    """
    if hyper_table.wavelengths is None:
        raise ValueError("Wavelengths are not defined in the HyperTable.")

    swir1_idx = np.argmin(np.abs(hyper_table.wavelengths - swir1_wavelength))
    swir2_idx = np.argmin(np.abs(hyper_table.wavelengths - swir2_wavelength))

    swir1_band = hyper_table.get_band(swir1_idx)
    swir2_band = hyper_table.get_band(swir2_idx)

    ndsi = (swir1_band - swir2_band) / (swir1_band + swir2_band + 1e-10)

    if image_shape is not None:
        plt.imshow(ndsi.reshape(image_shape), cmap=cmap)
        plt.colorbar(label="NDSI")
        plt.title("NDSI Heatmap")
        plt.axis("off")
        plt.show()

    return ndsi



