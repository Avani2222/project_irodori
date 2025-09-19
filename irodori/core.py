"""
This module defines the HyperTable class with the following functions and properties: __init__, 
shape, samples, bands, get_pixel, set_wavelengths, get_band, summary, and __repr__.
"""
import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any, List


class HyperTable:
    """
    A class to store and manage hyperspectral data in tabular (DataFrame) format.

    Each row corresponds to a sample/pixel and each column corresponds to a spectral band.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        wavelengths: Optional[Union[List[float], np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the HyperTable object.

        Parameters
        ----------
        data : pd.DataFrame
            Hyperspectral data in tabular format (rows = samples, cols = bands).
            Assumes the first column contains labels if wavelengths are provided for rest of the columns.
        wavelengths : list or np.ndarray, optional
            Wavelengths corresponding to columns (excluding labels).
        metadata : dict, optional
            Extra metadata about the dataset.
        
        Raises
        ------
        ValueError
            If wavelengths are provided but do not match number of columns.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("`data` must be a pandas DataFrame.")

        if data.empty:
            raise ValueError("`data` cannot be empty.")
        if data.empty:
            raise ValueError("Cannot initialize HyperTable with an empty DataFrame.")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        self.metadata = metadata or {}

        # Handle completely empty data
        if data.empty:
            self.labels = np.array([])
            self.data = pd.DataFrame()
            self.wavelengths = (
                np.asarray(wavelengths) if wavelengths is not None else np.array([])
            )
            self.spectra = np.empty((0, 0))
            return

        n_cols = data.shape[1]
        if wavelengths is not None:
            wl_array = np.asarray(wavelengths)
            if wl_array.ndim != 1:
                raise ValueError("Wavelengths must be a 1D array.")

    # Case A: wavelengths length equals number of columns -> purely spectral
            if len(wl_array) == n_cols:
                self.labels = None
                self.data = data.copy()

    # Case B: wavelengths length equals cols-1 -> first column is label
            elif len(wl_array) == (n_cols - 1):
                self.labels = data.iloc[:, 0].values
                self.data = data.iloc[:, 1:].copy()

            else:
                raise ValueError(
                f"Number of wavelengths ({len(wl_array)}) does not match "
                f"number of columns ({n_cols}) or number of bands ({n_cols - 1})."
                )

            self.wavelengths = wl_array


        else:
            # No wavelengths provided
            if n_cols == 1:
                # Only labels, no spectral bands
                self.labels = data.iloc[:, 0].values
                self.data = pd.DataFrame()
                self.wavelengths = np.array([])

            else:
                # Assume first column is labels, rest are spectral bands
                self.labels = data.iloc[:, 0].values
                self.data = data.iloc[:, 1:].copy()
                try:
                    # Try parsing column headers as numeric wavelengths
                    self.wavelengths = np.array(self.data.columns, dtype=float)
                except (ValueError, TypeError):
                    # Non-numeric headers → derived features (NDI, PCA, etc.)
                    self.wavelengths = None

        self.spectra = self.data.values  # n_samples x n_bands
        self.metadata = metadata or {}  # n_samples x n_bands# n_samples x n_bands

    @property
    def shape(self) -> tuple:
        """Returns the shape of the table (samples, bands)."""
        return self.data.shape

    @property
    def samples(self) -> int:
        """Returns the number of samples (rows)."""
        return self.data.shape[0]

    @property
    def bands(self) -> int:
        """Returns the number of spectral bands (columns)."""
        return self.data.shape[1]

    def get_pixel(self, index: int) -> np.ndarray:
        """
        Get the spectral signature of a single sample/pixel.

        Parameters
        ----------
        index : int
            Row index of the sample.

        Returns
        -------
        np.ndarray
            Array of spectral values for that pixel.
        """
        return self.data.iloc[index].values

    def set_wavelengths(self, start: float, end: float) -> None:
        """
        Generate and assign evenly spaced wavelengths between start and end.

        Parameters
        ----------
        start : float
            Starting wavelength (e.g., 400 nm).
        end : float
            Ending wavelength (e.g., 1000 nm).
        """
        self.wavelengths = np.linspace(start, end, self.bands)

    def get_band(self, band_index: int) -> np.ndarray:
        """
        Get all sample values for a given spectral band.

        Parameters
        ----------
        band_index : int
            Column index of the spectral band.

        Returns
        -------
        np.ndarray
            Array of values for the selected band.
        """
        return self.data.iloc[:, band_index].values

    def summary(self) -> pd.DataFrame:
        """
        Generate statistical summary of the dataset.

        Returns
        -------
        pd.DataFrame
            Statistical summary (count, mean, std, min, quartiles, max).
        """
        return self.data.describe()

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation of the object."""
        rows, cols = self.shape
        if self.wavelengths is not None:
            w_min = self.wavelengths.min()
            w_max = self.wavelengths.max()
            wl_range_str = f"wavelength_range=[{w_min:.1f} - {w_max:.1f}]nm"
        else:
            wl_range_str = "wavelengths=UNDEFINED"

        label_str = "YES" if self.labels is not None else "NO"
        return f"HyperTable(samples={rows}, bands={cols}, labels={label_str}, {wl_range_str})"

