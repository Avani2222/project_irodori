import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any, List


class HyperTable:
    """
    A class to store and manage hyperspectral data in tabular (DataFrame) format.

    This object represents
    hyperspectral data where each row corresponds to a sample/pixel and each
    column corresponds to a spectral band.

    Attributes
    ----------
    labels : np.ndarray or None
        Array of labels for each sample (first column of DataFrame if present).
    data : pd.DataFrame
        Hyperspectral dataset with samples as rows and bands as columns.
    wavelengths : np.ndarray or None
        Array of wavelength values corresponding to each band (column).
    metadata : dict
        Dictionary to store additional information (e.g., file source, sensor type).
    """

    def __init__(self,
                 data: pd.DataFrame,
                 wavelengths: Optional[Union[List[float], np.ndarray]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the HyperTable object.

        Parameters
        ----------
        data : pd.DataFrame
            Hyperspectral data in tabular format (rows = samples, cols = bands).
            Assumes the first column contains labels.
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
            raise ValueError("Input 'data' must be a pandas DataFrame.")

        # Separate labels (first column)
        self.labels = data.iloc[:, 0].values
        self.data = data.iloc[:, 1:]

        # Wavelengths validation
        if wavelengths is not None:
            wl_array = np.asarray(wavelengths)
            if wl_array.ndim != 1:
                raise ValueError("Wavelengths must be a 1D array.")
            if len(wl_array) != self.data.shape[1]:
                raise ValueError(
                    f"Number of wavelengths ({len(wl_array)}) does not match number of bands ({self.data.shape[1]})."
                )
            self.wavelengths = wl_array
        else:
            self.wavelengths = None

        self.metadata = metadata if metadata is not None else {}

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

        return f"HyperTable(samples={rows}, bands={cols}, labels=YES, {wl_range_str})"
