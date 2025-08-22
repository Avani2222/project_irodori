import pandas as pd
from typing import Optional, Union, Dict, Any, List
import numpy as np

from .hypertable import HyperTable   # assuming your class is in hypertable.py


def load_csv(filepath: str,
             wavelengths: Optional[Union[List[float], np.ndarray]] = None,
             metadata: Optional[Dict[str, Any]] = None,
             header: bool = True) -> HyperTable:
    """
    Load hyperspectral data from a CSV file into a HyperTable object.

    Parameters
    ----------
    filepath : str
        Path to the CSV file. Rows = samples, Columns = spectral bands.
    wavelengths : list or np.ndarray, optional
        Explicit list of wavelengths. If None, CSV column names will be used.
    metadata : dict, optional
        Extra metadata (e.g., file source, sensor type).
    header : bool, default=True
        Whether the CSV file has a header row. If False, columns will be auto-numbered.

    Returns
    -------
    HyperTable
        A HyperTable object containing the loaded data.
    """
    df = pd.read_csv(filepath, header=0 if header else None)
    return HyperTable(data=df, wavelengths=wavelengths, metadata=metadata)


def save_csv(hyper_table: HyperTable, filepath: str, include_header: bool = True):
    """
    Save a HyperTable object to a CSV file.

    Parameters
    ----------
    hyper_table : HyperTable
        The HyperTable object to be saved.
    filepath : str
        Path to save the CSV file.
    include_header : bool, default=True
        Whether to include column names (wavelengths) in the CSV.
    """
    if hyper_table.data is None:
        raise ValueError("No data to save in HyperTable.")

    hyper_table.data.to_csv(filepath, index=False, header=include_header)
