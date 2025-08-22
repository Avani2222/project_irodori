import pandas as pd
import numpy as np

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


