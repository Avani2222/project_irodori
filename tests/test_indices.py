# test_indices.py
import numpy as np
import pandas as pd
import pytest
from irodori.core import Hypertable

from irodori.indices import (
    compute_ndvi,
    compute_ndwi,
    compute_savi,
    compute_custom_index,
    compute_evi,
    compute_gndvi,
    compute_arvi,
    compute_mndwi,
    compute_ndsi,
)

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def sample_hyper_table():
    """
    Create a simple HyperTable with synthetic spectral bands.
    Columns represent bands at 470, 550, 660, 670, 800, 860, 1640, 1650, 2130 nm.
    """
    wavelengths = np.array([470, 550, 560, 660, 670, 800, 860, 1640, 1650, 2130])
    # 5 samples × 10 bands
    data = np.tile(np.linspace(0.1, 1.0, len(wavelengths)), (5, 1))
    df = pd.DataFrame(np.column_stack([np.arange(5), data]))  # first col = labels
    return HyperTable(df, wavelengths=wavelengths)


# -----------------------------
# Tests
# -----------------------------
def test_ndvi(sample_hyper_table):
    ndvi = compute_ndvi(sample_hyper_table)
    assert isinstance(ndvi, np.ndarray)
    assert ndvi.shape[0] == sample_hyper_table.samples
    assert np.all((ndvi >= -1) & (ndvi <= 1))


def test_ndwi(sample_hyper_table):
    ndwi = compute_ndwi(sample_hyper_table)
    assert ndwi.shape[0] == sample_hyper_table.samples
    assert np.all((ndwi >= -1) & (ndwi <= 1))


def test_savi(sample_hyper_table):
    savi = compute_savi(sample_hyper_table, L=0.5)
    assert savi.shape[0] == sample_hyper_table.samples
    assert np.all((savi >= -1) & (savi <= 1.5))  # SAVI can slightly exceed NDVI range


def test_custom_index(sample_hyper_table):
    formula = "(NIR - RED) / (NIR + RED)"
    band_map = {"RED": 660, "NIR": 800}
    custom = compute_custom_index(sample_hyper_table, formula, band_map)
    ndvi = compute_ndvi(sample_hyper_table)
    np.testing.assert_allclose(custom, ndvi, rtol=1e-6)


def test_evi(sample_hyper_table):
    evi = compute_evi(sample_hyper_table)
    assert evi.shape[0] == sample_hyper_table.samples
    # EVI often ranges between -1 and +3
    assert np.all((evi > -2) & (evi < 3))


def test_gndvi(sample_hyper_table):
    gndvi = compute_gndvi(sample_hyper_table)
    assert gndvi.shape[0] == sample_hyper_table.samples
    assert np.all((gndvi >= -1) & (gndvi <= 1))


def test_arvi(sample_hyper_table):
    arvi = compute_arvi(sample_hyper_table)
    assert arvi.shape[0] == sample_hyper_table.samples
    assert np.all((arvi >= -1) & (arvi <= 1))


def test_mndwi(sample_hyper_table):
    mndwi = compute_mndwi(sample_hyper_table)
    assert mndwi.shape[0] == sample_hyper_table.samples
    assert np.all((mndwi >= -1) & (mndwi <= 1))


def test_ndsi(sample_hyper_table):
    ndsi = compute_ndsi(sample_hyper_table)
    assert ndsi.shape[0] == sample_hyper_table.samples
    assert np.all((ndsi >= -1) & (ndsi <= 1))


# -----------------------------
# Error Handling
# -----------------------------
def test_missing_wavelengths_raises():
    df = pd.DataFrame(np.column_stack([np.arange(3), np.random.rand(3, 4)]))
    ht = HyperTable(df, wavelengths=None)
    with pytest.raises(ValueError):
        compute_ndvi(ht)
