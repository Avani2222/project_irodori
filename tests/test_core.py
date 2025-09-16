import pytest
import pandas as pd
import numpy as np

from your_module_name import HyperTable  # replace with actual filename

# ------------------------------
# Fixtures
# ------------------------------
@pytest.fixture
def sample_dataframe():
    # First column = labels, next columns = bands
    labels = [0, 1, 0]
    bands = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df = pd.DataFrame(np.column_stack([labels, bands]),
                      columns=["label", "b1", "b2", "b3"])
    return df

@pytest.fixture
def hypertable(sample_dataframe):
    return HyperTable(sample_dataframe)

# ------------------------------
# Tests
# ------------------------------
def test_init_and_shape(sample_dataframe):
    ht = HyperTable(sample_dataframe)
    assert ht.samples == 3
    assert ht.bands == 3
    assert ht.shape == (3, 3)


def test_invalid_data():
    with pytest.raises(ValueError):
        HyperTable(np.array([[1, 2], [3, 4]]))


def test_wavelength_validation(sample_dataframe):
    # Wrong length wavelengths
    with pytest.raises(ValueError):
        HyperTable(sample_dataframe, wavelengths=[400, 500])

    # Correct length
    ht = HyperTable(sample_dataframe, wavelengths=[400, 500, 600])
    assert np.allclose(ht.wavelengths, [400, 500, 600])


def test_get_pixel(hypertable):
    pixel = hypertable.get_pixel(0)
    assert np.array_equal(pixel, np.array([1, 2, 3]))


def test_get_band(hypertable):
    band = hypertable.get_band(1)
    assert np.array_equal(band, np.array([2, 5, 8]))


def test_set_wavelengths(hypertable):
    hypertable.set_wavelengths(400, 1000)
    assert len(hypertable.wavelengths) == hypertable.bands
    assert np.isclose(hypertable.wavelengths[0], 400)
    assert np.isclose(hypertable.wavelengths[-1], 1000)


def test_summary(hypertable):
    summary = hypertable.summary()
    assert isinstance(summary, pd.DataFrame)
    assert "mean" in summary.index


def test_repr_with_and_without_wavelengths(hypertable):
    r = repr(hypertable)
    assert "wavelengths=UNDEFINED" in r

    hypertable.set_wavelengths(400, 700)
    r2 = repr(hypertable)
    assert "wavelength_range" in r2
