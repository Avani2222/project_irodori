# test_similarity.py
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from irodori.core import HyperTable
import irodori.similarity as sim


# ------------------------------
# Fixtures
# ------------------------------
@pytest.fixture
@pytest.fixture
def small_ht():
    np.random.seed(0)
    samples, bands = 5, 6
    labels = np.random.randint(0, 2, size=samples)   # binary labels
    spectra = np.random.rand(samples, bands)

    # Build DataFrame with label column first
    df = pd.DataFrame(
        np.column_stack([labels, spectra]),
        columns=["Label"] + [f"band_{i}" for i in range(bands)]
    )

    return HyperTable(
        data=df,
        wavelengths=np.arange(bands),   # matches 6 spectral bands
        metadata={"source": "test"}
    )


@pytest.fixture
def reference():
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


# ------------------------------
# Tests
# ------------------------------
def test_spectral_angle_mapper(small_ht, reference):
    angles = sim.spectral_angle_mapper(small_ht, reference, visualize=False)
    assert angles.shape == (small_ht.samples,)
    assert np.all((angles >= 0) & (angles <= 180))

    with pytest.raises(ValueError):
        sim.spectral_angle_mapper(small_ht, np.array([1, 2, 3]), visualize=False)


def test_euclidean_distance(small_ht, reference):
    dists = sim.euclidean_distance(small_ht, reference, visualize=False)
    assert dists.shape == (small_ht.samples,)
    assert np.all(dists >= 0)

    with pytest.raises(ValueError):
        sim.euclidean_distance(small_ht, np.array([1, 2, 3]), visualize=False)


def test_spectral_information_divergence(small_ht, reference):
    sid = sim.spectral_information_divergence(small_ht, reference, visualize=False)
    assert sid.shape == (small_ht.samples,)
    assert np.all(sid >= 0)

    with pytest.raises(ValueError):
        sim.spectral_information_divergence(small_ht, np.array([1, 2]), visualize=False)


def test_band_ratio(small_ht):
    ratios = sim.band_ratio(small_ht, 0, 1, visualize=False)
    assert ratios.shape == (small_ht.samples,)
    assert np.all(np.isfinite(ratios))

    with pytest.raises(ValueError):
        sim.band_ratio(small_ht, 0, 99, visualize=False)


def test_spectral_correlation(small_ht, reference):
    corrs = sim.spectral_correlation(small_ht, reference, visualize=False)
    assert corrs.shape == (small_ht.samples,)
    assert np.all((corrs >= -1) & (corrs <= 1))


def test_cosine_similarity(small_ht, reference):
    sims = sim.cosine_similarity(small_ht, reference, visualize=False)
    assert sims.shape == (small_ht.samples,)
    assert np.all((sims >= -1) & (sims <= 1))


def test_sam_heatmap(reference):
    np.random.seed(0)
    cube = np.random.rand(4, 4, 6)  # 4x4 image, 6 bands
    heatmap = sim.sam_heatmap(cube, reference, in_degrees=True, figsize=(2, 2))
    assert heatmap.shape == (4, 4)
    assert np.all(heatmap >= 0)

    with pytest.raises(ValueError):
        sim.sam_heatmap(cube, np.array([1, 2, 3]))


def test_similarity_dashboard(small_ht, reference):
    df = sim.similarity_dashboard(small_ht, reference, figsize=(4, 3))
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"SAM", "Euclidean", "Correlation", "Cosine"}
    assert df.shape[0] == small_ht.samples
