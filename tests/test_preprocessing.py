# test_preprocessing.py
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from irodori.core import HyperTable
import irodori.preprocessing as pp


# ------------------------------
# Fixtures
# ------------------------------
@pytest.fixture
def small_ht():
    np.random.seed(42)
    data = pd.DataFrame(np.random.rand(5, 10), columns=[f"band_{i}" for i in range(10)])
    labels = np.array([0, 1, 0, 1, 0])
    wavelengths = np.linspace(400, 700, 10)
    return HyperTable(data=pd.concat([pd.Series(labels, name="label"), data], axis=1),
                      wavelengths=wavelengths,
                      metadata={"test": True})


# ------------------------------
# Tests
# ------------------------------
def test_minmax_scale(small_ht):
    ht_scaled = pp.minmax_scale(small_ht, feature_range=(0, 1), axis=0)
    assert isinstance(ht_scaled, HyperTable)
    assert_allclose(ht_scaled.data.min().min(), 0.0, rtol=1e-6, atol=1e-6)
    assert_allclose(ht_scaled.data.max().max(), 1.0, rtol=1e-6, atol=1e-6)


def test_standardize(small_ht):
    ht_std = pp.standardize(small_ht, axis=0)
    assert isinstance(ht_std, HyperTable)
    means = ht_std.data.mean(axis=0).values
    assert np.allclose(means, 0, atol=1e-7, rtol=1e-7)


def test_vector_normalize(small_ht):
    ht_norm = pp.vector_normalize(small_ht)
    norms = np.linalg.norm(ht_norm.data.values, axis=1)
    assert np.allclose(norms, 1, atol=1e-7)


def test_apply_savgol_filter(small_ht):
    ht_smooth = pp.apply_savgol_filter(small_ht, window_length=5, polyorder=2)
    assert isinstance(ht_smooth, HyperTable)


def test_band_average(small_ht):
    ht_avg = pp.band_average(small_ht, window_size=2)
    expected_bands = small_ht.bands // 2  # or ceil if implementation rounds up
    assert ht_avg.bands == expected_bands
    assert isinstance(ht_avg, HyperTable)


def test_pca_denoise(small_ht):
    ht_pca = pp.pca_denoise(small_ht, n_components=3)
    assert ht_pca.data.shape == small_ht.data.shape


def test_remove_noisy_bands_variance(small_ht):
    ht_filt = pp.remove_noisy_bands(small_ht, variance_threshold=0.0)
    assert isinstance(ht_filt, HyperTable)
    assert ht_filt.bands <= small_ht.bands


def test_select_wavelength_range(small_ht):
    ht_sel = pp.select_wavelength_range(small_ht, [(450, 600)])
    assert isinstance(ht_sel, HyperTable)
    assert np.all((ht_sel.wavelengths >= 450) & (ht_sel.wavelengths <= 600))


def test_mahalanobis_distance(small_ht):
    dists = pp.mahalanobis_distance(small_ht)
    assert len(dists) == small_ht.samples
    assert np.all(dists >= 0)


def test_isolation_forest_filter(small_ht):
    ht_filt, mask = pp.isolation_forest_filter(small_ht, return_mask=True)
    assert isinstance(ht_filt, HyperTable)
    assert mask.shape[0] == small_ht.samples


def test_correct_baseline(small_ht):
    ht_corr = pp.correct_baseline(small_ht, lam=1e3, p=0.1, niter=5)
    assert isinstance(ht_corr, HyperTable)


def test_normalize_vector(small_ht):
    ht_norm = pp.normalize_vector(small_ht)
    norms = np.linalg.norm(ht_norm.data.values, axis=1)
    assert np.allclose(norms[norms > 0], 1, atol=1e-7)


def test_spectral_shift(small_ht):
    ht_shift = pp.spectral_shift(small_ht, shift=1.5)
    assert isinstance(ht_shift, HyperTable)


def test_mixup(small_ht):
    ht_mix = pp.mixup(small_ht, alpha=0.5, n_samples=3)
    assert ht_mix.samples == 3


def test_spectral_derivative(small_ht):
    ht_deriv = pp.spectral_derivative(small_ht, order=1, window_length=5, polyorder=2)
    assert isinstance(ht_deriv, HyperTable)


def test_add_noise(small_ht):
    ht_noisy = pp.add_noise(small_ht, noise_level=0.05)
    assert isinstance(ht_noisy, HyperTable)


def test_spectral_index(small_ht):
    ht_idx = pp.spectral_index(small_ht, 0, 1)
    assert ht_idx.bands == 1


def test_resample_spectra(small_ht):
    new_wls = np.linspace(400, 700, 5)
    ht_res = pp.resample_spectra(small_ht, new_wls)
    assert ht_res.bands == 5


def test_estimate_snr(small_ht):
    snr = pp.estimate_snr(small_ht)
    assert snr.shape[0] == small_ht.samples


def test_multiplicative_scatter_correction(small_ht):
    ht_msc = pp.multiplicative_scatter_correction(small_ht)
    assert isinstance(ht_msc, HyperTable)


def test_standard_normal_variate(small_ht):
    ht_snv = pp.standard_normal_variate(small_ht)
    assert isinstance(ht_snv, HyperTable)


def test_savgol_first_derivative(small_ht):
    ht_fd = pp.savgol_first_derivative(small_ht, window_length=5, polyorder=2)
    assert isinstance(ht_fd, HyperTable)


def test_savgol_second_derivative(small_ht):
    ht_sd = pp.savgol_second_derivative(small_ht, window_length=5, polyorder=2)
    assert isinstance(ht_sd, HyperTable)


def test_baseline_als_function():
    y = np.linspace(1, 10, 50) + np.sin(np.linspace(0, 3, 50))
    baseline = pp.baseline_als(y, lam=1e2, p=0.01, niter=5)
    assert baseline.shape == y.shape


def test_apply_baseline_correction(small_ht):
    ht_corr = pp.apply_baseline_correction(small_ht, lam=1e3, p=0.1, niter=5)
    assert isinstance(ht_corr, HyperTable)


def test_resample_wavelengths(small_ht):
    new_wls = np.linspace(400, 700, 6)
    ht_res = pp.resample_wavelengths(small_ht, new_wls)
    assert ht_res.bands == 6


def test_remove_outliers_zscore(small_ht):
    ht_clean = pp.remove_outliers_zscore(small_ht, threshold=3.0)
    assert isinstance(ht_clean, HyperTable)
