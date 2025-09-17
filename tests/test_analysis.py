# test_analysis_module.py
import pytest
import numpy as np
import pandas as pd
from irodori.core import HyperTable
from irodori.analysis import (
    first_derivative,
    second_derivative,
    smooth_spectra,
    plot_spectral_signatures,
    plot_pixel_spectrum,
    plot_average_spectrum,
    plot_band_image,
    plot_band_histograms,
    anova_f_test,
    mutual_info_band_selection,
    band_correlation,
    spectral_entropy,
    cluster_bands,
    spectral_snr,
    spectral_peaks,
    spectral_angle_mapper,
    spectral_information_divergence,
    euclidean_distance,
    band_ratio,
    continuum_removal,
    plot_pca,
    pca_outlier_detection,
)

# ---------------------------
# Fixtures
# ---------------------------

@pytest.fixture
def mock_hyper_table():
    data = pd.DataFrame({
        400: [0.1, 0.2, 0.3, 0.4],
        450: [0.15, 0.25, 0.35, 0.45],
        500: [0.2, 0.3, 0.4, 0.5],
        550: [0.25, 0.35, 0.45, 0.55],
        600: [0.3, 0.4, 0.5, 0.6],
        650: [0.35, 0.45, 0.55, 0.65],
        700: [0.4, 0.5, 0.6, 0.7],
    })
    wavelengths = [400, 450, 500, 550, 600, 650, 700]
    return HyperTable(
        data=data,
        samples=data.shape[0],      # 4 samples
        bands=data.shape[1],        # 7 bands
        labels=np.array([0, 1, 0, 1]),  # numeric labels for testing
        wavelengths=wavelengths,
        metadata={"test": True}
    )

@pytest.fixture
def reference_spectrum(mock_hyper_table):
    # returns first sample's spectrum as numpy array
    return mock_hyper_table.data.iloc[0].values

@pytest.fixture
def image_shape(mock_hyper_table):
    # Must match total samples = 4
    return (2, 2)


# ---------------------------
# Tests
# ---------------------------

def test_first_derivative(mock_hyper_table):
    deriv_ht = first_derivative(mock_hyper_table, show_plot=False)
    assert deriv_ht.data.shape == mock_hyper_table.data.shape

def test_second_derivative(mock_hyper_table):
    deriv2_ht = second_derivative(mock_hyper_table)
    assert deriv2_ht.data.shape[1] == mock_hyper_table.bands

def test_smooth_spectra(mock_hyper_table):
    smooth_ht = smooth_spectra(mock_hyper_table, visualize=False)
    assert smooth_ht.data.shape == mock_hyper_table.data.shape

def test_plot_spectral_signatures(mock_hyper_table):
    plot_spectral_signatures(mock_hyper_table, sample_indices=[0, 1], labels=True, title="Test Plot")

def test_plot_pixel_spectrum(mock_hyper_table):
    plot_pixel_spectrum(mock_hyper_table, index=0, show_baseline=True)

def test_plot_average_spectrum(mock_hyper_table):
    plot_average_spectrum(mock_hyper_table, by_label=True)

def test_plot_band_image(mock_hyper_table, image_shape):
    plot_band_image(mock_hyper_table, band_index=0, image_shape=image_shape)

def test_plot_band_histograms(mock_hyper_table):
    plot_band_histograms(mock_hyper_table, band_indices=[0, 1, 2])

def test_anova_f_test(mock_hyper_table):
    result = anova_f_test(mock_hyper_table, top_k=3, visualize=False)
    assert len(result) == 3

def test_mutual_info_band_selection(mock_hyper_table):
    result = mutual_info_band_selection(mock_hyper_table, top_k=3)
    assert len(result) == 3

def test_band_correlation(mock_hyper_table):
    corr = band_correlation(mock_hyper_table)
    assert corr.shape == (mock_hyper_table.bands, mock_hyper_table.bands)

def test_spectral_entropy(mock_hyper_table):
    ent = spectral_entropy(mock_hyper_table, visualize=False)
    assert ent.shape[0] == mock_hyper_table.samples


def test_cluster_bands(mock_hyper_table):
    clusters = cluster_bands(mock_hyper_table, n_clusters=3)
    assert isinstance(clusters, dict)
    assert sum(len(v) for v in clusters.values()) == mock_hyper_table.bands

def test_spectral_snr(mock_hyper_table):
    snr = spectral_snr(mock_hyper_table, visualize=False)
    assert snr.shape[0] == mock_hyper_table.samples

def test_spectral_peaks(mock_hyper_table):
    peaks = spectral_peaks(mock_hyper_table, sample_indices=[0, 1], visualize=False)
    assert isinstance(peaks, dict)

def test_spectral_angle_mapper(mock_hyper_table, reference_spectrum):
    sam = spectral_angle_mapper(mock_hyper_table, reference_spectrum, visualize=False)
    assert sam.shape[0] == mock_hyper_table.samples

def test_spectral_information_divergence(mock_hyper_table, reference_spectrum):
    sid = spectral_information_divergence(mock_hyper_table, reference_spectrum, visualize=False)
    assert sid.shape[0] == mock_hyper_table.samples

def test_euclidean_distance(mock_hyper_table, reference_spectrum):
    dist = euclidean_distance(mock_hyper_table, reference_spectrum, visualize=False)
    assert dist.shape[0] == mock_hyper_table.samples

def test_band_ratio(mock_hyper_table):
    ratio = band_ratio(mock_hyper_table, band1=0, band2=1, visualize=False)
    assert ratio.shape[0] == mock_hyper_table.samples

def test_continuum_removal(mock_hyper_table):
    cr = continuum_removal(mock_hyper_table, sample_index=0, visualize=False)
    assert cr.shape[0] == mock_hyper_table.bands

def test_plot_pca(mock_hyper_table):
    plot_pca(mock_hyper_table, n_components=3)

def test_pca_outlier_detection(mock_hyper_table):
    outliers = pca_outlier_detection(mock_hyper_table, n_components=3, threshold=1.5, visualize=False)
    assert outliers.shape[0] == mock_hyper_table.samples
    assert outliers.dtype == bool
