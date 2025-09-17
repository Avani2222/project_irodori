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
    np.random.seed(42)
    samples, bands = 10, 50
    labels = np.random.randint(0, 2, size=samples)
    
    # Generate 50 spectral bands
    spectra = np.random.rand(samples, bands)
    
    # Build DataFrame: first column = labels, next 50 columns = bands
    df = pd.DataFrame(
        np.column_stack([labels, spectra]),
        columns=["Label"] + [f"Band_{i}" for i in range(bands)]
    )

ht = HyperTable(df, wavelengths=np.linspace(400, 1000, bands))
@pytest.fixture
def reference_spectrum(mock_hyper_table):
    return mock_hyper_table.data.values[0]

@pytest.fixture
def image_shape(mock_hyper_table):
    return (2, 5)  # Must match total samples = 10


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
    result = anova_f_test(mock_hyper_table, top_k=5, visualize=False)
    assert len(result) == 5

def test_mutual_info_band_selection(mock_hyper_table):
    result = mutual_info_band_selection(mock_hyper_table, top_k=5)
    assert len(result) == 5

def test_band_correlation(mock_hyper_table):
    corr = band_correlation(mock_hyper_table)
    assert corr.shape == (mock_hyper_table.bands, mock_hyper_table.bands)

def test_spectral_entropy(mock_hyper_table):
    ent = spectral_entropy(mock_hyper_table, visualize=False)
    assert ent.shape[0] == mock_hyper_table.bands

def test_cluster_bands(mock_hyper_table):
    clusters = cluster_bands(mock_hyper_table, n_clusters=3)
    assert isinstance(clusters, dict)
    assert sum(len(v) for v in clusters.values()) == mock_hyper_table.bands

def test_spectral_snr(mock_hyper_table):
    snr = spectral_snr(mock_hyper_table, visualize=False)
    assert snr.shape[0] == mock_hyper_table.bands

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
