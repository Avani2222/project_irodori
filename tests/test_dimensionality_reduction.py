# test_dimensionality.py
import numpy as np
import pandas as pd
import pytest
from irodori.core import Hypertable
from your_module_name import (
    pca_transform, ica_transform, visualize_embedding, nmf_decomposition,
    compute_mutual_info, lda_transform, kernel_pca_transform,
    factor_analysis_transform, isomap_transform, svd_transform,
    spectral_embedding_transform, mds_transform, kmeans_clustering,
    gmm_clustering, variance_per_band, anova_f_test, smooth_spectra
)

# ---- Dummy HyperTable for testing ----
class DummyHyperTable:
    def __init__(self, n_samples=20, n_bands=10, with_labels=True):
        self.data = pd.DataFrame(
            np.random.rand(n_samples, n_bands),
            columns=[f"b{i}" for i in range(n_bands)]
        )
        self.labels = np.random.randint(0, 3, size=n_samples) if with_labels else None
        self.wavelengths = np.linspace(400, 700, n_bands)
        self.bands = n_bands

@pytest.fixture
def ht():
    return DummyHyperTable()

@pytest.fixture
def ht_labels():
    return DummyHyperTable(with_labels=True)

# ---- Tests start ----
def test_pca_transform(ht):
    X = pca_transform(ht, n_components=3, visualize=False)
    assert X.shape == (ht.data.shape[0], 3)

def test_ica_transform(ht):
    X = ica_transform(ht, n_components=3, visualize=False)
    assert X.shape == (ht.data.shape[0], 3)

def test_visualize_embedding_tsne(ht):
    X = visualize_embedding(ht, method="tsne", n_components=2, perplexity=5, random_state=0)
    assert X.shape[1] == 2

def test_visualize_embedding_umap(ht):
    try:
        X = visualize_embedding(ht, method="umap", n_components=2, random_state=0)
        assert X.shape[1] == 2
    except ImportError:
        pytest.skip("UMAP not installed")

def test_visualize_embedding_invalid(ht):
    with pytest.raises(ValueError):
        visualize_embedding(ht, method="badmethod")

def test_nmf_decomposition(ht):
    W, H = nmf_decomposition(ht, n_components=3, visualize=False)
    assert W.shape[0] == ht.data.shape[0]
    assert H.shape[1] == ht.data.shape[1]

def test_compute_mutual_info_classification(ht_labels):
    y = ht_labels.labels
    mi = compute_mutual_info(ht_labels, y, task="classification", plot=False)
    assert len(mi) == ht_labels.data.shape[1]

def test_compute_mutual_info_regression(ht):
    y = np.random.rand(ht.data.shape[0])
    mi = compute_mutual_info(ht, y, task="regression", plot=False)
    assert len(mi) == ht.data.shape[1]

def test_compute_mutual_info_invalid_task(ht):
    with pytest.raises(ValueError):
        compute_mutual_info(ht, np.random.rand(ht.data.shape[0]), task="bad")

def test_lda_transform(ht_labels):
    X = lda_transform(ht_labels, n_components=2, visualize=False)
    assert X.shape[1] <= 2

def test_kernel_pca_transform(ht):
    X = kernel_pca_transform(ht, n_components=2, visualize=False)
    assert X.shape == (ht.data.shape[0], 2)

def test_factor_analysis_transform(ht):
    X = factor_analysis_transform(ht, n_components=2, visualize=False)
    assert X.shape == (ht.data.shape[0], 2)

def test_isomap_transform(ht):
    X = isomap_transform(ht, n_components=2, n_neighbors=3, visualize=False)
    assert X.shape == (ht.data.shape[0], 2)

def test_svd_transform(ht):
    X = svd_transform(ht, n_components=2, visualize=False)
    assert X.shape == (ht.data.shape[0], 2)

def test_spectral_embedding_transform(ht):
    X = spectral_embedding_transform(ht, n_components=2, n_neighbors=3, visualize=False)
    assert X.shape == (ht.data.shape[0], 2)

def test_mds_transform(ht):
    X = mds_transform(ht, n_components=2, visualize=False)
    assert X.shape == (ht.data.shape[0], 2)

def test_kmeans_clustering(ht):
    labels = kmeans_clustering(ht, n_clusters=3, visualize=False)
    assert len(labels) == ht.data.shape[0]

def test_gmm_clustering(ht):
    labels = gmm_clustering(ht, n_components=3, visualize=False)
    assert len(labels) == ht.data.shape[0]

def test_variance_per_band(ht):
    variances = variance_per_band(ht)
    assert len(variances) == ht.data.shape[1]

def test_anova_f_test(ht_labels):
    y = ht_labels.labels
    scores = anova_f_test(ht_labels, y, plot=False)
    assert len(scores) == ht_labels.data.shape[1]

def test_smooth_spectra(ht):
    smoothed = smooth_spectra(ht, window_length=5, polyorder=2)
    assert smoothed.shape == ht.data.shape
