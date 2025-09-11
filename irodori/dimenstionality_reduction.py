"""
Module for dimensionality reduction tasks on hyperspectral data including pca_transform, ica_transform, visualize_embedding, nmf_decomposition, compute_mutual_info, 
lda_transform, kernel_pca_transform, factor_analysis_transform, isomap_transform, svd_transform, spectral_embedding_transform, mds_transform, kmeans_clustering, 
gmm_clustering, variance_per_band, anova_f_test, smooth_spectra
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.decomposition import NMF
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from .core import Hypertable
from typing import Union, Optional, Literal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import Isomap
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import SpectralEmbedding, MDS
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import f_classif
from scipy.signal import savgol_filter

try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

def pca_transform(hyper_table: "HyperTable",
                  n_components: int = 3,
                  visualize: bool = True,
                  figsize: tuple = (7, 5)) -> np.ndarray:
    """
    Apply PCA to reduce hyperspectral data dimensionality.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    n_components : int, default=3
        Number of principal components to retain.
    visualize : bool, default=True
        If True, plots explained variance ratio and scatter plot of first PCs.
    figsize : tuple, default=(7, 5)
        Size of plots.

    Returns
    -------
    np.ndarray
        PCA-transformed data of shape (samples, n_components).
    """
    X = hyper_table.data.values.astype(float)

    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Visualization
    if visualize:
        # 1. Explained variance plot
        plt.figure(figsize=figsize)
        plt.plot(np.cumsum(pca.explained_variance_ratio_[:n_components]) * 100,
                 marker="o", color="darkblue")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance (%)")
        plt.title("PCA Explained Variance")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 2. Scatter plot (only if at least 2 PCs)
        if n_components >= 2:
            plt.figure(figsize=figsize)
            plt.scatter(X_pca[:, 0], X_pca[:, 1],
                        c="darkgreen", alpha=0.7, edgecolor="k")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("PCA Scatter Plot (PC1 vs PC2)")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    return X_pca



def ica_transform(hyper_table: "HyperTable",
                  n_components: int = 3,
                  visualize: bool = True,
                  figsize: tuple = (7, 5)) -> np.ndarray:
    """
    Apply Independent Component Analysis (ICA) to hyperspectral data.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    n_components : int, default=3
        Number of independent components to extract.
    visualize : bool, default=True
        If True, shows scatter plots of ICA components.
    figsize : tuple, default=(7, 5)
        Size of the plots.

    Returns
    -------
    np.ndarray
        ICA-transformed data of shape (samples, n_components).
    """
    X = hyper_table.data.values.astype(float)

    # Fit ICA
    ica = FastICA(n_components=n_components, random_state=42)
    X_ica = ica.fit_transform(X)

    # Visualization
    if visualize:
        # 2D scatter plot (IC1 vs IC2)
        if n_components >= 2:
            plt.figure(figsize=figsize)
            plt.scatter(X_ica[:, 0], X_ica[:, 1],
                        c="purple", alpha=0.7, edgecolor="k")
            plt.xlabel("IC1")
            plt.ylabel("IC2")
            plt.title("ICA Scatter Plot (IC1 vs IC2)")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        # 3D scatter plot (IC1 vs IC2 vs IC3)
        if n_components >= 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(X_ica[:, 0], X_ica[:, 1], X_ica[:, 2],
                       c="darkorange", alpha=0.7, edgecolor="k")
            ax.set_xlabel("IC1")
            ax.set_ylabel("IC2")
            ax.set_zlabel("IC3")
            ax.set_title("ICA 3D Scatter Plot")
            plt.tight_layout()
            plt.show()

    return X_ica

def visualize_embedding(hyper_table: "HyperTable",
                        method: str = "tsne",
                        n_components: int = 2,
                        perplexity: float = 30,
                        figsize: tuple = (7, 5),
                        random_state: int = 42) -> np.ndarray:
    """
    Visualize hyperspectral data using t-SNE or UMAP.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    method : str, default="tsne"
        Dimensionality reduction method: {"tsne", "umap"}.
    n_components : int, default=2
        Number of output dimensions (2 or 3).
    perplexity : float, default=30
        Perplexity for t-SNE (ignored for UMAP).
    figsize : tuple, default=(7, 5)
        Figure size.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Low-dimensional embedding of shape (samples, n_components).
    """
    X = hyper_table.data.values.astype(float)

    # Choose method
    if method.lower() == "tsne":
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            init="random",
            learning_rate="auto"
        )
    elif method.lower() == "umap":
        if not HAS_UMAP:
            raise ImportError("UMAP is not installed. Please install via `pip install umap-learn`.")
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state
        )
    else:
        raise ValueError("Method must be 'tsne' or 'umap'.")

    embedding = reducer.fit_transform(X)

    # Visualization
    if n_components == 2:
        plt.figure(figsize=figsize)
        if hyper_table.labels is not None:
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                                  c=hyper_table.labels, cmap="tab10", alpha=0.7, edgecolor="k")
            plt.colorbar(scatter, label="Labels")
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1],
                        c="steelblue", alpha=0.7, edgecolor="k")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title(f"{method.upper()} (2D)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        if hyper_table.labels is not None:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                                 c=hyper_table.labels, cmap="tab10", alpha=0.7, edgecolor="k")
            fig.colorbar(scatter, label="Labels")
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                       c="steelblue", alpha=0.7, edgecolor="k")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        ax.set_title(f"{method.upper()} (3D)")
        plt.tight_layout()
        plt.show()

    return embedding

def nmf_decomposition(hyper_table: "HyperTable",
                      n_components: int = 5,
                      visualize: bool = True,
                      figsize: tuple = (10, 5),
                      random_state: int = 42) -> tuple:
    """
    Perform Non-negative Matrix Factorization (NMF) on hyperspectral data.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    n_components : int, default=5
        Number of components (endmembers) to extract.
    visualize : bool, default=True
        Whether to visualize endmember spectra and abundance maps.
    figsize : tuple, default=(10, 5)
        Figure size for plots.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    W : np.ndarray
        Abundance matrix of shape (samples, n_components).
    H : np.ndarray
        Endmember spectra matrix of shape (n_components, bands).
    """
    X = hyper_table.data.values.astype(float)

    # Fit NMF
    model = NMF(n_components=n_components, init="nndsvda",
                random_state=random_state, max_iter=500)
    W = model.fit_transform(X)  # Abundances
    H = model.components_       # Endmembers

    if visualize:
        bands = np.arange(hyper_table.bands) if hyper_table.wavelengths is None else hyper_table.wavelengths

        # Plot endmember spectra
        plt.figure(figsize=figsize)
        for i in range(n_components):
            plt.plot(bands, H[i, :], label=f"Endmember {i+1}")
        plt.xlabel("Wavelength (nm)" if hyper_table.wavelengths is not None else "Band Index")
        plt.ylabel("Reflectance")
        plt.title("NMF Endmember Spectra")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Plot abundance distributions
        plt.figure(figsize=(8, 5))
        for i in range(n_components):
            plt.hist(W[:, i], bins=30, alpha=0.6, label=f"Endmember {i+1}")
        plt.xlabel("Abundance Value")
        plt.ylabel("Frequency")
        plt.title("Abundance Histograms (per Endmember)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return W, H

def compute_mutual_info(
    hyper_table: "HyperTable",
    y: np.ndarray,
    task: Literal["classification", "regression"] = "classification",
    n_neighbors: int = 3,
    plot: bool = True
) -> np.ndarray:
    """
    Compute mutual information between hyperspectral bands and target variable.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container with wavelengths and spectra (shape: n_samples x n_bands).
    y : np.ndarray
        Target labels or continuous values (shape: n_samples,).
    task : {"classification", "regression"}, default="classification"
        Whether the task is supervised classification or regression.
    n_neighbors : int, default=3
        Number of neighbors for MI estimation (used in sklearn).
    plot : bool, default=True
        If True, plots the MI score vs wavelength.

    Returns
    -------
    np.ndarray
        Mutual information scores for each band.
    """
    if hyper_table.spectra is None or hyper_table.wavelengths is None:
        raise ValueError("HyperTable must contain both spectra and wavelengths.")

    X = hyper_table.spectra  # shape (n_samples, n_bands)

    if task == "classification":
        mi = mutual_info_classif(X, y, discrete_features=False, n_neighbors=n_neighbors, random_state=42)
    elif task == "regression":
        mi = mutual_info_regression(X, y, discrete_features=False, n_neighbors=n_neighbors, random_state=42)
    else:
        raise ValueError("Task must be 'classification' or 'regression'.")

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(hyper_table.wavelengths, mi, marker="o", linewidth=1.5)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Mutual Information")
        plt.title(f"Mutual Information ({task.capitalize()})")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    return mi

def lda_transform(hyper_table: "HyperTable",
                  n_components: int = 2,
                  visualize: bool = True,
                  figsize: tuple = (7, 5)) -> np.ndarray:
    """
    Apply Linear Discriminant Analysis (LDA) for dimensionality reduction.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container with labels.
    n_components : int, default=2
        Number of discriminant components.
    visualize : bool, default=True
        If True, shows scatter plot of LDA components.
    figsize : tuple, default=(7, 5)
        Figure size.

    Returns
    -------
    np.ndarray
        LDA-transformed data of shape (samples, n_components).
    """
    if hyper_table.labels is None:
        raise ValueError("LDA requires class labels in HyperTable.")

    X = hyper_table.data.values.astype(float)
    y = hyper_table.labels

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X, y)

    if visualize and n_components == 2:
        plt.figure(figsize=figsize)
        scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1],
                              c=y, cmap="tab10", alpha=0.7, edgecolor="k")
        plt.colorbar(scatter, label="Class")
        plt.xlabel("LD1")
        plt.ylabel("LD2")
        plt.title("LDA Projection (2D)")
        plt.grid(alpha=0.3)
        plt.show()

    return X_lda

from sklearn.decomposition import KernelPCA

def kernel_pca_transform(hyper_table: "HyperTable",
                         n_components: int = 3,
                         kernel: str = "rbf",
                         gamma: Optional[float] = None,
                         visualize: bool = True,
                         figsize: tuple = (7, 5)) -> np.ndarray:
    """
    Apply Kernel PCA to hyperspectral data.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    n_components : int, default=3
        Number of components.
    kernel : str, default="rbf"
        Kernel type ("linear", "poly", "rbf", "sigmoid", etc.).
    gamma : float, optional
        Kernel coefficient for RBF/poly/sigmoid.
    visualize : bool, default=True
        Show scatter plot if n_components >= 2.
    figsize : tuple, default=(7, 5)
        Plot size.

    Returns
    -------
    np.ndarray
        Kernel PCA-transformed data.
    """
    X = hyper_table.data.values.astype(float)
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
    X_kpca = kpca.fit_transform(X)

    if visualize and n_components >= 2:
        plt.figure(figsize=figsize)
        plt.scatter(X_kpca[:, 0], X_kpca[:, 1],
                    c="darkred", alpha=0.7, edgecolor="k")
        plt.xlabel("KPCA1")
        plt.ylabel("KPCA2")
        plt.title(f"Kernel PCA ({kernel} kernel)")
        plt.grid(alpha=0.3)
        plt.show()

    return X_kpca

def factor_analysis_transform(hyper_table: "HyperTable",
                              n_components: int = 3,
                              visualize: bool = True,
                              figsize: tuple = (7, 5)) -> np.ndarray:
    """
    Apply Factor Analysis (FA) to hyperspectral data.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    n_components : int, default=3
        Number of latent factors.
    visualize : bool, default=True
        Show scatter if n_components >= 2.
    figsize : tuple, default=(7, 5)
        Plot size.

    Returns
    -------
    np.ndarray
        FA-transformed data.
    """
    X = hyper_table.data.values.astype(float)
    fa = FactorAnalysis(n_components=n_components, random_state=42)
    X_fa = fa.fit_transform(X)

    if visualize and n_components >= 2:
        plt.figure(figsize=figsize)
        plt.scatter(X_fa[:, 0], X_fa[:, 1],
                    c="teal", alpha=0.7, edgecolor="k")
        plt.xlabel("Factor 1")
        plt.ylabel("Factor 2")
        plt.title("Factor Analysis Projection")
        plt.grid(alpha=0.3)
        plt.show()

    return X_fa

def isomap_transform(hyper_table: "HyperTable",
                     n_components: int = 2,
                     n_neighbors: int = 5,
                     visualize: bool = True,
                     figsize: tuple = (7, 5)) -> np.ndarray:
    """
    Apply Isomap for non-linear dimensionality reduction.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    n_components : int, default=2
        Number of embedding dimensions.
    n_neighbors : int, default=5
        Number of neighbors for Isomap graph.
    visualize : bool, default=True
        Show scatter plot if n_components == 2.
    figsize : tuple, default=(7, 5)
        Plot size.

    Returns
    -------
    np.ndarray
        Isomap embedding.
    """
    X = hyper_table.data.values.astype(float)
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    X_iso = isomap.fit_transform(X)

    if visualize and n_components == 2:
        plt.figure(figsize=figsize)
        plt.scatter(X_iso[:, 0], X_iso[:, 1],
                    c="navy", alpha=0.7, edgecolor="k")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.title("Isomap Projection")
        plt.grid(alpha=0.3)
        plt.show()

    return X_iso      

def svd_transform(hyper_table: "HyperTable",
                  n_components: int = 3,
                  visualize: bool = True,
                  figsize: tuple = (7, 5)) -> np.ndarray:
    """
    Apply Truncated SVD (similar to PCA but works with sparse/high-dim data).

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    n_components : int, default=3
        Number of components.
    visualize : bool, default=True
        Show scatter if n_components >= 2.
    figsize : tuple, default=(7, 5)
        Plot size.

    Returns
    -------
    np.ndarray
        SVD-transformed data.
    """
    X = hyper_table.data.values.astype(float)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X)

    if visualize and n_components >= 2:
        plt.figure(figsize=figsize)
        plt.scatter(X_svd[:, 0], X_svd[:, 1],
                    c="darkblue", alpha=0.7, edgecolor="k")
        plt.xlabel("SVD1")
        plt.ylabel("SVD2")
        plt.title("Truncated SVD Projection")
        plt.grid(alpha=0.3)
        plt.show()

    return X_svd


def spectral_embedding_transform(hyper_table: "HyperTable",
                                 n_components: int = 2,
                                 n_neighbors: int = 5,
                                 visualize: bool = True,
                                 figsize: tuple = (7, 5)) -> np.ndarray:
    """
    Apply Spectral Embedding (Laplacian Eigenmaps).

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    n_components : int, default=2
        Number of embedding dimensions.
    n_neighbors : int, default=5
        Number of neighbors for affinity graph.
    visualize : bool, default=True
        Show scatter plot if n_components == 2.
    figsize : tuple, default=(7, 5)
        Plot size.

    Returns
    -------
    np.ndarray
        Spectral embedding.
    """
    X = hyper_table.data.values.astype(float)
    embedder = SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
    X_se = embedder.fit_transform(X)

    if visualize and n_components == 2:
        plt.figure(figsize=figsize)
        plt.scatter(X_se[:, 0], X_se[:, 1],
                    c="crimson", alpha=0.7, edgecolor="k")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.title("Spectral Embedding")
        plt.grid(alpha=0.3)
        plt.show()

    return X_se


def mds_transform(hyper_table: "HyperTable",
                  n_components: int = 2,
                  visualize: bool = True,
                  figsize: tuple = (7, 5)) -> np.ndarray:
    """
    Apply Multidimensional Scaling (MDS) for distance-preserving embedding.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    n_components : int, default=2
        Embedding dimension.
    visualize : bool, default=True
        Show scatter plot if n_components == 2.
    figsize : tuple, default=(7, 5)
        Plot size.

    Returns
    -------
    np.ndarray
        MDS embedding.
    """
    X = hyper_table.data.values.astype(float)
    mds = MDS(n_components=n_components, random_state=42, dissimilarity="euclidean")
    X_mds = mds.fit_transform(X)

    if visualize and n_components == 2:
        plt.figure(figsize=figsize)
        plt.scatter(X_mds[:, 0], X_mds[:, 1],
                    c="olive", alpha=0.7, edgecolor="k")
        plt.xlabel("MDS1")
        plt.ylabel("MDS2")
        plt.title("MDS Projection")
        plt.grid(alpha=0.3)
        plt.show()

    return X_mds


def kmeans_clustering(hyper_table: "HyperTable",
                      n_clusters: int = 5,
                      visualize: bool = True,
                      figsize: tuple = (7, 5)) -> np.ndarray:
    """
    Cluster hyperspectral samples using KMeans.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    n_clusters : int, default=5
        Number of clusters.
    visualize : bool, default=True
        Show scatter of first 2 PCs colored by cluster.
    figsize : tuple, default=(7, 5)
        Plot size.

    Returns
    -------
    np.ndarray
        Cluster labels for each sample.
    """
    X = hyper_table.data.values.astype(float)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X)

    if visualize:
        X_pca = PCA(n_components=2).fit_transform(X)
        plt.figure(figsize=figsize)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab20", alpha=0.7, edgecolor="k")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("KMeans Clustering (PCA Projection)")
        plt.colorbar(label="Cluster")
        plt.show()

    return labels


def gmm_clustering(hyper_table: "HyperTable",
                   n_components: int = 5,
                   visualize: bool = True,
                   figsize: tuple = (7, 5)) -> np.ndarray:
    """
    Cluster hyperspectral samples using Gaussian Mixture Model (GMM).

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    n_components : int, default=5
        Number of Gaussian components.
    visualize : bool, default=True
        Show scatter of first 2 PCs colored by cluster.
    figsize : tuple, default=(7, 5)
        Plot size.

    Returns
    -------
    np.ndarray
        Cluster labels for each sample.
    """
    X = hyper_table.data.values.astype(float)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(X)

    if visualize:
        X_pca = PCA(n_components=2).fit_transform(X)
        plt.figure(figsize=figsize)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab20", alpha=0.7, edgecolor="k")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("GMM Clustering (PCA Projection)")
        plt.colorbar(label="Cluster")
        plt.show()

    return labels


def variance_per_band(hyper_table: "HyperTable") -> np.ndarray:
    """
    Compute and plot variance of each spectral band.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.

    Returns
    -------
    np.ndarray
        Variance values for each band.
    """
    X = hyper_table.data.values.astype(float)
    variances = np.var(X, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(hyper_table.wavelengths if hyper_table.wavelengths is not None else np.arange(len(variances)),
             variances, marker="o", linewidth=1.5, color="darkblue")
    plt.xlabel("Wavelength (nm)" if hyper_table.wavelengths is not None else "Band Index")
    plt.ylabel("Variance")
    plt.title("Variance per Spectral Band")
    plt.grid(alpha=0.3)
    plt.show()

    return variances


def anova_f_test(hyper_table: "HyperTable",
                 y: np.ndarray,
                 plot: bool = True) -> np.ndarray:
    """
    Perform ANOVA F-test for supervised band ranking.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    y : np.ndarray
        Class labels.
    plot : bool, default=True
        Plot F-scores vs wavelength.

    Returns
    -------
    np.ndarray
        F-scores per band.
    """
    X = hyper_table.data.values.astype(float)
    f_scores, _ = f_classif(X, y)

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(hyper_table.wavelengths if hyper_table.wavelengths is not None else np.arange(len(f_scores)),
                 f_scores, marker="o", linewidth=1.5, color="firebrick")
        plt.xlabel("Wavelength (nm)" if hyper_table.wavelengths is not None else "Band Index")
        plt.ylabel("F-score")
        plt.title("ANOVA F-test Scores per Band")
        plt.grid(alpha=0.3)
        plt.show()

    return f_scores


def smooth_spectra(hyper_table: "HyperTable",
                   window_length: int = 11,
                   polyorder: int = 2) -> np.ndarray:
    """
    Apply Savitzky–Golay smoothing to hyperspectral spectra.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    window_length : int, default=11
        Length of the filter window (must be odd).
    polyorder : int, default=2
        Polynomial order.

    Returns
    -------
    np.ndarray
        Smoothed spectra.
    """
    X = hyper_table.data.values.astype(float)
    smoothed = savgol_filter(X, window_length=window_length, polyorder=polyorder, axis=1)
    return smoothed
