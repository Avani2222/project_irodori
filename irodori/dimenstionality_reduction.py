import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def plot_pca(hyper_table: "HyperTable",
             n_components: int = 3,
             figsize: tuple = (7, 5)) -> None:
    """
    Visualize PCA results for hyperspectral data.

    Parameters
    ----------
    hyper_table : HyperTable
        Hyperspectral data container.
    n_components : int, default=3
        Number of components to visualize (must be >= 2).
    figsize : tuple, default=(7, 5)
        Figure size for plots.

    Returns
    -------
    None
        Displays explained variance and scatter plots.
    """
    X = hyper_table.data.values.astype(float)

    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

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

    # 2. Scatter plot (PC1 vs PC2)
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

    # 3. Optional 3D scatter (PC1 vs PC2 vs PC3)
    if n_components >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                   c="teal", alpha=0.7, edgecolor="k")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("PCA 3D Scatter Plot")
        plt.tight_layout()
        plt.show()

