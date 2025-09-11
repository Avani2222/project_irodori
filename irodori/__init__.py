"""
irodori: An irodori data analysis library.

Provides classes and functions for:
- Irodori data handling (HyperTable)
- Preprocessing (denoising, normalization, baseline correction, spectral shifting, mixup, etc.)
- Vegetation and water indices (NDVI, GNDVI, SAVI, EVI, ARVI, MNDWI, NDSI, etc.)
- Spectral analysis & visualization (derivatives, entropy, clustering, band statistics, PCA/ICA plots)
- Dimensionality reduction & clustering (PCA, ICA, NMF, LDA, t-SNE, UMAP, Isomap, MDS, k-means, GMM)
- Similarity & statistical measures (SAM, SID, Euclidean, Correlation, Cosine)
- Classification (training, evaluation, cross-validation, ensemble classifiers, pipelines)
"""

# Core object
from .core import HyperTable

# IO utilities
from .io import (
    load_csv,
    save_csv
)

# Analysis
from .analysis import (
    first_derivative,
    second_derivative,
    smooth_spectra,
    plot_spectral_signatures,
    plot_pca,
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
    pca_outlier_detection
)

# Classification
from .classification import (
    split_data,
    scale_data,
    apply_pca,
    train_classifier,
    evaluate_classifier,
    plot_feature_importance,
    cross_validate_classifier,
    full_classification_pipeline,
    save_model,
    load_model,
    plot_precision_recall,
    classwise_metrics,
    top_k_accuracy,
    plot_calibration_curve,
    plot_tsne,
    plot_umap,
    permutation_importance_plot,
    build_voting_classifier,
    build_stacking_classifier,
    one_vs_rest_classifier,
    one_vs_one_classifier,
    compare_classifiers,
    optimize_threshold,
    apply_smote,
    batch_predict_and_report
)

# Dimensionality reduction
from .dimensionality_reduction import (
    pca_transform,
    ica_transform,
    visualize_embedding,
    nmf_decomposition,
    compute_mutual_info,
    lda_transform,
    kernel_pca_transform,
    factor_analysis_transform,
    isomap_transform,
    svd_transform,
    spectral_embedding_transform,
    mds_transform,
    kmeans_clustering,
    gmm_clustering,
    variance_per_band,
    anova_f_test as dr_anova_f_test,
    smooth_spectra as dr_smooth_spectra
)

# Indices
from .indices import (
    compute_ndvi,
    compute_ndwi,
    compute_savi,
    compute_custom_index,
    compute_evi,
    compute_gndvi,
    compute_arvi,
    compute_mndwi,
    compute_ndsi
)

# Preprocessing
from .preprocessing import (
    _rebuild_hypertable,
    minmax_scale,
    standardize,
    vector_normalize,
    apply_savgol_filter,
    band_average,
    pca_denoise,
    remove_noisy_bands,
    select_wavelength_range,
    mahalanobis_distance,
    isolation_forest_filter,
    correct_baseline,
    normalize_vector,
    spectral_shift,
    mixup,
    spectral_derivative,
    add_noise,
    spectral_index,
    resample_spectra,
    estimate_snr,
    multiplicative_scatter_correction,
    standard_normal_variate,
    savgol_first_derivative,
    savgol_second_derivative,
    baseline_als,
    apply_baseline_correction,
    resample_wavelengths,
    remove_outliers_zscore
)

# Similarity
from .similarity import (
    spectral_angle_mapper as sim_sam,
    euclidean_distance as sim_euclidean,
    spectral_information_divergence as sim_sid,
    band_ratio as sim_band_ratio,
    spectral_correlation,
    cosine_similarity,
    sam_heatmap,
    similarity_dashboard
)

__all__ = [
    # Core
    "HyperTable",

    # IO
    "load_csv",
    "save_csv",

    # Analysis
    "first_derivative",
    "second_derivative",
    "smooth_spectra",
    "plot_spectral_signatures",
    "plot_pca",
    "plot_pixel_spectrum",
    "plot_average_spectrum",
    "plot_band_image",
    "plot_band_histograms",
    "anova_f_test",
    "mutual_info_band_selection",
    "band_correlation",
    "spectral_entropy",
    "cluster_bands",
    "spectral_snr",
    "spectral_peaks",
    "spectral_angle_mapper",
    "spectral_information_divergence",
    "euclidean_distance",
    "band_ratio",
    "continuum_removal",
    "pca_outlier_detection",

    # Classification
    "split_data",
    "scale_data",
    "apply_pca",
    "train_classifier",
    "evaluate_classifier",
    "plot_feature_importance",
    "cross_validate_classifier",
    "full_classification_pipeline",
    "save_model",
    "load_model",
    "plot_precision_recall",
    "classwise_metrics",
    "top_k_accuracy",
    "plot_calibration_curve",
    "plot_tsne",
    "plot_umap",
    "permutation_importance_plot",
    "build_voting_classifier",
    "build_stacking_classifier",
    "one_vs_rest_classifier",
    "one_vs_one_classifier",
    "compare_classifiers",
    "optimize_threshold",
    "apply_smote",
    "batch_predict_and_report",

    # Dimensionality reduction
    "pca_transform",
    "ica_transform",
    "visualize_embedding",
    "nmf_decomposition",
    "compute_mutual_info",
    "lda_transform",
    "kernel_pca_transform",
    "factor_analysis_transform",
    "isomap_transform",
    "svd_transform",
    "spectral_embedding_transform",
    "mds_transform",
    "kmeans_clustering",
    "gmm_clustering",
    "variance_per_band",
    "dr_anova_f_test",
    "dr_smooth_spectra",

    # Indices
    "compute_ndvi",
    "compute_ndwi",
    "compute_savi",
    "compute_custom_index",
    "compute_evi",
    "compute_gndvi",
    "compute_arvi",
    "compute_mndwi",
    "compute_ndsi",

    # Preprocessing
    "_rebuild_hypertable",
    "minmax_scale",
    "standardize",
    "vector_normalize",
    "apply_savgol_filter",
    "band_average",
    "pca_denoise",
    "remove_noisy_bands",
    "select_wavelength_range",
    "mahalanobis_distance",
    "isolation_forest_filter",
    "correct_baseline",
    "normalize_vector",
    "spectral_shift",
    "mixup",
    "spectral_derivative",
    "add_noise",
    "spectral_index",
    "resample_spectra",
    "estimate_snr",
    "multiplicative_scatter_correction",
    "standard_normal_variate",
    "savgol_first_derivative",
    "savgol_second_derivative",
    "baseline_als",
    "apply_baseline_correction",
    "resample_wavelengths",
    "remove_outliers_zscore",

    # Similarity
    "sim_sam",
    "sim_euclidean",
    "sim_sid",
    "sim_band_ratio",
    "spectral_correlation",
    "cosine_similarity",
    "sam_heatmap",
    "similarity_dashboard",
]
