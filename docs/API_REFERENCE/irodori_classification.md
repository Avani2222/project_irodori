# Classification Module API Reference

This module provides a comprehensive set of tools for **classification tasks** using `HyperTable` objects. It includes utilities for train/test splitting, feature scaling, dimensionality reduction, classifier training and evaluation, visualization, ensemble methods, threshold optimization, handling class imbalance, and batch predictions.

---

## Table of Contents

1. [Train/Test Split](#1-train-test-split)  
2. [Scaling](#2-scaling)  
3. [Dimensionality Reduction (PCA)](#3-dimensionality-reduction-pca)  
4. [Train Classifier](#4-train-classifier)  
5. [Evaluate Classifier](#5-evaluate-classifier)  
6. [Feature Importance](#6-feature-importance)  
7. [Cross-Validation](#7-cross-validation)  
8. [Full Pipeline](#8-full-pipeline)  
9. [Save/Load Model](#9-save-load-model)  
10. [Precision-Recall Curves](#10-precision-recall-curves)  
11. [Class-wise Metrics](#11-class-wise-metrics)  
12. [Top-k Accuracy](#12-top-k-accuracy)  
13. [Calibration Curve](#13-calibration-curve)  
14. [t-SNE Visualization](#14-t-sne-visualization)  
15. [UMAP Visualization](#15-umap-visualization)  
16. [Permutation Feature Importance](#16-permutation-feature-importance)  
17. [Ensemble Classifiers](#17-ensemble-classifiers)  
18. [One-vs-Rest / One-vs-One](#18-one-vs-rest--one-vs-one)  
19. [Auto-Model Comparison](#19-auto-model-comparison)  
20. [Threshold Optimization](#20-threshold-optimization)  
21. [SMOTE Oversampling](#21-smote-oversampling)  
22. [Batch Prediction and Report](#22-batch-prediction-and-report)  

---

## 1. Train/Test Split

```python
split_data(ht: HyperTable, test_size: float = 0.2, random_state: int = 42)
,,,
Description: Split HyperTable data into training and testing sets.
Parameters:
ht: HyperTable object containing data and labels.
test_size (float): Proportion of data for testing (default 0.2).
random_state (int): Seed for reproducibility (default 42).
Returns: X_train, X_test, y_train, y_test
2. Scaling
scale_data(X_train, X_test, method: str = "standard")
Description: Scale features using standardization or min-max normalization.
Parameters:
X_train: Training feature matrix
X_test: Test feature matrix
method ('standard' or 'minmax', default 'standard')
Returns: X_train_scaled, X_test_scaled, scaler
3. Dimensionality Reduction (PCA)
apply_pca(X_train, X_test, n_components: int = 10)
Description: Reduce dimensionality of features using PCA.
Parameters:
n_components (int, default 10): Number of principal components.
Returns: X_train_pca, X_test_pca, pca
4. Train Classifier
train_classifier(X_train, y_train, method: str = "svm", **kwargs)
Description: Train a classifier on the training data.
Supported Methods: 'svm', 'rf', 'knn', 'mlp'
Returns: Trained classifier instance
5. Evaluate Classifier
evaluate_classifier(clf, X_test, y_test, plot_cm: bool = True, figsize: tuple = (7, 6))
Description: Evaluate classifier performance with accuracy, classification report, and confusion matrix.
Returns: accuracy, classification_report, confusion_matrix
6. Feature Importance
plot_feature_importance(clf: RandomForestClassifier, feature_names: list = None, top_k: int = 20)
Description: Plot top-k feature importances (Random Forest only).
Parameters:
feature_names (list, optional): Names of features
top_k (int): Number of top features to display
7. Cross-Validation
cross_validate_classifier(X, y, method: str = "svm", cv: int = 5, **kwargs)
Description: Perform k-fold cross-validation on the classifier.
Returns: mean_score, std_score
8. Full Pipeline
full_classification_pipeline(ht: HyperTable, test_size=0.2, scaling="standard", pca_components=None, classifier="svm", classifier_params=None)
Description: End-to-end pipeline: split → scale → PCA → train → evaluate.
Returns: Dictionary containing classifier, scaler, pca, accuracy, classification_report, confusion_matrix
9. Save/Load Model
save_model(clf, filepath: str)
load_model(filepath: str)
Description: Persist or load trained classifier using joblib.
10. Precision-Recall Curves
plot_precision_recall(clf, X_test, y_test, class_names=None)
Description: Plot precision-recall curves for binary or multiclass classification.
11. Class-wise Metrics
classwise_metrics(clf, X_test, y_test)
Description: Returns a table with precision, recall, F1-score, and support per class.
Returns: pd.DataFrame
12. Top-k Accuracy
top_k_accuracy(clf, X_test, y_test, k=3)
Description: Compute top-k accuracy for classifiers with predict_proba.
Returns: Float (accuracy)
13. Calibration Curve
plot_calibration_curve(clf, X_test, y_test, n_bins=10)
Description: Plot reliability diagram and return Brier score.
Returns: Brier score loss
14. t-SNE Visualization
plot_tsne(X, y, n_components=2, perplexity=30, random_state=42)
Description: Project feature space into 2D using t-SNE for visualization.
15. UMAP Visualization
plot_umap(X, y, n_neighbors=15, min_dist=0.1, random_state=42)
Description: Project feature space into 2D using UMAP for visualization.
16. Permutation Feature Importance
permutation_importance_plot(clf, X_test, y_test, feature_names=None, n_repeats=10, top_k=20)
Description: Plot top-k features using permutation importance.
17. Ensemble Classifiers
build_voting_classifier(X_train, y_train, estimators=None, voting="soft")
build_stacking_classifier(X_train, y_train, estimators=None, final_estimator=None)
Description: Train ensemble models: Voting or Stacking classifiers.
18. One-vs-Rest / One-vs-One
one_vs_rest_classifier(base_clf, X_train, y_train)
one_vs_one_classifier(base_clf, X_train, y_train)
Description: Wrap a base classifier for multiclass strategies.
19. Auto-Model Comparison
compare_classifiers(X_train, X_test, y_train, y_test, classifiers=None)
Description: Compare multiple classifiers on the same dataset.
Returns: pd.Series sorted by accuracy
20. Threshold Optimization
optimize_threshold(clf, X_val, y_val, metric="f1")
Description: Optimize probability threshold for binary classification.
Returns: Best threshold and corresponding score
21. SMOTE Oversampling
apply_smote(X_train, y_train)
Description: Balance class distribution using SMOTE.
Returns: X_res, y_res
22. Batch Prediction and Report
batch_predict_and_report(clf, X, y=None, filepath="predictions.csv")
Description: Perform batch predictions and save results to CSV.
Returns: pd.DataFrame with predictions, true labels, and probabilities
