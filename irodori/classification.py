"""Module for classification tasks including split_data, scale_data, apply_pca, train_classifier, evaluate_classifier, plot_feature_importance, 
cross_validate_classifier, full_classification_pipeline, plot_roc, hyperparameter_search, plot_confusion_heatmap, plot_learning_curve, save_model, load_model
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.manifold import TSNE
import umap
from imblearn.over_sampling import SMOTE

from .core import HyperTable
import joblib
# ------------------------------
# 1. Train/Test Split
# ------------------------------
def split_data(ht: HyperTable, test_size: float = 0.2, random_state: int = 42):
    """
    Split HyperTable into train and test sets.

    Parameters
    ----------
    ht : HyperTable
        HyperTable object containing data and labels.
    test_size : float, default=0.2
        Proportion of data to allocate to the test set.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Train and test splits for features and labels.
    """
    X = ht.data.values
    y = ht.labels
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ------------------------------
# 2. Scaling
# ------------------------------
def scale_data(X_train, X_test, method: str = "standard"):
    """
    Scale features using standardization or min-max normalization.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    X_test : np.ndarray
        Test feature matrix.
    method : {'standard', 'minmax'}, default='standard'
        Scaling method.

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
        Scaled feature matrices.
    scaler : object
        Fitted scaler instance.
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unknown scaling method")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ------------------------------
# 3. Dimensionality Reduction
# ------------------------------
def apply_pca(X_train, X_test, n_components: int = 10):
    """
    Apply PCA for dimensionality reduction.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    X_test : np.ndarray
        Test feature matrix.
    n_components : int, default=10
        Number of principal components.

    Returns
    -------
    X_train_pca, X_test_pca : np.ndarray
        Reduced feature matrices.
    pca : PCA
        Trained PCA instance.
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca


# ------------------------------
# 4. Train Classifier
# ------------------------------
def train_classifier(X_train, y_train, method: str = "svm", **kwargs):
    """
    Train a classifier on the training set.

    Supported methods: 'svm', 'rf', 'knn', 'mlp'.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    method : str, default='svm'
        Classifier type.
    **kwargs : dict
        Additional classifier parameters.

    Returns
    -------
    clf : classifier
        Trained classifier instance.
    """
    if method.lower() == "svm":
        clf = SVC(**kwargs)
    elif method.lower() == "rf":
        clf = RandomForestClassifier(**kwargs)
    elif method.lower() == "knn":
        clf = KNeighborsClassifier(**kwargs)
    elif method.lower() == "mlp":
        clf = MLPClassifier(**kwargs)
    else:
        raise ValueError("Unknown classifier method")
    
    clf.fit(X_train, y_train)
    return clf


# ------------------------------
# 5. Evaluate Classifier
# ------------------------------
def evaluate_classifier(clf, X_test, y_test, plot_cm: bool = True, figsize: tuple = (7, 6)):
    """
    Evaluate classifier performance with accuracy, report, and confusion matrix.

    Parameters
    ----------
    clf : classifier
        Trained classifier.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test labels.
    plot_cm : bool, default=True
        Whether to display confusion matrix.
    figsize : tuple, default=(7, 6)
        Figure size for the confusion matrix.

    Returns
    -------
    acc : float
        Accuracy score.
    report : dict
        Classification report.
    cm : np.ndarray
        Confusion matrix.
    """
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    if plot_cm:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=figsize)
        disp.plot(ax=ax, cmap="Blues", colorbar=True)
        plt.title("Confusion Matrix")
        plt.show()
    
    return acc, report, cm


# ------------------------------
# 6. Feature Importance
# ------------------------------
def plot_feature_importance(clf: RandomForestClassifier, feature_names: list = None, top_k: int = 20, figsize: tuple = (10, 5)):
    """
    Plot feature importance for Random Forest classifier.

    Parameters
    ----------
    clf : RandomForestClassifier
        Trained random forest model.
    feature_names : list, optional
        Feature names.
    top_k : int, default=20
        Number of top features to display.
    figsize : tuple, default=(10, 5)
        Figure size.
    """
    if not isinstance(clf, RandomForestClassifier):
        raise ValueError("Feature importance is only available for RandomForestClassifier.")
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]
    
    if feature_names is None:
        feature_names = [f"Band {i}" for i in range(len(importances))]
    
    plt.figure(figsize=figsize)
    sns.barplot(x=[feature_names[i] for i in indices],
                y=importances[indices],
                palette="viridis")
    plt.xticks(rotation=90)
    plt.ylabel("Importance")
    plt.title(f"Top {top_k} Feature Importances")
    plt.tight_layout()
    plt.show()


# ------------------------------
# 7. Cross-Validation
# ------------------------------
def cross_validate_classifier(X, y, method: str = "svm", cv: int = 5, **kwargs):
    """
    Perform cross-validation on a classifier.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    method : str, default='svm'
        Classifier type.
    cv : int, default=5
        Number of folds.
    **kwargs : dict
        Additional classifier parameters.

    Returns
    -------
    mean_score : float
        Mean cross-validation score.
    std_score : float
        Standard deviation of scores.
    """
    clf = train_classifier(X, y, method=method, **kwargs)
    scores = cross_val_score(clf, X, y, cv=cv)
    return scores.mean(), scores.std()


# ------------------------------
# 8. Full Pipeline
# ------------------------------
def full_classification_pipeline(ht: HyperTable,
                                 test_size: float = 0.2,
                                 scaling: str = "standard",
                                 pca_components: int = None,
                                 classifier: str = "svm",
                                 classifier_params: dict = None):
    """
    End-to-end classification pipeline:
    split → scale → PCA → train → evaluate.

    Parameters
    ----------
    ht : HyperTable
        HyperTable object containing data and labels.
    test_size : float, default=0.2
        Proportion for test split.
    scaling : {'standard', 'minmax'}, default='standard'
        Scaling method.
    pca_components : int, optional
        Number of PCA components.
    classifier : str, default='svm'
        Classifier type.
    classifier_params : dict, optional
        Parameters for classifier.

    Returns
    -------
    dict
        Pipeline outputs: classifier, scaler, PCA, accuracy, report, confusion matrix.
    """
    X_train, X_test, y_train, y_test = split_data(ht, test_size=test_size)
    X_train, X_test, scaler = scale_data(X_train, X_test, method=scaling)
    
    if pca_components is not None:
        X_train, X_test, pca = apply_pca(X_train, X_test, n_components=pca_components)
    else:
        pca = None
    
    clf = train_classifier(X_train, y_train, method=classifier, **(classifier_params or {}))
    acc, report, cm = evaluate_classifier(clf, X_test, y_test)
    
    return {
        "classifier": clf,
        "scaler": scaler,
        "pca": pca,
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm
    }

def save_model(clf, filepath: str):
    joblib.dump(clf, filepath)

def load_model(filepath: str):
    return joblib.load(filepath)

# ------------------------------
# 9. Precision-Recall Curves
# ------------------------------
def plot_precision_recall(clf, X_test, y_test, class_names=None):
    """
    Plot precision-recall curves for multiclass classification.

    Parameters
    ----------
    clf : classifier with predict_proba
        Trained classifier.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True labels.
    class_names : list, optional
        Names of the classes. If None, class indices are used.
    """
    if not hasattr(clf, "predict_proba"):
        raise ValueError("Classifier must support predict_proba.")

    y_score = clf.predict_proba(X_test)
    n_classes = len(np.unique(y_test))
    y_true_bin = label_binarize(y_test, classes=np.unique(y_test))

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_score[:, i])
        label = f"Class {class_names[i] if class_names else i} (AP = {ap:.2f})"
        plt.plot(recall, precision, lw=2, label=label)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="best")
    plt.show()


# ------------------------------
# 10. Class-wise Metrics Table
# ------------------------------
def classwise_metrics(clf, X_test, y_test):
    """
    Return a DataFrame with precision, recall, F1, and support per class.

    Parameters
    ----------
    clf : classifier
        Trained classifier.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True labels.

    Returns
    -------
    pd.DataFrame
        Classification report as a table.
    """
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return pd.DataFrame(report).transpose()


# ------------------------------
# 11. Top-k Accuracy
# ------------------------------
def top_k_accuracy(clf, X_test, y_test, k=3):
    """
    Compute top-k accuracy, i.e., fraction of samples where the true label
    is within the top-k predicted probabilities.

    Parameters
    ----------
    clf : classifier with predict_proba
        Trained classifier.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True labels.
    k : int
        Number of top predictions to consider.

    Returns
    -------
    float
        Top-k accuracy score.
    """
    if not hasattr(clf, "predict_proba"):
        raise ValueError("Classifier must support predict_proba.")
    probs = clf.predict_proba(X_test)
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(y_test[i] in top_k_preds[i] for i in range(len(y_test)))
    return correct / len(y_test)


# ------------------------------
# 12. Calibration Curve
# ------------------------------
def plot_calibration_curve(clf, X_test, y_test, n_bins=10):
    """
    Plot calibration curve (reliability diagram) and return Brier score.

    Parameters
    ----------
    clf : classifier with predict_proba
        Trained classifier.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True labels.
    n_bins : int
        Number of bins for calibration curve.

    Returns
    -------
    float
        Brier score loss.
    """
    if not hasattr(clf, "predict_proba"):
        raise ValueError("Classifier must support predict_proba.")
    probs = clf.predict_proba(X_test)[:, 1] if clf.predict_proba(X_test).ndim > 1 else clf.predict_proba(X_test)
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=n_bins)
    
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, "s-", label="Calibration curve")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.show()

    return brier_score_loss(y_test, probs)


# ------------------------------
# 13. t-SNE Visualization
# ------------------------------
def plot_tsne(X, y, n_components=2, perplexity=30, random_state=42):
    """
    Visualize feature space using t-SNE projection.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    n_components : int
        Number of dimensions (2 or 3).
    perplexity : float
        t-SNE perplexity parameter.
    random_state : int
        Random seed.
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y, palette="tab10")
    plt.title("t-SNE Projection of Feature Space")
    plt.show()


# ------------------------------
# 14. UMAP Visualization
# ------------------------------
def plot_umap(X, y, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Visualize feature space using UMAP projection.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    n_neighbors : int
        Number of neighbors for UMAP.
    min_dist : float
        Minimum distance parameter for UMAP.
    random_state : int
        Random seed.
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    X_embedded = reducer.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y, palette="tab10")
    plt.title("UMAP Projection of Feature Space")
    plt.show()


# ------------------------------
# 15. Permutation Feature Importance
# ------------------------------
def permutation_importance_plot(clf, X_test, y_test, feature_names=None, n_repeats=10, top_k=20):
    """
    Plot permutation feature importance.

    Parameters
    ----------
    clf : classifier
        Trained classifier.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True labels.
    feature_names : list, optional
        Feature names for plotting.
    n_repeats : int
        Number of permutations.
    top_k : int
        Number of top features to show.
    """
    result = permutation_importance(clf, X_test, y_test, n_repeats=n_repeats, random_state=42)
    indices = result.importances_mean.argsort()[::-1][:top_k]

    if feature_names is None:
        feature_names = [f"Band {i}" for i in range(X_test.shape[1])]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=[feature_names[i] for i in indices], y=result.importances_mean[indices])
    plt.xticks(rotation=90)
    plt.ylabel("Permutation Importance")
    plt.title(f"Top {top_k} Features (Permutation)")
    plt.show()


# ------------------------------
# 16. Ensemble Classifier
# ------------------------------
def build_voting_classifier(X_train, y_train, estimators=None, voting="soft"):
    """
    Train a VotingClassifier with multiple base estimators.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    estimators : list of tuples, optional
        List of (name, estimator) pairs.
    voting : str
        'soft' (probabilities) or 'hard' (majority vote).

    Returns
    -------
    VotingClassifier
        Trained voting classifier.
    """
    if estimators is None:
        estimators = [
            ("svm", SVC(probability=True)),
            ("rf", RandomForestClassifier()),
            ("knn", KNeighborsClassifier())
        ]
    clf = VotingClassifier(estimators=estimators, voting=voting)
    clf.fit(X_train, y_train)
    return clf


# ------------------------------
# 17. Stacking Classifier
# ------------------------------
def build_stacking_classifier(X_train, y_train, estimators=None, final_estimator=None):
    """
    Train a StackingClassifier.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    estimators : list of tuples, optional
        Base estimators.
    final_estimator : classifier, optional
        Meta-classifier.

    Returns
    -------
    StackingClassifier
        Trained stacking classifier.
    """
    if estimators is None:
        estimators = [
            ("rf", RandomForestClassifier()),
            ("knn", KNeighborsClassifier())
        ]
    if final_estimator is None:
        final_estimator = SVC(probability=True)

    clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
    clf.fit(X_train, y_train)
    return clf


# ------------------------------
# 18. One-vs-Rest and One-vs-One
# ------------------------------
def one_vs_rest_classifier(base_clf, X_train, y_train):
    """
    Wrap a classifier for One-vs-Rest multiclass strategy.

    Parameters
    ----------
    base_clf : classifier
        Base classifier.
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    OneVsRestClassifier
        Trained classifier.
    """
    clf = OneVsRestClassifier(base_clf)
    clf.fit(X_train, y_train)
    return clf


def one_vs_one_classifier(base_clf, X_train, y_train):
    """
    Wrap a classifier for One-vs-One multiclass strategy.

    Parameters
    ----------
    base_clf : classifier
        Base classifier.
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    OneVsOneClassifier
        Trained classifier.
    """
    clf = OneVsOneClassifier(base_clf)
    clf.fit(X_train, y_train)
    return clf


# ------------------------------
# 19. Auto-Model Comparison
# ------------------------------
def compare_classifiers(X_train, X_test, y_train, y_test, classifiers=None):
    """
    Train and compare multiple classifiers on the same dataset.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    X_test : np.ndarray
        Test feature matrix.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    classifiers : dict, optional
        Mapping from name to classifier.

    Returns
    -------
    pd.Series
        Classifier accuracies sorted in descending order.
    """
    if classifiers is None:
        classifiers = {
            "SVM": SVC(probability=True),
            "RandomForest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier(),
            "MLP": MLPClassifier(max_iter=500)
        }

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        results[name] = acc
    return pd.Series(results).sort_values(ascending=False)


# ------------------------------
# 20. Threshold Optimization
# ------------------------------
def optimize_threshold(clf, X_val, y_val, metric="f1"):
    """
    Find best probability threshold for binary classification.

    Parameters
    ----------
    clf : classifier with predict_proba
        Trained classifier.
    X_val : np.ndarray
        Validation feature matrix.
    y_val : np.ndarray
        Validation labels.
    metric : str
        'f1' or 'balanced_accuracy'.

    Returns
    -------
    float
        Best threshold.
    float
        Best score.
    """
    if not hasattr(clf, "predict_proba"):
        raise ValueError("Classifier must support predict_proba.")
    probs = clf.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 50)
    best_thresh, best_score = 0.5, -1

    from sklearn.metrics import f1_score, balanced_accuracy_score
    scorer = f1_score if metric == "f1" else balanced_accuracy_score

    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = scorer(y_val, preds)
        if score > best_score:
            best_score, best_thresh = score, t

    return best_thresh, best_score


# ------------------------------
# 21. Handle Class Imbalance (SMOTE)
# ------------------------------
def apply_smote(X_train, y_train):
    """
    Apply SMOTE oversampling to balance classes.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    np.ndarray
        Resampled features.
    np.ndarray
        Resampled labels.
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


# ------------------------------
# 22. Batch Prediction + Report
# ------------------------------
def batch_predict_and_report(clf, X, y=None, filepath="predictions.csv"):
    """
    Perform batch predictions and save results to CSV.

    Parameters
    ----------
    clf : classifier
        Trained classifier.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray, optional
        True labels (if available).
    filepath : str
        File path to save predictions.

    Returns
    -------
    pd.DataFrame
        DataFrame with predictions, true labels, and probabilities.
    """
    preds = clf.predict(X)
    proba = clf.predict_proba(X) if hasattr(clf, "predict_proba") else None
    df = pd.DataFrame({"Prediction": preds})
    if y is not None:
        df["True"] = y
    if proba is not None:
        df = pd.concat([df, pd.DataFrame(proba, columns=[f"Prob_Class_{i}" for i in range(proba.shape[1])])], axis=1)
    df.to_csv(filepath, index=False)
    return df




