import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from irodori.core import HyperTable
from irodori.classification import (
    split_data, scale_data, apply_pca, train_classifier, evaluate_classifier,
    plot_feature_importance, cross_validate_classifier, full_classification_pipeline,
    save_model, load_model, plot_precision_recall, classwise_metrics,
    top_k_accuracy, plot_calibration_curve, plot_tsne, plot_umap,
    permutation_importance_plot, build_voting_classifier, build_stacking_classifier,
    one_vs_rest_classifier, one_vs_one_classifier, compare_classifiers,
    optimize_threshold, apply_smote, batch_predict_and_report
)

# ------------------------------
# Fixtures
# ------------------------------
@pytest.fixture(scope="module")
def synthetic_data():
    X, y = make_classification(n_samples=200, n_features=20, n_classes=3, n_informative=5, random_state=42)
    return X, y

@pytest.fixture(scope="module")
def binary_data():
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, n_informative=5, random_state=42)
    return X, y

# ------------------------------
# Tests
# ------------------------------
def test_split_scale_pca(synthetic_data):
    X, y = synthetic_data
    from types import SimpleNamespace
    ht = SimpleNamespace(data=pd.DataFrame(X), labels=y)
    X_train, X_test, y_train, y_test = split_data(ht)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled, n_components=5)
    assert X_train_pca.shape[1] == 5


def test_train_and_evaluate_classifier(synthetic_data):
    X, y = synthetic_data
    clf = train_classifier(X, y, method="rf", n_estimators=10)
    acc, report, cm = evaluate_classifier(clf, X, y, plot_cm=False)
    assert 0 <= acc <= 1
    assert isinstance(report, dict)


def test_feature_importance_plot(synthetic_data):
    X, y = synthetic_data
    clf = RandomForestClassifier(n_estimators=10).fit(X, y)
    plot_feature_importance(clf, top_k=5)  # should run without error


def test_cross_validate(synthetic_data):
    X, y = synthetic_data
    mean_score, std_score = cross_validate_classifier(X, y, method="knn", cv=3)
    assert 0 <= mean_score <= 1


def test_full_pipeline(synthetic_data):
    X, y = synthetic_data
    from types import SimpleNamespace
    ht = SimpleNamespace(data=pd.DataFrame(X), labels=y)
    result = full_classification_pipeline(ht, pca_components=5, classifier="mlp", classifier_params={"max_iter":200})
    assert "accuracy" in result


def test_save_and_load_model(synthetic_data, tmp_path):
    X, y = synthetic_data
    clf = RandomForestClassifier(n_estimators=5).fit(X, y)
    filepath = tmp_path / "model.pkl"
    save_model(clf, filepath)
    loaded = load_model(filepath)
    assert isinstance(loaded, RandomForestClassifier)


def test_precision_recall(binary_data):
    X, y = binary_data
    clf = RandomForestClassifier().fit(X, y)
    plot_precision_recall(clf, X, y)


def test_classwise_metrics(synthetic_data):
    X, y = synthetic_data
    clf = RandomForestClassifier().fit(X, y)
    df = classwise_metrics(clf, X, y)
    assert isinstance(df, pd.DataFrame)


def test_top_k_accuracy(binary_data):
    X, y = binary_data
    clf = RandomForestClassifier().fit(X, y)
    score = top_k_accuracy(clf, X, y, k=2)
    assert 0 <= score <= 1


def test_calibration_curve(binary_data):
    X, y = binary_data
    clf = RandomForestClassifier().fit(X, y)
    brier = plot_calibration_curve(clf, X, y)
    assert brier >= 0


def test_tsne_umap(synthetic_data):
    X, y = synthetic_data
    plot_tsne(X, y)
    plot_umap(X, y)


def test_permutation_importance_plot(synthetic_data):
    X, y = synthetic_data
    clf = RandomForestClassifier().fit(X, y)
    permutation_importance_plot(clf, X, y, top_k=5)


def test_voting_and_stacking(synthetic_data):
    X, y = synthetic_data
    voting = build_voting_classifier(X, y)
    stacking = build_stacking_classifier(X, y)
    assert voting is not None and stacking is not None


def test_one_vs_wrappers(synthetic_data):
    X, y = synthetic_data
    base = RandomForestClassifier()
    ovr = one_vs_rest_classifier(base, X, y)
    ovo = one_vs_one_classifier(base, X, y)
    assert ovr is not None and ovo is not None


def test_compare_classifiers(synthetic_data):
    X, y = synthetic_data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    results = compare_classifiers(X_train, X_test, y_train, y_test)
    assert isinstance(results, pd.Series)


def test_optimize_threshold(binary_data):
    X, y = binary_data
    clf = RandomForestClassifier().fit(X, y)
    best_thresh, best_score = optimize_threshold(clf, X, y, metric="f1")
    assert 0 <= best_thresh <= 1


def test_smote(binary_data):
    X, y = binary_data
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, stratify=y)
    X_res, y_res = apply_smote(X_train, y_train)
    assert len(X_res) > len(X_train)


def test_batch_predict_and_report(binary_data, tmp_path):
    X, y = binary_data
    clf = RandomForestClassifier().fit(X, y)
    filepath = tmp_path / "predictions.csv"
    df = batch_predict_and_report(clf, X, y, filepath=filepath)
    assert filepath.exists()
    assert "Prediction" in df.columns

