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

from .core import HyperTable
import joblib
# ------------------------------
# 1. Train/Test Split
# ------------------------------
def split_data(ht: HyperTable, test_size: float = 0.2, random_state: int = 42):
    X = ht.data.values
    y = ht.labels
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ------------------------------
# 2. Scaling
# ------------------------------
def scale_data(X_train, X_test, method: str = "standard"):
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
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca


# ------------------------------
# 4. Train Classifier
# ------------------------------
def train_classifier(X_train, y_train, method: str = "svm", **kwargs):
    """
    Supported methods: 'svm', 'rf', 'knn', 'mlp'
    kwargs: classifier-specific parameters
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
# 6. Feature Importance (Random Forest)
# ------------------------------
def plot_feature_importance(clf: RandomForestClassifier, feature_names: list = None, top_k: int = 20, figsize: tuple = (10, 5)):
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
    clf = train_classifier(X, y, method=method, **kwargs)
    scores = cross_val_score(clf, X, y, cv=cv)
    return scores.mean(), scores.std()


# ------------------------------
# 8. Pipeline Example
# ------------------------------
def full_classification_pipeline(ht: HyperTable,
                                 test_size: float = 0.2,
                                 scaling: str = "standard",
                                 pca_components: int = None,
                                 classifier: str = "svm",
                                 classifier_params: dict = None):
    """
    Complete workflow: split → scale → PCA → train → evaluate
    """
    # Split
    X_train, X_test, y_train, y_test = split_data(ht, test_size=test_size)
    
    # Scale
    X_train, X_test, scaler = scale_data(X_train, X_test, method=scaling)
    
    # PCA
    if pca_components is not None:
        X_train, X_test, pca = apply_pca(X_train, X_test, n_components=pca_components)
    else:
        pca = None
    
    # Train
    clf = train_classifier(X_train, y_train, method=classifier, **(classifier_params or {}))
    
    # Evaluate
    acc, report, cm = evaluate_classifier(clf, X_test, y_test)
    
    return {
        "classifier": clf,
        "scaler": scaler,
        "pca": pca,
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm
    }

def plot_roc(clf, X_test, y_test, class_names=None):
    """
    Plot ROC curves for multiclass classification.
    """
    if not hasattr(clf, "predict_proba"):
        raise ValueError("Classifier must support predict_proba.")

    y_true_bin = label_binarize(y_test, classes=np.unique(y_test))
    y_score = clf.predict_proba(X_test)
    n_classes = y_true_bin.shape[1]

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.show()

def hyperparameter_search(X_train, y_train, method='svm', param_grid=None, cv=5, scoring='accuracy'):
    """
    Perform GridSearchCV to find the best hyperparameters.
    """
    clf = None
    if method.lower() == 'svm':
        clf = SVC(probability=True)
    elif method.lower() == 'rf':
        clf = RandomForestClassifier()
    elif method.lower() == 'mlp':
        clf = MLPClassifier(max_iter=500)
    elif method.lower() == 'knn':
        clf = KNeighborsClassifier()
    else:
        raise ValueError("Unsupported classifier for grid search")

    grid = GridSearchCV(clf, param_grid, cv=cv, scoring=scoring)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

def plot_confusion_heatmap(cm, class_names=None, figsize=(7, 6)):
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix Heatmap')
    plt.show()

def plot_learning_curve(clf, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'):
    train_sizes, train_scores, val_scores = learning_curve(clf, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    plt.xlabel('Training Size')
    plt.ylabel(scoring)
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.show()

def save_model(clf, filepath: str):
    joblib.dump(clf, filepath)

def load_model(filepath: str):
    return joblib.load(filepath)




