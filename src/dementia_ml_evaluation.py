"""
Machine learning pipeline for automatic dementia assessment.

This module includes:
- Feature selection using mutual information
- Training and hyperparameter tuning of multiple classifiers
- Model evaluation using ROC curves, learning curves and confusion matrices

Note:
Original data is not included due to confidentiality constraints.
This code reflects my personal contribution to a group project on dementia assessment.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import joblib  # Uncomment if saving models locally

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

warnings.filterwarnings("ignore")

# Data loading and preparation

def load_data(train_path=None, test_path=None):
    """
    Placeholder function to load datasets.
    Data not included due to confidentiality.
    
    Returns:
        train_df, test_df: pandas DataFrames or None
    """
    # Example usage if you have local CSVs:
    # train_df = pd.read_csv(train_path)
    # test_df = pd.read_csv(test_path)
    # return train_df, test_df
    return None, None

def encode_labels(df):
    """
    Map string labels to integers.
    """
    label_map = {"Control": 0, "Dementia": 1}
    df.iloc[:, -1] = df.iloc[:, -1].map(label_map)
    return df

def prepare_data(train_df, test_df):
    """
    Split DataFrames into features and labels.
    """
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].astype(int).values
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].astype(int).values
    return X_train, y_train, X_test, y_test

# Plotting functions

def plot_roc(model, Xtr, ytr, Xte, yte, name):
    if hasattr(model, "predict_proba"):
        pos_idx = list(model.classes_).index(1)
        tr_scores = model.predict_proba(Xtr)[:, pos_idx]
        te_scores = model.predict_proba(Xte)[:, pos_idx]
    else:
        tr_scores = model.decision_function(Xtr)
        te_scores = model.decision_function(Xte)
    fpr_tr, tpr_tr, _ = roc_curve(ytr, tr_scores)
    fpr_te, tpr_te, _ = roc_curve(yte, te_scores)
    plt.figure(figsize=(8,6))
    plt.plot(fpr_tr, tpr_tr, label=f"Train (AUC={auc(fpr_tr,tpr_tr):.2f})")
    plt.plot(fpr_te, tpr_te, label=f"Test (AUC={auc(fpr_te,tpr_te):.2f})")
    plt.plot([0,1], [0,1], "--", color="gray")
    plt.title(f"ROC – {name}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid(alpha=0.3)
    # plt.savefig(f"roc_curves/roc_{name.replace(' ','_')}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_learning(model, X, y, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sizes, tr, te = learning_curve(
        model, X, y, cv=skf, scoring="balanced_accuracy",
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    plt.figure(figsize=(8,6))
    plt.plot(sizes, tr.mean(axis=1), label="Train")
    plt.plot(sizes, te.mean(axis=1), label="CV")
    plt.title(f"Learning Curve – {name}")
    plt.xlabel("Training Size")
    plt.ylabel("Balanced Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    # plt.savefig(f"learning_curves/learning_{name.replace(' ','_')}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_confusion(model, Xtr, ytr, Xte, yte, name):
    ytr_pred = model.predict(Xtr)
    cm_tr = confusion_matrix(ytr, ytr_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_tr, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix – Train – {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    # plt.savefig(f"confusion_matrices/cm_train_{name.replace(' ','_')}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    yte_pred = model.predict(Xte)
    cm_te = confusion_matrix(yte, yte_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_te, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix – Test – {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    # plt.savefig(f"confusion_matrices/cm_test_{name.replace(' ','_')}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# Train and evaluate models

def train_models(X_train, y_train):
    n_features = X_train.shape[1]
    k_values = [k for k in [5, 10, 15, 20, 30] if k <= n_features]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {}

    # Logistic Regression
    lr_pipe = Pipeline([
        ("fs", SelectKBest(mutual_info_classif, k=min(10, n_features))),
        ("lr", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=3000))
    ])
    lr_grid = GridSearchCV(
        lr_pipe,
        {"fs__k": k_values, "lr__C": np.logspace(-3,3,20), "lr__penalty": ["l1","l2"]},
        scoring="recall",
        cv=skf,
        n_jobs=-1
    )
    lr_grid.fit(X_train, y_train)
    models["Logistic Regression"] = lr_grid.best_estimator_

    # Linear SVM
    lin_svm_pipe = Pipeline([
        ("fs", SelectKBest(mutual_info_classif, k=min(10, n_features))),
        ("svm", SVC(kernel="linear", class_weight="balanced", probability=True))
    ])
    lin_svm_grid = GridSearchCV(
        lin_svm_pipe,
        {"fs__k": k_values, "svm__C": [0.1,1,10]},
        scoring="balanced_accuracy",
        cv=skf,
        n_jobs=-1
    )
    lin_svm_grid.fit(X_train, y_train)
    models["Linear SVM"] = lin_svm_grid.best_estimator_

    # RBF SVM
    rbf_svm_pipe = Pipeline([
        ("fs", SelectKBest(mutual_info_classif, k=min(10, n_features))),
        ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True))
    ])
    rbf_svm_grid = GridSearchCV(
        rbf_svm_pipe,
        {"fs__k": k_values, "svm__C": [0.1,1,10], "svm__gamma": [0.001,0.01,0.1]},
        scoring="balanced_accuracy",
        cv=skf,
        n_jobs=-1
    )
    rbf_svm_grid.fit(X_train, y_train)
    models["RBF SVM"] = rbf_svm_grid.best_estimator_

    # Bagged Trees
    bagged_pipe = Pipeline([
        ("fs", SelectKBest(mutual_info_classif, k=min(10, n_features))),
        ("bag", BaggingClassifier(n_estimators=100, random_state=42))
    ])
    bagged_pipe.fit(X_train, y_train)
    models["Bagged Trees"] = bagged_pipe

    # Random Forest
    rf_pipe = Pipeline([
        ("fs", SelectKBest(mutual_info_classif, k=min(10, n_features))),
        ("rf", RandomForestClassifier(class_weight="balanced", random_state=42))
    ])
    rf_grid = GridSearchCV(
        rf_pipe,
        {"fs__k": k_values,
         "rf__n_estimators": [300,500],
         "rf__max_depth": [None,5,10],
         "rf__min_samples_leaf": [1,2,5]},
        scoring="recall",
        cv=skf,
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    models["Random Forest"] = rf_grid.best_estimator_

    return models

# Example usage (commented)

# train_df, test_df = load_data("train.csv","test.csv")
# train_df = encode_labels(train_df)
# test_df = encode_labels(test_df)
# X_train, y_train, X_test, y_test = prepare_data(train_df, test_df)
# models = train_models(X_train, y_train)
# for name, model in models.items():
#     plot_roc(model, X_train, y_train, X_test, y_test, name)
#     plot_learning(model, X_train, y_train, name)
#     plot_confusion(model, X_train, y_train, X_test, y_test, name)
#     print(name)
#     print(confusion_matrix(y_test, model.predict(X_test)))
#     print(classification_report(y_test, model.predict(X_test)))
#     # joblib.dump(model, f"{name.replace(' ','_')}.joblib")
