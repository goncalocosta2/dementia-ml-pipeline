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

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")

# CRIAR PASTAS RESULTADOS

os.makedirs("roccurves", exist_ok=True)
os.makedirs("confusionmatrices", exist_ok=True)
os.makedirs("learningcurves", exist_ok=True)

# LOAD DATA

df = pd.read_csv("merged_features2.csv")

featurecolumns = df.columns[1:-1]

Xall = df[featurecolumns].values
yall = df["label"].astype(int).values

featurenames = featurecolumns.tolist()

print("Dataset completo:", Xall.shape)
print("Distribuição classes:", np.bincount(yall))

# STRATIFIED SPLIT

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for trainidx, testidx in skf.split(Xall, yall):

    Xtrain_full = Xall[trainidx]
    Xtest_full = Xall[testidx]

    ytrain = yall[trainidx]
    ytest = yall[testidx]

    break

print("Train:", Xtrain_full.shape)
print("Test:", Xtest_full.shape)

# NORMALIZAÇÃO GLOBAl

scaler_global = StandardScaler()

Xtrain_scaled = scaler_global.fit_transform(Xtrain_full)
Xtest_scaled = scaler_global.transform(Xtest_full)

joblib.dump(scaler_global, "scaler_global.joblib")

# MUTUAL INFORMATION FEATURE SELECTION

mi = mutual_info_classif(Xtrain_scaled, ytrain)

midf = pd.DataFrame({
    "Feature": featurenames,
    "Score": mi
}).sort_values(by="Score", ascending=False)

print(midf)

topn = 10
topfeatures = midf["Feature"].values[:topn]

print("\nTop MI features:", topfeatures)

joblib.dump(topfeatures, "mi_selected_features.joblib")

mi_indices = [featurenames.index(f) for f in topfeatures]

Xtrain_mi = Xtrain_full[:, mi_indices]
Xtest_mi = Xtest_full[:, mi_indices]

# NORMALIZAÇÃO DAS FEATURES SELECIONADAS

scaler_mi = StandardScaler()

Xtrain_mi = scaler_mi.fit_transform(Xtrain_mi)
Xtest_mi = scaler_mi.transform(Xtest_mi)

joblib.dump(scaler_mi, "scaler_mi.joblib")

# MODELOS

cubicsvm = Pipeline([
    ("svm", SVC(
        kernel="poly",
        degree=3,
        C=1,
        gamma="scale",
        probability=True,
        class_weight="balanced"
    ))
])

coarsetree = Pipeline([
    ("tree", DecisionTreeClassifier(
        criterion="gini",
        max_leaf_nodes=5,
        random_state=42,
        class_weight="balanced"
    ))
])

cubicsvm.fit(Xtrain_mi, ytrain)
coarsetree.fit(Xtrain_mi, ytrain)

# FUNÇÕES DE AVALIAÇÃO

def plotroc(model, Xtr, ytr, Xte, yte, name):

    trscores = model.predict_proba(Xtr)[:,1]
    tescores = model.predict_proba(Xte)[:,1]

    fprtr, tprtr, _ = roc_curve(ytr, trscores)
    fprte, tprte, _ = roc_curve(yte, tescores)

    plt.figure()

    plt.plot(fprtr, tprtr, label=f"Train AUC={auc(fprtr,tprtr):.2f}")
    plt.plot(fprte, tprte, label=f"Test AUC={auc(fprte,tprte):.2f}")

    plt.plot([0,1],[0,1],"--")

    plt.title(name)
    plt.legend()

    plt.savefig(f"roccurves/roc_{name}.png")
    plt.close()


def plotconfusion(model, Xtr, ytr, Xte, yte, name):

    cm_train = confusion_matrix(ytr, model.predict(Xtr))
    cm_test = confusion_matrix(yte, model.predict(Xte))

    plt.figure()
    sns.heatmap(cm_train, annot=True, fmt="d")
    plt.title(name + "_train")
    plt.savefig(f"confusionmatrices/cm_{name}_train.png")
    plt.close()

    plt.figure()
    sns.heatmap(cm_test, annot=True, fmt="d")
    plt.title(name + "_test")
    plt.savefig(f"confusionmatrices/cm_{name}_test.png")
    plt.close()


def plot_learning_curve(model, X, y, name):

    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 10),
        shuffle=True,
        random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure()

    plt.plot(train_sizes, train_mean, label="Train")
    plt.plot(train_sizes, test_mean, label="Validation")

    plt.xlabel("Training size")
    plt.ylabel("Accuracy")
    plt.title(name)

    plt.legend()

    plt.savefig(f"learningcurves/lc_{name}.png")
    plt.close()

# AVALIAÇÃO

models = {
    "CubicSVM_MI": cubicsvm,
    "CoarseTree_MI": coarsetree
}

for name, model in models.items():

    plotroc(model, Xtrain_mi, ytrain, Xtest_mi, ytest, name)

    plotconfusion(model, Xtrain_mi, ytrain, Xtest_mi, ytest, name)

    plot_learning_curve(model, Xtrain_mi, ytrain, name)

    print("\n", name)

    print(confusion_matrix(ytest, model.predict(Xtest_mi)))
    print(classification_report(ytest, model.predict(Xtest_mi)))

# GUARDAR MODELOS

joblib.dump(cubicsvm, "best_cubicsvm_mi.joblib")
joblib.dump(coarsetree, "best_coarsetree_mi.joblib")
