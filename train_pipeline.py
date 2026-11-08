"""
train_pipeline.py
=================
End-to-end training script for the PCOS classification assignment.

Steps
-----
1. Load ``dataset/PCOS_data_without_infertility.xlsx`` and cross-check it with
   ``dataset/PCOS_infertility.csv``.
2. Stratified 75 / 25 hold-out split (``SEED = 42``).  The 25 % part is written to
   ``test_data.csv`` and is NEVER used for fitting or tuning.
3. For each of the five classifiers build a full pipeline
   (feature engineering -> winsorising -> imputation -> scaling/encoding -> model)
   and tune it with GridSearchCV (3 x 5-fold repeated stratified CV, scored on MCC).
4. Evaluate every tuned model on the hold-out set with the six required metrics.
5. Save pipelines to ``model/*.pkl`` and the scores to ``model/scoreboard.json``.

Run:  ``python train_pipeline.py``
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder,
                                   PowerTransformer, RobustScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from pcos_prep import (BINARY_COLUMNS, CONTINUOUS_COLUMNS, INPUT_COLUMNS, LABEL,
                       NOMINAL_COLUMNS, ClinicalFeatureBuilder, PercentileClipper,
                       load_raw_pcos)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "model"
SEED = 42
HOLDOUT_SHARE = 0.25


# --------------------------------------------------------------------------- #
#  Pipeline factory
# --------------------------------------------------------------------------- #
def make_feature_stage(scaler) -> Pipeline:
    """Feature engineering + column-wise preprocessing. ``scaler`` is applied to the
    continuous block only and can be swapped by GridSearchCV."""
    continuous_block = Pipeline([
        ("winsor", PercentileClipper(lower_pct=1.0, upper_pct=99.0)),
        ("fill", SimpleImputer(strategy="median")),
        ("rescale", scaler),
    ])
    flag_block = Pipeline([("fill", SimpleImputer(strategy="most_frequent"))])
    nominal_block = Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    router = ColumnTransformer(
        [("cont", continuous_block, CONTINUOUS_COLUMNS),
         ("flag", flag_block, BINARY_COLUMNS),
         ("nominal", nominal_block, NOMINAL_COLUMNS)],
        remainder="drop", verbose_feature_names_out=False,
    )
    return Pipeline([("engineer", ClinicalFeatureBuilder()), ("columns", router)])


SCALER_KEY = "features__columns__cont__rescale"

# name -> (estimator, param grid, file name)
CANDIDATES = {
    "Logistic Regression": (
        LogisticRegression(max_iter=4000, solver="liblinear", random_state=SEED),
        {SCALER_KEY: [RobustScaler(), PowerTransformer()],
         "classifier__C": [0.03, 0.1, 0.3, 1.0, 3.0],
         "classifier__penalty": ["l1", "l2"],
         "classifier__class_weight": [None, "balanced"]},
        "logreg_pipeline.pkl",
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=SEED),
        {"classifier__criterion": ["gini", "entropy"],
         "classifier__max_depth": [3, 4, 5, 6, None],
         "classifier__min_samples_leaf": [3, 6, 12],
         "classifier__ccp_alpha": [0.0, 0.005, 0.01]},
        "dtree_pipeline.pkl",
    ),
    "KNN": (
        KNeighborsClassifier(),
        {SCALER_KEY: [MinMaxScaler(), RobustScaler(), PowerTransformer()],
         "classifier__n_neighbors": [5, 9, 13, 17, 25],
         "classifier__weights": ["uniform", "distance"],
         "classifier__p": [1, 2]},
        "knn_pipeline.pkl",
    ),
    "Naive Bayes": (
        GaussianNB(),
        {SCALER_KEY: [PowerTransformer(), RobustScaler()],
         "classifier__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 5e-2]},
        "gnb_pipeline.pkl",
    ),
    "Support Vector Machine": (
        SVC(probability=True, random_state=SEED),
        {SCALER_KEY: [RobustScaler(), PowerTransformer()],
         "classifier__kernel": ["rbf", "linear"],
         "classifier__C": [0.1, 0.5, 1.0, 4.0],
         "classifier__gamma": ["scale", 0.005, 0.02]},
        "svm_pipeline.pkl",
    ),
}


def score_holdout(pipe, X: pd.DataFrame, y: pd.Series) -> dict:
    """Six required metrics + confusion matrix for one fitted pipeline."""
    guess = pipe.predict(X)
    risk = pipe.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, guess, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y, guess),
        "AUC": roc_auc_score(y, risk),
        "Precision": precision_score(y, guess, zero_division=0),
        "Recall": recall_score(y, guess, zero_division=0),
        "F1": f1_score(y, guess, zero_division=0),
        "MCC": matthews_corrcoef(y, guess),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def describe_params(best: dict) -> str:
    parts = []
    for key, val in best.items():
        short = key.split("__")[-1]
        if short == "rescale":
            val = type(val).__name__
        parts.append(f"{short}={val}")
    return ", ".join(parts)


def main() -> None:
    t0 = time.time()
    table = load_raw_pcos(DATA_DIR / "PCOS_data_without_infertility.xlsx",
                          DATA_DIR / "PCOS_infertility.csv")
    print(f"Loaded {table.shape[0]} rows x {table.shape[1] - 1} predictors; "
          f"PCOS share = {table[LABEL].mean():.3f}; missing cells = {int(table.isna().sum().sum())}")

    features, target = table[INPUT_COLUMNS], table[LABEL].astype(int)
    X_fit, X_hold, y_fit, y_hold = train_test_split(
        features, target, test_size=HOLDOUT_SHARE, stratify=target, random_state=SEED)
    print(f"Split: {len(X_fit)} train / {len(X_hold)} hold-out "
          f"(PCOS {y_fit.mean():.3f} / {y_hold.mean():.3f})")

    holdout_frame = X_hold.copy()
    holdout_frame[LABEL] = y_hold
    holdout_frame.to_csv(ROOT / "test_data.csv", index=False)

    folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    MODEL_DIR.mkdir(exist_ok=True)
    scoreboard = {}

    for label, (estimator, grid, fname) in CANDIDATES.items():
        base = Pipeline([("features", make_feature_stage(RobustScaler())),
                         ("classifier", estimator)])
        search = GridSearchCV(base, grid, scoring="matthews_corrcoef", cv=folds,
                              n_jobs=-1, refit=True)
        search.fit(X_fit, y_fit)
        tuned = search.best_estimator_
        joblib.dump(tuned, MODEL_DIR / fname, compress=3)

        result = score_holdout(tuned, X_hold, y_hold)
        result["TrainAccuracy"] = accuracy_score(y_fit, tuned.predict(X_fit))
        result["CV_MCC"] = float(search.best_score_)
        result["BestParams"] = describe_params(search.best_params_)
        result["File"] = fname
        scoreboard[label] = result
        print(f"{label:<24} CV-MCC={search.best_score_:.4f}  "
              + "  ".join(f"{k}={result[k]:.4f}" for k in
                          ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"])
              + f"  | {result['BestParams']}")

    meta = {
        "seed": SEED, "holdout_share": HOLDOUT_SHARE,
        "n_train": len(X_fit), "n_test": len(X_hold),
        "input_columns": INPUT_COLUMNS, "label": LABEL,
        "sklearn_version": sklearn.__version__,
        "cv": "RepeatedStratifiedKFold(5 folds x 3 repeats), scoring = MCC",
        "models": scoreboard,
    }
    (MODEL_DIR / "scoreboard.json").write_text(json.dumps(meta, indent=2, default=float))

    board = pd.DataFrame(scoreboard).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    print("\nHold-out comparison table\n" + board.astype(float).round(4).to_string())
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
