"""
model/train.py
---------------
One script, three jobs:

  1. Get the WDBC tumor-signal data -- from a local cache if one already
     exists, otherwise fetch it once and write the cache (see
     `get_wdbc_df` below).
  2. Fit all six mandated classifiers on an identical 80/20 stratified
     split and score each on the six required metrics.
  3. Persist ONE bundle file (`model/model_bundle.joblib`) containing the
     shared scaler, feature order, all five fitted models, and the metrics
     table -- plus `test_data.csv` at the repo root, the held-out split the
     Streamlit app expects to be uploaded.

Run:
    python model/train.py
"""

import os

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.dirname(THIS_FOLDER)

DATA_CACHE_PATH = os.path.join(THIS_FOLDER, "wdbc_raw_cache.csv")   # local cache
MODEL_BUNDLE_PATH = os.path.join(THIS_FOLDER, "model_bundle.joblib")  # single output
TARGET_COLUMN = "diagnosis"
RAND_SEED = 1974
TEST_SPLIT_PROPORTION = 0.2

MODEL_REGISTRY = {
    "Logistic Regression": LogisticRegression(max_iter=4500, random_state=RAND_SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RAND_SEED),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Gaussian Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=320, max_depth=8, random_state=RAND_SEED
    ),
}


def _fetch_data_from_uci() -> pd.DataFrame:
    """Live pull from the UCI archive (dataset id 17) via ucimlrepo."""
    from ucimlrepo import fetch_ucirepo

    wdbc = fetch_ucirepo(id=17)
    frame = wdbc.data.features.copy()
    label_col = wdbc.data.targets.columns[0]
    frame[TARGET_COLUMN] = wdbc.data.targets[label_col].map({"M": 1, "B": 0}).values
    return frame


def _fetch_data_from_sklearn_mirror() -> pd.DataFrame:
    """Identical WDBC records bundled with scikit-learn, used only when the
    UCI archive can't be reached (e.g. a locked-down sandbox)."""
    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.frame.copy()
    frame[TARGET_COLUMN] = 1 - frame["target"]   # 1 = malignant, matching UCI branch
    return frame.drop(columns=["target"])


def get_wdbc_df() -> pd.DataFrame:
    """Return the WDBC dataframe, fetching it over the network only the
    first time. Every subsequent run reads the local CSV cache instead."""
    if os.path.exists(DATA_CACHE_PATH):
        print(f"[train] Using cached dataset -> {DATA_CACHE_PATH}")
        return pd.read_csv(DATA_CACHE_PATH)

    try:
        frame = _fetch_data_from_uci()
        print("[train] Fetched dataset live from UCI (ucimlrepo, id=17).")
    except Exception as exc:  # noqa: BLE001
        print(f"[train] UCI fetch unavailable ({exc}); using local sklearn mirror.")
        frame = _fetch_data_from_sklearn_mirror()

    frame.to_csv(DATA_CACHE_PATH, index=False)
    print(f"[train] Cached raw dataset -> {DATA_CACHE_PATH} (deleted this file to re-fetch)")
    return frame


def score_bundle(y_true, y_pred, y_score) -> dict:
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_score), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def main():
    frame = get_wdbc_df()
    feature_cols = [c for c in frame.columns if c != TARGET_COLUMN]

    X = frame[feature_cols].astype(float)
    y = frame[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_PROPORTION, random_state=RAND_SEED, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    fitted_models, metrics = {}, {}
    for label, estimator in MODEL_REGISTRY.items():
        estimator.fit(X_train_scaled, y_train)
        preds = estimator.predict(X_test_scaled)
        scores = (
            estimator.predict_proba(X_test_scaled)[:, 1]
            if hasattr(estimator, "predict_proba")
            else estimator.decision_function(X_test_scaled)
        )
        metrics[label] = score_bundle(y_test, preds, scores)
        fitted_models[label] = estimator
        print(f"[train] {label:<28} -> {metrics[label]}")

    joblib.dump(
        {
            "scaler": scaler,
            "feature_order": feature_cols,
            "models": fitted_models,
            "metrics": metrics,
        },
        MODEL_BUNDLE_PATH,
    )
    print(f"\n[train] Wrote single bundle (scaler + {len(fitted_models)} models + metrics) -> {MODEL_BUNDLE_PATH}")

    test_export = X_test.copy()
    test_export[TARGET_COLUMN] = y_test.values
    test_csv_path = os.path.join(REPOSITORY_ROOT, "test_data.csv")
    test_export.to_csv(test_csv_path, index=False)
    print(f"[train] Wrote held-out test split ({len(test_export)} rows) -> {test_csv_path}")

    print("\n[train] Markdown metrics table (paste into README.md):\n")
    header = "| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |"
    sep = "|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for label, m in metrics.items():
        print(f"| {label} | {m['Accuracy']} | {m['AUC']} | {m['Precision']} | {m['Recall']} | {m['F1']} | {m['MCC']} |")


if __name__ == "__main__":
    main()
