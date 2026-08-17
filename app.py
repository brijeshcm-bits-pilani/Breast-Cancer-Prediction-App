"""
Onco-Signal Lab -- Breast Mass Diagnostic Assistant
====================================================
A Streamlit front-end for comparing five classifiers trained on the
Wisconsin Diagnostic Breast Cancer (WDBC) tumor-signal dataset.

Upload the held-out `test_data.csv`, pick a model from the sidebar, and
the app scores that model on the uploaded rows live -- accuracy, AUC,
precision, recall, F1, MCC, plus a confusion matrix and full
classification report.

Loads a single artifact: model/model_bundle.joblib (built by
model/train.py), which holds the shared scaler, feature order, all five
fitted models, and their held-out metrics.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_BUNDLE_PATH = os.path.join(THIS_FOLDER, "model", "model_bundle.joblib")
TARGET_COLUMN = "diagnosis"

PAGE_ACCENT = "#7A5CFA"
PAGE_ACCENT_SOFT = "#F1EEFF"

st.set_page_config(page_title="Onco-Signal Lab", page_icon="🩺", layout="wide")

st.markdown(
    f"""
    <style>
        html, body, [class*="css"] {{ font-family: 'Helvetica Neue', sans-serif; }}
        .os-hero {{
            padding: 1.6rem 2rem; border-radius: 18px;
            background: linear-gradient(135deg, {PAGE_ACCENT} 0%, #4E3AC7 100%);
            color: white; margin-bottom: 1.2rem;
        }}
        .os-hero h1 {{ margin-bottom: 0.2rem; font-size: 1.9rem; }}
        .os-hero p {{ margin: 0; opacity: 0.9; }}
        .os-metric-card {{ background: {PAGE_ACCENT_SOFT}; border-radius: 14px; padding: 0.9rem 0.4rem; text-align: center; }}
        .os-metric-card .val {{ font-size: 1.5rem; font-weight: 700; color: #3A2C99; }}
        .os-metric-card .lbl {{ font-size: 0.78rem; color: #5A4FA0; text-transform: uppercase; letter-spacing: 0.04em; }}
        section[data-testid="stSidebar"] {{ background-color: #FAF9FF; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_BUNDLE_PATH):
        st.error(
            "model/model_bundle.joblib not found. Run `python model/train.py` "
            "first to fetch the data and train all five models."
        )
        st.stop()
    return joblib.load(MODEL_BUNDLE_PATH)


def render_hero():
    st.markdown(
        """
        <div class="os-hero">
            <h1>Onco-Signal Lab</h1>
            <p>A side-by-side diagnostic assistant benchmarking five classifiers on
            fine-needle-aspirate tumor measurements (WDBC dataset).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(scores: dict):
    cols = st.columns(len(scores))
    for col, (label, value) in zip(cols, scores.items()):
        with col:
            st.markdown(
                f'<div class="os-metric-card"><div class="val">{value:.3f}</div>'
                f'<div class="lbl">{label}</div></div>',
                unsafe_allow_html=True,
            )


def score_bundle(y_true, y_pred, y_score) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    render_hero()
    model_bundle = load_model_bundle()
    scaler = model_bundle["scaler"]
    feature_order = model_bundle["feature_order"]
    models = model_bundle["models"]

    with st.sidebar:
        st.subheader("Controls")
        chosen_model_label = st.selectbox("Choose a classifier", list(models.keys()))
        st.caption(
            "Every model was trained on the same 80/20 stratified split of the "
            "WDBC dataset (30 nuclear-morphology features, 569 records)."
        )
        uploaded = st.file_uploader("Upload test_data.csv", type=["csv"])
        st.markdown("---")
        st.caption(
            "1 = malignant, 0 = benign. Only the held-out `test_data.csv` "
            "produced by `model/train.py` should be uploaded here, to keep "
            "the Streamlit free-tier footprint small."
        )

    if uploaded is None:
        st.info("Upload `test_data.csv` from the sidebar to see live results.")
        st.stop()

    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read that file as CSV: {exc}")
        st.stop()

    missing = [c for c in feature_order + [TARGET_COLUMN] if c not in raw.columns]
    if missing:
        st.error(f"Uploaded CSV is missing required column(s): {missing}")
        st.stop()

    X = raw[feature_order].astype(float)
    y_true = raw[TARGET_COLUMN].astype(int)
    X_scaled = scaler.transform(X)

    model = models[chosen_model_label]
    y_pred = model.predict(X_scaled)
    y_score = (
        model.predict_proba(X_scaled)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_scaled)
    )

    st.subheader(f"Results · {chosen_model_label}")
    render_metric_row(score_bundle(y_true, y_pred, y_score))

    left, right = st.columns([1, 1.3])

    with left:
        st.markdown("##### Confusion matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=["Benign (0)", "Malignant (1)"],
            yticklabels=["Benign (0)", "Malignant (1)"],
            cbar=False, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.markdown("##### Classification report")
        report_dict = classification_report(
            y_true, y_pred, target_names=["Benign", "Malignant"], output_dict=True
        )
        report_df = pd.DataFrame(report_dict).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    with st.expander("Row-level predictions"):
        preview = raw[feature_order[:4]].copy()
        preview["actual"] = y_true.map({0: "Benign", 1: "Malignant"})
        preview["predicted"] = pd.Series(y_pred).map({0: "Benign", 1: "Malignant"})
        preview["malignant_probability"] = np.round(y_score, 3)
        st.dataframe(preview, use_container_width=True)

    with st.expander("Compare all five models on this upload"):
        rows = []
        for label, m in models.items():
            p = m.predict(X_scaled)
            s = (
                m.predict_proba(X_scaled)[:, 1]
                if hasattr(m, "predict_proba")
                else m.decision_function(X_scaled)
            )
            rows.append({"Model": label, **score_bundle(y_true, p, s)})
        comparison = pd.DataFrame(rows).set_index("Model").round(4)
        st.dataframe(comparison, use_container_width=True)


if __name__ == "__main__":
    main()
