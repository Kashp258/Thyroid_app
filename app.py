"""
Thyroid Disorder Detection — Streamlit Demo
Stacking Ensemble: RF + XGB → LogisticRegression meta-learner
Pipeline exactly matches 02.ipynb (Steps 1–19)
"""
import sys
import streamlit as st

st.write(sys.version)
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Thyroid Disorder Detector",
    page_icon="🦋",
    layout="wide",
)

# ─────────────────────────────────────────────
# CONSTANTS — exactly from notebook
# ─────────────────────────────────────────────
FEATURES    = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
SEED        = 42

# Best hyperparameters from Bayesian optimisation (Steps 5 & 13)
RF_PARAMS = dict(
    n_estimators=289,
    max_depth=6,
    min_samples_split=8,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=SEED,
)
XGB_PARAMS = dict(
    n_estimators=395,
    max_depth=5,
    learning_rate=0.0186,
    subsample=0.72,
    colsample_bytree=0.71,
    eval_metric="logloss",
    random_state=SEED,
)

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# Mirrors Steps 1–2 of 02.ipynb exactly
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading harmonised dataset…")
def load_data(path: str = "harmonised_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["source"], errors="ignore")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    return df


# ─────────────────────────────────────────────
# MODEL TRAINING
# Mirrors Steps 6–7 (Stacking) + Step 13 (Robustness params)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training stacking ensemble…")
def build_model(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    rf  = RandomForestClassifier(**RF_PARAMS, n_jobs=-1)
    xgb = XGBClassifier(**XGB_PARAMS, n_jobs=-1)

    stack = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=LogisticRegression(
            penalty="l2", C=1.0, max_iter=1000, class_weight="balanced"
        ),
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )

    stack.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = stack.predict(X_test)
    y_prob = stack.predict_proba(X_test)[:, 1]

    metrics = {
        "AUC":       round(roc_auc_score(y_test, y_prob), 4),
        "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred),    4),
        "F1":        round(f1_score(y_test, y_pred),        4),
    }

    return stack, X_test, y_test, metrics


# ─────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────
def predict_single(model, values: dict):
    """Takes a dict of {feature: value} and returns (label, prob)."""
    X = pd.DataFrame([values], columns=FEATURES)
    prob  = model.predict_proba(X)[0, 1]
    label = int(prob >= 0.5)
    return label, prob


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
FEATURE_META = {
    "age": dict(
        label="Age (years)",
        min_value=0.01, max_value=455.0,
        value=35.0, step=1.0,
        help="Patient age in years. Dataset range: 0.01–455",
    ),
    "TSH": dict(
        label="TSH (mIU/L)",
        min_value=0.0, max_value=478.0,
        value=1.3, step=0.01,
        help="Thyroid Stimulating Hormone. Normal range ≈ 0.4–4.0 mIU/L",
    ),
    "T3": dict(
        label="T3 (nmol/L)",
        min_value=0.0005, max_value=10.6,
        value=1.9, step=0.01,
        help="Triiodothyronine. Normal range ≈ 1.1–2.6 nmol/L",
    ),
    "TT4": dict(
        label="TT4 (nmol/L)",
        min_value=0.002, max_value=430.0,
        value=100.0, step=1.0,
        help="Total Thyroxine. Normal range ≈ 60–160 nmol/L",
    ),
    "T4U": dict(
        label="T4U (ratio)",
        min_value=0.017, max_value=2.12,
        value=0.97, step=0.01,
        help="Thyroxine Uptake ratio. Normal ≈ 0.9–1.1",
    ),
    "FTI": dict(
        label="FTI (Free Thyroxine Index)",
        min_value=0.002, max_value=395.0,
        value=106.0, step=1.0,
        help="Free Thyroxine Index. Normal range ≈ 70–130",
    ),
}


def render_result(label: int, prob: float):
    if label == 1:
        st.error("🔴 **Prediction: ABNORMAL (Thyroid Disorder Detected)**")
    else:
        st.success("🟢 **Prediction: NORMAL (No Disorder Detected)**")

    col_a, col_b = st.columns(2)
    col_a.metric("Predicted Class", "Abnormal (1)" if label else "Normal (0)")
    col_b.metric("Abnormality Probability", f"{prob:.4f} ({prob*100:.1f}%)")

    # Probability bar
    st.progress(float(prob), text=f"Risk score: {prob:.4f}")


# ═══════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════
def main():
    # ── Header ──────────────────────────────────────────────
    st.title("🦋 Thyroid Disorder Detection")
    st.caption(
        "Explainable Stacking Ensemble · RF + XGB → Logistic Regression · "
        "Trained on Harmonised UCI Thyroid Dataset (n = 12 800)"
    )
    st.divider()

    # ── Load data ───────────────────────────────────────────
    try:
        df = load_data("harmonised_dataset.csv")
    except FileNotFoundError:
        st.error(
            "⚠️ `harmonised_dataset.csv` not found.  "
            "Place it in the same directory as `app.py` and re-run."
        )
        st.stop()

    # ── Train model ─────────────────────────────────────────
    model, X_test, y_test, test_metrics = build_model(df)

    # ── Sidebar — model card ─────────────────────────────────
    with st.sidebar:
        st.header("📊 Model Card")
        st.caption("Stacking Ensemble — test-set performance")
        for k, v in test_metrics.items():
            st.metric(k, v)

        st.divider()
        st.subheader("Pipeline Summary")
        st.markdown(
            """
| Step | Detail |
|------|--------|
| Features | age, TSH, T3, TT4, T4U, FTI |
| Label | 0 = Normal · 1 = Abnormal |
| Base models | Random Forest + XGBoost |
| Meta-learner | Logistic Regression |
| Train/test | 80 / 20 stratified |
| Dataset | 12 800 patients |
            """
        )
        st.divider()
        st.caption("Hyperparameters (Bayesian-tuned via Optuna, Step 5):")
        with st.expander("RF params"):
            st.json(RF_PARAMS)
        with st.expander("XGB params"):
            st.json(XGB_PARAMS)

    # ── Tabs ────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        ["🔬 Manual Input", "📋 Sample from Dataset", "📈 Dataset Overview"]
    )

    # ════════════════════════════════════════════
    # TAB 1 — Manual Input
    # ════════════════════════════════════════════
    with tab1:
        st.subheader("Enter Patient Biomarkers")
        st.caption(
            "All values are in the original clinical units used in the harmonised dataset."
        )

        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        input_vals = {}
        for i, feat in enumerate(FEATURES):
            meta = FEATURE_META[feat]
            with cols[i % 3]:
                input_vals[feat] = st.number_input(
                    label=meta["label"],
                    min_value=float(meta["min_value"]),
                    max_value=float(meta["max_value"]),
                    value=float(meta["value"]),
                    step=float(meta["step"]),
                    help=meta["help"],
                    key=f"manual_{feat}",
                )

        st.divider()

        if st.button("🔍 Predict", type="primary", use_container_width=True, key="btn_manual"):
            label, prob = predict_single(model, input_vals)
            render_result(label, prob)

            with st.expander("Input summary"):
                input_df = pd.DataFrame([input_vals])
                st.dataframe(input_df, use_container_width=True)

    # ════════════════════════════════════════════
    # TAB 2 — Sample from Dataset
    # ════════════════════════════════════════════
    with tab2:
        st.subheader("Load a Sample from the Harmonised Dataset")

        col_a, col_b, col_c = st.columns([2, 2, 1])
        with col_a:
            sample_class = st.selectbox(
                "Filter by true class",
                options=["All", "Normal (0)", "Abnormal (1)"],
                key="sample_class",
            )
        with col_b:
            sample_idx = st.number_input(
                "Sample index (within filtered subset)",
                min_value=0, max_value=500, value=0, step=1,
                key="sample_idx",
            )
        with col_c:
            random_btn = st.button("🎲 Random", use_container_width=True, key="btn_random")

        # Filter
        if sample_class == "Normal (0)":
            subset = df[df["class"] == 0].reset_index(drop=True)
        elif sample_class == "Abnormal (1)":
            subset = df[df["class"] == 1].reset_index(drop=True)
        else:
            subset = df.reset_index(drop=True)

        if random_btn:
            chosen_idx = int(np.random.randint(0, len(subset)))
        else:
            chosen_idx = min(int(sample_idx), len(subset) - 1)

        row = subset.iloc[chosen_idx]
        sample_vals = {f: float(row[f]) for f in FEATURES}
        true_label  = int(row["class"])

        st.divider()
        st.markdown(f"**Sample #{chosen_idx}** — True label: "
                    f"{'🔴 Abnormal (1)' if true_label else '🟢 Normal (0)'}")

        # Display feature values
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        for i, feat in enumerate(FEATURES):
            with [feat_col1, feat_col2, feat_col3][i % 3]:
                st.metric(FEATURE_META[feat]["label"], f"{sample_vals[feat]:.4f}")

        st.divider()

        if st.button("🔍 Predict Sample", type="primary",
                     use_container_width=True, key="btn_sample"):
            label, prob = predict_single(model, sample_vals)
            render_result(label, prob)

            match = label == true_label
            if match:
                st.success(f"✅ Correct prediction! Model matched the true label ({true_label}).")
            else:
                st.warning(
                    f"⚠️ Mismatch — True: {true_label}, Predicted: {label}. "
                    "This is a misclassified case."
                )

    # ════════════════════════════════════════════
    # TAB 3 — Dataset Overview
    # ════════════════════════════════════════════
    with tab3:
        st.subheader("Harmonised Dataset Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Samples", f"{len(df):,}")
        c2.metric("Features", len(FEATURES))
        c3.metric("Normal (0)", f"{(df['class']==0).sum():,}")
        c4.metric("Abnormal (1)", f"{(df['class']==1).sum():,}")

        st.divider()
        st.markdown("**Descriptive Statistics — Clinical Features**")
        st.dataframe(
            df[FEATURES + ["class"]].describe().round(4),
            use_container_width=True,
        )

        st.divider()
        st.markdown("**Feature-level Statistics by Class**")
        st.dataframe(
            df.groupby("class")[FEATURES].median().round(4),
            use_container_width=True,
        )

        st.divider()
        st.markdown("**Raw Data Preview (first 50 rows)**")
        st.dataframe(df[FEATURES + ["class"]].head(50), use_container_width=True)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
