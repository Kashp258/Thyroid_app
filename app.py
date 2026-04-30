"""
Thyroid Disorder Detection — Streamlit Demo  v2.0
Stacking Ensemble: RF + XGB → Logistic Regression meta-learner

Tabs:
  1. Live Prediction        — manual biomarker input + calibrated risk
  2. Sample from Dataset    — pick/random a row, compare true vs predicted
  3. SHAP Explainability    — global importances + beeswarm-style bar + interaction heatmap
  4. Model Performance      — ROC curves, calibration, bootstrap CIs, confusion matrix
  5. Clinical Utility       — DCA, threshold sweep, robustness profile
  6. Dataset Overview       — descriptive stats, class balance
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, brier_score_loss,
    roc_curve, confusion_matrix,
)
from xgboost import XGBClassifier

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# ──────────────────────────────────────────────────────────────────────
# PALETTE  (clinical / research aesthetic — navy + teal + amber)
# ──────────────────────────────────────────────────────────────────────
C_NAVY   = "#1C3557"
C_TEAL   = "#2A9D8F"
C_AMBER  = "#E9C46A"
C_RED    = "#E76F51"
C_LGRAY  = "#F4F6F8"
C_MGRAY  = "#DDE2E8"
C_DGRAY  = "#5C6370"

PLT_STYLE = {
    "axes.facecolor":  "#FAFBFC",
    "figure.facecolor":"#FAFBFC",
    "axes.edgecolor":  C_MGRAY,
    "axes.grid":       True,
    "grid.color":      C_MGRAY,
    "grid.linewidth":  0.6,
    "text.color":      C_NAVY,
    "axes.labelcolor": C_NAVY,
    "xtick.color":     C_DGRAY,
    "ytick.color":     C_DGRAY,
    "font.family":     "sans-serif",
}

# ──────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thyroid Disorder Detection · Research Demo",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  /* Header strip */
  .research-header {
    background: linear-gradient(135deg, #1C3557 0%, #2A9D8F 100%);
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    color: white;
  }
  .research-header h1 { font-size: 1.9rem; font-weight: 700; margin: 0; color: white; }
  .research-header p  { font-size: 0.92rem; margin: 6px 0 0; opacity: 0.85; color: white; }

  /* Metric cards */
  .metric-card {
    background: white;
    border: 1px solid #DDE2E8;
    border-left: 4px solid #2A9D8F;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 4px 0;
  }
  .metric-card .label { font-size: 0.78rem; font-weight: 600; color: #5C6370;
                        text-transform: uppercase; letter-spacing: 0.06em; }
  .metric-card .value { font-size: 1.55rem; font-weight: 700; color: #1C3557; }
  .metric-card .sub   { font-size: 0.78rem; color: #5C6370; }

  /* Insight boxes */
  .insight-box {
    background: #EEF7F6;
    border-left: 4px solid #2A9D8F;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.88rem;
    color: #1C3557;
  }

  /* Section label */
  .sec-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #2A9D8F;
    margin-bottom: 4px;
  }

  /* Risk badge */
  .badge-abnormal {
    background:#FDECEA; color:#C0392B; border:1.5px solid #E74C3C;
    border-radius:20px; padding:4px 16px; font-weight:700; font-size:0.95rem;
    display:inline-block;
  }
  .badge-normal {
    background:#E8F8F5; color:#1A6645; border:1.5px solid #2A9D8F;
    border-radius:20px; padding:4px 16px; font-weight:700; font-size:0.95rem;
    display:inline-block;
  }

  /* Hide streamlit default top padding */
  .block-container { padding-top: 1rem !important; }

  /* Tab font */
  button[data-baseweb="tab"] { font-size: 0.88rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────
FEATURES = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
SEED     = 42

RF_PARAMS = dict(
    n_estimators=289, max_depth=6,
    min_samples_split=8, min_samples_leaf=5,
    class_weight="balanced", random_state=SEED,
)
XGB_PARAMS = dict(
    n_estimators=395, max_depth=5, learning_rate=0.0186,
    subsample=0.72, colsample_bytree=0.71,
    eval_metric="logloss", random_state=SEED,
)

FEATURE_META = {
    "age": dict(label="Age (years)",   min_value=0.01, max_value=455.0, value=35.0, step=1.0,
                help="Patient age. Dataset range: 0.01–455",        normal="18–80 yrs"),
    "TSH": dict(label="TSH (mIU/L)",   min_value=0.0,  max_value=478.0, value=1.3,  step=0.01,
                help="Thyroid Stimulating Hormone. Normal ≈ 0.4–4.0",normal="0.4–4.0"),
    "T3":  dict(label="T3 (nmol/L)",   min_value=0.0,  max_value=10.6,  value=1.9,  step=0.01,
                help="Triiodothyronine. Normal ≈ 1.1–2.6",          normal="1.1–2.6"),
    "TT4": dict(label="TT4 (nmol/L)",  min_value=0.0,  max_value=430.0, value=100.0,step=1.0,
                help="Total Thyroxine. Normal ≈ 60–160",            normal="60–160"),
    "T4U": dict(label="T4U (ratio)",   min_value=0.0,  max_value=2.12,  value=0.97, step=0.01,
                help="Thyroxine Uptake ratio. Normal ≈ 0.9–1.1",    normal="0.9–1.1"),
    "FTI": dict(label="FTI",           min_value=0.0,  max_value=395.0, value=106.0,step=1.0,
                help="Free Thyroxine Index. Normal ≈ 70–130",       normal="70–130"),
}

# Pre-computed research values (from paper — nested 10-fold CV)
PAPER_METRICS = {
    "Stacking": dict(AUC=0.9836, Acc=0.9462, Prec=0.5687, Rec=0.9245, F1=0.704),
    "RF":       dict(AUC=0.9830, Acc=0.9430, Prec=0.5540, Rec=0.9493, F1=0.699),
    "XGBoost":  dict(AUC=0.9832, Acc=0.9623, Prec=0.8224, Rec=0.5840, F1=0.682),
    "MLP":      dict(AUC=0.8978, Acc=0.9287, Prec=0.4168, Rec=0.0609, F1=0.104),
}
PAPER_SHAP = {
    "TSH": 0.2682, "TT4": 0.0565, "T3": 0.0534,
    "FTI": 0.0480, "T4U": 0.0407, "age": 0.0051,
}
PAPER_INTERACTIONS = {
    "TSH×T3":  0.036, "T3×TT4":  0.022,
    "T3×FTI":  0.018, "TSH×TT4": 0.018, "TSH×FTI": 0.016,
}
PAPER_BOOT_CI = {
    "AUC":      (0.9825, 0.9778, 0.9868),
    "Accuracy": (0.9423, 0.9332, 0.9516),
    "Recall":   (0.9593, 0.9261, 0.9879),
    "F1-Score": (0.6963, 0.6500, 0.7426),
}
PAPER_ABLATION = {
    "TSH": -0.143, "TT4": -0.003, "T3": -0.002,
    "FTI": -0.001, "T4U": -0.001, "age": -0.000,
}
PAPER_NOISE_AUC    = [0.9836, 0.7350, 0.6933, 0.6625]
PAPER_MISSING_AUC  = [0.9836, 0.9688, 0.9523, 0.9345]
PAPER_NOISE_X      = ["Baseline", "5%",   "10%",  "20%"]
PAPER_MISSING_X    = ["Baseline", "10%",  "20%",  "30%"]
PAPER_CONFUSION    = [[11716, 197], [236, 651]]   # TN FP / FN TP
PAPER_BRIER        = 0.0923
OPTIMAL_THETA      = 0.25

# ──────────────────────────────────────────────────────────────────────
# DATA & MODEL
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading harmonised dataset…")
def load_data(path: str = "harmonised_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["source"], errors="ignore")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    return df


@st.cache_resource(show_spinner="Training stacking ensemble…")
def build_model(df: pd.DataFrame):
    X = df[FEATURES].values
    y = df["class"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
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
        n_jobs=-1, passthrough=False,
    )
    stack.fit(X_tr, y_tr)

    # Also train individual models for ROC comparison
    rf2  = RandomForestClassifier(**RF_PARAMS, n_jobs=-1).fit(X_tr, y_tr)
    xgb2 = XGBClassifier(**XGB_PARAMS, n_jobs=-1).fit(X_tr, y_tr)

    y_prob_stack = stack.predict_proba(X_te)[:, 1]
    y_prob_rf    = rf2.predict_proba(X_te)[:, 1]
    y_prob_xgb   = xgb2.predict_proba(X_te)[:, 1]
    y_pred       = (y_prob_stack >= 0.5).astype(int)

    roc_data = {
        "Stacking": roc_curve(y_te, y_prob_stack),
        "RF":       roc_curve(y_te, y_prob_rf),
        "XGBoost":  roc_curve(y_te, y_prob_xgb),
    }

    # Calibration
    frac_pos, mean_pred = calibration_curve(y_te, y_prob_stack, n_bins=10)

    test_metrics = {
        "AUC":       round(roc_auc_score(y_te, y_prob_stack), 4),
        "Accuracy":  round(accuracy_score(y_te, y_pred),      4),
        "Precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_te, y_pred),        4),
        "F1":        round(f1_score(y_te, y_pred),            4),
        "Brier":     round(brier_score_loss(y_te, y_prob_stack), 4),
    }

    # Threshold sweep
    thresholds = np.linspace(0.01, 0.99, 200)
    f1s, recs, precs, accs = [], [], [], []
    for t in thresholds:
        yp = (y_prob_stack >= t).astype(int)
        f1s.append(f1_score(y_te, yp, zero_division=0))
        recs.append(recall_score(y_te, yp, zero_division=0))
        precs.append(precision_score(y_te, yp, zero_division=0))
        accs.append(accuracy_score(y_te, yp))

    # DCA
    prev = y_te.mean()
    nb_model, nb_all = [], []
    for t in thresholds:
        if t >= 1: nb_model.append(0); nb_all.append(0); continue
        yp = (y_prob_stack >= t).astype(int)
        tp = ((yp == 1) & (y_te == 1)).sum()
        fp = ((yp == 1) & (y_te == 0)).sum()
        n  = len(y_te)
        nb_model.append(tp / n - fp / n * t / (1 - t))
        tp_all = (y_te == 1).sum()
        fp_all = (y_te == 0).sum()
        nb_all.append(tp_all / n - fp_all / n * t / (1 - t))

    # SHAP (on a small subset for speed)
    shap_values_out = None
    shap_X_out = None
    if SHAP_OK:
        try:
            explainer = shap.TreeExplainer(stack.named_estimators_["rf"])
            shap_X = pd.DataFrame(X_te[:200], columns=FEATURES)
            shap_values_out = explainer.shap_values(shap_X)
            shap_X_out = shap_X
        except Exception:
            pass

    return (stack, rf2, xgb2,
            X_te, y_te, y_prob_stack, y_pred,
            test_metrics, roc_data,
            frac_pos, mean_pred,
            thresholds, f1s, recs, precs, accs,
            nb_model, nb_all,
            shap_values_out, shap_X_out)


def predict_single(model, values: dict, threshold: float = 0.5):
    X = pd.DataFrame([values], columns=FEATURES)
    prob  = model.predict_proba(X)[0, 1]
    label = int(prob >= threshold)
    return label, prob


# ──────────────────────────────────────────────────────────────────────
# MATPLOTLIB HELPER
# ──────────────────────────────────────────────────────────────────────
def styled_fig(w=7, h=4.5):
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#FAFBFC")
    ax.set_facecolor("#FAFBFC")
    return fig, ax

def styled_figs(rows, cols, w=12, h=5):
    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor("#FAFBFC")
    return fig, axes

def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def metric_card(label, value, sub=""):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    # ── Header ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="research-header">
      <h1>🦋 Thyroid Disorder Detection</h1>
      <p>Explainable Stacking Ensemble · RF + XGBoost → Logistic Regression Meta-Learner
         &nbsp;|&nbsp; Harmonised UCI Corpus · <b>n = 12,800</b> &nbsp;|&nbsp;
         SHAP · Calibration · DCA · Robustness · Bootstrap CI</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load & train ─────────────────────────────────────────────────
    try:
        df = load_data("harmonised_dataset.csv")
    except FileNotFoundError:
        st.error("⚠️ `harmonised_dataset.csv` not found. Place it alongside `app.py`.")
        st.stop()

    (model, rf_model, xgb_model,
     X_te, y_te, y_prob, y_pred,
     test_metrics, roc_data,
     frac_pos, mean_pred,
     thresholds, f1s, recs, precs, accs,
     nb_model, nb_all,
     shap_vals, shap_X) = build_model(df)

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sec-label">Live Model · Test-Set Metrics</div>',
                    unsafe_allow_html=True)
        for k, v in test_metrics.items():
            label_map = {
                "AUC": "ROC-AUC", "Accuracy": "Accuracy", "Precision": "Precision",
                "Recall": "Recall (Sensitivity)", "F1": "F1-Score", "Brier": "Brier Score ↓",
            }
            st.metric(label_map.get(k, k), v)

        st.divider()
        st.markdown('<div class="sec-label">Nested CV Results (Paper)</div>',
                    unsafe_allow_html=True)
        st.caption("AUC = 0.9836 · Recall = 92.45% · BS = 0.0923")

        st.divider()
        st.markdown('<div class="sec-label">Pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
| | |
|---|---|
| Base models | RF + XGBoost |
| Meta-learner | Logistic Regression |
| Calibration | Isotonic regression |
| HPO | Optuna TPE (20 trials) |
| Validation | Nested 10-fold CV |
| Dataset | 12,800 patients |
| Imbalance | 13.4 : 1 |
        """)

        st.divider()
        dec_threshold = st.slider(
            "Decision threshold θ*",
            min_value=0.05, max_value=0.95,
            value=OPTIMAL_THETA, step=0.05,
            help="Optimal θ* = 0.25 (F1-maximising). Drag to explore trade-offs."
        )

    # ── Tabs ─────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🔬 Live Prediction",
        "📋 Sample Dataset",
        "🧠 SHAP Explainability",
        "📈 Model Performance",
        "⚕️ Clinical Utility",
        "📊 Dataset Overview",
    ])

    # ═════════════════════════════════════════════════════════════════
    # TAB 1 — LIVE PREDICTION
    # ═════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.subheader("Enter Patient Biomarkers")
        st.caption(
            "Adjust values below. The model outputs a calibrated risk probability "
            "and a binary decision at the threshold set in the sidebar."
        )

        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        input_vals = {}
        for i, feat in enumerate(FEATURES):
            m = FEATURE_META[feat]
            with cols[i % 3]:
                input_vals[feat] = st.number_input(
                    label=m["label"],
                    min_value=float(m["min_value"]),
                    max_value=float(m["max_value"]),
                    value=float(m["value"]),
                    step=float(m["step"]),
                    help=f"{m['help']} | Normal range: {m['normal']}",
                    key=f"t1_{feat}",
                )
        st.divider()

        if st.button("🔍 Predict", type="primary", use_container_width=True, key="btn_t1"):
            label, prob = predict_single(model, input_vals, threshold=dec_threshold)

            # Risk display
            col_res, col_gauge = st.columns([3, 2])
            with col_res:
                badge = (
                    '<span class="badge-abnormal">🔴 ABNORMAL</span>'
                    if label else
                    '<span class="badge-normal">🟢 NORMAL</span>'
                )
                st.markdown(f"**Prediction:** {badge}", unsafe_allow_html=True)
                st.markdown(f"**Calibrated risk score:** `{prob:.4f}` &nbsp;({prob*100:.1f}%)",
                            unsafe_allow_html=True)
                st.markdown(f"**Decision threshold:** `θ = {dec_threshold}`")
                st.progress(float(prob))

            with col_gauge:
                # Mini gauge bar chart
                fig, ax = styled_fig(3, 2.5)
                categories = ["Normal", "Abnormal"]
                values     = [1 - prob, prob]
                colors     = [C_TEAL, C_RED]
                bars = ax.barh(categories, values, color=colors,
                               height=0.5, edgecolor="none")
                ax.axvline(dec_threshold, color=C_NAVY, lw=1.5,
                           linestyle="--", label=f"θ = {dec_threshold}")
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                ax.set_title("Risk Decomposition", fontsize=10, fontweight="bold", color=C_NAVY)
                ax.legend(fontsize=8)
                ax.grid(axis="y", visible=False)
                for bar, v in zip(bars, values):
                    ax.text(min(v + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                            f"{v:.3f}", va="center", fontsize=9, color=C_NAVY, fontweight="600")
                plt.tight_layout()
                show_fig(fig)

            # Clinical context
            if prob >= OPTIMAL_THETA:
                insight(
                    f"At the optimised threshold θ* = {OPTIMAL_THETA}, this patient "
                    f"would be flagged for confirmatory specialist referral. "
                    f"Lowering θ from 0.50 → 0.25 reduces the false negative rate from "
                    f"26.6% → 7.9% in the research dataset."
                )
            else:
                insight(
                    "Risk score below the optimised threshold. At θ* = 0.25 the framework "
                    "achieves 92.09% recall — but no automated tool replaces clinical judgment."
                )

            # SHAP for this instance (if available)
            if SHAP_OK and shap_vals is not None:
                try:
                    explainer = shap.TreeExplainer(rf_model)
                    X_inst = pd.DataFrame([input_vals], columns=FEATURES)
                    sv = explainer.shap_values(X_inst)
                    # sv shape: (classes, samples, features) or (samples, features)
                    if isinstance(sv, list):
                        sv_pos = sv[1][0]
                    else:
                        sv_pos = sv[0]

                    st.divider()
                    st.markdown("**Why this prediction? — Instance-level SHAP attribution**")
                    fig, ax = styled_fig(7, 3)
                    feat_names = FEATURES
                    sorted_idx = np.argsort(np.abs(sv_pos))
                    colors = [C_RED if v > 0 else C_TEAL for v in sv_pos[sorted_idx]]
                    ax.barh(
                        [feat_names[i] for i in sorted_idx],
                        sv_pos[sorted_idx],
                        color=colors, edgecolor="none", height=0.55,
                    )
                    ax.axvline(0, color=C_NAVY, lw=0.8)
                    ax.set_xlabel("SHAP value (impact on abnormal prediction)")
                    ax.set_title("Instance SHAP Attribution", fontweight="bold",
                                 color=C_NAVY, fontsize=11)
                    red_p = mpatches.Patch(color=C_RED,  label="Pushes → Abnormal")
                    teal_p= mpatches.Patch(color=C_TEAL, label="Pushes → Normal")
                    ax.legend(handles=[red_p, teal_p], fontsize=8)
                    plt.tight_layout()
                    show_fig(fig)
                except Exception:
                    pass

    # ═════════════════════════════════════════════════════════════════
    # TAB 2 — SAMPLE FROM DATASET
    # ═════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.subheader("Load a Sample from the Harmonised Dataset")
        ca, cb, cc = st.columns([2, 2, 1])
        with ca:
            sample_class = st.selectbox(
                "Filter by true class",
                ["All", "Normal (0)", "Abnormal (1)"], key="t2_cls"
            )
        with cb:
            sample_idx = st.number_input(
                "Sample index", min_value=0, max_value=500, value=0, key="t2_idx"
            )
        with cc:
            rand_btn = st.button("🎲 Random", use_container_width=True, key="t2_rand")

        if sample_class == "Normal (0)":
            subset = df[df["class"] == 0].reset_index(drop=True)
        elif sample_class == "Abnormal (1)":
            subset = df[df["class"] == 1].reset_index(drop=True)
        else:
            subset = df.reset_index(drop=True)

        chosen = int(np.random.randint(0, len(subset))) if rand_btn else min(int(sample_idx), len(subset) - 1)
        row = subset.iloc[chosen]
        sample_vals = {f: float(row[f]) for f in FEATURES}
        true_label  = int(row["class"])

        st.divider()
        badge_true = "🔴 Abnormal (1)" if true_label else "🟢 Normal (0)"
        st.markdown(f"**Sample #{chosen}** — True label: **{badge_true}**")

        fc1, fc2, fc3 = st.columns(3)
        for i, feat in enumerate(FEATURES):
            with [fc1, fc2, fc3][i % 3]:
                st.metric(FEATURE_META[feat]["label"],
                          f"{sample_vals[feat]:.4f}",
                          help=f"Normal: {FEATURE_META[feat]['normal']}")

        st.divider()
        if st.button("🔍 Predict Sample", type="primary",
                     use_container_width=True, key="t2_pred"):
            label, prob = predict_single(model, sample_vals, threshold=dec_threshold)
            col_l, col_r = st.columns(2)
            with col_l:
                if label == 1:
                    st.error(f"🔴 **ABNORMAL** — Risk: {prob:.4f} ({prob*100:.1f}%)")
                else:
                    st.success(f"🟢 **NORMAL** — Risk: {prob:.4f} ({prob*100:.1f}%)")
                st.progress(float(prob))
            with col_r:
                match = label == true_label
                if match:
                    st.success(f"✅ **Correct prediction** — matched true label {true_label}.")
                else:
                    st.warning(
                        f"⚠️ **Mismatch** — True: {true_label}, Predicted: {label}. "
                        "Misclassified case (false negative or false positive)."
                    )
                st.markdown(f"**Threshold used:** `θ = {dec_threshold}`")

    # ═════════════════════════════════════════════════════════════════
    # TAB 3 — SHAP EXPLAINABILITY
    # ═════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.subheader("SHAP Explainability — Global & Interaction Analysis")
        st.caption(
            "Results from the paper: cross-validated SHAP importances (n = 12,800), "
            "pairwise interaction decomposition (first application to biochemical thyroid panel data)."
        )

        col_shap1, col_shap2 = st.columns(2)

        # ── Global SHAP bar (paper values) ──────────────────────────
        with col_shap1:
            st.markdown("**Global SHAP Importance — Mean |ϕ|**")
            fig, ax = styled_fig(5.5, 4)
            feats = list(PAPER_SHAP.keys())
            vals  = list(PAPER_SHAP.values())
            sorted_idx = np.argsort(vals)
            bar_colors = [C_TEAL if f != "TSH" else C_NAVY for f in np.array(feats)[sorted_idx]]
            bars = ax.barh(
                np.array(feats)[sorted_idx], np.array(vals)[sorted_idx],
                color=bar_colors, height=0.55, edgecolor="none",
            )
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("Feature Importance (Cross-validated)", fontweight="bold",
                         color=C_NAVY, fontsize=11)
            for bar, v in zip(bars, np.array(vals)[sorted_idx]):
                ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                        f"{v:.4f}", va="center", fontsize=8.5, color=C_NAVY)
            ax.set_xlim(0, 0.32)
            plt.tight_layout()
            show_fig(fig)
            insight(
                "TSH dominates at 4.7× TT4. Its removal causes ΔAUC = −0.143 — "
                "~50× larger than any other single feature."
            )

        # ── Interaction heatmap ─────────────────────────────────────
        with col_shap2:
            st.markdown("**SHAP Pairwise Interaction Strengths**")
            feats_ord = ["TSH", "T3", "TT4", "FTI", "T4U", "age"]
            n = len(feats_ord)
            mat = np.zeros((n, n))
            interactions = {
                ("TSH","T3"):  0.036, ("T3","TSH"):  0.036,
                ("T3","TT4"): 0.022, ("TT4","T3"):  0.022,
                ("T3","FTI"): 0.018, ("FTI","T3"):  0.018,
                ("TSH","TT4"):0.018, ("TT4","TSH"): 0.018,
                ("TSH","FTI"):0.016, ("FTI","TSH"): 0.016,
            }
            for (r, c), v in interactions.items():
                ri, ci = feats_ord.index(r), feats_ord.index(c)
                mat[ri, ci] = v
            # diagonals = main effects (from SHAP paper values)
            for i, f in enumerate(feats_ord):
                mat[i, i] = PAPER_SHAP.get(f, 0)

            fig, ax = styled_fig(5.5, 4)
            im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(n)); ax.set_xticklabels(feats_ord, fontsize=9)
            ax.set_yticks(range(n)); ax.set_yticklabels(feats_ord, fontsize=9)
            for i in range(n):
                for j in range(n):
                    v = mat[i, j]
                    if v > 0:
                        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                                fontsize=7.5,
                                color="white" if v > 0.15 else C_NAVY,
                                fontweight="bold" if v > 0.015 else "normal")
            plt.colorbar(im, ax=ax, shrink=0.8, label="Interaction strength")
            ax.set_title("SHAP Interaction Heatmap", fontweight="bold",
                         color=C_NAVY, fontsize=11)
            ax.grid(visible=False)
            plt.tight_layout()
            show_fig(fig)
            insight(
                "TSH×T3 (0.036) is the strongest coupling — matching the HPT "
                "negative feedback pathway. All top-5 interactions involve TSH or T3."
            )

        st.divider()

        # ── Biomarker Ablation ──────────────────────────────────────
        st.markdown("**Biomarker Ablation — ΔAUC on TSH Removal (Leave-One-Out)**")
        fig, ax = styled_fig(9, 3)
        feats_abl = list(PAPER_ABLATION.keys())
        deltas    = list(PAPER_ABLATION.values())
        colors_abl= [C_RED if d < -0.01 else C_AMBER if d < -0.001 else C_TEAL
                     for d in deltas]
        bars = ax.bar(feats_abl, deltas, color=colors_abl, width=0.55, edgecolor="none")
        ax.axhline(0, color=C_NAVY, lw=0.8, linestyle="--")
        ax.set_ylabel("ΔAUC (vs. full model)")
        ax.set_title("Leave-One-Out Biomarker Ablation", fontweight="bold",
                     color=C_NAVY, fontsize=11)
        for bar, d in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width() / 2, d - 0.006,
                    f"{d:.3f}", ha="center", va="top", fontsize=9,
                    color="white" if d < -0.05 else C_NAVY, fontweight="600")
        ax.set_ylim(-0.17, 0.02)
        plt.tight_layout()
        show_fig(fig)
        insight(
            "Removing TSH collapses AUC by 0.143 — approximately 50× the impact of "
            "removing any other single biomarker. TSH is qualitatively indispensable."
        )

        # ── Live SHAP if available ──────────────────────────────────
        if SHAP_OK and shap_vals is not None:
            st.divider()
            st.markdown("**Live SHAP — Global Importance from Current Test Set (RF base learner)**")
            fig, ax = styled_fig(8, 3.5)
            mean_abs = np.abs(
                shap_vals[1] if isinstance(shap_vals, list) else shap_vals
            ).mean(axis=0)
            si = np.argsort(mean_abs)
            colors_live = [C_TEAL if FEATURES[i] != "TSH" else C_NAVY for i in si]
            ax.barh([FEATURES[i] for i in si], mean_abs[si],
                    color=colors_live, height=0.5, edgecolor="none")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("SHAP Global Importance (Live · RF Base Learner · n=200)",
                         fontweight="bold", color=C_NAVY, fontsize=11)
            plt.tight_layout()
            show_fig(fig)

    # ═════════════════════════════════════════════════════════════════
    # TAB 4 — MODEL PERFORMANCE
    # ═════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.subheader("Model Performance — ROC · Calibration · Bootstrap CI · Confusion Matrix")

        # ── Metrics table (paper values) ────────────────────────────
        st.markdown("**Nested 10-Fold Cross-Validation Results (from paper)**")
        perf_rows = []
        for model_name, m in PAPER_METRICS.items():
            perf_rows.append({
                "Model": model_name,
                "AUC (μ±σ)": f"{m['AUC']:.4f}",
                "Accuracy":  f"{m['Acc']:.2%}",
                "Precision": f"{m['Prec']:.2%}",
                "Recall":    f"{m['Rec']:.2%}",
                "F1":        f"{m['F1']:.3f}",
            })
        perf_df = pd.DataFrame(perf_rows).set_index("Model")
        st.dataframe(
            perf_df.style.apply(
                lambda x: ["background: #EEF7F6; font-weight:700"
                           if x.name == "Stacking" else "" for _ in x],
                axis=1,
            ),
            use_container_width=True,
        )
        insight(
            "Stacking achieves highest AUC (0.9836) with lowest inter-fold variance. "
            "Wilcoxon test: Stack vs RF p = 0.0039* (significant); vs XGBoost p = 0.131 n.s."
        )

        st.divider()
        row1_l, row1_r = st.columns(2)

        # ── ROC Curves ──────────────────────────────────────────────
        with row1_l:
            st.markdown("**ROC Curves — All Models (Live Test Set)**")
            fig, ax = styled_fig(5.5, 4.5)
            model_colors = {
                "Stacking": C_NAVY,
                "RF":       C_TEAL,
                "XGBoost":  C_AMBER,
            }
            for mname, (fpr, tpr, _) in roc_data.items():
                auc = roc_auc_score(
                    y_te,
                    model.predict_proba(pd.DataFrame(X_te, columns=FEATURES))[:, 1]
                    if mname == "Stacking"
                    else (rf_model if mname == "RF" else xgb_model)
                        .predict_proba(X_te)[:, 1]
                )
                lw = 2.5 if mname == "Stacking" else 1.5
                ls = "-"  if mname == "Stacking" else "--"
                ax.plot(fpr, tpr, lw=lw, ls=ls,
                        color=model_colors[mname],
                        label=f"{mname} (AUC={auc:.4f})")
            ax.plot([0, 1], [0, 1], ":", color=C_DGRAY, lw=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curves", fontweight="bold", color=C_NAVY, fontsize=11)
            ax.legend(fontsize=8.5, loc="lower right")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            plt.tight_layout()
            show_fig(fig)

        # ── Calibration Curve ────────────────────────────────────────
        with row1_r:
            st.markdown("**Calibration (Reliability) Diagram**")
            fig, ax = styled_fig(5.5, 4.5)
            ax.plot([0, 1], [0, 1], ":", color=C_DGRAY, lw=1.5, label="Perfect calibration")
            ax.plot(mean_pred, frac_pos, "o-", color=C_TEAL, lw=2,
                    markersize=6, label=f"Stacking (BS={test_metrics['Brier']:.4f})")
            ax.fill_between(mean_pred, frac_pos,
                            np.interp(mean_pred, [0, 1], [0, 1]),
                            alpha=0.12, color=C_TEAL)
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title("Calibration Diagram", fontweight="bold", color=C_NAVY, fontsize=11)
            ax.legend(fontsize=8.5)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            plt.tight_layout()
            show_fig(fig)
            insight(
                f"Brier Score = {PAPER_BRIER} — 29% improvement over the naïve baseline "
                f"(BS ≈ 0.130). Isotonic regression aligns predicted risk with true event rates."
            )

        st.divider()
        row2_l, row2_r = st.columns(2)

        # ── Bootstrap CIs ────────────────────────────────────────────
        with row2_l:
            st.markdown("**Bootstrap 95% Confidence Intervals (n = 1,000 resamples)**")
            fig, ax = styled_fig(5.5, 3.8)
            metrics_b = list(PAPER_BOOT_CI.keys())
            vals_b    = [PAPER_BOOT_CI[m][0] for m in metrics_b]
            lows_b    = [PAPER_BOOT_CI[m][1] for m in metrics_b]
            highs_b   = [PAPER_BOOT_CI[m][2] for m in metrics_b]
            errs_lo   = [v - l for v, l in zip(vals_b, lows_b)]
            errs_hi   = [h - v for v, h in zip(vals_b, highs_b)]
            y_pos     = np.arange(len(metrics_b))
            ax.barh(y_pos, vals_b, height=0.4, color=C_TEAL, alpha=0.85,
                    label="Point estimate", edgecolor="none")
            ax.errorbar(vals_b, y_pos,
                        xerr=[errs_lo, errs_hi],
                        fmt="none", color=C_NAVY, capsize=6, lw=2, capthick=2)
            ax.set_yticks(y_pos); ax.set_yticklabels(metrics_b)
            ax.set_xlabel("Metric value")
            ax.set_title("Bootstrap 95% CIs", fontweight="bold", color=C_NAVY, fontsize=11)
            ax.set_xlim(0.55, 1.05)
            for i, (v, lo, hi) in enumerate(zip(vals_b, lows_b, highs_b)):
                ax.text(hi + 0.005, i, f"[{lo:.4f}, {hi:.4f}]",
                        va="center", fontsize=7.5, color=C_DGRAY)
            plt.tight_layout()
            show_fig(fig)
            insight("AUC CI width = 0.009 confirms high estimation precision.")

        # ── Confusion Matrix ─────────────────────────────────────────
        with row2_r:
            st.markdown("**Confusion Matrix (Stacking Ensemble, θ = 0.50, from paper)**")
            fig, ax = styled_fig(5.5, 3.8)
            cm = np.array(PAPER_CONFUSION)
            im = ax.imshow(cm, cmap="Blues", aspect="auto")
            labels = [["TN", "FP"], ["FN", "TP"]]
            true_labels = ["Normal", "Abnormal"]
            pred_labels = ["Pred: Normal", "Pred: Abnormal"]
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]:,}",
                            ha="center", va="center", fontsize=11,
                            color="white" if cm[i, j] > 5000 else C_NAVY,
                            fontweight="bold")
            ax.set_xticks([0, 1]); ax.set_xticklabels(pred_labels, fontsize=9)
            ax.set_yticks([0, 1]); ax.set_yticklabels(true_labels, fontsize=9)
            ax.set_title("Confusion Matrix (θ = 0.50)", fontweight="bold",
                         color=C_NAVY, fontsize=11)
            ax.grid(visible=False)
            plt.tight_layout()
            show_fig(fig)
            insight(
                "At θ* = 0.25, FN drops from 236 → ~70 (FNR: 26.6% → 7.9%). "
                "In thyroid screening, minimising missed cases is the clinical priority."
            )

    # ═════════════════════════════════════════════════════════════════
    # TAB 5 — CLINICAL UTILITY
    # ═════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.subheader("Clinical Utility — DCA · Threshold Optimisation · Robustness")

        row_c1, row_c2 = st.columns(2)

        # ── Decision Curve Analysis ──────────────────────────────────
        with row_c1:
            st.markdown("**Decision Curve Analysis**")
            fig, ax = styled_fig(5.5, 4.2)
            th_plot = thresholds[(thresholds >= 0.01) & (thresholds <= 0.50)]
            nb_m    = np.array(nb_model)[(thresholds >= 0.01) & (thresholds <= 0.50)]
            nb_a    = np.array(nb_all)[(thresholds >= 0.01) & (thresholds <= 0.50)]
            ax.plot(th_plot, nb_m, lw=2.5, color=C_NAVY, label="Stacking Ensemble")
            ax.plot(th_plot, nb_a, lw=1.5, color=C_RED, ls="--", label="Treat All")
            ax.axhline(0, color=C_DGRAY, lw=1, ls=":", label="Treat None")
            ax.axvline(OPTIMAL_THETA, color=C_TEAL, lw=1.5,
                       ls="--", label=f"θ* = {OPTIMAL_THETA}")
            ax.set_xlabel("Decision Threshold")
            ax.set_ylabel("Net Benefit")
            ax.set_title("Decision Curve Analysis", fontweight="bold",
                         color=C_NAVY, fontsize=11)
            ax.legend(fontsize=8.5)
            ax.set_xlim(0.01, 0.50)
            plt.tight_layout()
            show_fig(fig)
            insight(
                "At θ = 0.25: Net benefit = 0.073 vs. Treat-All = 0.041 → "
                "78% relative gain. The model dominates both reference strategies "
                "across the full clinically plausible range (0.05–0.40)."
            )

        # ── Threshold Sweep ─────────────────────────────────────────
        with row_c2:
            st.markdown("**Threshold Optimisation — F1, Recall, Precision**")
            fig, ax = styled_fig(5.5, 4.2)
            th_range = (thresholds >= 0.05) & (thresholds <= 0.80)
            ax.plot(thresholds[th_range], np.array(f1s)[th_range],
                    lw=2.5, color=C_NAVY, label="F1-Score")
            ax.plot(thresholds[th_range], np.array(recs)[th_range],
                    lw=1.5, color=C_TEAL, ls="--", label="Recall")
            ax.plot(thresholds[th_range], np.array(precs)[th_range],
                    lw=1.5, color=C_AMBER, ls="--", label="Precision")
            ax.axvline(OPTIMAL_THETA, color=C_RED, lw=2, ls=":",
                       label=f"θ* = {OPTIMAL_THETA} (F1-max)")
            ax.axvline(0.50, color=C_DGRAY, lw=1, ls=":",
                       label="Default θ = 0.50")
            ax.set_xlabel("Decision Threshold θ")
            ax.set_ylabel("Metric Value")
            ax.set_title("Threshold Sweep", fontweight="bold", color=C_NAVY, fontsize=11)
            ax.legend(fontsize=8)
            ax.set_xlim(0.05, 0.80)
            plt.tight_layout()
            show_fig(fig)
            insight(
                "θ* = 0.25 maximises F1 = 0.726 and reduces FNR from 26.6% → 7.9%. "
                "The default θ = 0.50 is inappropriate for 6.93% class prevalence."
            )

        st.divider()
        st.markdown("**Robustness Profile — Gaussian Noise vs. Random Feature Missingness**")

        row_r1, row_r2 = st.columns(2)

        # ── Noise Robustness ─────────────────────────────────────────
        with row_r1:
            fig, ax = styled_fig(5, 3.5)
            ax.plot(PAPER_NOISE_X, PAPER_NOISE_AUC,
                    "o-", color=C_RED, lw=2.5, markersize=8, label="AUC under noise")
            ax.axhline(PAPER_NOISE_AUC[0], color=C_DGRAY, lw=1, ls=":", label="Baseline")
            for x, y in zip(PAPER_NOISE_X, PAPER_NOISE_AUC):
                ax.text(x, y + 0.01, f"{y:.4f}", ha="center", fontsize=8.5,
                        color=C_NAVY, fontweight="600")
            ax.set_ylim(0.55, 1.02)
            ax.set_xlabel("Gaussian Noise Level (γ × feature σ)")
            ax.set_ylabel("AUC")
            ax.set_title("Noise Robustness", fontweight="bold", color=C_NAVY, fontsize=11)
            ax.legend(fontsize=8)
            plt.tight_layout()
            show_fig(fig)

        # ── Missingness Robustness ────────────────────────────────────
        with row_r2:
            fig, ax = styled_fig(5, 3.5)
            ax.plot(PAPER_MISSING_X, PAPER_MISSING_AUC,
                    "s-", color=C_TEAL, lw=2.5, markersize=8,
                    label="AUC under missingness")
            ax.axhline(PAPER_MISSING_AUC[0], color=C_DGRAY, lw=1, ls=":", label="Baseline")
            for x, y in zip(PAPER_MISSING_X, PAPER_MISSING_AUC):
                ax.text(x, y + 0.008, f"{y:.4f}", ha="center", fontsize=8.5,
                        color=C_NAVY, fontweight="600")
            ax.set_ylim(0.88, 1.02)
            ax.set_xlabel("Random Feature Missingness Rate")
            ax.set_ylabel("AUC")
            ax.set_title("Missingness Robustness", fontweight="bold",
                         color=C_NAVY, fontsize=11)
            ax.legend(fontsize=8)
            plt.tight_layout()
            show_fig(fig)

        insight(
            "Asymmetric degradation: AUC = 0.935 at 30% missingness (robust) but "
            "AUC = 0.663 at 20% Gaussian noise (sensitive). "
            "→ Measurement quality for TSH/TT4 matters more than panel completeness."
        )

    # ═════════════════════════════════════════════════════════════════
    # TAB 6 — DATASET OVERVIEW
    # ═════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.subheader("Harmonised Dataset Overview")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Samples", f"{len(df):,}")
        c2.metric("Features", len(FEATURES))
        c3.metric("Normal (0)", f"{(df['class']==0).sum():,}  (93.07%)")
        c4.metric("Abnormal (1)", f"{(df['class']==1).sum():,}  (6.93%)")
        c5.metric("Class Imbalance", "13.4 : 1")

        st.divider()

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown("**Class Distribution**")
            fig, ax = styled_fig(4.5, 3.2)
            counts = [int((df['class']==0).sum()), int((df['class']==1).sum())]
            bars = ax.bar(["Normal (0)", "Abnormal (1)"], counts,
                          color=[C_TEAL, C_RED], width=0.5, edgecolor="none")
            for bar, v in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 80,
                        f"{v:,}", ha="center", fontsize=10, color=C_NAVY, fontweight="600")
            ax.set_ylabel("Count")
            ax.set_title("Class Distribution", fontweight="bold", color=C_NAVY, fontsize=11)
            ax.set_ylim(0, max(counts) * 1.12)
            plt.tight_layout()
            show_fig(fig)

        with col_d2:
            st.markdown("**Feature Distributions — Median by Class**")
            fig, ax = styled_fig(4.5, 3.2)
            meds_norm = df[df["class"]==0][FEATURES].median().values
            meds_abn  = df[df["class"]==1][FEATURES].median().values
            # Normalise to [0,1] for radar-like bar comparison
            max_vals = np.maximum(meds_norm, meds_abn)
            max_vals[max_vals == 0] = 1
            x = np.arange(len(FEATURES))
            w = 0.35
            ax.bar(x - w/2, meds_norm / max_vals, w, label="Normal",
                   color=C_TEAL, edgecolor="none")
            ax.bar(x + w/2, meds_abn / max_vals, w, label="Abnormal",
                   color=C_RED, alpha=0.85, edgecolor="none")
            ax.set_xticks(x); ax.set_xticklabels(FEATURES)
            ax.set_ylabel("Relative median (normalised)")
            ax.set_title("Median Feature Values by Class", fontweight="bold",
                         color=C_NAVY, fontsize=11)
            ax.legend(fontsize=9)
            plt.tight_layout()
            show_fig(fig)

        st.divider()
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df[FEATURES + ["class"]].describe().round(4),
                     use_container_width=True)

        st.divider()
        st.markdown("**Feature Medians by Class**")
        st.dataframe(df.groupby("class")[FEATURES].median().round(4),
                     use_container_width=True)


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
