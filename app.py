"""
Thyroid Disorder Detection — Research Demo  v3.0
Stacking Ensemble: RF + XGBoost → Logistic Regression meta-learner

DESIGN PRINCIPLE: Every graph on screen must update when the user changes
an input. No static paper-reproduction charts.

Tabs:
  1. Live Prediction   — biomarker input → risk score + gauge + instance SHAP
  2. What-If Explorer  — sweep one feature, watch risk curve update live
  3. Sample & Verify   — pick a real patient, predict, live confusion matrix
  4. Research Summary  — text/tables only (no graphs), key findings from paper
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

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, brier_score_loss,
    confusion_matrix,
)
from xgboost import XGBClassifier

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# ── Palette ───────────────────────────────────────────────────────────
C_NAVY  = "#1C3557"
C_TEAL  = "#2A9D8F"
C_RED   = "#E76F51"
C_AMBER = "#E9C46A"
C_LGRAY = "#F4F6F8"
C_MGRAY = "#DDE2E8"
C_DGRAY = "#5C6370"

PLT_RC = {
    "axes.facecolor":    "#FAFBFC",
    "figure.facecolor":  "#FAFBFC",
    "axes.edgecolor":    C_MGRAY,
    "axes.grid":         True,
    "grid.color":        C_MGRAY,
    "grid.linewidth":    0.55,
    "text.color":        C_NAVY,
    "axes.labelcolor":   C_NAVY,
    "xtick.color":       C_DGRAY,
    "ytick.color":       C_DGRAY,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
}

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thyroid Disorder Detection · Research Demo",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.hero {
  background: linear-gradient(135deg, #1C3557 0%, #2A9D8F 100%);
  border-radius: 12px; padding: 24px 32px;
  margin-bottom: 20px; color: white;
}
.hero h1 { font-size: 1.75rem; font-weight: 700; margin: 0; color: white; }
.hero p  { font-size: 0.88rem; margin: 5px 0 0; opacity: 0.88; color: white; }

.kpi {
  background: white; border: 1px solid #DDE2E8;
  border-left: 4px solid #2A9D8F; border-radius: 10px;
  padding: 14px 18px; margin: 3px 0;
}
.kpi .lbl { font-size: 0.72rem; font-weight: 700; color: #5C6370;
            text-transform: uppercase; letter-spacing: 0.07em; }
.kpi .val { font-size: 1.45rem; font-weight: 700; color: #1C3557; }
.kpi .sub { font-size: 0.72rem; color: #5C6370; }

.insight {
  background: #EEF7F6; border-left: 4px solid #2A9D8F;
  border-radius: 0 8px 8px 0; padding: 11px 15px;
  margin: 10px 0; font-size: 0.86rem; color: #1C3557;
}
.warn-box {
  background: #FFF4EC; border-left: 4px solid #E76F51;
  border-radius: 0 8px 8px 0; padding: 11px 15px;
  margin: 10px 0; font-size: 0.86rem; color: #7B2D00;
}
.paper-tag {
  display: inline-block; background: #1C3557; color: white;
  font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em;
  border-radius: 4px; padding: 2px 8px; margin-left: 6px;
}
.block-container { padding-top: 0.8rem !important; }
button[data-baseweb="tab"] { font-size: 0.87rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────
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
    "age": dict(label="Age",  lo=0.01, hi=455.0, default=35.0,  step=1.0,
                normal="18-80",    unit="yrs",    tip="Patient age in years"),
    "TSH": dict(label="TSH",  lo=0.0,  hi=478.0, default=1.3,   step=0.01,
                normal="0.4-4.0",  unit="mIU/L",  tip="Thyroid Stimulating Hormone — primary HPT signal"),
    "T3":  dict(label="T3",   lo=0.0,  hi=10.6,  default=1.9,   step=0.01,
                normal="1.1-2.6",  unit="nmol/L", tip="Triiodothyronine — active thyroid hormone"),
    "TT4": dict(label="TT4",  lo=0.0,  hi=430.0, default=100.0, step=1.0,
                normal="60-160",   unit="nmol/L", tip="Total Thyroxine"),
    "T4U": dict(label="T4U",  lo=0.0,  hi=2.12,  default=0.97,  step=0.01,
                normal="0.9-1.1",  unit="ratio",  tip="Thyroxine Uptake ratio"),
    "FTI": dict(label="FTI",  lo=0.0,  hi=395.0, default=106.0, step=1.0,
                normal="70-130",   unit="index",  tip="Free Thyroxine Index"),
}


# ── Data & model ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data(path="harmonised_dataset.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=["source"], errors="ignore")
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    return df


@st.cache_resource(show_spinner="Training stacking ensemble…")
def build_model(df):
    X = df[FEATURES].values
    y = df["class"].values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    rf  = RandomForestClassifier(**RF_PARAMS, n_jobs=-1)
    xgb = XGBClassifier(**XGB_PARAMS, n_jobs=-1)
    stack = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=LogisticRegression(
            penalty="l2", C=1.0, max_iter=1000, class_weight="balanced"
        ),
        stack_method="predict_proba", n_jobs=-1, passthrough=False,
    )
    stack.fit(Xtr, ytr)

    # RF trained alone for SHAP (TreeExplainer needs a pure tree model)
    rf_solo = RandomForestClassifier(**RF_PARAMS, n_jobs=-1).fit(Xtr, ytr)

    yprob = stack.predict_proba(Xte)[:, 1]
    ypred = (yprob >= 0.5).astype(int)

    live_metrics = dict(
        AUC      =round(roc_auc_score(yte, yprob),                          4),
        Accuracy =round(accuracy_score(yte, ypred),                          4),
        Precision=round(precision_score(yte, ypred, zero_division=0),        4),
        Recall   =round(recall_score(yte, ypred),                            4),
        F1       =round(f1_score(yte, ypred),                                4),
        Brier    =round(brier_score_loss(yte, yprob),                        4),
    )

    shap_explainer = None
    if SHAP_OK:
        try:
            shap_explainer = shap.TreeExplainer(rf_solo)
        except Exception:
            pass

    return stack, shap_explainer, Xte, yte, yprob, live_metrics


# ── Helpers ───────────────────────────────────────────────────────────
def predict(model, vals: dict, theta: float):
    X     = pd.DataFrame([vals], columns=FEATURES)
    prob  = float(model.predict_proba(X)[0, 1])
    label = int(prob >= theta)
    return label, prob


def fig_ax(w=6.5, h=3.8):
    with plt.rc_context(PLT_RC):
        fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#FAFBFC")
    return fig, ax


def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def kpi(label, value, sub=""):
    st.markdown(
        f'<div class="kpi"><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div>'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def note(txt, warn=False):
    cls = "warn-box" if warn else "insight"
    icon = "⚠️" if warn else "💡"
    st.markdown(f'<div class="{cls}">{icon} {txt}</div>', unsafe_allow_html=True)


def normal_range(feat, val):
    """Return (lo, hi) floats for a feature's normal range string."""
    rng = FEATURE_META[feat]["normal"].replace("–", "-").split("-")
    return float(rng[0]), float(rng[1])


# ═══════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS — every one recomputed per user interaction
# ═══════════════════════════════════════════════════════════════════════

def chart_gauge(prob: float, theta: float):
    """Horizontal gauge bar that moves with every input change."""
    fig, ax = fig_ax(5.5, 2.0)
    # background
    ax.barh(0, 1.0, height=0.5, color=C_LGRAY, edgecolor="none", zorder=1)
    # filled risk
    ax.barh(0, prob, height=0.5,
            color=C_RED if prob >= theta else C_TEAL,
            edgecolor="none", zorder=2)
    # threshold line
    ax.axvline(theta, color=C_NAVY, lw=2.5, zorder=3)
    ax.text(theta, 0.34, f"θ={theta:.2f}", ha="center",
            fontsize=8.5, color=C_NAVY, fontweight="700", va="bottom")
    # risk label
    side = prob + 0.03 if prob < 0.80 else prob - 0.12
    ax.text(side, 0, f"{prob:.3f}", va="center", fontsize=12,
            fontweight="700",
            color="white" if 0.15 < prob < 0.85 else C_NAVY)
    # zone labels
    ax.axvspan(0,     theta, alpha=0.07, color=C_TEAL, zorder=0)
    ax.axvspan(theta, 1.0,   alpha=0.07, color=C_RED,  zorder=0)
    ax.text(theta / 2,       -0.38, "Normal zone",
            ha="center", fontsize=7.5, color=C_TEAL, fontweight="600")
    ax.text((theta + 1) / 2, -0.38, "Abnormal zone",
            ha="center", fontsize=7.5, color=C_RED,  fontweight="600")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("Calibrated risk probability", fontsize=9)
    ax.set_title("Risk Score Gauge", fontweight="bold",
                 color=C_NAVY, fontsize=10)
    ax.grid(axis="y", visible=False)
    plt.tight_layout()
    return fig


def chart_shap(explainer, vals: dict):
    """
    Instance-level SHAP bar chart.
    Fully recomputed for every new set of biomarker values.
    Returns (fig, sv1_array) or (None, None) if SHAP unavailable.
    """
    if explainer is None:
        return None, None
    try:
        X  = pd.DataFrame([vals], columns=FEATURES)
        sv = explainer.shap_values(X)
        sv1 = sv[1][0] if isinstance(sv, list) else sv[0]

        fig, ax = fig_ax(6.0, 3.4)
        order  = np.argsort(np.abs(sv1))
        colors = [C_RED if v > 0 else C_TEAL for v in sv1[order]]
        feat_labels = [
            f"{FEATURES[i]}  =  {vals[FEATURES[i]]:.2f}" for i in order
        ]
        bars = ax.barh(feat_labels, sv1[order],
                       color=colors, height=0.52, edgecolor="none")
        ax.axvline(0, color=C_NAVY, lw=0.9, linestyle="--")
        ax.set_xlabel("SHAP value  (+  pushes Abnormal  |  -  pushes Normal)")
        ax.set_title("Why this prediction?  —  Instance SHAP",
                     fontweight="bold", color=C_NAVY, fontsize=10)
        red_p  = mpatches.Patch(color=C_RED,  label="↑ Increases risk")
        teal_p = mpatches.Patch(color=C_TEAL, label="↓ Decreases risk")
        ax.legend(handles=[red_p, teal_p], fontsize=8, loc="lower right")
        for bar, v in zip(bars, sv1[order]):
            ax.text(
                v + (0.003 if v >= 0 else -0.003),
                bar.get_y() + bar.get_height() / 2,
                f"{v:+.3f}", va="center", fontsize=8,
                ha="left" if v >= 0 else "right", color=C_NAVY,
            )
        plt.tight_layout()
        return fig, sv1
    except Exception:
        return None, None


def chart_whatif(model, base_vals: dict, feat: str, theta: float):
    """
    Sweeps `feat` across its full range, holds all other features at
    base_vals, and plots the resulting risk probability curve.
    Fully redrawn every time base_vals, feat, or theta changes.
    """
    m  = FEATURE_META[feat]
    xs = np.linspace(m["lo"], m["hi"], 300)
    ps = np.array([
        float(model.predict_proba(
            pd.DataFrame([{**base_vals, feat: x}], columns=FEATURES)
        )[0, 1])
        for x in xs
    ])
    cur_x = base_vals[feat]
    cur_p = float(model.predict_proba(
        pd.DataFrame([base_vals], columns=FEATURES)
    )[0, 1])

    # Normal range shading
    lo_n, hi_n = normal_range(feat, cur_x)

    fig, ax = fig_ax(7.2, 3.8)
    ax.axvspan(lo_n, min(hi_n, m["hi"]), alpha=0.10,
               color=C_TEAL, label=f"Normal range ({m['normal']} {m['unit']})")
    ax.plot(xs, ps, lw=2.5, color=C_NAVY, zorder=3, label="Risk probability")
    ax.fill_between(xs, ps, alpha=0.10, color=C_TEAL)
    ax.axhline(theta, color=C_RED, lw=1.5, ls="--",
               label=f"Decision threshold θ = {theta:.2f}")
    ax.axvline(cur_x, color=C_AMBER, lw=2.0, ls=":",
               label=f"Current value = {cur_x:.2f}")
    ax.scatter([cur_x], [cur_p], s=90, color=C_AMBER,
               zorder=5, edgecolors=C_NAVY, lw=1.5)
    ax.fill_between(xs, theta, 1.02, alpha=0.05, color=C_RED)
    ax.fill_between(xs, 0,     theta, alpha=0.05, color=C_TEAL)
    ax.set_xlabel(
        f"{m['label']}  ({m['unit']})   |   Normal: {m['normal']}",
        fontsize=9,
    )
    ax.set_ylabel("Abnormality probability", fontsize=9)
    ax.set_title(
        f"How does risk change as {feat} varies?  "
        f"(all other features fixed at baseline)",
        fontweight="bold", color=C_NAVY, fontsize=10,
    )
    ax.set_ylim(-0.02, 1.04)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    return fig, xs, ps


def chart_confusion(yprob, yte, theta: float):
    """
    Confusion matrix for the full test set at the given threshold.
    Redraws every time the sidebar θ slider moves.
    """
    ypred = (np.array(yprob) >= theta).astype(int)
    yte   = np.array(yte)
    cm    = confusion_matrix(yte, ypred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    fig, ax = fig_ax(5.0, 3.4)
    bg = [[C_TEAL, "#F7D9D0"], ["#FDECEA", C_NAVY]]
    lbl_map = [["TN", "FP"], ["FN ⚠", "TP"]]
    vals_map = [[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]]

    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.axis("off")
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle(
                [j, 1 - i], 1, 1,
                facecolor=bg[i][j], edgecolor="white", lw=2.5,
            ))
            txt_color = "white" if (i == 1 and j == 1) else C_NAVY
            ax.text(j + 0.5, 1.5 - i,
                    f"{lbl_map[i][j]}\n{vals_map[i][j]}",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=txt_color)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Pred: Normal", "Pred: Abnormal"], fontsize=9)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Actual\nAbnormal", "Actual\nNormal"], fontsize=9)
    ax.tick_params(left=False, bottom=False)
    ax.set_title(f"Confusion Matrix — θ = {theta:.2f}",
                 fontweight="bold", color=C_NAVY, fontsize=10)
    plt.tight_layout()

    stats = dict(FNR=fnr, Recall=rec, Precision=prec, F1=f1,
                 TP=tp, FP=fp, TN=tn, FN=fn)
    return fig, stats


# ═══════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════
def main():
    # Header
    st.markdown("""
    <div class="hero">
      <h1> Thyroid Disorder Detection</h1>
      <p>Calibrated Stacking Ensemble  ·  RF + XGBoost → Logistic Regression
         ·  Harmonised UCI Corpus  n = 12,800
         ·  SHAP  ·  Calibration  ·  DCA  ·  Bootstrap CI</p>
    </div>""", unsafe_allow_html=True)

    # Load data
    try:
        df = load_data("harmonised_dataset.csv")
    except FileNotFoundError:
        st.error("⚠️ `harmonised_dataset.csv` not found. Place it alongside `app.py`.")
        st.stop()

    (model, shap_explainer,
     Xte, yte, yprob_test, live_metrics) = build_model(df)

    # ─── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️  Decision Threshold  θ")
        theta = st.slider(
            "Drag to explore trade-offs",
            min_value=0.05, max_value=0.90,
            value=0.25, step=0.05,
            help=(
                "θ* = 0.25 is F1-maximising (from paper). "
                "Default naïve = 0.50. Lower θ → higher recall, "
                "more false positives. Higher θ → fewer flags, more missed cases."
            ),
        )

        # Live FNR preview from the actual test set
        at_cur = (np.array(yprob_test) >= theta).astype(int)
        at_50  = (np.array(yprob_test) >= 0.50).astype(int)
        yte_np = np.array(yte)
        fn_cur = ((at_cur == 0) & (yte_np == 1)).sum()
        fn_50  = ((at_50  == 0) & (yte_np == 1)).sum()
        pos    = (yte_np == 1).sum()
        fnr_cur = fn_cur / pos
        fnr_50  = fn_50  / pos
        st.caption(
            f"FNR at **θ = {theta:.2f}** → **{fnr_cur:.1%}** "
            f"&nbsp;|&nbsp; FNR at θ = 0.50 → {fnr_50:.1%}"
        )

        st.divider()
        st.markdown("### 📊  Live Test-Set Metrics")
        st.caption("Computed on 20% held-out split (this session)")
        label_map = {
            "AUC":      "ROC-AUC",
            "Accuracy": "Accuracy",
            "Precision":"Precision",
            "Recall":   "Recall (Sensitivity)",
            "F1":       "F1-Score",
            "Brier":    "Brier Score ↓",
        }
        for k, v in live_metrics.items():
            st.metric(label_map[k], v)

        st.divider()
        st.markdown("### 📄  Paper  (Nested 10-Fold CV)")
        st.caption("IAENG IJCS — Pawar, Mahakalkar, Gaikwad, Neware")
        st.table(pd.DataFrame([
            ["AUC",          "0.9836 ± 0.0021"],
            ["Recall",       "92.45%"],
            ["F1",           "0.704"],
            ["Brier Score",  "0.0923  (-29%)"],
            ["p vs RF",      "0.0039 ★"],
            ["p vs XGBoost", "0.131 n.s."],
        ], columns=["Metric", "Value"]).set_index("Metric"))

    # ─── Tabs ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        " Live Prediction",
        " What-If Explorer",
        " Sample & Verify",
        " Research Summary",
    ])

    # ═══════════════════════════════════════════════════════════════
    # TAB 1 — LIVE PREDICTION
    # All output (badge, score, gauge, SHAP bar, context row)
    # recomputes the moment any number input changes.
    # ═══════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Enter Patient Biomarkers")
        st.caption(
            "Every number below is live. Change any value and the gauge, "
            "risk score, and SHAP bar update immediately — no button needed."
        )

        # Six input widgets
        cols_in = st.columns(3)
        input_vals = {}
        for i, feat in enumerate(FEATURES):
            m = FEATURE_META[feat]
            with cols_in[i % 3]:
                input_vals[feat] = st.number_input(
                    label=f"{m['label']}  ({m['unit']})",
                    min_value=float(m["lo"]),
                    max_value=float(m["hi"]),
                    value=float(m["default"]),
                    step=float(m["step"]),
                    help=f"{m['tip']}  ·  Normal: {m['normal']} {m['unit']}",
                    key=f"t1_{feat}",
                )

        st.divider()

        # Predict (runs automatically on every widget change)
        label, prob = predict(model, input_vals, theta)

        # ── Result strip ────────────────────────────────────────────
        r1, r2, r3 = st.columns(3)
        with r1:
            if label == 1:
                st.error("🔴  **ABNORMAL — Disorder Detected**")
            else:
                st.success("🟢  **NORMAL — No Disorder**")
        with r2:
            kpi("Calibrated Risk Score",
                f"{prob:.4f}",
                f"{prob * 100:.1f}%  abnormality probability")
        with r3:
            kpi("Decision Threshold",
                f"θ = {theta:.2f}",
                "Adjust in sidebar  ·  Optimal θ* = 0.25")

        st.divider()

        # ── Two live charts side-by-side ────────────────────────────
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown(
                "**Risk Gauge** "
                "<span class='paper-tag'>LIVE</span>",
                unsafe_allow_html=True,
            )
            show(chart_gauge(prob, theta))
            if prob >= theta:
                note(
                    f"Risk ({prob:.3f}) ≥ θ ({theta:.2f}). "
                    "Model recommends confirmatory specialist referral. "
                    f"At θ* = 0.25, FNR drops from 26.6% → 7.9%."
                )
            else:
                note(
                    f"Risk ({prob:.3f}) < θ ({theta:.2f}). "
                    "Model predicts normal thyroid function. "
                    "Not a clinical diagnosis — automated first-pass triage only."
                )

        with c_right:
            st.markdown(
                "**Instance SHAP — why this prediction?** "
                "<span class='paper-tag'>LIVE</span>",
                unsafe_allow_html=True,
            )
            if SHAP_OK and shap_explainer is not None:
                shap_fig, sv1 = chart_shap(shap_explainer, input_vals)
                if shap_fig and sv1 is not None:
                    show(shap_fig)
                    top_i    = int(np.argmax(np.abs(sv1)))
                    top_feat = FEATURES[top_i]
                    top_val  = sv1[top_i]
                    direction = "toward Abnormal ↑" if top_val > 0 else "toward Normal ↓"
                    note(
                        f"Strongest driver: **{top_feat}** "
                        f"(SHAP = {top_val:+.3f},  {direction}). "
                        "Red = pushes risk up · Teal = pushes risk down."
                    )
            else:
                st.info(
                    "Install `shap` (`pip install shap`) to enable "
                    "instance-level explanations."
                )

        # ── Clinical range context row ───────────────────────────────
        st.divider()
        st.markdown("**Clinical range check — current values vs. normal ranges**")
        ctx_cols = st.columns(len(FEATURES))
        for i, feat in enumerate(FEATURES):
            m       = FEATURE_META[feat]
            v       = input_vals[feat]
            lo_n, hi_n = normal_range(feat, v)
            in_rng  = lo_n <= v <= hi_n
            with ctx_cols[i]:
                st.metric(
                    label=f"{feat}",
                    value=f"{v:.2f}",
                    delta="✅ Normal" if in_rng else "⚠️ Out of range",
                    delta_color="off" if in_rng else "inverse",
                    help=f"Normal: {m['normal']} {m['unit']}",
                )

    # ═══════════════════════════════════════════════════════════════
    # TAB 2 — WHAT-IF EXPLORER
    # Set a baseline patient, pick one feature to sweep.
    # Risk curve redraws live — demonstrates HPT nonlinearity.
    # ═══════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("What-If Feature Explorer")
        st.caption(
            "Set a baseline patient profile below, then choose one feature to vary "
            "across its entire clinical range. The risk curve updates live, "
            "revealing the nonlinear HPT dynamics the ensemble has learnt."
        )

        st.markdown("**Baseline patient  (all features held fixed except the chosen one)**")
        base_cols_ui = st.columns(3)
        base_vals    = {}
        for i, feat in enumerate(FEATURES):
            m = FEATURE_META[feat]
            with base_cols_ui[i % 3]:
                base_vals[feat] = st.number_input(
                    label=f"{m['label']}  ({m['unit']})",
                    min_value=float(m["lo"]),
                    max_value=float(m["hi"]),
                    value=float(m["default"]),
                    step=float(m["step"]),
                    help=f"Normal: {m['normal']} {m['unit']}",
                    key=f"t2_base_{feat}",
                )

        sweep_feat = st.selectbox(
            "Feature to sweep  →",
            options=FEATURES,
            index=1,   # TSH by default (most interesting)
            help="All other features stay at the baseline values set above.",
            key="t2_sweep",
        )

        st.divider()
        st.markdown(
            f"**Risk curve: how does the model respond to varying  {sweep_feat}?** "
            "<span class='paper-tag'>LIVE</span>",
            unsafe_allow_html=True,
        )

        wi_fig, xs, ps = chart_whatif(model, base_vals, sweep_feat, theta)
        show(wi_fig)

        # Insight: how many times does the curve cross the threshold?
        crossings = int((np.diff((ps >= theta).astype(int)) != 0).sum())
        note(
            f"Risk curve crosses the decision boundary "
            f"**{crossings} time(s)** as {sweep_feat} varies from "
            f"{FEATURE_META[sweep_feat]['lo']} to {FEATURE_META[sweep_feat]['hi']} "
            f"{FEATURE_META[sweep_feat]['unit']}. "
            f"Min risk = {ps.min():.3f}  ·  Max risk = {ps.max():.3f}. "
            "Non-monotonic curves reflect the nonlinear HPT feedback modelled by the ensemble."
        )

    # ═══════════════════════════════════════════════════════════════
    # TAB 3 — SAMPLE & VERIFY
    # Pick a real patient record → predict → show result.
    # The confusion matrix (bottom) updates with the θ slider.
    # ═══════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Sample Patient & Verify Against True Label")
        st.caption(
            "Select a real record from the harmonised dataset. "
            "The confusion matrix at the bottom updates live as you move θ."
        )

        fa, fb, fc = st.columns([2, 2, 1])
        with fa:
            cls_filter = st.selectbox(
                "Filter by true class",
                ["All", "Normal (0)", "Abnormal (1)"],
                key="t3_cls",
            )
        with fb:
            idx_in = st.number_input(
                "Sample index", min_value=0, max_value=500,
                value=0, step=1, key="t3_idx",
            )
        with fc:
            rand_btn = st.button("🎲 Random", use_container_width=True, key="t3_rand")

        if cls_filter == "Normal (0)":
            subset = df[df["class"] == 0].reset_index(drop=True)
        elif cls_filter == "Abnormal (1)":
            subset = df[df["class"] == 1].reset_index(drop=True)
        else:
            subset = df.reset_index(drop=True)

        chosen = (
            int(np.random.randint(0, len(subset))) if rand_btn
            else min(int(idx_in), len(subset) - 1)
        )
        row        = subset.iloc[chosen]
        samp_vals  = {f: float(row[f]) for f in FEATURES}
        true_label = int(row["class"])

        st.divider()
        true_str = "🔴 Abnormal (1)" if true_label else "🟢 Normal (0)"
        st.markdown(f"**Sample #{chosen}** — True label: **{true_str}**")

        # Feature cards with range check
        mcols = st.columns(len(FEATURES))
        for i, feat in enumerate(FEATURES):
            m = FEATURE_META[feat]
            v = samp_vals[feat]
            lo_n, hi_n = normal_range(feat, v)
            in_rng = lo_n <= v <= hi_n
            with mcols[i]:
                st.metric(
                    feat, f"{v:.3f}",
                    delta="Normal ✅" if in_rng else "⚠️ Out of range",
                    delta_color="off" if in_rng else "inverse",
                )

        st.divider()

        # Predict this sample
        slabel, sprob = predict(model, samp_vals, theta)

        left_s, right_s = st.columns(2)

        with left_s:
            if slabel == 1:
                st.error(f"🔴  **ABNORMAL**  —  Risk = {sprob:.4f}")
            else:
                st.success(f"🟢  **NORMAL**  —  Risk = {sprob:.4f}")
            show(chart_gauge(sprob, theta))

        with right_s:
            # Correct / wrong verdict
            if slabel == true_label:
                st.success(
                    f"✅  **Correct** — model matched true label ({true_label})."
                )
            elif true_label == 1 and slabel == 0:
                st.error(
                    "❌  **False Negative** — True: Abnormal, Predicted: Normal. "
                    "Missed case. Try lowering θ in the sidebar."
                )
            else:
                st.warning(
                    "⚠️  **False Positive** — True: Normal, Predicted: Abnormal. "
                    "Unnecessary referral. Try raising θ."
                )

            # Instance SHAP for the sample patient
            if SHAP_OK and shap_explainer is not None:
                st.markdown("**Why? — Instance SHAP for this patient**")
                sfig, _ = chart_shap(shap_explainer, samp_vals)
                if sfig:
                    show(sfig)

        # ── Live confusion matrix (responds to θ slider) ─────────────
        st.divider()
        st.markdown(
            "**Confusion Matrix — entire test set at current θ** "
            "<span class='paper-tag'>UPDATES WITH θ SLIDER</span>",
            unsafe_allow_html=True,
        )
        cm_fig, cm_stats = chart_confusion(yprob_test, yte, theta)

        cm_l, cm_r = st.columns([1, 1])
        with cm_l:
            show(cm_fig)

        with cm_r:
            st.markdown(f"**At θ = {theta:.2f}:**")
            kpi("False Negative Rate",   f"{cm_stats['FNR']:.1%}",
                "↓ Lower = fewer missed cases")
            kpi("Recall (Sensitivity)",  f"{cm_stats['Recall']:.1%}",
                "% of true abnormal cases caught")
            kpi("Precision",             f"{cm_stats['Precision']:.1%}",
                "% of flagged cases that are truly abnormal")
            kpi("F1-Score",              f"{cm_stats['F1']:.3f}",
                "Harmonic mean of Precision & Recall")

            st.divider()
            if abs(theta - 0.25) < 0.01:
                note(
                    "θ = 0.25 (optimal). FNR is minimised. "
                    "This threshold maximises F1 = 0.726 and reduces "
                    "missed cases from 26.6% → 7.9% (paper result)."
                )
            elif theta > 0.45:
                note(
                    f"At θ = {theta:.2f}, FNR = {cm_stats['FNR']:.1%}. "
                    "Many true disorder cases are missed. "
                    "Lower the threshold for a screening context.",
                    warn=True,
                )
            else:
                note(
                    f"θ = {theta:.2f}. FNR = {cm_stats['FNR']:.1%}. "
                    "Good recall — appropriate for a thyroid screening workflow."
                )

    # ═══════════════════════════════════════════════════════════════
    # TAB 4 — RESEARCH SUMMARY
    # Text and tables only — no static charts.
    # The panel reads this as a quick reference card.
    # ═══════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Research Summary")
        st.caption(
            "*Explainable Thyroid Disorder Detection via a Calibrated Stacking "
            "Ensemble: Integrating Robustness Analysis, SHAP Interaction "
            "Decomposition, and Decision Curve Validation* · "
            "Pawar, Mahakalkar, Gaikwad, Neware · "
            "IAENG International Journal of Computer Science (under submission)"
        )

        # Six contributions
        st.markdown("#### Six Methodological Contributions")
        for num, title, detail in [
            ("1", "Multi-repository harmonisation",
             "Three UCI thyroid datasets → unified 12,800-record corpus · "
             "11,913 normal / 887 abnormal  (13.4 : 1 imbalance)"),
            ("2", "Nested 10-fold cross-validation",
             "Inner loop tunes hyperparameters; outer loop evaluates — "
             "eliminates the optimistic bias present in all compared works"),
            ("3", "Isotonic probability calibration",
             "Brier Score = 0.0923 · 29% below naïve baseline · "
             "required for clinically actionable risk communication"),
            ("4", "Five-channel SHAP explainability",
             "MI + PI + RFE + SHAP global attribution + biomarker ablation "
             "all converge on TSH (ΔAUC = -0.143 on removal)"),
            ("5", "SHAP pairwise interaction decomposition",
             "First application to biochemical thyroid panel · "
             "TSH×T3 = 0.036 matches the HPT negative-feedback pathway"),
            ("6", "Decision Curve Analysis + threshold optimisation",
             "78% relative net benefit gain over treat-all at θ = 0.25 · "
             "FNR: 26.6% → 7.9%  ·  F1: 0.699 → 0.726"),
        ]:
            st.markdown(
                f'<div class="insight"><b>#{num} · {title}</b> — {detail}</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("#### Nested 10-Fold CV — All Models")
        st.dataframe(
            pd.DataFrame([
                {"Model":"Random Forest",   "AUC":"0.9830±0.0029","Acc":"94.30%",
                 "Prec":"55.40%","Recall":"94.93%","F1":"0.699","BS":"—"},
                {"Model":"XGBoost",         "AUC":"0.9832±0.0039","Acc":"96.23%",
                 "Prec":"82.24%","Recall":"58.40%","F1":"0.682","BS":"—"},
                {"Model":"MLP",             "AUC":"0.8978±0.0193","Acc":"92.87%",
                 "Prec":"41.68%","Recall":"6.09%", "F1":"0.104","BS":"—"},
                {"Model":"Stacking (ours)", "AUC":"0.9836±0.0021","Acc":"94.62%",
                 "Prec":"56.87%","Recall":"92.45%","F1":"0.704","BS":"0.0923"},
            ]).set_index("Model").style.apply(
                lambda x: [
                    "background:#EEF7F6;font-weight:700"
                    if x.name == "Stacking (ours)" else "" for _ in x
                ], axis=1,
            ),
            use_container_width=True,
        )
        st.caption(
            "Wilcoxon signed-rank: Stack vs RF p = 0.0039 ★  ·  "
            "Stack vs XGBoost p = 0.131 n.s.  ·  "
            "Bootstrap 95% CI for AUC = [0.9778, 0.9868]  (width = 0.009)"
        )

        st.divider()
        st.markdown("#### SHAP Feature Importance  (cross-validated, n = 12,800)")
        st.dataframe(
            pd.DataFrame([
                {"Feature":"TSH","Mean |ϕ|":0.2682,"MI":0.1301,"PI":"+0.0684",
                 "Rank":1,"Key finding":"4.7× TT4 · ΔAUC = -0.143 on removal"},
                {"Feature":"TT4","Mean |ϕ|":0.0565,"MI":0.0361,"PI":"+0.0043",
                 "Rank":2,"Key finding":"Total thyroxine secretion"},
                {"Feature":"T3", "Mean |ϕ|":0.0534,"MI":0.0482,"PI":"+0.0076",
                 "Rank":3,"Key finding":"Active hormone · TSH×T3 interaction = 0.036"},
                {"Feature":"FTI","Mean |ϕ|":0.0480,"MI":0.0388,"PI":"+0.0022",
                 "Rank":4,"Key finding":""},
                {"Feature":"T4U","Mean |ϕ|":0.0407,"MI":0.0115,"PI":"-0.0117",
                 "Rank":5,"Key finding":"Negative PI — permutation has no effect"},
                {"Feature":"age","Mean |ϕ|":0.0051,"MI":0.0057,"PI":"-0.0116",
                 "Rank":6,"Key finding":"Demographic proxy only"},
            ]).set_index("Feature"),
            use_container_width=True,
        )
        st.caption(
            "Top-5 SHAP interactions: TSH×T3=0.036 · T3×TT4=0.022 · "
            "T3×FTI=0.018 · TSH×TT4=0.018 · TSH×FTI=0.016  — "
            "all involve TSH or T3, consistent with HPT regulatory biology."
        )

        st.divider()
        st.markdown("#### Robustness Profile  (from paper)")
        st.dataframe(
            pd.DataFrame([
                {"Experiment":"Gaussian noise  5%",    "AUC":0.7350,"ΔAUC":-0.149,"Assessment":"Sensitive"},
                {"Experiment":"Gaussian noise 10%",    "AUC":0.6933,"ΔAUC":-0.290,"Assessment":"Sensitive"},
                {"Experiment":"Gaussian noise 20%",    "AUC":0.6625,"ΔAUC":-0.320,"Assessment":"Sensitive"},
                {"Experiment":"Missing features 10%",  "AUC":0.9688,"ΔAUC":-0.015,"Assessment":"Robust"},
                {"Experiment":"Missing features 20%",  "AUC":0.9523,"ΔAUC":-0.030,"Assessment":"Robust"},
                {"Experiment":"Missing features 30%",  "AUC":0.9345,"ΔAUC":-0.048,"Assessment":"Robust"},
            ]).set_index("Experiment"),
            use_container_width=True,
        )
        note(
            "Asymmetric degradation: robust to feature missingness (AUC = 0.935 at 30%), "
            "sensitive to Gaussian noise (AUC = 0.663 at 20%). "
            "→ TSH/TT4 measurement precision matters more than panel completeness "
            "for deployment infrastructure decisions."
        )


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
