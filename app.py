"""
PCOS Classifier Lab - Streamlit dashboard
=========================================
Evaluates the five pre-trained PCOS classifiers stored in ``model/`` on a test CSV.

* Sidebar  : data source (upload a CSV or use the bundled ``test_data.csv``) + page navigation
* Pages    : 1) Single-model evaluation   2) All-model leaderboard
             3) Patient simulator (live what-if scoring)   4) Project notes

Models are only *loaded* (joblib) and used for ``predict`` / ``predict_proba``;
nothing is re-fitted on the uploaded data.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from pcos_prep import INPUT_COLUMNS, LABEL, harmonise_headers, repair_typos

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
BUNDLED_CSV = HERE / "test_data.csv"
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

# one fixed colour per model (validated categorical palette, dark surface)
MODEL_HUES = {
    "Logistic Regression": "#3987e5",
    "Decision Tree": "#d95926",
    "KNN": "#199e70",
    "Naive Bayes": "#c98500",
    "Support Vector Machine": "#d55181",
}
INK, INK_SOFT, INK_MUTED = "#e8eef2", "#b4c0c8", "#7f8b93"
GRID, AXIS, PANEL = "#26323a", "#36444d", "#141c22"

st.set_page_config(page_title="PCOS Classifier Lab", page_icon="🧬", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1280px;}
  div[data-testid="stMetric"] {background: #141c22;}
  div[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
  .lab-banner {border-left: 4px solid #2bb3a3; padding: .55rem 1rem; background: #121a20;
               border-radius: 0 8px 8px 0; color: #b4c0c8; margin-bottom: 1rem; font-size: .95rem;}
  .lab-banner b {color: #e8eef2;}
  .lab-tag {display:inline-block; font-size:.75rem; letter-spacing:.06em; text-transform:uppercase;
            color:#2bb3a3; font-weight:600; margin-bottom:.15rem;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  Loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading saved pipelines ...")
def fetch_artifacts() -> tuple[dict, dict]:
    card = json.loads((MODEL_DIR / "scoreboard.json").read_text())
    pipes = {name: joblib.load(MODEL_DIR / info["File"]) for name, info in card["models"].items()}
    return pipes, card


def prepare_upload(raw: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Normalise headers, coerce numbers and check the schema.
    Returns (clean frame or None, blocking problems, informational notes)."""
    problems, notes = [], []
    frame = repair_typos(harmonise_headers(raw))
    absent = [c for c in INPUT_COLUMNS if c not in frame.columns]
    if absent:
        problems.append(f"{len(absent)} required feature column(s) missing: "
                        + ", ".join(absent[:10]) + (" ..." if len(absent) > 10 else ""))
    if LABEL not in frame.columns:
        problems.append(f"Target column `{LABEL}` (or the raw `PCOS (Y/N)`) not found.")
    if problems:
        return None, problems, notes

    before = len(frame)
    frame = frame.dropna(subset=[LABEL])
    if len(frame) < before:
        notes.append(f"Dropped {before - len(frame)} row(s) with a blank target.")
    if not frame[LABEL].isin([0, 1]).all():
        return None, [f"`{LABEL}` must contain only 0 (no PCOS) and 1 (PCOS)."], notes
    if frame.empty:
        return None, ["No usable rows left after cleaning."], notes

    gaps = int(frame[INPUT_COLUMNS].isna().sum().sum())
    if gaps:
        notes.append(f"{gaps} blank/non-numeric feature cell(s) will be imputed by the saved "
                     "pipelines (training-set medians / modes).")
    surplus = [c for c in frame.columns if c not in INPUT_COLUMNS + [LABEL]]
    if surplus:
        notes.append(f"Ignoring {len(surplus)} extra column(s): {', '.join(map(str, surplus[:6]))}")
    frame[LABEL] = frame[LABEL].astype(int)
    return frame[INPUT_COLUMNS + [LABEL]].reset_index(drop=True), problems, notes


@st.cache_data(show_spinner="Scoring models on the test data ...")
def run_all_models(frame: pd.DataFrame) -> dict:
    """Predict with every saved pipeline (no fitting) and compute the six metrics."""
    pipes, _ = fetch_artifacts()
    X, y = frame[INPUT_COLUMNS], frame[LABEL]
    both_classes = y.nunique() == 2
    out = {}
    for name, pipe in pipes.items():
        guess = pipe.predict(X)
        risk = pipe.predict_proba(X)[:, 1]
        out[name] = {
            "pred": guess, "proba": risk,
            "metrics": {
                "Accuracy": accuracy_score(y, guess),
                "AUC": roc_auc_score(y, risk) if both_classes else float("nan"),
                "Precision": precision_score(y, guess, zero_division=0),
                "Recall": recall_score(y, guess, zero_division=0),
                "F1": f1_score(y, guess, zero_division=0),
                "MCC": matthews_corrcoef(y, guess),
            },
        }
    return out


try:
    PIPES, CARD = fetch_artifacts()
except Exception as err:  # pragma: no cover - surfaced in the UI
    st.error(f"Could not load the saved models from `model/`: {err}")
    st.stop()


# --------------------------------------------------------------------------- #
#  Plot helpers (Plotly, dark)
# --------------------------------------------------------------------------- #
def _style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL,
        font=dict(color=INK_SOFT, size=13),
        hoverlabel=dict(bgcolor="#0e1418", bordercolor=AXIS, font=dict(color=INK)),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(color=INK_SOFT)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=INK_MUTED))
    return fig


def confusion_heatmap(y_true, y_pred) -> go.Figure:
    grid = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # colour = share of the actual class (row-normalised) so the minority class is not drowned out
    row_share = grid / np.clip(grid.sum(axis=1, keepdims=True), 1, None)
    names = ["No PCOS (0)", "PCOS (1)"]
    tags = [["True negative", "False positive"], ["False negative", "True positive"]]
    fig = go.Figure(go.Heatmap(
        z=row_share, x=names, y=names, showscale=False, xgap=3, ygap=3, zmin=0, zmax=1,
        colorscale=[[0, "#16222c"], [0.5, "#1c5cab"], [1, "#6da7ec"]],
        customdata=np.dstack([grid, np.array(tags, dtype=object)]),
        hovertemplate=("<b>%{customdata[0]}</b> patients (%{z:.0%} of actual %{y})"
                       "<br>%{customdata[1]} · predicted %{x}<extra></extra>"),
    ))
    for i in range(2):
        for j in range(2):
            fig.add_annotation(x=names[j], y=names[i], showarrow=False,
                               text=(f"<b>{grid[i, j]}</b><br><span style='font-size:11px'>"
                                     f"{tags[i][j]} · {row_share[i, j]:.0%}</span>"),
                               font=dict(size=20, color="#0b0b0b" if row_share[i, j] > 0.7 else INK))
    fig.update_yaxes(autorange="reversed", title="Actual", showgrid=False)
    fig.update_xaxes(title="Predicted", side="bottom", showgrid=False)
    return _style(fig, 360)


def roc_figure(y_true, curves: dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                             line=dict(color=INK_MUTED, width=1, dash="dot"), hoverinfo="skip"))
    for name, risk in curves.items():
        fpr, tpr, _ = roc_curve(y_true, risk)
        auc = roc_auc_score(y_true, risk)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.3f})",
            line=dict(color=MODEL_HUES.get(name, "#3987e5"), width=2, shape="hv"),
            hovertemplate=f"{name}<br>FPR %{{x:.2f}} · TPR <b>%{{y:.2f}}</b><extra></extra>"))
    fig.update_xaxes(title="False positive rate", range=[-0.01, 1.01])
    fig.update_yaxes(title="True positive rate", range=[-0.01, 1.02])
    return _style(fig, 380 if len(curves) == 1 else 440)


def metric_bars(board: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name in board.index:
        fig.add_trace(go.Bar(
            name=name, x=METRIC_ORDER, y=board.loc[name, METRIC_ORDER],
            marker=dict(color=MODEL_HUES.get(name), line=dict(color=PANEL, width=2),
                        cornerradius=4),
            hovertemplate=f"<b>%{{y:.4f}}</b><br>{name} · %{{x}}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    fig.update_yaxes(range=[0, 1.0], title="Score (0-1)", dtick=0.2)
    return _style(fig, 420)


# --------------------------------------------------------------------------- #
#  Sidebar - data source
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🧬 PCOS Classifier Lab")
    st.caption("Five saved scikit-learn pipelines · evaluated live · never retrained")
    st.divider()
    st.markdown('<span class="lab-tag">Step 1 · Test data</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a test CSV (features + `PCOS` target)", type=["csv"])
    if st.button("Use bundled test_data.csv", icon="📦", width="stretch"):
        st.session_state["source"] = "bundled"
    if uploaded is not None:
        st.session_state["source"] = "upload"
    elif st.session_state.get("source") == "upload":
        st.session_state["source"] = "bundled"
    st.session_state.setdefault("source", "bundled")

    with open(BUNDLED_CSV, "rb") as fh:
        st.download_button("Download sample test_data.csv", fh.read(), "test_data.csv",
                           mime="text/csv", icon="⬇️", width="stretch")
    st.divider()

raw_frame, source_label = None, ""
if st.session_state["source"] == "upload" and uploaded is not None:
    try:
        raw_frame, source_label = pd.read_csv(uploaded), f"Uploaded · {uploaded.name}"
    except Exception as err:
        st.sidebar.error(f"Could not parse the CSV: {err}")
else:
    raw_frame, source_label = pd.read_csv(BUNDLED_CSV), "Bundled · test_data.csv"

DATA, ISSUES, NOTES = (None, ["No data loaded."], []) if raw_frame is None else prepare_upload(raw_frame)

with st.sidebar:
    if DATA is not None:
        st.success(f"{source_label}\n\n**{len(DATA)}** rows · **{int(DATA[LABEL].sum())}** PCOS "
                   f"({DATA[LABEL].mean():.0%})", icon="✅")
    else:
        for msg in ISSUES:
            st.error(msg)


def need_data() -> bool:
    if DATA is None:
        st.warning("Load a valid test CSV from the sidebar to continue.", icon="⚠️")
        with st.expander("Expected columns"):
            st.code(", ".join(INPUT_COLUMNS + [LABEL]))
        return False
    for note in NOTES:
        st.info(note, icon="ℹ️")
    return True


# --------------------------------------------------------------------------- #
#  Page 1 - single model
# --------------------------------------------------------------------------- #
def page_single_model():
    st.title("Model evaluator")
    st.markdown('<div class="lab-banner">Pick one of the five <b>pre-trained</b> pipelines. '
                'All metrics below are computed right now on the loaded test data '
                '(positive class = <b>PCOS</b>, threshold 0.5).</div>', unsafe_allow_html=True)
    if not need_data():
        return
    results = run_all_models(DATA)
    y = DATA[LABEL]

    pick_col, info_col = st.columns([1, 2])
    choice = pick_col.selectbox("Classifier", list(PIPES), index=0, key="chosen_model")
    info = CARD["models"][choice]
    info_col.markdown(f"<span class='lab-tag'>Tuned settings</span><br>"
                      f"<code>{info['BestParams']}</code><br>"
                      f"<span style='color:{INK_MUTED};font-size:.85rem'>file: model/{info['File']}"
                      f" · CV-MCC on training folds: {info['CV_MCC']:.3f}</span>",
                      unsafe_allow_html=True)

    res = results[choice]
    tiles = st.columns(6)
    for tile, key in zip(tiles, METRIC_ORDER):
        val = res["metrics"][key]
        tile.metric(key, "n/a" if pd.isna(val) else f"{val:.4f}", border=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Confusion matrix")
        st.plotly_chart(confusion_heatmap(y, res["pred"]), width="stretch",
                        config={"displayModeBar": False})
    with right:
        st.subheader("ROC curve")
        if y.nunique() == 2:
            st.plotly_chart(roc_figure(y, {choice: res["proba"]}), width="stretch",
                            config={"displayModeBar": False})
        else:
            st.info("ROC/AUC needs both classes in the test data.")

    st.subheader("Classification report")
    raw_report = classification_report(
        y, res["pred"], labels=[0, 1], target_names=["No PCOS (0)", "PCOS (1)"],
        output_dict=True, zero_division=0)
    acc = raw_report.pop("accuracy", accuracy_score(y, res["pred"]))
    report = pd.DataFrame(raw_report).T
    report.loc["accuracy"] = [np.nan, np.nan, acc, len(y)]
    shown = report.copy().astype(object)
    for col in ["precision", "recall", "f1-score"]:
        shown[col] = [("" if pd.isna(v) else f"{v:.4f}") for v in report[col]]
    shown["support"] = report["support"].astype(int).astype(str)
    st.dataframe(shown, width="stretch")

    st.subheader("Row-level predictions")
    preds = DATA.copy()
    preds.insert(0, "P(PCOS)", np.round(res["proba"], 4))
    preds.insert(0, "Predicted", res["pred"])
    preds.insert(0, "Actual", y.values)
    preds.insert(0, "Outcome", np.select(
        [(preds.Actual == 1) & (preds.Predicted == 1), (preds.Actual == 0) & (preds.Predicted == 0),
         (preds.Actual == 0) & (preds.Predicted == 1)], ["TP", "TN", "FP"], "FN"))
    preds = preds.drop(columns=[LABEL])
    only_errors = st.toggle("Show only misclassified rows", value=False)
    view = preds[preds.Outcome.isin(["FP", "FN"])] if only_errors else preds
    st.dataframe(view, width="stretch", height=300,
                 column_config={"P(PCOS)": st.column_config.ProgressColumn(
                     "P(PCOS)", min_value=0.0, max_value=1.0, format="%.3f")})
    st.download_button("Download predictions", preds.to_csv(index=False).encode(),
                       f"pcos_predictions_{choice.lower().replace(' ', '_')}.csv",
                       mime="text/csv", icon="⬇️")


# --------------------------------------------------------------------------- #
#  Page 2 - leaderboard
# --------------------------------------------------------------------------- #
def page_leaderboard():
    st.title("Leaderboard · all five models")
    st.markdown('<div class="lab-banner">Every saved pipeline scored on the same test rows. '
                'Best value per metric is highlighted; the ranking uses <b>MCC</b>, which stays '
                'honest under the ~2:1 class imbalance.</div>', unsafe_allow_html=True)
    if not need_data():
        return
    results = run_all_models(DATA)
    y = DATA[LABEL]
    board = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T[METRIC_ORDER]
    board.index.name = "ML Model Name"
    board = board.sort_values("MCC", ascending=False)

    leader = board.index[0]
    short = {"Logistic Regression": "Logistic Reg.", "Support Vector Machine": "SVM"}
    a, b, c = st.columns([1.3, 1, 1])
    a.metric("Top model by MCC", short.get(leader, leader), border=True, help=leader)
    b.metric("Its MCC", f"{board.loc[leader, 'MCC']:.4f}",
             f"{board.loc[leader, 'MCC'] - board['MCC'].iloc[1]:+.4f} vs runner-up", border=True)
    c.metric("Best AUC", f"{board['AUC'].max():.4f}", board["AUC"].idxmax(),
             delta_color="off", border=True)

    st.dataframe(board.style.format("{:.4f}").highlight_max(axis=0, color="#1c5cab"),
                 width="stretch")
    st.download_button("Download comparison table", board.round(4).to_csv().encode(),
                       "pcos_model_comparison.csv", mime="text/csv", icon="⬇️")

    st.subheader("Metric comparison")
    st.plotly_chart(metric_bars(board), width="stretch", config={"displayModeBar": False})

    st.subheader("ROC curves")
    if y.nunique() == 2:
        st.plotly_chart(roc_figure(y, {n: results[n]["proba"] for n in board.index}),
                        width="stretch", config={"displayModeBar": False})

    st.subheader("Confusion counts")
    rows = []
    for name in board.index:
        tn, fp, fn, tp = confusion_matrix(y, results[name]["pred"], labels=[0, 1]).ravel()
        rows.append({"ML Model Name": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                     "Missed PCOS (FN)": f"{fn} of {fn + tp}"})
    st.dataframe(pd.DataFrame(rows).set_index("ML Model Name"), width="stretch")


# --------------------------------------------------------------------------- #
#  Page 3 - patient simulator (what-if scoring of one patient)
# --------------------------------------------------------------------------- #
SYMPTOM_LABELS = {
    "Weight gain": "WeightGain", "Excess hair growth": "HairGrowth",
    "Skin darkening": "SkinDarkening", "Hair loss": "HairLoss", "Pimples / acne": "Pimples",
}
LIFESTYLE_LABELS = {"Fast-food habit": "FastFood", "Regular exercise": "RegExercise"}


def build_patient_row(age, height, weight, irregular, period_days, symptoms, lifestyle,
                      follicles=None, hormones=None) -> pd.DataFrame:
    """One-row frame in the training column order. Anything the user did not give stays NaN,
    so each saved pipeline fills it with its own training median / mode (no retraining)."""
    row = {col: np.nan for col in INPUT_COLUMNS}
    row.update(Age=age, HeightCm=height, WeightKg=weight, BMI=weight / (height / 100) ** 2,
               CycleType=4.0 if irregular else 2.0, PeriodDays=period_days)
    for label, col in {**SYMPTOM_LABELS, **LIFESTYLE_LABELS}.items():
        row[col] = float(label in symptoms or label in lifestyle)
    if follicles:
        row["FollicleCountL"], row["FollicleCountR"] = follicles
    if hormones:
        row["FSH"], row["LH"], row["AMH"] = hormones
        row["FSHtoLH"] = hormones[0] / max(hormones[1], 0.05)
    return pd.DataFrame([row])[INPUT_COLUMNS]


def page_simulator():
    st.title("Patient simulator")
    st.markdown('<div class="lab-banner">Move the controls and watch all five saved models re-score '
                'one hypothetical patient <b>live</b>. Anything you leave out is filled in by each '
                'pipeline with its training median, so the ◇ markers show the '
                '<b>typical-patient baseline</b>.</div>', unsafe_allow_html=True)

    controls, readout = st.columns([5, 6], gap="large")
    with controls:
        with st.container(border=True):
            st.markdown('<span class="lab-tag">1 · Body</span>', unsafe_allow_html=True)
            age = st.slider("Age (years)", 18, 50, 28)
            height = st.slider("Height (cm)", 135, 185, 158)
            weight = st.slider("Weight (kg)", 30, 110, 58)
            st.caption(f"BMI ≈ {weight / (height / 100) ** 2:.1f}")
        with st.container(border=True):
            st.markdown('<span class="lab-tag">2 · Cycle & symptoms</span>', unsafe_allow_html=True)
            cycle = st.segmented_control("Cycle pattern", ["Regular", "Irregular"],
                                         default="Regular", key="sim_cycle") or "Regular"
            period_days = st.slider("Period length (days)", 0, 12, 5)
            symptoms = st.pills("Symptoms present", list(SYMPTOM_LABELS),
                                selection_mode="multi", key="sim_symptoms") or []
            lifestyle = st.pills("Lifestyle", list(LIFESTYLE_LABELS),
                                 selection_mode="multi", key="sim_lifestyle") or []
        with st.container(border=True):
            st.markdown('<span class="lab-tag">3 · Optional test results</span>',
                        unsafe_allow_html=True)
            follicles = hormones = None
            if st.toggle("I have ultrasound follicle counts", key="sim_scan"):
                c1, c2 = st.columns(2)
                follicles = (c1.number_input("Left ovary", 0, 30, 6, key="sim_fl"),
                             c2.number_input("Right ovary", 0, 30, 6, key="sim_fr"))
            if st.toggle("I have hormone results", key="sim_hormones"):
                c1, c2, c3 = st.columns(3)
                hormones = (c1.number_input("FSH", 0.1, 60.0, 5.0, step=0.1, key="sim_fsh"),
                            c2.number_input("LH", 0.1, 60.0, 3.0, step=0.1, key="sim_lh"),
                            c3.number_input("AMH", 0.0, 40.0, 4.0, step=0.1, key="sim_amh"))

    patient = build_patient_row(age, height, weight, cycle == "Irregular", period_days,
                                symptoms, lifestyle, follicles, hormones)
    typical = pd.DataFrame([{col: np.nan for col in INPUT_COLUMNS}])[INPUT_COLUMNS]
    names = list(PIPES)
    risk = {n: float(PIPES[n].predict_proba(patient)[0, 1]) for n in names}
    base = {n: float(PIPES[n].predict_proba(typical)[0, 1]) for n in names}
    flagged = [n for n in names if risk[n] >= 0.5]
    given = int(patient.notna().sum(axis=1).iloc[0])

    with readout:
        if len(flagged) >= 3:
            st.warning(f"**{len(flagged)} of 5 models** flag a PCOS-like profile.", icon="⚠️")
        elif flagged:
            st.info(f"**{len(flagged)} of 5 models** flag PCOS - the models disagree.", icon="↔️")
        else:
            st.success("**0 of 5 models** flag a PCOS-like profile.", icon="✅")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=names, x=[risk[n] for n in names], orientation="h", name="This patient",
            marker=dict(color=[MODEL_HUES[n] for n in names], line=dict(color=PANEL, width=2),
                        cornerradius=4),
            text=[f"{risk[n]:.0%}" for n in names], textposition="outside", cliponaxis=False,
            textfont=dict(color=INK), hovertemplate="%{y}<br>P(PCOS) <b>%{x:.1%}</b><extra></extra>"))
        fig.add_trace(go.Scatter(
            y=names, x=[base[n] for n in names], mode="markers", name="Typical patient",
            marker=dict(symbol="diamond-open", size=11, color=INK_SOFT, line=dict(width=2)),
            hovertemplate="%{y}<br>typical patient <b>%{x:.1%}</b><extra></extra>"))
        fig.add_vline(x=0.5, line=dict(color=INK_MUTED, width=1, dash="dot"),
                      annotation_text="decision threshold 0.5", annotation_font_color=INK_MUTED)
        fig.update_xaxes(range=[0, 1.12], tickformat=".0%", title="Probability of PCOS")
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig = _style(fig, 380)
        fig.update_layout(legend=dict(orientation="h", y=1.14, x=0, font=dict(color=INK_SOFT)),
                          margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        table = pd.DataFrame({
            "P(PCOS)": [risk[n] for n in names],
            "vs typical patient": [risk[n] - base[n] for n in names],
            "Verdict": ["⚠️ PCOS-like" if risk[n] >= 0.5 else "✅ Not flagged" for n in names],
        }, index=pd.Index(names, name="ML Model Name"))
        st.dataframe(table.style.format({"P(PCOS)": "{:.1%}", "vs typical patient": "{:+.1%}"}),
                     width="stretch")
        st.caption(f"{given} of {len(INPUT_COLUMNS)} inputs supplied; the other "
                   f"{len(INPUT_COLUMNS) - given} are imputed by each pipeline. Ultrasound "
                   "follicle counts carry the most weight, so results without them are "
                   "less informative.")
    st.warning("Educational demo of the trained models - not a diagnosis.", icon="🩺")


# --------------------------------------------------------------------------- #
#  Page 4 - notes
# --------------------------------------------------------------------------- #
def page_notes():
    st.title("Project notes")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(f"""
**Goal.** Predict whether a patient has *Polycystic Ovary Syndrome* (binary target `PCOS`)
from 41 clinical, hormonal and ultrasound attributes.

**Data.** Kaggle PCOS dataset (Prasoon Kottarathil, 10 hospitals in Kerala, India):
541 patients, 364 negative / 177 positive. `PCOS_data_without_infertility.xlsx`
cross-checked against `PCOS_infertility.csv` on the serial number.

**Hold-out design.** Stratified **75 / 25** split, `random_state={CARD['seed']}` →
{CARD['n_train']} training rows / {CARD['n_test']} test rows (`test_data.csv`).

**Pipeline inside every `.pkl`.**
1. `ClinicalFeatureBuilder` – adds *IrregularCycle*, *FollicleTotal*, *LHtoFSH*, *SymptomScore*
2. `PercentileClipper` – winsorises continuous columns at the training 1st/99th percentiles
3. `SimpleImputer` – median (continuous) / most-frequent (flags, blood group)
4. Scaler chosen by grid search – `RobustScaler`, `PowerTransformer` (Yeo-Johnson) or `MinMaxScaler`
5. One-hot encoding of blood group → classifier

**Tuning.** `GridSearchCV` with {CARD['cv']}.
""")
    with right:
        st.markdown("<span class='lab-tag'>Tuned hyper-parameters</span>", unsafe_allow_html=True)
        for name, m in CARD["models"].items():
            st.markdown(f"**{name}**  \n`{m['BestParams']}`")
        st.markdown("<span class='lab-tag'>Environment</span>", unsafe_allow_html=True)
        st.caption(f"scikit-learn {CARD['sklearn_version']} (pickles) · Streamlit {st.__version__}")

    st.markdown("<span class='lab-tag'>Training-time hold-out scores</span>", unsafe_allow_html=True)
    ref = pd.DataFrame({n: {k: m[k] for k in METRIC_ORDER + ["TrainAccuracy", "CV_MCC"]}
                        for n, m in CARD["models"].items()}).T
    ref.index.name = "ML Model Name"
    st.dataframe(ref.style.format("{:.4f}"), width="stretch")
    st.caption("Frozen at training time. The other pages recompute the six metrics live from the "
               "loaded test file, so they match exactly when the bundled `test_data.csv` is used.")
    st.warning("Educational project only - not a diagnostic tool.", icon="🩺")


nav = st.navigation([
    st.Page(page_single_model, title="Model evaluator", icon="🔬", default=True),
    st.Page(page_leaderboard, title="Leaderboard", icon="🏁", url_path="leaderboard"),
    st.Page(page_simulator, title="Patient simulator", icon="🎛️", url_path="simulator"),
    st.Page(page_notes, title="Project notes", icon="📘", url_path="notes"),
], position="sidebar")
nav.run()
