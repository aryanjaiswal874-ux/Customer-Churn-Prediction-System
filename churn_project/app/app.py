"""
Customer Churn Prediction App
=============================
Streamlit app with two modes:
  1. Manual single-customer prediction
  2. CSV batch upload with download
"""

import io
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Project root ───────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_generator import get_feature_columns
from utils.model_utils import get_feature_importance
from utils.preprocessing import (
    clean_dataframe,
    get_required_columns,
    prepare_for_prediction,
    suggest_column_fixes,
    validate_uploaded_dataframe,
)
from utils.visualizations import (
    plot_churn_distribution,
    plot_churn_probability_histogram,
    plot_feature_importance,
    plot_model_comparison,
    plot_risk_segmentation,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PIPELINE_PATH = PROJECT_ROOT / "models" / "churn_pipeline.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "training_metrics.json"
SAMPLE_CSV_PATH = PROJECT_ROOT / "data" / "sample_template.csv"

# ══════════════════════════════════════════════════════════════════════════════
# Page config – must be first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
    border-bottom: 1px solid #334155;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    text-align: center;
}
.app-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    color: #F1F5F9;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-header .subtitle {
    color: #64748B;
    font-size: 0.95rem;
    margin-top: 4px;
}

/* ── Metric cards ── */
.metric-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .label { color: #64748B; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-card .value { color: #F1F5F9; font-size: 2rem; font-weight: 700; margin-top: 4px; }
.metric-card .value.danger { color: #EF4444; }
.metric-card .value.success { color: #22C55E; }

/* ── Prediction badge ── */
.pred-badge-churn {
    background: #450A0A; border: 1px solid #EF4444;
    color: #FCA5A5; border-radius: 8px;
    padding: 0.9rem 1.4rem; font-size: 1.3rem; font-weight: 700;
    text-align: center; margin-top: 0.5rem;
}
.pred-badge-stay {
    background: #052E16; border: 1px solid #22C55E;
    color: #86EFAC; border-radius: 8px;
    padding: 0.9rem 1.4rem; font-size: 1.3rem; font-weight: 700;
    text-align: center; margin-top: 0.5rem;
}

/* ── Info/warning boxes ── */
.info-box {
    background: #0C1A2E; border-left: 4px solid #3B82F6;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.6rem 0;
    color: #93C5FD; font-size: 0.9rem;
}
.warn-box {
    background: #1C1008; border-left: 4px solid #F59E0B;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.6rem 0;
    color: #FCD34D; font-size: 0.9rem;
}
.success-box {
    background: #052E16; border-left: 4px solid #22C55E;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.6rem 0;
    color: #86EFAC; font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline loader (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model pipeline…")
def load_pipeline():
    """Load the saved sklearn pipeline from disk."""
    if not PIPELINE_PATH.exists():
        return None
    with open(PIPELINE_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar(pipeline_data, metrics_data):
    with st.sidebar:
        st.markdown("## 📡 Churn Predictor")
        st.markdown("---")

        # Model info
        if pipeline_data:
            best = pipeline_data.get("best_model_name", "Unknown")
            st.markdown(f"**Active Model:** `{best}`")

            if metrics_data:
                scores = metrics_data["all_results"].get(best, {})
                st.markdown(f"**Test Accuracy:** `{scores.get('Accuracy', 0):.1%}`")
                st.markdown(f"**F1 Score:** `{scores.get('F1 Score', 0):.1%}`")
                st.markdown(f"**ROC-AUC:** `{scores.get('ROC-AUC', 0):.1%}`")
        else:
            st.warning("⚠️ Model not found. Run `python train.py` first.")

        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown(
            """
**Manual Mode**
- Fill in customer details
- Click **Predict** to get instant result

**CSV Upload Mode**
- Upload a CSV with required columns
- Download predictions with churn scores
- Visualize churn distribution

**Need the template?**
Download the sample CSV below.
"""
        )

        # Sample CSV download
        if SAMPLE_CSV_PATH.exists():
            with open(SAMPLE_CSV_PATH, "rb") as f:
                st.download_button(
                    label="📥 Download Sample CSV Template",
                    data=f,
                    file_name="sample_churn_template.csv",
                    mime="text/csv",
                    width="stretch",
                )

        st.markdown("---")
        st.markdown("### 🔑 Required Columns")
        cols = get_required_columns()
        for c in cols:
            st.markdown(f"- `{c}`")

        st.markdown("---")
        st.caption("Built with Python · Scikit-learn · Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
# Manual Input Mode
# ══════════════════════════════════════════════════════════════════════════════
def render_manual_mode(pipeline_data):
    st.markdown("### 👤 Enter Customer Details")

    numerical_cols, _ = get_feature_columns()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col2:
        st.markdown("**📶 Service Details**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 55.0, step=0.5)
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges), step=10.0
        )
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["Yes", "No", "No phone service"]
        )
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col3:
        st.markdown("**🔒 Add-ons & Contract**")
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    col4, col5 = st.columns(2)
    with col4:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col5:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    if st.button("🔮 Predict Churn", type="primary", width="stretch"):
        if pipeline_data is None:
            st.error("❌ Model not loaded. Please run `python train.py` first.")
            return

        input_data = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "SeniorCitizen": senior,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                }
            ]
        )

        try:
            pipeline = pipeline_data["pipeline"]
            prediction = pipeline.predict(input_data)[0]
            probability = None
            if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
                prob_arr = pipeline.predict_proba(input_data)[0]
                classes = pipeline.classes_
                yes_idx = list(classes).index("Yes")
                probability = prob_arr[yes_idx]

            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")

            r1, r2, r3 = st.columns(3)
            with r1:
                if prediction == "Yes":
                    st.markdown('<div class="pred-badge-churn">⚠️ WILL CHURN</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="pred-badge-stay">✅ WILL STAY</div>', unsafe_allow_html=True)
            with r2:
                if probability is not None:
                    pct = probability * 100
                    color = "danger" if pct > 50 else "success"
                    st.markdown(
                        f'<div class="metric-card"><div class="label">Churn Probability</div>'
                        f'<div class="value {color}">{pct:.1f}%</div></div>',
                        unsafe_allow_html=True,
                    )
            with r3:
                risk = "🔴 High" if (probability or 0) > 0.66 else ("🟡 Medium" if (probability or 0) > 0.33 else "🟢 Low")
                st.markdown(
                    f'<div class="metric-card"><div class="label">Risk Level</div>'
                    f'<div class="value">{risk}</div></div>',
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            logger.exception("Manual prediction failed")


# ══════════════════════════════════════════════════════════════════════════════
# CSV Upload Mode
# ══════════════════════════════════════════════════════════════════════════════
def render_csv_mode(pipeline_data):
    st.markdown("### 📂 Batch CSV Prediction")

    st.markdown(
        '<div class="info-box">Upload a CSV file with customer data. '
        "Required columns are listed in the sidebar. "
        "Download the sample template to get started.</div>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=["csv"], help="Max 200 MB"
    )

    if uploaded_file is None:
        st.info("👆 Upload a CSV file to run batch predictions.")
        return

    # ── Read CSV ──────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        return

    st.markdown(f"**Loaded {len(df):,} rows × {len(df.columns)} columns**")
    st.markdown("#### 🔍 Data Preview (first 5 rows)")
    st.dataframe(df.head(), width="stretch")

    # ── Validate columns ──────────────────────────────────────────────────────
    is_valid, missing_cols, extra_cols = validate_uploaded_dataframe(df)

    if not is_valid:
        st.error(f"❌ **{len(missing_cols)} required column(s) missing from your CSV:**")
        for col in missing_cols:
            st.markdown(f"- `{col}`")

        # Auto-detect possible fixes
        suggestions = suggest_column_fixes(df)
        has_suggestions = any(v is not None for v in suggestions.values())
        if has_suggestions:
            st.markdown(
                '<div class="warn-box">💡 <b>Possible column name mismatches detected:</b><br>' +
                "<br>".join(
                    f"  • Missing <code>{k}</code> → did you mean <code>{v}</code>?"
                    for k, v in suggestions.items()
                    if v is not None
                ) +
                "</div>",
                unsafe_allow_html=True,
            )
        st.stop()

    if extra_cols:
        st.markdown(
            f'<div class="info-box">ℹ️ Extra columns will be ignored: {", ".join(f"<code>{c}</code>" for c in extra_cols)}</div>',
            unsafe_allow_html=True,
        )

    if pipeline_data is None:
        st.error("❌ Model not loaded. Please run `python train.py` first.")
        return

    # ── Run predictions ───────────────────────────────────────────────────────
    if st.button("🚀 Run Batch Predictions", type="primary", width="stretch"):
        with st.spinner("Processing predictions…"):
            try:
                cleaned = clean_dataframe(df)
                X = prepare_for_prediction(cleaned)

                pipeline = pipeline_data["pipeline"]
                predictions = pipeline.predict(X)
                probabilities = None
                if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
                    prob_matrix = pipeline.predict_proba(X)
                    classes = list(pipeline.classes_)
                    yes_idx = classes.index("Yes")
                    probabilities = prob_matrix[:, yes_idx]

                # Add result columns to original df
                result_df = df.copy()
                result_df["Churn Prediction"] = predictions
                if probabilities is not None:
                    result_df["Churn Probability (%)"] = np.round(probabilities * 100, 2)
                    result_df["Risk Level"] = pd.cut(
                        probabilities,
                        bins=[-0.001, 0.33, 0.66, 1.001],
                        labels=["Low", "Medium", "High"],
                    )

                # ── Summary metrics ───────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 📊 Prediction Summary")

                total = len(result_df)
                churn_count = (result_df["Churn Prediction"] == "Yes").sum()
                stay_count = total - churn_count
                churn_rate = churn_count / total * 100

                m1, m2, m3, m4 = st.columns(4)
                metrics = [
                    ("Total Customers", f"{total:,}", ""),
                    ("Will Churn", f"{churn_count:,}", "danger"),
                    ("Will Stay", f"{stay_count:,}", "success"),
                    ("Churn Rate", f"{churn_rate:.1f}%", "danger" if churn_rate > 30 else ""),
                ]
                for col, (label, value, cls) in zip([m1, m2, m3, m4], metrics):
                    with col:
                        st.markdown(
                            f'<div class="metric-card"><div class="label">{label}</div>'
                            f'<div class="value {cls}">{value}</div></div>',
                            unsafe_allow_html=True,
                        )

                # ── Results table ─────────────────────────────────────────────
                st.markdown("#### 📋 Results Table")
                display_cols = ["Churn Prediction"]
                if "Churn Probability (%)" in result_df.columns:
                    display_cols += ["Churn Probability (%)", "Risk Level"]
                st.dataframe(
                    result_df[list(df.columns) + display_cols].head(100),
                    width="stretch",
                )

                # ── Visualizations ────────────────────────────────────────────
                st.markdown("### 📈 Visualizations")

                viz1, viz2 = st.columns(2)
                with viz1:
                    st.pyplot(plot_churn_distribution(result_df["Churn Prediction"]))
                with viz2:
                    if probabilities is not None:
                        st.pyplot(plot_churn_probability_histogram(probabilities))

                if probabilities is not None:
                    viz3, viz4 = st.columns(2)
                    with viz3:
                        st.pyplot(plot_risk_segmentation(probabilities))
                    with viz4:
                        num_cols, cat_cols = get_feature_columns()
                        feat_names, importances = get_feature_importance(
                            pipeline, num_cols, cat_cols
                        )
                        if len(feat_names):
                            st.pyplot(plot_feature_importance(feat_names, importances))

                # ── Download ──────────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 📥 Download Results")
                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="⬇️ Download Predictions CSV",
                    data=csv_buffer.getvalue().encode(),
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    width="stretch",
                    type="primary",
                )
                st.markdown(
                    '<div class="success-box">✅ Predictions complete! Download your results above.</div>',
                    unsafe_allow_html=True,
                )

            except KeyError as e:
                st.error(f"❌ Column '{e}' is missing in the uploaded data. Check the sidebar for required columns.")
                logger.exception("Column error during batch prediction")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                logger.exception("Batch prediction failed")


# ══════════════════════════════════════════════════════════════════════════════
# Model Performance Tab
# ══════════════════════════════════════════════════════════════════════════════
def render_model_performance(pipeline_data, metrics_data):
    st.markdown("### 🏆 Model Training Results")

    if metrics_data is None:
        st.warning("No training metrics found. Run `python train.py` first.")
        return

    all_results = metrics_data["all_results"]
    best = metrics_data["best_model"]

    st.markdown(f"**Selected Model: `{best}`** ← chosen by highest F1 Score")
    st.markdown("---")

    # Comparison chart
    st.pyplot(plot_model_comparison(all_results))

    # Metrics table
    st.markdown("#### Detailed Metrics Table")
    rows = []
    for model_name, scores in all_results.items():
        row = {"Model": model_name + (" ✓" if model_name == best else "")}
        row.update({k: f"{v:.4f}" for k, v in scores.items()})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch")

    # Feature importance
    if pipeline_data:
        st.markdown("---")
        st.markdown("#### 🔍 Feature Importances")
        num_cols, cat_cols = get_feature_columns()
        feat_names, importances = get_feature_importance(
            pipeline_data["pipeline"], num_cols, cat_cols
        )
        if len(feat_names):
            st.pyplot(plot_feature_importance(feat_names, importances, top_n=15))
        else:
            st.info("Feature importances not available for this model type.")


# ══════════════════════════════════════════════════════════════════════════════
# Main App
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Header
    st.markdown(
        """
        <div class="app-header">
            <h1>📡 Customer Churn Prediction App</h1>
            <div class="subtitle">Telecom Customer Retention Intelligence · Powered by Machine Learning</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load resources
    pipeline_data = load_pipeline()
    metrics_data = load_metrics()

    # Sidebar
    render_sidebar(pipeline_data, metrics_data)

    # Tabs
    tab1, tab2, tab3 = st.tabs(
        ["👤 Manual Prediction", "📂 CSV Batch Upload", "📊 Model Performance"]
    )

    with tab1:
        render_manual_mode(pipeline_data)

    with tab2:
        render_csv_mode(pipeline_data)

    with tab3:
        render_model_performance(pipeline_data, metrics_data)


if __name__ == "__main__":
    main()
