"""
DS Playbook — Interactive Streamlit Dashboard.
Run with: streamlit run app/dashboard.py
Requires: pip install -e . from repo root
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from dskit.eda import missing_report
    from dskit.features import encode_categoricals, scale_features, handle_missing
    from dskit.models import compare_models
    from dskit.viz import (
        plot_missing_values,
        plot_correlation_matrix,
        plot_distributions,
        plot_model_comparison,
    )
except ImportError:
    st.error(
        "dskit not found. Run `pip install -e .` from the repo root first."
    )
    st.stop()

st.set_page_config(
    page_title="DS Playbook",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧪 DS Playbook — Interactive Data Explorer")
st.caption("Upload any CSV and get instant EDA, correlation analysis, and model benchmarking.")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    st.divider()
    st.caption("DS Playbook v0.1.0")

if uploaded_file is None:
    st.info("📂 Upload a CSV file using the sidebar to get started.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Modules", "4")
    col2.metric("Notebooks", "4")
    col3.metric("ML Models", "5+")
    col4.metric("Dashboard Tabs", "4")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success(f"Loaded **{df.shape[0]:,} rows × {df.shape[1]} columns**")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 EDA", "🔗 Correlations", "🤖 Models"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())
    c4.metric("Numeric Columns", df.select_dtypes(include="number").shape[1])
    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Column Types")
    dtype_df = pd.DataFrame({
        "Column": df.dtypes.index,
        "Type": df.dtypes.astype(str).values,
        "Unique Values": [df[c].nunique() for c in df.columns],
        "Missing": [df[c].isnull().sum() for c in df.columns],
    })
    st.dataframe(dtype_df, use_container_width=True)

with tab2:
    st.subheader("Missing Values")
    missing = missing_report(df)
    if missing.empty:
        st.success("No missing values found!")
    else:
        st.dataframe(missing, use_container_width=True)
        fig = plot_missing_values(df)
        if fig:
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Numeric Summary Statistics")
    st.dataframe(df.describe(include="number"), use_container_width=True)

    st.subheader("Distributions")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        selected = st.multiselect(
            "Select columns",
            numeric_cols,
            default=numeric_cols[:min(4, len(numeric_cols))],
        )
        if selected:
            fig = plot_distributions(df, cols=selected)
            if fig:
                st.pyplot(fig)
                plt.close(fig)

with tab3:
    st.subheader("Correlation Matrix")
    numeric_only = df.select_dtypes(include="number")
    if numeric_only.shape[1] < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
    else:
        method = st.selectbox("Method", ["pearson", "spearman", "kendall"])
        fig = plot_correlation_matrix(df, method=method)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Top Correlated Pairs")
        corr = numeric_only.corr(method=method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        top = upper.stack().sort_values(ascending=False).head(10)
        top_df = top.reset_index()
        top_df.columns = ["Feature 1", "Feature 2", "Correlation"]
        st.dataframe(top_df, use_container_width=True)

with tab4:
    st.subheader("Auto ML Benchmark")
    st.info("Select a target column and task type, then click Run.")
    target = st.selectbox("Target column", df.columns.tolist())
    task = st.selectbox("Task type", ["classification", "regression"])
    cv_folds = st.slider("CV Folds", 2, 10, 5)

    if st.button("🚀 Run Model Comparison", type="primary"):
        with st.spinner("Training models... this may take a minute"):
            try:
                y = df[target]
                X_raw = df.drop(columns=[target])
                X_clean = handle_missing(X_raw)
                X_enc = encode_categoricals(X_clean)
                X_scaled = scale_features(X_enc)
                results = compare_models(X_scaled, y, task=task, cv=cv_folds)
                st.success("Done!")
                st.dataframe(results, use_container_width=True)
                fig = plot_model_comparison(results)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as exc:
                st.error(f"Error during model training: {exc}")
