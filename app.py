# app.py
import streamlit as st
import pandas as pd
from main import (load_and_inspect, detect_column_types,
                  handle_missing_values, feature_engineering,
                  run_h2o_automl, generate_lime_explanation,
                  summarize_lime_explanations, generate_narrative,
                  plot_feature_importance)

st.set_page_config(page_title="Automated DS", layout="centered")

st.title("🤖 Automated DS Web App")

# 1) File uploader
uploaded = st.file_uploader("Upload your CSV or Excel file", type=["csv","xlsx"])
if not uploaded:
    st.info("Please upload a dataset to get started.")
    st.stop()

# 2) Read into DataFrame
df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
st.write("### Preview of your data", df.head())

# 3) Auto-detect column types & show dropdowns
types = detect_column_types(df)
all_cols = df.columns.tolist()

target = st.selectbox("Select the *target* column to predict", all_cols)
predictors = st.multiselect(
    "Select *predictor* columns (leave blank to use all except target)",
    [c for c in all_cols if c != target],
    default=[c for c in all_cols if c != target]
)

# 4) Run button
if st.button("Run AutoML & Explain"):
    with st.spinner("Running AutoML… this can take a few minutes"):
        df2 = handle_missing_values(df.copy(), types)
        df2 = feature_engineering(df2)
        model = run_h2o_automl(df2, target)
        exps = generate_lime_explanation(df2[predictors + [target]], model, target,
                                         num_features=5, num_samples=10)
        summary = summarize_lime_explanations(exps)
        narrative = generate_narrative(summary, target)

    st.success("✅ Analysis Complete")
    st.subheader("Key Insights")
    st.markdown(narrative)
    st.subheader("Feature Impact Chart")
    fig = plot_feature_importance(summary, target)  # modify plot to return fig instead of show()
    st.pyplot(fig)
