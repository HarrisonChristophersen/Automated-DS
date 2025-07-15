# main.py (modified for trend analysis with plain language and improved labels)
import os
import pandas as pd
import numpy as np

# Import our custom modules
from importing import load_and_inspect, detect_column_types
from preprocessing import handle_missing_values, feature_engineering
from modeling import run_h2o_automl
from explainability import (
    generate_lime_explanation,
    summarize_lime_explanations,
    generate_narrative,
    plot_feature_importance,
)

def main():
    # ---------------------------
    # 1. Data Import and Inspection
    # ---------------------------
    file_path = "C:/Users/purpl/Downloads/Amazon Sales.xlsx"  # update as needed
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print("Loading and inspecting data...")
    df = load_and_inspect(file_path)
    
    # ---------------------------
    # 2. Detect Column Types
    # ---------------------------
    column_types = detect_column_types(df)
    
    # ---------------------------
    # 3. Handle Missing Values
    # ---------------------------
    df_clean = handle_missing_values(
        df, 
        num_strategy="median", 
        cat_strategy="mode", 
        date_strategy="ffill", 
        drop_threshold=0.5, 
        delete_rows=False
    )
    
    # ---------------------------
    # 4. Feature Engineering
    # ---------------------------
    df_processed = feature_engineering(df_clean, column_types, scale_numeric=True)
    
       # ---------------------------
    # 5. Model Configuration (User Input)
    # ---------------------------
    print("\nAvailable columns:", list(df_processed.columns))
    
    # Keep asking until a valid target column is provided
    while True:
        target = input("Enter the target column (result) from the above columns: ").strip()
        if target in df_processed.columns:
            break
        else:
            print(f"Target column '{target}' not found in the dataset. Please try again.")

    predictors_input = input(
        "Enter comma-separated predictor columns (or leave blank for all columns except target): "
    ).strip()
    if predictors_input:
        predictor_cols = [col.strip() for col in predictors_input.split(",") if col.strip() in df_processed.columns]
        if not predictor_cols:
            print("No valid predictor columns entered. Using all columns except the target.")
            predictor_cols = [col for col in df_processed.columns if col != target]
        df_model = df_processed[predictor_cols + [target]]
    else:
        df_model = df_processed.copy()
        predictor_cols = [col for col in df_model.columns if col != target]

    # ---------------------------
    # 6. Model Training with H2O AutoML
    # ---------------------------
    print(f"\nRunning H2O AutoML to predict '{target}' ...")
    aml_model = run_h2o_automl(df_model, target=target, max_models=5, max_runtime_secs=60)
    best_model = aml_model.leader
    
    # ---------------------------
    # 7. Trend Analysis and Feature Importance
    # ---------------------------
    print("\nFeature Importance:")
    try:
        varimp = best_model.varimp(use_pandas=True)
        print(varimp)
    except Exception as e:
        print("Variable importance not available for this model. Error:", e)
    
    # ---------------------------
    # 8. Aggregated LIME Explanations for Trend Insights
    # ---------------------------
    print("\nGenerating aggregated LIME explanations for trend insights...")
    X_train = df_model[predictor_cols].values

    # Filter out constant columns (i.e. columns with near-zero variance) from predictors for LIME
    var_values = np.var(X_train, axis=0)
    non_constant_indices = np.where(var_values > 1e-6)[0]
    if len(non_constant_indices) < X_train.shape[1]:
        print("Warning: Some predictor columns are constant and will be dropped for LIME analysis.")
    X_train_filtered = X_train[:, non_constant_indices]
    predictor_cols_filtered = [predictor_cols[i] for i in non_constant_indices]

    num_samples = min(5, len(X_train_filtered))
    sample_indices = np.random.choice(range(len(X_train_filtered)), size=num_samples, replace=False)

    aggregated_explanations = []
    for idx in sample_indices:
        instance = X_train_filtered[idx]
        exp = generate_lime_explanation(X_train_filtered, predictor_cols_filtered, instance, best_model, mode='regression')
        aggregated_explanations.append(exp.as_list())

    # ---------------------------
    # 9. Summarize and Visualize Explanations
    # ---------------------------
    summary = summarize_lime_explanations(aggregated_explanations)
    narrative = generate_narrative(summary, target)
    
    print("\nSummary of Trends Based on Aggregated LIME Explanations:")
    print(narrative)
    
    # Plot the aggregated feature importance using the target name
    plot_feature_importance(summary, target)
    
if __name__ == '__main__':
    main()
