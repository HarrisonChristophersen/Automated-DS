# explainability.py
import pandas as pd
import numpy as np
import h2o
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

def h2o_predict_fn(data, best_model, feature_names):
    """
    A wrapper function to generate predictions using the H2O model.
    
    :param data: NumPy array of input features (one instance or multiple)
    :param best_model: The H2O model object (e.g. aml.leader)
    :param feature_names: List of predictor column names
    :return: Predictions as a NumPy array.
    """
    # Convert the numpy array to a DataFrame with correct feature names
    df_data = pd.DataFrame(data, columns=feature_names)
    # Convert the DataFrame to an H2OFrame
    h2o_data = h2o.H2OFrame(df_data)
    # Generate predictions using the best H2O model
    predictions = best_model.predict(h2o_data)
    # Convert predictions to a DataFrame and then to a NumPy array
    return predictions.as_data_frame().values

def generate_lime_explanation(df, model, target, num_features=5, num_samples=10):
    X = df.drop(columns=[target]).values
    feature_names = df.drop(columns=[target]).columns.tolist()
    explainer = LimeTabularExplainer(
        X,
        feature_names=feature_names,
        class_names=[target],
        verbose=False
    )
    explanations = []
    for row in X[:num_samples]:
        exp = explainer.explain_instance(
            row,
            lambda z: model.predict(h2o.H2OFrame([z])).as_data_frame().values[0],
            num_features=num_features
        )
        explanations.append(exp)
    return explanations

def extract_feature(condition):
    """
    Extracts the feature name from a LIME condition string.
    For example, from "-0.15 < Pts <= 0.67" it returns "Pts",
    and from "Ast > 0.60" it returns "Ast".
    """
    # Remove any extraneous characters
    condition = condition.replace("(", "").replace(")", "").replace("'", "").replace('"', "")
    parts = condition.split()
    # If the first part is numeric, assume the feature is in the third position.
    try:
        float(parts[0])
        return parts[2]
    except ValueError:
        return parts[0]

def summarize_lime_explanations(exps):
    agg = {}
    for exp in exps:
        for feat, weight in exp.as_list():
            agg[feat] = agg.get(feat, 0) + weight
    count = len(exps)
    for feat in agg:
        agg[feat] /= count
    return agg


def generate_narrative(summary, target):
    lines = [f"• `{feat}` has an average impact of {imp:.3f} on `{target}`"
             for feat, imp in summary.items()]
    return "\n".join(lines)

def plot_feature_importance(summary, target):
    feats, imps = zip(*summary.items())
    plt.figure(figsize=(8, 5))
    plt.barh(feats, imps)
    plt.xlabel(f"Impact on {target}")
    plt.title("Feature Importance (LIME)")
    plt.tight_layout()
    plt.show()
    