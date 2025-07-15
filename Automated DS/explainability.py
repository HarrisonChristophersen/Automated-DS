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

def generate_lime_explanation(X_train, feature_names, instance, best_model, mode='regression'):
    """
    Generate a LIME explanation for a single instance.
    
    :param X_train: Training data as a NumPy array
    :param feature_names: List of predictor names
    :param instance: A single instance (1D NumPy array) from the dataset
    :param best_model: The H2O model object (e.g. aml.leader)
    :param mode: 'regression' or 'classification'
    :return: The LIME explanation object
    """
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode=mode
    )
    
    # Create a prediction function that passes best_model and feature_names
    predict_fn = lambda data: h2o_predict_fn(data, best_model, feature_names)
    
    # Generate the explanation for the given instance
    exp = explainer.explain_instance(instance, predict_fn)
    return exp

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

def summarize_lime_explanations(aggregated_explanations):
    """
    Aggregates LIME explanation contributions for each feature across multiple instances.
    
    :param aggregated_explanations: List of lists of tuples (condition, contribution) 
                                    for each sample.
    :return: Dictionary mapping feature names to average contribution.
    """
    feature_contribs = {}
    
    for explanation in aggregated_explanations:
        for condition, contrib in explanation:
            feature = extract_feature(condition)
            if feature not in feature_contribs:
                feature_contribs[feature] = []
            feature_contribs[feature].append(contrib)
    
    # Compute average contribution per feature
    summary = {}
    for feature, contribs in feature_contribs.items():
        summary[feature] = sum(contribs) / len(contribs)
    
    return summary

def generate_narrative(summary, target_name):
    """
    Converts the aggregated feature contributions into a plain language narrative.
    
    :param summary: Dictionary mapping feature names to average contributions.
    :param target_name: The name of the target variable.
    :return: String with plain language statements.
    """
    narrative_lines = []
    for feature, avg in summary.items():
        if avg > 0:
            narrative_lines.append(f"Increases in {feature} tend to increase {target_name}.")
        elif avg < 0:
            narrative_lines.append(f"Increases in {feature} tend to decrease {target_name}.")
        else:
            narrative_lines.append(f"{feature} does not have a significant impact on {target_name}.")
    return "\n".join(narrative_lines)

def plot_feature_importance(summary, target_name):
    """
    Plots a bar chart of the aggregated feature contributions.
    
    :param summary: Dictionary mapping feature names to average contributions.
    :param target_name: The name of the target variable.
    """
    features = list(summary.keys())
    avg_contribs = [summary[feat] for feat in features]
    
    plt.figure(figsize=(10, 6))
    plt.bar(features, avg_contribs, color='skyblue')
    plt.xlabel("Features")
    plt.ylabel(f"Average Contribution to {target_name}")
    plt.title(f"Aggregated Feature Impact on {target_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
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
    df_data = pd.DataFrame(data, columns=feature_names)
    h2o_data = h2o.H2OFrame(df_data)
    predictions = best_model.predict(h2o_data)
    return predictions.as_data_frame().values

def generate_lime_explanation(X_train, feature_names, instance, best_model, mode='regression'):
    """
    Generate a LIME explanation for a single instance.
    
    :param X_train: Training data as a NumPy array
    :param feature_names: List of predictor names
    :param instance: A single instance (1D NumPy array) from the dataset
    :param best_model: The H2O model object (e.g. aml.leader)
    :param mode: 'regression' or 'classification'
    :return: The LIME explanation object
    """
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode=mode
    )
    predict_fn = lambda data: h2o_predict_fn(data, best_model, feature_names)
    exp = explainer.explain_instance(instance, predict_fn)
    return exp

def extract_feature(condition):
    """
    Extracts the feature name from a LIME condition string.
    
    For example, from "-0.15 < Pts <= 0.67" it returns "Pts",
    and from "Ast > 0.60" it returns "Ast".
    
    This function has been updated to handle cases where the condition
    may not be in the expected format.
    """
    # Remove extraneous characters
    condition = condition.replace("(", "").replace(")", "").replace("'", "").replace('"', "")
    parts = condition.split()
    # If there are at least 3 parts and the first part is numeric, assume the feature is in position 3.
    if len(parts) >= 3:
        try:
            float(parts[0])
            return parts[2]
        except ValueError:
            return parts[0]
    else:
        return parts[0]

def summarize_lime_explanations(aggregated_explanations):
    """
    Aggregates LIME explanation contributions for each feature across multiple instances.
    
    :param aggregated_explanations: List of lists of tuples (condition, contribution) for each sample.
    :return: Dictionary mapping feature names to average contribution.
    """
    feature_contribs = {}
    for explanation in aggregated_explanations:
        for condition, contrib in explanation:
            feature = extract_feature(condition)
            if feature not in feature_contribs:
                feature_contribs[feature] = []
            feature_contribs[feature].append(contrib)
    summary = {feature: sum(contribs) / len(contribs) for feature, contribs in feature_contribs.items()}
    return summary

def generate_narrative(summary, target_name):
    """
    Converts the aggregated feature contributions into a plain language narrative.
    
    :param summary: Dictionary mapping feature names to average contributions.
    :param target_name: The name of the target variable.
    :return: String with plain language statements.
    """
    narrative_lines = []
    for feature, avg in summary.items():
        if avg > 0:
            narrative_lines.append(f"Increases in {feature} tend to increase {target_name}.")
        elif avg < 0:
            narrative_lines.append(f"Increases in {feature} tend to decrease {target_name}.")
        else:
            narrative_lines.append(f"{feature} does not have a significant impact on {target_name}.")
    return "\n".join(narrative_lines)

def plot_feature_importance(summary, target_name):
    """
    Plots a bar chart of the aggregated feature contributions.
    
    :param summary: Dictionary mapping feature names to average contributions.
    :param target_name: The name of the target variable.
    """
    features = list(summary.keys())
    avg_contribs = [summary[feat] for feat in features]
    
    plt.figure(figsize=(10, 6))
    plt.bar(features, avg_contribs, color='skyblue')
    plt.xlabel("Features")
    plt.ylabel(f"Average Contribution to {target_name}")
    plt.title(f"Aggregated Feature Impact on {target_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
