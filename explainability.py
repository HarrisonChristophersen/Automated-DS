import numpy as np
import pandas as pd
import h2o
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

def generate_lime_explanation(df: pd.DataFrame,
                              model,
                              target: str,
                              num_features: int = 5,
                              num_samples: int = 10):
    """
    Generate LIME explanations for the first `num_samples` rows,
    encoding categoricals as integer codes and handling z properly.
    """
    # 1) Identify features (everything except target)
    feature_cols = [c for c in df.columns if c != target]
    data = df[feature_cols].copy()

    # 2) Determine which columns are categorical
    categorical_features = [
        i for i, col in enumerate(feature_cols)
        if not pd.api.types.is_numeric_dtype(data[col])
    ]

    # 3) Encode all categorical columns to integer codes
    for idx in categorical_features:
        col = feature_cols[idx]
        data[col] = pd.Categorical(data[col]).codes

    # 4) Convert to numpy for LIME
    X = data.values

    # 5) Define a prediction function that accepts 2D arrays
    def predict_fn(z_array: np.ndarray) -> np.ndarray:
        """
        z_array: np.ndarray of shape (n_rows, n_features).
        Returns: np.ndarray of shape (n_rows,) with the model's predictions.
        """
        # Build a DataFrame with the same columns
        df_z = pd.DataFrame(z_array, columns=feature_cols)
        # Convert to H2OFrame, predict, convert back to pandas
        preds = model.predict(h2o.H2OFrame(df_z)).as_data_frame()
        # If regression, H2O returns a column named 'predict'
        return preds.iloc[:, 0].values

    # 6) Create the Lime explainer
    explainer = LimeTabularExplainer(
        training_data=X,
        feature_names=feature_cols,
        class_names=[target],
        categorical_features=categorical_features,
        verbose=False,
    )

    # 7) Explain the first `num_samples` instances
    explanations = []
    for i in range(min(num_samples, X.shape[0])):
        exp = explainer.explain_instance(
            X[i],
            predict_fn,
            num_features=num_features
        )
        explanations.append(exp)

    return explanations


def summarize_lime_explanations(exps):
    """
    Aggregate feature weights across multiple LIME explanations.
    """
    agg = {}
    for exp in exps:
        for feat, weight in exp.as_list():
            agg[feat] = agg.get(feat, 0) + weight
    count = len(exps)
    return {feat: total / count for feat, total in agg.items()}


def generate_narrative(summary, target):
    """
    Turn the aggregated summary into a bullet-point narrative.
    """
    return "\n".join(
        f"• `{feat}` has an average impact of {imp:.3f} on `{target}`"
        for feat, imp in summary.items()
    )


def plot_feature_importance(summary, target):
    """
    Create and return a Matplotlib Figure of horizontal bar chart
    for feature impacts.
    """
    feats, imps = zip(*summary.items())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feats, imps)
    ax.set_xlabel(f"Impact on {target}")
    ax.set_title("Feature Importance (LIME)")
    fig.tight
