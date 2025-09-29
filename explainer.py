"""
Model explainability utilities using LIME
Version: 2.0 - Refactored Architecture
"""
import numpy as np
import pandas as pd
import h2o
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from config import LIMEConfig
from logger import LoggerMixin


class ModelExplainer(LoggerMixin):
    """Handles LIME-based model explanations"""
    
    def __init__(self, config: Optional[LIMEConfig] = None):
        self.config = config or LIMEConfig()
        self.explainer = None
        self.explanations = []
    
    def generate_explanations(
        self,
        df: pd.DataFrame,
        model: h2o.estimators.H2OEstimator,
        target: str,
        sample_indices: Optional[List[int]] = None
    ) -> List:
        """
        Generate LIME explanations for model predictions.
        
        Args:
            df: DataFrame with features and target
            model: Trained H2O model
            target: Target column name
            sample_indices: Specific row indices to explain (default: first N rows)
            
        Returns:
            List of LIME explanation objects
        """
        # Prepare features
        feature_cols = [col for col in df.columns if col != target]
        data = df[feature_cols].copy()
        
        # Identify categorical features
        categorical_indices = [
            i for i, col in enumerate(feature_cols)
            if not pd.api.types.is_numeric_dtype(data[col])
        ]
        
        # Encode categorical columns
        encoding_maps = {}
        for idx in categorical_indices:
            col = feature_cols[idx]
            data[col] = pd.Categorical(data[col])
            encoding_maps[col] = dict(enumerate(data[col].cat.categories))
            data[col] = data[col].cat.codes
        
        X = data.values
        
        # Define prediction function
        def predict_fn(z_array: np.ndarray) -> np.ndarray:
            """Predict function for LIME"""
            df_z = pd.DataFrame(z_array, columns=feature_cols)
            
            # Decode categorical columns
            for idx in categorical_indices:
                col = feature_cols[idx]
                df_z[col] = df_z[col].round().astype(int)
                df_z[col] = df_z[col].map(
                    lambda x: encoding_maps[col].get(x, encoding_maps[col][0]) 
                    if x in encoding_maps[col] else encoding_maps[col][0]
                )
            
            try:
                preds = model.predict(h2o.H2OFrame(df_z)).as_data_frame()
                return preds.iloc[:, 0].values
            except Exception as e:
                self.logger.error(f"Prediction error in LIME: {str(e)}")
                return np.zeros(len(df_z))
        
        # Create explainer
        self.explainer = LimeTabularExplainer(
            training_data=X,
            feature_names=feature_cols,
            class_names=[target],
            categorical_features=categorical_indices,
            verbose=False,
            mode='regression'
        )
        
        # Determine samples to explain
        if sample_indices is None:
            num_samples = min(self.config.num_samples, len(X))
            sample_indices = range(num_samples)
        
        self.logger.info(f"Generating LIME explanations for {len(sample_indices)} samples...")
        
        # Generate explanations
        self.explanations = []
        for i in sample_indices:
            try:
                exp = self.explainer.explain_instance(
                    X[i],
                    predict_fn,
                    num_features=self.config.num_features
                )
                self.explanations.append(exp)
            except Exception as e:
                self.logger.warning(f"Failed to explain instance {i}: {str(e)}")
        
        self.logger.info(f"Generated {len(self.explanations)} explanations")
        return self.explanations
    
    def summarize_explanations(self) -> Dict[str, float]:
        """
        Aggregate feature importance across all explanations.
        
        Returns:
            Dictionary mapping features to average importance
        """
        if not self.explanations:
            raise ValueError("No explanations available. Generate explanations first.")
        
        importance_sum = {}
        importance_count = {}
        
        for exp in self.explanations:
            for feature, weight in exp.as_list():
                # Clean feature name (remove value constraints like ">5.0")
                clean_feature = feature.split()[0] if ' ' in feature else feature
                
                if clean_feature not in importance_sum:
                    importance_sum[clean_feature] = 0
                    importance_count[clean_feature] = 0
                
                importance_sum[clean_feature] += abs(weight)
                importance_count[clean_feature] += 1
        
        # Calculate averages
        summary = {
            feat: importance_sum[feat] / importance_count[feat]
            for feat in importance_sum
        }
        
        # Sort by importance
        summary = dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))
        
        self.logger.info(f"Feature importance summary computed for {len(summary)} features")
        return summary
    
    def generate_narrative(self, summary: Dict[str, float], target: str) -> str:
        """
        Generate human-readable narrative from feature importance.
        
        Args:
            summary: Feature importance summary
            target: Target column name
            
        Returns:
            Narrative text
        """
        if not summary:
            return "No feature importance data available."
        
        lines = [f"### Feature Impact on '{target}'\n"]
        
        for i, (feature, importance) in enumerate(summary.items(), 1):
            impact_level = self._get_impact_level(importance, list(summary.values()))
            lines.append(f"{i}. **{feature}**: {impact_level} (importance: {importance:.4f})")
        
        return "\n".join(lines)
    
    def _get_impact_level(self, importance: float, all_importances: List[float]) -> str:
        """Categorize feature importance into levels"""
        max_importance = max(all_importances)
        relative_importance = importance / max_importance if max_importance > 0 else 0
        
        if relative_importance > 0.7:
            return "Very High Impact"
        elif relative_importance > 0.4:
            return "High Impact"
        elif relative_importance > 0.2:
            return "Moderate Impact"
        else:
            return "Low Impact"
    
    def plot_feature_importance(
        self, 
        summary: Dict[str, float], 
        target: str,
        top_n: int = 10,
        figsize: tuple = (10, 6)
    ) -> plt.Figure:
        """
        Create feature importance visualization.
        
        Args:
            summary: Feature importance summary
            target: Target column name
            top_n: Number of top features to display
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        if not summary:
            self.logger.warning("No feature importance to plot")
            return None
        
        # Get top N features
        items = list(summary.items())[:top_n]
        features, importances = zip(*items)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(features))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        
        bars = ax.barh(y_pos, importances, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Average Feature Importance', fontsize=12)
        ax.set_title(f'Top {len(features)} Features Impacting "{target}"', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{imp:.4f}',
                   ha='left', va='center', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        fig.tight_layout()
        self.logger.info("Feature importance plot created")
        return fig
    
    def save_explanations(self, output_path: str):
        """
        Save explanations to HTML files.
        
        Args:
            output_path: Directory to save explanation files
        """
        from pathlib import Path
        
        if not self.explanations:
            self.logger.warning("No explanations to save")
            return
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, exp in enumerate(self.explanations):
            file_path = output_dir / f"explanation_{i}.html"
            exp.save_to_file(str(file_path))
        
        self.logger.info(f"Saved {len(self.explanations)} explanations to {output_path}")