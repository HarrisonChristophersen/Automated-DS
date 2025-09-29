"""
Complete ML pipeline orchestrating all components
Version: 2.0 - Refactored Architecture
"""
import pandas as pd
from typing import Optional, Dict, List
from pathlib import Path
import json

from config import AppConfig
from logger import setup_logger
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from explainer import ModelExplainer


class MLPipeline:
    """End-to-end machine learning pipeline"""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.logger = setup_logger('MLPipeline', self.config.log_level)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor(self.config.preprocessing)
        self.trainer = ModelTrainer(self.config.automl)
        self.explainer = ModelExplainer(self.config.lime)
        
        # Pipeline state
        self.raw_data = None
        self.processed_data = None
        self.column_types = None
        self.model = None
        self.explanations = None
        self.results = {}
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file"""
        self.logger.info(f"Loading data from {file_path}")
        self.raw_data = self.data_loader.load(file_path)
        
        # Inspect data
        inspection = self.data_loader.inspect(self.raw_data)
        self.results['data_inspection'] = inspection
        
        # Detect column types
        self.column_types = self.data_loader.detect_column_types(self.raw_data)
        
        return self.raw_data
    
    def preprocess_data(self, target: str) -> pd.DataFrame:
        """Preprocess data"""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.logger.info("Preprocessing data")
        self.processed_data = self.preprocessor.preprocess(
            self.raw_data, 
            self.column_types,
            target
        )
        
        return self.processed_data
    
    def train_model(
        self, 
        target: str, 
        exclude_cols: Optional[List[str]] = None
    ):
        """Train AutoML model"""
        if self.processed_data is None:
            raise ValueError("No processed data. Call preprocess_data() first.")
        
        self.logger.info("Training model")
        self.model = self.trainer.train(
            self.processed_data, 
            target, 
            exclude_cols
        )
        
        # Get model performance
        performance = self.trainer.get_model_performance()
        self.results['model_performance'] = performance
        
        # Get leaderboard
        self.results['leaderboard'] = self.trainer.get_leaderboard()
        
        return self.model
    
    def explain_model(self, target: str):
        """Generate model explanations"""
        if self.model is None:
            raise ValueError("No model trained. Call train_model() first.")
        
        if self.processed_data is None:
            raise ValueError("No processed data available.")
        
        self.logger.info("Generating explanations")
        
        # Generate LIME explanations
        self.explanations = self.explainer.generate_explanations(
            self.processed_data,
            self.model,
            target
        )
        
        # Summarize explanations
        summary = self.explainer.summarize_explanations()
        self.results['feature_importance'] = summary
        
        # Generate narrative
        narrative = self.explainer.generate_narrative(summary, target)
        self.results['narrative'] = narrative
        
        return summary
    
    def run(
        self, 
        file_path: str, 
        target: str,
        exclude_cols: Optional[List[str]] = None
    ) -> Dict:
        """
        Run complete pipeline.
        
        Args:
            file_path: Path to data file
            target: Target column name
            exclude_cols: Columns to exclude from training
            
        Returns:
            Dictionary containing all results
        """
        try:
            # Execute pipeline steps
            self.load_data(file_path)
            self.preprocess_data(target)
            self.train_model(target, exclude_cols)
            self.explain_model(target)
            
            self.logger.info("Pipeline completed successfully")
            self.results['status'] = 'success'
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.results['status'] = 'failed'
            self.results['error'] = str(e)
            raise
        
        finally:
            # Cleanup
            if hasattr(self, 'trainer'):
                self.trainer.shutdown_h2o()
        
        return self.results
    
    def save_results(self, output_dir: str):
        """Save all results to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        if self.processed_data is not None:
            data_path = output_path / "processed_data.csv"
            self.processed_data.to_csv(data_path, index=False)
            self.logger.info(f"Saved processed data to {data_path}")
        
        # Save model
        if self.model is not None:
            model_path = self.trainer.save_model(str(output_path / "model"))
            self.results['model_path'] = model_path
        
        # Save leaderboard
        if 'leaderboard' in self.results:
            lb_path = output_path / "leaderboard.csv"
            self.results['leaderboard'].to_csv(lb_path, index=False)
            self.logger.info(f"Saved leaderboard to {lb_path}")
        
        # Save feature importance plot
        if 'feature_importance' in self.results:
            target = self.results.get('target', 'target')
            fig = self.explainer.plot_feature_importance(
                self.results['feature_importance'], 
                target
            )
            if fig:
                plot_path = output_path / "feature_importance.png"
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved feature importance plot to {plot_path}")
        
        # Save explanations as HTML
        if self.explanations:
            self.explainer.save_explanations(str(output_path / "explanations"))
        
        # Save results summary as JSON
        json_results = {
            k: v for k, v in self.results.items() 
            if k not in ['leaderboard', 'data_inspection']
        }
        # Convert non-serializable objects
        if 'model_performance' in json_results:
            json_results['model_performance'] = {
                k: float(v) if isinstance(v, (int, float)) else str(v)
                for k, v in json_results['model_performance'].items()
            }
        
        summary_path = output_path / "results_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        self.logger.info(f"Saved results summary to {summary_path}")
        
        self.logger.info(f"All results saved to {output_dir}")
    
    def generate_report(self, target: str) -> str:
        """Generate comprehensive text report"""
        lines = ["=" * 60]
        lines.append("AUTOMATED ML PIPELINE REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Data summary
        if 'data_inspection' in self.results:
            insp = self.results['data_inspection']
            lines.append("DATA SUMMARY")
            lines.append("-" * 60)
            lines.append(f"Rows: {insp['shape'][0]:,}")
            lines.append(f"Columns: {insp['shape'][1]}")
            lines.append(f"Duplicates: {insp['duplicates']}")
            lines.append(f"Memory Usage: {insp['memory_usage_mb']:.2f} MB")
            lines.append("")
        
        # Model performance
        if 'model_performance' in self.results:
            perf = self.results['model_performance']
            lines.append("MODEL PERFORMANCE")
            lines.append("-" * 60)
            lines.append(f"Model ID: {perf.get('model_id', 'N/A')}")
            lines.append(f"Model Type: {perf.get('model_type', 'N/A')}")
            for metric, value in perf.items():
                if metric not in ['model_id', 'model_type']:
                    lines.append(f"{metric.upper()}: {value:.4f}")
            lines.append("")
        
        # Feature importance
        if 'narrative' in self.results:
            lines.append("FEATURE IMPORTANCE")
            lines.append("-" * 60)
            lines.append(self.results['narrative'])
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)