"""
Model training utilities using H2O AutoML
Version: 2.0 - Refactored Architecture
"""
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from typing import Optional, List, Dict
from pathlib import Path
from config import AutoMLConfig
from logger import LoggerMixin


class ModelTrainer(LoggerMixin):
    """Handles H2O AutoML model training and management"""
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        self.config = config or AutoMLConfig()
        self.model = None
        self.leaderboard = None
        self.h2o_initialized = False
    
    def initialize_h2o(self):
        """Initialize H2O cluster"""
        if not self.h2o_initialized:
            try:
                h2o.init(strict_version_check=False)
                self.h2o_initialized = True
                self.logger.info("H2O cluster initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize H2O: {str(e)}")
                raise
    
    def shutdown_h2o(self):
        """Shutdown H2O cluster"""
        if self.h2o_initialized:
            try:
                h2o.cluster().shutdown()
                self.h2o_initialized = False
                self.logger.info("H2O cluster shut down")
            except:
                pass
    
    def train(
        self, 
        df: pd.DataFrame, 
        target: str,
        exclude_cols: Optional[List[str]] = None
    ) -> h2o.estimators.H2OEstimator:
        """
        Train H2O AutoML model.
        
        Args:
            df: Training DataFrame
            target: Target column name
            exclude_cols: Columns to exclude from training
            
        Returns:
            Best trained model
        """
        self.initialize_h2o()
        
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")
        
        # Convert to H2O Frame
        hf = h2o.H2OFrame(df)
        
        # Prepare feature columns
        exclude_cols = exclude_cols or []
        exclude_cols.append(target)
        feature_cols = [col for col in hf.columns if col not in exclude_cols]
        
        self.logger.info(f"Training AutoML with {len(feature_cols)} features, target: {target}")
        self.logger.info(f"Max models: {self.config.max_models}, "
                        f"Max runtime: {self.config.max_runtime_secs}s")
        
        # Configure AutoML
        aml = H2OAutoML(
            max_models=self.config.max_models,
            max_runtime_secs=self.config.max_runtime_secs,
            seed=self.config.seed,
            nfolds=self.config.nfolds,
            balance_classes=self.config.balance_classes,
            verbosity="info"
        )
        
        # Train
        try:
            aml.train(x=feature_cols, y=target, training_frame=hf)
            self.model = aml.leader
            self.leaderboard = aml.leaderboard
            
            self.logger.info(f"Training complete. Best model: {self.model.model_id}")
            self.logger.info(f"Leader model type: {self.model.algo}")
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for the best model"""
        if self.model is None:
            raise ValueError("No model has been trained yet")
        
        perf = self.model.model_performance()
        
        metrics = {
            'model_id': self.model.model_id,
            'model_type': self.model.algo
        }
        
        # Safely get metrics that exist
        try:
            if hasattr(perf, 'auc') and perf.auc() is not None:
                metrics['auc'] = perf.auc()
        except:
            pass
            
        try:
            if hasattr(perf, 'rmse') and perf.rmse() is not None:
                metrics['rmse'] = perf.rmse()
        except:
            pass
            
        try:
            if hasattr(perf, 'mae') and perf.mae() is not None:
                metrics['mae'] = perf.mae()
        except:
            pass
            
        try:
            if hasattr(perf, 'r2') and perf.r2() is not None:
                metrics['r2'] = perf.r2()
        except:
            pass
            
        try:
            if hasattr(perf, 'logloss') and perf.logloss() is not None:
                metrics['logloss'] = perf.logloss()
        except:
            pass
        
        return metrics
    
    def save_model(self, path: str) -> str:
        """
        Save trained model to disk.
        
        Args:
            path: Directory path to save model
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model has been trained yet")
        
        Path(path).mkdir(parents=True, exist_ok=True)
        model_path = h2o.save_model(model=self.model, path=path, force=True)
        self.logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path: str):
        """
        Load a saved model.
        
        Args:
            model_path: Path to saved model
        """
        self.initialize_h2o()
        self.model = h2o.load_model(model_path)
        self.logger.info(f"Model loaded from {model_path}")
        return self.model
    
    def get_leaderboard(self) -> pd.DataFrame:
        """
        Get AutoML leaderboard as pandas DataFrame.
        
        Returns:
            Leaderboard DataFrame
        """
        if self.leaderboard is None:
            raise ValueError("No leaderboard available. Train a model first.")
        
        return self.leaderboard.as_data_frame()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.shutdown_h2o()