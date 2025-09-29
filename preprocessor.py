"""
Data preprocessing utilities
Version: 2.0 - Refactored Architecture
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from config import PreprocessingConfig
from logger import LoggerMixin


class DataPreprocessor(LoggerMixin):
    """Handles data preprocessing operations"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.scalers = {}
        self.encoding_maps = {}
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        column_types: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Handle missing values using configured strategies.
        
        Args:
            df: Input DataFrame
            column_types: Dictionary of column types
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        # Drop columns with too many missing values
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > self.config.missing_threshold].index.tolist()
        
        if cols_to_drop:
            self.logger.warning(
                f"Dropping {len(cols_to_drop)} columns with >{self.config.missing_threshold*100}% missing: "
                f"{cols_to_drop}"
            )
            df = df.drop(columns=cols_to_drop)
            
            # Update column_types
            for col_type in column_types:
                column_types[col_type] = [
                    c for c in column_types[col_type] if c not in cols_to_drop
                ]
        
        # Handle numeric columns
        for col in column_types['numeric']:
            if col not in df.columns:
                continue
                
            if df[col].isnull().any():
                if self.config.numeric_strategy == 'median':
                    fill_value = df[col].median()
                elif self.config.numeric_strategy == 'mean':
                    fill_value = df[col].mean()
                elif self.config.numeric_strategy == 'interpolate':
                    df[col] = df[col].interpolate(method='linear')
                    continue
                else:
                    fill_value = df[col].median()
                
                df[col].fillna(fill_value, inplace=True)
                self.logger.debug(f"Filled {col} with {self.config.numeric_strategy}: {fill_value:.3f}")
        
        # Handle categorical columns
        for col in column_types['categorical']:
            if col not in df.columns:
                continue
                
            if df[col].isnull().any():
                if self.config.categorical_strategy == 'mode':
                    mode_val = df[col].mode()
                    fill_value = mode_val[0] if len(mode_val) > 0 else 'MISSING'
                else:
                    fill_value = 'MISSING'
                
                df[col].fillna(fill_value, inplace=True)
                self.logger.debug(f"Filled {col} with: {fill_value}")
        
        # Handle datetime columns
        for col in column_types['datetime']:
            if col not in df.columns:
                continue
                
            df[col] = pd.to_datetime(df[col], errors='coerce')
            if df[col].isnull().any():
                df[col] = df[col].ffill()  # Use ffill() instead of fillna(method='ffill')
        
        self.logger.info("Missing value handling complete")
        return df
    
    def scale_features(
        self, 
        df: pd.DataFrame, 
        numeric_cols: List[str],
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Scale numeric features.
        
        Args:
            df: Input DataFrame
            numeric_cols: List of numeric column names
            method: Scaling method ('standard' or 'robust')
            
        Returns:
            DataFrame with scaled features
        """
        if not numeric_cols or not self.config.scale_numeric:
            return df
        
        df = df.copy()
        
        # Filter to only columns present in df
        cols_to_scale = [col for col in numeric_cols if col in df.columns]
        
        if not cols_to_scale:
            return df
        
        if method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        self.scalers['numeric'] = scaler
        
        self.logger.info(f"Scaled {len(cols_to_scale)} numeric features using {method} scaler")
        return df
    
    def engineer_features(
        self, 
        df: pd.DataFrame, 
        datetime_cols: List[str]
    ) -> pd.DataFrame:
        """
        Create engineered features from existing columns.
        
        Args:
            df: Input DataFrame
            datetime_cols: List of datetime column names
            
        Returns:
            DataFrame with engineered features
        """
        if not self.config.engineer_features:
            return df
        
        df = df.copy()
        new_features = []
        
        # Extract datetime features
        for col in datetime_cols:
            if col not in df.columns:
                continue
                
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_quarter'] = df[col].dt.quarter
            
            new_features.extend([
                f'{col}_year', f'{col}_month', f'{col}_day', 
                f'{col}_dayofweek', f'{col}_quarter'
            ])
        
        if new_features:
            self.logger.info(f"Created {len(new_features)} datetime features")
        
        return df
    
    def preprocess(
        self, 
        df: pd.DataFrame, 
        column_types: Dict[str, List[str]],
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Execute full preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            column_types: Dictionary of column types
            target_col: Target column name (won't be scaled if specified)
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("Starting preprocessing pipeline...")
        
        # Handle missing values
        df = self.handle_missing_values(df, column_types)
        
        # Engineer features
        df = self.engineer_features(df, column_types['datetime'])
        
        # Scale numeric features (exclude target)
        numeric_to_scale = [
            col for col in column_types['numeric'] 
            if col != target_col and col in df.columns
        ]
        df = self.scale_features(df, numeric_to_scale)
        
        self.logger.info("Preprocessing pipeline complete")
        return df
