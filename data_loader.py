"""
Data loading and inspection utilities
Version: 2.0 - Refactored Architecture
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from logger import LoggerMixin


class DataLoader(LoggerMixin):
    """Handles data loading and initial inspection"""
    
    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls'}
    
    def load(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV or Excel file with error handling.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        try:
            if suffix == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            self.logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns from {path.name}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading file: {str(e)}")
            raise
    
    def inspect(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate comprehensive data inspection report.
        
        Args:
            df: DataFrame to inspect
            
        Returns:
            Dictionary containing inspection results
        """
        report = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Add numeric column statistics
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            report['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        self.logger.info(f"Data inspection complete: {report['shape'][0]} rows, "
                        f"{report['shape'][1]} columns, {report['duplicates']} duplicates")
        
        return report
    
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detect and categorize column types.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping type categories to column lists
        """
        numeric_cols = list(df.select_dtypes(include='number').columns)
        
        datetime_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            elif col not in numeric_cols:
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].dropna().head(100), errors='raise')
                    datetime_cols.append(col)
                except:
                    pass
        
        categorical_cols = [
            col for col in df.columns 
            if col not in numeric_cols and col not in datetime_cols
        ]
        
        types = {
            'numeric': numeric_cols,
            'datetime': datetime_cols,
            'categorical': categorical_cols
        }
        
        self.logger.info(f"Column types detected - Numeric: {len(numeric_cols)}, "
                        f"Datetime: {len(datetime_cols)}, Categorical: {len(categorical_cols)}")
        
        return types