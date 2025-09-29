"""
Configuration management for Automated DS
Version: 2.0 - Refactored Architecture
"""
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class AutoMLConfig:
    """Configuration for H2O AutoML"""
    max_models: int = 20
    max_runtime_secs: int = 300
    seed: int = 42
    nfolds: int = 5
    balance_classes: bool = False
    
    
@dataclass
class LIMEConfig:
    """Configuration for LIME explanations"""
    num_features: int = 10
    num_samples: int = 10
    
    
@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing"""
    missing_threshold: float = 0.5
    numeric_strategy: str = 'median'  # 'median', 'mean', 'interpolate'
    categorical_strategy: str = 'mode'  # 'mode', 'constant'
    scale_numeric: bool = True
    engineer_features: bool = True
    

@dataclass
class AppConfig:
    """Main application configuration"""
    automl: AutoMLConfig = field(default_factory=AutoMLConfig)
    lime: LIMEConfig = field(default_factory=LIMEConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    log_level: str = 'INFO'
    output_dir: str = 'outputs'
    
    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
