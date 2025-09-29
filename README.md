# AutoDS

Automated Data Science Pipeline with H2O AutoML and LIME Explainability

## Overview

AutoDS is a production-ready machine learning pipeline that automates model training, generates interpretable explanations, and provides both command-line and web interfaces for data analysis.

## Features

- Automated end-to-end ML workflow from data loading to model deployment
- H2O AutoML for automatic model selection and hyperparameter tuning
- LIME-based model explanations for interpretability
- Dual interface: CLI tool and Streamlit web application
- Comprehensive data preprocessing and feature engineering
- Interactive visualizations and performance metrics
- Model persistence and result export capabilities

## Requirements

- Python 3.8 or higher
- Java 11+ (required by H2O)
- 4GB+ RAM recommended
- pip package manager

## Installation

### Clone Repository

```bash
git clone https://github.com/HarrisonChristophersen/Automated-DS.git
cd Automated-DS
```

### Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python main.py --help
```

## Quick Start

### Web Interface

```bash
streamlit run app.py
```

Navigate to http://localhost:8501 in your browser.

### Command Line

```bash
python main.py data.csv target_column
```

## Usage

### CLI Examples

Basic usage:
```bash
python main.py sales.csv Revenue
```

With custom parameters:
```bash
python main.py data.csv target \
  --max_models 30 \
  --max_secs 600 \
  --num_features 15 \
  --output results/
```

Full configuration:
```bash
python main.py sales.xlsx Revenue \
  --max_models 50 \
  --max_secs 1200 \
  --nfolds 10 \
  --exclude customer_id,order_id \
  --output analysis_results/
```

### CLI Options

**AutoML Configuration:**
- `--max_models N` - Maximum models to train (default: 20)
- `--max_secs N` - Maximum runtime in seconds (default: 300)
- `--nfolds N` - Cross-validation folds (default: 5)

**Explainability:**
- `--num_features N` - Features to show in explanations (default: 10)
- `--num_samples N` - Instances to explain (default: 10)

**Preprocessing:**
- `--missing_threshold F` - Drop columns with missing % above threshold (default: 0.5)
- `--numeric_strategy {median,mean,interpolate}` - Numeric imputation method
- `--no_scaling` - Disable feature scaling
- `--no_feature_engineering` - Disable automatic feature engineering

**Output:**
- `--output DIR` - Output directory (default: outputs)
- `--exclude COLS` - Comma-separated columns to exclude
- `--log_level {DEBUG,INFO,WARNING,ERROR}` - Logging verbosity

### Programmatic Usage

```python
from pipeline import MLPipeline
from config import AppConfig

# Initialize pipeline
pipeline = MLPipeline()

# Run analysis
results = pipeline.run(
    file_path='data.csv',
    target='target_column',
    exclude_cols=['id', 'timestamp']
)

# Access results
print(results['model_performance'])
print(results['feature_importance'])

# Generate report
report = pipeline.generate_report('target_column')
print(report)

# Save outputs
pipeline.save_results('output_directory')
```

## Project Structure

```
Automated-DS/
├── config.py              # Configuration management
├── logger.py              # Logging utilities
├── data_loader.py         # Data loading and inspection
├── preprocessor.py        # Data preprocessing
├── model_trainer.py       # H2O AutoML training
├── explainer.py           # LIME explainability
├── pipeline.py            # ML pipeline orchestration
├── main.py                # CLI entry point
├── app.py                 # Streamlit web interface
├── requirements.txt       # Python dependencies
├── README.md              # Documentation
└── outputs/               # Default output directory
```

## Configuration

### Custom Configuration File

Create `custom_config.py`:

```python
from config import AppConfig, AutoMLConfig, LIMEConfig, PreprocessingConfig

config = AppConfig(
    automl=AutoMLConfig(
        max_models=50,
        max_runtime_secs=1800,
        nfolds=10
    ),
    lime=LIMEConfig(
        num_features=20,
        num_samples=25
    ),
    preprocessing=PreprocessingConfig(
        missing_threshold=0.3,
        numeric_strategy='interpolate',
        scale_numeric=True,
        engineer_features=True
    ),
    log_level='DEBUG',
    output_dir='custom_results'
)
```

Use in code:
```python
from pipeline import MLPipeline
from custom_config import config

pipeline = MLPipeline(config)
results = pipeline.run('data.csv', 'target')
```

## Output Files

The pipeline generates:
- `processed_data.csv` - Cleaned and preprocessed data
- `model/` - Saved H2O model files
- `leaderboard.csv` - Model comparison metrics
- `feature_importance.png` - Visualization of key drivers
- `explanations/` - Individual LIME explanations (HTML)
- `results_summary.json` - Analysis results in JSON format

## Best Practices

### Data Preparation

- Ensure target column is clearly defined
- Remove obvious ID columns before analysis
- Check for sufficient data (minimum 100 rows recommended)
- Verify data types are correct

### Model Training

- Start with default settings for baseline
- Increase `max_models` for better accuracy
- Allow more `max_secs` for larger datasets
- Use cross-validation for reliable performance estimates

### Interpretation

- Focus on top 3-5 features for actionable insights
- Consider domain knowledge when evaluating feature importance
- Review correlation analysis for multicollinearity
- Validate model performance on held-out data

## Troubleshooting

### H2O Connection Issues

If H2O fails to start:
```bash
# Check Java installation
java -version

# Kill existing H2O instances
pkill -f h2o.jar
```

### Memory Errors

For large datasets:
- Reduce `max_models` parameter
- Disable feature engineering with `--no_feature_engineering`
- Process data in chunks if possible

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Performance Considerations

- Small datasets (< 1000 rows): Use default settings
- Medium datasets (1000-100K rows): Increase `max_runtime_secs` to 600+
- Large datasets (> 100K rows): Consider sampling or using CLI for better resource management

## Limitations

- H2O requires Java runtime environment
- Limited to tabular data (CSV/Excel formats)
- Memory usage scales with dataset size
- Browser-based storage not supported in artifacts

## Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- H2O.ai for H2O AutoML framework
- LIME for model interpretability
- Streamlit for web interface capabilities

## Contact

Harrison Christophersen - HarrisonChris43@gmail.com

Project Link: https://github.com/HarrisonChristophersen/Automated-DS
