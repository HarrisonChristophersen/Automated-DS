Automated DS

A command-line and web-based tool for automated data analysis and forecasting using H2O AutoML and LIME explainability.

Project Overview

Automated DS allows users to:

Upload a CSV or Excel dataset

Automatically preprocess data (missing values, scaling, feature engineering)

Train an H2O AutoML model on a chosen target column

Generate LIME-based explanations for model insights

View narrative summaries and feature impact charts

Interact via CLI or a Streamlit web UI

Prerequisites

Python 3.8+

Java 11 (required by H2O)

pip (Python package manager)

Installation

Clone the repo or download the project files:

git clone https://github.com/HarrisonChristophersen/Automated-DS.git
cd Automated-DS

(Optional) Create and activate a virtual environment:

python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Usage

Command-Line Interface (CLI)

Run the main script to analyze a dataset and view insights:

python main.py <path/to/file.csv|xlsx> <target_column> [--max_models 20] [--max_secs 300]

<target_column>: the column to predict (e.g., Amount, Qty).

--max_models: maximum AutoML models to train (default 20).

--max_secs: maximum AutoML runtime in seconds (default 300).

Examples:

python main.py sample_data.csv target
python main.py sales_data.xlsx Amount --max_models 10 --max_secs 600

Web Interface (Streamlit)

Launch the app:

streamlit run app.py

In your browser, open http://localhost:8501

Upload a dataset, choose target and predictors, then click Run AutoML & Explain.

Project Structure

Automated-DS/
├─ importing.py       # Load and inspect data
├─ preprocessing.py   # Missing value handling & feature engineering
├─ modeling.py        # H2O AutoML training
├─ explainability.py  # LIME explanations & plotting
├─ main.py            # CLI entry point
├─ app.py             # Streamlit web app
├─ requirements.txt   # Python dependencies
└─ sample_data.csv    # Example dataset

Contributing

Feel free to open issues or pull requests to improve the pipeline, add new features, or fix bugs.