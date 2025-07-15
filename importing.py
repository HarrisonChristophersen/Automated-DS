#importing.py
import pandas as pd

def load_and_inspect(file_path):
    """
    Loads a CSV or Excel file and displays basic data insights.
    
    :param file_path: Path to the dataset file (.csv or .xlsx)
    :return: Pandas DataFrame
    """
    # Detect file type
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel.")

    # Display basic info
    print("\n🔍 Dataset Overview:")
    print(df.info())

    # Show first few rows
    print("\n📊 First 5 Rows:")
    print(df.head())

    # Check for missing values
    print("\n⚠️ Missing Values:")
    print(df.isnull().sum())

    # Summary statistics
    print("\n📈 Summary Statistics:")
    print(df.describe(include='all'))

    return df

def detect_column_types(df):
    """
    Detects numeric, categorical, and datetime columns in the dataset.

    :param df: Pandas DataFrame
    :return: Dictionary with categorized column names
    """
    column_types = {"numeric": [], "categorical": [], "datetime": []}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types["numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or is_datetime_column(df[col]):
            column_types["datetime"].append(col)
        else:
            column_types["categorical"].append(col)

    print("\n🔍 **Column Type Detection:**")
    print("🔢 Numeric Columns:", column_types["numeric"])
    print("🔠 Categorical Columns:", column_types["categorical"])
    print("📅 Datetime Columns:", column_types["datetime"])

    return column_types

def is_datetime_column(series):
    """Attempts to convert a column to datetime and checks if successful."""
    try:
        pd.to_datetime(series, errors="coerce")
        return True
    except:
        return False

# Example Usage:
# df = load_and_inspect("your_file.csv")
# column_types = detect_column_types(df)

