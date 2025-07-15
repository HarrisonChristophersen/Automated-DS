#importing.py
import pandas as pd

def load_and_inspect(file_path):
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type: must be .csv or .xls/.xlsx")
    print(f"🔍 Loaded {len(df)} rows × {len(df.columns)} columns")
    return df

def detect_column_types(df):
    numeric = list(df.select_dtypes(include='number').columns)
    datetime = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical = [c for c in df.columns if c not in numeric + datetime]
    print(f"Types → numeric: {numeric}, datetime: {datetime}, categorical: {categorical}")
    return {'numeric': numeric, 'datetime': datetime, 'categorical': categorical}

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

