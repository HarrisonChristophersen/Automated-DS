#preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df, num_strategy="median", cat_strategy="mode", date_strategy="ffill", drop_threshold=0.5, delete_rows=False):
    """
    Handles missing values in a DataFrame based on user-selected strategies.

    :param df: Pandas DataFrame
    :param num_strategy: Strategy for numeric columns ('mean', 'median', 'interpolate', 'none')
    :param cat_strategy: Strategy for categorical columns ('mode', 'unknown', 'none')
    :param date_strategy: Strategy for datetime columns ('ffill', 'bfill', 'none')
    :param drop_threshold: Max % of missing values allowed before dropping a column.
    :param delete_rows: If True, delete rows with any missing values instead of filling.
    :return: DataFrame with missing values handled.
    """
    missing_info = df.isnull().sum() / len(df)

    print("\n⚠️ Missing Data Report:")
    print(missing_info[missing_info > 0])  # Show only columns with missing values

    if delete_rows:
        df = df.dropna()
        print("\n🗑 Deleted all rows with missing values!")
        return df

    # Drop columns with too much missing data
    cols_to_drop = missing_info[missing_info > drop_threshold].index
    df = df.drop(columns=cols_to_drop)
    print(f"\n🗑 Dropped Columns (>{drop_threshold*100}% missing): {list(cols_to_drop)}")

    # Handle missing values based on user-selected strategies
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):  # Numeric columns
            if num_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif num_strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif num_strategy == "interpolate":
                df[col] = df[col].interpolate()
        
        elif pd.api.types.is_object_dtype(df[col]):  # Categorical columns
            if cat_strategy == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])
            elif cat_strategy == "unknown":
                df[col] = df[col].fillna("Unknown")

        elif pd.api.types.is_datetime64_any_dtype(df[col]):  # Datetime columns
            if date_strategy == "ffill":
                df[col] = df[col].fillna(method='ffill')
            elif date_strategy == "bfill":
                df[col] = df[col].fillna(method='bfill')

    print("\n✅ Missing values handled based on selected strategies!")
    return df

# Example Usage:
# df_cleaned = handle_missing_values(df, num_strategy="interpolate", cat_strategy="unknown", date_strategy="bfill", delete_rows=True)

def feature_engineering(df, column_types, scale_numeric=False):
    """
    Performs feature engineering on the DataFrame based on detected column types.
    
    :param df: Pandas DataFrame
    :param column_types: Dictionary with keys 'numeric', 'categorical', 'datetime'
    :param scale_numeric: Boolean indicating whether to scale numeric columns
    :return: Processed DataFrame ready for modeling
    """
    # Process categorical variables: One-hot encoding
    if column_types.get("categorical"):
        # Only include columns that still exist in df
        valid_categoricals = [col for col in column_types["categorical"] if col in df.columns]
        if valid_categoricals:
            df = pd.get_dummies(df, columns=valid_categoricals, drop_first=True)
            print("✅ Categorical features encoded using one-hot encoding.")

    # Process datetime variables: Extract common features
    if column_types.get("datetime"):
        valid_datetime = [col for col in column_types["datetime"] if col in df.columns]
        for dt_col in valid_datetime:
            df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
            df[f"{dt_col}_year"] = df[dt_col].dt.year
            df[f"{dt_col}_month"] = df[dt_col].dt.month
            df[f"{dt_col}_day"] = df[dt_col].dt.day
            df[f"{dt_col}_dayofweek"] = df[dt_col].dt.dayofweek
            # Optionally, drop the original datetime column
            df.drop(columns=[dt_col], inplace=True)
        print("✅ Datetime features extracted (year, month, day, day of week).")

    # Optionally, scale numeric features
    if scale_numeric and column_types.get("numeric"):
        valid_numeric = [col for col in column_types["numeric"] if col in df.columns]
        if valid_numeric:
            scaler = StandardScaler()
            df[valid_numeric] = scaler.fit_transform(df[valid_numeric])
            print("✅ Numeric features scaled using StandardScaler.")
    
    return df


# Example Usage:
# Assuming you've already loaded your DataFrame and detected column types:
# from importing import load_and_inspect
# from your_detection_module import detect_column_types
#
# df = load_and_inspect("your_file.csv")
# column_types = detect_column_types(df)
# df_processed = feature_engineering(df, column_types, scale_numeric=True)
