#preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df, strategies, drop_threshold=0.5, delete_rows=False):
    cols_to_drop = [c for c in df if df[c].isna().mean() > drop_threshold]
    if cols_to_drop:
        print(f"🗑 Dropping cols (>{drop_threshold*100}% missing): {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    if delete_rows:
        df = df.dropna()
    else:
        # fill numeric
        for c in strategies['numeric']:
            if strategies.get('num_strategy', 'median') == 'median':
                df[c].fillna(df[c].median(), inplace=True)
            else:
                df[c].fillna(df[c].mean(), inplace=True)
        # fill categorical
        for c in strategies['categorical']:
            df[c].fillna('MISSING', inplace=True)
        # fill datetime
        for c in strategies['datetime']:
            df[c] = pd.to_datetime(df[c], errors='coerce').fillna(method='ffill')

    # scale numeric
    if strategies['numeric']:
        scaler = StandardScaler()
        df[strategies['numeric']] = scaler.fit_transform(df[strategies['numeric']])
        print("✅ Scaled numeric features.")
    return df

# Example Usage:
# df_cleaned = handle_missing_values(df, num_strategy="interpolate", cat_strategy="unknown", date_strategy="bfill", delete_rows=True)

def feature_engineering(df):
    for c in df.select_dtypes(include='datetime').columns:
        df[f"{c}_year"] = df[c].dt.year
        df[f"{c}_month"] = df[c].dt.month
    print("🛠 Feature engineering complete.")
    return df


# Example Usage:
# Assuming you've already loaded your DataFrame and detected column types:
# from importing import load_and_inspect
# from your_detection_module import detect_column_types
#
# df = load_and_inspect("your_file.csv")
# column_types = detect_column_types(df)
# df_processed = feature_engineering(df, column_types, scale_numeric=True)
