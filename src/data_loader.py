import os, pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean', 'unified_daily.csv')

def load_unified() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    for c in df.select_dtypes('object').columns:
        df[c] = df[c].astype('category')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df
