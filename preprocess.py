import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocess(df: pd.DataFrame, method: str = 'minmax') -> (np.ndarray, object):
    if not df.select_dtypes(include=['object']).empty:
        raise ValueError("Non-numeric columns present in DataFrame")
    if df.isnull().values.any():
        raise ValueError("Missing values detected in DataFrame")
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    X_scaled = scaler.fit_transform(df)
    return X_scaled, scaler
