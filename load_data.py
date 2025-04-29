import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import DATA_PATH


def load_and_preprocess_data():
    """
    Loads the fetal health dataset, separates labels,
    checks data consistency, and scales the features.
    Returns:
        df (pd.DataFrame): Original features without labels
        fetal_labels (pd.Series): Original fetal_health labels
        X_scaled (np.ndarray): Scaled feature matrix
    """
    df = pd.read_csv(DATA_PATH)
    fetal_labels = df['fetal_health'].copy()
    df = df.drop(columns=['fetal_health'])

    all_numeric = df.map(lambda x: isinstance(x, (int, float))).all().all()
    has_text_columns = df.select_dtypes(include=['object']).empty
    has_missing_values = not df.isnull().values.any()

    assert all_numeric and has_text_columns and has_missing_values, \
        "Data must be numeric, non-null, and without object columns"

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df)
    return df, fetal_labels, X_scaled
