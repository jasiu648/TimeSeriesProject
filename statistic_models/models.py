import pandas as pd
from sklearn.impute import KNNImputer

def impute_mean(data):

    if isinstance(data, pd.DataFrame):
        return data.fillna(data.mean())
    else:
        raise ValueError("Input must be a pandas DataFrame or Series.")

def impute_median(data):

    if isinstance(data, pd.DataFrame):
        return data.fillna(data.median())
    else:
        raise ValueError("Input must be a pandas DataFrame or Series.")

def impute_mode(data):

    if isinstance(data, pd.DataFrame):
        return data.fillna(data.mode().iloc[0])
    else:
        raise ValueError("Input must be a pandas DataFrame or Series.")

def impute_ffill(data):

    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.ffill().bfill()
    else:
        raise ValueError("Input must be a pandas DataFrame or Series.")

def impute_bfill(data):

    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.bfill().ffill()
    else:
        raise ValueError("Input must be a pandas DataFrame or Series.")

def impute_knn(data, n_neighbors=5):

    if isinstance(data, pd.Series):
        df = data.to_frame()
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed = imputer.fit_transform(df)
        return pd.Series(imputed.flatten(), index=data.index, name=data.name)
    elif isinstance(data, pd.DataFrame):
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed = imputer.fit_transform(data)
        return pd.DataFrame(imputed, columns=data.columns, index=data.index)
    else:
        raise ValueError("Input must be a pandas DataFrame or Series.")