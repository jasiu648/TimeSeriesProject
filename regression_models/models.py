import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

def impute_ridge(df, alpha=1.0):

    df_imputed = df.ffill().bfill().copy()
    for col in df.columns:
        if df[col].isnull().any():
            not_null_mask = df[col].notnull()
            null_mask = df[col].isnull()
            X_train = df.loc[not_null_mask].drop(columns=[col])
            y_train = df.loc[not_null_mask, col]
            X_pred = df.loc[null_mask].drop(columns=[col])

            # Fill NaNs in features with column mean for regression
            X_train_filled = X_train.fillna(X_train.mean())
            X_pred_filled = X_pred.fillna(X_train.mean())

            if X_train.shape[1] > 0 and len(y_train) > 0 and len(X_pred) > 0:
                model = Ridge(alpha=alpha)
                model.fit(X_train_filled, y_train)
                df_imputed.loc[null_mask, col] = model.predict(X_pred_filled)
    return df_imputed