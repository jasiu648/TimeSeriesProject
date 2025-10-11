import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

def evaluate_imputation(df, imputer_func, mask_fraction=0.1, random_state=None):

    np.random.seed(random_state)

    df = df.copy()

    # Step 1: Impute to make complete reference (no NaNs)
    df_complete = imputer_func(df)
    
    mask = df_complete.notnull()
    total_values = mask.values.sum()
    n_mask = int(total_values * mask_fraction)
    idx = np.argwhere(mask.values)
    mask_indices = idx[np.random.choice(len(idx), n_mask, replace=False)]

    # Store reference values and mask them
    reference_values = df_complete.values[mask_indices[:, 0], mask_indices[:, 1]].copy()
    df_masked = df_complete.copy()
    df_masked.values[mask_indices[:, 0], mask_indices[:, 1]] = np.nan

    df_imputed = imputer_func(df_masked)
    imputed_values = df_imputed.values[mask_indices[:, 0], mask_indices[:, 1]]

    mae = mean_absolute_error(reference_values, imputed_values)
    rmse = root_mean_squared_error(reference_values, imputed_values)
    smape_val = smape(reference_values, imputed_values)

    return {'smape': smape_val, 'mae': mae, 'rmse': rmse}