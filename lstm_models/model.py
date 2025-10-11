import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from sklearn.preprocessing import MinMaxScaler

def impute_lstm(df, n_steps=5, epochs=20, batch_size=32):

    df_imputed = df.copy()
    for col in df.columns:
        series = df_imputed[col]
        
        # Fill initial and final NaNs for sequence creation
        series_ffill = series.ffill().bfill()
        scaler = MinMaxScaler()
        values = scaler.fit_transform(series_ffill.values.reshape(-1, 1)).flatten()

        # Prepare supervised learning data
        X, y, nan_indices = [], [], []
        for i in range(n_steps, len(values)):
            if np.isnan(series.values[i]):
                nan_indices.append(i)
            X.append(values[i - n_steps:i])
            y.append(values[i])

        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Mask out targets that were originally NaN
        mask = ~np.isnan(series.values[n_steps:])
        X_train, y_train = X[mask], y[mask]

        # Build and train LSTM model
        model = Sequential([
            Masking(mask_value=0., input_shape=(n_steps, 1)),
            LSTM(32, activation='tanh'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Predict missing values
        for idx in nan_indices:
            if idx >= n_steps:
                x_input = values[idx - n_steps:idx].reshape((1, n_steps, 1))
                pred = model.predict(x_input, verbose=0)[0, 0]
                # Inverse scale
                imputed_value = scaler.inverse_transform([[pred]])[0, 0]
                df_imputed.at[df_imputed.index[idx], col] = imputed_value

    return df_imputed