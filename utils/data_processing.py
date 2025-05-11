from typing import Sequence
from dataclasses import dataclass

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

def read_excel(filename: str, sheet_name: str, header: int | Sequence[int]) -> pd.DataFrame:
    return pd.read_excel(filename, sheet_name=sheet_name, header=header)

# Scaling to [0-1] for RNN algos (LSTM, GRU, ...)
def scaler(x_min: float, x_max: float):
    def _scaler(x):
        return (x - x_min) / (x_max - x_min)
    return _scaler

# Rescaling scaled values to original
def rescaler(x_min: float, x_max: float):
    def _rescaler(x):
        return x * (x_max - x_min) + x_min
    return _rescaler

def process_data(df: pd.DataFrame, lags: int = 7):
    processed_file = "datasets/processed_scats_data_october_2006.parquet"
    
    if os.path.exists(processed_file):
        # Load the processed data
        processed_data = pd.read_parquet(processed_file)
        
        # Extract the data
        X_latlong = processed_data[['lat', 'long']].values
        X_flow = processed_data[[f'flow_{i}' for i in range(lags)]].values
        y = processed_data['target'].values
        
        # Get the flow min and max for rescaling
        flow_min = processed_data['flow_min'].iloc[0]
        flow_max = processed_data['flow_max'].iloc[0]
        flow_rescaler = rescaler(flow_min, flow_max)
        
    else:
        flow_columns = [f"V{str(i).zfill(2)}" for i in range(96)]  # Creates V00 to V95
        scat_grouped = df.groupby(['NB_LATITUDE', 'NB_LONGITUDE'])[flow_columns].apply(lambda x: x.values.tolist())

        scat_latlong = np.array(scat_grouped.index.to_list())

        # We would return those scalers in case we want to inverse transform it
        lat_scaler = MinMaxScaler()
        long_scaler = MinMaxScaler()

        lats = scat_latlong[:, 0].reshape(-1, 1) # Get the lat column and reshaped it to 1D values
        longs = scat_latlong[:, 1].reshape(-1, 1) # Get the long column and reshaped it to 1D values

        lats_scaled = lat_scaler.fit_transform(lats) # Fit the scaler to the lats format and scale it to [0-1]
        longs_scaled = long_scaler.fit_transform(longs) # Fit the scaler to the longs format and scale it to [0-1]

        scat_latlong_scaled = np.hstack((lats_scaled, longs_scaled))

        scat_data = scat_grouped.values

        flow_max = np.array(scat_data.max()).max()
        flow_min = np.array(scat_data.min()).min()

        # We would return scaler and rescaler in case we want to scale and inverse scale it
        flow_scaler = scaler(flow_min, flow_max) # Get the scaler of flow data
        flow_rescaler = rescaler(flow_min, flow_max) # Get the rescaler of flow data

        X_latlong = []
        X_flow = []
        y = []

        for i, flow in enumerate(scat_grouped.values):
            flow = np.array(flow, dtype=float).flatten()
            flow = np.vectorize(flow_scaler)(flow)

            indices = np.arange(lags, len(flow))
            offset = np.arange(-lags, 0)

            for idx in indices:
                past_flow = flow[idx + offset]  # past lags
                target = flow[idx]              # current target
                X_flow.append(past_flow)
                X_latlong.append(scat_latlong_scaled[i])
                y.append(target)

        # Convert lists to numpy arrays
        X_latlong = np.array(X_latlong)
        X_flow = np.array(X_flow)
        y = np.array(y)

        # Create DataFrame with optimized data types
        processed_df = pd.DataFrame({
            'lat': X_latlong[:, 0].astype('float32'),  # Use float32 instead of float64
            'long': X_latlong[:, 1].astype('float32'),
            **{f'flow_{i}': X_flow[:, i].astype('float32') for i in range(lags)},
            'target': y.astype('float32'),
            'flow_min': flow_min,
            'flow_max': flow_max
        })

        # Save as parquet file with compression
        processed_df.to_parquet(processed_file, compression='gzip')

    # Combine into a single X tuple for train/test/validation split
    combined_X = list(zip(X_latlong, X_flow))

    # First split: 80% train, 20% test+val
    X_train_comb, X_testval_comb, y_train, y_testval = train_test_split(combined_X, y, test_size=0.95, random_state=42)

    # Second split: 50% train, 50% validation (10% test, 10% validation)
    X_test_comb, X_val_comb, y_test, y_val = train_test_split(X_testval_comb, y_testval, test_size=0.5, random_state=42)

    # Unzip the combined tuples back into separate arrays
    X_latlong_train, X_flow_train = zip(*X_train_comb)
    X_latlong_val, X_flow_val = zip(*X_val_comb)
    X_latlong_test, X_flow_test = zip(*X_test_comb)

    # Convert back to np.ndarray
    X_latlong_train = np.array(X_latlong_train)
    X_flow_train = np.array(X_flow_train)
    X_latlong_val = np.array(X_latlong_val)
    X_flow_val = np.array(X_flow_val)
    X_latlong_test = np.array(X_latlong_test)
    X_flow_test = np.array(X_flow_test)

    flow_dataset = Dataset(
        X_flow_train, y_train,
        X_flow_val, y_val,
        X_flow_test, y_test
    )

    return flow_dataset, flow_rescaler
    # return X_latlong_train, X_flow_train, y_train, X_latlong_test, X_flow_test, y_test, flow_scaler, flow_rescaler, lat_scaler, long_scaler
