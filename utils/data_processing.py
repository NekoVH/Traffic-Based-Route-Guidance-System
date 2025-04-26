from typing import Sequence

import numpy as np
import pandas as pd

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

def process_data(df: pd.DataFrame):
