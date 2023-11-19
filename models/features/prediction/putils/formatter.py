import numpy as np
import pandas as pd
from typing import Tuple


def create_sequences(input: pd.DataFrame, n_past: int, n_future: int) -> Tuple:
    X, y = [], []
    # For each time step
    for i in range(n_past, len(input) - n_future + 1):
        X.append(input[i - n_past : i, :])
        y.append(input[i : i + n_future, 0])
    return np.array(X), np.array(y)
