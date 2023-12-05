import numpy as np
from typing import Tuple


def create_sequences(
    input: np.ndarray, n_past: int, n_future: int, is_overlap=True
) -> Tuple:
    X, y = [], []
    skip = 1 if is_overlap else n_future
    for i in range(n_past, len(input) - n_future + 1, skip):
        X.append(input[i - n_past : i, :])
        y.append(input[i : i + n_future, 0])
    return np.array(X), np.array(y)
