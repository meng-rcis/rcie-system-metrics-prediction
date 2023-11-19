import pandas as pd
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from interface import IBaseModel
from constant.columns import INDEX_COL


class GP(IBaseModel):
    def __init__(self):
        self.dataset = None
        self.training_dataset = None
        self.model = None
        self.X = None
        self.y = None

    def PrepareParameters(
        self,
        dataset: pd.DataFrame,
        feature: str,
        start_index: int,
        end_index: int,
    ):
        # Copy dataset to avoid changing the original dataset
        cp_dataset = dataset.copy()
        # Set up configuration
        self.dataset = cp_dataset[[feature]]  # Make sure it's a DataFrame
        self.training_dataset = (
            self.dataset.iloc[start_index:]
            if end_index is None
            else self.dataset.iloc[start_index:end_index]
        )
        # Convert index to 'time_num'
        self.training_dataset["time_num"] = (
            self.training_dataset.index - self.training_dataset.index[0]
        ).total_seconds()
        self.X = self.training_dataset["time_num"].values.reshape(-1, 1)
        self.y = self.training_dataset[feature].values.reshape(-1, 1)

    def ConfigModel(self, config: dict):
        length_scale = config.get("length_scale", 1.0)
        noise_level = config.get("noise_level", 1.0)
        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
        self.model = GaussianProcessRegressor(kernel=kernel)
        self.model.fit(self.X, self.y)

    def Predict(self, config: dict) -> pd.DataFrame:
        # Forecast future values
        last_time = self.training_dataset["time_num"].iloc[-1]
        steps = config.get("steps", 1)
        start, stop = last_time + 5, last_time + 5 * steps
        future_times = np.linspace(start, stop, steps).reshape(-1, 1)
        future_predictions, _future_std = self.model.predict(
            future_times, return_std=True
        )
        # Convert future numerical times back to datetime
        future_datetimes = pd.to_datetime(
            self.training_dataset.index[0]
        ) + pd.to_timedelta(future_times.flatten(), unit="s")
        # Create a DataFrame for future predictions
        prediction = pd.DataFrame(
            {
                INDEX_COL: future_datetimes,
                "cpu_usage": future_predictions.flatten(),
            }
        )
        prediction.set_index(INDEX_COL, inplace=True)
        return prediction
