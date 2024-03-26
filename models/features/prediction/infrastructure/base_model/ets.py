import time
import pandas as pd

from typing import Any
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from interface import IBaseModel


class ETS(IBaseModel):
    def __init__(self):
        self.dataset = None
        self.training_dataset = None
        self.model = None

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
        self.dataset = cp_dataset[feature]
        self.training_dataset = (
            self.dataset.iloc[start_index:]
            if end_index is None
            else self.dataset.iloc[start_index:end_index]
        )

    def TrainModel(self, config: dict) -> Any:
        """
        Train an ETS model on a given time series.
        - series: Pandas Series object representing the time series data.
        - trend: Type of trend component.
        - seasonal: Type of seasonal component.
        - seasonal_periods: The number of periods in a complete seasonal cycle.
        """
        start_time = time.time()
        trend = config.get("trend", None)
        seasonal = config.get("seasonal", None)
        seasonal_periods = config.get("seasonal_periods", None)
        model = ExponentialSmoothing(
            self.training_dataset,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
        )
        self.model = model.fit()
        end_time = time.time()
        print(f"[ETS] Training time: {end_time - start_time} seconds")
        return self.model

    def Predict(self, config: dict) -> pd.DataFrame:
        start_time = time.time()
        steps = config.get("steps", 1)
        prediction = self.model.forecast(steps=steps)
        end_time = time.time()
        print(f"[ETS] Prediction time: {end_time - start_time} seconds")
        return prediction

    def SaveModel(self, model: Any):
        self.model = model
