import pandas as pd

from typing import Any
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAXM
from interface import IBaseModel


class SARIMA(IBaseModel):
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

    def ConfigModel(self, config: dict) -> Any:
        """
        Train a SARIMA model on a given time series.
        - series: Pandas Series object representing the time series data.
        - order: A tuple representing the (p,d,q) parameters for ARIMA.
        - seasonal_order: A tuple representing the (P,D,Q,s) seasonal parameters.
        """
        order = config.get("order", (1, 1, 1))  # Non-seasonal order
        seasonal_order = config.get("seasonal_order", (1, 1, 1, 12))  # Seasonal order
        model = SARIMAXM(
            self.training_dataset,
            order=order,
            seasonal_order=seasonal_order,
        )
        self.model = model.fit()
        return self.model

    def Predict(self, config: dict) -> pd.DataFrame:
        steps = config.get("steps", 1)
        return self.model.forecast(steps=steps)

    def SaveModel(self, model: Any):
        self.model = model