import pandas as pd

from typing import Any
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAXM
from interface import IBaseModel


class SARIMAX(IBaseModel):
    def __init__(self):
        self.dataset = None
        self.training_dataset = None
        self.model = None
        self.exog = None  # Exogenous variables

    def PrepareParameters(
        self,
        dataset: pd.DataFrame,
        feature: str,
        start_index: int,
        end_index: int,
        exog=None,  # Adding exogenous data
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
        if exog is not None:
            self.exog = exog.iloc[start_index:]
            if end_index is not None:
                self.exog = self.exog.iloc[:end_index]

    def ConfigModel(self, config: dict) -> Any:
        """
        Train a SARIMAX model on a given time series.
        - series: Pandas Series object representing the time series data.
        - order: A tuple representing the (p,d,q) parameters for ARIMA.
        - seasonal_order: A tuple representing the (P,D,Q,s) seasonal parameters.
        """
        order = config.get("order", (1, 1, 1))  # Non-seasonal order
        seasonal_order = config.get("seasonal_order", (1, 1, 1, 12))  # Seasonal order
        model = SARIMAXM(
            self.training_dataset,
            exog=self.exog,
            order=order,
            seasonal_order=seasonal_order,
        )
        self.model = model.fit()
        return self.model

    def Predict(self, config: dict) -> pd.DataFrame:
        steps = config.get("steps", 1)
        exog_future = config.get(
            "exog_future", None
        )  # Exogenous variables for the forecast period
        return self.model.get_forecast(steps=steps, exog=exog_future).predicted_mean

    def SaveModel(self, model: Any):
        self.model = model
