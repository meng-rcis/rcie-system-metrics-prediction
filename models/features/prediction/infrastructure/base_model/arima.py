import pandas as pd

from statsmodels.tsa.arima.model import ARIMA as ARIMAM
from interface import IBaseModel


class ARIMA(IBaseModel):
    def __init__(self):
        self.dataset = None
        self.training_dataset = None
        self.model = None

    def ConfigModel(
        self,
        dataset: pd.DataFrame,
        feature: str,
        start_index: int,
        end_index: int,
        prediction_steps: int,
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

    def TrainModel(self, config: dict):
        """
        Train an ARIMA model on a given time series.
        - series: Pandas Series object representing the time series data.
        - order: A tuple representing the (p,d,q) parameters for ARIMA.
        """
        order = config.get("order", None)
        model = ARIMAM(self.training_dataset, order=order)
        self.model = model.fit()

    def TuneModel(self, config: dict):
        pass

    def Predict(self, config: dict) -> pd.DataFrame:
        steps = config.get("steps", 1)
        return self.model.forecast(steps=steps)
