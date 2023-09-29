import os
import sys

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
import pandas as pd
from interface.model import IModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ETS(IModel):
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
        self.training_dataset = self.dataset.iloc[start_index:end_index]

    def TrainModel(self, config: dict):
        """
        Train an ETS model on a given time series.
        - series: Pandas Series object representing the time series data.
        - trend: Type of trend component.
        - seasonal: Type of seasonal component.
        - seasonal_periods: The number of periods in a complete seasonal cycle.
        """
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

    def TuneModel(self, config: dict):
        pass

    def Predict(self, config: dict) -> pd.DataFrame:
        prediction = self.model.forecast(steps=config["steps"])
        return prediction
