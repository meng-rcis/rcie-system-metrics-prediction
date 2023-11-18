import os
import sys

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from constant.columns import INDEX_COL, FREQUENCY
from models.features.prediction.interface.base_model import IBaseModel
from prophet import Prophet as ProphetL
import pandas as pd
import logging


class Prophet(IBaseModel):
    def __init__(self):
        self.dataset = None
        self.training_dataset = None
        self.model = None
        self.feature = None
        self.prediction_steps = None

    def ConfigModel(
        self,
        dataset: pd.DataFrame,
        feature: str,
        start_index: int,
        end_index: int,
        prediction_steps: int,
    ):
        # Suppress cmdstanpy INFO log messages
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        # Copy dataset to avoid changing the original dataset
        cp_dataset = dataset.copy()
        cp_dataset.reset_index(inplace=True)
        # Set up configuration
        self.feature = feature
        self.dataset = cp_dataset[[INDEX_COL, feature]]
        df_prophet = self.dataset.rename(columns={INDEX_COL: "ds", self.feature: "y"})
        self.training_dataset = (
            df_prophet[start_index:]
            if end_index is None
            else df_prophet[start_index:end_index]
        )
        self.prediction_steps = prediction_steps

    def TrainModel(self, config: dict):
        self.model = ProphetL()
        self.model.fit(self.training_dataset)

    def TuneModel(self, config: dict):
        pass

    def Predict(self, config: dict):
        steps = config.get("steps", 1)
        future = self.model.make_future_dataframe(periods=steps, freq=FREQUENCY)
        forecast = self.model.predict(future)
        forecast.rename(columns={"yhat": self.feature, "ds": INDEX_COL}, inplace=True)
        forecast.set_index(INDEX_COL, inplace=True)
        prediction = forecast.iloc[-steps:][
            [self.feature]
        ]  # Only select the last prediction steps
        return prediction
