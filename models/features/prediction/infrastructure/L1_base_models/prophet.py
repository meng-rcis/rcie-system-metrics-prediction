import os
import sys

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from interface.model import IModel
from prophet import Prophet as Prophet_MODEL
import pandas as pd


class Prophet(IModel):
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
        self.dataset = dataset[feature]

    def TrainModel(self, config: dict):
        self.model = Prophet_MODEL()
        self.model.fit(self.training_dataset)

    def TuneModel(self, config: dict):
        pass

    def Predict(self, config: dict):
        dataframe = self.model.make_future_dataframe(periods=config["steps"])
        prediction = self.model.predict(dataframe)
        return prediction
