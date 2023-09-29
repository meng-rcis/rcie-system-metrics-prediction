import os
import sys

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from constant.columns import INDEX_COL
from interface.model import IModel
from prophet import Prophet as Prophet_MODEL
import pandas as pd


class Prophet(IModel):
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
        dataset.reset_index(inplace=True)
        self.feature = feature
        self.dataset = dataset[[INDEX_COL, feature]]
        df_prophet = self.dataset.rename(columns={INDEX_COL: "ds", feature: "y"})
        self.training_dataset = df_prophet[start_index:end_index]

    def TrainModel(self, config: dict):
        self.model = Prophet_MODEL()
        self.model.fit(self.training_dataset)

    def TuneModel(self, config: dict):
        pass

    def Predict(self, config: dict):
        future = self.model.make_future_dataframe(periods=config["steps"])
        forecast = self.model.predict(future)
        forecast.rename(columns={"yhat": self.feature, "ds": INDEX_COL}, inplace=True)
        forecast.set_index(INDEX_COL, inplace=True)
        # Only select the last 'config["steps"]' rows for prediction
        prediction = forecast.iloc[-config["steps"] :][[self.feature]]
        print("prophet prediction: ", prediction)
        return prediction
