import os
import sys

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from models.features.prediction.interface.base_model import IBaseModel
from models.features.prediction.interface.l1 import IL1
from infrastructure.l1_base_model.arima import ARIMA
from infrastructure.l1_base_model.ets import ETS
from infrastructure.l1_base_model.lstm import LSTM
from infrastructure.l1_base_model.cnn import CNN
from infrastructure.l1_base_model.gru import GRU
from infrastructure.l1_base_model.gp import GP
import config.control as models_config
import pconstant.models_id as models_id
import pandas as pd


# NOTE: Purpose of the L1 is to let the user to define the base models and its configurations in a single place
class L1(IL1):
    def __init__(self, model_ids: list[str]):
        self.models = self.InitiateModels(model_ids)

    # NOTE: A function to prepare all models
    def InitiateModels(self, model_ids: list[str]) -> list[dict]:
        models = []
        for model_id in model_ids:
            models.append(
                {
                    "id": model_id,
                    "instance": self.getModel(model_id),
                    "setup_config": self.getSetupConfig(model_id),
                    "prediction_config": self.getPredictionConfig(model_id),
                }
            )
        return models

    # NOTE: A function to execute the training process of all models
    def TrainModels(
        self,
        dataset: pd.DataFrame,
        feature: str,
        start_index: int = 0,
        end_index: int = None,
        prediction_steps: int = 1,
    ):
        for model in self.models:
            model["instance"].ConfigModel(
                dataset=dataset,
                feature=feature,
                start_index=start_index,
                end_index=end_index,
                prediction_steps=prediction_steps,
            )
            model["instance"].TrainModel(model["setup_config"])

    # NOTE: A function to execute the prediction process of all models
    def Predict(self, steps: int) -> pd.DataFrame:
        predictions = pd.DataFrame()

        for model in self.models:
            prediction = model["instance"].Predict(
                {**model["prediction_config"], "steps": steps}
            )
            predictions[model["id"]] = prediction

        return predictions

    # NOTE: A function to get the model instance based on the model id
    def getModel(self, model_id: str) -> IBaseModel:
        if model_id == models_id.ARIMA:
            return ARIMA()
        elif model_id == models_id.ETS:
            return ETS()
        elif model_id == models_id.LSTM:
            return LSTM()
        elif model_id == models_id.CNN:
            return CNN()
        elif model_id == models_id.GRU:
            return GRU()
        elif model_id == models_id.GP:
            return GP()

        raise Exception("[getModel] Model ID not found: ", model_id)

    # NOTE: A function to get the default setup configuration of each model
    def getSetupConfig(self, model_id: str) -> dict:
        if model_id == models_id.ARIMA:
            return models_config.SETUP_ARIMA_CONFIG
        elif model_id == models_id.ETS:
            return models_config.SETUP_ETS_CONFIG
        elif model_id == models_id.LSTM:
            return models_config.SETUP_LSTM_CONFIG
        elif model_id == models_id.CNN:
            return models_config.SETUP_CNN_CONFIG
        elif model_id == models_id.GRU:
            return models_config.SETUP_GRU_CONFIG
        elif model_id == models_id.GP:
            return models_config.SETUP_GP_CONFIG

        raise Exception("[getSetupConfig] Model ID not found: ", model_id)

    # NOTE: A function to get the default prediction configuration of each model
    def getPredictionConfig(self, model_id: str) -> dict:
        if model_id == models_id.ARIMA:
            return models_config.PREDICTION_ARIMA_CONFIG
        elif model_id == models_id.ETS:
            return models_config.PREDICTION_ETS_CONFIG
        elif model_id == models_id.LSTM:
            return models_config.PREDICTION_LSTM_CONFIG
        elif model_id == models_id.CNN:
            return models_config.PREDICTION_CNN_CONFIG
        elif model_id == models_id.GRU:
            return models_config.PREDICTION_GRU_CONFIG
        elif model_id == models_id.GP:
            return models_config.PREDICTION_GP_CONFIG

        raise Exception("[getPredictionConfig] Model ID not found: ", model_id)
