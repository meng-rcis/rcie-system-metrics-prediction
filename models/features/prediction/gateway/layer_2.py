import os
import sys

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from models.features.prediction.interface.meta_model import IMetaModel
from infrastructure.l2_meta_model.regression_stack import RidgeRegression
from infrastructure.l2_meta_model.tree_stack import RandomForest
from infrastructure.l2_meta_model.neural_stack import FeedforwardNeuralNetwork
from config.control import (
    SETUP_RIDGE_REGRESSION_CONFIG,
    SETUP_RANDOM_FOREST_CONFIG,
    SETUP_FEEDFORWARD_NEURAL_NETWORK_CONFIG,
    PREDICTION_RIDGE_REGRESSION_CONFIG,
    PREDICTION_RANDOM_FOREST_CONFIG,
    PREDICTION_FEEDFORWARD_NEURAL_NETWORK_CONFIG,
)
import pconstant.models_id as models_id
import pandas as pd


# NOTE: Purpose of the GatewayL2 is to let the user to define the base models and its configurations in a single place
class GatewayL2:
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

    # NOTE: A function to get the model instance based on the model id
    def getModel(self, model_id: str) -> IMetaModel:
        if model_id == models_id.REGRESSION_STACK:
            return RidgeRegression()
        elif model_id == models_id.TREE_STACK:
            return RandomForest()
        elif model_id == models_id.NEURAL_STACK:
            return FeedforwardNeuralNetwork()

        raise Exception("Model ID not found: ", model_id)

    # NOTE: A function to get the default setup configuration of each model
    def getSetupConfig(self, model_id: str) -> dict:
        if model_id == models_id.REGRESSION_STACK:
            return SETUP_RIDGE_REGRESSION_CONFIG
        elif model_id == models_id.TREE_STACK:
            return SETUP_RANDOM_FOREST_CONFIG
        elif model_id == models_id.NEURAL_STACK:
            return SETUP_FEEDFORWARD_NEURAL_NETWORK_CONFIG

        raise Exception("Model ID not found: ", model_id)

    # NOTE: A function to get the default prediction configuration of each model
    def getPredictionConfig(self, model_id: str) -> dict:
        if model_id == models_id.REGRESSION_STACK:
            return PREDICTION_RIDGE_REGRESSION_CONFIG
        elif model_id == models_id.TREE_STACK:
            return PREDICTION_RANDOM_FOREST_CONFIG
        elif model_id == models_id.NEURAL_STACK:
            return PREDICTION_FEEDFORWARD_NEURAL_NETWORK_CONFIG

        raise Exception("Model ID not found: ", model_id)

    # NOTE: A function to execute the training process of all models
    def TrainModels(
        self,
        dataset: pd.DataFrame,
        features: list[str],
        target: str,
        start_index: int = 0,
        end_index: int = -1,
    ):
        for model in self.models:
            model["instance"].ConfigModel(
                dataset=dataset,
                features=features,
                target=target,
                start_index=start_index,
                end_index=end_index,
            )
            model["instance"].TrainModel(model["setup_config"])

    # NOTE: A function to execute the prediction process of all models
    def Predict(self, input: pd.DataFrame) -> pd.DataFrame:
        predictions = pd.DataFrame()

        for model in self.models:
            prediction = model["instance"].Predict(
                {**model["prediction_config"], "input": input}
            )
            predictions[model["id"]] = prediction

        return predictions
