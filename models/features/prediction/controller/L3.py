from models.features.prediction.interface.L3 import IL3

from sklearn.metrics import mean_squared_error
from math import exp
from manager.data_manager import DataManager
from pconstant.feature_header import TIME
import pandas as pd


class L3(IL3):
    def __init__(
        self,
        meta_model_ids: list[str],
        meta_prediction_source: str,
        target_col: str,
        alpha: float = 1.0,
    ):
        self.model_ids = meta_model_ids
        self.meta_prediction_source = meta_prediction_source
        self.target_col = target_col
        self.alpha = alpha
        self.data_manager = DataManager()

    def FindModelWeights(self) -> object:
        # Define all models with equal weight if the file is not found
        if self.data_manager.IsFileExist(self.meta_prediction_source) == False:
            return {item: 1 / len(self.model_ids) for item in self.model_ids}

        # Otherwise, calculate the weight
        meta_prediction = self.data_manager.ReadCSV(
            self.meta_prediction_source, index_col=TIME
        )

        # Calculate RMSE for each model
        rmse = {
            model: mean_squared_error(
                meta_prediction[self.target_col], meta_prediction[model], squared=False
            )
            for model in self.model_ids
        }

        # Calculate the exponent weights
        exp_weights = {
            model: exp(-self.alpha * rmse[model]) for model in self.model_ids
        }

        # Normalize the weights so they sum to 1
        total_weight = sum(exp_weights.values())
        normalized_weights = {
            model: weight / total_weight for model, weight in exp_weights.items()
        }

        return normalized_weights

    def Predict(self, input: pd.DataFrame, weights: object) -> pd.DataFrame:
        # Calculate weighted predictions
        weighted_preds = sum(input[model] * weight for model, weight in weights.items())

        # Convert to a DataFrame
        result_df = pd.DataFrame({"Predicted": weighted_preds})
        return result_df
