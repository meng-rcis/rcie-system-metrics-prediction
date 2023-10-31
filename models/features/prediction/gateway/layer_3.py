from manager.data_manager import DataManager
import pandas as pd


class GatewayL3:
    def __init__(self, meta_model_ids: list[str], meta_prediction_source: str):
        self.model_ids = meta_model_ids
        self.meta_prediction_source = meta_prediction_source
        self.data_manager = DataManager()

    def FindModelWeights(self) -> object:
        # Define all models with equal weight if the file is not found
        if self.data_manager.IsFileExist(self.meta_prediction_source) == False:
            return {item: 1 for item in self.model_ids}

        # Otherwise, calculate the weight
        meta_prediction = self.data_manager.ReadCSV(self.meta_prediction_source)
        weights = {}  # TODO: Calculate the weight
        return weights

    def Predict(self, input: pd.DataFrame, weights: object) -> pd.DataFrame:
        # Calculate weighted predictions
        weighted_preds = sum(input[model] * weight for model, weight in weights.items())

        # Convert to a DataFrame
        result_df = pd.DataFrame({"Predicted": weighted_preds})
        return result_df
