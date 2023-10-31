import os
import sys

import pandas as pd

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from models.features.prediction.interface.meta_model import IMetaModel
from sklearn.linear_model import Ridge


class RidgeRegression(IMetaModel):
    def __init__(self):
        self.dataset = None
        self.model = None

    def ConfigModel(
        self,
        dataset: pd.DataFrame,
        features: list[str],
        target: str,
        start_index: int,
        end_index: int,
        prediction_steps: int,
    ):
        print(dataset.head())
        # print type of dataset
        print(type(dataset))

    def TrainModel(self, input):
        pass

    def TuneModel(self, input):
        pass

    def Predict(self, input):
        pass
