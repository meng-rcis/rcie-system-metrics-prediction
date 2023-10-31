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


class RidgeRegression(IMetaModel):
    def __init__(self):
        pass

    def ConfigModel(
        self,
        dataset: pd.DataFrame,
        features: str,
        start_index: int,
        end_index: int,
        prediction_steps: int,
    ):
        pass

    def TrainModel(self, input):
        pass

    def TuneModel(self, input):
        pass

    def Predict(self, input):
        pass
