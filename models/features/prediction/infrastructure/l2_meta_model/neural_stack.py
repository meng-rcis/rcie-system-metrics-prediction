import os
import sys

import pandas as pd

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from interface.model import IModel


class LinearLayerNeuralNetwork(IModel):
    def __init__(self):
        pass

    def ConfigModel(
        self,
        dataset: pd.DataFrame,
        feature: str,
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
