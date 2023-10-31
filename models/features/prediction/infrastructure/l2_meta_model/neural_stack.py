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


class LinearLayerNeuralNetwork(IMetaModel):
    def __init__(self):
        pass

    def ConfigModel(
        self,
        dataset: pd.DataFrame,
        features: list[str],
        target: str,
        start_index: int,
        end_index: int,
    ):
        pass

    def TrainModel(self, config: dict):
        pass

    def TuneModel(self, config: dict):
        pass

    def Predict(self, config: dict):
        pass
