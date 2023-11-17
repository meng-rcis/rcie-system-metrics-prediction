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
from sklearn.linear_model import LinearRegression


class RidgeRegression(IMetaModel):
    def __init__(self):
        self.dataset = None
        self.training_dataset = None
        self.model = None
        self.X = None
        self.y = None

    def ConfigModel(
        self,
        dataset: pd.DataFrame,
        features: list[str],
        target: str,
        start_index: int,
        end_index: int,
    ):
        cp_dataset = dataset.copy()
        self.training_dataset = (
            cp_dataset.iloc[start_index:]
            if end_index is None
            else cp_dataset.iloc[start_index:end_index]
        )
        self.X = self.training_dataset[features]
        self.y = self.training_dataset[target]

    def TrainModel(self, config: dict):
        """
        Train a Ridge Regression model
        - alpha: Regularization strength; must be a positive float.
        """
        alpha = config.get("alpha", 1.0)
        model = Ridge(alpha=alpha)
        model.fit(self.X, self.y)
        self.model = model

    def TuneModel(self, config: dict):
        pass

    def Predict(self, config: dict):
        input = config.get("input", None)
        if input is None:
            raise ValueError("Input is not provided")
        return self.model.predict(input)


class LR(IMetaModel):
    def __init__(self):
        self.dataset = None
        self.training_dataset = None
        self.model = None
        self.X = None
        self.y = None

    def ConfigModel(
        self,
        dataset: pd.DataFrame,
        features: list[str],
        target: str,
        start_index: int,
        end_index: int,
    ):
        cp_dataset = dataset.copy()
        self.training_dataset = (
            cp_dataset.iloc[start_index:]
            if end_index is None
            else cp_dataset.iloc[start_index:end_index]
        )
        self.X = self.training_dataset[features]
        self.y = self.training_dataset[target]

    def TrainModel(self, config: dict):
        model = LinearRegression()
        model.fit(self.X, self.y)
        self.model = model

    def TuneModel(self, config: dict):
        pass

    def Predict(self, config: dict):
        input = config.get("input", None)
        if input is None:
            raise ValueError("Input is not provided")
        return self.model.predict(input)
