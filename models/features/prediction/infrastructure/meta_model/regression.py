import time
import pandas as pd

from typing import Any
from sklearn.linear_model import Ridge, LinearRegression as LRL
from interface import IMetaModel


class RidgeRegression(IMetaModel):
    def __init__(self):
        self.dataset = None
        self.training_dataset = None
        self.model = None
        self.X = None
        self.y = None

    def PrepareParameters(
        self,
        dataset: pd.DataFrame,
        features: list[str],
        target: str,
        start_index: int,
        end_index: int,
        config: dict,
    ):
        override_features = config.get("override_features", [])
        cp_dataset = dataset.copy()
        self.training_dataset = (
            cp_dataset.iloc[start_index:]
            if end_index is None
            else cp_dataset.iloc[start_index:end_index]
        )
        self.X = (
            self.training_dataset[features]
            if len(override_features) == 0
            else self.training_dataset[override_features]
        )
        self.y = self.training_dataset[target]

    def TrainModel(self, config: dict) -> Any:
        """
        Train a Ridge Regression model
        - alpha: Regularization strength; must be a positive float.
        """
        start_time = time.time()
        alpha = config.get("alpha", 1.0)
        model = Ridge(alpha=alpha)
        model.fit(self.X, self.y)
        self.model = model
        end_time = time.time()
        print(f"[RidgeRegression] Training time: {end_time - start_time} seconds")
        return self.model

    def Predict(self, config: dict):
        start_time = time.time()
        override_features = config.get("override_features", [])
        input = config.get("input", None)
        if input is None:
            raise ValueError("Input is not provided")
        if len(override_features) != 0:
            input = input[override_features]
        predicted = self.model.predict(input)
        end_time = time.time()
        print(f"[RidgeRegression] Prediction time: {end_time - start_time} seconds")
        return predicted

    def SaveModel(self, model: Any):
        self.model = model


class LinearRegression(IMetaModel):
    def __init__(self):
        self.dataset = None
        self.training_dataset = None
        self.model = None
        self.X = None
        self.y = None

    def PrepareParameters(
        self,
        dataset: pd.DataFrame,
        features: list[str],
        target: str,
        start_index: int,
        end_index: int,
        config: dict,
    ):
        override_features = config.get("override_features", [])
        cp_dataset = dataset.copy()
        self.training_dataset = (
            cp_dataset.iloc[start_index:]
            if end_index is None
            else cp_dataset.iloc[start_index:end_index]
        )
        self.X = (
            self.training_dataset[features]
            if len(override_features) == 0
            else self.training_dataset[override_features]
        )
        self.y = self.training_dataset[target]

    def TrainModel(self, config: dict) -> Any:
        start_time = time.time()
        model = LRL()
        model.fit(self.X, self.y)
        self.model = model
        end_time = time.time()
        print(f"[LinearRegression] Training time: {end_time - start_time} seconds")
        return self.model

    def Predict(self, config: dict):
        start_time = time.time()
        override_features = config.get("override_features", [])
        input = config.get("input", None)
        if input is None:
            raise ValueError("Input is not provided")
        if len(override_features) != 0:
            input = input[override_features]
        predicted = self.model.predict(input)
        end_time = time.time()
        print(f"[LinearRegression] Prediction time: {end_time - start_time} seconds")
        return predicted

    def SaveModel(self, model: Any):
        self.model = model
