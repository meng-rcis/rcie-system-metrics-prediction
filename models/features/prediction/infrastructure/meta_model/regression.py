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
        alpha = config.get("alpha", 1.0)
        model = Ridge(alpha=alpha)
        model.fit(self.X, self.y)
        self.model = model
        return self.model

    def Predict(self, config: dict):
        override_features = config.get("override_features", [])
        input = config.get("input", None)
        if input is None:
            raise ValueError("Input is not provided")
        if len(override_features) != 0:
            input = input[override_features]
        return self.model.predict(input)

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
        model = LRL()
        model.fit(self.X, self.y)
        self.model = model
        return self.model

    def Predict(self, config: dict):
        override_features = config.get("override_features", [])
        input = config.get("input", None)
        if input is None:
            raise ValueError("Input is not provided")
        if len(override_features) != 0:
            input = input[override_features]
        return self.model.predict(input)

    def SaveModel(self, model: Any):
        self.model = model
