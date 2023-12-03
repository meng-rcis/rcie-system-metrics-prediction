import pandas as pd

from typing import Any
from interface import IMetaModel
from sklearn.ensemble import RandomForestRegressor


class RandomForest(IMetaModel):
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
        Train a Random Forest model
        - n_estimators: The number of trees in the forest.
        - random_state: Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True), and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features).
        """
        n_estimators = config.get("n_estimators", 100)
        max_features = config.get("max_features", 1)
        random_state = config.get("random_state", 0)
        verbose = config.get("verbose", 0)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose,
        )
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
