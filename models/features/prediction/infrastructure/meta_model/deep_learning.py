import pandas as pd

from typing import Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from interface import IMetaModel


class FeedforwardNeuralNetwork(IMetaModel):
    def __init__(self):
        self.dataset = None
        self.training_dataset = None
        self.model = None
        self.X = None
        self.y = None
        self.scaled_X = None
        self.scaled_y = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def PrepareParameters(
        self,
        dataset: pd.DataFrame,
        features: list[str],
        target: str,
        start_index: int,
        end_index: int,
        preparation_config: dict,
    ):
        override_features = preparation_config.get("override_features", [])
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
        self.scaled_X = self.scaler_X.fit_transform(self.X)
        self.scaled_y = self.scaler_y.fit_transform(self.y.values.reshape(-1, 1))

    def TrainModel(self, config: dict) -> Any:
        # Splitting data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            self.scaled_X,
            self.scaled_y,
            test_size=config.get("validation_split", 0.2),
            random_state=0,
        )
        # Define the model architecture
        model = Sequential(
            [
                Dense(50, activation="relu", input_shape=(X_train.shape[1],)),
                Dense(50, activation="relu"),
                Dense(1),
            ]
        )
        # Compile the model
        model.compile(optimizer="adam", loss="mse")
        # Train the model
        model.fit(
            X_train,
            y_train,
            epochs=config.get("epochs", 1),
            batch_size=config.get("batch_size", 1),
            verbose=config.get("verbose", "auto"),
            validation_data=(X_val, y_val),
        )
        self.model = model
        return self.model

    def Predict(self, config: dict):
        input = config.get("input", None)
        verbose = config.get("verbose", 0)
        if input is None:
            raise ValueError("Input is not provided")
        scaled_input = self.scaler_X.transform(input)
        yhat = self.model.predict(scaled_input, verbose=verbose)
        return self.scaler_y.inverse_transform(yhat)

    def SaveModel(self, model: Any):
        self.model = model
