import os
import sys

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D
from models.features.prediction.interface.base_model import IBaseModel
from constant.columns import FREQUENCY
from pconstant.models_id import CNN as CNN_ID

import numpy as np
import pandas as pd


class CNN(IBaseModel):
    def __init__(self):
        self.dataset = None
        self.training_dataset = None
        self.scaled_training_dataset = None
        self.model = None
        self.feature = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def ConfigModel(
        self,
        dataset: pd.DataFrame,
        feature: str,
        start_index: int,
        end_index: int,
        prediction_steps: int,
    ):
        # Copy dataset to avoid changing the original dataset
        cp_dataset = dataset.copy()
        # Set the dataset and training dataset
        self.feature = feature
        self.dataset = cp_dataset[feature]
        self.training_dataset = (
            self.dataset.iloc[start_index:]
            if end_index is None
            else self.dataset.iloc[start_index:end_index]
        )
        self.scaled_training_dataset = self.scaler.fit_transform(
            self.training_dataset.values.reshape(-1, 1)
        )

    def TrainModel(self, config: dict):
        n_past = config.get("n_past", 5)
        steps = config.get("steps", 1)
        X, y = self.create_sequences(
            self.scaled_training_dataset,
            n_past,
            steps,
        )
        # CNN Model
        model = Sequential()
        model.add(
            Conv1D(
                filters=64,
                kernel_size=2,
                activation="relu",
                input_shape=(X.shape[1], X.shape[2]),
            )
        )
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        model.add(Dense(y.shape[1]))
        model.compile(optimizer="adam", loss="mse")
        # Train the model
        model.fit(
            X,
            y,
            epochs=config.get("epochs", 1),
            verbose=config.get("verbose", "auto"),
            batch_size=config.get("batch_size", 32),
            validation_split=config.get("validation_split", 0.2),
        )
        self.model = model

    def TuneModel(self, config: dict):
        pass

    def Predict(self, config: dict) -> pd.DataFrame:
        n_past = config.get("n_past", 5)
        batch_size = config.get("batch_size", 1)
        features = config.get("features", 1)
        verbose = config.get("verbose", "auto")
        # Forecast
        x_input = self.scaled_training_dataset[-n_past:]  # Last sequence in data
        x_input_values = x_input.reshape((batch_size, n_past, features))
        yhat = self.model.predict(x_input_values, verbose=verbose)
        # Invert scaling
        yhat_original = self.scaler.inverse_transform(yhat)
        # Transform the prediction results to a DataFrame
        # Get the last datetime from the training dataset
        last_datetime = self.training_dataset.index[-1]
        # Calculate the datetime values for the predicted results
        # Assuming your data has a frequency of 5 seconds (as per your previous example)
        prediction_datetimes = pd.date_range(
            start=last_datetime, periods=len(yhat_original[0]) + 1, freq=FREQUENCY
        )[1:]
        # Convert the prediction results to a DataFrame with the calculated datetime index
        prediction_df = pd.DataFrame(
            yhat_original[0], columns=[CNN_ID], index=prediction_datetimes
        )
        return prediction_df

    def create_sequences(
        self, input: pd.DataFrame, n_past: int, n_future: int
    ) -> Tuple:
        X, y = [], []
        # For each time step
        for i in range(n_past, len(input) - n_future + 1):
            X.append(input[i - n_past : i, :])
            y.append(input[i : i + n_future, 0])
        return np.array(X), np.array(y)
