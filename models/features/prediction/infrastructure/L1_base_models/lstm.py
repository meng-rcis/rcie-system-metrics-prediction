import os
import sys

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
import pandas as pd
import numpy as np
from interface.model import IModel

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


class LSTM(IModel):
    def __init__(self):
        self.dataset = None
        self.training_dataset = None
        self.model = None
        self.feature = None
        self.default_prediction_steps = 1
        self.default_seq_length = 5
        self.scaler = MinMaxScaler()

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
        # Set up configuration
        self.feature = feature
        self.dataset = cp_dataset[self.feature]
        self.training_dataset = self.dataset.iloc[start_index:end_index]

    def TrainModel(self, config: dict):
        # Normalize data
        self.training_dataset = self.scaler.fit_transform(
            self.training_dataset.values.reshape(-1, 1)
        )
        # Group data for LSTM
        steps_ahead = config.get("steps", self.default_prediction_steps)
        seq_length = config.get("seq_length", self.default_seq_length)
        X, y = self.create_sequences(self.training_dataset, seq_length, steps_ahead)
        # LSTM Model
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(50, activation="relu", input_shape=(X.shape[1], 1)))
        model.add(keras.layers.Dense(1))
        model.compile(optimizer="adam", loss="mse")
        # Reshape input for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        # Train the model
        epochs = config.get("epochs", 1)
        verbose = config.get("verbose", "auto")
        self.model = model.fit(X, y, epochs=epochs, verbose=verbose)

    def TuneModel(self, config: dict):
        pass

    def Predict(self, config: dict) -> pd.DataFrame:
        seq_length = config.get("seq_length", self.default_seq_length)
        verbose = config.get("verbose", "auto")
        # Forecast
        x_input = self.training_dataset.values[-seq_length:]  # Last sequence in data
        x_input = x_input.reshape((1, seq_length, 1))  # Reshape for LSTM
        yhat = self.model.predict(x_input, verbose=verbose)
        # Invert scaling
        yhat_original = self.scaler.inverse_transform(yhat)
        prediction = yhat_original[0, 0]
        return prediction

    # Preprocess data for LSTM
    def create_sequences(data, seq_length, steps_ahead):
        sequences, target = [], []
        for i in range(len(data) - seq_length - steps_ahead + 1):
            sequences.append(data[i : (i + seq_length)])
            target.append(data[i + seq_length + steps_ahead - 1])
        return np.array(sequences), np.array(target)
