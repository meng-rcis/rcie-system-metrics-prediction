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
from config.control import FEATURES
from constant.columns import FREQUENCY
from pconstant.models_id import LSTM as LSTM_ID
from interface.model import IModel

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM as Layer_LSTM


class LSTM(IModel):
    def __init__(self):
        self.dataset = None
        self.training_dataset = None
        self.scaled_training_dataset = None
        self.model = None
        self.feature = None
        self.default_values = {
            "prediction_steps": 1,
            "n_past": 5,
            "model_batch_size": 32,
            "reshape_batch_size": 1,
            "features": 1,
            "epochs": 1,
            "validation_split": 0.2,
            "verbose": "auto",
        }
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.use_all_features = False

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
        self.dataset = (
            cp_dataset[FEATURES] if self.use_all_features else cp_dataset[feature]
        )  # TO-DO: Fix cp_dataset[FEATURES]
        self.training_dataset = self.dataset.iloc[start_index:end_index]

    def TrainModel(self, config: dict):
        # Normalize data
        self.scaled_training_dataset = self.scaler.fit_transform(
            self.training_dataset.values.reshape(-1, 1)
        )
        # Group data for LSTM
        X, y = self.create_sequences(
            self.scaled_training_dataset,
            config.get("n_past", self.default_values.get("n_past")),
            config.get("steps", self.default_values.get("prediction_steps")),
        )
        # LSTM Model
        model = Sequential()
        model.add(
            Layer_LSTM(
                64,
                activation="relu",
                input_shape=(X.shape[1], X.shape[2]),
                return_sequences=True,
            )
        )
        model.add(Layer_LSTM(32, activation="relu", return_sequences=False))
        model.add(Dense(y.shape[1]))
        model.compile(optimizer="adam", loss="mse")
        self.model = model
        # Train the model
        model.fit(
            X,
            y,
            epochs=config.get("epochs", self.default_values.get("epochs")),
            verbose=config.get("verbose", self.default_values.get("verbose")),
            batch_size=config.get(
                "batch_size", self.default_values.get("model_batch_size")
            ),
            validation_split=config.get(
                "validation_split", self.default_values.get("validation_split")
            ),
        )

    def TuneModel(self, config: dict):
        pass

    def Predict(self, config: dict) -> pd.DataFrame:
        n_past = config.get("n_past", self.default_values.get("n_past"))
        batch_size = config.get(
            "batch_size", self.default_values.get("reshape_batch_size")
        )
        features = config.get("features", self.default_values.get("features"))
        verbose = config.get("verbose", self.default_values.get("verbose"))
        # Forecast
        x_input = self.scaled_training_dataset[-n_past:]  # Last sequence in data
        x_input_values = x_input.reshape(
            (batch_size, n_past, features)
        )  # Reshape for LSTM
        yhat = self.model.predict(x_input_values, verbose=verbose)
        # Invert scaling
        yhat_original = self.scaler.inverse_transform(yhat)
        # Get the last datetime from the training dataset
        last_datetime = self.training_dataset.index[-1]
        # Calculate the datetime values for the predicted results
        # Assuming your data has a frequency of 5 seconds (as per your previous example)
        prediction_datetimes = pd.date_range(
            start=last_datetime, periods=len(yhat_original[0]) + 1, freq=FREQUENCY
        )[1:]
        # Convert the prediction results to a DataFrame with the calculated datetime index
        prediction_df = pd.DataFrame(
            yhat_original[0], columns=[LSTM_ID], index=prediction_datetimes
        )
        return prediction_df

    # Preprocess data for LSTM
    def create_sequences(self, input, n_past, n_future):
        X, y = [], []
        # For each time step
        for i in range(n_past, len(input) - n_future + 1):
            X.append(input[i - n_past : i, :])
            y.append(input[i : i + n_future, 0])
        return np.array(X), np.array(y)
