import time
import pandas as pd
import tensorflow as tf

from typing import Any
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM as LSTML
from pconstant.models_id import LSTM as LSTM_ID
from interface import IBaseModel
from putils.formatter import create_sequences


class LSTM(IBaseModel):
    def __init__(self):
        self.dataset = None
        self.training_dataset = None
        self.scaled_training_dataset = None
        self.model = None
        self.feature = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def PrepareParameters(
        self,
        dataset: pd.DataFrame,
        feature: str,
        start_index: int,
        end_index: int,
    ):
        # Copy dataset to avoid changing the original dataset
        cp_dataset = dataset.copy()
        # Set up configuration
        self.feature = feature
        self.dataset = cp_dataset[feature]
        self.training_dataset = (
            self.dataset.iloc[start_index:]
            if end_index is None
            else self.dataset.iloc[start_index:end_index]
        )
        # Normalize data
        self.scaled_training_dataset = self.scaler.fit_transform(
            self.training_dataset.values.reshape(-1, 1)
        )

    def TrainModel(self, config: dict) -> Any:
        start_time = time.time()
        # Group data for LSTM
        X, y = create_sequences(
            self.scaled_training_dataset,
            config.get("n_past", 5),
            config.get("steps", 1),
        )
        # LSTM Model
        model = Sequential()
        model.add(
            LSTML(
                64,
                activation="relu",
                # dropout=0.2,
                # recurrent_dropout=0.2,
                input_shape=(X.shape[1], X.shape[2]),
                return_sequences=True,
            )
        )
        model.add(LSTML(32, activation="relu", return_sequences=False))
        model.add(Dense(y.shape[1]))
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        # Train the model
        model.fit(
            X,
            y,
            epochs=config.get("epochs", 1),
            verbose=config.get("verbose", "auto"),
            batch_size=config.get("batch_size", 32),
            validation_split=config.get("validation_split", 0.2),
            use_multiprocessing=config.get("use_multiprocessing", True),
        )
        self.model = model
        end_time = time.time()
        print(f"[LSTM] Training time: {end_time - start_time} seconds")
        return self.model

    def Predict(self, config: dict) -> pd.DataFrame:
        start_time = time.time()
        n_past = config.get("n_past", 5)
        batch_size = config.get("batch_size", 1)
        features = config.get("features", 1)
        verbose = config.get("verbose", "auto")
        frequency = config.get("frequency", "5S")
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
            start=last_datetime,
            periods=len(yhat_original[0]) + 1,
            freq=frequency,
        )[1:]
        # Convert the prediction results to a DataFrame with the calculated datetime index
        prediction_df = pd.DataFrame(
            yhat_original[0],
            columns=[LSTM_ID],
            index=prediction_datetimes,
        )
        end_time = time.time()
        print(f"[LSTM] Prediction time: {end_time - start_time} seconds")
        return prediction_df

    def SaveModel(self, model: Any):
        self.model = model
