import os
import sys
import pandas as pd

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from pconstant.models_id import GRU as GRU_ID
from models.features.prediction.interface.base_model import IBaseModel
from models.features.prediction.putils.formatter import create_sequences

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import GRU as GRUL, Dense
from keras.optimizers import Adam


class GRU(IBaseModel):
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

    def TrainModel(self, config: dict):
        # Group data for GRU
        X, y = create_sequences(
            self.scaled_training_dataset,
            config.get("n_past", 5),
            config.get("steps", 1),
        )
        # GRU Model
        model = Sequential()
        model.add(
            GRUL(
                64,
                activation="relu",
                input_shape=(X.shape[1], X.shape[2]),
                return_sequences=True,
            )
        )
        model.add(GRUL(32, activation="relu", return_sequences=False))
        model.add(Dense(y.shape[1]))
        model.compile(optimizer=Adam(), loss="mse")

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
        frequency = config.get("frequency", "5S")
        # Forecast
        x_input = self.scaled_training_dataset[-n_past:]  # Last sequence in data
        x_input_values = x_input.reshape(
            (batch_size, n_past, features)
        )  # Reshape for GRU
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
            columns=[GRU_ID],
            index=prediction_datetimes,
        )
        return prediction_df