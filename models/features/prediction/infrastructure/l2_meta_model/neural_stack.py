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
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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

    def ConfigModel(
        self,
        dataset: pd.DataFrame,
        features: list[str],
        target: str,
        start_index: int,
        end_index: int,
    ):
        cp_dataset = dataset.copy()
        self.training_dataset = cp_dataset.iloc[start_index:end_index]
        self.X = self.training_dataset[features]
        self.y = self.training_dataset[target]
        self.scaled_X = self.scaler_X.fit_transform(self.X)
        self.scaled_y = self.scaler_y.fit_transform(self.y.values.reshape(-1, 1))

    def TrainModel(self, config: dict):
        # Splitting data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            self.scaled_X,
            self.y,
            test_size=config.get("validation_split", 0.2),
            random_state=0,
        )
        # Define the model architecture
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    10, activation="relu", input_shape=(X_train.shape[1],)
                ),
                tf.keras.layers.Dense(10, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        # Compile the model
        model.compile(optimizer="adam", loss="mse")
        # Train the model
        model.fit(
            X_train,
            y_train,
            epochs=config.get("epochs", 1),
            verbose=config.get("verbose", "auto"),
            validation_data=(X_val, y_val),
        )
        self.model = model

    def TuneModel(self, config: dict):
        pass

    def Predict(self, config: dict):
        input = config.get("input", None)
        if input is None:
            raise ValueError("Input is not provided")
        scaled_input = self.scaler_X.transform(input)
        yhat = self.model.predict(scaled_input)
        return self.scaler_y.inverse_transform(yhat)
