import os
import sys

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from interface.model import IModel
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


class CNN(IModel):
    def __init__(self, n_in=10):
        self.n_in = n_in
        self.n_out = None
        self.dataset = None
        self.training_dataset = None
        self.testing_dataset = None
        self.model = None
        self.scaled_feature_values = None
        self.X = None
        self.y = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def ConfigModel(
        self,
        dataset: pd.DataFrame,
        feature: str,
        start_index: int,
        end_index: int,
        prediction_steps: int,
    ):
        # Set the dataset and training dataset
        self.dataset = dataset[feature]
        self.training_dataset = self.dataset.iloc[start_index:end_index]
        self.testing_dataset = self.dataset.iloc[
            end_index : end_index + prediction_steps
        ]
        self.n_out = prediction_steps

        # Scale the feature values to be between 0 and 1
        feature_values = self.training_dataset.values.reshape(-1, 1)
        feature_values_test = self.testing_dataset.values.reshape(-1, 1)
        self.scaled_feature_values = self.scaler.fit_transform(feature_values)
        self.scaled_feature_values_test = self.scaler.fit_transform(feature_values_test)

        # Split the feature values into X and y
        Xy = self.series_to_supervised(
            data=feature_values, n_in=self.n_in, n_out=self.n_out
        )
        Xy_test = self.series_to_supervised(
            data=feature_values_test, n_in=self.n_in, n_out=self.n_out
        )
        self.X, self.y = Xy[:, : self.n_in], Xy[:, self.n_in :]
        self.X_test, self.y_test = Xy_test[:, : self.n_in], Xy_test[:, self.n_in :]

    def TrainModel(self, config):
        model = self.define_model(n_in=self.n_in, n_out=self.n_out)
        self.model = model.fit(
            x=self.X, y=self.y, epochs=1000, batch_size=32, verbose=1
        )

    def TuneModel(self, config):
        pass

    def Predict(self, config):
        prediction = self.model.predict(self.X_test)
        # y_test = self.scaler.inverse_transform(self.y_test)
        y_pred = self.scaler.inverse_transform(prediction)
        return y_pred

    def series_to_supervised(self, data, n_in, n_out):
        agg = []
        for i in range(len(data) - n_in - n_out + 1):
            agg.append(data[i : i + n_in + n_out])
        return np.array(agg)

    def define_model(self, n_in: int, n_out: int):
        model = keras.models.Sequential()
        model.add(
            keras.layers.Conv1D(
                filters=64, kernel_size=3, activation="relu", input_shape=(n_in, 1)
            )
        )
        model.add(keras.layers.MaxPooling1D(pool_size=2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(50, activation="relu"))
        model.add(
            keras.layers.Dense(n_out, activation="linear")
        )  # linear activation for regression problems
        model.compile(optimizer="adam", loss="mse")
        return model
