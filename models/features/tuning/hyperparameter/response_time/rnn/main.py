import os
import sys
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler

dir_path = os.path.dirname(os.path.realpath(__file__))
models_path = os.path.join(dir_path, "../../../../../../")
prediction_path = os.path.join(dir_path, "../../../../../features/prediction/")
sys.path.append(models_path)
sys.path.append(prediction_path)

from models.features.prediction.putils.formatter import create_sequences
from models.features.prediction.config.control import CONFIG
from models.features.prediction.config.path import BASE_DATASET_PATH, BEFORE_FILTER_FILE
from models.features.prediction.manager import DataManager

FEATURE = "response_time"
VERSION = "tune"
SAVE_PATH = f"./models/features/tuning/hyperparameter/{FEATURE}/rnn/source/rnn_{VERSION}_result.csv"
# Best score=0.0061
# Best parameters:
#               - n_past=21
#               - epochs=139
#               - batch_size=30
#               - learning_rate=0.038230
#               - neurons_l1=63
#               - neurons_l2=59
#               - activation_function_l1=sigmoid
#               - activation_function_l2=sigmoid
MODEL_CONFIG = {
    "n_past": 21,
    "epochs": 139,
    "batch_size": 30,
    "learning_rate": 0.038230,
    "neurons_l1": 63,
    "neurons_l2": 59,
    "activation_function_l1": "sigmoid",
    "activation_function_l2": "sigmoid",
}
# MODEL_CONFIG = {
#     "n_past": 30,
#     "epochs": 50,
#     "batch_size": 32,
#     "learning_rate": 0.001,
#     "neurons_l1": 50,
#     "neurons_l2": 50,
#     "activation_function_l1": "relu",
#     "activation_function_l2": "relu",
# }


def define_model(X, y, config):
    # RNN Model
    model = Sequential()
    model.add(
        SimpleRNN(
            config.get("neurons_l1", 50),
            activation=config.get("activation_function_l1", "relu"),
            input_shape=(X.shape[1], X.shape[2]),
            return_sequences=True,
        )
    )
    model.add(
        SimpleRNN(
            config.get("neurons_l2", 50),
            activation=config.get("activation_function_l2", "relu"),
        )
    )
    model.add(Dense(y.shape[1]))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.get("learning_rate", 0.001)
        ),
        loss="mse",
    )

    # Train the model
    model.fit(
        X,
        y,
        epochs=config.get("epochs", 1),
        verbose=config.get("verbose", 0),
        batch_size=config.get("batch_size", 32),
        validation_split=config.get("validation_split", 0.2),
    )
    return model


def main():
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = DataManager.LoadDataset(BASE_DATASET_PATH)
    raw_dataset = DataManager.LoadDataset(BEFORE_FILTER_FILE)
    dataset_cp = dataset.copy()[FEATURE]
    raw_dataset_cp = raw_dataset.copy()[FEATURE]
    time_indices = dataset_cp.index

    n_past = MODEL_CONFIG["n_past"]
    starting_training_index = CONFIG["START_TRAINING_INDEX"]
    steps = CONFIG["PREDICTION_STEPS"]
    base_training_size = CONFIG["INITIAL_BASE_TRAINING_SIZE"]
    redefine_interval = CONFIG["T_REDEFINE_MODEL_INTERVAL"]
    auto_created_base_result_size = CONFIG["T_AUTO_CREATED_BASE_RESULT_SIZE"]
    redefine_interval_loop_required = auto_created_base_result_size // redefine_interval
    cumulative_training_size = base_training_size

    # Initialize an empty DataFrame to store all results
    all_results_df = pd.DataFrame()

    for i in range(redefine_interval_loop_required):
        # Prepare data
        start_time = time.time()
        print(f"Redefine Model: {i + 1}")
        next_prediction = cumulative_training_size + redefine_interval
        end_index = min(next_prediction, len(dataset_cp))
        training_dataset, testing_dataset = (
            dataset_cp[starting_training_index:cumulative_training_size],
            dataset_cp[cumulative_training_size - n_past : end_index],
        )
        scaled_training_dataset = scaler.fit_transform(
            training_dataset.values.reshape(-1, 1)
        )
        scaled_testing_dataset = scaler.transform(testing_dataset.values.reshape(-1, 1))
        X_train, y_train = create_sequences(scaled_training_dataset, n_past, steps)
        X_test, y_test = create_sequences(scaled_testing_dataset, n_past, steps, False)

        # Define model
        model = define_model(X_train, y_train, MODEL_CONFIG)

        # Predict
        print(f"Predict Model: {i + 1}")
        yhat = model.predict(X_test)
        yhat_original = scaler.inverse_transform(yhat)
        y_test = scaler.inverse_transform(y_test)

        # Prepare result
        result_df = pd.DataFrame(
            {
                "Time": time_indices[cumulative_training_size:end_index],
                "RNN": yhat_original.flatten(),
                "Actual": y_test.flatten(),
                "Raw": raw_dataset_cp[cumulative_training_size:end_index],
            }
        )
        all_results_df = pd.concat([all_results_df, result_df], ignore_index=True)
        cumulative_training_size += redefine_interval
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Cumulative Size: {cumulative_training_size} [{total_time} Seconds]\n")

    print("Save Results")
    all_results_df.to_csv(SAVE_PATH, index=False)
    print("Done")


if __name__ == "__main__":
    main()
