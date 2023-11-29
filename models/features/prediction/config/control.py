import math
import pconstant.models_id as models_id
import infrastructure.base_model as base_models
import infrastructure.meta_model as meta_models

from constant.columns import FREQUENCY

# NOTE: Define the time (units -> milliseconds) that the model will predict again (not used yet)
# STEP:
# - Update actual + raw (filtered case) data in source meta training data (from k to t points)
# - Train meta models (from 0 to t points)
# - Predict new data with PREDICTION_STEPS by meta models (from t to t + PREDICTION_STEPS points)
# - Update predicted data in source meta training data (from t to t + PREDICTION_STEPS points)
# PREDICTION_TIME = 1  # Should be time interval

"""
Main Configuration
define the default setup configuration here

SELECTED_FEATURE - the selected feature to be predicted
START_TRAINING_INDEX - the starting of training index of dataset
PREDICTION_STEPS - the number of steps to be predicted
INITIAL_BASE_TRAINING_SIZE - the number of initial base training size
INITIAL_META_TRAINING_SIZE - the number of initial meta training size
AUTO_CREATED_FINAL_RESULT_SIZE - the number of L3 size that will be automatically generated initially
ALPHA - the number on how we value RMSE of L2 models to generate the final result (in L3)
IS_FILTERED - use filter (reduce noise) or not
IS_HIDE_WARNING - define whether or not it should hide warning
IS_PARALLEL_PROCESSING - use parallel processing or not (main switch to use parallel processing or not)
IS_PARALLEL_PROCESSING_FOR_L2 - use parallel processing for L2 or not
IS_CLEANING_ROWS_REQUIRED_INITIALLY - define cleaning rows is required initially or not
IS_UPDATING_CSV_REQUIRED_INITIALLY - define updating csv is required initially or not
IS_SETUP_META_MODEL_DATASET_REQUIRED - setup meta model dataset or not
MANUALLY_MOVE_L2_L3_FILES_TO_ARCHIVE_FOLDER - force to move L2 & L3 files to archive folder or not (used to start the new main process)
BASE_MODELS_IDS - define the list of base model ids here
META_MODELS_IDS - define the list of meta model ids here
"""

CONFIG = {
    # SELECTED_FEATURE's options: "cpu_usage", "memory_usage", "bandwidth_inbound", "bandwidth_outbound", "tps", "response_time"
    "SELECTED_FEATURE": "bandwidth_inbound",
    "START_TRAINING_INDEX": 0,
    "PREDICTION_STEPS": 5,
    "INITIAL_BASE_TRAINING_SIZE": 1000,
    "INITIAL_META_TRAINING_SIZE": 1000,
    "AUTO_CREATED_FINAL_RESULT_SIZE": 250,
    "ALPHA": 500.0,
    "IS_FILTERED": True,
    "IS_HIDE_WARNING": True,
    "IS_PARALLEL_PROCESSING": True,
    "IS_PARALLEL_PROCESSING_FOR_L2": True,
    "IS_CLEANING_ROWS_REQUIRED_INITIALLY": True,
    "IS_UPDATING_CSV_REQUIRED_INITIALLY": True,
    "IS_SETUP_META_MODEL_DATASET_REQUIRED": True,
    "MANUALLY_MOVE_L2_L3_FILES_TO_ARCHIVE_FOLDER": False,
    "BASE_MODELS_IDS": [
        models_id.ARIMA,
        models_id.SARIMA,
        models_id.ETS,
        models_id.GP,
        models_id.RNN,
        models_id.LSTM,
        models_id.CNN,
        models_id.GRU,
        models_id.TCN,
    ],
    "META_MODELS_IDS": [
        models_id.REGRESSION_STACK,
        models_id.TREE_STACK,
        models_id.NEURAL_STACK,
    ],
}

"""
L1 Layer Models Configuration
define the default setup configuration of each model here

verbose - 0: silent, 1: progress bar, 2: one line per epoch
"""

L1_MODELS = {
    models_id.ARIMA: base_models.ARIMA(),
    models_id.SARIMA: base_models.SARIMA(),
    models_id.SARIMAX: base_models.SARIMAX(),
    models_id.ETS: base_models.ETS(),
    models_id.GP: base_models.GP(),
    models_id.RNN: base_models.RNN(),
    models_id.LSTM: base_models.LSTM(),
    models_id.CNN: base_models.CNN(),
    models_id.GRU: base_models.GRU(),
    models_id.TCN: base_models.TCN(),
}

SETUP_L1_CONFIG = {
    models_id.ARIMA: {
        "order": (1, 1, 1),
    },
    models_id.SARIMA: {
        "order": (1, 1, 1),
        "seasonal_order": (1, 1, 1, 12),  # 12 -> 12 * 5 seconds = 1 minute
    },
    models_id.SARIMAX: {
        "order": (1, 1, 1),
        "seasonal_order": (1, 1, 1, 12),  # 12 -> 12 * 5 seconds = 1 minute
    },
    models_id.ETS: {
        "trend": "add",
        "seasonal": "add",
        "seasonal_periods": 12,  # 12 -> 12 * 5 seconds = 1 minute
    },
    models_id.GP: {
        "length_scale": 1.0,
        "noise_level": 1.0,
    },
    models_id.RNN: {
        "n_past": 30,
        "epochs": 50,
        "batch_size": 32,
        "validation_split": 0.2,
        "verbose": 0,
    },
    models_id.LSTM: {
        "n_past": 30,
        "epochs": 50,
        "batch_size": 32,
        "validation_split": 0.2,
        "verbose": 0,
    },
    models_id.CNN: {
        "n_past": 30,
        "epochs": 50,
        "batch_size": 32,
        "validation_split": 0.2,
        "verbose": 0,
    },
    models_id.GRU: {
        "n_past": 30,
        "epochs": 50,
        "batch_size": 32,
        "validation_split": 0.2,
        "verbose": 0,
    },
    models_id.TCN: {
        "n_past": 30,
        "epochs": 50,
        "batch_size": 32,
        "validation_split": 0.2,
        "verbose": 0,
    },
}

PREDICTION_L1_CONFIG = {
    models_id.ARIMA: {},
    models_id.SARIMA: {},
    models_id.SARIMAX: {
        "exog_future": None,
    },
    models_id.ETS: {},
    models_id.GP: {},
    models_id.RNN: {
        "n_past": SETUP_L1_CONFIG.get(models_id.RNN, {}).get("n_past", 10),
        "verbose": 0,
        "batch_size": 1,
        "features": 1,
        "frequency": FREQUENCY,
    },
    models_id.LSTM: {
        "n_past": SETUP_L1_CONFIG.get(models_id.LSTM, {}).get("n_past", 10),
        "verbose": 0,
        "batch_size": 1,
        "features": 1,
        "frequency": FREQUENCY,
    },
    models_id.CNN: {
        "n_past": SETUP_L1_CONFIG.get(models_id.CNN, {}).get("n_past", 10),
        "verbose": 0,
        "batch_size": 1,
        "features": 1,
        "frequency": FREQUENCY,
    },
    models_id.GRU: {
        "n_past": SETUP_L1_CONFIG.get(models_id.GRU, {}).get("n_past", 10),
        "verbose": 0,
        "batch_size": 1,
        "features": 1,
        "frequency": FREQUENCY,
    },
    models_id.TCN: {
        "n_past": SETUP_L1_CONFIG.get(models_id.TCN, {}).get("n_past", 10),
        "verbose": 0,
        "batch_size": 1,
        "features": 1,
        "frequency": FREQUENCY,
    },
}

"""
L2 Layer Models Configuration
define the default setup configuration of each model here (for meta models)
"""

L2_MODELS = {
    models_id.REGRESSION_STACK: meta_models.LinearRegression(),
    models_id.TREE_STACK: meta_models.RandomForest(),
    models_id.NEURAL_STACK: meta_models.FeedforwardNeuralNetwork(),
}

SETUP_L2_CONFIG = {
    "LINEAR_REGRESSION": {},
    "RIDGE_REGRESSION": {
        "alpha": 1.0,
    },
    "RANDOM_FOREST": {
        "verbose": 0,
    },
    "FEEDFORWARD_NEURAL_NETWORK": {
        "epochs": 50,
        "batch_size": 32,
        "validation_split": 0.2,
        "verbose": 0,
    },
}

PREDICTION_L2_CONFIG = {
    "LINEAR_REGRESSION": {},
    "RIDGE_REGRESSION": {},
    "RANDOM_FOREST": {},
    "FEEDFORWARD_NEURAL_NETWORK": {"verbose": 0},
}

# override_features: the list of features that will be used for training - overriding the default features (BASE_MODELS_IDS)
#   - DEFAULT: [], the default features will be used
PREPARATION_L2_CONFIG = {
    "LINEAR_REGRESSION": {"override_features": []},
    "RIDGE_REGRESSION": {"override_features": []},
    "RANDOM_FOREST": {"override_features": []},
    "FEEDFORWARD_NEURAL_NETWORK": {"override_features": []},
}
