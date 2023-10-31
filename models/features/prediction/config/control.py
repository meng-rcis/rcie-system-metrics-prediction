import os
import sys
import math

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
import pconstant.models_id as models_id

# NOTE: Define the selected feature to be predicted here
FEATURES = [
    "cpu_usage",
    "memory_usage",
    "bandwidth_inbound",
    "bandwidth_outbound",
    "tps",
    "response_time",
]
SELECTED_FEATURE = "cpu_usage"

# NOTE: Define the number of steps to be predicted here
PREDICTION_STEPS = 2

# NOTE: Define the time (units -> milliseconds) that the model will predict again (not used yet)
# STEP:
# - Update actual + raw (filtered case) data in source meta training data (from k to t points)
# - Train meta models (from 0 to t points)
# - Predict new data with PREDICTION_STEPS by meta models (from t to t + PREDICTION_STEPS points)
# - Update predicted data in source meta training data (from t to t + PREDICTION_STEPS points)
PREDICTION_TIME = 1  # Should be time interval

# NOTE: Define the number of initial base training size here
INITIAL_BASE_TRAINING_SIZE = 1000

# NOTE: Define the number of initial meta training size here
INITIAL_META_TRAINING_SIZE = 100

# NOTE: Define the number of initial training size here for meta models
BATCH_SIZE = math.ceil(INITIAL_META_TRAINING_SIZE / PREDICTION_STEPS)
INITIAL_META_TRAINING_SIZE_IN_MAIN = BATCH_SIZE * PREDICTION_STEPS
INITIAL_BASE_TRAINING_SIZE_IN_MAIN = (
    INITIAL_META_TRAINING_SIZE_IN_MAIN + INITIAL_BASE_TRAINING_SIZE
)

# NOTE: Use filter (reduce noise) or not
IS_FILTERED = True

# NOTE: Setup meta model dataset or not
IS_SETUP_META_MODEL_DATASET_REQUIRED = True

# NOTE: Define the default setup configuration (hyperparameter) of each model here
SETUP_ARIMA_CONFIG = {"order": (1, 1, 1)}
SETUP_ETS_CONFIG = {
    "trend": "add",
    "seasonal": "add",
    "seasonal_periods": 12,
}  # 12 -> 12 * 5 seconds = 1 minute
SETUP_PROPHET_CONFIG = {}
SETUP_LSTM_CONFIG = {
    "n_past": 10,
    "steps": PREDICTION_STEPS,
    "epochs": 65,
    "batch_size": 16,
    "validation_split": 0.2,
    "verbose": 0,  # 0: silent, 1: progress bar, 2: one line per epoch
}

# NOTE: Define the default prediction configuration of each model here
PREDICTION_ARIMA_CONFIG = {}
PREDICTION_ETS_CONFIG = {}
PREDICTION_PROPHET_CONFIG = {}
PREDICTION_LSTM_CONFIG = {
    "n_past": SETUP_LSTM_CONFIG.get("n_past", 10),
    "steps": SETUP_LSTM_CONFIG.get("steps", PREDICTION_STEPS),
    "verbose": 0,  # 0: silent, 1: progress bar, 2: one line per epoch
    "batch_size": 1,
    "features": 1,
}

# NOTE: Define the default setup configuration (hyperparameter) of each model here (for meta models)
SETUP_RIDGE_REGRESSION_CONFIG = {
    "alpha": 1.0,
}
SETUP_RANDOM_FOREST_CONFIG = {
    "n_estimators": 100,
    "random_state": 0,
}
SETUP_FEEDFORWARD_NEURAL_NETWORK_CONFIG = {}

# NOTE: Define the default prediction configuration of each model here (for meta models)
PREDICTION_RIDGE_REGRESSION_CONFIG = {}
PREDICTION_RANDOM_FOREST_CONFIG = {}
PREDICTION_FEEDFORWARD_NEURAL_NETWORK_CONFIG = {}

# NOTE: Define the list of base model ids here
BASE_MODELS_IDS = [models_id.ARIMA, models_id.ETS]
# BASE_MODELS_IDS = [LSTM]

# NOTE: Define the list of meta model ids here
META_MODELS_IDS = [
    models_id.REGRESSION_STACK,
    models_id.TREE_STACK,
    models_id.NEURAL_STACK,
]

# NOTE: Define the starting of training index of dataset
START_TRAINING_INDEX = 0
