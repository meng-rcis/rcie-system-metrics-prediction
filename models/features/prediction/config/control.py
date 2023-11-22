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

# NOTE: Define the selected feature to be predicted here
SELECTED_FEATURE = "cpu_usage"  # "cpu_usage", "memory_usage", "bandwidth_inbound", "bandwidth_outbound", "tps", "response_time"

# NOTE: Define the starting of training index of dataset
START_TRAINING_INDEX = 0

# NOTE: Define the number of steps to be predicted here
PREDICTION_STEPS = 5

# NOTE: Define the number of initial base training size here
INITIAL_BASE_TRAINING_SIZE = 1000

# NOTE: Define the number of initial meta training size here
INITIAL_META_TRAINING_SIZE = 1000

"""
The system will automatically run the main process with the given loop range (RANGE_REQUIRED_TO_AUTO_GENERATE_FINAL_RESULT_SIZE)
Therefore, we don't require to manually run the main process one by one
Expected Result - we would automatically have the final prediction (L3 prediction) with the given data size (INITIAL_FINAL_RESULT_SIZE) saved in the L3 prediction dataset (L3_PREDICTION_DATASET_PATH)
-> define the number of initial L3 size here
"""
INITIAL_FINAL_RESULT_SIZE = 300
RANGE_REQUIRED_TO_AUTO_GENERATE_FINAL_RESULT_SIZE = (
    math.ceil(INITIAL_FINAL_RESULT_SIZE / PREDICTION_STEPS)
    if INITIAL_FINAL_RESULT_SIZE is not None
    else None
)

# NOTE: Use parallel processing or not
IS_PARALLEL_PROCESSING = True
IS_PARALLEL_PROCESSING_FOR_L2 = True and IS_PARALLEL_PROCESSING

# NOTE: Use filter (reduce noise) or not
IS_FILTERED = True

# NOTE: Setup meta model dataset or not
# When you want to create new meta files (L2 & L3) but you already have base file (L1), don't forget to update MANUALLY_MOVE_L2_L3_FILES_TO_ARCHIVE_FOLDER = True
IS_SETUP_META_MODEL_DATASET_REQUIRED = True

# NOTE: Force to move L2 & L3 files to archive folder or not
MANUALLY_MOVE_L2_L3_FILES_TO_ARCHIVE_FOLDER = False
IS_MOVE_FILE_TO_ARCHIVE_REQUIRED = (
    IS_SETUP_META_MODEL_DATASET_REQUIRED or MANUALLY_MOVE_L2_L3_FILES_TO_ARCHIVE_FOLDER
)

# NOTE: Update CSV initially or not
IS_UPDATE_CSV_REQUIRED_INITIALLY = False

# NOTE: Define cleaning rows is required initially or not
IS_CLEANING_ROWS_REQUIRED_INITIALLY = True

# NOTE: Define whether or not it should hide warning
IS_HIDE_WARNING = True

# NOTE: Define the list of base model ids here
# BASE_MODELS_IDS = [
#     models_id.ARIMA,
#     models_id.SARIMA,
#     models_id.ETS,
#     models_id.GP,
#     models_id.LSTM,
#     models_id.CNN,
#     models_id.GRU,
#     models_id.TCN,
# ]
BASE_MODELS_IDS = [models_id.RNN]

# NOTE: Define the list of meta model ids here
META_MODELS_IDS = [
    models_id.REGRESSION_STACK,
    models_id.TREE_STACK,
    models_id.NEURAL_STACK,
]

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
        "n_estimators": 300,
        "max_features": 1,
        "random_state": 0,
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

"""
L3 Layer Configuration
define configuration for L3 layer here
"""

ALPHA = 500.0
