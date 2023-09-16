import os
import sys

# Add path to the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from pconstant.models_id import ARIMA

# NOTE: Define the default setup configuration (hyperparameter) of each model here
SETUP_ARIMA_CONFIG = { 'order': (5, 0, 0) }
SETUP_ETS_CONFIG = {}
SETUP_PROPHET_CONFIG = {}

# NOTE: Define the default prediction configuration of each model here
PREDICTION_ARIMA_CONFIG = {}
PREDICTION_ETS_CONFIG = {}
PREDICTION_PROPHET_CONFIG = {}

# NOTE: Define the list of base model ids here
BASE_MODELS_IDS = [ARIMA]

# NOTE: Define the list of meta model ids here
META_MODELS_IDS = []

# NOTE: Define the selected feature to be predicted here
SELECTED_FEATURE = 'tps'