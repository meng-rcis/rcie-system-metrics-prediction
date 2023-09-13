import os
import sys

# Add path to the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import pconstant.models_id as models_id

BASE_HEADERS = [ 
    models_id.ARIMA,
    models_id.LSTM,
    models_id.CNN,
    models_id.PROPHET,
    models_id.ETS,
]

META_HEADERS = [
    models_id.REGRESSION_STACK,
    models_id.TREE_STACK,
    models_id.NEURAL_STACK,
]