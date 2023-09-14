import os
import sys

# Add path to the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from interface.model import IModel
from infrastructure.L1_base_models.arima import ARIMA
from infrastructure.L1_base_models.ets import ETS
from infrastructure.L1_base_models.prophet import Prophet
from config.model import SETUP_ARIMA_CONFIG, SETUP_ETS_CONFIG, SETUP_PROPHET_CONFIG, PREDICTION_ARIMA_CONFIG, PREDICTION_ETS_CONFIG, PREDICTION_PROPHET_CONFIG
import pconstant.models_id as models_id
import pandas as pd

# NOTE: Purpose of the GatewayL1 is to let the user to define the base models and its configurations in a single place
class GatewayL1:
    def __init__(self, model_ids: list[str]):
        self.models = self.prepareModels(model_ids)

    # NOTE: A function to prepare all models
    def prepareModels(self, model_ids: list[str]) -> list[dict]:
        models = []
        for model_id in model_ids:
            models.append({ 
                'id': model_id,
                'instance': self.getModel(model_id),
                'setup_config': self.getSetupConfig(model_id),
                'prediction_config': self.getPredictionConfig(model_id),
            })
        return models
    
    # NOTE: A function to get the model instance based on the model id
    def getModel(self, model_id: str) -> IModel:
        if model_id == models_id.ARIMA:
            return ARIMA()
        elif model_id == models_id.ETS:
            return ETS()
        elif model_id == models_id.PROPHET:
            return Prophet()
        
        raise Exception('Model ID not found: ', model_id)
    
    # NOTE: A function to get the default setup configuration of each model
    def getSetupConfig(self, model_id: str) -> dict:
        if model_id == models_id.ARIMA:
            return SETUP_ARIMA_CONFIG
        elif model_id == models_id.ETS:
            return SETUP_ETS_CONFIG
        elif model_id == models_id.PROPHET:
            return SETUP_PROPHET_CONFIG

        raise Exception('Model ID not found: ', model_id)
    
    # NOTE: A function to get the default prediction configuration of each model
    def getPredictionConfig(self, model_id: str) -> dict:
        if model_id == models_id.ARIMA:
            return PREDICTION_ARIMA_CONFIG
        elif model_id == models_id.ETS:
            return PREDICTION_ETS_CONFIG
        elif model_id == models_id.PROPHET:
            return PREDICTION_PROPHET_CONFIG
            
        raise Exception('Model ID not found: ', model_id)

    # NOTE: A function to execute the training process of all models
    def TrainModels(self, dataset: pd.DataFrame):
        for model in self.models:
            model['instance'].AssignDataset(dataset)
            model['instance'].TrainModel(model['setup_config'])
    
    # NOTE: A function to execute the prediction process of all models
    def Predict(self, step: int) -> pd.DataFrame:
        predictions = pd.DataFrame()

        for model in self.models:
            config = model['prediction_config']
            predictions[model['id']] = model['instance'].Predict({ **config, 'step': step })
        
        return predictions