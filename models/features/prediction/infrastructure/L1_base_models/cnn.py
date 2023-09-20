import os
import sys

# Add path to the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from interface.model import IModel
import pandas as pd

class CNN( IModel ):
    def __init__(self):
        self.dataset = None
        self.model = None
    
    def ConfigModel(
            self, 
            dataset: pd.DataFrame, 
            feature: str, 
            start_index: int, 
            end_index: int,
            prediction_steps: int,
        ):
        self.dataset = dataset[feature]

    def TrainModel(self, config):
        pass

    def TuneModel(self, config):
        pass

    def Predict(self, config):
        prediction = self.model.predict(x=config['x'])