import os
import sys

# Add path to the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from interface.model import IModel
import pandas as pd

class ARIMA( IModel ):
    def __init__(self):
        self.dataset = None
        self.model = None
    
    def AssignDataset(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def TrainModel(self, config: dict):
        def train_arima(series, order=(1,1,1)):
            """
            Train an ARIMA model on a given time series.
            Parameters:
            - series: Pandas Series object representing the time series data.
            - order: A tuple representing the (p,d,q) parameters for ARIMA.
            Returns:
            - model_fit: The trained ARIMA model.
            """
            model = ARIMA(series, order=order)
            model_fit = model.fit()
            return model_fit
        
        self.model = train_arima(self.dataset, config.order)

    def TuneModel(self, config: dict):
        pass

    def Predict(self, config: dict) -> pd.DataFrame:
        prediction = self.model.forecast(config.steps)
        return pd.DataFrame(prediction)