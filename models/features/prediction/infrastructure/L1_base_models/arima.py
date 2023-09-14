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

    def TrainModel(self, input: any):
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
        
        return train_arima(self.dataset, input.order)

    def TuneModel(self, input):
        pass

    def Predict(self, input) -> pd.DataFrame:
        pass