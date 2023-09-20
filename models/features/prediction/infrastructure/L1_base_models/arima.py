import os
import sys

# Add path to the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from statsmodels.tsa.arima.model import ARIMA as ARIMA_MODEL
from interface.model import IModel
import pandas as pd

class ARIMA( IModel ):
    def __init__(self):
        self.dataset = None
        self.training_dataset = None
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
        self.training_dataset = self.dataset.iloc[start_index:end_index]

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
            model = ARIMA_MODEL(series, order=order)
            model_fit = model.fit()
            return model_fit
        
        self.model = train_arima(
            series=self.training_dataset,
            order=config['order']
        )

    def TuneModel(self, config: dict):
        pass

    def Predict(self, config: dict) -> pd.DataFrame:
        return self.model.forecast(steps=config['steps'])
