import abc
import pandas as pd

class IModel( abc.ABC ):
    @abc.abstractmethod
    def AssignDataset(self, dataset: pd.DataFrame):
        pass

    @abc.abstractmethod
    def TrainModel(self, config: dict):
        pass

    @abc.abstractmethod
    def TuneModel(self, config: dict):
        pass 

    @abc.abstractmethod
    def Predict(self, config: dict) -> pd.DataFrame:
        pass


