import abc
import pandas as pd

class IModel( abc.ABC ):
    @abc.abstractmethod
    def AssignDataset(self, dataset: pd.DataFrame):
        pass

    @abc.abstractmethod
    def TrainModel(self, input: any) -> any:
        pass

    @abc.abstractmethod
    def TuneModel(self, input: any) -> any:
        pass 

    @abc.abstractmethod
    def Predict(self, input: any) -> pd.DataFrame:
        pass


