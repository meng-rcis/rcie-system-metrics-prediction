import abc
import pandas as pd


class IBaseModel(abc.ABC):
    @abc.abstractmethod
    def PrepareParameters(
        self,
        dataset: pd.DataFrame,
        feature: str,
        start_index: int,
        end_index: int,
    ):
        pass

    @abc.abstractmethod
    def ConfigModel(self, config: dict):
        pass

    @abc.abstractmethod
    def Predict(self, config: dict) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def SaveModel(self):
        pass

    @abc.abstractmethod
    def LoadModel(self):
        pass
