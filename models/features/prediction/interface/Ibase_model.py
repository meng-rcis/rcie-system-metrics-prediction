import abc
import pandas as pd


class IBaseModel(abc.ABC):
    @abc.abstractmethod
    def ConfigModel(
        self,
        dataset: pd.DataFrame,
        feature: str,
        start_index: int,
        end_index: int,
    ):
        pass

    @abc.abstractmethod
    def TrainModel(self, config: dict):
        pass

    @abc.abstractmethod
    def Predict(self, config: dict) -> pd.DataFrame:
        pass
