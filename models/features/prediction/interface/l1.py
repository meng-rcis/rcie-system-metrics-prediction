import abc
import pandas as pd

from models.features.prediction.interface import IBaseModel


class IL1(abc.ABC):
    @abc.abstractmethod
    def InitiateModels(self, model_ids: list[str]) -> list[dict]:
        pass

    @abc.abstractmethod
    def TrainModels(
        self,
        dataset: pd.DataFrame,
        feature: str,
        start_index: int = 0,
        end_index: int = None,
        prediction_steps: int = 1,
    ):
        pass

    @abc.abstractmethod
    def Predict(self, steps: int) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def getPredictionConfig(self, model_id: str) -> dict:
        pass

    @abc.abstractmethod
    def getSetupConfig(self, model_id: str) -> dict:
        pass

    @abc.abstractmethod
    def getModel(self, model_id: str) -> IBaseModel:
        pass
