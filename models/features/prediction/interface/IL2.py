import abc
import pandas as pd

from interface import IMetaModel


class IL2(abc.ABC):
    @abc.abstractmethod
    def InitiateModels(self, model_ids: list[str]) -> list[dict]:
        pass

    @abc.abstractmethod
    def TrainModels(
        self,
        dataset: pd.DataFrame,
        features: list[str],
        target: str,
        start_index: int = 0,
        end_index: int = None,
    ):
        pass

    @abc.abstractmethod
    def Predict(self, steps: int) -> pd.DataFrame:
        pass
