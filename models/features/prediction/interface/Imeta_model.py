import abc
import pandas as pd

from typing import Any


class IMetaModel(abc.ABC):
    @abc.abstractmethod
    def PrepareParameters(
        self,
        dataset: pd.DataFrame,
        features: list[str],
        target: str,
        start_index: int,
        end_index: int,
    ):
        pass

    @abc.abstractmethod
    def TrainModel(self, config: dict) -> Any:
        pass

    @abc.abstractmethod
    def Predict(self, config: dict) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def SaveModel(self, model: Any):
        pass
