import abc
import pandas as pd


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
    def ConfigModel(self, config: dict):
        pass

    @abc.abstractmethod
    def Predict(self, config: dict) -> pd.DataFrame:
        pass
