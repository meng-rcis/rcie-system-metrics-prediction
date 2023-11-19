import abc
import pandas as pd


class IL3(abc.ABC):
    @abc.abstractmethod
    def FindModelWeights(self) -> object:
        pass

    @abc.abstractmethod
    def Predict(self, input: pd.DataFrame, weights: object) -> pd.DataFrame:
        pass
