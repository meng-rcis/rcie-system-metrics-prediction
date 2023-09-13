import pandas as pd

class Calculator:
    def __init__(self, config):
        self.config = config

    def CalculateWeight(self, data: pd.DataFrame):
        raise NotImplementedError