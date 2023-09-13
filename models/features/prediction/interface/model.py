import abc

class IModel( abc.ABC ):
    @abc.abstractmethod
    def TrainModel(self, input: Any) -> Any:
        pass

    @abc.abstractmethod
    def TuneModel(self, input: Any) -> Any:
        pass 

    @abc.abstractmethod
    def Predict(self, input: Any) -> Any:
        pass


