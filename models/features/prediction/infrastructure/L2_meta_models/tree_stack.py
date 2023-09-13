import os
import sys

# Add path to the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from interface.model import IModel

class TreeStack( IModel ):
    def __init__(self):
        pass

    def TrainModel(self, input):
        pass

    def TuneModel(self, input):
        pass

    def Predict(self, input):
        pass