import os
import sys

# Add path to the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from manager.data_manager import DataManager
from utils.predictor import Predictor
from utils.calculator import Calculator

class MainManager():
    def __init__(self, prediction_step, is_update_csv_required=False):
        self.prediction_step = prediction_step
        self.is_update_csv_required = is_update_csv_required
        self.base_models = []
        self.meta_models = []
        self.data_manager = DataManager()
        self.predictor = Predictor()
        self.calculator = Calculator()

    def Run(self):
        # Write latest actual data (row: previous+step) if required
        if self.is_update_csv_required:
            self.UpdateCSVToLatest()

        # Train base models with the latest data in based CSV
        self.TrainBaseModels()

        # Train meta models with the latest data in CSV-1
        self.TrainMetaModels()

        # Calculate weight of each meta model with the data in CSV-2
        weights = self.CalculateWeight()

        # Predict the next step using prediction_step based on the base models
        base_results = self.PredictBaseModels()

        # Predict the next step using prediction_step based on the meta models
        meta_results = self.PredictMetaModels(base_results)

        # Find final result by weight averaging of the prediction result from meta models
        self.PredictFinalResultWithWeightAverage(meta_results, weights)

        # Update the flag to indicate that the CSV is updated
        self.is_update_csv_required = True

    def UpdateCSVToLatest(self):
        print("Updating CSV to latest...")
        pass

    def CalculateWeight(self):
        print("Calculating weight...")
        meta_prediction = self.data_manager.ReadCSV('')
        weights = self.calculator.CalculateWeight(meta_prediction)
        return weights
    
    def TrainBaseModels(self):
        print("Training base models...")
        pass

    def TrainMetaModels(self):
        print("Training meta models...")
        pass
    
    def PredictBaseModels(self):
        print("Predicting base models...")

        NO_OF_BASE_MODELS = 5
        for i in range(NO_OF_BASE_MODELS):
            # Predict the next step using prediction_step based on the base models (parallel?)
            self.predictor.Predict()

            # Write the prediction result into CSV-1 file
            self.data_manager.WriteCSV('', [], [[]])
    
    def PredictMetaModels(self, base_results):
        print("Predicting meta models...")

        NO_OF_META_MODELS = 3
        for i in range(NO_OF_META_MODELS):
            # Predict the next step using prediction_step based on the meta models
            self.predictor.Predict()

            # Write the prediction result into CSV-2 file
            self.data_manager.WriteCSV('', [], [[]])

    def PredictFinalResultWithWeightAverage(self, meta_results, weights):
        print("Predicting final result...")
        pass