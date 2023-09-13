import os
import sys

# Add path to the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from manager.data_manager import DataManager

class MainManager():
    def __init__(self, predictionStep, isUpdateCSVRequired=False):
        self.predictionStep = predictionStep
        self.isUpdateCSVRequired = isUpdateCSVRequired 
        self.data_manager = DataManager()

    def Run(self):
        # Write latest actual data (row: previous+step) if required
        if self.isUpdateCSVRequired:
            self.UpdateCSVToLatest()

        # Train base models with the latest data in based CSV

        # Train meta models with the latest data in CSV-1

        # Calculate weight of each meta model with the data in CSV-2

        # Predict the next step using predictionStep based on the base models

        # Write the prediction result into CSV-1 file

        # Predict the next step using predictionStep based on the meta models

        # Write the prediction result into CSV-2 file

        # Find final result by weight averaging of the prediction result from meta models

        # Update the flag to indicate that the CSV is updated
        self.isUpdateCSVRequired = True

    def UpdateCSVToLatest(self):
        print("Updating CSV to latest...")
        pass
    