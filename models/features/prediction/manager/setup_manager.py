import os
import sys
import math

# Add path to the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from manager.data_manager import DataManager
import pandas as pd

class SetupManager():
    def __init__(
            self,
            meta_source: str,
            data_source: pd.DataFrame,
            initial_base_training_size: int = 100,
            initial_meta_training_size: int = 10,
            prediction_step: int = 1,
        ):
        self.data_source = data_source
        self.meta_source = meta_source
        self.data_manager = DataManager(csv_file=meta_source)
        self.initial_base_training_size = initial_base_training_size
        self.initial_meta_training_size = initial_meta_training_size
        self.prediction_step = prediction_step

    def PrepareMetaModelDataset(self):
        # Check dataset has enough rows for training
        if self.isDatasetValid() == False:
            raise Exception("Dataset does not have enough rows for training. Please check the dataset.")
        
        # Loop to split dataset with given number of rows
        meta_total_rows = 0
        count = 0
        while meta_total_rows < self.initial_meta_training_size:
            # Train base models 
            print(f"[In Progress Loop - {count}] Training base models...")

            # Predict the next step using prediction_step based on the base models
            print(f"[In Progress Loop - {count}] Predicting the next step...")

            # Write the prediction result into CSV file
            print(f"[In Progress Loop - {count}] Writing the prediction result into CSV file...")

            # Increment meta_total_rows by the number of added rows

            # Print the number of rows in the meta dataset
            print("[In Progress Loop - {count}] Total number of rows in meta dataset: ", meta_total_rows)

            count += 1
        
        # Print the result
        print("[Complete] number of rows in meta dataset: ", meta_total_rows)
        print("[Complete] Meta dataset located at ", self.meta_source," is ready for training")

        return

    def isDatasetValid(self): 
        # Count number of rows in dataset
        dataset_size = self.data_source.count()

        # Count number of rows required for training
        dataset_size_required = self.initial_base_training_size + self.initial_meta_training_size

        # Return True if dataset has enough rows for training, otherwise return False
        return dataset_size >= dataset_size_required