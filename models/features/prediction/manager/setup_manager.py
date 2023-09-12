import os
import sys

# Add path to the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from manager.data_manager import DataManager
from constant_csv.models_header import BASE_HEADERS
import pandas as pd

class SetupManager():
    def __init__(
            self,
            meta_training_path: str,
            base_training_dataset: pd.DataFrame,
            initial_base_training_size: int = 100,
            initial_meta_training_size: int = 10,
            prediction_step: int = 1,
        ):
        self.meta_training_path = meta_training_path
        self.base_training_dataset = base_training_dataset
        self.data_manager = DataManager()
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
            self.data_manager.WriteCSV(
                filename=self.meta_training_path,
                header=BASE_HEADERS,
                rows=[[1,2,3,4,5]] # TODO: Replace with the actual prediction result
            )
            
            # Increment meta_total_rows by the number of added rows
            meta_total_rows += self.prediction_step
            count += 1

            # Print the increase result
            print(f"[In Progress Loop - {count}] number of rows in meta dataset: ", meta_total_rows)
        
        # Print the result
        print("[Complete] number of rows in meta dataset: ", meta_total_rows)
        print("[Complete] Meta dataset located at ", self.meta_training_path," is ready for training")

        return

    def isDatasetValid(self): 
        print("Checking dataset...")
        # Count number of rows in dataset
        dataset_size = self.base_training_dataset.count()[0]

        # Count number of rows required for training
        dataset_size_required = self.initial_base_training_size + self.initial_meta_training_size

        # Return True if dataset has enough rows for training, otherwise return False
        return dataset_size >= dataset_size_required