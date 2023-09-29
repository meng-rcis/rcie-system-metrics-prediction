import os
import sys

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from manager.data_manager import DataManager
from gateway.layer_1 import GatewayL1
import pandas as pd


class SetupManager:
    def __init__(
        self,
        dataset: pd.DataFrame,
        selected_feature: str,
        meta_training_path: str,
        base_model_ids: list[str],
        initial_base_training_size: int = 100,
        initial_meta_training_size: int = 10,
        prediction_steps: int = 1,
    ):
        self.dataset = dataset
        self.selected_feature = selected_feature
        self.meta_training_path = meta_training_path
        self.data_manager = DataManager()
        self.base_gateway = GatewayL1(base_model_ids)
        self.initial_base_training_size = initial_base_training_size
        self.initial_meta_training_size = initial_meta_training_size
        self.prediction_steps = prediction_steps
        self.base_model_ids = base_model_ids

    def PrepareMetaModelDataset(self):
        # Check dataset has enough rows for training
        if self.isDatasetValid() == False:
            raise Exception(
                "Dataset does not have enough rows for training. Please check the dataset."
            )

        # Loop to split dataset with given number of rows
        meta_total_rows = 0
        count = 0
        while meta_total_rows < self.initial_meta_training_size:
            # Train base models
            print(f"[In Progress Loop - {count}] Training base models...")
            first_training_index = meta_total_rows
            last_training_index = meta_total_rows + self.initial_base_training_size
            self.base_gateway.TrainModels(
                dataset=self.dataset,
                feature=self.selected_feature,
                start_index=first_training_index,
                end_index=last_training_index,
                prediction_steps=self.prediction_steps,
            )

            # Predict the next step using prediction_steps based on the base models
            print(f"[In Progress Loop - {count}] Predicting the next step...")
            prediction_result = self.base_gateway.Predict(steps=self.prediction_steps)
            actual_result = self.dataset[self.selected_feature].iloc[
                last_training_index : last_training_index + self.prediction_steps
            ]

            # Extract the prediction result into CSV format
            print(
                f"[In Progress Loop - {count}] Extracting the prediction result into CSV format..."
            )
            rows, header = self.data_manager.ExtractPredictionToCSV(
                prediction_result,
                actual_result,
                self.base_model_ids,
            )

            # Write the prediction result into CSV file
            print(
                f"[In Progress Loop - {count}] Writing the prediction result into CSV file..."
            )
            self.data_manager.WriteCSV(
                path=self.meta_training_path, header=header, rows=rows
            )

            # Increment meta_total_rows by the number of added rows
            meta_total_rows += self.prediction_steps
            count += 1

            # Print the increase result
            print(
                f"[In Progress Loop - {count}] number of rows in meta dataset: ",
                meta_total_rows,
            )

        # Print the result
        print("[Complete] number of rows in meta dataset: ", meta_total_rows)
        print(
            "[Complete] Meta dataset located at",
            self.meta_training_path,
            "is ready for training",
        )

    def isDatasetValid(self):
        print("Checking dataset...")
        # Count number of rows in dataset
        dataset_size = self.dataset.count()[0]

        # Count number of rows required for training
        dataset_size_required = (
            self.initial_base_training_size + self.initial_meta_training_size
        )

        # Return True if dataset has enough rows for training, otherwise return False
        return dataset_size >= dataset_size_required
