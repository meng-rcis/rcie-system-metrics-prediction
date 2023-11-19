import time
import pandas as pd

from manager import DataManager
from controller import L1
from config.control import START_TRAINING_INDEX
from config.path import BEFORE_FILTER_FILE
from putils.printer import print_loop_message
from putils.path import generate_meta_archive_directory_path


class SetupManager:
    def __init__(
        self,
        dataset: pd.DataFrame,
        selected_feature: str,
        l1_prediction_path: str,
        base_model_ids: list[str],
        start_training_index: int = START_TRAINING_INDEX,
        initial_base_training_size: int = 100,
        initial_meta_training_size: int = 10,
        prediction_steps: int = 1,
        is_filtered: bool = False,
        is_parallel_processing: bool = False,
    ):
        self.dataset = dataset
        self.selected_feature = selected_feature
        self.l1_prediction_path = l1_prediction_path
        self.data_manager = DataManager()
        self.base_gateway = L1(
            model_ids=base_model_ids, is_parallel_processing=is_parallel_processing
        )
        self.start_training_index = start_training_index
        self.initial_base_training_size = initial_base_training_size
        self.initial_meta_training_size = initial_meta_training_size
        self.prediction_steps = prediction_steps
        self.base_model_ids = base_model_ids
        self.is_filtered = is_filtered
        self.loop_count = 1
        self.before_filter_dataset = (
            self.data_manager.LoadDataset(BEFORE_FILTER_FILE) if is_filtered else None
        )

    def PrepareMetaModelDataset(self):
        # Check dataset has enough rows for training
        if self.isDatasetValid() == False:
            error_message = "Not Enough Rows for Training Base Models"
            raise Exception(error_message)

        # Move the outdated training file to archive directory
        meta_archive_directory_path = generate_meta_archive_directory_path(layer="l1")
        self.data_manager.MoveCSV(self.l1_prediction_path, meta_archive_directory_path)

        # Loop to split dataset with given number of rows
        meta_total_rows = 0

        while meta_total_rows < self.initial_meta_training_size:
            start_time = time.time()
            print_loop_message(self.loop_count, "Setup", "Started")

            # Train base models
            print_loop_message(self.loop_count, "Setup", "Training Base Models...")
            last_training_index = (
                self.start_training_index
                + meta_total_rows
                + self.initial_base_training_size
            )
            self.base_gateway.TrainModels(
                dataset=self.dataset,
                feature=self.selected_feature,
                start_index=self.start_training_index,
                end_index=last_training_index,
                steps=self.prediction_steps,
            )

            # Predict the next step using prediction_steps based on the base models
            print_loop_message(self.loop_count, "Setup", "Predicting Result...")
            prediction_result = self.base_gateway.Predict(steps=self.prediction_steps)
            actual_result = self.dataset[self.selected_feature].iloc[
                last_training_index : last_training_index + self.prediction_steps
            ]
            before_filter_result = None
            if self.is_filtered:
                before_filter_result = self.before_filter_dataset[
                    self.selected_feature
                ].iloc[
                    last_training_index : last_training_index + self.prediction_steps
                ]

            # Extract the prediction result into CSV format
            print_loop_message(
                self.loop_count, "Setup", "Extracting Result into CSV Format..."
            )
            rows, headers = self.data_manager.ExtractSetupPredictionToCSV(
                prediction_result=prediction_result,
                actual_result=actual_result,
                model_ids=self.base_model_ids,
                before_filter_dataset=before_filter_result,
            )

            # Write the prediction result into CSV file
            print_loop_message(self.loop_count, "Setup", "Writing Result into CSV...")
            self.data_manager.WriteCSV(
                path=self.l1_prediction_path, headers=headers, rows=rows
            )

            # Increment meta_total_rows by the number of added rows
            meta_total_rows += self.prediction_steps
            print_loop_message(
                self.loop_count, "Setup", "Total Meta Rows:", meta_total_rows
            )

            # Print the time taken to complete the loop
            end_time = time.time()
            diff_time = round(end_time - start_time, 2)
            print_loop_message(
                self.loop_count,
                "Setup",
                "Finished",
                f"[Total Time: {diff_time} seconds]",
                "\n",
            )

            self.loop_count += 1

        print("[Complete] Meta Dataset Located At:", self.l1_prediction_path)

    def isDatasetValid(self):
        # Count number of rows in dataset
        dataset_size = self.dataset.count()[0]

        # Count number of rows required for training
        required = self.initial_base_training_size + self.initial_meta_training_size

        # Return True if dataset has enough rows for training, otherwise return False
        return dataset_size >= required
