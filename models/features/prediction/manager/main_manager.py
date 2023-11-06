import os
import sys
import pandas as pd

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from config.control import START_TRAINING_INDEX
from config.path import BEFORE_FILTER_FILE
from manager.data_manager import DataManager
from gateway.layer_1 import GatewayL1
from gateway.layer_2 import GatewayL2
from gateway.layer_3 import GatewayL3
from putils.printer import print_loop_message
from putils.path import generate_meta_archive_directory_path
from pconstant.feature_header import ACTUAL, RAW, TIME


class MainManager:
    def __init__(
        self,
        dataset: pd.DataFrame,
        selected_feature: str,
        l1_prediction_path: str,
        l2_prediction_path: str,
        l3_prediction_path: str,
        base_model_ids: list[str],
        meta_model_ids: list[str],
        initial_base_training_size: int,
        start_training_index: int = START_TRAINING_INDEX,
        prediction_steps: int = 1,
        is_filtered: bool = False,
        is_update_csv_required_initially: bool = False,
        is_move_to_archive_required: bool = False,
    ):
        self.dataset = dataset
        self.selected_feature = selected_feature
        self.l1_prediction_path = l1_prediction_path
        self.l2_prediction_path = l2_prediction_path
        self.l3_prediction_path = l3_prediction_path
        self.base_model_ids = base_model_ids
        self.meta_model_ids = meta_model_ids
        self.meta_target = RAW if is_filtered else ACTUAL
        self.prediction_steps = prediction_steps
        self.initial_base_training_size = initial_base_training_size
        self.start_training_index = start_training_index
        self.is_first_run = True
        self.is_filtered = is_filtered
        self.is_update_csv_required = is_update_csv_required_initially
        self.l1_gateway = GatewayL1(base_model_ids)
        self.l2_gateway = GatewayL2(meta_model_ids)
        self.l3_gateway = GatewayL3(
            meta_model_ids=meta_model_ids, meta_prediction_source=l2_prediction_path
        )
        self.data_manager = DataManager()
        self.loop_count = 0
        self.is_move_to_archive_required = is_move_to_archive_required
        self.before_filter_dataset = self.data_manager.LoadDataset(BEFORE_FILTER_FILE)
        self.meta_file_destinations = [
            self.l1_prediction_path,
            self.l2_prediction_path,
            self.l3_prediction_path,
        ]

    def Run(self, range: int = None):
        # Validate file if it is the first run
        if self.is_first_run:
            self.validateProcess()
            self.is_first_run = False

        if self.is_move_to_archive_required:
            # Move the L2 outdated file to archive directory
            l2_archive_dir = generate_meta_archive_directory_path(layer="l2")
            self.data_manager.MoveCSV(self.l2_prediction_path, l2_archive_dir)

            # Move the L3 outdated file to archive directory
            l3_archive_dir = generate_meta_archive_directory_path(layer="l3")
            self.data_manager.MoveCSV(self.l3_prediction_path, l3_archive_dir)

        # Prepare the initial CSV for L2 and L3 before manually running the loop
        if range is not None:
            for _ in range:
                # Process prediction
                self.ProcessPrediction()
                self.loop_count += 1

        while True:
            # Wait for user to press Enter to continue
            print("Press Enter to continue...")
            input()

            # Process prediction
            self.ProcessPrediction()

    def ProcessPrediction(self):
        # Write latest actual data (row: previous+step) if required
        if self.is_update_csv_required:
            self.updateCSVToLatest()

        # Train base models with the latest data in based CSV
        self.trainBaseModels()

        # Train meta models with the latest data in CSV-1
        self.trainMetaModels()

        # Calculate weight of each meta model with the data in CSV-2
        weights = self.calculateWeight()

        # Predict the next step using prediction_steps based on the base models
        base_results = self.predictBaseModels()

        # Predict the next step using prediction_steps based on the meta models
        meta_results = self.predictMetaModels(base_results)

        # Find final result by weight averaging of the prediction result from meta models
        final_result = self.predictFinalResultWithWeightAverage(meta_results, weights)

        # Share index to the meta and final's prediction result
        self.shareIndex(base_results, meta_results, final_result)

        # Save the prediction result of base and meta models
        self.savePredictionResults(
            l1_df=base_results,
            l2_df=meta_results,
            l3_df=final_result,
        )

        # Update the flag to indicate that the CSV is updated
        self.is_update_csv_required = True

        print_loop_message(self.loop_count, "Main", "Finished", "\n")

        # Increase loop count
        self.loop_count += 1

    def validateProcess(self):
        print("Validating file...")
        isL1MetaFileExist = self.data_manager.IsFileExist(self.l1_prediction_path)

        if isL1MetaFileExist == False:
            message = f"File {self.l1_prediction_path} is not found. Run setup first."
            raise Exception(message)

    def updateCSVToLatest(self):
        print_loop_message(self.loop_count, "Main", "Updating CSV to latest...")
        for dest in self.meta_file_destinations:
            self.updateCSVPredictionToLatest(dest)

    def updateCSVPredictionToLatest(self, dir: str):
        if (
            dir != self.l1_prediction_path
            and dir != self.l2_prediction_path
            and dir != self.l3_prediction_path
        ):
            raise Exception("dir must be either L1, L2 or L3")
        dest = self.data_manager.ReadCSV(path=dir, index_col=TIME)
        self.data_manager.UpdateDestinationToLatest(
            src=self.dataset,
            dest=dest,
            src_target=self.selected_feature,
            dest_target=ACTUAL,
        )
        if self.is_filtered:
            self.data_manager.UpdateDestinationToLatest(
                src=self.before_filter_dataset,
                dest=dest,
                src_target=self.selected_feature,
                dest_target=RAW,
            )
        self.data_manager.UpdateRowsInCSV(
            path=dir,
            updated_rows=dest,
            index_col_name=TIME,
        )

    def calculateWeight(self):
        print_loop_message(self.loop_count, "Main", "Calculating weight...")
        weights = self.l3_gateway.FindModelWeights()
        return weights

    def trainBaseModels(self):
        print_loop_message(self.loop_count, "Main", "Training base models...")
        meta_rows_count = self.data_manager.CountWithoutHeader(self.l1_prediction_path)
        last_training_index = (
            self.start_training_index
            + self.initial_base_training_size
            + meta_rows_count
        )
        self.l1_gateway.TrainModels(
            dataset=self.dataset,
            feature=self.selected_feature,
            start_index=self.start_training_index,
            end_index=last_training_index,
            prediction_steps=self.prediction_steps,
        )

    def trainMetaModels(self):
        print_loop_message(self.loop_count, "Main", "Training meta models...")
        dataset = self.data_manager.ReadCSV(
            path=self.l1_prediction_path, index_col=TIME
        )
        self.l2_gateway.TrainModels(
            dataset=dataset,
            features=self.base_model_ids,
            target=self.meta_target,
        )

    def predictBaseModels(self) -> pd.DataFrame:
        print_loop_message(self.loop_count, "Main", "Predicting base models...")
        prediction_result = self.l1_gateway.Predict(steps=self.prediction_steps)
        return prediction_result

    def predictMetaModels(self, base_results: pd.DataFrame) -> pd.DataFrame:
        print_loop_message(self.loop_count, "Main", "Predicting meta models...")
        prediction_result = self.l2_gateway.Predict(input=base_results)
        return prediction_result

    def predictFinalResultWithWeightAverage(
        self, meta_results: pd.DataFrame, weights: object
    ) -> pd.DataFrame:
        print_loop_message(self.loop_count, "Main", "Predicting final result...")
        prediction_result = self.l3_gateway.Predict(input=meta_results, weights=weights)
        return prediction_result

    def shareIndex(self, base: pd.DataFrame, *results: pd.DataFrame):
        """
        Args:
        - base (pd.DataFrame): The base dataframe
        - results (List[pd.DataFrame]): List of dataframes to update their index based on base dataframe's index
        """
        print_loop_message(self.loop_count, "Main", "Adding index...")
        for result in results:
            result.index = base.index

    def savePredictionResults(
        self,
        l1_df: pd.DataFrame,
        l2_df: pd.DataFrame,
        l3_df: pd.DataFrame,
    ):
        print_loop_message(self.loop_count, "Main", "Extracting data...")
        l1_rows, l1_headers = self.data_manager.ExtractMainPredictionToCSV(l1_df)
        l2_rows, l2_headers = self.data_manager.ExtractMainPredictionToCSV(l2_df)
        l3_rows, l3_headers = self.data_manager.ExtractMainPredictionToCSV(l3_df)

        # Add Raw header to l1_headers if self.is_filtered = True
        if self.is_filtered:
            l1_headers.append("Raw")
            l2_headers.append("Raw")
            l3_headers.append("Raw")

        print_loop_message(self.loop_count, "Main", "Saving data into CSV...")
        self.data_manager.WriteCSV(
            path=self.l1_prediction_path, headers=l1_headers, rows=l1_rows
        )
        self.data_manager.WriteCSV(
            path=self.l2_prediction_path, headers=l2_headers, rows=l2_rows
        )
        self.data_manager.WriteCSV(
            path=self.l3_prediction_path, headers=l3_headers, rows=l3_rows
        )
