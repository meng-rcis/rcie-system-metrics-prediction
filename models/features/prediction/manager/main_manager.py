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
from manager.data_manager import DataManager
from gateway.layer_1 import GatewayL1
from gateway.layer_2 import GatewayL2
from gateway.layer_3 import GatewayL3
from putils.printer import print_loop_message


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
        initial_meta_training_size: int,
        start_training_index: int = START_TRAINING_INDEX,
        prediction_steps: int = 1,
        is_filtered: bool = False,
        is_update_csv_required: bool = False,
    ):
        self.dataset = dataset
        self.selected_feature = selected_feature
        self.l1_prediction_path = l1_prediction_path
        self.l2_prediction_path = l2_prediction_path
        self.l3_prediction_path = l3_prediction_path
        self.base_model_ids = base_model_ids
        self.meta_model_ids = meta_model_ids
        self.meta_target = "Raw" if is_filtered else "Actual"
        self.prediction_steps = prediction_steps
        self.initial_base_training_size = initial_base_training_size
        self.initial_meta_training_size = initial_meta_training_size
        self.start_training_index = start_training_index
        self.is_first_run = True
        self.is_filtered = is_filtered
        self.is_update_csv_required = is_update_csv_required
        self.l1_gateway = GatewayL1(base_model_ids)
        self.l2_gateway = GatewayL2(meta_model_ids)
        self.l3_gateway = GatewayL3(
            meta_model_ids=meta_model_ids, meta_prediction_source=l2_prediction_path
        )
        self.data_manager = DataManager()
        self.loop_count = 0

    def Run(self):
        # Validate file if it is the first run
        if self.is_first_run:
            self.validateProcess()
            self.is_first_run = False

        while True:
            # Wait for user to press Enter to continue
            print_loop_message(self.loop_count, "Main", "Press Enter to continue...")
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

        print_loop_message(self.loop_count, "Main", "Base:", base_results.head())

        # Predict the next step using prediction_steps based on the meta models
        meta_results = self.predictMetaModels(base_results)

        print_loop_message(self.loop_count, "Main", "Meta:", meta_results.head())

        # Find final result by weight averaging of the prediction result from meta models
        final_result = self.predictFinalResultWithWeightAverage(meta_results, weights)

        # Save the prediction result of base and meta models
        self.savePredictionResult(
            base_model_results=base_results,
            meta_model_results=meta_results,
            final_result=final_result,
        )

        # Update the flag to indicate that the CSV is updated
        self.is_update_csv_required = True

        # Increase loop count
        self.loop_count += 1

    def validateProcess(self):
        print_loop_message(self.loop_count, "Main", "Validating file...")
        isL1MetaFileExist = self.data_manager.IsFileExist(self.l1_prediction_path)

        if isL1MetaFileExist == False:
            message = f"File {self.l1_prediction_path} is not found. Run setup first."
            raise Exception(message)

    def updateCSVToLatest(self):
        print_loop_message(self.loop_count, "Main", "Updating CSV to latest...")
        pass

    def calculateWeight(self):
        print_loop_message(self.loop_count, "Main", "Calculating weight...")
        weights = self.l3_gateway.FindModelWeights()
        return weights

    # Fix to find last index
    def trainBaseModels(self, meta_increase_size: int = 0):
        print_loop_message(self.loop_count, "Main", "Training base models...")
        last_training_index = (
            self.start_training_index
            + meta_increase_size
            + self.initial_base_training_size
        ) # TODO: Fix to find total rows of meta CSV
        self.l1_gateway.TrainModels(
            dataset=self.dataset,
            feature=self.selected_feature,
            start_index=self.start_training_index,
            end_index=last_training_index,
            prediction_steps=self.prediction_steps,
        )

    def trainMetaModels(self):
        print_loop_message(self.loop_count, "Main", "Training meta models...")
        dataset = self.data_manager.ReadCSV(self.l1_prediction_path)
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

    def savePredictionResult(
        self,
        base_model_results: pd.DataFrame,
        meta_model_results: pd.DataFrame,
        final_result: pd.DataFrame,
    ):
        pass
