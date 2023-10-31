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
from putils.calculator import Calculator


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
        self.data_manager = DataManager()
        self.calculator = Calculator()

    def Run(self):
        # Validate file if it is the first run
        if self.is_first_run:
            self.validateProcess()
            self.is_first_run = False

        # Wait for user to press Enter to continue
        print("Press Enter to continue...")
        input()

        # Write latest actual data (row: previous+step) if required
        if self.is_update_csv_required:
            self.updateCSVToLatest()

        # Train base models with the latest data in based CSV
        self.trainBaseModels()

        # Train meta models with the latest data in CSV-1
        self.trainMetaModels()

        return

        # Calculate weight of each meta model with the data in CSV-2
        weights = self.calculateWeight()

        # Predict the next step using prediction_steps based on the base models
        base_results = self.predictBaseModels()

        # Predict the next step using prediction_steps based on the meta models
        meta_results = self.predictMetaModels(base_results)

        # Find final result by weight averaging of the prediction result from meta models
        self.predictFinalResultWithWeightAverage(meta_results, weights)

        # Update the flag to indicate that the CSV is updated
        self.is_update_csv_required = True

    def validateProcess(self):
        print("Validating file...")
        isL1MetaFileExist = self.data_manager.IsFileExist(self.l1_prediction_path)

        if isL1MetaFileExist == False:
            message = f"File {self.l1_prediction_path} is not found. Run setup first."
            raise Exception(message)

    def updateCSVToLatest(self):
        print("Updating CSV to latest...")
        pass

    def calculateWeight(self):
        print("Calculating weight...")
        if self.data_manager.IsFileExist(self.l2_prediction_path) == False:
            return {item: 1 for item in self.meta_model_ids}

        meta_prediction = self.data_manager.ReadCSV(self.l2_prediction_path)
        weights = self.calculator.CalculateWeight(meta_prediction)
        return weights

    def trainBaseModels(self, meta_increase_size: int = 0):
        print("Training base models...")
        last_training_index = (
            self.start_training_index
            + meta_increase_size
            + self.initial_base_training_size
        )
        self.l1_gateway.TrainModels(
            dataset=self.dataset,
            feature=self.selected_feature,
            start_index=self.start_training_index,
            end_index=last_training_index,
            prediction_steps=self.prediction_steps,
        )

    def trainMetaModels(self, meta_increase_size: int = 0):
        print("Training meta models...")
        dataset = self.data_manager.ReadCSV(self.l1_prediction_path)
        last_training_index = meta_increase_size + self.initial_meta_training_size
        self.l2_gateway.TrainModels(
            dataset=dataset,
            features=self.base_model_ids,
            target=self.meta_target,
            end_index=last_training_index,
            prediction_steps=self.prediction_steps,
        )

    def predictBaseModels(self):
        print("Predicting base models...")
        pass

    def predictMetaModels(self, base_results):
        print("Predicting meta models...")
        pass

    def predictFinalResultWithWeightAverage(self, meta_results, weights):
        print("Predicting final result...")
        pass
