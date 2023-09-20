import os
import sys

# Add path to the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from manager.data_manager import DataManager
from manager.setup_manager import SetupManager
from manager.main_manager import MainManager
from config.model import BASE_MODELS_IDS, SELECTED_FEATURE, PREDICTION_STEPS
from config.path import META_TRAINING_PATH, DATASET_PATH

def main():
    # Trigger Model Setup if Required
    isSetupMetaModelDatasetRequired = True
    if isSetupMetaModelDatasetRequired:
        print("Preparing meta model dataset...")
        DataManagerInstance = DataManager()
        SetupManagerInstance = SetupManager(
            selected_feature=SELECTED_FEATURE,
            meta_training_path=META_TRAINING_PATH,
            base_model_ids=BASE_MODELS_IDS,
            prediction_steps=PREDICTION_STEPS,
            base_training_dataset=DataManagerInstance.LoadDataset(DATASET_PATH),
            )
        SetupManagerInstance.PrepareMetaModelDataset()

    # Start Main Process
    print("Starting main process...")
    MainManagerInstance = MainManager(
        prediction_steps=PREDICTION_STEPS
        )
    # MainManagerInstance.Run()

main()