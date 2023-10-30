import os
import sys

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from manager.data_manager import DataManager
from manager.setup_manager import SetupManager
from manager.main_manager import MainManager
from config.control import (
    BASE_MODELS_IDS,
    META_MODELS_IDS,
    SELECTED_FEATURE,
    PREDICTION_STEPS,
    INITIAL_BASE_TRAINING_SIZE,
    INITIAL_META_TRAINING_SIZE,
    INITIAL_BASE_TRAINING_SIZE_IN_MAIN,
    INITIAL_META_TRAINING_SIZE_IN_MAIN,
    IS_FILTERED,
    IS_SETUP_META_MODEL_DATASET_REQUIRED,
)
from config.path import (
    L1_META_DATASET_PATH,
    L2_META_DATASET_PATH,
    FINAL_PREDICTION_PATH,
    BASE_DATASET_PATH,
)


def main():
    # Trigger Model Setup if Required
    if IS_SETUP_META_MODEL_DATASET_REQUIRED:
        print("Preparing meta model dataset...")
        DataManagerInstance = DataManager()
        SetupManagerInstance = SetupManager(
            dataset=DataManagerInstance.LoadDataset(BASE_DATASET_PATH),
            selected_feature=SELECTED_FEATURE,
            meta_training_path=L1_META_DATASET_PATH,
            base_model_ids=BASE_MODELS_IDS,
            prediction_steps=PREDICTION_STEPS,
            initial_base_training_size=INITIAL_BASE_TRAINING_SIZE,
            initial_meta_training_size=INITIAL_META_TRAINING_SIZE,
            is_filtered=IS_FILTERED,
        )
        SetupManagerInstance.PrepareMetaModelDataset()

    # Start Main Process
    print("Starting main process...")
    MainManagerInstance = MainManager(
        base_dataset=DataManagerInstance.LoadDataset(BASE_DATASET_PATH),
        selected_feature=SELECTED_FEATURE,
        l1_meta_training_path=L1_META_DATASET_PATH,
        l2_meta_training_path=L2_META_DATASET_PATH,
        final_prediction_path=FINAL_PREDICTION_PATH,
        base_model_ids=BASE_MODELS_IDS,
        meta_model_ids=META_MODELS_IDS,
        initial_base_training_size=INITIAL_BASE_TRAINING_SIZE_IN_MAIN,
        initial_meta_training_size=INITIAL_META_TRAINING_SIZE_IN_MAIN,
        prediction_steps=PREDICTION_STEPS,
        is_filtered=IS_FILTERED,
    )
    MainManagerInstance.Run()


main()
