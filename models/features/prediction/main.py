import os
import sys
import warnings

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
    ALPHA,
    IS_FILTERED,
    IS_SETUP_META_MODEL_DATASET_REQUIRED,
    IS_FORCE_MOVE_FILE_TO_ARCHIVE_FOLDER,
    IS_UPDATE_CSV_REQUIRED_INITIALLY,
    IS_CLEANING_ROWS_REQUIRED_INITIALLY,
    IS_HIDE_WARNING,
    RANGE_REQUIRED_TO_AUTO_GENERATE_FINAL_RESULT_SIZE,
)
from config.path import (
    L1_PREDICTION_DATASET_PATH,
    L2_PREDICTION_DATASET_PATH,
    L3_PREDICTION_DATASET_PATH,
    BASE_DATASET_PATH,
)


def main():
    if IS_HIDE_WARNING:
        warnings.filterwarnings("ignore")

    DataManagerInstance = DataManager()

    # Trigger Model Setup if Required
    if IS_SETUP_META_MODEL_DATASET_REQUIRED:
        print("Preparing meta model dataset...")
        SetupManagerInstance = SetupManager(
            dataset=DataManagerInstance.LoadDataset(BASE_DATASET_PATH),
            selected_feature=SELECTED_FEATURE,
            l1_prediction_path=L1_PREDICTION_DATASET_PATH,
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
        dataset=DataManagerInstance.LoadDataset(BASE_DATASET_PATH),
        selected_feature=SELECTED_FEATURE,
        l1_prediction_path=L1_PREDICTION_DATASET_PATH,
        l2_prediction_path=L2_PREDICTION_DATASET_PATH,
        l3_prediction_path=L3_PREDICTION_DATASET_PATH,
        initial_base_training_size=INITIAL_BASE_TRAINING_SIZE,
        base_model_ids=BASE_MODELS_IDS,
        meta_model_ids=META_MODELS_IDS,
        prediction_steps=PREDICTION_STEPS,
        alpha=ALPHA,
        is_filtered=IS_FILTERED,
        is_update_csv_required_initially=IS_UPDATE_CSV_REQUIRED_INITIALLY,
        is_clean_rows_required_initially=IS_CLEANING_ROWS_REQUIRED_INITIALLY,
        is_move_to_archive_required=IS_SETUP_META_MODEL_DATASET_REQUIRED
        or IS_FORCE_MOVE_FILE_TO_ARCHIVE_FOLDER,
    )
    MainManagerInstance.Run(auto_loop=RANGE_REQUIRED_TO_AUTO_GENERATE_FINAL_RESULT_SIZE)


main()
