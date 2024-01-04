import os
import sys
import warnings
import math

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from manager import MainManager, DataManager, SetupManager
from config.control import CONFIG
from config.path import (
    L1_PREDICTION_DATASET_PATH,
    L2_PREDICTION_DATASET_PATH,
    L3_PREDICTION_DATASET_PATH,
    BASE_DATASET_PATH,
)


def main():
    if CONFIG["IS_HIDE_WARNING"]:
        warnings.filterwarnings("ignore")

    DataManagerInstance = DataManager()
    # Trigger Model Setup if Required
    if CONFIG["IS_SETUP_META_MODEL_DATASET_REQUIRED"]:
        print("Preparing meta model dataset...")
        SetupManagerInstance = SetupManager(
            dataset=DataManagerInstance.LoadDataset(BASE_DATASET_PATH),
            selected_feature=CONFIG["SELECTED_FEATURE"],
            l1_prediction_path=L1_PREDICTION_DATASET_PATH,
            l2_prediction_path=L2_PREDICTION_DATASET_PATH,
            l3_prediction_path=L3_PREDICTION_DATASET_PATH,
            base_model_ids=CONFIG["BASE_MODELS_IDS"],
            prediction_steps=CONFIG["PREDICTION_STEPS"],
            initial_base_training_size=CONFIG["INITIAL_BASE_TRAINING_SIZE"],
            initial_meta_training_size=CONFIG["INITIAL_META_TRAINING_SIZE"],
            is_parallel_processing=CONFIG["IS_PARALLEL_PROCESSING"],
            is_filtered=CONFIG["IS_FILTERED"],
        )
        SetupManagerInstance.PrepareMetaModelDataset()

    if CONFIG["IS_MAIN_PROCESS_REQUIRED"] == False:
        print("Main process is not required. Exiting...")
        return

    # Start Main Process
    print("Starting main process...")
    MainManagerInstance = MainManager(
        dataset=DataManagerInstance.LoadDataset(BASE_DATASET_PATH),
        selected_feature=CONFIG["SELECTED_FEATURE"],
        l1_prediction_path=L1_PREDICTION_DATASET_PATH,
        l2_prediction_path=L2_PREDICTION_DATASET_PATH,
        l3_prediction_path=L3_PREDICTION_DATASET_PATH,
        initial_base_training_size=CONFIG["INITIAL_BASE_TRAINING_SIZE"],
        base_model_ids=CONFIG["BASE_MODELS_IDS"],
        meta_model_ids=CONFIG["META_MODELS_IDS"],
        prediction_steps=CONFIG["PREDICTION_STEPS"],
        alpha=CONFIG["ALPHA"],
        is_filtered=CONFIG["IS_FILTERED"],
        is_parallel_processing=CONFIG["IS_PARALLEL_PROCESSING"],
        is_parallel_processing_for_l2=CONFIG["IS_PARALLEL_PROCESSING_FOR_L2"],
        is_update_csv_required_initially=CONFIG["IS_UPDATING_CSV_REQUIRED_INITIALLY"],
        is_clean_rows_required_initially=CONFIG["IS_CLEANING_ROWS_REQUIRED_INITIALLY"]
        and CONFIG["IS_UPDATING_CSV_REQUIRED_INITIALLY"] == False,
        is_move_to_archive_required=CONFIG[
            "MANUALLY_MOVE_L2_L3_FILES_TO_ARCHIVE_FOLDER"
        ],
    )

    auto_loop = (
        math.ceil(CONFIG["AUTO_CREATED_FINAL_RESULT_SIZE"] / CONFIG["PREDICTION_STEPS"])
        if CONFIG["AUTO_CREATED_FINAL_RESULT_SIZE"] > 0
        else 0
    )
    MainManagerInstance.Run(auto_loop=auto_loop)


if __name__ == "__main__":
    main()
