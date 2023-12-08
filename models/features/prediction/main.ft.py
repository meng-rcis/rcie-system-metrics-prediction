import os
import sys

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from manager import DataManager
from putils.printer import print_loop_message
from pconstant.feature_header import RAW, ACTUAL, TIME
from controller import L2, L3
from config.control import CONFIG
from config.path import (
    BASE_DATASET_PATH,
    BEFORE_FILTER_FILE,
    L1_PREDICTION_DATASET_PATH,
    L2_PREDICTION_DATASET_PATH,
    L3_PREDICTION_DATASET_PATH,
)


def main():
    (
        alpha,
        is_filtered,
        meta_model_ids,
        selected_feature,
        steps,
        cumulative_training_size,
        is_parallel_processing_for_l2,
        auto_created_final_result_size,
    ) = (
        CONFIG["ALPHA"],
        CONFIG["IS_FILTERED"],
        CONFIG["META_MODELS_IDS"],
        CONFIG["SELECTED_FEATURE"],
        CONFIG["PREDICTION_STEPS"],
        CONFIG["INITIAL_BASE_TRAINING_SIZE"],
        CONFIG["IS_PARALLEL_PROCESSING_FOR_L2"],
        CONFIG["AUTO_CREATED_FINAL_RESULT_SIZE"],
    )

    df_l1 = DataManager.ReadCSV(
        path=L1_PREDICTION_DATASET_PATH,
        index_col=TIME,
    )
    base_df = DataManager.LoadDataset(path=BASE_DATASET_PATH)
    before_filter_df = (
        DataManager.LoadDataset(path=BEFORE_FILTER_FILE) if is_filtered else None
    )

    def updateCSVPredictionToLatest(dir: str):
        if DataManager.IsFileExist(dir) == False:
            return
        dest = DataManager.ReadCSV(path=dir, index_col=TIME)
        DataManager.UpdateDestinationToLatest(
            src=base_df,
            dest=dest,
            src_target=selected_feature,
            dest_target=ACTUAL,
        )
        if is_filtered:
            DataManager.UpdateDestinationToLatest(
                src=before_filter_df,
                dest=dest,
                src_target=selected_feature,
                dest_target=RAW,
            )
        DataManager.UpdateRowsInCSV(
            path=dir,
            updated_rows=dest,
            index_col_name=TIME,
        )

    l2_gateway = L2(
        target=ACTUAL,
        model_ids=meta_model_ids,
        is_parallel_processing=is_parallel_processing_for_l2,
    )
    l3_gateway = L3(
        meta_model_ids=meta_model_ids,
        meta_prediction_source=L2_PREDICTION_DATASET_PATH,
        target_col=RAW if is_filtered else ACTUAL,
        alpha=alpha,
    )

    loop_required = auto_created_final_result_size // steps
    for i in range(loop_required):
        # Prepare Dataset
        print_loop_message(i + 1, "MainFT", "Preparing dataset...")
        next_prediction = cumulative_training_size + steps
        end_index = min(next_prediction, len(df_l1))
        training_dataset, testing_dataset = (
            df_l1[:cumulative_training_size],
            df_l1[cumulative_training_size:end_index],
        )

        # L2 Prediction
        print_loop_message(i + 1, "MainFT", "Training L2 models...")
        l2_gateway.TrainModels(
            dataset=training_dataset,
            features=[],
        )
        print_loop_message(i + 1, "MainFT", "Predicting L2 models...")
        l2_prediction = l2_gateway.Predict(input=testing_dataset)

        # L3 Prediction
        print_loop_message(i + 1, "MainFT", "Finding weights in L3 models...")
        weights = l3_gateway.FindModelWeights()
        print_loop_message(i + 1, "MainFT", "Predicting L3 models...")
        l3_prediction = l3_gateway.Predict(
            input=l2_prediction,
            weights=weights,
        )

        # Share Index
        l2_prediction.index = testing_dataset.index
        l3_prediction.index = testing_dataset.index

        # Save to CSV
        print_loop_message(i + 1, "MainFT", "Saving to CSV...")
        l2_rows, l2_headers = DataManager.ExtractMainPredictionToCSV(l2_prediction)
        l3_rows, l3_headers = DataManager.ExtractMainPredictionToCSV(l3_prediction)

        if is_filtered:
            l2_headers.append("Raw")
            l3_headers.append("Raw")

        DataManager.WriteCSV(
            path=L2_PREDICTION_DATASET_PATH,
            headers=l2_headers,
            rows=l2_rows,
        )
        DataManager.WriteCSV(
            path=L3_PREDICTION_DATASET_PATH,
            headers=l3_headers,
            rows=l3_rows,
        )

        # Update CSV
        print_loop_message(i + 1, "MainFT", "Updating CSV...\n")
        updateCSVPredictionToLatest(dir=L2_PREDICTION_DATASET_PATH)
        updateCSVPredictionToLatest(dir=L3_PREDICTION_DATASET_PATH)

        # Update Cumulative Training Size
        cumulative_training_size += steps


if __name__ == "__main__":
    main()
