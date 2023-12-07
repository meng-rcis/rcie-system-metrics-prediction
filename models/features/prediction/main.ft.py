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
from pconstant.feature_header import ACTUAL
from controller import L1, L2, L3
from config.control import CONFIG
from config.path import (
    L1_PREDICTION_DATASET_PATH,
    L2_PREDICTION_DATASET_PATH,
    L3_PREDICTION_DATASET_PATH,
    BASE_DATASET_PATH,
)


def main():
    df_l1 = DataManager().ReadCSV(L1_PREDICTION_DATASET_PATH)
    base_training_size = CONFIG["INITIAL_BASE_TRAINING_SIZE"]
    redefine_interval = CONFIG["T_REDEFINE_MODEL_INTERVAL"]
    auto_created_base_result_size = CONFIG["T_AUTO_CREATED_BASE_RESULT_SIZE"]
    l2_gateway = L2(
        target=ACTUAL,
        model_ids=CONFIG["META_MODELS_IDS"],
        is_parallel_processing=CONFIG["IS_PARALLEL_PROCESSING_FOR_L2"],
    )

    redefine_interval_loop_required = auto_created_base_result_size // redefine_interval
    cumulative_training_size = base_training_size

    # # Initialize an empty DataFrame to store all results
    # all_results_df = pd.DataFrame()

    # for i in range(redefine_interval_loop_required):
    #     # Prepare data
    #     start_time = time.time()
    #     print(f"Redefine Model: {i + 1}")
    #     next_prediction = cumulative_training_size + redefine_interval
    #     end_index = min(next_prediction, len(dataset_cp))
    #     training_dataset, testing_dataset = (
    #         dataset_cp[starting_training_index:cumulative_training_size],
    #         dataset_cp[cumulative_training_size - n_past : end_index],
    #     )
    #     scaled_training_dataset = scaler.fit_transform(
    #         training_dataset.values.reshape(-1, 1)
    #     )
    #     scaled_testing_dataset = scaler.transform(testing_dataset.values.reshape(-1, 1))
    #     X_train, y_train = create_sequences(scaled_training_dataset, n_past, steps)
    #     X_test, y_test = create_sequences(scaled_testing_dataset, n_past, steps, False)

    #     # Define model
    #     model = define_model(X_train, y_train, MODEL_CONFIG)

    #     # Predict
    #     print(f"Predict Model: {i + 1}")
    #     yhat = model.predict(X_test)
    #     yhat_original = scaler.inverse_transform(yhat)
    #     y_test = scaler.inverse_transform(y_test)

    #     # Prepare result
    #     result_df = pd.DataFrame(
    #         {
    #             "Time": time_indices[cumulative_training_size:end_index],
    #             "RNN": yhat_original.flatten(),
    #             "Actual": y_test.flatten(),
    #             "Raw": raw_dataset_cp[cumulative_training_size:end_index],
    #         }
    #     )
    #     all_results_df = pd.concat([all_results_df, result_df], ignore_index=True)
    #     cumulative_training_size += redefine_interval
    #     end_time = time.time()
    #     total_time = end_time - start_time
    #     print(f"Cumulative Size: {cumulative_training_size} [{total_time} Seconds]\n")

    # print("Save Results")
    # all_results_df.to_csv(SAVE_PATH, index=False)
    # print("Done")
