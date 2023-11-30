import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from functools import reduce
from constant.path import PATH_FROM_ROOT
from constant.columns import COLS
import pandas as pd
import models.features.prediction.pconstant.models_id as models_id


CONFIG = {
    # "cpu_usage", "memory_usage", "inbound_bandwidth", "outbound_bandwidth", "tps", "response_time"
    "features": ["cpu_usage", "memory_usage", "inbound_bandwidth"],
    "dataset": PATH_FROM_ROOT,
    "file_source": "./models/features/tuning",
    "file_destination": "./models/label/source/dataset.csv",
    "config": {
        "cpu_usage": {
            "chosen_layer": "l3",
            "chosen_feature": "Predicted",
        },
        "memory_usage": {
            "chosen_layer": "l1",
            "chosen_feature": models_id.LSTM,
        },
        "inbound_bandwidth": {
            "chosen_layer": "l2",
            "chosen_feature": models_id.REGRESSION_STACK,
        },
        "outbound_bandwidth": {
            "chosen_layer": "l1",
            "chosen_feature": models_id.LSTM,
        },
        "tps": {},
        "response_time": {},
    },
}


def merge_dataframes_on_time(dfs):
    # Prepare DataFrames for merging
    prepared_dfs = []
    for d in dfs:
        for key, df in d.items():
            # Make a copy of the DataFrame to avoid SettingWithCopyWarning
            df_copy = df.copy()
            # Rename the feature column(s) except 'Time' with the dictionary key
            df_copy.rename(
                columns={col: key if col != "Time" else col for col in df_copy.columns},
                inplace=True,
            )
            prepared_dfs.append(df_copy)

    # Merge all DataFrames on the 'Time' column
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on="Time", how="inner"), prepared_dfs
    )
    return merged_df


def join_tuning_file():
    # Load Dataset
    dataset = pd.read_csv(CONFIG["dataset"], skiprows=1, header=None, names=COLS)
    dataset.rename(columns={"time": "Time"}, inplace=True)
    predicted_metrics = []
    for feature in CONFIG["features"]:
        # Load Tuning File
        feature_config = CONFIG["config"][feature]
        file_tuning = f"{CONFIG['file_source']}/{feature}/source/{feature_config['chosen_layer']}.csv"
        tuning = pd.read_csv(file_tuning)
        # Join Dataset and Tuning File
        predicted_metrics.append(
            {feature: tuning[["Time", feature_config["chosen_feature"]]]}
        )

    selected_df = dataset[["Time", "status"]]
    predicted_metrics.append({"status": selected_df})

    # Merge all DataFrames on the 'Time' column
    merged_df = merge_dataframes_on_time(predicted_metrics)
    print("Merged DataFrame:\n", merged_df.head())

    # Save to CSV
    merged_df.to_csv(CONFIG["file_destination"], index=False)
    print(f"Saved to {CONFIG['file_destination']}")


join_tuning_file()
