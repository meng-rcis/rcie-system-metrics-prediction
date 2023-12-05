import pandas as pd


def remove_column_from_csv(csv_file_path, column_names):
    df = pd.read_csv(csv_file_path)
    for column_name in column_names:
        if column_name in df.columns:
            df = df.drop(column_name, axis=1)
    df.to_csv(csv_file_path, index=False)


def loop_remove_column_from_csv(csv_file_paths, column_names):
    for csv_file_path in csv_file_paths:
        remove_column_from_csv(csv_file_path, column_names)


loop_remove_column_from_csv(
    [
        "./models/features/source/l1_prediction_dataset/prediction_result_filtered.csv",
    ],
    ["ETS", "SARIMA"],
)
