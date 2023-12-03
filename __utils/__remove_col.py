import pandas as pd


def remove_column_from_csv(csv_file_path, column_name):
    df = pd.read_csv(csv_file_path)
    if column_name in df.columns:
        df = df.drop(column_name, axis=1)
        df.to_csv(csv_file_path, index=False)


def loop_remove_column_from_csv(csv_file_paths, column_name):
    for csv_file_path in csv_file_paths:
        remove_column_from_csv(csv_file_path, column_name)


loop_remove_column_from_csv(
    [
        "./models/features/tuning/cpu_usage/source/l1.csv",
        "./models/features/tuning/memory_usage/source/l1.csv",
        "./models/features/tuning/inbound_bandwidth/source/l1.csv",
        "./models/features/tuning/outbound_bandwidth/source/l1.csv",
        "./models/features/tuning/tps/source/l1.csv",
        "./models/features/tuning/response_time/source/l1.csv",
    ],
    "GP",
)
