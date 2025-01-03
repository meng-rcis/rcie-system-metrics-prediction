import os
import csv
import pandas as pd

"""
0: Slice the window, 
1: Slice the window with normalize data (value-mean)
2: Slice the window with normalize data (value-mean)/std
3: Combine 0 and 1
"""
MODE = 0
SIZE, TARGET_Y = 10, 10
WINDOW_SIZE = f"{SIZE}_{TARGET_Y}"
TARGET = "./models/label/source/archive/target_widow_slide_dataset.csv"
DEST = f"./models/label/extra/window_slice/source/{WINDOW_SIZE}/dataset.csv"
EXPANDED_COLS = [
    "cpu_usage",
    "memory_usage",
    "bandwidth_inbound",
    "bandwidth_outbound",
    "tps",
    "response_time",
]

if MODE == 1:
    DEST = (
        f"./models/label/extra/window_slice/source/{WINDOW_SIZE}_normalize/dataset.csv"
    )
elif MODE == 2:
    DEST = f"./models/label/extra/window_slice/source/{WINDOW_SIZE}_normalize_std/dataset.csv"
elif MODE == 3:
    DEST = f"./models/label/extra/window_slice/source/{WINDOW_SIZE}_combine_0_1/dataset.csv"


def ensure_directory_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def find_mean(df, cols) -> dict:
    means = {}
    for col in cols:
        means[col] = df[col].mean()
    return means


def find_std(df, cols) -> dict:
    stds = {}
    for col in cols:
        stds[col] = df[col].std()
    return stds


def slice_window(size: int, target_y: int, mode: int = 0) -> None:
    ensure_directory_exists(TARGET)  # Ensure the TARGET directory exists
    ensure_directory_exists(DEST)  # Ensure the DEST directory exists

    df = pd.read_csv(TARGET)
    means = find_mean(df, EXPANDED_COLS)
    stds = find_std(df, EXPANDED_COLS)

    # Create expanded columns with the given windows size
    headers = ["Time", "status"]
    for col in EXPANDED_COLS:
        header_size = size if MODE != 3 else SIZE * 2
        for i in range(0, header_size):
            headers.append(f"{col}_{i}")

    # Loop through every rows
    rows = []
    for i in range(0, len(df) - size):
        # Slice the DataFrame with the window size
        df_slice = df[i : i + size]
        # Get the target value and time
        target = df_slice["status"].iloc[target_y]
        target_time = df_slice["Time"].iloc[target_y]

        # Get the expanded row
        row = [target_time, target]
        for col in EXPANDED_COLS:
            col_values = df_slice[col].values
            for value in col_values:
                if mode == 0:
                    row.append(value)
                elif mode == 1:
                    value = value - means[col]
                    row.append(value)
                elif mode == 2:
                    value = (value - means[col]) / stds[col]
                    row.append(value)
                elif mode == 3:
                    value_mean = value - means[col]
                    value_std = (value - means[col]) / stds[col]
                    row.append(value_mean)
                    row.append(value_std)
        rows = rows + [row]

    # Write the expanded rows to the new CSV file
    with open(DEST, "w") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


slice_window(size=SIZE, target_y=TARGET_Y - 1, mode=MODE)
