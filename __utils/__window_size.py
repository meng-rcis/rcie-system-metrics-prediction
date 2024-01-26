import os
import csv
import pandas as pd

SIZE, TARGET_Y = 16, 8
WINDOW_SIZE = f"{SIZE}_{TARGET_Y}"
TARGET = "./models/label/source/dataset.csv"
DEST = f"./models/label/extra/window_slice/source/{WINDOW_SIZE}/dataset.csv"
EXPANDED_COLS = [
    "cpu_usage",
    "memory_usage",
    "bandwidth_inbound",
    "bandwidth_outbound",
    "tps",
    "response_time",
]


def ensure_directory_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def slice_window(size: int, target_y: int) -> None:
    ensure_directory_exists(TARGET)  # Ensure the TARGET directory exists
    ensure_directory_exists(DEST)  # Ensure the DEST directory exists

    df = pd.read_csv(TARGET)
    # Create expanded columns with the given windows size
    headers = ["Time", "status"]
    for col in EXPANDED_COLS:
        for i in range(0, size):
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
                row.append(value)

        rows = rows + [row]

    # Write the expanded rows to the new CSV file
    with open(DEST, "w") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


slice_window(size=SIZE, target_y=TARGET_Y)
