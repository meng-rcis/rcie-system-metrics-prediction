import pickle
import csv
import pandas as pd


def convert_pickle_csv(pickle_file, csv_file):
    with open(pickle_file, "rb") as file:
        data = pickle.load(file)

    if isinstance(data, pd.DataFrame):
        data.to_csv(csv_file, index=True)
    else:
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)

    print(f"Converted {pickle_file} to {csv_file}")
    return csv_file


convert_pickle_csv("dump/filtered_df.p", "dump/filtered_df.csv")
