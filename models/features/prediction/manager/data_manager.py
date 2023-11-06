import os
import csv
import pandas as pd
import pickle
import shutil

from typing import Any, List, Dict, Tuple
from pconstant.feature_header import ACTUAL, RAW, TIME


class DataManager:
    def __init__(self):
        pass

    @staticmethod
    def ExtractSetupPredictionToCSV(
        prediction_result: Dict[str, pd.Series],
        actual_result: Dict[str, pd.Series],
        model_ids: List[str],
        before_filter_dataset: Dict[str, pd.Series] = None,
    ) -> Tuple[List[List[Any]], List[str]]:
        # Get the union of all indices from all Series to ensure we capture all unique indices
        all_indices = sorted(
            set().union(*(prediction_result[model_id].index for model_id in model_ids))
        )

        rows = []
        for idx in all_indices:
            current_row = [idx]  # Start with the index (datetime) itself
            for model_id in model_ids:
                current_row.append(prediction_result[model_id].get(idx, None))
            current_row.append(actual_result.get(str(idx), None))
            rows.append(current_row)

            if before_filter_dataset is not None:
                current_row.append(before_filter_dataset.get(str(idx), None))

        # Creating the headers
        headers = [TIME] + model_ids + [ACTUAL]
        headers = headers + [RAW] if before_filter_dataset is not None else headers
        return rows, headers

    @staticmethod
    def ExtractMainPredictionToCSV(
        df: pd.DataFrame,
    ) -> Tuple[List[List[Any]], List[str]]:
        headers = [TIME] + list(df.columns) + [ACTUAL]
        rows = [[index] + row.tolist() for index, row in zip(df.index, df.values)]
        return rows, headers

    @staticmethod
    def LoadDataset(path: str):
        try:
            with open(path, "rb") as file:
                return pickle.load(open(path, "rb"))
        except Exception as e:
            print(f"Error loading dataset from {path}: {e}")
            raise

    @staticmethod
    def WriteCSV(path: str, headers: List[str], rows: List[List[Any]]):
        # Check if file exists and has content
        file_exists = os.path.exists(path)

        with open(path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # If file doesn't exist or is empty, write the headers
            if not file_exists or os.path.getsize(path) == 0:
                writer.writerow(headers)

            # Write the data rows
            writer.writerows(rows)

    @staticmethod
    def ReadCSV(path: str, index_col: str = None):
        return pd.read_csv(
            path,
            index_col=index_col,
        )

    @staticmethod
    def MoveCSV(file_path: str, dest_directory: str):
        # Check if the file exists
        if not os.path.exists(file_path):
            return

        # Extract the file name from the path
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(dest_directory, file_name)

        # Ensure the destination directory exists
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)

        # Move the file to the new directory
        try:
            shutil.move(file_path, dest_path)
            print(f"File '{file_path}' moved to '{dest_path}'!")
        except PermissionError:
            print(f"Permission denied to move file '{file_path}' to '{dest_path}'!")
        except Exception as e:
            print(f"An error occurred while moving: {e}")

    @staticmethod
    def IsFileExist(path: str) -> bool:
        return os.path.exists(path)

    @staticmethod
    def CountWithoutHeader(path: str) -> int:
        if os.path.exists(path) == False:
            return 0

        with open(path, "r") as file:
            return len(file.readlines()) - 1

    @staticmethod
    def UpdateDestinationToLatest(
        src: pd.DataFrame,
        dest: pd.DataFrame,
        src_target: str,
        dest_target: str,
    ):
        # Loop through dest DataFrame
        for idx in dest.index:
            # Check if the dest_target value in dest is NaN
            if pd.isna(dest.at[idx, dest_target]):
                # Retrieve the value from src DataFrame using the same index and the src_target column
                value = src.at[idx, src_target]

                # Update the value in dest DataFrame
                dest.at[idx, dest_target] = value

    @staticmethod
    def UpdateRowsInCSV(path: str, updated_rows: pd.DataFrame, index_col_name: str):
        # Read the original CSV using the specified column as the index
        original_df = pd.read_csv(path, index_col=index_col_name)

        # Update rows based on the index
        original_df.update(updated_rows.set_index(index_col_name))

        # Append rows that don't exist in the original dataframe
        new_rows = updated_rows.set_index(index_col_name).loc[
            ~updated_rows[index_col_name].isin(original_df.index)
        ]
        combined_df = pd.concat([original_df, new_rows])

        # Save back to CSV
        combined_df.reset_index().to_csv(path, index=False)
