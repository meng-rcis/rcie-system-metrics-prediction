import os
import csv
import pandas as pd
import pickle
import shutil

from typing import Any, List, Dict, Tuple


class DataManager:
    def __init__(self):
        pass

    @staticmethod
    def ExtractPredictionToCSV(
        prediction_result: Dict[str, pd.Series],
        actual_result: Dict[str, pd.Series],
        model_ids: List[str],
        before_filter_dataset: Dict[str, pd.Series] = None,
    ) -> Tuple[List[List[Any]], List[str]]:
        # Get the union of all indices from all Series to ensure we capture all unique indices
        all_indices = sorted(
            set().union(*(prediction_result[model_id].index for model_id in model_ids))
        )

        extracted_data = []
        for idx in all_indices:
            current_row = [idx]  # Start with the index (datetime) itself
            for model_id in model_ids:
                current_row.append(prediction_result[model_id].get(idx, None))
            current_row.append(actual_result.get(str(idx), None))
            extracted_data.append(current_row)

            if before_filter_dataset is not None:
                current_row.append(before_filter_dataset.get(str(idx), None))

        # Creating the header
        header = ["Time"] + model_ids + ["Actual"]
        header = header + ["Raw"] if before_filter_dataset is not None else header
        return extracted_data, header

    @staticmethod
    def LoadDataset(path: str):
        try:
            with open(path, "rb") as file:
                return pickle.load(open(path, "rb"))
        except Exception as e:
            print(f"Error loading dataset from {path}: {e}")
            raise

    @staticmethod
    def WriteCSV(path: str, header: List[str], rows: List[List[Any]]):
        # Check if file exists and has content
        file_exists = os.path.exists(path)

        with open(path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # If file doesn't exist or is empty, write the header
            if not file_exists or os.path.getsize(path) == 0:
                writer.writerow(header)

            # Write the data rows
            writer.writerows(rows)

    @staticmethod
    def ReadCSV(path: str):
        return pd.read_csv(path)

    @staticmethod
    def MoveCSV(file_path: str, dest_directory: str):
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File '{file_path}' not found!")
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
