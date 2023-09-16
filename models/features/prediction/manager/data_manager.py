import os
import csv

from typing import Any, List, Dict, Tuple
import pickle
import pandas as pd

class DataManager:
    def __init__(self):
        pass
    
    @staticmethod
    def ExtractPredictionToCSV(
        prediction_result: Dict[str, pd.Series], 
        actual_result: Dict[str, pd.Series], 
        model_ids: List[str]
        ) -> Tuple[List[List[Any]], List[str]]:
        # Get the union of all indices from all Series to ensure we capture all unique indices
        all_indices = sorted(set().union(*(prediction_result[model_id].index for model_id in model_ids)))
        
        extracted_data = []
        for idx in all_indices:
            current_row = [idx]  # Start with the index itself
            for model_id in model_ids:
                current_row.append(prediction_result[model_id].get(idx, None))
            current_row.append(actual_result.get(idx, None))
            extracted_data.append(current_row)
        
        # Creating the header
        header = ['index'] + model_ids + ['actual']
        return extracted_data, header

    
    @staticmethod
    def LoadDataset(path: str):
        try:
            with open(path, 'rb') as file:
                return pickle.load(open(path, 'rb'))
        except Exception as e:
            print(f"Error loading dataset from {path}: {e}")
            raise
    
    @staticmethod
    def WriteCSV(path: str, header: List[str], rows: List[List[Any]]):
        # Check if file exists and has content
        file_exists = os.path.exists(path)
        
        with open(path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # If file doesn't exist or is empty, write the header
            if not file_exists or os.path.getsize(path) == 0:
                writer.writerow(header)

            # Write the data rows
            writer.writerows(rows)

    @staticmethod
    def ReadCSV(path: str):
        return pd.read_csv(path)
    
    