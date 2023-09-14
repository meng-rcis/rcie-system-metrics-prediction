import os
import csv

import pickle
import pandas as pd

class DataManager:
    def __init__(self):
        pass
    
    def LoadDataset(self, filename: str):
        try:
            with open(filename, 'rb') as file:
                return pickle.load(open(filename, 'rb'))
        except Exception as e:
            print(f"Error loading dataset from {filename}: {e}")
            raise
    
    def WriteCSV(self, filename: str, header: list[str], rows: list[list[str]]):
        # Check if file exists and has content
        file_exists = os.path.exists(filename)
        
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # If file doesn't exist or is empty, write the header
            if not file_exists or os.path.getsize(filename) == 0:
                writer.writerow(header)

            # Write the data rows
            writer.writerows(rows)

    def ReadCSV(self, filename: str):
        return pd.read_csv(filename)
    
    