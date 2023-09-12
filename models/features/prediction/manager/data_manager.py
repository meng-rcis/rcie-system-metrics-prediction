import os
import sys
import csv

# Add path to the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import utils.load as load
import pandas as pd

class DataManager:
    def __init__(self):
        pass
    
    def LoadDataset(self, filename: str):
        return load(filename)
    
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
    
    