import os
import sys

# Add path to the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from manager.data_manager import DataManager
from manager.setup_manager import SetupManager
from manager.main_manager import MainManager

def main():
    # Trigger Model Setup if Required
    isSetupMetaModelDatasetRequired = True
    if isSetupMetaModelDatasetRequired:
        print("Preparing meta model dataset...")
        DataManagerInstance = DataManager()
        SetupManagerInstance = SetupManager(
            meta_training_path='../source/training/meta.csv',
            base_training_dataset=DataManagerInstance.LoadDataset('../../../dump/df.p'),
            )
        SetupManagerInstance.PrepareMetaModelDataset()

    # Start Main Process
    print("Starting main process...")
    MainManagerInstance = MainManager()
    MainManagerInstance.Run()

main()