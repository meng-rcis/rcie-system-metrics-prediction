import os
import sys

# Add path to the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from manager.setup_manager import SetupManager

def main():
    # Trigger Model Setup if Required
    isSetupMetaModelDatasetRequired = False
    if isSetupMetaModelDatasetRequired:
        print("Preparing meta model dataset...")
        SetupInstance = SetupManager()
        SetupInstance.PrepareMetaModelDataset()

    # Start Main Process
    # MainManager.Run()
