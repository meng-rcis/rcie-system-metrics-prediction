import os
import sys

# Add path to the root folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from config.control import IS_FILTERED

# Define meta training dataset file
META_FILE = "meta_df_filtered.csv" if IS_FILTERED else "meta_df.csv"

# Define meta training dataset path
META_TRAINING_PATH = "models/features/source/meta_training_dataset/" + META_FILE

# Define base training dataset path
DATASET_PATH = "dump/"
