import os
import sys
import time

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

# Define meta training dataset archive path
META_ARCHIVE_DIRECTORY = "models/features/source/meta_training_dataset/archive/" + str(
    round(time.time() * 1000)
)

# Define base training dataset path
DATASET_PATH = "dump/"
