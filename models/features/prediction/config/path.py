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

# Define meta training dataset archive path
META_ARCHIVE_DIRECTORY = "models/features/source/meta_training_dataset/archive/" + str(
    round(time.time() * 1000)
)

"""
This is the path for the base training dataset
Models in layer 1 will be trained with this dataset
"""

DUMP_BASE_DIRECTORY = "dump/"

# Define base training dataset file
BASE_FILE = "filtered_df.p" if IS_FILTERED else "df.p"

# Define base training dataset path
BASE_DATASET_PATH = DUMP_BASE_DIRECTORY + BASE_FILE

# Define base training dataset file before filter
BEFORE_FILTER_FILE = DUMP_BASE_DIRECTORY + "df.p"

"""
This is the path for the meta training dataset
Models in layer 1 will store their prediction results in this dataset
Models in layer 2 will be trained with this dataset
"""

# Define meta training dataset file
META_FILE = "meta_df_filtered.csv" if IS_FILTERED else "meta_df.csv"

# Define meta training dataset path
META_DATASET_PATH = "models/features/source/meta_training_dataset/" + META_FILE
