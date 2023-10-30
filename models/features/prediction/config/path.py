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

"""
This is the path for the base training dataset
Base models will be trained with this dataset
"""

DUMP_BASE_DIRECTORY = "dump/"

# Define base training dataset file
BASE_FILE = "filtered_df.p" if IS_FILTERED else "df.p"

# Define base training dataset path
BASE_DATASET_PATH = DUMP_BASE_DIRECTORY + BASE_FILE

# Define base training dataset file before filter
BEFORE_FILTER_FILE = DUMP_BASE_DIRECTORY + "df.p"

"""
This is base meta configuration data
"""

# Define meta source folder
META_SOURCE_DIRECTORY = "models/features/source/"

# Define meta training dataset file
META_FILE = "meta_df_filtered.csv" if IS_FILTERED else "meta_df.csv"

"""
This is the path for the meta training dataset
Base models will store their prediction results in this dataset
L1 meta models will be trained with this dataset
"""

# Define L1 meta training dataset path
L1_META_DATASET_PATH = META_SOURCE_DIRECTORY + "l1_meta_training_dataset/" + META_FILE

"""
This is the path for the meta training dataset
L1 meta models will store their prediction results in this dataset
L2 meta models will be trained with this dataset
"""

# Define L2 meta training dataset path
L2_META_DATASET_PATH = META_SOURCE_DIRECTORY + "l2_meta_training_dataset/" + META_FILE

"""
This is the path for the meta training dataset
L3 meta models will store their prediction results in this dataset
"""

# Define L3 meta training prediction result dataset path
FINAL_PREDICTION_PATH = META_SOURCE_DIRECTORY + "final_prediction_dataset/" + META_FILE
