import os
import sys
import shutil

DEFAULT_SIZE = "1250"
BASE = "./models/features/tuning/"
BASE_TEMPLATE = BASE + "template"
BASE_L1 = "./models/features/source/l1_prediction_dataset"
BASE_L2 = "./models/features/source/l2_prediction_dataset"
BASE_L3 = "./models/features/source/l3_prediction_dataset"
SRC_L1 = BASE_L1 + "/prediction_result_filtered.csv"
SRC_L2 = BASE_L2 + "/prediction_result_filtered.csv"
SRC_L3 = BASE_L3 + "/prediction_result_filtered.csv"


def create_dest_file_path(metric, layer):
    return BASE + metric + "/source/" + layer + ".csv"


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_archive_path(metric, layer):
    return (
        "./models/features/source/"
        + layer
        + "_prediction_dataset/archive/"
        + metric
        + "-1250/prediction_result_filtered.csv"
    )


def copy_folder(source, dest):
    shutil.copytree(source, dest)


def copy_file(source, dest):
    shutil.copy(source, dest)


def move_file_to_tuning():
    if len(sys.argv) < 2:
        print("Please specify metric")
        return
    size = ""
    if len(sys.argv) < 3:
        print("Using default size: " + DEFAULT_SIZE)
        size = DEFAULT_SIZE
    else:
        size = sys.argv[2]

    metric = sys.argv[1]
    TARGET = BASE + metric
    copy_folder(BASE_TEMPLATE, TARGET)
    print("Created folder: " + TARGET)

    copy_file(SRC_L1, create_dest_file_path(metric, "l1"))
    copy_file(SRC_L2, create_dest_file_path(metric, "l2"))
    copy_file(SRC_L3, create_dest_file_path(metric, "l3"))
    print("Copied files to tuning folder")

    create_folder(BASE_L1 + "/archive/" + metric + "-" + size)
    create_folder(BASE_L2 + "/archive/" + metric + "-" + size)
    create_folder(BASE_L3 + "/archive/" + metric + "-" + size)
    copy_file(SRC_L1, create_archive_path(metric, "l1"))
    copy_file(SRC_L2, create_archive_path(metric, "l2"))
    copy_file(SRC_L3, create_archive_path(metric, "l3"))
    print("Copied files to archive folder")

    print("Done")


move_file_to_tuning()
