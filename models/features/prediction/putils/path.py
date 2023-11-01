import time


def generate_meta_archive_directory_path(
    layer: str, folder_name: str = str(round(time.time() * 1000))
) -> str:
    if layer != "l1" and layer != "l2" and layer != "l3":
        raise Exception("Invalid layer")
    path = (
        "models/features/source/" + layer + "_prediction_dataset/archive/" + folder_name
    )
    return path
