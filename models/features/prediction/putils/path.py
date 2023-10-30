import time


def generate_meta_archive_directory(layer: str) -> str:
    if layer != "l1" and layer != "l2" and layer != "final":
        raise Exception("Invalid layer")
    if layer == "final":
        path = (
            "models/features/source/"
            + "final_prediction_dataset/archive/"
            + str(round(time.time() * 1000))
        )
        return path
    path = (
        "models/features/source/"
        + layer
        + "_meta_training_dataset/archive/"
        + str(round(time.time() * 1000))
    )
    return path
