import numpy as np
import pandas as pd


# Define functions
def compute_rmse(actual, predicted):
    return np.sqrt(((actual - predicted) ** 2).mean())


def compute_mape(actual, predicted):
    return 100 * np.mean(np.abs((actual - predicted) / actual))


def compute_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))


def load_data(
    layer: str, datafile: str, archived: str | int = None, last_appearance: int = None
):
    dir = (
        "../source/" + layer + "_prediction_dataset/" + datafile
        if archived is None
        else "../source/"
        + layer
        + "_prediction_dataset/archive/"
        + str(archived)
        + "/"
        + datafile
    )
    df = pd.read_csv(dir)
    df["Time"] = pd.to_datetime(df["Time"])
    df["FormattedTime"] = df["Time"].dt.strftime("%H:%M:%S")
    df = df.sort_values(by="FormattedTime")
    df = df.set_index("FormattedTime")
    # Reduce size of appearance
    if last_appearance is not None:
        df = df.tail(last_appearance)
    return df


def load_data_from_tuned_folder(layer: str, last_appearance: int = None):
    dir = "./source/" + layer + ".csv"
    df = pd.read_csv(dir)
    df["Time"] = pd.to_datetime(df["Time"])
    df["FormattedTime"] = df["Time"].dt.strftime("%H:%M:%S")
    df = df.sort_values(by="FormattedTime")
    df = df.set_index("FormattedTime")
    # Reduce size of appearance
    if last_appearance is not None:
        df = df.tail(last_appearance)
    return df
