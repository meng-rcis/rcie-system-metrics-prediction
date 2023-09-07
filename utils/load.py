import pickle

def load_dataset(path: str):
    return pickle.load(open(path, 'rb'))

def load_features(path: str):
    return pickle.load(open(path, 'rb'))

def load_labels(path: str):
    return pickle.load(open(path, 'rb'))