import pickle

def load(path: str):
    return pickle.load(open(path, 'rb'))
