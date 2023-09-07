import pickle

def load_dataset():
    return pickle.load(open('../dump/df.p', 'rb'))