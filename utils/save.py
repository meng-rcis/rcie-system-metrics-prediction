import pickle

def save_dataset(df, path='../dump/df.p'):
    pickle.dump(df, open(path, 'wb'))
