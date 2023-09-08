import pickle

def save_dataset(df):
    pickle.dump(df, open('../dump/df.p', 'wb'))
