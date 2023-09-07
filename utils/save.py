import pickle

def save_dataset(df):
    pickle.dump(df, open('../dump/df.p', 'wb'))

def save_features(features):
    pickle.dump(features, open('../dump/features.p', 'wb'))

def save_labels(labels):
    pickle.dump(labels, open('../dump/labels.p', 'wb'))