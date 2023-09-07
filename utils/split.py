def split_dataset(df, train_ratio: int, start_index=0, range:int=None): 
    if train_ratio > 1 or train_ratio < 0:
        raise ValueError('train_ratio must be between 0 and 1')
    if start_index < 0 or start_index >= len(df):
        raise ValueError('start_index must be between 0 and len(df)')
    if range is not None and range <= 0:    
        raise ValueError('range must be greater than 0')
    if range is not None and start_index + range > len(df):
        raise ValueError('start_index + range must be less than len(df)')
    
    end_index = start_index + range if range is not None else len(df)
    selected = df.iloc[start_index:end_index]
    train_end_index = int(len(selected) * train_ratio)
    train = selected.iloc[:train_end_index]
    test = selected.iloc[train_end_index:]

    return train, test
