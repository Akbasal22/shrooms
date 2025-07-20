import pandas as pd
import numpy as np

def featurize_data(path):
    df = pd.read_csv(path)
    class_column = df['class']
    newdf = pd.DataFrame()
    newdf = pd.get_dummies(df.drop(columns=['class']))
    newdf['class'] = class_column
    newdf.to_csv('processed.csv')


def get_x_y(path):
    df = pd.read_csv(path)
    y = df['class'].to_numpy()
    x = df.drop(columns=['class']).to_numpy()


    folds = np.array_split(np.arange(8124), 10)

    splits = []
    for i in range(10):
        val_idx = folds[i]
        train_idx = np.concatenate(folds[:i] + folds[i+1:])
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]
        splits.append((x_train, y_train, x_val, y_val))
    
    return splits

featurize_data('./mushrooms.csv')


get_x_y('./processed.csv')
# lstm

