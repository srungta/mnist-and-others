import numpy as np

def flatten_dataset(dataset):
    shape = dataset.shape
    return np.reshape(dataset, (shape[0], (shape[1] * shape[2])))