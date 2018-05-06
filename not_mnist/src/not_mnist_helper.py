import numpy as np
import pickle
import os
from matplotlib import pyplot as plt
import time

from constants import *
from commonconstants import NOT_MNIST_PICKLES_DIR
from file_helper import read_from_pickle
from data_helper import flatten_dataset

def get_datasets(filename=FINAL_DATASET_FILENAME_SMALL):
    data_root = NOT_MNIST_PICKLES_DIR
    pickle_file = os.path.join(data_root, filename)
    datasets = read_from_pickle(pickle_file)
    return datasets['train_dataset'], datasets['train_labels'], datasets['valid_dataset'], datasets['valid_labels'], datasets['test_dataset'], datasets['test_labels']


def reformat(dataset, labels):
    flat_dataset = flatten_dataset(dataset)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(NUM_OF_CLASSES) == labels[:, None]).astype(np.float32)
    return flat_dataset.astype(np.float32), labels

