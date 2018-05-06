from file_helper import read_from_pickle, file_exists
from constants import MNIST_FLATTENED_NORMALISED_PICKLE, MNIST_PICKLE
from setup import setup_flattened_normalised_mnist_pickle, setup_mnist_pickle

def get_mnist_data(flattened = True):
    if flattened:
        if(file_exists(MNIST_FLATTENED_NORMALISED_PICKLE) == False):
            setup_flattened_normalised_mnist_pickle()
        dataset = read_from_pickle(MNIST_FLATTENED_NORMALISED_PICKLE)
        return dataset['x_train'],dataset['y_train'],dataset['x_test'],dataset['y_test']
    else:
        if(file_exists(MNIST_PICKLE) == False):
            setup_mnist_pickle()
        dataset = read_from_pickle(MNIST_PICKLE)
        return dataset['x_train'],dataset['y_train'],dataset['x_test'],dataset['y_test']

