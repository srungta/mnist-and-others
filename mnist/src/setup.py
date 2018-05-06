from __future__ import print_function
from keras.datasets import mnist
import pickle

from constants import MNIST_FLATTENED_NORMALISED_PICKLE, MNIST_PICKLE
from file_helper import file_exists, read_from_pickle
from data_helper import flatten_dataset

def setup_flattened_normalised_mnist_pickle():
    
    print('Setting up normalised mnist pickle:')

    destination_file = MNIST_FLATTENED_NORMALISED_PICKLE
    if file_exists(destination_file) == True:
        print('MNIST normalised file already exists.')
        return
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = flatten_dataset(x_train)
    x_test = flatten_dataset(x_test)
    
    print('\t\tTraining data (x_train) : ', x_train.shape)
    print('\t\tTraining labels (y_train) : ', y_train.shape)
    print('\t\tTest data (y_train) : ', x_test.shape)    
    print('\t\tTest labels (y_test) : ', y_test.shape)

    dataset = {
        'x_train': x_train,
        'y_train':y_train,
        'x_test': x_test, 
        'y_test': y_test
        }
    with open(destination_file, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

def setup_mnist_pickle():
    print('Setting up raw mnist pickle:')

    destination_file = MNIST_PICKLE
    if file_exists(destination_file) == True:
        print('MNIST file already exists.')
        return
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    print('\t\tTraining data (x_train) : ', x_train.shape)
    print('\t\tTraining labels (y_train) : ', y_train.shape)
    print('\t\tTest data (y_train) : ', x_test.shape)    
    print('\t\tTest labels (y_test) : ', y_test.shape)
    
    dataset = {
        'x_train': x_train,
        'y_train':y_train,
        'x_test': x_test, 
        'y_test': y_test
        }
    with open(destination_file, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    setup_flattened_normalised_mnist_pickle()
    setup_mnist_pickle()