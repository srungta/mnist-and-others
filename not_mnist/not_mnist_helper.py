import numpy as np
import pickle
import os
from matplotlib import pyplot as plt
import time
from constants import *


def read_from_pickle(pickle_file):
    pickle_file = open(pickle_file, "rb")
    emp = pickle.load(pickle_file)
    pickle_file.close()
    return emp


def get_datasets(filename=FINAL_DATASET_FILENAME_SMALL):
    data_root = '.\data'
    pickle_file = os.path.join(data_root, filename)
    datasets = read_from_pickle(pickle_file)
    return datasets['train_dataset'], datasets['train_labels'], datasets['valid_dataset'], datasets['valid_labels'], datasets['test_dataset'], datasets['test_labels']


def flatten_dataset(dataset):
    shape = dataset.shape
    return np.reshape(dataset, (shape[0], (shape[1]*shape[2])))


def reformat(dataset, labels):
    flat_dataset = flatten_dataset(dataset)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(NUM_OF_CLASSES) == labels[:, None]).astype(np.float32)
    return flat_dataset.astype(np.float32), labels


def print_dataset_details(dataset):
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))


def get_random_image(dataset):
    depth = dataset.shape[0]
    index = np.random.randint(0, depth - 1)
    return dataset[index, :, :]


def show_image(image):
    plt.imshow(image)
    plt.show()


def get_overlaps(dataset1, dataset2):
    dataset1.flags.writeable = False
    dataset2.flags.writeable = False
    start = time.clock()
    set1 = set([hash(bytes(image)) for image in dataset1])
    set2 = set([hash(bytes(image)) for image in dataset2])
    overlaps = set1.intersection(set2)
    return overlaps, time.clock() - start


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])
