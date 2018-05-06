import os

#====================== FILE PATHS ==================================

FINAL_DATASET_FILENAME = 'notMNIST.pickle'
FINAL_DATASET_FILENAME_SMALL = 'notMNIST_small.pickle'
NOT_MNIST_FILENAME_LARGE = 'notMNIST_large.tar.gz'
NOT_MNIST_FILENAME_SMALL = 'notMNIST_small.tar.gz'

#============= URLS ========================================
DATASET_DOWNLOAD_URL = 'https://commondatastorage.googleapis.com/books1000/'

#============= IMAGE DETAILS ========================================

IMAGE_SIZE = 28  # Pixel width and height.
NUM_OF_CLASSES = 10
PIXEl_DEPTh = 255.0  # Number of levels per pixel.

#============= MISCELLANEOUS ========================================
NUMPY_SEED = 133

#============= DATA SIZE ========================================
MINIMUM_TEST_SAMPLES_PER_CLASS = 1800
MINIMUM_TRAIN_SAMPLES_PER_CLASS = 45000
TEST_SIZE = 1000
TRAINING_SIZE = 20000
VALIDATION_SIZE = 1000

TEST_SIZE_SMALL = 100
TRAINING_SIZE_SMALL = 2000
VALIDATION_SIZE_SMALL = 100

#============= HYPERPARAMETERS ========================================
LEARNING_RATE = 0.5
BATCH_SIZE = 128
