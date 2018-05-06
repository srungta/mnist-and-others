import os

# Folder paths
__this_folder = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(__this_folder, '..', 'data'))
MNIST_DATA_DIR = os.path.join(DATA_DIR,'mnist')
MNIST_PICKLES_DIR = os.path.join(MNIST_DATA_DIR,'pickles')
NOT_MNIST_DATA_DIR = os.path.join(DATA_DIR,'not_mnist')
NOT_MNIST_PICKLES_DIR = os.path.join(NOT_MNIST_DATA_DIR,'pickles')

# File paths
MNIST_NORMALISED_PICKLE = os.path.join(MNIST_PICKLES_DIR,'mnist_normalised.pickle')

