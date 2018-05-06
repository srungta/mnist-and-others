import os

#============= FOLDERS PATHS ========================================
__this_folder = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(__this_folder, '..', 'data'))

#--------- MNIST ---------
MNIST_DATA_DIR = os.path.join(DATA_DIR,'mnist')
MNIST_PICKLES_DIR = os.path.join(MNIST_DATA_DIR,'pickles')

#--------- NOT MNIST ---------
NOT_MNIST_DATA_DIR = os.path.join(DATA_DIR,'not_mnist')
NOT_MNIST_PICKLES_DIR = os.path.join(NOT_MNIST_DATA_DIR,'pickles')
NOT_MNIST_ZIPS_DIR = os.path.join(NOT_MNIST_DATA_DIR,'zips')
NOT_MNIST_IMAGES_DIR = os.path.join(NOT_MNIST_DATA_DIR,'images')

#====================== FILE PATHS ==================================
#----------- MNIST ---------
MNIST_FLATTENED_NORMALISED_PICKLE = os.path.join(MNIST_PICKLES_DIR,'mnist_normalised.pickle')
MNIST_PICKLE = os.path.join(MNIST_PICKLES_DIR,'mnist.pickle')

