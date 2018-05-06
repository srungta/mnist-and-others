# imports
from __future__ import print_function
from IPython.display import display, Image
from six.moves import cPickle as pickle
from six.moves.urllib.request import urlretrieve
from sklearn.linear_model import LogisticRegression
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile

from constants import *
from commonconstants import NOT_MNIST_ZIPS_DIR, NOT_MNIST_IMAGES_DIR, NOT_MNIST_PICKLES_DIR
from file_helper import get_file_name, join_paths

np.random.seed(NUMPY_SEED)
last_percent_reported = None


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
      slow internet connections. Reports every 5% change in download progress.
      """
    global last_percent_reported
    percent = int(count*blockSize*100 / totalSize)
    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write('%s%%' % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
        last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
    dest_filename = os.path.join(NOT_MNIST_ZIPS_DIR, filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(
            DATASET_DOWNLOAD_URL + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception(
            'Failed to verify ' + dest_filename + '. Get using abrowser.'
        )
    return dest_filename


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(get_file_name(filename))[0])[0]  # remove .tar.gz
    root = join_paths(NOT_MNIST_IMAGES_DIR, root)
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' %
              (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(NOT_MNIST_IMAGES_DIR)
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != NUM_OF_CLASSES:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                NUM_OF_CLASSES, len(data_folders)))
    print(data_folders)
    return data_folders


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), IMAGE_SIZE, IMAGE_SIZE),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (imageio.imread(image_file).astype(float) -
                          PIXEl_DEPTh / 2) / PIXEl_DEPTh
            if image_data.shape != (IMAGE_SIZE, IMAGE_SIZE):
                raise Exception('Unexpected image shape: %s' %
                                str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except (IOError, ValueError) as e:
            print('Could not read:', image_file,
                  ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    NUM_OF_CLASSES = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, IMAGE_SIZE)
    train_dataset, train_labels = make_arrays(train_size, IMAGE_SIZE)
    vsize_per_class = valid_size // NUM_OF_CLASSES
    tsize_per_class = train_size // NUM_OF_CLASSES

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class+tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


def get_dataset_filenames():
    train_filename = maybe_download(NOT_MNIST_FILENAME_LARGE, 247336696)
    test_filename = maybe_download(NOT_MNIST_FILENAME_SMALL, 8458043)

    train_folders = maybe_extract(train_filename)
    test_folders = maybe_extract(test_filename)

    train_datasets = maybe_pickle(
        train_folders, MINIMUM_TRAIN_SAMPLES_PER_CLASS)
    test_datasets = maybe_pickle(test_folders, MINIMUM_TEST_SAMPLES_PER_CLASS)
    return train_datasets, test_datasets


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def main(train_size, valid_size, test_size, filename):
    train_datasets, test_datasets = get_dataset_filenames()
    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
        train_datasets, train_size, valid_size)
    _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
    pickle_file = os.path.join(NOT_MNIST_PICKLES_DIR, filename)

    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise


if __name__ == '__main__':
    # Large
    main(TRAINING_SIZE, VALIDATION_SIZE, TEST_SIZE, FINAL_DATASET_FILENAME)    
    # Small
    main(TRAINING_SIZE_SMALL, VALIDATION_SIZE_SMALL, TEST_SIZE_SMALL, FINAL_DATASET_FILENAME_SMALL)