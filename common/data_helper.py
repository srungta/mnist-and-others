import numpy as np
import matplotlib.pyplot as plt
import itertools

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def flatten_dataset(dataset):
    shape = dataset.shape
    return np.reshape(dataset, (shape[0], (shape[1] * shape[2])))

def get_overlaps(dataset1, dataset2):
    dataset1.flags.writeable = False
    dataset2.flags.writeable = False
    start = time.clock()
    set1 = set([hash(bytes(image)) for image in dataset1])
    set2 = set([hash(bytes(image)) for image in dataset2])
    overlaps = set1.intersection(set2)
    return overlaps, time.clock() - start

def get_random_image(dataset):
    depth = dataset.shape[0]
    index = np.random.randint(0, depth - 1)
    return dataset[index, :, :]

def plot_confusion_matrix(cm, classes, normalise=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks,classes)
    if normalise:
        cm = cm.astype('float' / cm.sum(axis=1)[:, np.newaxis])
    thresh = cm.max() / 2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[0])):
        plt.text(j,i,cm[i,j], horizontalalignment='center', color ='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def print_dataset_details(dataset):
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))

def show_image(image):
    plt.imshow(image)
    plt.show()