import numpy as np

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

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

def print_dataset_details(dataset):
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))

def show_image(image):
    plt.imshow(image)
    plt.show()