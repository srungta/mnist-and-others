from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import time

from not_mnist_helper import print_dataset_details, read_from_pickle, get_random_image, show_image, get_datasets, get_overlaps, flatten_dataset
from not_mnist_setup import get_dataset_filenames


def check_image_sanity():
    # Check if images are still accessible
    train_datasets, test_datasets = get_dataset_filenames()
    data = read_from_pickle(test_datasets[0])
    print_dataset_details(data)
    show_image(get_random_image(data))
    del train_datasets
    del test_datasets


def get_overlaps_in_datasets(train_dataset, valid_dataset, test_dataset):
    # Check overlaps

    overlap, timetaken = get_overlaps(train_dataset, valid_dataset)
    print("Number of overlaps between training and validation dataset : ",
          len(overlap), " [computes in time :", timetaken, " ]")

    overlap, timetaken = get_overlaps(train_dataset, test_dataset)
    print("Number of overlaps between test and training dataset : ",
          len(overlap), " [computes in time :", timetaken, " ]")

    overlap, timetaken = get_overlaps(test_dataset, valid_dataset)
    print("Number of overlaps between test and validation dataset : ",
          len(overlap), " [computes in time :", timetaken, " ]")


def perform_logistics_regression(X_train, train_labels, X_test, test_labels):
    lg = LogisticRegression()
    start_train = time.clock()
    lg.fit(X_train, train_labels)
    train_end = time.clock() - start_train
    print("Training took ", train_end - start_train, " time.")
    y_pred = lg.predict(X_test)
    cmatrix = confusion_matrix(test_labels, y_pred)
    print("testing took ", time.clock() - train_end, " time.")
    print(cmatrix)


def main():
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_datasets()
    # check_image_sanity()
    # get_overlaps_in_datasets(train_dataset, valid_dataset, test_dataset)
    # print_dataset_details(train_dataset)
    # print_dataset_details(train_labels)
    # print_dataset_details(valid_dataset)
    # print_dataset_details(valid_labels)
    # print_dataset_details(test_dataset)
    # print_dataset_details(test_labels)

    X_train = flatten_dataset(train_dataset)
    print_dataset_details(X_train)

    X_test = flatten_dataset(test_dataset)
    print_dataset_details(X_test)

    perform_logistics_regression(X_train, train_labels, X_test, test_labels)


if __name__ == "__main__":
    main()
