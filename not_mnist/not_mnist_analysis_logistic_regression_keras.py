from not_mnist_helper import get_datasets
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

batch_size = 120
num_classes = 10
epochs = 2


def main():
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_datasets()
    model = Sequential()
    model.add(Dense(10, activation='softmax', input_shape=valid_dataset.shape))
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(valid_dataset, valid_labels, epochs=epochs,
              validation_data=(test_dataset, test_labels))
    score = model.evaluate(test_dataset, test_labels, verbose=1)
    print('Test loss : ', score[0])
    print('Test accuracy : ', score[1])


if __name__ == "__main__":
    main()
    pass
