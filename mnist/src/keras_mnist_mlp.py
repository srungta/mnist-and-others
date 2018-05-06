import keras
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from time import time

from mnist_helper import get_mnist_data

batch_size = 120
num_classes = 10
epochs = 2
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

x_train, y_train, x_test, y_test = get_mnist_data(True)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[tensorboard])

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss : ', score[0])
print('Test accuracy : ', score[1])
