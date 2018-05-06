from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

from commonconstants import MNIST_FLATTENED_NORMALISED_PICKLE
from file_helper import read_from_pickle
from mnist_helper import get_mnist_data

# HYPERPARAMETERS
epochs = 10
encoding_dim = 32
batch_size = 256
train_size = 6000
test_size = 1000

# SET UP MODELS
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# SET UP DATA

x_train ,_ , x_test ,_ = get_mnist_data(True)
print(x_train.shape)
print(x_test.shape)

x_train = x_train[:train_size]
x_test = x_test[:test_size]

print(x_train.shape)
print(x_test.shape)

# TRAINING

autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size,
                shuffle=True, validation_data=(x_test, x_test))


encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


# VISUALIZATION

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
