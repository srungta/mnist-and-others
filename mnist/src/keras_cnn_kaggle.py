#!/usr/bin/env python

# Description of the module
''' Classifying mnist using keras + CNN
This module implements mnist classification 
using Keras as described in the kernel at 
https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6/code
'''
# Imports
## In-built modules

## Third-party modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

## Custom modules
from commonconstants import MNIST_KAGGLE_TRAIN, MNIST_KAGGLE_TEST
from data_helper import plot_confusion_matrix, print_dataset_details

# Global initialisations
np.random.seed(2)
sns.set(style='white', context='notebook', palette='deep')
reduce_size = False

# Data preparation
## Load data
train = pd.read_csv(MNIST_KAGGLE_TRAIN)
test = pd.read_csv(MNIST_KAGGLE_TEST)
test_size = 28000
# Reduce data size for prototype
if reduce_size:
    train = train.head(1000)
    test = test.head(100)
    test_size = 100


Y_train = train.label
X_train = train.drop(labels = ["label"], axis = 1)
del train

g = sns.countplot(Y_train)
print(Y_train.value_counts())

## Check for null and missing values
print(X_train.isnull().any().describe())
print(test.isnull().any().describe())

## Normalization
X_train = X_train / 255.0
test = test / 255.0

## Reshape
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

## Label encoding
Y_train = to_categorical(Y_train, num_classes=10)

## Split training and valdiation set
random_seed = 2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = random_seed)
# Modelling
## Define the model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters=32, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters=64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))

## Set the optimizer and annealer
optimiser = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer= optimiser, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience = 3, verbose=1, min_lr = 0.00001)
epochs = 2
batch_size = 64

## Data augmentation
datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False,featurewise_std_normalization=False,samplewise_std_normalization=False,
                             zca_whitening=False,rotation_range=10, zoom_range=0.1, width_shift_range=0.1, horizontal_flip=False, height_shift_range=0.1,
                             vertical_flip=False)

datagen.fit(X_train)

# Evaluate the model
# history = model.fit(X_train, Y_train, batch_size=batch_size, epochs= epochs,
# verbose = 2, validation_data=(X_val,Y_val))
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = batch_size), epochs= epochs,validation_data=(X_val,Y_val),verbose=2,
                              steps_per_epoch= X_train.shape[0] // batch_size, callbacks=[learning_rate_reduction])

## Training and validation curves
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label='Training loss')
ax[0].plot(history.history['val_loss'], color='r', label='validation loss', axes= ax[0])
legend = ax[0].legend(loc='best', shadow = True)

ax[1].plot(history.history['acc'], color='b', label='Training accuracy')
ax[1].plot(history.history['val_acc'], color='r', label='Validation accuracy')
legend = ax[1].legend(loc='best', shadow = True)    
plt.show()

## Confusion matrix

Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_val, axis=1)
confusion_mtx = confusion_matrix(Y_true,Y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes = range(10))

errors = (Y_pred_classes - Y_true !=0)
Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols,sharex=True, sharey= True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow(img_errors[error].reshape(28,28))
            ax[row,col].set_title('Predicted : {}\n True : {}'.format(pred_errors[error], obs_errors[error]))
            n +=1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

plt.show()
# Prediction and submition
## Predict and Submit results

results = model.predict(test)
results = np.argmax(results, axis = 1)
results = pd.Series(results, name='Label')
submission = pd.concat([pd.Series(range(1,test_size+1), name='ImageId'), results], axis = 1)
submission.to_csv('cnn_mnist.csv', index = False)
