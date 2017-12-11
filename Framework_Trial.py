#author: Samet Kalkan

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.utils import np_utils


train_data = np.load("../train_data.npy")
train_label = np.load("../train_label.npy")

#splits train_data with ratio 1/4
validation_data = train_data[int(len(train_data)*3/4):]
validation_label = train_label[int(len(train_data)*3/4):]

#normalization
train_data = train_data / 255.0
validation_data = validation_data / 255.0

#number of class
num_classes = len(np.unique(train_label))

#One-hot encode target
#for example if label is 5 converts it [0,0,0,0,1]
train_label = np_utils.to_categorical(train_label, num_classes)
validation_label = np_utils.to_categorical(validation_label, num_classes)

model = Sequential()

#convolutional layer with 5x5 32 filters and with relu activation function
#input_shape: shape of the each data
#kernel_size: size of the filter
#strides: default (1,1)
#activation: activation function such as "relu","sigmoid"
model.add(Conv2D(32, kernel_size=(5,5), input_shape=(200, 200, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

# Add layer to flatten input
model.add(Flatten())

#beginning of fully connected neural network.
# # Add fully connected layer of 10 units with a ReLU activation function
model.add(Dense(10, activation='relu'))

# Add dropout layer
model.add(Dropout(0.5))

# Add fully connected layer with a softmax activation function
model.add(Dense(num_classes, activation='softmax'))


# Compile neural network
model.compile(loss='categorical_crossentropy', # Cross-entropy
                optimizer='rmsprop', # Root Mean Square Propagation
                metrics=['accuracy']) # Accuracy performance metric


#begin traing the data
model.fit(train_data, # train data
            train_label, # label
            epochs=2, # Number of epochs
            batch_size=64, # Number of observations per batch
            validation_data=(validation_data, validation_label)) # Data for evaluation
