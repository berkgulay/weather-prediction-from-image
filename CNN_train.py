#author: Samet Kalkan

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.utils import np_utils

from keras import regularizers

np.random.seed(0)

def seperateData(v_data, v_label, cl):
    seperatedData = []
    seperatedLabel = []
    for i in range(len(v_data)):
        if cl==int(v_label[i]):
            seperatedData.append(v_data[i])
            seperatedLabel.append(cl)
    return (np.array(seperatedData), np_utils.to_categorical(np.array(seperatedLabel), 5))

size = 50
train_data = np.load("data/"+str(size)+"/train_data.npy")
train_label = np.load("data/"+str(size)+"/train_label.npy")

#normalization
train_data = train_data / 255.0

train_data = train_data.reshape(train_data.shape[0], size, size, 3)

#number of class
num_classes = 5 #Cloudy,Sunny,Rainy,Snowy,Foggy

#for example if label is 4 converts it [0,0,0,0,1]
train_label = np_utils.to_categorical(train_label, num_classes)

model = Sequential()

#convolutional layer with 5x5 32 filters and with relu activation function
#input_shape: shape of the each data
#kernel_size: size of the filter
#strides: default (1,1)
#activation: activation function such as "relu","sigmoid"
model.add(Conv2D(32, kernel_size=(1,1),input_shape=(size, size,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

#beginning of fully connected neural network.
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))

# Add fully connected layer with a softmax activation function
model.add(Dense(num_classes, activation='softmax'))

# Compile neural network
model.compile(loss='categorical_crossentropy', # Cross-entropy
                optimizer='rmsprop', # Root Mean Square Propagation
                metrics=['accuracy']) # Accuracy performance metric


#begin traing the data
history = model.fit(train_data, # train data
            train_label, # label
            epochs=100, # Number of epochs
            verbose=2,
            batch_size=64)

#model.save_weights("modelsCNN/trainedModel.h5",overwrite=True)
model.save("modelsCNN/trainedModel.h5",overwrite=True)
