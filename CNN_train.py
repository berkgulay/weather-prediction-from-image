#author: Samet Kalkan

import os


import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.utils import np_utils
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import cv2

from keras import regularizers

np.random.seed(0)

def getAccuracyOfClass(validation_label, y):
    c = 0
    for i in range(len(y)):
        if y[i] == np.argmax(validation_label[i]):
            c += 1
    return c/len(y)

def seperateData(v_data, v_label, cl):
    seperatedData = []
    seperatedLabel = []
    for i in range(len(v_data)):
        if cl==int(v_label[i]):
            seperatedData.append(v_data[i])
            seperatedLabel.append(cl)
    return (np.array(seperatedData), np_utils.to_categorical(np.array(seperatedLabel), 5))

size = 50
train_data = np.load("../models"+str(size)+"/train_data_concat1000.npy")
train_label = np.load("../models"+str(size)+"/train_label_concat1000.npy")

train_data = np.array(train_data)
train_label = np.array(train_label)

print(train_data.shape,train_label.shape)


validation_data = train_data[int(len(train_data)*4/5):]
validation_label = train_label[int(len(train_label)*4/5):]
train_data = train_data[0:int(len(train_data)*4/5)]
train_label = train_label[0:int(len(train_label)*4/5)]

#normalization
train_data = train_data / 255.0
validation_data = validation_data / 255.0

print(len(train_data),len(train_label))
print(len(validation_data),len(validation_label))


train_data = train_data.reshape(train_data.shape[0], size, size, 3)
validation_data = validation_data.reshape(validation_data.shape[0], size, size, 3)

vd = [seperateData(validation_data,validation_label, i) for i in range(5)]




#number of class
num_classes = 5 #Cloudy,Sunny,Rainy,Snowy,Foggy

#for example if label is 4 converts it [0,0,0,0,1]
train_label = np_utils.to_categorical(train_label, num_classes)
validation_label = np_utils.to_categorical(validation_label, num_classes)


model = Sequential()

#convolutional layer with 5x5 32 filters and with relu activation function
#input_shape: shape of the each data
#kernel_size: size of the filter
#strides: default (1,1)
#activation: activation function such as "relu","sigmoid"
model.add(Conv2D(64, kernel_size=(5,5), strides=2,input_shape=(size, size,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
model.add(Conv2D(32, kernel_size=(3,3), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(Flatten())

#beginning of fully connected neural network.
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))




#model.add(Dense(50, activation='relu'))

# Add fully connected layer with a softmax activation function
model.add(Dense(num_classes, activation='softmax'))

# Compile neural network
model.compile(loss='categorical_crossentropy', # Cross-entropy
                optimizer='rmsprop', # Root Mean Square Propagation
                metrics=['accuracy']) # Accuracy performance metric


#begin traing the data
history = model.fit(train_data, # train data
            train_label, # label
            epochs=20, # Number of epochs
            verbose=2,
            batch_size=64)


y = model.predict_classes(validation_data,verbose=0)
acc = getAccuracyOfClass(validation_label, y)
print("General Accuracy for Validation Data:", acc)
print("-----------------------------")

for i in range(len(vd)):
    y = model.predict_classes(vd[i][0],verbose=0)
    acc = getAccuracyOfClass(vd[i][1], y)
    print("Accuracy for class " + str(i) + ": ", acc)
    print("-----------------------------")


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
