#author: Samet Kalkan

import os


import numpy as np
from keras.utils import np_utils
from keras.models import load_model


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

size = "50"
validation_data = np.load("data/"+size+"/validation_data.npy")
validation_label = np.load("data/"+size+"/validation_label.npy")

#normalization
validation_data = validation_data / 255.0

vd = [seperateData(validation_data,validation_label, i) for i in range(5)]


#number of class
num_classes = 5 #Cloudy,Sunny,Rainy,Snowy,Foggy

#for example if label is 4 converts it [0,0,0,0,1]
validation_label = np_utils.to_categorical(validation_label, num_classes)


model = load_model("modelsCNN/trainedModel.h5")

y = model.predict_classes(validation_data,verbose=0)
acc = getAccuracyOfClass(validation_label, y)
print("General Accuracy for Validation Data:", acc)
print("-----------------------------")

for i in range(len(vd)):
    y = model.predict_classes(vd[i][0],verbose=0)
    acc = getAccuracyOfClass(vd[i][1], y)
    print("Accuracy for class " + str(i) + ": ", acc)
    print("-----------------------------")

"""
plt.plot(y.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""
h = model.evaluate(validation_data,validation_label)