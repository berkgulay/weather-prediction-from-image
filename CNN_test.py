#author: Samet Kalkan

import numpy as np
from keras.utils import np_utils
from keras.models import load_model



"""
    returns accuracy of given label
    validation label: expected
    y: predicted label
"""
def getAccuracyOfClass(validation_label, y):
    c = 0
    for i in range(len(y)):
        if y[i] == np.argmax(validation_label[i]):
            c += 1
    return c/len(y)

"""
    seperates validation data and label according to class no
    cl: indicates which labels will be seperated.
    returns an array that stores [val_data,val_label] in each index for each class.
"""
def separateData(v_data, v_label):
    vd=[ [[],[]] for i in range(5) ]
    for i in range(len(v_data)):
        cl=int(v_label[i])
        vd[cl][0].append(v_data[i])
        vd[cl][1].append(cl)
    for i in range(5):
        vd[i][0] = np.array(vd[i][0])
        vd[i][1] = np.array(np_utils.to_categorical(np.array(vd[i][1])))
    return vd



validation_data = np.load("../concat100/validation_data.npy")
validation_label = np.load("../concat100/validation_label.npy")

#normalization
validation_data = validation_data / 255.0


# each index stores a list which stores validation data and its label according to index no
# vd[0] = [val,lab] for class 0
# vd[1] = [val,lab] for class 1 and so on
vd = separateData(validation_data,validation_label)

#number of class
num_classes = 5 #Cloudy,Sunny,Rainy,Snowy,Foggy

#for example if label is 4 converts it [0,0,0,0,1]
validation_label = np_utils.to_categorical(validation_label, num_classes)


#loads trained model and architecture
model = load_model("modelsCNN/size100/trainedModelE40.h5")


#-------predicting part-------
y = model.predict_classes(validation_data,verbose=0)
acc = getAccuracyOfClass(validation_label, y)
print("General Accuracy for Validation Data:", acc)
print("-----------------------------")

for i in range(len(vd)):
    v_data = vd[i][0]
    v_label = vd[i][1]
    y = model.predict_classes(v_data,verbose=0)
    acc = getAccuracyOfClass(v_label, y)
    print("Accuracy for class " + str(i) + ": ", acc)
    print("-----------------------------")

