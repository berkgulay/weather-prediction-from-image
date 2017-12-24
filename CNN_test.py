#author: Samet Kalkan

import numpy as np
from keras.utils import np_utils
from keras.models import load_model
import tools as T


validation_data = np.load("../concat100/validation_data.npy")
validation_label = np.load("../concat100/validation_label.npy")

# normalization
validation_data = validation_data / 255.0


# each index stores a list which stores validation data and its label according to index no
# vd[0] = [val,lab] for class 0
# vd[1] = [val,lab] for class 1 and so on
vd = T.separate_data(validation_data, validation_label)

# number of class
num_classes = 5  # Cloudy,Sunny,Rainy,Snowy,Foggy

# for example if label is 4 converts it [0,0,0,0,1]
validation_label = np_utils.to_categorical(validation_label, num_classes)


# loads trained model and architecture
model = load_model("modelsCNN/size100/trainedModelE20.h5")


# -------predicting part-------
y = model.predict_classes(validation_data, verbose=0)
acc = T.get_accuracy_of_class(T.binary_to_class(validation_label), y)
print("General Accuracy for Validation Data:", acc)
print("-----------------------------")

for i in range(len(vd)):
    v_data = vd[i][0]
    v_label = vd[i][1]
    y = model.predict_classes(v_data,verbose=0)
    acc = T.get_accuracy_of_class(v_label, y)
    print("Accuracy for class " + str(i) + ": ", acc)
    print("-----------------------------")





