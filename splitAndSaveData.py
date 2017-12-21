#author: Samet Kalkan

import numpy as np

"""
    splits all data into train data and validation data
    then saves them
"""


size = "50"

data = np.load("../models"+size+"/train_data_concat1000.npy")
label = np.load("../models"+size+"/train_label_concat1000.npy")
#----spliting part---
#4/5 of data is train data and rest is validation
validation_data = data[int(len(data)*4/5):]
validation_label = label[int(len(label)*4/5):]
train_data = data[0:int(len(data)*4/5)]
train_label = label[0:int(len(label)*4/5)]

np.save("data/"+size+"/train_data.npy", train_data)
np.save("data/"+size+"/train_label.npy", train_label)
np.save("data/"+size+"/validation_data.npy", validation_data)
np.save("data/"+size+"/validation_label.npy", validation_label)





