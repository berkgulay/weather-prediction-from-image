import numpy as np


size = "50"
train_data = np.load("../models"+size+"/train_data_concat1000.npy")
train_label = np.load("../models"+size+"/train_label_concat1000.npy")

train_data = np.array(train_data)
train_label = np.array(train_label)


validation_data = train_data[int(len(train_data)*4/5):]
validation_label = train_label[int(len(train_label)*4/5):]
train_data = train_data[0:int(len(train_data)*4/5)]
train_label = train_label[0:int(len(train_label)*4/5)]

np.save("data/"+size+"/train_data.npy", train_data)
np.save("data/"+size+"/train_label.npy",train_label)
np.save("data/"+size+"/validation_data.npy",validation_data)
np.save("data/"+size+"/validation_label.npy",validation_label)
