
import numpy as np
import random

def shuffle(train_data,train_label):
    temp = list(zip(train_data,train_label))
    random.shuffle(temp)
    return zip(*temp)

def concanate():
    train_data = np.load("../models100/train_data8.npy")[:1000]
    train_label = np.load("../models100/train_label8.npy")[:1000]


    tempData = np.load("../models/train_data9.npy")[:1000]
    tempLabel = np.load("../models/train_label9.npy")[:1000]

    train_data = np.concatenate((train_data, tempData), axis=0)
    train_label = np.concatenate((train_label, tempLabel), axis=0)

    tempData = np.load("../models/train_data.npy")
    tempLabel = np.load("../models/train_label.npy")

    train_data = np.concatenate((train_data, tempData), axis=0)
    train_label = np.concatenate((train_label, tempLabel), axis=0)

    tempData = None
    tempLabel = None
    train_data, train_label = shuffle(train_data, train_label)
    np.save("../models/train_data_concat1000.npy", train_data)
    np.save("../models/train_label_concat1000.npy", train_label)

concanate()