#author: Samet Kalkan

import numpy as np
import random
import os
import PIL.ImageOps
from keras.preprocessing import image as image_utils

"""
This script reads all images in a directory given,
adds it to an array and labels each image, then saves those model. 
"""

imageRoot = "../images/"

train_data = []
train_label = []

#list of directory of classes in given path
classesDir = os.listdir(imageRoot)

for cls in classesDir:
    classList = os.listdir(imageRoot + cls + "/") #image list in a class directory
    for imageName in classList:

        img = image_utils.load_img(imageRoot + cls + "/" + imageName, target_size=(200,200)) #open an image
        img = PIL.ImageOps.invert(img) #inverts it
        img = image_utils.img_to_array(img) #converts it to array

        train_data.append(img)
        train_label.append(int(cls))

def shuffle(train_data,train_label):
    temp = list(zip(train_data,train_label))
    random.shuffle(temp)

    return zip(*temp)

def saveModel(path, model):
    np.save(path,model)


train_data,train_label = shuffle(train_data,train_label)
saveModel("../train_data.npy", np.array(train_data))
saveModel("../train_label.npy", np.array(train_label))


