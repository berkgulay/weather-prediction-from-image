#author: Samet Kalkan

import numpy as np
import os
import PIL.ImageOps
from keras.preprocessing import image as image_utils

"""
This script does that reads all images in a directory given,
adds it to an array and labels each image, then saves those model. 
"""

imageRoot = "../images/"

train_data = []
train_label = []

#list of directory of classes in given path
classesDir = os.listdir(imageRoot)

i = 0
for cls in classesDir:
    classList = os.listdir(imageRoot + cls + "/") #image list in a class directory
    for imageName in classList:

        img = image_utils.load_img(imageRoot + cls + "/" + imageName, target_size=(200,200)) #open an image
        img = PIL.ImageOps.invert(img) #inverts it
        img = image_utils.img_to_array(img) #converts it to array

        train_data.append(img)
        train_label.append(i)
    i+=1

train_data = np.array(train_data)


def saveModel(path, model):
    np.save(path,model)

saveModel("../train_data.npy", train_data)
saveModel("../train_label.npy", np.array(train_label))


