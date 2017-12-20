#author: Samet Kalkan, Berk Gulay

import numpy as np
import random
import os
import PIL.ImageOps
from keras.preprocessing import image as image_utils

"""
This script reads all images in a directory given,
adds it to an array and labels each image, then saves those model. 
"""

def shuffle(train_data,train_label):
    temp = list(zip(train_data,train_label))
    random.shuffle(temp)

    return zip(*temp)

def saveModel(path, model):
    np.save(path,model)


imageRoot = "../DataSets/WarmthOfImage/cropped/" #Change this root directory of images to create model for them
batch_size_for_models = 5000 #5000 sized batch models


train_data = []
train_label = []

#list of directory of classes in given path
classesDir = os.listdir(imageRoot)

counter = 0 #counter to check size of batch, if 5000 save model and flush lists
fc = 0 #file counter to name models
for cls in classesDir:
    classList = os.listdir(imageRoot + cls + "/") #image list in a class directory
    for imageName in classList:
        counter += 1

        img = image_utils.load_img(imageRoot + cls + "/" + imageName, target_size=(200,200)) #open an image
        img = PIL.ImageOps.invert(img) #inverts it
        img = image_utils.img_to_array(img) #converts it to array

        train_data.append(img)
        train_label.append(int(cls))

        if(counter == batch_size_for_models):
            train_data, train_label = shuffle(train_data, train_label)
            saveModel("../DataSets/WarmthOfImage/models/cropped/train_data"+ str(fc) +".npy", np.array(train_data)) #model root to save image models(image)
            saveModel("../DataSets/WarmthOfImage/models/cropped/train_label"+ str(fc) +".npy", np.array(train_label)) #model root to save image models(label))

            train_data = []
            train_label = []
            fc += 1
            counter = 0


#rest of images which stays in list , add their models to model root lastly
if(len(train_data)!=0):
    train_data,train_label = shuffle(train_data,train_label)
    saveModel("../DataSets/WarmthOfImage/models/cropped/train_data.npy", np.array(train_data)) #model root to save image models(image)
    saveModel("../DataSets/WarmthOfImage/models/cropped/train_label.npy", np.array(train_label)) #model root to save image models(label)


