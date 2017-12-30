#author: Berk Gulay

import cv2
import Features
import numpy as np
import os

def describe(image_path,cropped_image_path,cont=True,bright=True,haze=True,sharpness=True,color_hist=True,intensity_hist=True,white_thresh=175):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cropped_image_rgb = cv2.imread(cropped_image_path, cv2.IMREAD_COLOR)

    description_array = np.reshape(cropped_image_rgb,(np.size(cropped_image_rgb)))

    (norm_cont,c,max_b,avg_d,avg_b) = Features.contrast(image)
    if(cont == True):
        description_array = np.append(description_array,[norm_cont],axis=0)
    if (bright == True):
        bright_value = Features.brightness(image)
        description_array = np.append(description_array, [bright_value], axis=0)
    if (haze == True):
        haze_value = Features.haze(c,max_b,avg_d,avg_b)
        description_array = np.append(description_array, [haze_value], axis=0)
    if (sharpness == True):
        sharp_value = Features.sharpness(image)
        description_array = np.append(description_array, [sharp_value], axis=0)
    if (color_hist == True):
        color_h = Features.color_hist(image)
        description_array = np.concatenate((description_array, color_h), axis=0)
    if (intensity_hist == True):
        intensity = Features.intensity_hist(image,white_thresh)
        description_array = np.append(description_array, [intensity], axis=0)

    return description_array


def create_features(standart_image_dir,cropped_image_dir,dest,cont=True,bright=True,haze=True,sharpness=True,color_hist=True,intensity_hist=True,white_thresh = 175):
    img_features = []
    img_labels = []
    batch_counter = 0
    fc = 1

    dirs = os.listdir(standart_image_dir)
    i=0
    for dir_name in dirs:
        class_images = os.listdir(standart_image_dir+'/'+dir_name+'/')
        for image_name in class_images:
            i+=1
            print(i)
            if(image_name[-4:]=='.jpg'): # if image extension is jpg
                std_img_path = standart_image_dir+'/'+dir_name+'/'+image_name
                cropped_img_path = cropped_image_dir+'/'+dir_name+'/'+image_name

                img_feature = describe(std_img_path,cropped_img_path,cont,bright,haze,sharpness,color_hist,intensity_hist,white_thresh)
                img_label = [int(dir_name)]

                img_features.append(img_feature)
                img_labels.append(img_label)
                batch_counter +=1

            if(batch_counter==500 or (dir_name==dirs[-1] and image_name==class_images[-1])):
                img_features = np.array(img_features)
                img_labels = np.array(img_labels)

                np.save(dest + '/features'+str(fc)+'.npy', img_features)
                np.save(dest + '/labels'+str(fc)+'.npy', img_labels)

                batch_counter = 0
                img_features = []
                img_labels = []
                fc += 1

    return (np.shape(img_features),np.shape(img_labels))
