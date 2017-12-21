#author: Berk Gulay

import cv2
import Features
import numpy as np

def describe(image_path,cropped_image_path,cont=True,bright=True,haze=True,sharpness=True,color_hist=True,intensity_hist=True,white_thresh = 175):
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
