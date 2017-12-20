#author Berk Gulay , Mert Surucuoglu

import numpy as np
import cv2

def contrast():
    pass

def brightness(rgb_image):
    brightness_list = []
    for i in np.arange(len(rgb_image)):
        for j in np.arange(len(rgb_image[i])):
            pixel_r = rgb_image[i][j][0]
            pixel_g = rgb_image[i][j][1]
            pixel_b = rgb_image[i][j][2]

            brightness = (0.299 * pixel_r) + (0.587 * pixel_g) + (0.114 * pixel_b)
            brightness_list.append(brightness)

    return (sum(brightness_list)/len(brightness_list))


def haze():
    pass

def color_hist(rgb_image, numofbin = 256):
    histR, bins = np.histogram(rgb_image[:, 0], np.arange(0, numofbin + 1), density=True)
    histG, bins = np.histogram(rgb_image[:, 1], np.arange(0, numofbin + 1), density=True)
    histB, bins = np.histogram(rgb_image[:, 2], np.arange(0, numofbin + 1), density=True)
    hist = np.concatenate((histR, histG, histB), axis=0)
    return hist

def intensity_hist(image,white_threshold):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    white_pixels = 0
    black_pixels = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image.item(i, j) < white_threshold):
                black_pixels += 1
            else:
                white_pixels += 1

    return (white_pixels / (black_pixels + white_pixels))

def sharpness(image):
    image = image.convert('L')  # to grayscale
    array = np.asarray(image, dtype=np.int32)

    dx = np.diff(array)[1:, :]  # remove the first row
    dy = np.diff(array, axis=0)[:, 1:]  # remove the first column
    dnorm = np.sqrt(dx ** 2 + dy ** 2)
    sharpness = np.average(dnorm)

    return sharpness


