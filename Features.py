#author Berk Gulay , Mert Surucuoglu

import numpy as np
import cv2
import math

def contrast(rgb_image):
    d_primes = []
    b_primes = []

    for i in np.arange(len(rgb_image)):
        for j in np.arange(len(rgb_image[i])):
            d_primes.append(min((rgb_image[i][j][0],rgb_image[i][j][1],rgb_image[i][j][2])))
            b_primes.append(max((rgb_image[i][j][0], rgb_image[i][j][1], rgb_image[i][j][2])))

    max_b = max(b_primes)
    avg_d = sum(d_primes) / len(d_primes)
    avg_b = sum(b_primes) / len(b_primes)
    contrast = avg_d - avg_b
    normalized_contrast = (contrast) / (255)

    return (normalized_contrast,contrast,max_b,avg_d,avg_b)

def brightness(rgb_image):
    brightness_list = []
    for i in np.arange(len(rgb_image)):
        for j in np.arange(len(rgb_image[i])):
            pixel_r = rgb_image[i][j][0]
            pixel_g = rgb_image[i][j][1]
            pixel_b = rgb_image[i][j][2]

            brightness = (0.299 * pixel_r) + (0.587 * pixel_g) + (0.114 * pixel_b)
            brightness_list.append(brightness)

    brightness = sum(brightness_list)/len(brightness_list)
    normalized_brightness = (brightness) / (255)

    return normalized_brightness


def haze(contrast,max_b,avg_d,avg_b,lamb=1/3):
    A = (lamb * max_b) + ((1 - lamb) * avg_b)
    x1 = (A - avg_d) / A
    x2 = contrast / A

    haze = math.exp((-0.5 * ((5.1 * x1) + (2.9 * x2))) + 0.2461) #normalized value

    return haze


def color_hist(rgb_image, numofbin = 256):
    histR, bins = np.histogram(rgb_image[:, 0], np.arange(0, numofbin + 1), density=True)
    histG, bins = np.histogram(rgb_image[:, 1], np.arange(0, numofbin + 1), density=True)
    histB, bins = np.histogram(rgb_image[:, 2], np.arange(0, numofbin + 1), density=True)
    hist = np.concatenate((histR, histG, histB), axis=0)
    return hist

def intensity_hist(rgb_image,white_threshold):
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    white_pixels = 0
    black_pixels = 0
    for i in np.arange(rgb_image.shape[0]):
        for j in np.arange(rgb_image.shape[1]):
            if (rgb_image.item(i, j) < white_threshold):
                black_pixels += 1
            else:
                white_pixels += 1

    return (white_pixels / (black_pixels + white_pixels))

def sharpness(rgb_image):
    image = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2GRAY)  # to grayscale
    array = np.asarray(image, dtype=np.int32)

    dx = np.diff(array)[1:, :]  # remove the first row
    dy = np.diff(array, axis=0)[:, 1:]  # remove the first column
    dnorm = np.sqrt(dx ** 2 + dy ** 2)

    sharpness = np.average(dnorm)
    normalized_sharpness = sharpness / 9

    return normalized_sharpness
