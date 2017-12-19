#author Mert Surucuoglu, Berk Gulay

import numpy as np

def contrast():
    pass

def brightness():
    pass


def haze():
    pass

def color_hist(rgb_image, numofbin = 256):
    histR, bins = np.histogram(rgb_image[:, 0], np.arange(0, numofbin + 1), density=True)
    histG, bins = np.histogram(rgb_image[:, 1], np.arange(0, numofbin + 1), density=True)
    histB, bins = np.histogram(rgb_image[:, 2], np.arange(0, numofbin + 1), density=True)
    hist = np.concatenate((histR, histG, histB), axis=0)
    return hist

def intensity_hist():
    pass

def sharpness(image):
    image = image.convert('L')  # to grayscale
    array = np.asarray(image, dtype=np.int32)

    dx = np.diff(array)[1:, :]  # remove the first row
    dy = np.diff(array, axis=0)[:, 1:]  # remove the first column
    dnorm = np.sqrt(dx ** 2 + dy ** 2)
    sharpness = np.average(dnorm)

    return sharpness





