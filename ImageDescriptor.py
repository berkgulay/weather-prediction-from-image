#author: Berk Gulay

import cv2

def describe(image):
    image = cv2.imread('../DataSets/WarmthOfImage/train/3/79734660.jpg', cv2.IMREAD_COLOR)