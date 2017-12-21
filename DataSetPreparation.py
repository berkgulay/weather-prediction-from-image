#author Berk Gulay

import ImageResizer as rs
import os

#root directory for source images(which will be cropped)
path = '../train/1/'
#root directory as destination to save cropped images(Prepared images will be saved in here)
dest = '../cropped100/1'

for filename in os.listdir(path):
    rs.resize_image(100,  #crop size for all images (just change it to define crop size)
                    path+filename,
                    dest,
                    filename)
