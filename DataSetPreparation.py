#author Berk Gulay

import ImageResizer as rs
import os

#root directory for source images(which will be cropped)
path = 'C:/Users/Berk/Desktop/BerkG/ComputerEngineering/Semester5/GitHub/DataSets/WarmthOfImage/0/'
#root directory as destination to save cropped images(Prepared images will be saved in here)
dest = 'C:/Users/Berk/Desktop/BerkG/ComputerEngineering/Semester5/GitHub/DataSets/WarmthOfImage/cropped/0'

for filename in os.listdir(path):
    rs.resize_image(200,  #crop size for all images (just change it to define crop size)
                    path+filename,
                    dest,
                    filename)