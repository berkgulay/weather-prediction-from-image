#author Berk Gulay

import Resize_Image as rs
import os

path = 'C:/Users/Berk/Desktop/BerkG/ComputerEngineering/Semester5/GitHub/DataSets/WarmthOfImage/0/'
dest = 'C:/Users/Berk/Desktop/BerkG/ComputerEngineering/Semester5/GitHub/DataSets/WarmthOfImage/cropped/0'

for filename in os.listdir(path):
    rs.resize_image(200,
                    path+filename,
                    dest,
                    filename)



