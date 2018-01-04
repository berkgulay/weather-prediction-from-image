
import cv2
import ImageDescriptor as id
from keras.models import load_model
from keras.preprocessing import image as image_utils
from PIL import Image
import PIL.ImageOps
import tools as T

#for i in ["0","1","2","3"]:
#T.prepare_data_set("../cektiklerim/test/", "../cektiklerim/cropped100/", 100)

#T.image_to_matrix("../myNewSet/cropped/", "../myNewSet/model/", 200)


#id.create_features("../berkfoto/deneme", "../berkfoto/cropped40berk", "../berkfoto/feature")

def predict_image_with_CNN(path, model):
    """
        predicts an image
        Returns:
            path of the image and its class
    """
    img = image_utils.load_img(path, target_size=(100, 100))  # open an image
    img = PIL.ImageOps.invert(img)  # inverts it
    img = image_utils.img_to_array(img)  # converts it to array
    img = img/255.0
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

    y = model.predict_classes(img, verbose=0, batch_size=1)
    return path, T.classes[y[0]]

def predict_image_with_RF(org_path, cropped_path, clf):
    """
        predicts an image
        Returns:
            path of the image and its class
    """
    feature = id.describe(org_path, cropped_path)
    y = clf.predict(feature)
    return (org_path, y[0])



