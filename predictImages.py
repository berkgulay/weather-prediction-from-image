
import cv2
import ImageDescriptor as id
from keras.models import load_model


#T.prepare_data_set("../berkfoto/deneme/0/","../berkfoto/cropped40berk/0/", 40)

#T.image_to_matrix("../berkfoto/cropped40berk/",)


#id.create_features("../berkfoto/deneme", "../berkfoto/cropped40berk", "../berkfoto/feature")

def predict_image_with_CNN(path):
    """
        predicts an image
        Returns:
            path of the image and its class
    """
    # "../berkfoto/cropped/0/0ea78fdd-70f7-4c46-b88a-e3d61a767dbe.jpg"
    model = load_model("modelsCNN/size100/trainedModelE20.h5")
    img = cv2.imread(path)
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    y = model.predict_classes(img, verbose=0)
    return (path, y[0])

def predict_image_with_RF(org_path, cropped_path, clf):
    """
        predicts an image
        Returns:
            path of the image and its class
    """
    feature = id.describe(org_path, cropped_path)
    feature = feature.reshape(1,feature.shape[0])
    y = clf.predict(feature)
    return (org_path, y[0])











