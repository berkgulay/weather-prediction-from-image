#author Berk Gulay

from sklearn import svm
import numpy as np
from sklearn.model_selection import KFold
import random

# This function seperates into sub classes for getting accuracy for each classes
def separate_test_data(test_data, test_label):

    cloudy_data = []
    cloudy_label = []

    sunny_data = []
    sunny_label = []

    rainy_data = []
    rainy_label = []

    snowy_data = []
    snowy_label = []

    foggy_data = []
    foggy_label = []

    for i in range(len(test_data)):
        if int(test_label[i]) == 0:
            cloudy_data.append(test_data[i])
            cloudy_label.append(0)
        elif int(test_label[i]) == 1:
            sunny_data.append(test_data[i])
            sunny_label.append(1)
        elif int(test_label[i]) == 2:
            rainy_data.append(test_data[i])
            rainy_label.append(2)
        elif int(test_label[i]) == 3:
            snowy_data.append(test_data[i])
            snowy_label.append(3)
        elif int(test_label[i]) == 4:
            foggy_data.append(test_data[i])
            foggy_label.append(4)

    return (    np.array(cloudy_data),np.array(cloudy_label),
                np.array(sunny_data),np.array(sunny_label),
                np.array(rainy_data),np.array(rainy_label),
                np.array(snowy_data),np.array(snowy_label),
                np.array(foggy_data), np.array(foggy_label)
            )

#K-fold splits into 10 and shuffles the indexes
def k_fold(data,data_label,fold_num):
    kf = KFold(n_splits=fold_num,shuffle=True)
    kf.get_n_splits(data)

    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for train_index, test_index in kf.split(data):

        train_data, test_data= data[train_index], data[test_index]
        train_label, test_label = data_label[train_index], data_label[test_index]

    return (train_data,
            train_label,
            test_data,
            test_label)

def shuffle(train_data,train_label):
    temp = list(zip(train_data,train_label))
    random.shuffle(temp)

    return zip(*temp)


data = np.load('../DataSets/WarmthOfImage/SVM-DT/features_labels/features.npy')
data_label = np.load('../DataSets/WarmthOfImage/SVM-DT/features_labels/labels.npy')
data_label = np.reshape(data_label,(np.shape(data)[0],))

#Shuffle:
data , data_label = shuffle(data,data_label)

data = np.array(data)
data_label = np.array(data_label)

#train_data,train_label,test_data,test_label = k_fold(data,data_label,10)


#try gamma = 3 or 2 and 0.2 or 0.1 , kernel= polynomial with degree 3,5,7,10 coef0 as 0,10,100
#try C as 4 and 2 and 0.5
#try divided tol by 1/10
#try decision_function_shape as 'ovo'
#try class_weight as default
#try with no shrink (default with shrink change it)
svc = svm.SVC(kernel='rbf',gamma=0.5,C=1.2,class_weight='balanced',max_iter=500,decision_function_shape='ovr',tol=0.001,cache_size=1000,probability=True)
svc.fit(data[0:-1200],data_label[0:-1200])


# makes a list for each class seperately to get each accuracy of them
splitted_test = separate_test_data(data[-1200:], data_label[-1200:])
validation_data = [(splitted_test[t],splitted_test[t+1]) for t in range(0,10,2)]



num_of_matches = 0
num_test_data = 0
# get an accuracy in here
for j in range(len(validation_data)):
    features_for_class = validation_data[j][0]  # features of a test image
    labels_for_class = validation_data[j][1]  # label of a test image

    predicted = svc.predict(features_for_class)

    count_for_each = 0
    for i in range(len(predicted)):
        if predicted[i] == labels_for_class[i]:
            count_for_each += 1
            num_of_matches += 1
        num_test_data += 1
    print("Accuracy for", j, "class : ", count_for_each / len(predicted))
print("General Accuracy : ", num_of_matches / num_test_data)
print("0 is Cloudy\n1 is Sunny\n2 is Rainy\n3 is Snowy \n4 is Foggy\n")
