#author Berk Gulay

from sklearn import svm
import numpy as np
from sklearn.model_selection import KFold
import tools as T



data = np.load('../features/features.npy')
data_label = np.load('../features/labels.npy')
data_label = np.reshape(data_label, (np.shape(data)[0],))

data, data_label = T.shuffle(data, data_label)
data = np.array(data)
data_label = np.array(data_label)

#train_data,train_label,test_data,test_label = k_fold(data,data_label,10)

svc = svm.SVC(kernel='rbf',gamma=0.001,C=1.2,class_weight='balanced',max_iter=500,decision_function_shape='ovr',tol=0.001,cache_size=1000,probability=True)



split_size = 10
kf = KFold(n_splits=split_size, shuffle=True)
kf.get_n_splits(data)

overallAccuracies = np.zeros(5)
generalOverallAccuracy = 0

k=0
for train_index, test_index in kf.split(data):
    # print("TRAIN:", train_index, "TEST:", test_index)
    train_data, test_data = data[train_index], data[test_index]
    train_label, test_label = data_label[train_index], data_label[test_index]

    # puts all features into a single array
    train_data_reshaped = train_data.reshape((len(train_data), -1))
    test_data_reshaped = test_data.reshape((len(test_data), -1))
    svc.fit( train_data_reshaped, train_label )
    # Train the Classifier to take the training features and learn how they relate to the training(the species)
    # makes a list to get each accuracy
    separated_test_data = T.separate_data(test_data_reshaped,test_label)
    # to get a accuracy
    k+=1
    print("Fold:", k)
    for i in range(len(separated_test_data)):
        v_data = separated_test_data[i][0]
        v_label = separated_test_data[i][1]
        y = svc.predict(v_data)
        acc = T.get_accuracy_of_class(v_label, y)
        overallAccuracies[i]+=acc
        print("Accuracy for class " + T.classes[i] + ": ", acc)
    generalAccuracy =  T.get_accuracy_of_class(test_label,svc.predict(test_data_reshaped))
    generalOverallAccuracy+= generalAccuracy
    print("General Accuracy:", generalAccuracy)
    print("---------------------------------------------")

print("OVERALL ACCURACY\n-------------------------------------------------")
for i in range(len(overallAccuracies)):
    print("Overall Accuracy For", T.classes[i], overallAccuracies[i]/split_size)
print("Overall Accuracy", generalOverallAccuracy/split_size)