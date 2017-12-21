#author Berk Gulay

from sklearn import svm
import numpy as np
from sklearn.model_selection import KFold

data = np.load('../DataSets/WarmthOfImage/SVM-DT/features_labels/features.npy')
data_label = np.load('../DataSets/WarmthOfImage/SVM-DT/features_labels/labels.npy')
data_label = np.reshape(data_label,(np.shape(data)[0],))

# This function seperates into sub classes for getting accuracy for each classes
def separate_data(_test_data, _test_label, cl):
    separated_data = []
    separated_label = []
    for i in range(len(_test_data)):
        if cl == int(_test_label[i]):
            separated_data.append(_test_data[i])
            separated_label.append(cl)
    return np.array(separated_data), np.array(separated_label)

#K-fold splits into 10 and shuffles the indexes
kf = KFold(n_splits=10,shuffle=True)
kf.get_n_splits(data)
for train_index, test_index in kf.split(data):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train_data, test_data= data[train_index], data[test_index]
    train_label, test_label = data_label[train_index], data_label[test_index]


#try gamma = 3 or 2 and 0.2 or 0.1 , kernel= polynomial with degree 3,5,7,10 coef0 as 0,10,100
#try C as 4 and 2 and 0.5
#try divided tol by 1/10
#try decision_function_shape as 'ovo'
#try class_weight as default
#try with no shrink (default with shrink change it)
svc = svm.SVC(kernel='rbf',gamma=0.5,C=1.2,class_weight='balanced',max_iter=500,decision_function_shape='ovr',tol=0.001,cache_size=1000,probability=True)

svc.fit()

predictions = svc.predict()
