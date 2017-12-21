from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
import numpy as np

data = np.load("train_data_concat1000.npy")
data_label = np.load("train_label_concat1000.npy")

""""test_data = data[int(len(data)*4/5):]
test_label = data_label[int(len(data_label)*4/5):]
train_data = data[0:int(len(data)*4/5)]
train_label = data_label[0:int(len(data_label)*4/5)]"""""

kf = KFold(n_splits=10,shuffle=True)
kf.get_n_splits(data)
for train_index, test_index in kf.split(data):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train_data, test_data= data[train_index], data[test_index]
    train_label, test_label = data_label[train_index], data_label[test_index]

train_data = train_data.reshape((len(train_data), -1))
test_data = test_data.reshape((len(test_data), -1))

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(train_data, train_label)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
print(clf.feature_importances_)
print("--------")
predicted=clf.predict(test_data)
print(predicted)
print("--------")

print(test_label)
count=0
num_of_matches=0
for i in predicted:
    if i==test_label[count]:
        num_of_matches+=1
print(num_of_matches/len(predicted))



