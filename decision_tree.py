#author: Mert Surucuoglu
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

def seperateData(v_data, v_label, cl):
    seperatedData = []
    seperatedLabel = []
    for i in range(len(v_data)):
        if cl==int(v_label[i]):
            seperatedData.append(v_data[i])
            seperatedLabel.append(cl)
    return (np.array(seperatedData), np.array(seperatedLabel), 5)


data = np.load("train_data_concat1000.npy")
data_label = np.load("train_label_concat1000.npy")

"""test_data = data[int(len(data)*4/5):]
test_label = data_label[int(len(data_label)*4/5):]
train_data = data[0:int(len(data)*4/5)]
train_label = data_label[0:int(len(data_label)*4/5)]"""

kf = KFold(n_splits=10,shuffle=True)
kf.get_n_splits(data)
for train_index, test_index in kf.split(data):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train_data, test_data= data[train_index], data[test_index]
    train_label, test_label = data_label[train_index], data_label[test_index]

train_data = train_data.reshape((len(train_data), -1))
test_data = test_data.reshape((len(test_data), -1))

estimator = DecisionTreeClassifier(max_leaf_nodes=50, random_state=10)
estimator.fit(train_data, train_label)



validation_data = [seperateData(test_data, test_label, i) for i in range(5)]
maches=0
total=0
for j in range(len(validation_data)):
    v_data = validation_data[j][0]
    v_label = validation_data[j][1]
    y = estimator.predict(v_data)
    c=0
    for i in range(len(y)):
        if y[i]==v_label[i]:
            c+=1
            maches+=1
        total+=1
    print("Acc for",j,": ",c/len(y))
print(maches/total)


"""for j in [0,1,2,3,4]:
    c = 0
    for i in range(len(data_label)):
        if j==data_label[i]:
            c+=1
    print(c)"""""
