#author: Mert Surucuoglu
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
import numpy as np

#This function seperates into sub classes for getting accuracy for each classes
def seperateData(v_data, v_label, cl):
    seperatedData = []
    seperatedLabel = []
    for i in range(len(v_data)):
        if cl==int(v_label[i]):
            seperatedData.append(v_data[i])
            seperatedLabel.append(cl)
    return (np.array(seperatedData), np.array(seperatedLabel), 5)

#loads the data from npy
data = np.load("../WorkStation/train_data_concat1000.npy")
data_label = np.load("../WorkStation/train_label_concat1000.npy")

#K-fold splits into 10 and shuffles the indexes
kf = KFold(n_splits=10,shuffle=True)
kf.get_n_splits(data)
for train_index, test_index in kf.split(data):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train_data, test_data= data[train_index], data[test_index]
    train_label, test_label = data_label[train_index], data_label[test_index]

#puts all features into a single array
train_data = train_data.reshape((len(train_data), -1))
test_data = test_data.reshape((len(test_data), -1))


# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(bootstrap=True,
                       class_weight='balanced_subsample',
                       criterion='gini',
                       max_depth=5,
                       max_features='auto',
                       max_leaf_nodes=50,
                       min_impurity_decrease=0.0,
                       min_impurity_split=None,
                       min_samples_leaf=1,
                       min_samples_split=2,
                       min_weight_fraction_leaf=0.0,
                       n_estimators=100,#The number of trees in the forest
                       n_jobs=1,
                       oob_score=False,
                       random_state=None,
                       verbose=0,
                       warm_start=False)


# Train the Classifier to take the training features and learn how they relate to the training(the species)
clf.fit(train_data, train_label)

# makes a list to get each accuracy
validation_data = [seperateData(test_data, test_label, i) for i in range(5)]
num_of_matches=0
num_test_data=0

for j in range(len(validation_data)):
    v_data = validation_data[j][0] # features of a test image
    v_label = validation_data[j][1] # label of a test image
    predicted = clf.predict(v_data)
    count_for_each = 0
    for i in range(len(predicted)):
        if predicted[i]==v_label[i]:
            count_for_each+=1
            num_of_matches+=1
        num_test_data+=1
    print("Accuracy for",j,"class : ",count_for_each/len(predicted))
print("General Accuracy : ",num_of_matches/num_test_data)
print("0 is Cloudy\n1 is Sunny\n2 is Rainy\n3 is Snowy \n4 is Foggy\n")




