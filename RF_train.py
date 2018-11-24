# author: Mert Surucuoglu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np
import tools as T

# loads the data from npy
data = np.load('../features/features.npy')
data_label = np.load('../features/labels.npy')
data_label = np.reshape(data_label, (np.shape(data)[0],))

# K-fold splits into 10 and shuffles the indexes
split_size = 10
kf = KFold(n_splits=split_size, shuffle=True)
kf.get_n_splits(data)

overallAccuracies = np.zeros(5)
generalOverallAccuracy = 0

for train_index, test_index in kf.split(data):
    # print("TRAIN:", train_index, "TEST:", test_index)
    train_data, test_data = data[train_index], data[test_index]
    train_label, test_label = data_label[train_index], data_label[test_index]

    # puts all features into a single array
    train_data_reshaped = train_data.reshape((len(train_data), -1))
    test_data_reshaped = test_data.reshape((len(test_data), -1))

    # Create a random forest Classifier. By convention, clf means 'Classifier'
    clf = RandomForestClassifier(bootstrap=False,
                                 max_leaf_nodes=None,
                                 n_estimators=12,  # The number of trees in the forest
                                 min_weight_fraction_leaf=0.0,
                                 )

    # Train the Classifier to take the training features and learn how they relate to the training(the species)
    clf.fit(train_data_reshaped, train_label)
    # makes a list to get each accuracy
    separated_test_data = T.separate_data(test_data_reshaped,test_label)
    # to get a accuracy
    for i in range(len(separated_test_data)):
        v_data = separated_test_data[i][0]
        v_label = separated_test_data[i][1]
        y = clf.predict(v_data)
        acc = T.get_accuracy_of_class(v_label, y)
        overallAccuracies[i]+=acc
        print("Accuracy for class " + T.classes[i] + ": ", acc)
    generalAccuracy =  T.get_accuracy_of_class(test_label,clf.predict(test_data_reshaped))
    generalOverallAccuracy+= generalAccuracy
    print("General Accuracy:", generalAccuracy)
    print("---------------------------------------------")

print("OVERALL ACCURACY\n-------------------------------------------------")
for i in range(len(overallAccuracies)):
    print("Overall Accuracy For", T.classes[i], overallAccuracies[i]/split_size)
print("Overall Accuracy", generalOverallAccuracy/split_size)










