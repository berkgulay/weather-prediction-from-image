# author: Mert Surucuoglu
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier


# This function seperates into sub classes for getting accuracy for each classes
def separate_data(_test_data, _test_label, cl):
    separated_data = []
    separated_label = []
    for i in range(len(_test_data)):
        if cl == int(_test_label[i]):
            separated_data.append(_test_data[i])
            separated_label.append(cl)
    return np.array(separated_data), np.array(separated_label)


train_data = np.array([])
test_data = np.array([])
# loads the data from npy
data = np.load("../WorkStation/train_data_concat1000.npy")
data_label = np.load("../WorkStation/train_label_concat1000.npy")

# K-fold splits into 10 and shuffles the indexes
kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(data)
for train_index, test_index in kf.split(data):
    # print("TRAIN:", train_index, "TEST:", test_index)
    train_data, test_data = data[train_index], data[test_index]
    train_label, test_label = data_label[train_index], data_label[test_index]

# puts all features into a single array
train_data_reshaped = train_data.reshape((len(train_data), -1))
test_data_reshaped = test_data.reshape((len(test_data), -1))

# Create a Decision Tree Classifier.
estimator = DecisionTreeClassifier(max_leaf_nodes=50, random_state=None)
estimator.fit(train_data_reshaped, train_label)

# makes a list to get each accuracy
separated_test_data = [separate_data(test_data_reshaped, test_label, i) for i in range(5)]
num_of_matches = 0
num_test_data = 0

# get an accuracy in here
for j in range(len(separated_test_data)):
    v_data = separated_test_data[j][0]  # features of a test image
    v_label = separated_test_data[j][1]  # label of a test image
    predicted = estimator.predict(v_data)
    count_for_each = 0
    for i in range(len(predicted)):
        if predicted[i] == v_label[i]:
            count_for_each += 1
            num_of_matches += 1
        num_test_data += 1
    print("Accuracy for", j, "class : ", count_for_each / len(predicted))
print("General Accuracy : ", num_of_matches / num_test_data)
print("0 is Cloudy\n1 is Sunny\n2 is Rainy\n3 is Snowy \n4 is Foggy\n")
