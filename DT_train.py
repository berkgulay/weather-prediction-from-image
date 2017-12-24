# author: Mert Surucuoglu
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, tree


# This function seperates into sub classes for getting accuracy for each classes
def separate_data(_test_data, _test_label, class_num):
    separated_data = []
    separated_label = []
    for i in range(len(_test_data)):
        if class_num == int(_test_label[i]):
            separated_data.append(_test_data[i])
            separated_label.append(class_num)
    return np.array(separated_data), np.array(separated_label)


def get_accuracy():
    num_of_matches = 0
    num_test_data = 0
    # get an accuracy in here
    for j in range(len(separated_test_data)):
        test_feature = separated_test_data[j][0]  # features of a test image
        test_label = separated_test_data[j][1]  # label of a test image
        predicted = clf.predict(test_feature)
        count_for_each = 0
        for i in range(len(predicted)):
            if predicted[i] == test_label[i]:
                count_for_each += 1
                num_of_matches += 1
            num_test_data += 1
        if j == 0:  # Cloudy
            cloudy_acc = count_for_each / len(predicted)
            print("Accuracy for cloudy class : ", count_for_each / len(predicted))
        elif j == 1:  # Sunny
            sunny_acc = count_for_each / len(predicted)
            print("Accuracy for sunny class : ", count_for_each / len(predicted))

        elif j == 2:  # Rainy
            rainy_acc = count_for_each / len(predicted)
            print("Accuracy for rainy class : ", count_for_each / len(predicted))

        elif j == 3:  # Snowy
            snowy_acc = count_for_each / len(predicted)
            print("Accuracy for snowy class : ", count_for_each / len(predicted))

        elif j == 4:  # Foggy
            foggy_acc = count_for_each / len(predicted)
            print("Accuracy for foggy class : ", count_for_each / len(predicted))

    print("General Accuracy : ", num_of_matches / num_test_data)
    print("-------------------------------------------------")
    return num_of_matches / num_test_data, cloudy_acc, sunny_acc, rainy_acc, snowy_acc, foggy_acc


train_data = np.array([])
test_data = np.array([])
# loads the data from npy
data = np.load('../WorkStation/features.npy')
data_label = np.load('../WorkStation/labels.npy')
data_label = np.reshape(data_label, (np.shape(data)[0],))
total_accuracy = 0
cloudy_accuracy = 0
sunny_accuracy = 0
rainy_accuracy = 0
snowy_accuracy = 0
foggy_accuracy = 0
# K-fold splits into 10 and shuffles the indexes
split_size = 10
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
    clf = DecisionTreeClassifier(max_leaf_nodes=40, min_samples_leaf=20, max_depth=20, random_state=None)
    clf.fit(train_data_reshaped, train_label)

    # makes a list to get each accuracy
    separated_test_data = [separate_data(test_data_reshaped, test_label, i) for i in range(5)]
    accuracies = get_accuracy()
    total_accuracy += accuracies[0]
    cloudy_accuracy += accuracies[1]
    sunny_accuracy += accuracies[2]
    rainy_accuracy += accuracies[3]
    snowy_accuracy += accuracies[4]
    foggy_accuracy += accuracies[5]
print("OVERALL ACCURACY\n-------------------------------------------------")
print("Overall Accuracy For Cloudy Class :", cloudy_accuracy / split_size)
print("Overall Accuracy For Sunny Class :", sunny_accuracy / split_size)
print("Overall Accuracy For Rainy Class :", rainy_accuracy / split_size)
print("Overall Accuracy For Snowy Class :", snowy_accuracy / split_size)
print("Overall Accuracy For Foggy Class :", foggy_accuracy / split_size)
print("Overall Accuracy: ", total_accuracy / split_size)
