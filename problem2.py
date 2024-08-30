# KNN
import pandas as pd


# Helpers

# Custom length function
def custom_len(arr):
    count = 0
    for _ in arr:
        count += 1
    return count


def _sum(arr):
    result = 0
    for elem in arr:
        result += elem
    return result


def _sqrt(x):
    if x == 0:
        return 0
    guess = x / 2
    while True:
        new_guess = (guess + x / guess) / 2
        if abs(new_guess - guess) < 1e-6:
            return new_guess
        guess = new_guess


# d[1] = [0, 0, 1, 1] class values counter = {0 -> 2, 1 -> 2}
def custom_counter(arr):  # tie case ?
    counter = {}
    for elem in arr:
        if elem in counter:
            counter[elem] += 1
        else:
            counter[elem] = 1

    return counter


def Bubble_sort(arr):
    n = custom_len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


def calc_mean(_data):
    total = _sum(_data)
    _mean = total / custom_len(_data)
    return _mean


def calc_std(_data):
    m = calc_mean(_data)
    squared_diff = [(x - m) ** 2 for x in _data]
    variance = _sum(squared_diff) / custom_len(_data)
    _std = variance ** 0.5
    return _std


# Loading data
data = pd.read_csv("BankNote_Authentication.csv")

######################################################################

# randomize the data (since the class 1 aligns above each other)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

######################################################################

# Split data
train_size = int(0.7 * len(data))
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

######################################################################

# Normalize data
for col in data.columns[:-1]:
    mean = calc_mean(train_data[col])  # mean and std should be implemented from scratch !!
    std = calc_std(train_data[col])
    train_data.loc[:, col] = (train_data[col] - mean) / std
    test_data.loc[:, col] = (test_data[col] - mean) / std


######################################################################

# Euclidean distance function
def euclidean_distance(x1, x2):
    return _sqrt(_sum((x1 - x2) ** 2))  # test


######################################################################

# KNN function
def knn(train_data, test_data, k):
    correct = 0
    total = len(test_data)

    for i in range(len(test_data)):
        distances = []
        for j in range(len(train_data)):
            dist = euclidean_distance(test_data.iloc[i, :-1], train_data.iloc[j, :-1])  # test
            distances.append((dist, train_data.iloc[j, -1]))
        # Sort the distances in ascending order based on the distance
        Bubble_sort(distances)
        # get the nearest classes then count the occurrence
        top_classes = [d[1] for d in distances[:k]]  # ? d[1] = [0, 0, 1, 0] - # top_classes = [0, 0, 1] if k = 3
        # ties are handled here by taking the first class of the tied classes
        firstClassValue = train_data.iloc[0, -1]
        counter = custom_counter(top_classes)
        max_count = max(counter.values())  # Occurrence: 2 of class 0 - 2 of class 1
        values_list = list(counter.values())
        predicted_class = None
        for key, value in counter.items():  # key is the class
            if value == max_count:
                if custom_len(
                        values_list) == 2:  # if the length of list is 2 (index of class 0 and 1), else the list has class 0 or 1 only
                    if values_list[0] == values_list[1]:  # Tie case
                        if key == firstClassValue:
                            predicted_class = key
                            break
                        else:
                            if key == 1:
                                predicted_class = 0
                                break
                            else:  # key == 0
                                predicted_class = 1
                                break
                    else:  # Not tie case (normal case)
                        predicted_class = key
                        break
                else:
                    predicted_class = key
                    break

        if predicted_class == test_data.iloc[i, -1]:  # calc accuracy
            correct += 1

    accuracy = correct / total

    return accuracy


######################################################################


# Experiment with different values of k
for k in range(1, 10):
    print(f'\nKNN Model is Working using K = {k}, Please be Patient...\n')
    accuracy = knn(train_data, test_data, k)
    correct_instances = int(accuracy * len(test_data))
    total_instances = len(test_data)
    print(f"K = {k}, Correctly Classified Instances = {correct_instances}/{total_instances}, Accuracy = {accuracy}")
