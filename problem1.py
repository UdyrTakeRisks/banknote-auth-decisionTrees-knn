# Decision Tree
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("BankNote_Authentication.csv")

# Separate features and target variable
X = data.drop(columns=['class'])
y = data['class']


# Function to run experiments

def run_experiment(X, y, train_size):
    accuracies = []
    tree_sizes = []
    random_seed = 42
    random_states = [random_seed + i for i in range(5)]
    for state in random_states:  # run five times
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
                                                            random_state=state)  # randomize states

        # Train decision tree classifier
        DT = DecisionTreeClassifier()
        DT.fit(X_train, y_train)

        # Make predictions
        y_pred = DT.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # accuracy = DT.score(y_test, y_pred) ??
        accuracies.append(accuracy)

        # Get tree size (number of its nodes)
        tree_size = DT.tree_.node_count
        tree_sizes.append(tree_size)

    return accuracies, tree_sizes


# Experiment with fixed train_test split ratio (25%)
print("Fixed train_test split ratio (25%):")
accuracies, tree_sizes = run_experiment(X, y, train_size=0.25)  # 0.25 How ??
print(f"Experiment (1):")
print("  Accuracy:", accuracies)
print("  Tree sizes:", tree_sizes)

# Experiment with different range of train_test split ratio
train_test_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
mean_accuracies = []
max_accuracies = []
min_accuracies = []
mean_tree_sizes = []
mean_final_tree_sizes = []
max_tree_sizes = []
min_tree_sizes = []

for ratio in train_test_ratios:
    accuracies, tree_sizes = run_experiment(X, y, train_size=ratio)
    mean_accuracies.append(np.mean(accuracies))
    max_accuracies.append(np.max(accuracies))
    min_accuracies.append(np.min(accuracies))
    mean_tree_sizes.append(np.mean(tree_sizes))
    # for every ratio, get the last value in the tree_sizes this will be the final tree nodes
    mean_final_tree_sizes.append(tree_sizes[-1])
    max_tree_sizes.append(np.max(tree_sizes))
    min_tree_sizes.append(np.min(tree_sizes))

# Print mean, max, and min accuracies and tree sizes for each split ratio
print("\nExperiment with different train_test split ratios:")
print(f"Experiment (2):")
for i, ratio in enumerate(train_test_ratios):
    print(f"Train-Test Split Ratio: {ratio * 100}% - {100 - ratio * 100}%")
    print("  Mean Accuracy:", mean_accuracies[i])
    print("  Max Accuracy:", max_accuracies[i])
    print("  Min Accuracy:", min_accuracies[i])
    print("  Mean Tree Size:", mean_tree_sizes[i])
    print("  Max Tree Size:", max_tree_sizes[i])
    print("  Min Tree Size:", min_tree_sizes[i])
    print()  # newline

# Draw plots
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(train_test_ratios, mean_accuracies, marker='o')
plt.title('Mean Accuracy vs Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(train_test_ratios, mean_tree_sizes, marker='o')
# plt.title('Mean Tree Size vs Training Set Size')
# plt.xlabel('Training Set Size')
# plt.ylabel('Mean Tree Size')

plt.subplot(1, 2, 2)
plt.plot(train_test_ratios, mean_final_tree_sizes, marker='o')
plt.title('Mean Final Tree Size vs Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Final Tree Size')

plt.tight_layout()
plt.show()
