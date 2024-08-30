import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns


def _sqrt(x):
    if x == 0:
        return 0
    guess = x / 2
    while True:
        new_guess = (guess + x / guess) / 2
        if abs(new_guess - guess) < 1e-6:
            return new_guess
        guess = new_guess


print('square root: ', _sqrt(25), '\n')

print('square root: ', _sqrt(100), '\n')

print('square root: ', _sqrt(50), '\n')

# Loading data
data = pd.read_csv("BankNote_Authentication.csv")

######################################################################
# randomize the data (since the class 1 aligns above each other)
data = data.sample(frac=1, random_state=92).reset_index(drop=True)

# Split data
train_size = int(0.7 * len(data))
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

print('train data: \n', train_data, '\n')

print('test data: \n', test_data, '\n')

print('TRAIN DATA WITH INDEX: \n')
for index, row in train_data.iterrows():
    print(index, '\n', row)

firstClassValue = train_data.iloc[0, -1]
print(firstClassValue)


def _sum(arr):
    result = 0
    for elem in arr:
        result += elem
    return result


x = [1, 2, 3, 4, 5]
print('sum: ', _sum(x), '\n')

# data.plot(kind='scatter', x='variance', y='skewness')
# plot.show()

sns.scatterplot(x='variance', y='skewness', hue='class', data=data)
plot.xlabel('variance')
plot.ylabel('skewness')
plot.title('Scatter Plot of Features with Target Class')
plot.show()

sns.scatterplot(x='curtosis', y='entropy', hue='class', data=data)
plot.xlabel('curtosis')
plot.ylabel('entropy')
plot.title('Scatter Plot of Features with Target Class')
plot.show()

sns.scatterplot(x='variance', y='curtosis', hue='class', data=data)
plot.xlabel('variance')
plot.ylabel('curtosis')
plot.title('Scatter Plot of Features with Target Class')
plot.show()

sns.scatterplot(x='variance', y='entropy', hue='class', data=data)
plot.xlabel('variance')
plot.ylabel('entropy')
plot.title('Scatter Plot of Features with Target Class')
plot.show()

sns.scatterplot(x='skewness', y='curtosis', hue='class', data=data)
plot.xlabel('skewness')
plot.ylabel('curtosis')
plot.title('Scatter Plot of Features with Target Class')
plot.show()

sns.scatterplot(x='skewness', y='entropy', hue='class', data=data)
plot.xlabel('skewness')
plot.ylabel('entropy')
plot.title('Scatter Plot of Features with Target Class')
plot.show()

# print(data)
#
# for index, row in data.iterrows():
#     print(row)
firstClassValue = 1  # found in train file
counter = {0: 2, 1: 1}
max_count = max(counter.values())  # Occurrence: 2 of class 0 - 2 of class 1
values_list = list(counter.values())
for key, value in counter.items():  # key is the class
    if value == max_count:
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

print('Predicted class: ', predicted_class)
