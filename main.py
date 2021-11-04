# Import definition
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold

# Constants definition
K = [1, 3, 5, 9]
GROUP_NUMBER = 16  # Our group number

# ----------------------------------------- PREPROCESSING AND DATA FILTERING ----------------------------------------- #

# Loading dataset into working desk
data = arff.loadarff('./data/breast.w.arff')
df = pd.DataFrame(data[0])

# Removes NaN values from dataset by deleting rows
df.dropna(axis=0, how="any", inplace=True)

# Gets X (data matrix) and y (target values column matrix)
X = df.drop("Class", axis=1).to_numpy()
y = df["Class"].to_numpy()

# Performs some preprocessing by turning labels into binaries (benign is 1)
# We are doing a "double conversion" to convert everything to Binary type
for count, value in enumerate(y):
    if value == b"benign":
        y[count] = "yes"
    else:
        y[count] = "no"
lb = LabelBinarizer()
y = lb.fit_transform(y)

# Holds testing and training accuracy for each K (exercise 5.i)
d_i = {
    "train": [],
    "test": []
}

# Holds testing and training accuracy for each K (exercise 5.ii)
d_ii = {
    "train": [],
    "test": []
}

# Creates a k fold cross validator
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=GROUP_NUMBER)

# --------------------------------------------------- QUESTION 5.i --------------------------------------------------- #

# Iterates over each number of selected features and creates a decision tree
for k in K:

    # Selects the best k features using mutual information
    X_new = SelectKBest(mutual_info_classif, k=k).fit_transform(X, y.ravel())

    # Creates tree with k max number of features
    tree = DecisionTreeClassifier(criterion="gini", max_features=k)

    # Performs a cross validate to train and get accuracy values for this tree
    values = cross_validate(tree, X_new, y, cv=skf, scoring="accuracy", return_train_score=True)

    # Gets train and test accuracy and appends them to dictionary
    acc_train = np.mean(values["train_score"])
    acc_test = np.mean(values["test_score"])
    d_i["train"].append(acc_train)
    d_i["test"].append(acc_test)

print(f"Dict 5.i: {d_i}\n")

# Plots graph
plt.plot(K, d_i["train"], marker='o')
plt.plot(K, d_i["test"], marker='o')
plt.title("Question 5.i")
plt.legend(["train", "test"])
plt.ylabel("Accuracy")
plt.xlabel("K selected features")
plt.show()

# --------------------------------------------------- QUESTION 5.ii -------------------------------------------------- #

# Iterates over each number for maximum depth and creates a decision tree
for k in K:

    # Creates tree with k max depth
    tree = DecisionTreeClassifier(criterion="gini", max_depth=k)

    # Performs a cross val
    values = cross_validate(tree, X, y, cv=skf, scoring="accuracy", return_train_score=True)

    # Gets train and test accuracy and appends them to dictionary
    acc_train = np.mean(values["train_score"])
    acc_test = np.mean(values["test_score"])
    d_ii["train"].append(acc_train)
    d_ii["test"].append(acc_test)

print(f"Dict 5.ii: {d_ii}\n")

# Plots graph
plt.plot(K, d_ii["train"], marker='o')
plt.plot(K, d_ii["test"], marker='o')
plt.title("Question 5.ii")
plt.legend(["train", "test"])
plt.ylabel("Accuracy")
plt.xlabel("K max tree depth")
plt.show()

# ---------------------------------------------------- QUESTION 6 ---------------------------------------------------- #


# ---------------------------------------------------- QUESTION 7 ---------------------------------------------------- #
