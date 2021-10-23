# Import definition
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif

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

# Creates a k fold cross validator with 10 splits
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=GROUP_NUMBER)

# --------------------------------------------------- QUESTION 5.i --------------------------------------------------- #

# Iterates over each number of selected features and creates a decision tree
for k in K:

    # Selects the best k features using mutual information
    X_new = SelectKBest(mutual_info_classif, k=k).fit_transform(X, y.ravel())

    # Creates tree with k max number of features and using information gain
    tree = DecisionTreeClassifier(criterion="entropy", max_features=k)

# --------------------------------------------------- QUESTION 5.ii -------------------------------------------------- #

# Iterates over each number for maximum depth and creates a decision tree
for k in K:

    # Creates tree with k max depth
    tree = DecisionTreeClassifier(max_depth=k)

# ---------------------------------------------------- QUESTION 6 ---------------------------------------------------- #


# ---------------------------------------------------- QUESTION 7 ---------------------------------------------------- #
