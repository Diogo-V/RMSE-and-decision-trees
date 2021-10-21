# Import definition
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

# ----------------------------------------- PREPROCESSING AND DATA FILTERING ----------------------------------------- #

# Loading dataset into working desk
data = arff.loadarff('breast.w.arff')
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
