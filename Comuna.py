import pandas as pd
import numpy as np

input_train = {
    "y1": [1, 1, 0, 1, 2, 1, 2, 0],
    "y2": [1, 1, 2, 2, 0, 1, 0, 2],
    "y3": [0, 5, 4, 3, 7, 1, 2, 9],
    "output": [1, 3, 2, 0, 6, 4, 5, 7]
}

input_test = {
    "y1": [2, 1],
    "y2": [0, 2],
    "y3": [0, 1],
    "output": [2, 4]
}

train_df = pd.DataFrame.from_dict(input_train)
test_df = pd.DataFrame.from_dict(input_test)


def basis_func(j, x):
    return pow(np.linalg.norm(x), j)


temp_matrix = []

for i in range(8):
    new_x = []
    new_line = []

    new_x.append(input_train["y1"][i])
    new_x.append(input_train["y2"][i])
    new_x.append(input_train["y3"][i])

    for j in range(4):
        new_line.append(basis_func(j, new_x))

    print("Run #", i, "\nNew X: ", new_x, "\nNew Line: ", new_line)

    temp_matrix.append(new_line)

phi = np.matrix(temp_matrix)

test_temp = []

for output in input_train["output"]:
    test_temp.append([output])

test_outputs = np.matrix(test_temp)

phi_transposed = phi.transpose()
phi_pinv = np.linalg.pinv(phi)
weights = phi_pinv * test_outputs

print("\nPretty phi matrix:\n")
print(phi)

print("\nPretty transposed phi matrix:\n")
print(phi_transposed)

print("\nPretty pseudo-inverse phi matrix:\n")
print(phi_pinv)

print("\nPretty test outputs matrix:\n")
print(test_outputs)

print("\nPretty weights matrix:\n")
print(weights)
