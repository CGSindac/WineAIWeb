import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.linalg import solve, inv

np.set_printoptions(precision=4, suppress=True)

# Loading csv file
data_frame = pd.read_csv("./data/winequality-white.csv", sep=';')

# Initialize Data
feature_names = data_frame.drop(columns=["quality"]).columns.tolist() 

print(len(feature_names))
print(feature_names)
X_raw = data_frame[feature_names].to_numpy()
Y = data_frame["quality"].to_numpy()

# Computing Product of X^T X
X = np.asarray(X_raw, dtype=float)
X_transposed = X.T

Xt_mul_X = X_transposed @ X

# Stop Here ( Refer to Parts D Onwards)
Xt_mul_Y = X_transposed @ Y
Xt_mul_X_inverse = np.linalg.inv(Xt_mul_X)

determinant = np.linalg.det(Xt_mul_X)
theta = Xt_mul_X_inverse @ Xt_mul_Y

predictions = (X @ theta).flatten()

plt.figure(figsize=(7,4))

# Plot Points
plt.plot(data_frame["quality"], 'o-', label='Actual')
plt.plot(predictions,'s--', label='Predicted')

plt.xlim(0, 10)

# Visualization
plt.xlabel("Observation")
plt.ylabel("Quality")
plt.title("Normal Equation Regression")
plt.legend()
plt.grid(True)
plt.show()



