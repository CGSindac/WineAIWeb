import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fit_normal_equation(X, y):
    """
    Ordinary Least Squares via the Normal Equation:
        b = (X^T X)^-1 X^T y
    with an intercept column of ones prepended.
    TODO: complete this function. Fall back to np.linalg.pinv if X^T X is singular.
    """
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    XtX = X_aug.T @ X_aug
    Xty = X_aug.T @ y
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        beta = np.linalg.pinv(XtX) @ Xty
    else:
        beta = np.linalg.inv(XtX) @ Xty
    return beta

def predict(X, beta):
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    return X_aug @ beta

def get_full_weights():
    return fit_normal_equation(X_train, y_train)

def get_pca_weights():
    return fit_normal_equation(Z_train, y_train)

def calculate_full(new_data) -> float:
    new_data = np.atleast_2d(new_data)
    new_data_scaled  = (new_data - mu) @ S
    weights = get_full_weights()
    print(len(new_data_scaled), len(weights))
    return predict(new_data_scaled, weights)

def calculate_pca(new_data) -> float:
    return predict(new_data, get_pca_weights())

np.set_printoptions(precision=4, suppress=True)

df = pd.read_csv("./data/winequality-white.csv", sep=None, engine="python")
df.columns = df.columns.str.strip()


# Import Dataset to Python
feature_names = df.drop(columns=["quality"]).columns.tolist()
X_raw = df[feature_names].to_numpy()
y = df["quality"].to_numpy()


# Compute Mean and Standard deviation of the raw X matrix
mu = X_raw.mean(axis=0)
sigma = X_raw.std(axis=0)
S = np.diag(1 / sigma)

# Transform the Raw X Matrix
X_centered = X_raw - mu
X_transformed = X_centered @ S

# Split the dataset into training and test sets
rng = np.random.default_rng(42)
n = X_transformed.shape[0]
idx = rng.permutation(n)
train_size = int(0.8 * n)
train_idx, test_idx = idx[:train_size], idx[train_size:]


X_train, X_test = X_transformed[train_idx], X_transformed[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

XtX_train = X_train.T @ X_train # NOTE UNUSED

# Compute Eigenvalues and Eigenvectors & Covariance matrix of training features
C = np.cov(X_train, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]

eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]


V = eigvecs # Build Eigenvecton Marix
Lambda =  np.diag(eigvals) # Build Diagonal Eigenvalue matrix

explained_var_ratio = eigvals / eigvals.sum()
cum_var = np.cumsum(explained_var_ratio)
k = int(np.argmax(cum_var >= 0.95)) + 1

V_k = V[:, :k]
Z_train = X_train @ V_k
Z_test = X_test @ V_k
