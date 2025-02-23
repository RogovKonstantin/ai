# File: model_no_numpy.py
import pandas as pd
import matplotlib.pyplot as plt
import random

# -----------------------------
# 1. Load numeric train/test data
# -----------------------------
train_df = pd.read_csv("train_numeric.csv")
test_df = pd.read_csv("test_numeric.csv")

# -----------------------------
# 2. Compute Mean & Std for Normalization
# -----------------------------
def compute_mean_std(column):
    """Compute mean and standard deviation for normalization."""
    mean = sum(column) / len(column)
    variance = sum((x - mean) ** 2 for x in column) / len(column)
    std_dev = variance ** 0.5
    return mean, std_dev if std_dev != 0 else 1  # Prevent division by zero

# Normalize numeric features
for col in ['age', 'bmi', 'children']:
    mean_val, std_val = compute_mean_std(train_df[col].tolist())
    train_df[col] = [(x - mean_val) / std_val for x in train_df[col]]
    test_df[col] = [(x - mean_val) / std_val for x in test_df[col]]

# Normalize target variable (log-space)
y_mean, y_std = compute_mean_std(train_df["charges"].tolist())
train_df["charges"] = [(x - y_mean) / y_std for x in train_df["charges"]]
test_df["charges"] = [(x - y_mean) / y_std for x in test_df["charges"]]

# -----------------------------
# 3. Select Only Relevant Features
# -----------------------------
selected_features = ["age", "bmi", "smoker"]
X_train = train_df[selected_features].values.tolist()
y_train = [[val] for val in train_df["charges"].tolist()]
X_test  = test_df[selected_features].values.tolist()
y_test  = [[val] for val in test_df["charges"].tolist()]

# -----------------------------
# 4. Add Bias Column
# -----------------------------
def add_bias(X):
    return [[1] + row for row in X]

X_train_bias = add_bias(X_train)
X_test_bias  = add_bias(X_test)

# -----------------------------
# 5. Initialize Parameters
# -----------------------------
num_features = len(X_train_bias[0])
theta = [[random.uniform(-0.1, 0.1)] for _ in range(num_features)]

# -----------------------------
# 6. Define Matrix Operations
# -----------------------------
def mat_mult(A, B):
    """Multiply matrix A (m x n) with matrix B (n x p)."""
    m, n = len(A), len(A[0])
    p = len(B[0])
    result = [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)] for i in range(m)]
    return result

def transpose(A):
    """Transpose a matrix."""
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def scalar_mult_matrix(s, A):
    """Multiply each element in matrix A by scalar s."""
    return [[s * val for val in row] for row in A]

def matrix_subtract(A, B):
    """Subtract matrix B from A element-wise."""
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

# -----------------------------
# 7. Define MSE & Gradient
# -----------------------------
def mse(X, y, theta):
    """Compute Mean Squared Error (MSE)."""
    m = len(X)
    predictions = mat_mult(X, theta)
    error = matrix_subtract(predictions, y)
    squared_error = [[val[0] ** 2] for val in error]
    return sum(row[0] for row in squared_error) / (2 * m)

def gradient(X, y, theta, lambda_reg=0.01):
    """Compute the gradient of MSE with L2 regularization."""
    m = len(X)
    predictions = mat_mult(X, theta)
    error = matrix_subtract(predictions, y)
    X_T = transpose(X)
    grad = mat_mult(X_T, error)

    # Add L2 regularization
    for j in range(len(theta)):
        grad[j][0] += lambda_reg * theta[j][0]

    return scalar_mult_matrix(1 / m, grad)

# -----------------------------
# 8. Gradient Descent
# -----------------------------
learning_rate = 1e-2
num_iterations = 5000
mse_history = []

for i in range(num_iterations):
    grad = gradient(X_train_bias, y_train, theta)

    for j in range(len(theta)):
        theta[j][0] -= learning_rate * grad[j][0]

    cost = mse(X_train_bias, y_train, theta)
    mse_history.append(cost)

    if i % 1000 == 0:
        print(f"Iteration {i}, MSE: {cost}")

# -----------------------------
# 9. Final MSE & R² on Test Set
# -----------------------------
y_pred_test = mat_mult(X_test_bias, theta)
test_mse_log = 2 * mse(X_test_bias, y_test, theta)

# Compute R² manually
def mean(values):
    """Compute mean of a list."""
    return sum(values) / len(values)

y_test_vals = [row[0] for row in y_test]
y_pred_vals = [row[0] for row in y_pred_test]
mean_y_test = mean(y_test_vals)

sst_test = sum((val - mean_y_test) ** 2 for val in y_test_vals)
ssr_test = sum((y_pred_vals[i] - y_test_vals[i]) ** 2 for i in range(len(y_test_vals)))
r2_test = 1 - (ssr_test / sst_test)

print("\nFinal Test MSE:", test_mse_log)
print("Final Test R²:", r2_test)

# -----------------------------
# 10. Plot MSE vs. Iterations
# -----------------------------
plt.figure()
plt.plot(mse_history, label="MSE")
plt.xlabel("Итерации")
plt.ylabel("MSE")
plt.title("Изменение MSE по итерациям ")
plt.legend()
plt.savefig("no_libs_mse_vs_iteration.png")
plt.close()

# -----------------------------
# 11. Predicted vs. Actual (log-space)
# -----------------------------
plt.figure()
plt.scatter(y_test_vals, y_pred_vals, alpha=0.6)
plt.plot([min(y_test_vals), max(y_test_vals)],
         [min(y_test_vals), max(y_test_vals)],
         'r--')
plt.xlabel("Фактические значения")
plt.ylabel("Предсказанные значения")
plt.title("Предсказанные против фактических")
plt.savefig("predicted_vs_actual_no_libs.png")
plt.close()

# -----------------------------
# 12. Save learned theta (without NumPy)
# -----------------------------
with open("theta_no_numpy.txt", "w") as f:
    for row in theta:
        f.write(str(row[0]) + "\n")

print("Saved no-libs model parameters to theta_no_numpy.txt.")
print("Plots saved: no_libs_mse_vs_iteration.png, predicted_vs_actual_no_libs.png")
