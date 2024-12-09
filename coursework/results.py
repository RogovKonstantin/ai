import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


# Read dataset
data = pd.read_csv('datasets/dataset_final.csv')

# Select features and target
X = data.drop(columns=['TenYearCHD'])
y = data['TenYearCHD']

# Scale features to help with convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# --- Model 1: Sklearn Logistic Regression ---
# Initialize and train sklearn model
sklearn_model = SklearnLogisticRegression(random_state=0, max_iter=1000)  # Increased max_iter
sklearn_model.fit(X_train, y_train)

# Make predictions with sklearn model
y_train_pred_sklearn = sklearn_model.predict(X_train)
y_test_pred_sklearn = sklearn_model.predict(X_test)

# Evaluate sklearn model
train_accuracy_sklearn = accuracy_score(y_train, y_train_pred_sklearn)
test_accuracy_sklearn = accuracy_score(y_test, y_test_pred_sklearn)
train_mse_sklearn = mean_squared_error(y_train, y_train_pred_sklearn)
test_mse_sklearn = mean_squared_error(y_test, y_test_pred_sklearn)

# --- Model 2: Custom Logistic Regression ---
# Custom Logistic Regression Implementation

def sigmoid(z):
    # Clip z to avoid overflow
    z = max(min(z, 100), -100)
    return 1 / (1 + (2.718281828459045 ** -z))

def predict(X, weights):
    y_pred = []
    for x in X:
        z = sum(w * x_i for w, x_i in zip(weights, x))
        y_pred.append(1 if sigmoid(z) >= 0.5 else 0)
    return y_pred

def logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    weights = [0.0] * len(X[0])

    for epoch in range(epochs):
        for i in range(len(X)):
            z = sum(w * x_i for w, x_i in zip(weights, X[i]))  # Weighted sum
            prediction = sigmoid(z)
            for j in range(len(weights)):
                weights[j] += learning_rate * (y[i] - prediction) * X[i][j]

    return weights

# Convert to list for custom logistic regression
X_train_custom = X_train.tolist()
X_test_custom = X_test.tolist()

# Train custom model
weights = logistic_regression(X_train_custom, y_train.tolist())

# Make predictions with custom model
y_train_pred_custom = predict(X_train_custom, weights)
y_test_pred_custom = predict(X_test_custom, weights)

# Evaluate custom model
train_accuracy_custom = accuracy_score(y_train, y_train_pred_custom)
test_accuracy_custom = accuracy_score(y_test, y_test_pred_custom)
train_mse_custom = mean_squared_error(y_train, y_train_pred_custom)
test_mse_custom = mean_squared_error(y_test, y_test_pred_custom)

# --- Results ---
print("Sklearn Logistic Regression:")
print("Training Accuracy:", train_accuracy_sklearn)
print("Test Accuracy:", test_accuracy_sklearn)
print("Training MSE:", train_mse_sklearn)
print("Test MSE:", test_mse_sklearn)

print("\nCustom Logistic Regression:")
print("Training Accuracy:", train_accuracy_custom)
print("Test Accuracy:", test_accuracy_custom)
print("Training MSE:", train_mse_custom)
print("Test MSE:", test_mse_custom)
